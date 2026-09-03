"""Persona run — the actuary doing a rate review.

Steps mirror docs/RELEASE_0.4_PLAN.md §"Persona runs": roles/recode/derived/
filter/split come from the project file (the data grids are canvas widgets
Playwright cannot type into), then in the browser: check the variables and the
split, add an interaction and a linear term on the Design page, choose the
offset on the Model page, fit, review A/E by every rating factor and by pair,
open the interaction table, download Excel + scorer + script, reload and check
the fit survived, and finally run the exported script and compare it with the
downloaded scorer.
"""

from __future__ import annotations

import time
import zipfile

from ._helpers import (
    assert_clean,
    click,
    download,
    edit_grid_cell,
    goto_page,
    run_python,
    select,
    settle,
    tab,
    wait_text,
)
from .conftest import SERVER_PYTHON


def test_actuary_rate_review(actuary_server, browser, e2e_dir):
    srv, project_path = actuary_server
    t0 = time.time()
    pg = browser.new_page(viewport={"width": 1500, "height": 1000})
    pg.goto(srv.url, wait_until="networkidle")
    settle(pg)
    assert_clean(pg, "project page")
    assert "actuary" in pg.get_by_test_id("stSidebar").inner_text()

    # -- Variables: roles are visible and a target is set
    goto_page(pg, "Variables")
    assert_clean(pg, "variables")
    body = pg.locator('[data-testid="stMain"]').inner_text()
    assert "target" in body and "ClaimNb" in body

    # -- Split: the balance table shows train and holdout
    goto_page(pg, "Split")
    assert_clean(pg, "split")
    body = pg.locator('[data-testid="stMain"]').inner_text()
    assert "train" in body and "holdout" in body

    # -- Design: Density becomes piecewise-linear, VehPower × VehGas is added
    goto_page(pg, "Design")
    assert_clean(pg, "design")
    select(pg, "Variable", "Density")
    select(pg, "Kind", "linear")
    assert_clean(pg, "design kind=linear")
    assert "rounded outward" in pg.locator('[data-testid="stMain"]').inner_text()
    select(pg, "First variable", "VehPower")
    select(pg, "Second variable", "VehGas")
    click(pg, "Add interaction")
    assert_clean(pg, "add interaction")
    assert "VehPower×VehGas" in pg.locator('[data-testid="stMain"]').inner_text()

    # -- Model: offset = log(current premium) (the derived column), then fit
    goto_page(pg, "Model")
    assert_clean(pg, "model")
    select(pg, "Offset", "log_current_premium")
    click(pg, "Fit model")
    assert_clean(pg, "fit")
    main_text = pg.locator('[data-testid="stMain"]').inner_text()
    assert "Fitted and up to date" in main_text
    assert "log_current_premium" in main_text

    # -- Diagnostics: A/E by every rating factor, then by pair
    goto_page(pg, "Diagnostics")
    assert_clean(pg, "diagnostics")
    for var in [
        "DrivAge",
        "VehAge",
        "BonusMalus",
        "Density",
        "VehPower",
        "VehGas",
        "Region",
    ]:
        select(pg, "Variable", var)
        assert_clean(pg, f"A/E {var}")
        assert pg.locator(".js-plotly-plot").count() >= 1
    tab(pg, "A/E by pair")
    select(pg, "Rows", "VehPower")
    select(pg, "Columns", "VehGas")
    assert_clean(pg, "A/E by pair")

    # -- Rate tables: cap one relativity (VehGas row 2 -> 1.05), then the
    #    interaction table and its exports
    goto_page(pg, "Rate tables")
    assert_clean(pg, "rate tables")
    select(pg, "Variable", "VehGas")
    edit_grid_cell(pg, 0, 1, "1.05", expect="1 adjustment")
    assert_clean(pg, "cap relativity")
    main = pg.locator('[data-testid="stMain"]')
    select(pg, "Variable", "VehPower×VehGas")
    assert_clean(pg, "interaction table")
    assert "Cells multiply" in main.inner_text()
    xlsx = download(pg, "Excel rate tables", e2e_dir)
    names = zipfile.ZipFile(xlsx).read("xl/workbook.xml").decode()
    assert "VehPower×VehGas (matrix)" in names and "Density" in names
    scorer = download(pg, "Scorer (.easyglm)", e2e_dir)

    # -- Export: the script
    goto_page(pg, "Export")
    assert_clean(pg, "export")
    script = download(pg, "Download script", e2e_dir)
    src = script.read_text()
    assert "InteractionEncoder(" in src and "LinearEncoder(" in src

    # -- Project: save, then reload the browser and check the fit is restored
    goto_page(pg, "Project & data")
    click(pg, "Save project")
    assert_clean(pg, "save")
    pg.goto(srv.url, wait_until="networkidle")
    settle(pg)
    goto_page(pg, "Model")
    assert "Fitted and up to date" in pg.locator('[data-testid="stMain"]').inner_text()
    goto_page(pg, "Rate tables")
    assert wait_text(pg, "1 adjustment"), pg.locator(
        '[data-testid="stMain"]'
    ).inner_text()[
        :500
    ]  # the cap survived the reload

    # -- The exported script reproduces the downloaded scorer
    out = run_python(
        SERVER_PYTHON,
        f"""
import runpy, numpy as np, polars as pl
from pathlib import Path
from easy_glm.engine import RateModel
ns = runpy.run_path({str(script)!r})
rebuilt = [p for p in Path('.').glob('*.easyglm')]
assert rebuilt, 'script wrote no .easyglm'
a = RateModel.from_json(rebuilt[0]); b = RateModel.from_json({str(scorer)!r})
df = pl.read_parquet({str(project_path.parent / 'policies.parquet')!r}).head(5000)
df = df.with_columns(pl.col('Area').cast(pl.Utf8).replace({{'E': 'D', 'F': 'D'}}), pl.col('current_premium').log().alias('log_current_premium'))
pa = a.predict(df, exposure_col=None); pb = b.predict(df, exposure_col=None)
assert b.metadata.offset_col == 'log_current_premium', b.metadata
assert len(b.snapshots) >= 2, 'adjustment snapshot missing from the downloaded scorer'
print('maxdiff', float(np.max(np.abs(pa / pb - 1))))
""",
        e2e_dir,
    )
    assert "maxdiff" in out
    assert float(out.split("maxdiff")[1].split()[0]) < 1e-9
    print(f"actuary persona: {time.time() - t0:.0f}s")
