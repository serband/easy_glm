"""Persona run — the data scientist comparing models.

freq_v1 (CV lasso) is fitted; freq_v2 is created and given an interaction;
Density is switched to a linear term (the variable design is shared by all
models, so freq_v1 goes stale and is refitted); both are compared on the
Diagnostics page (challenger overlay, Gini, double lift), the pair search runs,
the Compare page shows both models with the table of relativities that differ,
freq_v2 is promoted to champion there, both scripts and scorers export, the HTML
report is downloaded and opened in the browser, and the Gini shown on screen is
recomputed from the downloaded scorer.
"""

from __future__ import annotations

import re
import time

from ._helpers import (
    assert_clean,
    click,
    download,
    fill,
    goto_page,
    run_python,
    select,
    select_sidebar,
    settle,
    tab,
    wait_text,
)
from .conftest import SERVER_PYTHON


def test_data_scientist_model_comparison(scientist_server, browser, e2e_dir):
    srv, _project_path = scientist_server
    t0 = time.time()
    pg = browser.new_page(viewport={"width": 1500, "height": 1000})
    pg.goto(srv.url, wait_until="networkidle")
    settle(pg)

    # -- fit freq_v1 with CV
    goto_page(pg, "Model")
    click(pg, "Fit model")
    assert_clean(pg, "fit v1")
    main = pg.locator('[data-testid="stMain"]')
    assert "Fitted and up to date" in main.inner_text()

    # -- create freq_v2
    fill(pg, "New model name", "freq_v2")
    click(pg, "Create")
    assert_clean(pg, "create v2")
    assert "freq_v2" in pg.get_by_test_id("stSidebar").inner_text()

    # -- Model: interaction + linear Density for freq_v2
    goto_page(pg, "Model")
    select(pg, "Variable", "Density")
    select(pg, "Kind", "linear")
    select(pg, "Model", "freq_v2")
    select(pg, "First variable", "DrivAge")
    select(pg, "Second variable", "VehPower")
    click(pg, "Add interaction")
    assert_clean(pg, "add interaction v2")

    # -- the variable design is shared by every model, so freq_v1 is now stale
    #    (the workbench says so) and gets refitted with Density linear; freq_v2
    #    differs from it only by the interaction
    goto_page(pg, "Model")
    select(pg, "Model", "freq_v1")
    assert "Spec changed since the last fit" in main.inner_text()
    click(pg, "Fit model")
    assert_clean(pg, "refit v1")
    select(pg, "Model", "freq_v2")
    click(pg, "Fit model")
    assert_clean(pg, "fit v2")
    assert "Fitted and up to date" in main.inner_text()

    # -- Diagnostics: challenger overlay + pair search on v1. The challenger is
    #    chosen once in the sidebar; Diagnostics, Rate tables and Compare follow it
    select_sidebar(pg, "Default comparison model", "freq_v2")
    goto_page(pg, "Diagnostics")
    select(pg, "Model", "freq_v1")
    assert_clean(pg, "challenger")
    assert "freq_v2" in pg.get_by_test_id("stMain").inner_text()
    tab(pg, "Lift")
    gini_text = main.inner_text()
    assert "challenger" in gini_text
    m = re.search(r"Normalised Gini \(holdout\): ([0-9.]+)", gini_text)
    assert m, gini_text[:400]
    shown_gini = float(m.group(1))
    tab(pg, "Double lift")
    assert pg.locator(".js-plotly-plot").count() >= 1
    tab(pg, "Residual factors")
    click(pg, "Search pairs")
    assert_clean(pg, "pair search")
    # the Show selectbox defaults to the top-ranked pair: a real pair of predictors
    show = pg.get_by_test_id("stSelectbox").filter(has_text="Show").first.inner_text()
    assert " × " in show, show
    assert pg.get_by_test_id("stDataFrame").count() >= 1  # the ranked pairs table
    assert "search bands" in main.inner_text()  # the heatmap of the shown pair

    # -- Compare: both models side by side, and which relativities differ
    goto_page(pg, "Compare")
    assert_clean(pg, "compare")
    assert "freq_v1" in main.inner_text() and "freq_v2" in main.inner_text()
    assert "Project champion: freq_v1" in main.inner_text()
    tab(pg, "Relativities that differ")
    assert wait_text(pg, "log_diff"), main.inner_text()[:600]
    # the diff table itself is a canvas grid: assert the element and the caption
    assert pg.get_by_test_id("stDataFrame").count() >= 1
    assert re.search(r"\*?\*?\d+\*?\*? row\(s\)", main.inner_text()), main.inner_text()[
        :600
    ]

    # -- promote v2 from the Compare page and export both scripts
    click(pg, "Make freq_v2 champion")
    assert_clean(pg, "champion")
    assert wait_text(pg, "Project champion: freq_v2")
    goto_page(pg, "Export")
    scorers = {}
    for name in ("freq_v1", "freq_v2"):
        select(pg, "Model", name)
        script = download(pg, "Download script", e2e_dir)
        assert "fit_glm(" in script.read_text()
        scorers[name] = download(pg, "Scorer (.easyglm)", e2e_dir)

    # -- the HTML report: both models in one self-contained file that opens in
    #    the browser without a console error and without fetching anything
    select(pg, "Model", "freq_v2")
    select(pg, "Include a comparison with", "freq_v1")
    report = download(pg, "Download HTML report", e2e_dir)
    html = report.read_text()
    assert "freq_v1" in html and "freq_v2" in html
    assert 'id="compare"' in html
    assert not re.search(r'(?:src|href)\s*=\s*["\']https?://', html)
    problems: list[str] = []
    viewer = browser.new_page()
    viewer.on(
        "console", lambda m: problems.append(m.text) if m.type == "error" else None
    )
    viewer.on("pageerror", lambda e: problems.append(str(e)))
    viewer.on(
        "request",
        lambda r: (
            problems.append(f"external request {r.url}")
            if not r.url.startswith("file:")
            else None
        ),
    )
    viewer.goto(report.resolve().as_uri(), wait_until="networkidle")
    viewer.wait_for_timeout(300)
    assert viewer.locator("section.variable").count() >= 7  # one per rating factor
    viewer.close()
    assert problems == []
    # the Gini shown for freq_v1 on the holdout equals workflow.gini on the
    # downloaded scorer (same split seed, same rows)
    out = run_python(
        SERVER_PYTHON,
        f"""
import numpy as np, polars as pl
from easy_glm.engine import RateModel
from easy_glm.workflow import gini
df = pl.read_parquet({str(_project_path.parent / 'policies.parquet')!r})
df = df.filter(pl.col('Exposure') > 0.02)
df = df.with_columns(pl.col('Area').cast(pl.Utf8).replace({{'E': 'D', 'F': 'D'}}))
is_train = np.random.default_rng(7).random(df.height) < 0.7
hold = df.filter(pl.Series(~is_train))
rm = RateModel.from_json({str(scorers['freq_v1'])!r})
pred = rm.predict(hold, exposure_col=None)
w = hold['Exposure'].to_numpy()
print('gini', gini(hold['ClaimNb'].to_numpy(), pred * w, w))
""",
        e2e_dir,
    )
    computed = float(out.split("gini")[1].split()[0])
    assert abs(computed - shown_gini) < 5e-4, (computed, shown_gini)
    print(f"data-scientist persona: {time.time() - t0:.0f}s")
