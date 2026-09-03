"""Persona run — the data scientist comparing models.

freq_v1 (CV lasso) is fitted; freq_v2 is created and given an interaction;
Density is switched to a linear term (the variable design is shared by all
models, so freq_v1 goes stale and is refitted); both are compared on the
Diagnostics page (challenger overlay, Gini, double lift), the pair search runs,
freq_v2 is promoted to champion, both scripts and scorers export, and the Gini
shown on screen is recomputed from the downloaded scorer. TODO(D3): assert on
the Compare page once it exists.
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
    settle,
    tab,
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

    # -- Design: interaction + linear Density for freq_v2
    goto_page(pg, "Design")
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

    # -- Diagnostics: challenger overlay + pair search on v1
    goto_page(pg, "Diagnostics")
    select(pg, "Model", "freq_v1")
    select(pg, "Compare with", "freq_v2")
    assert_clean(pg, "challenger")
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

    # -- promote v2 and export both scripts
    goto_page(pg, "Model")
    select(pg, "Model", "freq_v2")
    click(pg, "Make champion")
    assert_clean(pg, "champion")
    goto_page(pg, "Export")
    scorers = {}
    for name in ("freq_v1", "freq_v2"):
        select(pg, "Model", name)
        script = download(pg, "Download script", e2e_dir)
        assert "fit_glm(" in script.read_text()
        scorers[name] = download(pg, "Scorer (.easyglm)", e2e_dir)
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
