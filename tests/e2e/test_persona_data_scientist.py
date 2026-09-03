"""Persona run — the data scientist comparing models.

freq_v1 (CV lasso) is fitted, cloned to freq_v2 with an interaction and a
linear Density term, both are compared on the Diagnostics page (challenger
overlay, Gini, double lift), the pair search runs on freq_v1, freq_v2 is
promoted to champion, and both scripts export. TODO(D3): assert on the
Compare page once it exists.
"""

from __future__ import annotations

import time

from ._helpers import (
    assert_clean,
    click,
    download,
    fill,
    goto_page,
    select,
    settle,
    tab,
)


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

    # -- fit freq_v2
    goto_page(pg, "Model")
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
    assert "challenger" in main.inner_text()
    tab(pg, "Double lift")
    assert pg.locator(".js-plotly-plot").count() >= 1
    tab(pg, "Residual factors")
    click(pg, "Search pairs")
    assert_clean(pg, "pair search")
    assert "DrivAge × VehPower" in main.inner_text() or "×" in main.inner_text()

    # -- promote v2 and export both scripts
    goto_page(pg, "Model")
    select(pg, "Model", "freq_v2")
    click(pg, "Make champion")
    assert_clean(pg, "champion")
    goto_page(pg, "Export")
    for name in ("freq_v1", "freq_v2"):
        select(pg, "Model", name)
        script = download(pg, "Download script", e2e_dir)
        assert "fit_glm(" in script.read_text()
    print(f"data-scientist persona: {time.time() - t0:.0f}s")
