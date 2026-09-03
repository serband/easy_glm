"""Screenshot driver for scripts/checks/d3_d4_compare_report.py (runs under an
interpreter that has Playwright; it needs no easy_glm). Usage:

    python _d3_screens.py <base_url> <out_dir>

Fits both models, photographs the Compare page, downloads the HTML report from
the Export page and photographs two of its sections. Prints DONE on success.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)


def settle(pg, timeout=300):
    t0 = time.time()
    pg.wait_for_timeout(500)
    while time.time() - t0 < timeout:
        if pg.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        pg.wait_for_timeout(300)
    pg.wait_for_timeout(400)


def nav(pg, title):
    pg.get_by_test_id("stSidebarNav").get_by_role("link", name=title).click()
    settle(pg)


def select(pg, label, value, scope="stMain"):
    box = (
        pg.get_by_test_id(scope)
        .get_by_test_id("stSelectbox")
        .filter(has_text=label)
        .first
    )
    box.locator("input").click()
    box.locator("input").fill(value)
    pg.keyboard.press("Enter")
    settle(pg)


def shot(pg, name, anchor=None):
    """A viewport-sized picture (kept small) scrolled to ``anchor``."""
    if anchor is not None:
        anchor.scroll_into_view_if_needed()
        pg.wait_for_timeout(400)
    else:
        pg.mouse.wheel(0, -20000)
        pg.wait_for_timeout(300)
    path = OUT / f"{name}.png"
    pg.screenshot(path=str(path))
    exc = pg.locator('[data-testid="stException"]').all_inner_texts()
    print(f"{name}: {path.stat().st_size // 1024} KB, exceptions={len(exc)}")


with sync_playwright() as p:
    browser = p.chromium.launch()
    pg = browser.new_page(
        viewport={"width": 1400, "height": 900}, device_scale_factor=1
    )
    pg.goto(BASE, wait_until="networkidle")
    settle(pg)

    # -- fit both models
    nav(pg, "Model")
    for model in ("freq_v1", "freq_v2"):
        select(pg, "Model", model)
        pg.get_by_role("button", name="Fit model", exact=True).click()
        settle(pg, 600)

    # -- Compare page
    nav(pg, "Compare")
    main = pg.locator('[data-testid="stMain"]')
    # anchor on the metrics table so the holdout rows are in frame, not cut off
    shot(pg, "d3_compare_metrics", pg.get_by_text("Metrics side by side").first)
    pg.get_by_role("tab", name="A/E by variable", exact=True).click()
    settle(pg)
    select(pg, "Variable", "DrivAge")
    shot(pg, "d3_compare_ae", pg.locator(".js-plotly-plot").first)
    pg.get_by_role("tab", name="Relativities that differ", exact=True).click()
    settle(pg)
    # anchor on the tolerance box so the headline, the caption and the table
    # are all in frame (there are three grids on the page now)
    shot(pg, "d3_compare_diff", pg.get_by_text("Report a band when").first)
    assert "row(s)" in main.inner_text(), main.inner_text()[:400]

    # -- the report, downloaded from the Export page and opened in the browser
    nav(pg, "Export")
    select(pg, "Model", "freq_v2")  # the richer model: interaction + linear term
    select(pg, "Include a comparison with", "freq_v1")
    with pg.expect_download(timeout=180_000) as dl:
        pg.get_by_role("button", name="Download HTML report", exact=False).first.click()
    report = OUT.parent / "report.html"
    dl.value.save_as(report)
    print(f"report: {report.stat().st_size // 1024} KB")

    problems: list[str] = []
    viewer = browser.new_page(
        viewport={"width": 1200, "height": 900}, device_scale_factor=1
    )
    viewer.on(
        "console", lambda m: problems.append(m.text) if m.type == "error" else None
    )
    viewer.on("pageerror", lambda e: problems.append(str(e)))
    viewer.goto(report.resolve().as_uri(), wait_until="networkidle")
    viewer.wait_for_timeout(400)
    shot(viewer, "d4_report_summary")
    shot(viewer, "d4_report_variable", viewer.locator("#var-drivage h4").first)
    shot(
        viewer,
        "d4_report_interaction",
        viewer.locator("#interactions h4").first,
    )
    shot(viewer, "d4_report_compare", viewer.locator("#compare h3").first)
    print(f"console problems: {problems}")
    assert not problems, problems
    report.unlink(missing_ok=True)
    browser.close()

print("DONE")
