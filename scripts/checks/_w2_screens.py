"""Screenshot driver for scripts/checks/w2_pages.py (runs under an interpreter
that has Playwright; it needs no easy_glm). Usage:

    python _w2_screens.py <base_url> <out_dir>
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


def select(pg, label, value):
    box = pg.get_by_test_id("stSelectbox").filter(has_text=label).first
    box.locator("input").click()
    box.locator("input").fill(value)
    pg.keyboard.press("Enter")
    settle(pg)


def shot(pg, name, locator=None):
    path = str(OUT / f"{name}.png")
    if locator is not None:
        locator.screenshot(path=path)
    else:
        pg.screenshot(path=path, full_page=True)
    exc = pg.locator('[data-testid="stException"]').all_inner_texts()
    print(f"{name}: exceptions={len(exc)}")


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)
    pg.goto(BASE, wait_until="networkidle")
    settle(pg)
    nav(pg, "Model")
    pg.get_by_role("button", name="Fit model", exact=True).click()
    settle(pg, 600)
    nav(pg, "Design")
    select(pg, "Variable", "Density")
    select(pg, "Kind", "linear")
    main = pg.locator('[data-testid="stMain"]')
    pg.get_by_text("Piecewise-linear").first.scroll_into_view_if_needed()
    shot(pg, "w2_design_linear", main)
    pg.get_by_text("Interactions", exact=True).first.scroll_into_view_if_needed()
    shot(pg, "w2_design_interactions", main)
    nav(pg, "Diagnostics")
    pg.get_by_role("tab", name="A/E by pair", exact=True).click()
    settle(pg)
    select(pg, "Rows", "DrivAge")
    select(pg, "Columns", "VehPower")
    shot(pg, "w2_diagnostics_pair", main)
    nav(pg, "Rate tables")
    opts = pg.get_by_test_id("stSelectbox").filter(has_text="Variable").first
    opts.locator("input").click()
    opts.locator("input").fill("×")
    pg.keyboard.press("Enter")
    settle(pg)
    shot(pg, "w2_tables_interaction", main)
    select(pg, "Variable", "Density")
    shot(pg, "w2_tables_linear", main)
    b.close()
print("DONE")
