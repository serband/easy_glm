"""Playwright helpers for Streamlit's widgets (sidebar nav, selectbox, ...)."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


def settle(pg, timeout: float = 240.0) -> float:
    """Wait until Streamlit's 'running' indicator has appeared and gone."""
    t0 = time.time()
    status = pg.locator('[data-testid="stStatusWidget"]')
    try:  # the indicator appears a moment after an action; give it a chance
        status.wait_for(state="attached", timeout=1500)
    except Exception:  # noqa: BLE001 - fast reruns never show it
        pass
    try:
        status.wait_for(state="detached", timeout=timeout * 1000)
    except Exception:  # noqa: BLE001
        pass
    pg.wait_for_timeout(150)
    return time.time() - t0


def wait_text(pg, text: str, timeout: float = 15.0) -> bool:
    """Poll the main panel until ``text`` appears (Streamlit may still be
    delivering the rerun after ``settle`` returns)."""
    t0 = time.time()
    main = pg.locator('[data-testid="stMain"]')
    while time.time() - t0 < timeout:
        if text in main.inner_text():
            return True
        pg.wait_for_timeout(250)
    return False


def edit_grid_cell(
    pg,
    grid_index: int,
    row: int,
    value: str,
    *,
    expect: str,
    column: int | None = None,
) -> None:
    """Type ``value`` into data row ``row`` (0-based, header excluded) of the
    ``grid_index``-th data editor on the page, then wait until ``expect`` shows
    in the main panel; retried up to three times because the editor is a canvas
    inside a scroller that intercepts pointer events (double-click opens the
    overlay editor, Enter commits and reruns).

    The cell is the row's **last** column by default. ``column`` (0-based) picks
    another one: the grid is a canvas, so there is no element to address — the
    row's first cell is clicked and the selection is walked right with the arrow
    keys, then Enter opens the same overlay editor."""
    last_seen = ""
    for attempt in range(3):
        grid = pg.get_by_test_id("stDataFrame").nth(grid_index)
        grid.scroll_into_view_if_needed()
        pg.wait_for_timeout(300)
        scroller = grid.locator(".stDataFrameGlideDataEditor").first
        box = scroller.bounding_box()
        assert box, "grid not visible"
        y = 52 + 35 * row
        if column is None:
            scroller.dblclick(position={"x": box["width"] - 40, "y": y})
        else:
            scroller.click(position={"x": 40, "y": y})
            pg.wait_for_timeout(200)
            for _ in range(column):
                pg.keyboard.press("ArrowRight")
            pg.wait_for_timeout(200)
            pg.keyboard.press("Enter")  # open the editor on the selected cell
        pg.wait_for_timeout(300)
        overlay = pg.locator("#portal input, #portal textarea")
        shots = os.environ.get("EASY_GLM_E2E_SHOTS")  # diagnostics for a flaky edit
        if shots:
            pg.screenshot(path=f"{shots}/edit_{attempt}_overlay.png", full_page=True)
            print(
                f"[edit_grid_cell] attempt {attempt}: box={box} "
                f"grids={pg.get_by_test_id('stDataFrame').count()} overlay={overlay.count()}"
            )
        if overlay.count() == 0:
            pg.keyboard.press("Escape")
            continue
        for _ in range(16):
            pg.keyboard.press("Backspace")
        for _ in range(16):
            pg.keyboard.press("Delete")
        pg.keyboard.type(value)
        if overlay.first.input_value() != value:
            if shots:
                print(f"[edit_grid_cell] overlay value {overlay.first.input_value()!r}")
            pg.keyboard.press("Escape")
            continue
        pg.wait_for_timeout(400)  # let the number editor flush the typed value
        pg.keyboard.press("Enter")
        settle(pg)
        if wait_text(pg, expect, timeout=10):
            return
        last_seen = pg.locator('[data-testid="stMain"]').inner_text()[:400]
        if shots:
            pg.screenshot(path=f"{shots}/edit_{attempt}_after.png", full_page=True)
            print("[edit_grid_cell] committed but no effect")
    raise AssertionError(
        f"grid edit did not register after 3 attempts; page: {last_seen}"
    )


def goto_page(pg, title: str) -> None:
    """Client-side navigation through the sidebar (keeps the session)."""
    pg.get_by_test_id("stSidebarNav").get_by_role("link", name=title).click()
    settle(pg)


def exceptions(pg) -> list[str]:
    return pg.locator('[data-testid="stException"]').all_inner_texts()


def assert_clean(pg, where: str = "") -> None:
    exc = exceptions(pg)
    assert not exc, f"{where}: {exc[0][:800]}"


def select(pg, label: str, value: str) -> None:
    """Pick ``value`` in the selectbox whose label contains ``label``."""
    box = pg.get_by_test_id("stSelectbox").filter(has_text=label).first
    box.locator("input").click()
    box.locator("input").fill(value)
    pg.keyboard.press("Enter")
    settle(pg)


def click(pg, name: str, *, exact: bool = True) -> None:
    pg.get_by_role("button", name=name, exact=exact).first.click()
    settle(pg)


def tab(pg, name: str) -> None:
    pg.get_by_role("tab", name=name, exact=True).click()
    settle(pg)


def fill(pg, label: str, value: str) -> None:
    box = pg.get_by_label(label, exact=False).first
    box.fill(value)
    box.press("Enter")
    settle(pg)


def download(pg, button: str, folder: Path) -> Path:
    btn = pg.get_by_role("button", name=button, exact=False).first
    try:
        btn.wait_for(state="visible", timeout=15_000)
    except Exception:  # noqa: BLE001 - dump the page so the failure is diagnosable
        shot = folder / f"missing_{button.split()[0].lower()}.png"
        pg.screenshot(path=str(shot), full_page=True)
        text = pg.locator('[data-testid="stMain"]').inner_text()
        raise AssertionError(
            f"no button {button!r} on the page (screenshot {shot}):\n{text[:1200]}"
        ) from None
    with pg.expect_download(timeout=120_000) as d:
        btn.click()
    path = folder / d.value.suggested_filename
    d.value.save_as(path)
    return path


def run_python(python: str, code: str, cwd: Path) -> str:
    proc = subprocess.run(
        [python, "-c", code], cwd=cwd, capture_output=True, text=True, timeout=900
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stdout


def texts(pg, test_id: str) -> list[str]:
    return pg.get_by_test_id(test_id).all_inner_texts()
