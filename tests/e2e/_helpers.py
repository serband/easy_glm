"""Playwright helpers for Streamlit's widgets (sidebar nav, selectbox, ...)."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


def settle(pg, timeout: float = 240.0) -> float:
    """Wait until Streamlit's 'running' indicator disappears."""
    t0 = time.time()
    pg.wait_for_timeout(500)
    while time.time() - t0 < timeout:
        if pg.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        pg.wait_for_timeout(300)
    pg.wait_for_timeout(300)
    return time.time() - t0


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
    with pg.expect_download(timeout=120_000) as d:
        pg.get_by_role("button", name=button, exact=False).first.click()
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
