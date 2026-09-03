"""Break-it run — a hostile tester on a real workbench server (W3).

The blocking findings of docs/reviews/w2-breakage.md that need a real browser
(file upload, two tabs on one project file) are driven end to end here; the
rest are AppTest cases in tests/test_w3_hardening.py. Every step must end
with a message on the page, never a traceback, and the project file on disk
must only ever hold what the user meant to keep.
"""

from __future__ import annotations

import time
from pathlib import Path

from ._helpers import (
    assert_clean,
    click,
    edit_grid_cell,
    fill,
    goto_page,
    settle,
    tab,
    wait_text,
)


def _main(pg) -> str:
    return pg.locator('[data-testid="stMain"]').inner_text()


def _click_main(pg, name: str) -> None:
    """Click a button in the main panel (the sidebar has its own Save button)."""
    pg.get_by_test_id("stMain").get_by_role("button", name=name, exact=True).click()
    settle(pg)


def _file_contains(path: Path, text: str, timeout: float = 10.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if text in path.read_text():
            return True
        time.sleep(0.25)
    return False


def test_breakit(breakit_server, browser, e2e_dir):
    srv, project_path = breakit_server
    t0 = time.time()
    original = project_path.read_text()
    pg = browser.new_page(viewport={"width": 1500, "height": 1000})
    pg.goto(srv.url, wait_until="networkidle")
    settle(pg)
    assert_clean(pg, "project page")

    # -- 1. a broken project file is uploaded: a message, the open project is
    #       untouched and its file is not rewritten
    bad = e2e_dir / "truncated.easyglm-project.json"
    bad.write_text('{"name": "x", "data": {')
    pg.locator('input[type="file"]').first.set_input_files(str(bad))
    settle(pg)
    click(pg, "Load uploaded project")
    assert_clean(pg, "bad project upload")
    assert wait_text(pg, "Not a valid easy_glm project"), _main(pg)[:800]
    assert "breakit" in pg.get_by_test_id("stSidebar").inner_text()
    assert project_path.read_text() == original

    # -- 2. saving to a folder that cannot exist: a message, and the project
    #       keeps autosaving to its own file
    fill(pg, "Project file", "/nonexistent_dir_easy_glm/x.json")
    _click_main(pg, "Save project")
    assert_clean(pg, "save to a bad path")
    assert wait_text(pg, "Could not save"), _main(pg)[:800]
    fill(pg, "Project name", "breakit renamed")
    assert _file_contains(project_path, "breakit renamed"), "autosave lost its file"

    # -- 3. a derived column that cannot be computed is refused with a message;
    #       the data steps keep working
    goto_page(pg, "Variables")
    assert_clean(pg, "variables")
    tab(pg, "Derived columns")
    fill(pg, "New column name", "boom")
    fill(pg, "Expression", "pl.col('no_such_column') * 2")
    click(pg, "Add derived column")
    assert_clean(pg, "bad derived column")
    assert wait_text(pg, "no_such_column"), _main(pg)[:800]
    assert '"boom"' not in project_path.read_text()
    goto_page(pg, "Split")
    assert_clean(pg, "split after the bad derived column")
    assert wait_text(pg, "train"), _main(pg)[:800]

    # -- 4. the roles grid in a real browser: renaming a column carries the
    #       role and the model reference *and* rewrites the row filter that
    #       names it (docs/reviews/w3-hardening.md S2), so the data steps keep
    #       working instead of "unable to find column Exposure"
    goto_page(pg, "Variables")
    assert_clean(pg, "variables before the rename")
    edit_grid_cell(
        pg,
        0,
        2,  # the Exposure row
        "exposure_years",
        column=1,  # the "rename to" column
        expect="exposure_years",
    )
    assert_clean(pg, "roles grid rename")
    saved = project_path.read_text()
    assert '"exposure_years"' in saved, saved[:800]
    assert "pl.col('exposure_years') > 0.02" in saved, "the row filter was not renamed"
    goto_page(pg, "Split")
    assert_clean(pg, "split after the rename")
    assert "unable to find column" not in _main(pg), _main(pg)[:800]
    assert wait_text(pg, "train"), _main(pg)[:800]

    # -- 5. a model name with a slash is refused before the model exists
    goto_page(pg, "Model")
    assert_clean(pg, "model")
    fill(pg, "New model name", "a/b")
    assert_clean(pg, "bad model name")
    assert wait_text(pg, "cannot contain"), _main(pg)[:800]
    assert pg.get_by_role("button", name="Create", exact=True).first.is_disabled()
    assert '"a/b"' not in project_path.read_text()

    # -- 6. two tabs on one project file: the second tab's autosave pauses
    #       instead of overwriting the first tab's work; "Reload from disk"
    #       takes the first tab's version and the tab keeps working
    pg2 = browser.new_page(viewport={"width": 1500, "height": 1000})
    pg2.goto(srv.url, wait_until="networkidle")
    settle(pg2)
    goto_page(pg2, "Model")
    assert_clean(pg2, "second tab")
    fill(pg, "Notes", "from tab one")
    assert _file_contains(project_path, "from tab one")
    fill(pg2, "Notes", "from tab two")
    assert_clean(pg2, "second tab edit")
    assert wait_text(pg2, "changed by another browser tab"), _main(pg2)[:800]
    assert "from tab two" not in project_path.read_text(), "tab two overwrote tab one"
    click(pg2, "Reload from disk")
    assert_clean(pg2, "reload from disk")
    assert not wait_text(pg2, "changed by another browser tab", timeout=2)
    notes = pg2.get_by_label("Notes", exact=False).first.input_value()
    assert notes == "from tab one", notes
    assert "from tab two" not in project_path.read_text()
    fill(pg2, "Notes", "tab two after reload")
    assert _file_contains(
        project_path, "tab two after reload"
    ), "autosave stayed paused"
    assert_clean(pg2, "second tab after reload")

    # -- 7. the other branch: the first tab (whose copy is now the stale one)
    #       overwrites, and its autosave resumes afterwards
    fill(pg, "Notes", "tab one wins")
    assert_clean(pg, "first tab edit")
    assert wait_text(pg, "changed by another browser tab"), _main(pg)[:800]
    assert "tab one wins" not in project_path.read_text()
    click(pg, "Overwrite with this tab's version")
    assert_clean(pg, "overwrite")
    assert _file_contains(project_path, "tab one wins"), "overwrite did not save"
    assert not wait_text(pg, "changed by another browser tab", timeout=2)
    fill(pg, "Notes", "tab one after overwrite")
    assert _file_contains(
        project_path, "tab one after overwrite"
    ), "autosave stayed paused after Overwrite"
    assert_clean(pg, "first tab after overwrite")
    print(f"break-it run: {time.time() - t0:.0f}s")
