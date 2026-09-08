"""Project roles and split gate sidebar navigation and direct page requests."""

import json
import sys
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from test_w3_hardening import _frame, _project, wk

import easy_glm.app as app

DOWNSTREAM = ("Explore", "Model", "Diagnostics", "Compare", "Rate tables", "Export")


@pytest.fixture
def navigation(tmp_path, monkeypatch):
    data = tmp_path / "data.parquet"
    _frame().write_parquet(data)
    project = _project(data)
    path = tmp_path / "navigation.easyglm-project.json"
    pages = {}
    original = st.navigation

    def capture(groups, **kwargs):
        pages.update({page.title: page for page in groups["Workflow"]})
        return original(groups, **kwargs)

    monkeypatch.setattr(st, "navigation", capture)
    monkeypatch.setattr(sys, "argv", ["main.py", f"--project={path}"])

    def start():
        project.to_json(path)
        at = AppTest.from_file(
            str(Path(app.__file__).with_name("main.py")), default_timeout=60
        ).run()
        assert not at.exception
        return at

    return project, pages, start


def visit(at, pages, title):
    # AppTest.switch_page only supports file-backed pages. Use the registered
    # page hash to exercise the same routing for callable pages/direct URLs.
    at._page_hash = pages[title]._script_hash
    at.run()
    assert not at.exception


def assert_locked(pages):
    assert all(pages[name].visibility == "hidden" for name in DOWNSTREAM)
    assert pages["Variables"].visibility == "visible"
    assert pages["Project & data"].visibility == "visible"


@pytest.mark.parametrize("title", DOWNSTREAM)
def test_missing_project_target_blocks_even_existing_model_and_direct_url(
    navigation, title
):
    project, pages, start = navigation
    project.data.roles.pop("ClaimNb")
    # A model override must not stand in for project roles needed by Explore.
    assert project.models["freq"].target == "ClaimNb"
    at = start()
    assert_locked(pages)
    visit(at, pages, title)
    assert [heading.value for heading in at.title] == ["Variables"]
    assert not any(button.label == "Fit model" for button in at.button)


def test_valid_roles_unlock_without_a_model_or_all_columns_assigned(navigation):
    project, pages, start = navigation
    project.models.clear()
    project.data.roles.pop("Region")
    at = start()
    assert all(pages[name].visibility == "visible" for name in DOWNSTREAM)
    visit(at, pages, "Model")
    assert [heading.value for heading in at.title] == ["Model design and fit"]
    assert not any(button.label == "Fit model" for button in at.button)


@pytest.mark.parametrize(
    "role_change",
    ["no_predictors", "missing_target", "two_targets", "missing_predictor"],
)
def test_invalid_role_assignments_block_navigation(navigation, role_change):
    project, pages, start = navigation
    if role_change == "no_predictors":
        project.data.roles = {
            name: role
            for name, role in project.data.roles.items()
            if role != "predictor"
        }
    elif role_change == "missing_target":
        project.data.roles.pop("ClaimNb")
        project.data.roles["gone"] = "target"
    elif role_change == "two_targets":
        project.data.roles["Exposure"] = "target"
    else:
        project.data.roles["gone"] = "predictor"
    start()
    assert_locked(pages)


def test_split_still_required_after_roles_assigned(navigation):
    project, pages, start = navigation
    project.data.split.column = "missing_split"
    at = start()
    assert_locked(pages)
    assert any("split on Variables" in item.value for item in at.sidebar.caption)
    visit(at, pages, "Explore")
    assert [heading.value for heading in at.title] == ["Variables"]


def test_json_assignment_unlocks_and_table_removal_relocks(navigation):
    project, pages, start = navigation
    project.data.roles.pop("ClaimNb")
    at = start()
    visit(at, pages, "Variables")
    at.toggle(key=wk(at, "bulk_roles_toggle")).set_value(True).run()
    editor = at.text_area(key=wk(at, "bulk_roles_json_v2"))
    payload = json.loads(editor.value)
    payload["roles"]["unassigned"].remove("ClaimNb")
    payload["assignments"]["target"] = "ClaimNb"
    editor.set_value(json.dumps(payload)).run()
    assert_locked(pages)  # unapplied text cannot unlock modelling
    at.button(key=wk(at, "bulk_roles_apply")).click().run()
    assert all(pages[name].visibility == "visible" for name in DOWNSTREAM)
    at.toggle(key=wk(at, "bulk_roles_toggle")).set_value(False).run()
    at.session_state[wk(at, "roles_grid")] = {
        "edited_rows": {1: {"role": "unassigned"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run()
    assert_locked(pages)
    visit(at, pages, "Model")
    assert [heading.value for heading in at.title] == ["Variables"]


def test_external_role_removal_relocks_on_rerun(navigation):
    _project, pages, start = navigation
    at = start()
    visit(at, pages, "Model")
    at.session_state["project"].apply_role_change("ClaimNb", "ignore")
    at.run()
    assert not at.exception
    assert_locked(pages)
    assert [heading.value for heading in at.title] == ["Variables"]
