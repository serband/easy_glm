"""Cross-editor project updates must not be overwritten by stale widget values."""

import json

import pytest
from test_w3_hardening import _frame, _project, _run, _script, wk

from easy_glm.workflow import Project


@pytest.fixture
def workspace(tmp_path):
    data = tmp_path / "policies.parquet"
    _frame().write_parquet(data)
    project = tmp_path / "sync.easyglm-project.json"
    _project(data).to_json(project)
    return {"data": data, "project": project}


def test_external_roles_refresh_json_without_manual_reset(workspace):
    at = _run(_script("pages_variables", str(workspace["project"])))
    at.toggle(key=wk(at, "bulk_roles_toggle")).set_value(True).run()
    p = at.session_state["_project"]
    p.apply_role_change("Region", "ignore")
    at.run()
    payload = json.loads(at.text_area(key=wk(at, "bulk_roles_json_v2")).value)
    assert "Region" not in payload["roles"]["predictor"]


def test_external_model_selection_survives_render(workspace):
    at = _run(_script("pages_model", str(workspace["project"])))
    p = at.session_state["_project"]
    p.models["freq"].predictors.remove("Region")
    at.run()
    assert "Region" not in at.multiselect(key=wk(at, "preds_freq")).value
    assert "Region" not in p.models["freq"].predictors
    p.models["freq"].predictors.append("Region")
    at.run()
    assert "Region" in at.multiselect(key=wk(at, "preds_freq")).value
    assert not at.exception


def test_table_json_roundtrip_and_saved_model_removal(workspace):
    at = _run(_script("pages_variables", str(workspace["project"])))
    key = wk(at, "roles_grid")
    at.session_state[key] = {
        "edited_rows": {5: {"role": "ignore"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run()
    assert at.session_state["_project"].data.roles["Region"] == "ignore"
    at.toggle(key=wk(at, "bulk_roles_toggle")).set_value(True).run()
    editor = at.text_area(key=wk(at, "bulk_roles_json_v2"))
    payload = json.loads(editor.value)
    assert "Region" not in payload["roles"]["predictor"]
    payload["roles"]["predictor"].append("Region")
    editor.set_value(json.dumps(payload)).run()
    at.button(key=wk(at, "bulk_roles_apply")).click().run()
    at.toggle(key=wk(at, "bulk_roles_toggle")).set_value(False).run()
    grid = at.dataframe[0].value
    assert grid.loc[grid["column"] == "Region", "role"].iloc[0] == "predictor"
    saved = Project.from_json(workspace["project"])
    assert saved.data.roles["Region"] == "predictor"
    # Re-enabling a project predictor does not add it to every fitted model.
    assert "Region" not in saved.models["freq"].predictors
    assert not at.exception


def test_unapplied_json_survives_rerun_and_external_refresh(workspace):
    at = _run(_script("pages_variables", str(workspace["project"])))
    at.toggle(key=wk(at, "bulk_roles_toggle")).set_value(True).run()
    draft = '{"unfinished":'
    at.text_area(key=wk(at, "bulk_roles_json_v2")).set_value(draft).run()
    at.run()
    assert at.text_area(key=wk(at, "bulk_roles_json_v2")).value == draft
    at.session_state["_project"].apply_role_change("Region", "ignore")
    at.run()
    assert at.session_state[wk(at, "bulk_roles_previous_draft")] == draft
    payload = json.loads(at.text_area(key=wk(at, "bulk_roles_json_v2")).value)
    assert "Region" not in payload["roles"]["predictor"]
    assert not at.exception


def test_residual_unassigned_factor_updates_roles_json_table_and_model(workspace):
    at = _run(
        _script(
            "pages_diagnostics",
            str(workspace["project"]),
            fit=True,
            prelude="S.project().models['freq'].predictors.remove('Region'); S.project().data.roles.pop('Region')",
        )
    )
    at.button(key=wk(at, "rfs_go")).click().run()
    at.multiselect(key=wk(at, "rfs_add_selection")).set_value(["Region"]).run()
    at.button(key=wk(at, "rfs_add_selected")).click().run()
    assert not at.exception
    saved = Project.from_json(workspace["project"])
    assert saved.data.roles["Region"] == "predictor"
    assert "Region" in saved.models["freq"].predictors
    at = _run(_script("pages_model", str(workspace["project"])))
    assert "Region" in at.multiselect(key=wk(at, "preds_freq")).value
    assert "Region" in at.dataframe[0].value["variable"].tolist()
    at.session_state["_page"] = "pages_variables"
    at.run()
    grid = at.dataframe[0].value
    assert grid.loc[grid["column"] == "Region", "role"].iloc[0] == "predictor"
    at.toggle(key=wk(at, "bulk_roles_toggle")).set_value(True).run()
    payload = json.loads(at.text_area(key=wk(at, "bulk_roles_json_v2")).value)
    assert "Region" in payload["roles"]["predictor"]
    assert not at.exception


def test_residual_interaction_is_visible_in_model_without_new_roles(workspace):
    at = _run(_script("pages_diagnostics", str(workspace["project"]), fit=True))
    roles = dict(at.session_state["_project"].data.roles)
    at.button(key=wk(at, "rps_go")).click().run()
    [button for button in at.button if button.label == "Add interaction"][
        0
    ].click().run()
    saved = Project.from_json(workspace["project"])
    assert len(saved.models["freq"].interactions) == 1
    interaction = saved.models["freq"].interactions[0]
    at = _run(_script("pages_model", str(workspace["project"])))
    assert any(interaction.name in info.value for info in at.info)
    assert at.session_state["_project"].data.roles == roles
    assert not at.exception
