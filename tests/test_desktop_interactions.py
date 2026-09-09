"""Atomic model interaction edits preserve fitted and post-fit state boundaries."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import pytest
from test_desktop_models import model_session as model_session
from test_desktop_models import revision, save_model, wait_fit

from easy_glm.desktop.modeling import ModelEdit, edit_model
from easy_glm.workflow.project import (
    Adjustment,
    Interaction,
    ModelConfig,
    Project,
    TableSnapshot,
    VariableDesign,
)


def interaction_project():
    project = Project()
    project.data.roles = {
        "Y": "target",
        "A": "predictor",
        "B": "predictor",
        "C": "predictor",
    }
    project.data.split.mode = "random"
    pairs = [
        Interaction(
            "A", "B", min_cell_exposure=0.00123456789, penalty_weight=2.7, alpha=0.07
        ),
        Interaction("A", "C"),
    ]
    edits = [
        Adjustment("A", 0, 10, 1.2),
        Adjustment(pairs[0].name, 0, 10, 1.3, from_b=0, to_b=5, cell=True),
        Adjustment(pairs[1].name, 0, 10, 0.9, from_b=5, to_b=10, cell=True),
    ]
    project.models["Frequency"] = ModelConfig(
        target="Y",
        predictors=["A", "B", "C"],
        interactions=pairs,
        monotone={"A": "increasing"},
        notes="Keep these notes",
        base_rate_override=0.12,
        adjustments=deepcopy(edits),
        snapshots=[
            TableSnapshot(
                "Reviewed", adjustments=deepcopy(edits), base_rate_override=0.09
            )
        ],
    )
    project.design.variables["A"] = VariableDesign(
        kind="linear", knots=[20, 40], clamp=[0, 80], penalty_weight=2
    )
    return project


def edited(project, interactions, **fields):
    return edit_model(
        project,
        ModelEdit(
            session_id="test",
            revision=0,
            name="Frequency",
            fields={"interactions": interactions, **fields},
        ),
    )


def test_add_pair_uses_defaults_and_keeps_advanced_original_settings():
    project = interaction_project()
    before = project.to_dict()
    result = edited(
        project, [{"a": "A", "b": "B"}, {"a": "A", "b": "C"}, {"a": "B", "b": "C"}]
    )
    cfg = result.models["Frequency"]
    assert cfg.interactions[:2] == project.models["Frequency"].interactions
    assert cfg.interactions[2] == Interaction("B", "C")
    assert cfg.adjustments == project.models["Frequency"].adjustments
    assert cfg.snapshots == project.models["Frequency"].snapshots
    assert cfg.monotone == {"A": "increasing"}
    assert cfg.notes == "Keep these notes"
    assert cfg.base_rate_override == 0.12
    assert result.design == project.design
    assert project.to_dict() == before


def test_pair_reversal_is_a_noop_and_keeps_cell_coordinates_and_legacy_alpha():
    project = interaction_project()
    pair = asdict(project.models["Frequency"].interactions[0])
    pair.update(a="B", b="A")
    assert edited(project, [pair, {"a": "C", "b": "A"}]).to_dict() == project.to_dict()


def test_explicit_removal_cleans_only_its_adjustments_and_snapshots():
    project = interaction_project()
    before = project.to_dict()
    result = edited(project, [{"a": "A", "b": "C"}], predictors=["A", "C"])
    cfg = result.models["Frequency"]
    removed = project.models["Frequency"].interactions[0].name
    expected = [
        a for a in project.models["Frequency"].adjustments if a.variable != removed
    ]
    assert cfg.adjustments == expected
    assert cfg.snapshots[0].adjustments == expected
    assert cfg.snapshots[0].base_rate_override == 0.09
    assert cfg.base_rate_override == 0.12
    assert cfg.interactions == [Interaction("A", "C")]
    assert project.to_dict() == before


def test_empty_pair_list_removes_both_pairs_but_keeps_main_adjustments():
    project = interaction_project()
    result = edited(project, [])
    cfg = result.models["Frequency"]
    assert cfg.interactions == []
    assert cfg.adjustments == project.models["Frequency"].adjustments[:1]
    assert cfg.snapshots[0].adjustments == cfg.adjustments


def test_settings_edit_preserves_legacy_alpha_and_other_pairs():
    project = interaction_project()
    result = edited(
        project,
        [
            {"a": "A", "b": "B", "min_cell_exposure": 0, "penalty_weight": 0},
            {"a": "A", "b": "C"},
        ],
    )
    cfg = result.models["Frequency"]
    assert cfg.interactions[0] == Interaction("A", "B", 0, 0, alpha=0.07)
    assert cfg.interactions[1] == project.models["Frequency"].interactions[1]
    assert cfg.adjustments == project.models["Frequency"].adjustments


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        "A × B",
        [None],
        ["A × B"],
        [{"a": "A"}],
        [{"a": [], "b": "B"}],
        [{"a": " ", "b": "B"}],
        [{"a": "A", "b": "A"}],
        [{"a": "A", "b": "B"}, {"a": "B", "b": "A"}],
        [{"a": "A", "b": "missing"}],
        [{"a": "A", "b": "Y"}],
        [{"a": "A", "b": "B", "unsupported": 1}],
        [{"a": "A", "b": "B", "alpha": 0.03}],
        [{"a": "B", "b": "C", "alpha": 0.03}],
    ],
)
def test_invalid_pair_payload_is_atomic(value):
    project = interaction_project()
    before = project.to_dict()
    with pytest.raises(ValueError):
        edited(project, value)
    assert project.to_dict() == before


@pytest.mark.parametrize(
    "field,bad",
    [
        ("min_cell_exposure", -0.1),
        ("min_cell_exposure", 1),
        ("min_cell_exposure", True),
        ("min_cell_exposure", "0.005"),
        ("min_cell_exposure", None),
        ("min_cell_exposure", float("nan")),
        ("min_cell_exposure", float("inf")),
        ("penalty_weight", -1),
        ("penalty_weight", True),
        ("penalty_weight", "1"),
        ("penalty_weight", None),
        ("penalty_weight", float("nan")),
        ("penalty_weight", float("inf")),
    ],
)
def test_invalid_numeric_setting_is_atomic(field, bad):
    project = interaction_project()
    before = project.to_dict()
    with pytest.raises(ValueError):
        edited(project, [{"a": "A", "b": "B", field: bad}])
    assert project.to_dict() == before


def test_parent_deselection_requires_explicit_pair_removal():
    project = interaction_project()
    before = project.to_dict()
    with pytest.raises(ValueError, match="interaction parent 'B'"):
        edited(project, [{"a": "A", "b": "B"}], predictors=["A", "C"])
    assert project.to_dict() == before


def test_interaction_api_noop_retains_fit_and_change_requires_explicit_refit(
    model_session,
):
    client, original, _ = model_session
    save_model(client)
    pair = {"a": "Age", "b": "Region", "min_cell_exposure": 0.005, "penalty_weight": 1}
    response = client.post(
        "/api/models/save",
        json={
            **revision(client),
            "name": "Frequency",
            "fields": {"interactions": [pair]},
        },
    )
    assert response.status_code == 200, response.text
    assert client.get("/api/jobs").json() == {}
    assert client.get("/api/workbench").json()["models"]["Frequency"][
        "interactions"
    ] == [{**pair, "alpha": None}]
    assert (
        client.post("/api/models/Frequency/fit", json=revision(client)).status_code
        == 202
    )
    fit = wait_fit(client)
    assert fit["status"] == "complete", fit
    current = revision(client)
    before = client.get("/api/project").json()
    invalid = client.post(
        "/api/models/save",
        json={
            **current,
            "name": "Frequency",
            "fields": {"interactions": [pair, {"a": "Region", "b": "Age"}]},
        },
    )
    assert invalid.status_code == 422
    assert client.get("/api/project").json() == before
    noop = client.post(
        "/api/models/save",
        json={
            **current,
            "name": "Frequency",
            "fields": {"interactions": [{**pair, "a": "Region", "b": "Age"}]},
        },
    )
    assert noop.status_code == 200, noop.text
    assert revision(client) == current
    assert client.get("/api/jobs").json()["Frequency"]["id"] == fit["id"]
    assert client.get("/api/results/Frequency").status_code == 200
    changed = client.post(
        "/api/models/save",
        json={
            **current,
            "name": "Frequency",
            "fields": {"interactions": [{**pair, "min_cell_exposure": 0.01}]},
        },
    )
    assert changed.status_code == 200, changed.text
    stale = client.get("/api/jobs").json()["Frequency"]
    assert stale["status"] == "stale"
    assert stale["id"] == fit["id"]
    assert client.get("/api/results/Frequency").status_code == 409
    assert not original.models
