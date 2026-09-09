"""The local API's real model-to-results path and stale-fit protections."""

from __future__ import annotations

import time
from copy import deepcopy

import numpy as np
import polars as pl
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from easy_glm.desktop.modeling import ModelEdit, edit_model
from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import Interaction, ModelConfig, Project, VariableDesign


@pytest.fixture
def model_session():
    rng = np.random.default_rng(44)
    raw = pl.DataFrame(
        {
            "Claims": rng.poisson(0.4, 1200),
            "Exposure": np.ones(1200),
            "Age": rng.integers(18, 80, 1200),
            "Region": rng.choice(["A", "B", "C"], 1200),
        }
    )
    project = Project(name="Model test")
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "Age": "predictor",
        "Region": "predictor",
    }
    project.data.split.mode = "random"
    with TestClient(
        create_app(project, raw, port=8780), base_url="http://127.0.0.1:8780"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        yield client, project, raw


def revision(client):
    snapshot = client.get("/api/variables").json()
    return {key: snapshot[key] for key in ("session_id", "revision")}


def save_model(client):
    response = client.post(
        "/api/models/save",
        json={
            **revision(client),
            "name": "Frequency",
            "create": True,
            "fields": {
                "predictors": ["Age", "Region"],
                "divide_target_by_weight": True,
                "penalty": {"alpha": 0.01, "cv": None},
            },
            "n_bins": 8,
        },
    )
    assert response.status_code == 200, response.text


def wait_fit(client):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = client.get("/api/jobs").json()["Frequency"]
        if result["status"] not in ("queued", "running"):
            return result
        time.sleep(0.05)
    pytest.fail("Fit did not complete within 30 seconds")


def test_real_fit_diagnostics_tables_and_invalidation(model_session):
    client, project, _ = model_session
    info = client.get("/api/workbench").json()
    assert info["problems"] == []
    assert info["split"]["mode"] == "random"
    assert not project.column_with_role("split")
    assert sum(info["counts"].values()) == 1200
    save_model(client)
    assert (
        client.post("/api/models/Frequency/fit", json=revision(client)).status_code
        == 202
    )
    start = time.perf_counter()
    assert client.get("/health").status_code == 200
    assert client.get("/api/variables").status_code == 200
    assert time.perf_counter() - start < 1
    status = wait_fit(client)
    assert status["status"] == "complete", status
    result = client.get("/api/results/Frequency").json()
    assert set(result["metrics"]) == {"train", "holdout", "all"}
    assert result["metrics"]["train"]["expected"] > 0
    assert result["lift"]["holdout"]
    assert result["base_rate"] > 0
    table = client.get("/api/results/Frequency/table?variable=Age&limit=2").json()
    assert len(table["rows"]) == 2
    assert table["total"] > 2
    assert "relativity" in table["columns"]
    assert (
        client.get("/api/results/Frequency/table?variable=Age&limit=501").status_code
        == 422
    )
    state = client.get("/api/variables").json()
    setup = state["setup"]
    setup["renames"]["Age"] = "DriverAge"
    assert (
        client.post(
            "/api/variables/apply", json={**revision(client), "setup": setup}
        ).status_code
        == 200
    )
    assert client.get("/api/jobs").json()["Frequency"]["status"] == "stale"
    assert client.get("/api/results/Frequency").status_code == 409
    assert not project.models  # input project never mutated


def test_missing_split_can_be_set_without_a_source_split_role(model_session):
    client, _, _ = model_session
    bad = client.post(
        "/api/split",
        json={
            **revision(client),
            "mode": "column",
            "column": "missing",
            "train_value": 1,
        },
    )
    assert bad.status_code == 422
    response = client.post(
        "/api/split",
        json={
            **revision(client),
            "mode": "random",
            "column": "test_split",
            "fraction": 0.8,
            "seed": 77,
        },
    )
    assert response.status_code == 200
    info = client.get("/api/workbench").json()
    assert info["counts"]["train"] > 0 and info["counts"]["holdout"] > 0
    assert info["split"]["seed"] == 77
    assert info["problems"] == []


def test_bad_model_and_concurrent_or_stale_request_are_refused(model_session):
    client, _, _ = model_session
    assert client.get("/api/results/Frequency").status_code == 409
    invalid = client.post(
        "/api/models/save",
        json={
            **revision(client),
            "name": "Bad",
            "create": True,
            "fields": {"predictors": ["Claims"]},
        },
    )
    assert invalid.status_code == 422
    save_model(client)
    old = revision(client)
    assert client.post("/api/models/Frequency/fit", json=old).status_code == 202
    assert client.post("/api/models/Frequency/fit", json=old).status_code == 409
    deadline = time.monotonic() + 5
    while client.get("/api/jobs").json()["Frequency"]["status"] == "queued":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert client.get("/api/jobs").json()["Frequency"]["status"] == "running"
    assert client.post("/api/models/Frequency/cancel", json=old).status_code == 200
    assert client.get("/api/jobs").json()["Frequency"]["status"] == "cancelled"
    assert client.get("/api/results/Frequency").status_code == 409


def test_existing_interactions_and_expert_settings_survive_basic_edits():
    project = Project()
    project.data.roles = {"Y": "target", "A": "predictor", "B": "predictor"}
    project.data.split.mode = "random"
    project.models["Existing"] = ModelConfig(
        target="Y",
        predictors=["A", "B"],
        interactions=[Interaction("A", "B")],
        monotone={"A": "increasing"},
        notes="keep notes",
    )
    project.design.variables["A"] = VariableDesign(
        kind="linear", knots=[20, 40], clamp=[0, 80], penalty_weight=2
    )
    before = deepcopy(project.to_dict())
    result = edit_model(
        project,
        ModelEdit(
            session_id="test",
            revision=0,
            name="Existing",
            fields={"penalty": {"alpha": 0.02, "cv": None}},
        ),
    )
    assert (
        result.models["Existing"].interactions
        == project.models["Existing"].interactions
    )
    assert result.models["Existing"].monotone == {"A": "increasing"}
    assert result.design == project.design
    assert project.to_dict() == before
    with pytest.raises(ValueError):
        edit_model(
            project,
            ModelEdit(
                session_id="test",
                revision=0,
                name="Existing",
                fields={"predictors": ["A"]},
            ),
        )


def test_model_and_split_edits_invalidate_completed_results(model_session):
    client, _, _ = model_session
    save_model(client)
    for endpoint, changes in (
        (
            "/api/models/save",
            {"name": "Frequency", "fields": {"penalty": {"alpha": 0.02}}},
        ),
        ("/api/split", {"mode": "random", "column": "traintest", "seed": 99}),
    ):
        assert (
            client.post("/api/models/Frequency/fit", json=revision(client)).status_code
            == 202
        )
        assert wait_fit(client)["status"] == "complete"
        assert (
            client.post(endpoint, json={**revision(client), **changes}).status_code
            == 200
        )
        assert client.get("/api/jobs").json()["Frequency"]["status"] == "stale"
        assert client.get("/api/results/Frequency").status_code == 409


def test_worker_failure_is_actionable_and_has_no_results(model_session):
    client, _, _ = model_session
    save_model(client)
    assert (
        client.post(
            "/api/models/save",
            json={
                **revision(client),
                "name": "Frequency",
                "fields": {"family": "gamma"},
            },
        ).status_code
        == 200
    )
    assert (
        client.post("/api/models/Frequency/fit", json=revision(client)).status_code
        == 202
    )
    status = wait_fit(client)
    assert status["status"] == "failed"
    assert status["message"]
    assert not status["applicable"]
    assert client.get("/api/results/Frequency").status_code == 409
