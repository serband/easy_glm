"""Desktop new-data source lifecycle contracts."""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from easy_glm.desktop.jobs import model_key
from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import ModelConfig, Project


@pytest.fixture
def client(tmp_path):
    source = tmp_path / "new_book.csv"
    pl.DataFrame({"Band": ["A", "B"], "Exposure": [1.0, 2.0]}).write_csv(source)
    project = Project(name="Scoring sources")
    project.models["saved"] = ModelConfig(target="Claims", weight="Exposure")
    before = model_key(project, "saved")
    with TestClient(
        create_app(
            project, pl.DataFrame({"Claims": [1], "Exposure": [1.0]}), port=8892
        ),
        base_url="http://127.0.0.1:8892",
    ) as api:
        api.headers["X-EasyGLM-Token"] = api.get("/api/session").json()["token"]
        yield api, source, before, project


def _revision(api: TestClient) -> dict[str, object]:
    payload = api.get("/api/workbench").json()
    return {key: payload[key] for key in ("session_id", "revision")}


def test_file_registration_keeps_training_fit_identity_and_detects_change(
    client,
) -> None:
    api, source, before, project = client
    response = api.post(
        "/api/datasets/file",
        json={**_revision(api), "name": "Renewal book", "path": str(source)},
    )
    assert response.status_code == 200, response.text
    item = response.json()["dataset"]
    assert response.json()["revision"] == 0
    assert item["source"] == "file"
    assert item["status"] == "available"
    # Registry descriptors live outside DataConfig/model_key inputs.
    assert model_key(project, "saved") == before

    pl.DataFrame({"Band": ["changed"]}).write_csv(source)
    listed = api.get("/api/datasets").json()["datasets"]
    assert listed[0]["status"] == "changed"
    assert "changed" in listed[0]["message"].lower()


def test_scoring_sources_round_trip_is_optional_and_file_only(client) -> None:
    api, source, _before, project = client
    response = api.post(
        "/api/datasets/file",
        json={**_revision(api), "name": "Renewal book", "path": str(source)},
    )
    assert response.status_code == 200
    # The browser registry is session-only until project JSON is requested; the
    # original caller object remains untouched and old empty projects omit it.
    assert "scoring_sources" not in project.to_dict()
    item = response.json()["dataset"]
    assert item["digest"]


def test_saved_binomial_model_lists_probability_without_exposure(tmp_path) -> None:
    rng = np.random.default_rng(72)
    raw = pl.DataFrame(
        {
            "Claim": rng.binomial(1, 0.3, 140),
            "Age": rng.integers(18, 80, 140),
        }
    )
    project = Project(name="Binomial scoring")
    project.data.roles = {"Claim": "target", "Age": "predictor"}
    project.data.split.mode = "random"
    with TestClient(
        create_app(project, raw, port=8893), base_url="http://127.0.0.1:8893"
    ) as api:
        api.headers["X-EasyGLM-Token"] = api.get("/api/session").json()["token"]
        saved = api.post(
            "/api/models/save",
            json={
                **_revision(api),
                "name": "Claim probability",
                "create": True,
                "fields": {
                    "predictors": ["Age"],
                    "family": "binomial",
                    "link": "logit",
                    "penalty": {"alpha": 0.01, "cv": None},
                },
                "n_bins": 4,
            },
        )
        assert saved.status_code == 200, saved.text
        assert (
            api.post(
                "/api/models/Claim probability/fit", json=_revision(api)
            ).status_code
            == 202
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = api.get("/api/jobs").json()["Claim probability"]
            if status["status"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        assert status["status"] == "complete", status
        models = api.get("/api/scoring/models").json()["models"]
        assert len(models) == 1
        assert models[0]["probability"] is True
        assert models[0]["output_basis"] == "probability"
        assert [scope["id"] for scope in models[0]["scopes"]] == ["full"]
        assert models[0]["exposure_column"] is None
        assert models[0]["allowed_exposure_columns"] == []
        book = tmp_path / "new_claims.csv"
        pl.DataFrame({"Age": [22, 64]}).write_csv(book)
        dataset = api.post(
            "/api/datasets/file",
            json={**_revision(api), "name": "New claims", "path": str(book)},
        )
        assert dataset.status_code == 200, dataset.text
        preview = api.post(
            "/api/scoring/preview",
            json={
                **_revision(api),
                "dataset_id": dataset.json()["dataset"]["id"],
                "model_id": models[0]["id"],
                "scope": "full",
                "column_prefix": "",
            },
        )
        assert preview.status_code == 200, preview.text
        assert preview.json()["metadata"]["output_column"].startswith(
            "easyglm_full_chain_"
        )


def test_legacy_physical_model_exposes_one_real_component(tmp_path) -> None:
    rng = np.random.default_rng(73)
    raw = pl.DataFrame(
        {
            "Claims": rng.poisson(0.25, 140),
            "Exposure": np.ones(140),
            "Age": rng.integers(18, 80, 140),
        }
    )
    project = Project(name="Legacy physical scoring")
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "Age": "predictor",
    }
    project.data.split.mode = "random"
    with TestClient(
        create_app(project, raw, port=8894), base_url="http://127.0.0.1:8894"
    ) as api:
        api.headers["X-EasyGLM-Token"] = api.get("/api/session").json()["token"]
        response = api.post(
            "/api/models/save",
            json={
                **_revision(api),
                "name": "Frequency",
                "create": True,
                "fields": {
                    "predictors": ["Age"],
                    "divide_target_by_weight": True,
                    "penalty": {"alpha": 0.01, "cv": None},
                },
                "n_bins": 4,
            },
        )
        assert response.status_code == 200, response.text
        assert (
            api.post("/api/models/Frequency/fit", json=_revision(api)).status_code
            == 202
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = api.get("/api/jobs").json()["Frequency"]
            if status["status"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        assert status["status"] == "complete", status
        model = api.get("/api/scoring/models").json()["models"][0]
        assert model["kind"] == "physical"
        assert [item["component_id"] for item in model["components"]] == ["legacy"]
        assert model["allowed_exposure_columns"] == ["Exposure"]
        book = tmp_path / "new_frequency.csv"
        pl.DataFrame({"Age": [25], "Exposure": [0.5]}).write_csv(book)
        dataset = api.post(
            "/api/datasets/file",
            json={**_revision(api), "name": "Renewals", "path": str(book)},
        ).json()["dataset"]
        selected = api.post(
            "/api/scoring/preview",
            json={
                **_revision(api),
                "dataset_id": dataset["id"],
                "model_id": model["id"],
                "scope": "model",
                "component_ids": ["legacy"],
                "exposure_col": "Exposure",
            },
        )
        assert selected.status_code == 200, selected.text
