"""Durable provenance and safe factor scope for the visual model flow."""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from easy_glm.app.state import model_hash as streamlit_model_hash
from easy_glm.desktop.jobs import model_key
from easy_glm.desktop.modeling import stack_factor_stages, validate_stack_factor_scope
from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import ModelConfig, ModelLineage, Project
from easy_glm.workflow.stack_types import ComponentRecipe


def _project() -> tuple[Project, pl.DataFrame]:
    raw = pl.DataFrame(
        {
            "y": [0, 1],
            "w": [1.0, 1.0],
            "age": [30, 40],
            "region": ["A", "B"],
            "id": [1, 2],
        }
    )
    project = Project(name="flow")
    project.data.roles = {"y": "target", "w": "weight", "age": "predictor", "id": "id"}
    project.models["Base"] = ModelConfig(target="y", weight="w")
    return project, raw


def test_lineage_round_trip_is_omitted_when_absent_and_ignored_by_fit_key():
    project, _ = _project()
    before = model_key(project, "Base")
    streamlit_before = streamlit_model_hash(project, "Base")
    project.models["Base"].lineage = ModelLineage(
        parent_name="Older",
        parent_fit_id="123",
        parent_key="a" * 64,
        parent_prefix_fingerprint="b" * 64,
        prefix_length=1,
    )
    encoded = project.to_dict()
    assert encoded["version"] == 5
    restored = Project.from_dict(encoded)
    assert restored.models["Base"].lineage == project.models["Base"].lineage
    assert model_key(project, "Base") == before
    assert streamlit_model_hash(project, "Base") == streamlit_before
    assert streamlit_model_hash(restored, "Base") == streamlit_before

    project.models["Base"].lineage = None
    without_lineage = project.to_dict()
    assert without_lineage["version"] == 2
    assert "lineage" not in without_lineage["models"]["Base"]


def test_stack_scope_excludes_controls_but_allows_unassigned_and_repeated_factors():
    project, raw = _project()
    config = project.models["Base"]
    config.components = [
        ComponentRecipe("first", "First", "glm", ("age", "region")),
        ComponentRecipe("again", "Again", "glm", ("age",)),
    ]
    stages = stack_factor_stages(project, raw, set(raw.columns), config)
    assert stages[0] == ["age", "region"]
    validate_stack_factor_scope(project, raw, set(raw.columns), config)

    config.components[0] = ComponentRecipe("bad", "Bad", "glm", ("y",))
    with pytest.raises(ValueError, match="unavailable"):
        validate_stack_factor_scope(project, raw, set(raw.columns), config)


def test_catalogue_attributes_only_enter_scope_after_its_pinned_lookup():
    project, raw = _project()
    config = project.models["Base"]
    config.components = [
        ComponentRecipe("bad", "Bad", "glm", ("make",)),
        ComponentRecipe("lookup", "Lookup", "catalogue", ("make",), {"key": "region"}),
        ComponentRecipe("after", "After", "glm", ("make",)),
    ]
    with pytest.raises(ValueError, match="unavailable"):
        validate_stack_factor_scope(project, raw, set(raw.columns), config)

    config.components = config.components[1:]
    validate_stack_factor_scope(project, raw, set(raw.columns), config)


def test_saved_branch_can_only_refit_its_full_recipe_without_live_parent():
    rng = np.random.default_rng(4)
    raw = pl.DataFrame(
        {
            "y": rng.poisson(0.3, 180),
            "w": np.ones(180),
            "age": rng.integers(20, 70, 180),
            "region": rng.choice(["A", "B"], 180),
        }
    )
    project = Project(name="reopened")
    project.data.roles = {
        "y": "target",
        "w": "weight",
        "age": "predictor",
        "region": "predictor",
    }
    project.data.split.mode = "random"
    main = ComponentRecipe("main", "Main", "glm", ("age",), {"alpha": 0.01})
    project.models["Base"] = ModelConfig(target="y", weight="w", components=[main])
    project.models["Child"] = ModelConfig(
        target="y",
        weight="w",
        components=[
            main,
            ComponentRecipe("region", "Region", "glm", ("region",), {"alpha": 0.01}),
        ],
        lineage=ModelLineage("Base", "old-fit", "old-key", "old-prefix", 1),
    )
    with TestClient(
        create_app(project, raw, port=8891), base_url="http://127.0.0.1:8891"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]

        def revision(full_refit: bool = False) -> dict[str, object]:
            return {
                **{
                    key: client.get("/api/variables").json()[key]
                    for key in ("session_id", "revision")
                },
                "full_refit": full_refit,
            }

        assert client.post("/api/models/Child/fit", json=revision()).status_code == 409
        assert (
            client.post("/api/models/Child/fit", json=revision(True)).status_code == 202
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = client.get("/api/jobs").json()["Child"]
            if status["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert status["status"] == "complete", status


def test_full_refit_rejects_an_imported_protected_stack_factor():
    project, raw = _project()
    raw = raw.with_columns(pl.Series("policy_id", [1, 2]))
    project.data.roles["policy_id"] = "id"
    project.models["Base"].components = [
        ComponentRecipe("bad", "Bad", "glm", ("policy_id",), {"alpha": 0.01})
    ]
    project.models["Catalogue"] = ModelConfig(
        target="y",
        weight="w",
        components=[
            ComponentRecipe(
                "lookup", "Forged lookup", "catalogue", ("y",), {"key": "age"}
            )
        ],
    )
    with TestClient(
        create_app(project, raw, port=8892), base_url="http://127.0.0.1:8892"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        revision = client.get("/api/variables").json()
        response = client.post(
            "/api/models/Base/fit",
            json={
                "session_id": revision["session_id"],
                "revision": revision["revision"],
                "full_refit": True,
            },
        )
        assert response.status_code == 422
        assert "unavailable" in response.json()["detail"]
        catalogue = client.post(
            "/api/models/Catalogue/fit",
            json={
                "session_id": revision["session_id"],
                "revision": revision["revision"],
                "full_refit": True,
            },
        )
        assert catalogue.status_code == 422
        assert "protected columns" in catalogue.json()["detail"]
