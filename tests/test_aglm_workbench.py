"""Training grid preflight for AGLM interactions, without fitting CatBoost."""

from dataclasses import asdict

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from easy_glm.desktop.server import create_app
from easy_glm.workflow.model_plan import ModelPlan
from easy_glm.workflow.pair_stages import interaction_grid
from easy_glm.workflow.project import Project, VariableDesign


def test_aglm_interaction_grid_limits_are_based_on_cells() -> None:
    train = pl.DataFrame(
        {
            "Age": np.linspace(18.0, 90.0, 500),
            "VehicleAge": np.linspace(0.0, 20.0, 500),
            "Region": ["North", "South"] * 250,
        }
    )
    project = Project()
    project.design.variables["Age"] = VariableDesign(kind="aglm")
    project.design.variables["VehicleAge"] = VariableDesign(kind="aglm_l")
    too_large = interaction_grid(project, train, "Age", "VehicleAge")
    assert too_large["cells"] == 10_404
    assert not too_large["allowed"]
    assert "smaller full break grid" in too_large["message"]
    assert interaction_grid(project, train, "Age", "Region")["allowed"]

    project.design.variables["Age"].n_breaks = 5
    project.design.variables["VehicleAge"].n_breaks = 5
    coarse = interaction_grid(project, train, "Age", "VehicleAge")
    assert coarse["allowed"] and coarse["cells"] == 49


def test_workbench_grid_preview_is_read_only_and_pair_override_keeps_main_curve() -> (
    None
):
    data = pl.DataFrame(
        {
            "Claims": [0, 1] * 250,
            "Age": np.linspace(18.0, 90.0, 500),
            "VehicleAge": np.linspace(0.0, 20.0, 500),
            "traintest": [1] * 400 + [0] * 100,
        }
    )
    project = Project()
    project.data.roles = {
        "Claims": "target",
        "Age": "predictor",
        "VehicleAge": "predictor",
        "traintest": "split",
    }
    project.design.variables["Age"] = VariableDesign(kind="aglm")
    project.design.variables["VehicleAge"] = VariableDesign(kind="aglm_l")
    project.model_plan = ModelPlan.from_dict(
        {
            "format_version": 2,
            "mode": "single",
            "family": "poisson",
            "link": "log",
            "divide_target_by_weight": False,
            "models": [
                {
                    "model_id": "main",
                    "name": "Main effects",
                    "core": {
                        "component_id": "main",
                        "name": "Main effects",
                        "kind": "glm",
                        "factors": ["Age", "VehicleAge"],
                        "settings": {"alpha": 0.01},
                    },
                    "pair_corrections": [],
                }
            ],
            "adopted_root": None,
            "fit_refs": [],
        }
    )
    before = project.to_dict()
    with TestClient(
        create_app(project, data, port=8898), base_url="http://127.0.0.1:8898"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        state_before = client.get("/api/model-plan").json()
        path = "/api/core-models/main/interaction-grid"
        result = client.post(path, json={"a": "Age", "b": "VehicleAge"})
        assert result.status_code == 200, result.text
        assert result.json()["cells"] == 10_404
        assert not result.json()["allowed"]

        pair_design = {
            "defaults": asdict(project.design.defaults),
            "variables": {
                "Age": asdict(VariableDesign(kind="step", n_bins=5)),
                "VehicleAge": asdict(VariableDesign(kind="step", n_bins=5)),
            },
        }
        result = client.post(
            path,
            json={
                "a": "Age",
                "b": "VehicleAge",
                "design": pair_design,
            },
        )
        assert result.status_code == 200, result.text
        assert result.json()["allowed"]
        assert (
            client.get("/api/model-plan").json()["revision"] == state_before["revision"]
        )
        protected = client.post(path, json={"a": "Age", "b": "Claims"})
        assert protected.status_code == 422
    assert project.to_dict() == before
