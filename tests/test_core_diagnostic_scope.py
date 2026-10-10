"""Logical v2 model-plan checkpoints for stack diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from easy_glm.desktop.stack_views import (
    diagnostic_scopes,
    stack_diagnostics,
    stack_result,
)
from easy_glm.workflow.model_plan import PLAN_FORMAT_MODELS, ModelPlan, PlannedModel
from easy_glm.workflow.project import ModelConfig, Project
from easy_glm.workflow.stack_types import (
    ComponentArtifact,
    ComponentRecipe,
    ConstantScorer,
    StackScorer,
)


def _recipe(component_id: str, name: str, kind: str, *factors: str) -> ComponentRecipe:
    return ComponentRecipe(component_id, name, kind, factors)


def _artifact(recipe: ComponentRecipe, multiplier: float) -> ComponentArtifact:
    return ComponentArtifact(recipe, "saved", ConstantScorer(np.log(multiplier)))


def _saved_v2_run() -> tuple[Project, pl.DataFrame, SimpleNamespace]:
    plan = ModelPlan(
        format_version=PLAN_FORMAT_MODELS,
        mode="sequential",
        family="poisson",
        link="log",
        divide_target_by_weight=True,
        models=[
            PlannedModel(
                "main",
                "Main GLM",
                _recipe("main", "Main effects", "glm", "Age"),
                (
                    _recipe("pair-1", "Pair 1", "interaction", "Age", "Region"),
                    _recipe("pair-2", "Pair 2", "interaction", "Age", "Make"),
                ),
            ),
            PlannedModel(
                "vehicle",
                "Vehicle",
                _recipe("vehicle", "Vehicle", "catboost", "Make"),
            ),
        ],
    )
    project = Project(name="saved v2", model_plan=plan)
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "split": "split",
        "Age": "predictor",
        "Region": "predictor",
        "Make": "predictor",
    }
    project.data.split.mode = "column"
    project.data.split.column = "split"
    project.data.split.train_value = 1
    config = ModelConfig(
        family="poisson",
        link="log",
        target="Claims",
        weight="Exposure",
        divide_target_by_weight=True,
    )
    recipes = list(plan.physical_stages)
    stack = StackScorer(
        "poisson",
        "log",
        "Claims",
        "Exposure",
        None,
        True,
        None,
        [
            _artifact(recipes[0], 2.0),
            _artifact(recipes[1], 1.5),
            _artifact(recipes[2], 0.5),
            _artifact(recipes[3], 2.0),
        ],
    )
    frame = pl.DataFrame(
        {
            "Claims": [1, 2, 3, 4],
            "Exposure": [1.0, 1.0, 1.0, 1.0],
            "split": [1, 1, 1, 0],
            "Age": [25, 40, 60, 35],
            "Region": ["A", "A", "B", "B"],
            "Make": ["X", "Y", "X", "Y"],
        }
    )
    run = SimpleNamespace(
        stack=stack,
        config=config,
        name="Saved chain",
        created_at="2026-10-10T00:00:00Z",
        project_snapshot=project.to_dict(),
    )
    return project, frame, run


def test_v2_diagnostics_group_pairs_under_their_saved_glm_and_score_prefixes():
    project, frame, run = _saved_v2_run()

    # Prove scope labels come from the saved snapshot, not this later mutable plan.
    project.model_plan.models[0] = PlannedModel(
        "main",
        "Changed current plan",
        project.model_plan.models[0].core,
        project.model_plan.models[0].pair_corrections,
    )
    scopes = diagnostic_scopes(
        run.stack, ModelPlan.from_dict(run.project_snapshot["model_plan"])
    )
    assert [scope["id"] for scope in scopes] == ["main", "pair-1", "pair-2", ""]
    assert [scope["label"] for scope in scopes] == [
        "Main effects — Main GLM",
        "Main GLM + Pair 1",
        "After Main GLM (2 tables)",
        "After Vehicle",
    ]
    assert [scope["logical_model_name"] for scope in scopes] == [
        "Main GLM",
        "Main GLM",
        "Main GLM",
        "Vehicle",
    ]

    result = stack_result(project, frame, run)
    assert result["diagnostic_scopes"] == scopes
    first_pair = stack_diagnostics(
        project, frame, run, subset="train", through="pair-1"
    )
    assert first_pair["scope"]["label"] == "Main GLM + Pair 1"
    # Main effects are 2, then Pair 1 multiplies by 1.5: three train rows at 3.
    assert first_pair["metrics"]["expected"] == pytest.approx(9.0)
    main_with_two_pairs = stack_diagnostics(
        project, frame, run, subset="train", through="pair-2"
    )
    assert main_with_two_pairs["metrics"]["expected"] == pytest.approx(4.5)
    final = stack_diagnostics(project, frame, run, subset="train")
    assert final["scope"]["label"] == "After Vehicle"
    assert final["metrics"]["expected"] == pytest.approx(9.0)


def test_v2_scope_falls_back_to_saved_physical_labels_when_snapshot_does_not_match():
    _project, _frame, run = _saved_v2_run()
    snapshot = Project.from_dict(run.project_snapshot)
    snapshot.model_plan.models[0] = PlannedModel(
        "main",
        "Main GLM",
        _recipe("main", "Main effects", "glm", "Age"),
        (_recipe("different", "Different pair", "interaction", "Age", "Region"),),
    )
    # The run's physical Pair 1 remains the only reliable source of its label.
    scopes = diagnostic_scopes(run.stack, snapshot.model_plan)
    assert [scope["id"] for scope in scopes] == ["main", "pair-1", "pair-2", ""]
    assert scopes[1]["label"] == "After Pair 1"
    assert "logical_model_name" not in scopes[1]
