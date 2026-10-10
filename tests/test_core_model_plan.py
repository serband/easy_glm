"""Logical model-plan schema: named cores own their interaction tables."""

from __future__ import annotations

import pytest

from easy_glm.workflow.model_plan import (
    PLAN_FORMAT_MODELS,
    AdoptedRoot,
    ModelPlan,
    PlannedModel,
    structural_problems,
)
from easy_glm.workflow.project import Project
from easy_glm.workflow.stack_types import ComponentRecipe


def recipe(component_id: str, name: str, kind: str, *factors: str) -> ComponentRecipe:
    return ComponentRecipe(component_id, name, kind, factors, {"alpha": 0.01})


def v2_plan() -> ModelPlan:
    main = PlannedModel(
        "main",
        "Main GLM",
        recipe("main", "Main effects", "glm", "DriverAge", "Region"),
        (
            recipe(
                "main-pair-1",
                "DriverAge × Region",
                "interaction",
                "DriverAge",
                "Region",
            ),
            recipe(
                "main-pair-2",
                "DriverAge × VehicleAge",
                "interaction",
                "DriverAge",
                "VehicleAge",
            ),
        ),
    )
    vehicle = PlannedModel(
        "vehicle",
        "Vehicle",
        recipe("vehicle", "Vehicle CatBoost", "catboost", "Make", "VehicleAge"),
    )
    return ModelPlan(
        format_version=PLAN_FORMAT_MODELS,
        mode="sequential",
        family="poisson",
        link="log",
        models=[main, vehicle],
    )


def test_v2_serialises_named_models_and_rebuilds_physical_projection():
    plan = v2_plan()
    encoded = plan.to_dict()
    assert encoded["format_version"] == 2
    assert "stages" not in encoded
    assert [model["name"] for model in encoded["models"]] == ["Main GLM", "Vehicle"]
    assert [
        pair["component_id"] for pair in encoded["models"][0]["pair_corrections"]
    ] == [
        "main-pair-1",
        "main-pair-2",
    ]

    restored = ModelPlan.from_dict(encoded)
    assert restored.to_dict() == encoded
    assert [stage.component_id for stage in restored.stages] == [
        "main",
        "main-pair-1",
        "main-pair-2",
        "vehicle",
    ]
    assert restored.logical_model_for_component("main-pair-2").name == "Main GLM"
    assert [part.name for part in restored.components_through("main-pair-1")] == [
        "Main effects",
        "DriverAge × Region",
    ]


def test_v2_ref_binds_logical_owner_and_exact_physical_checkpoint():
    plan = v2_plan()
    plan.fit_refs = [
        {
            "model_id": "main",
            "through_component_id": "main-pair-2",
            "model_name": "French motor — Main GLM",
            "fit_id": "fit-123",
            "prefix_fingerprint": "a" * 64,
        }
    ]
    assert ModelPlan.from_dict(plan.to_dict()).fit_refs == plan.fit_refs

    raw = plan.to_dict()
    raw["fit_refs"][0]["model_id"] = "vehicle"
    with pytest.raises(ValueError, match="belongs to another model"):
        ModelPlan.from_dict(raw)


def test_v2_has_one_active_reference_per_model_and_preserves_adopted_snapshot():
    plan = v2_plan()
    plan.adopted_root = AdoptedRoot(
        "Accepted base",
        "fit-base",
        "b" * 64,
        (recipe("accepted", "Accepted score", "glm", "LegacyFactor"),),
    )
    plan.fit_refs = [
        {
            "model_id": "main",
            "through_component_id": "main",
            "model_name": "Main GLM fit",
            "fit_id": "fit-main",
            "prefix_fingerprint": "c" * 64,
        }
    ]
    encoded = plan.to_dict()
    assert ModelPlan.from_dict(encoded).adopted_root == plan.adopted_root
    encoded["fit_refs"].append({**encoded["fit_refs"][0], "fit_id": "duplicate"})
    with pytest.raises(ValueError, match="one active reference"):
        ModelPlan.from_dict(encoded)

    collision = plan.to_dict()
    collision["fit_refs"] = []
    collision["adopted_root"]["physical_prefix"][0]["component_id"] = "main"
    with pytest.raises(ValueError, match="overlap"):
        ModelPlan.from_dict(collision)


def test_v2_rejects_interaction_as_a_peer_model_or_under_catboost():
    pair = recipe("pair", "A × B", "interaction", "A", "B")
    with pytest.raises(ValueError, match="core must be GLM"):
        PlannedModel("pair", "Interaction", pair)
    with pytest.raises(ValueError, match="Only a GLM"):
        PlannedModel(
            "tree",
            "Tree",
            recipe("tree", "Tree", "catboost", "A"),
            (pair,),
        )
    with pytest.raises(ValueError, match="must equal"):
        PlannedModel("logical", "GLM", recipe("physical", "GLM", "glm", "A"))
    with pytest.raises(ValueError, match="must not contain duplicates"):
        PlannedModel(
            "main",
            "Main",
            recipe("main", "Main", "glm", "A"),
            (recipe("pair", "A × A", "interaction", "A", "A"),),
        )
    with pytest.raises(ValueError, match="same interaction twice"):
        PlannedModel(
            "main",
            "Main",
            recipe("main", "Main", "glm", "A"),
            (
                recipe("pair-1", "A × B", "interaction", "A", "B"),
                recipe("pair-2", "B × A", "interaction", "B", "A"),
            ),
        )


def test_legacy_v1_round_trip_is_unchanged_and_still_allows_peer_interactions():
    raw = {
        "plan_id": "legacy",
        "mode": "sequential",
        "family": "poisson",
        "link": "log",
        "tweedie_power": 1.5,
        "divide_target_by_weight": True,
        "stages": [
            recipe("main", "Main", "glm", "A").to_dict(),
            recipe("pair", "A × B", "interaction", "A", "B").to_dict(),
        ],
        "adopted_model_name": None,
        "fit_refs": [],
    }
    restored = ModelPlan.from_dict(raw)
    assert restored.format_version == 1
    assert restored.to_dict() == raw
    assert not structural_problems(restored, {"poisson"})


def test_project_version_only_changes_for_the_v2_plan_format():
    legacy = Project(model_plan=ModelPlan(stages=[recipe("main", "Main", "glm", "A")]))
    assert legacy.to_dict()["version"] == 6

    logical = Project(model_plan=v2_plan())
    encoded = logical.to_dict()
    assert encoded["version"] == 7
    assert (
        Project.from_dict(encoded).model_plan.to_dict() == logical.model_plan.to_dict()
    )
