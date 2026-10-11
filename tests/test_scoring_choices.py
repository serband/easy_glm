"""Pure frozen-fit route choice projection."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from easy_glm.desktop.scoring_choices import scoring_targets
from easy_glm.engine.models import FromToRow, ModelMetadata, VariableConfig
from easy_glm.engine.rate_model import RateModel
from easy_glm.workflow.model_plan import ModelPlan, PlannedModel
from easy_glm.workflow.project import Derived, Project
from easy_glm.workflow.scoring import _prepare
from easy_glm.workflow.stack_types import (
    ComponentArtifact,
    ComponentRecipe,
    ConstantScorer,
    LookupScorer,
    StackScorer,
)


def test_standalone_logit_is_one_immutable_probability_target() -> None:
    project = Project()
    rate_model = RateModel(
        0.2,
        {},
        metadata=ModelMetadata(link="logit", target="Claim", weight_col="Exposure"),
    )
    fitted = SimpleNamespace(
        name="Claim model", rate_model=rate_model, stack=None, created_at="now"
    )
    before = rate_model.to_dict()

    targets = scoring_targets(project, fitted, {"fit_id": "fit-1"})

    assert len(targets) == 1
    assert targets[0]["id"] == "fit-1/overall"
    assert targets[0]["purpose"] == "overall"
    assert targets[0]["output_kind"] == "probability"
    assert targets[0]["output_label"] == "Predicted probability"
    assert targets[0]["expected_label"] == "Expected probability"
    assert targets[0]["request"] == {
        "model_id": "fit-1",
        "scope": "full",
        "component_ids": [],
        "through": None,
    }
    assert targets[0]["exposure_column"] is None
    assert targets[0]["allowed_exposure_columns"] == []
    assert rate_model.to_dict() == before


def test_standalone_poisson_unit_response_is_frequency() -> None:
    metadata = ModelMetadata(link="log", divide_target_by_weight=True)
    metadata.family = "poisson"
    rate_model = RateModel(1.0, {}, metadata=metadata)

    target = scoring_targets(
        Project(),
        SimpleNamespace(name="Frequency", rate_model=rate_model, stack=None),
        {"fit_id": "fit"},
    )[0]

    assert target["output_kind"] == "frequency"


def test_renamed_and_derived_requirements_are_source_columns() -> None:
    project = Project()
    project.data.renames = {"RawAge": "Age"}
    project.data.derived = [Derived("AgeBand", "pl.col('Age') // 10")]
    rate_model = RateModel(
        1.0,
        {"AgeBand": VariableConfig("numeric", [FromToRow(0, 100, 1.0)])},
        metadata=ModelMetadata(link="log"),
    )
    fitted = SimpleNamespace(
        name="Saved", rate_model=rate_model, stack=None, created_at=""
    )
    target = scoring_targets(project, fitted, {"fit_id": "fit"})[0]
    source_book = pl.DataFrame({"RawAge": [24, 57]})
    prepared = _prepare(source_book, project, {"AgeBand"})
    assert target["request"] == {
        "model_id": "fit",
        "scope": "full",
        "component_ids": [],
        "through": None,
    }
    assert target["required_columns"] == ["RawAge"]
    assert prepared.columns == ["Age", "AgeBand"]
    assert prepared["AgeBand"].to_list() == [2, 5]


def test_flat_stack_uses_actual_components_and_real_prefixes() -> None:
    project = Project()
    stack = StackScorer("poisson", "log", "Claims", "Exposure", None, True, None)
    stack.components = [
        ComponentArtifact(
            ComponentRecipe("main", "Main", "glm", ("Age",)), "", ConstantScorer(0)
        ),
        ComponentArtifact(
            ComponentRecipe("vehicle", "Vehicle", "catboost", ("Vehicle",)),
            "",
            ConstantScorer(0),
        ),
    ]
    fitted = SimpleNamespace(
        name="Saved", rate_model=None, stack=stack, created_at="then"
    )

    targets = scoring_targets(project, fitted, {"fit_id": "frozen"})

    ratings = [item for item in targets if item["purpose"] == "rating"]
    assert [item["request"]["component_ids"] for item in ratings] == [
        ["main"],
        ["vehicle"],
    ]
    assert ratings[0]["output_kind"] == "frequency"
    checkpoints = [item for item in targets if item["purpose"] == "checkpoint"]
    assert checkpoints[0]["request"]["through"] == "main"
    assert checkpoints[0]["included_models"] == ["Main"]


def test_hidden_baseline_does_not_displace_root_catboost() -> None:
    project = Project()
    stack = StackScorer("gaussian", "log", "Cost", "Exposure", None, False, None)
    stack.components = [
        ComponentArtifact(
            ComponentRecipe("baseline", "Internal", "baseline", ()),
            "",
            ConstantScorer(0),
        ),
        ComponentArtifact(
            ComponentRecipe("root", "Root CatBoost", "catboost", ("Age",)),
            "",
            ConstantScorer(0),
        ),
        ComponentArtifact(
            ComponentRecipe("later", "Later", "catboost", ("Vehicle",)),
            "",
            ConstantScorer(0),
        ),
    ]
    fitted = SimpleNamespace(name="Saved", rate_model=None, stack=stack)

    ratings = [
        item
        for item in scoring_targets(project, fitted, {"fit_id": "fit"})
        if item["purpose"] == "rating"
    ]

    assert [item["output_kind"] for item in ratings] == ["rate", "multiplier"]


def test_logical_owned_group_uses_opaque_backend_id_and_unions_requirements() -> None:
    main = ComponentRecipe("main", "Main", "glm", ("Age",))
    pair = ComponentRecipe("pair", "Age × Region", "interaction", ("Age", "Region"))
    vehicle = ComponentRecipe("vehicle", "Vehicle", "catboost", ("Vehicle",))
    project = Project()
    project.model_plan = ModelPlan(
        format_version=2,
        models=[
            PlannedModel("main", "Main model", main, (pair,)),
            PlannedModel("vehicle", "Vehicle model", vehicle),
        ],
        fit_refs=[
            {
                "model_id": "main",
                "through_component_id": "pair",
                "model_name": "Saved",
                "fit_id": "frozen",
                "prefix_fingerprint": "x",
            },
            {
                "model_id": "vehicle",
                "through_component_id": "vehicle",
                "model_name": "Saved",
                "fit_id": "frozen",
                "prefix_fingerprint": "x",
            },
        ],
    )
    stack = StackScorer("poisson", "log", "Claims", "Exposure", None, True, None)
    stack.components = [
        ComponentArtifact(main, "", ConstantScorer(0)),
        ComponentArtifact(pair, "", ConstantScorer(0)),
        ComponentArtifact(vehicle, "", ConstantScorer(0)),
    ]
    fitted = SimpleNamespace(
        name="Saved", rate_model=None, stack=stack, created_at="then"
    )
    before_project = project.to_dict()
    before_stack = stack.to_dict()

    targets = scoring_targets(
        project, fitted, {"fit_id": "frozen", "id": "frozen:main"}
    )

    main_rating = next(item for item in targets if item["id"] == "frozen/main")
    vehicle_rating = next(item for item in targets if item["id"] == "frozen/vehicle")
    assert main_rating["request"]["model_id"] == "frozen:main"
    assert main_rating["request"]["component_ids"] == ["main", "pair"]
    assert {"Age", "Region"}.issubset(main_rating["required_columns"])
    assert vehicle_rating["request"]["model_id"] == "frozen:vehicle"
    assert vehicle_rating["request"]["component_ids"] == ["vehicle"]
    assert main_rating["exposure_column"] is None
    assert main_rating["allowed_exposure_columns"] == []
    assert project.to_dict() == before_project
    assert stack.to_dict() == before_stack


def test_overall_groups_all_frozen_components_even_when_only_last_model_is_selectable() -> (
    None
):
    main = ComponentRecipe("main", "Main", "glm", ("Age",))
    pair = ComponentRecipe("pair", "Age × Region", "interaction", ("Age", "Region"))
    vehicle = ComponentRecipe("vehicle", "Vehicle", "catboost", ("Vehicle",))
    project = Project()
    project.model_plan = ModelPlan(
        format_version=2,
        models=[
            PlannedModel("main", "Main model", main, (pair,)),
            PlannedModel("vehicle", "Vehicle model", vehicle),
        ],
        # A later frozen fit has only the Vehicle route reference.  It still
        # contains the earlier Main + pair artifacts in its accepted stack.
        fit_refs=[
            {
                "model_id": "vehicle",
                "through_component_id": "vehicle",
                "model_name": "Saved",
                "fit_id": "frozen",
                "prefix_fingerprint": "x",
            }
        ],
    )
    stack = StackScorer("poisson", "log", "Claims", "Exposure", None, True, None)
    stack.components = [
        ComponentArtifact(main, "", ConstantScorer(0)),
        ComponentArtifact(pair, "", ConstantScorer(0)),
        ComponentArtifact(vehicle, "", ConstantScorer(0)),
    ]

    targets = scoring_targets(
        project,
        SimpleNamespace(name="Saved", rate_model=None, stack=stack),
        {"fit_id": "frozen"},
    )

    assert targets[0]["included_models"] == [
        "Main model + 1 accepted pair table(s)",
        "Vehicle model",
    ]
    assert [item["id"] for item in targets if item["purpose"] == "rating"] == [
        "frozen/vehicle"
    ]


def test_catalogue_accepted_score_needs_key_but_native_candidate_needs_attributes() -> (
    None
):
    recipe = ComponentRecipe(
        "catalogue", "Vehicle catalogue", "catalogue", ("Vehicle", "Power")
    )
    stack = StackScorer("poisson", "log", "Claims", "Exposure", None, True, None)
    stack.components = [
        ComponentArtifact(
            recipe,
            "",
            LookupScorer(
                "Vehicle",
                pl.DataFrame({"Vehicle": ["A"], "Power": [90], "multiplier": [1.0]}),
                attributes=("Power",),
                native_scorer=object(),
            ),
        )
    ]

    targets = scoring_targets(
        Project(),
        SimpleNamespace(name="Saved", rate_model=None, stack=stack),
        {"fit_id": "fit"},
    )
    rating = next(item for item in targets if item["purpose"] == "rating")
    candidate = next(item for item in targets if item["purpose"] == "candidate")

    assert rating["required_columns"] == ["Vehicle"]
    assert rating["exposure_column"] is None
    assert candidate["required_columns"] == ["Power", "Vehicle"]
    assert candidate["output_kind"] == "candidate_multiplier"
    assert candidate["candidate_for"] == rating["id"]
    assert candidate["native_available"] is True
    assert candidate["exposure_column"] is None


def test_legacy_target_uses_friendly_saved_name_and_never_exposes_virtual_parts() -> (
    None
):
    legacy = ComponentRecipe(
        "legacy",
        "Accepted legacy scorer",
        "legacy",
        ("Age",),
        settings={"model_name": "Friendly saved model"},
    )
    virtual = ComponentRecipe("legacy_main", "Inspection main", "glm", ("Age",))
    later = ComponentRecipe("later", "Later correction", "catboost", ("Vehicle",))
    stack = StackScorer("poisson", "log", "Claims", "Exposure", None, True, None)
    stack.components = [
        ComponentArtifact(legacy, "", ConstantScorer(0)),
        ComponentArtifact(later, "", ConstantScorer(0)),
    ]
    stack.virtual_components = [ComponentArtifact(virtual, "", ConstantScorer(0))]
    before = stack.to_dict()

    targets = scoring_targets(
        Project(),
        SimpleNamespace(name="Saved", rate_model=None, stack=stack),
        {"fit_id": "fit"},
    )

    assert targets[0]["included_models"] == [
        "Friendly saved model",
        "Later correction",
    ]
    rating = next(item for item in targets if item["purpose"] == "rating")
    assert rating["label"] == "Friendly saved model"
    assert rating["included_models"] == ["Friendly saved model"]
    assert rating["request"]["component_ids"] == ["legacy"]
    checkpoint = next(item for item in targets if item["purpose"] == "checkpoint")
    assert checkpoint["label"] == "Through Friendly saved model"
    assert all("legacy_main" not in item["id"] for item in targets)
    assert stack.to_dict() == before
