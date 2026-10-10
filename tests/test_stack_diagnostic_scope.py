"""Diagnostic scope contracts for saved multiplicative model stacks."""

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
from easy_glm.workflow.project import ModelConfig, Project
from easy_glm.workflow.stack_types import (
    ComponentArtifact,
    ComponentRecipe,
    ConstantScorer,
    LookupScorer,
    StackScorer,
)


def _artifact(
    component_id: str, name: str, kind: str, scorer: object, factors: tuple[str, ...]
) -> ComponentArtifact:
    return ComponentArtifact(
        ComponentRecipe(component_id, name, kind, factors), "saved", scorer
    )


def _saved_stack() -> tuple[Project, pl.DataFrame, SimpleNamespace]:
    project = Project(name="Scoped diagnostics")
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "traintest": "split",
        "PolicyID": "id",
        "DriverAge": "predictor",
        "Unmodelled": "unassigned",
    }
    project.data.split.mode = "column"
    project.data.split.column = "traintest"
    project.data.split.train_value = 1
    project.data.split.holdout_value = 0
    config = ModelConfig(
        family="poisson",
        link="log",
        target="Claims",
        weight="Exposure",
        offset="external_offset",
        divide_target_by_weight=True,
    )
    catalogue = pl.DataFrame(
        {
            "PolicyID": [101, 102, 103, 104],
            "Make": ["A", "B", "A", "C"],
            "multiplier": [1.5, 2.0, 1.25, 0.75],
        }
    )
    baseline = _artifact(
        "main", "Main effects", "glm", ConstantScorer(np.log(1.25)), ("DriverAge",)
    )
    vehicle = _artifact(
        "vehicle",
        "Vehicle file",
        "catalogue",
        LookupScorer("PolicyID", catalogue, ("Make",)),
        ("Make",),
    )
    occupation = _artifact(
        "occupation",
        "Occupation correction",
        "glm",
        ConstantScorer(np.log(0.8)),
        ("Unmodelled",),
    )
    stack = StackScorer(
        "poisson",
        "log",
        "Claims",
        "Exposure",
        "external_offset",
        True,
        None,
        [baseline, vehicle, occupation],
    )
    frame = pl.DataFrame(
        {
            "Claims": [2, 1, 4, 1],
            "Exposure": [1.0, 2.0, 1.0, 1.0],
            "traintest": [1, 1, 1, 0],
            "PolicyID": [101, 102, 103, 104],
            "DriverAge": [22, 38, 51, 67],
            "Unmodelled": ["x", "y", "x", "z"],
            "external_offset": np.log([2.0, 2.0, 2.0, 2.0]),
        }
    )
    run = SimpleNamespace(
        stack=stack,
        config=config,
        name="Accepted motor stack",
        created_at="2026-10-10T00:00:00Z",
    )
    return project, frame, run


def test_scopes_are_physical_cumulative_prefixes_with_one_final_entry() -> None:
    project, frame, run = _saved_stack()
    scopes = diagnostic_scopes(run.stack)
    assert [scope["id"] for scope in scopes] == ["main", "vehicle", ""]
    assert [scope["included_models"] for scope in scopes] == [
        ["Main effects"],
        ["Main effects", "Vehicle file"],
        ["Main effects", "Vehicle file", "Occupation correction"],
    ]
    assert scopes[-1]["label"] == "Final fit — after Occupation correction"

    # The external offset stays in every prefix.  The later occupation
    # correction only appears in the accepted final score.
    train = frame.filter(pl.col("traintest") == 1)
    main = run.stack.prefix_through("main").predict_unit(train)
    vehicle = run.stack.prefix_through("vehicle").predict_unit(train)
    final = run.stack.prefix_through().predict_unit(train)
    np.testing.assert_allclose(main, np.repeat(2.5, train.height))
    np.testing.assert_allclose(vehicle, np.array([3.75, 5.0, 3.125]))
    np.testing.assert_allclose(final, vehicle * 0.8)

    result = stack_result(project, frame, run)
    assert result["diagnostic_scopes"] == scopes
    assert {"DriverAge", "Unmodelled", "Make"} <= set(result["review_variables"])
    assert "PolicyID" not in result["review_variables"]


def test_diagnostics_use_the_saved_scope_not_later_catalogue_or_component() -> None:
    project, frame, run = _saved_stack()
    initial = stack_diagnostics(project, frame, run, subset="train", through="main")
    after_vehicle = stack_diagnostics(
        project, frame, run, subset="train", through="vehicle", variable="Make"
    )
    final = stack_diagnostics(project, frame, run, subset="train")
    final_by_id = stack_diagnostics(
        project, frame, run, subset="train", through="occupation"
    )

    assert initial["saved_model"] == "Accepted motor stack"
    assert initial["scope"]["id"] == "main"
    assert "Make" not in initial["review_variables"]
    assert after_vehicle["scope"]["id"] == "vehicle"
    assert after_vehicle["ae"]
    assert final["scope"]["id"] == ""
    assert final_by_id["scope"] == final["scope"]
    assert final_by_id["metrics"] == final["metrics"]
    # The catalogue changes rows differently, so a saved final scorer must not
    # be a scalar re-labelling of the first-stage output.
    assert final["metrics"]["expected"] != pytest.approx(initial["metrics"]["expected"])

    with pytest.raises(ValueError, match="unavailable"):
        stack_diagnostics(project, frame, run, subset="train", through="missing")
    with pytest.raises(ValueError, match="unavailable for A/E"):
        stack_diagnostics(
            project, frame, run, subset="train", through="main", variable="Make"
        )


def test_internal_and_virtual_components_are_never_diagnostic_scopes() -> None:
    project, frame, run = _saved_stack()
    baseline = _artifact(
        "__baseline__", "Starting level", "baseline", ConstantScorer(0.0), ()
    )
    virtual = _artifact(
        "legacy_main", "Legacy main effects", "glm", ConstantScorer(0.0), ()
    )
    run.stack.components.insert(0, baseline)
    run.stack.virtual_components.append(virtual)
    scopes = diagnostic_scopes(run.stack)
    assert [scope["id"] for scope in scopes] == ["main", "vehicle", ""]
    with pytest.raises(ValueError, match="unavailable"):
        stack_diagnostics(project, frame, run, subset="holdout", through="legacy_main")


def test_legacy_is_a_real_boundary_and_correction_baseline_is_retained() -> None:
    _project, frame, run = _saved_stack()
    legacy = _artifact(
        "legacy", "Accepted legacy scorer", "legacy", ConstantScorer(np.log(1.1)), ()
    )
    run.stack.components = [legacy, *run.stack.components[1:]]
    assert [scope["id"] for scope in diagnostic_scopes(run.stack)] == [
        "legacy",
        "vehicle",
        "",
    ]

    baseline = _artifact(
        "__baseline__", "Starting level", "baseline", ConstantScorer(np.log(1.7)), ()
    )
    run.stack.components.insert(0, baseline)
    prefix = run.stack.prefix_through("legacy")
    train = frame.filter(pl.col("traintest") == 1)
    np.testing.assert_allclose(
        prefix.predict_unit(train), np.repeat(2.0 * 1.7 * 1.1, train.height)
    )
