"""Fold-safe tuning of mixed stack corrections."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from sklearn.model_selection import KFold

from easy_glm.workflow.project import ModelConfig, Project
from easy_glm.workflow.stack_tuning import _fold_loss, tune_component
from easy_glm.workflow.stack_types import (
    ComponentArtifact,
    ComponentRecipe,
    ConstantScorer,
    StackScorer,
    validate_catboost_tuning_settings,
)


def _config(recipe: ComponentRecipe) -> tuple[Project, ModelConfig]:
    project = Project()
    config = ModelConfig(family="poisson", target="y", weight="w", components=[recipe])
    project.models["Stack"] = config
    return project, config


def test_correction_first_no_change_uses_fold_training_baseline() -> None:
    recipe = ComponentRecipe(
        "pair",
        "Pair",
        "interaction",
        ("a", "b"),
        {"trials": 1, "cv": 3, "seed": 17, "time_limit_minutes": 0},
    )
    project, config = _config(recipe)
    frame = pl.DataFrame(
        {
            "a": ["A", "A", "B", "B", "A", "B", "A", "B", "A"],
            "b": ["X", "Y", "X", "Y", "X", "Y", "Y", "X", "X"],
            "y": [0.0, 1.0, 9.0, 2.0, 3.0, 7.0, 1.0, 8.0, 2.0],
            "w": [1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0],
        }
    )
    chosen, evidence = tune_component(project, frame, config, recipe, [])
    assert chosen.settings["no_change"] is True
    manual_loss = manual_weight = 0.0
    for fit_index, score_index in KFold(
        n_splits=3, shuffle=True, random_state=17
    ).split(np.arange(frame.height)):
        fold_train, fold_validation = frame[fit_index], frame[score_index]
        level = float(
            np.sum(fold_train["y"].to_numpy() * fold_train["w"].to_numpy())
            / fold_train["w"].sum()
        )
        loss, weight = _fold_loss(
            config, fold_validation, np.full(fold_validation.height, level)
        )
        manual_loss += loss
        manual_weight += weight
    assert evidence["prefix_cv_loss"] == pytest.approx(manual_loss / manual_weight)
    assert evidence["selected_cv_loss"] == pytest.approx(manual_loss / manual_weight)
    assert evidence["catboost_tuning_seconds"] == 0


def test_prefix_is_refitted_for_each_fold_and_reused_across_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easy_glm.workflow import stack, stack_tuning

    recipe = ComponentRecipe(
        "candidate",
        "Candidate",
        "catboost",
        ("x",),
        {"trials": 2, "cv": 3, "seed": 5, "time_limit_minutes": 1},
    )
    prefix_recipe = ComponentRecipe("prefix", "Prefix", "glm", ("x",))
    project, config = _config(recipe)
    frame = pl.DataFrame(
        {
            "x": list(range(12)),
            "y": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0],
            "w": [1.0] * 12,
        }
    )
    fitted_rows: list[tuple[int, ...]] = []
    candidate_prefix_levels: list[float] = []

    def fold_prefix(_project, rows, _config, recipes, _upcoming):
        assert recipes == [prefix_recipe]
        fitted_rows.append(tuple(rows["x"].to_list()))
        level = float(rows["y"].mean())
        result = stack._new_stack(config)
        result.components.append(
            ComponentArtifact(prefix_recipe, "", ConstantScorer(math.log(level)))
        )
        return result

    def fold_candidate(_project, rows, _config, candidate, prefix: StackScorer):
        candidate_prefix_levels.append(float(prefix.predict_unit(rows)[0]))
        result = prefix.clone()
        result.components.append(ComponentArtifact(candidate, "", ConstantScorer(0.0)))
        return result, 0.01

    monkeypatch.setattr(stack, "fit_prefix", fold_prefix)
    monkeypatch.setattr(stack_tuning, "_fit_fold_candidate", fold_candidate)
    _, evidence = tune_component(project, frame, config, recipe, [prefix_recipe])
    assert len(fitted_rows) == 3
    assert len(set(fitted_rows)) == 3
    assert len(candidate_prefix_levels) == 6
    assert candidate_prefix_levels[:3] == pytest.approx(candidate_prefix_levels[3:])
    assert evidence["catboost_tuning_seconds"] == pytest.approx(0.06)
    assert evidence["selected_no_change"] is True


def test_catboost_budget_zero_keeps_valid_no_change_candidate() -> None:
    recipe = ComponentRecipe(
        "cb",
        "CatBoost",
        "catboost",
        ("x",),
        {"trials": 4, "cv": 2, "seed": 11, "time_limit_minutes": 0},
    )
    project, config = _config(recipe)
    frame = pl.DataFrame(
        {"x": [0, 1, 0, 1, 0, 1], "y": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0], "w": [1.0] * 6}
    )
    chosen, evidence = tune_component(project, frame, config, recipe, [])
    assert chosen.settings["no_change"] is True
    assert evidence["valid_candidates"] == 1
    assert evidence["budget_exhausted"] is True


def test_catboost_budget_stops_before_next_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easy_glm.workflow import stack_tuning

    recipe = ComponentRecipe(
        "cb",
        "CatBoost",
        "catboost",
        ("x",),
        {"trials": 2, "cv": 3, "seed": 11, "time_limit_minutes": 1 / 60},
    )
    project, config = _config(recipe)
    frame = pl.DataFrame(
        {
            "x": list(range(9)),
            "y": [1.0, 2.0, 3.0] * 3,
            "w": [1.0] * 9,
        }
    )
    calls = 0

    def slow_fit(_project, rows, _config, candidate, prefix):
        nonlocal calls
        calls += 1
        return prefix.clone(), 2.0

    monkeypatch.setattr(stack_tuning, "_fit_fold_candidate", slow_fit)
    selected, evidence = tune_component(project, frame, config, recipe, [])
    assert calls == 1
    assert selected.settings["no_change"] is True
    assert evidence["budget_exhausted"] is True
    assert evidence["catboost_tuning_seconds"] == pytest.approx(2.0)


def test_real_catboost_candidate_is_evaluated_on_original_response() -> None:
    recipe = ComponentRecipe(
        "cb",
        "CatBoost",
        "catboost",
        ("x",),
        {
            "tune": True,
            "tuning_budget": "trials",
            "trials": 1,
            "cv": 2,
            "seed": 5,
            "time_limit_minutes": 1,
            "iterations": 15,
            "depth": 2,
            "learning_rate": 0.1,
        },
    )
    project, config = _config(recipe)
    config.weight = None
    frame = pl.DataFrame(
        {"x": np.tile([0, 1, 2, 3], 12), "y": np.tile([1.0, 1.2, 1.4, 1.6], 12)}
    )
    chosen, evidence = tune_component(project, frame, config, recipe, [])
    assert not chosen.settings.get("no_change", False)
    assert evidence["selected_no_change"] is False
    assert evidence["selected_params"] is not None
    assert evidence["selected_cv_loss"] < evidence["prefix_cv_loss"]
    assert evidence["catboost_tuning_seconds"] > 0


def test_zero_trials_preserves_reviewed_component_without_claiming_cv() -> None:
    recipe = ComponentRecipe(
        "vehicle",
        "Vehicle",
        "catalogue",
        ("power",),
        {"trials": 0, "cv": 2, "time_limit_minutes": 0},
        ({"operation": "cap", "lower": 0.8},),
    )
    project, config = _config(recipe)
    selected, evidence = tune_component(
        project, pl.DataFrame({"y": [1.0, 2.0]}), config, recipe, []
    )
    assert selected.amendments == recipe.amendments
    assert evidence["kind"] == "tuning_skipped"
    assert "selected_cv_loss" not in evidence


def test_explicit_trial_budget_uses_optuna_from_its_first_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easy_glm.workflow import stack_tuning

    recipe = ComponentRecipe(
        "cb",
        "CatBoost",
        "catboost",
        ("x",),
        {
            "tune": True,
            "tuning_budget": "trials",
            "trials": 1,
            "cv": 2,
            "seed": 7,
            "iterations": 12,
        },
    )
    project, config = _config(recipe)
    frame = pl.DataFrame({"x": [0, 1, 2, 3], "y": [1.0, 2.0, 3.0, 4.0], "w": [1.0] * 4})
    seen: list[ComponentRecipe] = []

    def candidate(_project, _rows, _config, candidate, prefix):
        seen.append(candidate)
        result = prefix.clone()
        result.components.append(ComponentArtifact(candidate, "", ConstantScorer(0.0)))
        return result, 0.01

    monkeypatch.setattr(stack_tuning, "_fit_fold_candidate", candidate)
    selected, evidence = tune_component(project, frame, config, recipe, [])
    assert evidence["kind"] == "optuna_catboost_tuning"
    assert evidence["attempted_trials"] == evidence["completed_trials"] == 1
    assert evidence["pruned_trials"] == 0
    assert len(seen) == 2  # two CV folds, one sampled candidate
    assert selected.settings["no_change"] is True
    assert evidence["selected_no_change"] is True
    assert evidence["selected_params"] is None
    assert "No sampled CatBoost setting improved" in evidence["reason"]


def test_explicit_time_budget_has_no_hidden_trial_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easy_glm.workflow import stack_tuning

    recipe = ComponentRecipe(
        "cb",
        "CatBoost",
        "catboost",
        ("x",),
        {
            "tune": True,
            "tuning_budget": "time",
            "time_limit_minutes": 10 / 60,
            "cv": 2,
            "seed": 7,
        },
    )
    project, config = _config(recipe)
    frame = pl.DataFrame({"x": [0, 1, 2, 3], "y": [1.0, 2.0, 3.0, 4.0], "w": [1.0] * 4})

    def candidate(_project, _rows, _config, candidate, prefix):
        result = prefix.clone()
        result.components.append(ComponentArtifact(candidate, "", ConstantScorer(0.0)))
        return result, 0.5

    monkeypatch.setattr(stack_tuning, "_fit_fold_candidate", candidate)
    _, evidence = tune_component(project, frame, config, recipe, [])
    # Two folds consume one second per completed trial. Ten seconds permits ten
    # completed trials, demonstrating that time mode does not retain an 8 cap.
    assert evidence["completed_trials"] == 10
    assert evidence["attempted_trials"] == 10
    assert (
        evidence["stop_reason"] == "time limit reached after the current CatBoost fit"
    )


def test_explicit_tuning_keeps_no_change_when_every_tree_is_worse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easy_glm.workflow import stack_tuning

    recipe = ComponentRecipe(
        "cb",
        "CatBoost",
        "catboost",
        ("x",),
        {"tune": True, "tuning_budget": "trials", "trials": 2, "cv": 2},
    )
    project, config = _config(recipe)
    frame = pl.DataFrame({"x": [0, 1, 2, 3], "y": [1.0] * 4, "w": [1.0] * 4})

    def worse(_project, _rows, _config, candidate, prefix):
        result = prefix.clone()
        result.components.append(
            ComponentArtifact(candidate, "", ConstantScorer(math.log(0.1)))
        )
        return result, 0.01

    monkeypatch.setattr(stack_tuning, "_fit_fold_candidate", worse)
    selected, evidence = tune_component(project, frame, config, recipe, [])
    assert selected.settings["no_change"] is True
    assert evidence["selected_no_change"] is True
    assert evidence["selected_cv_loss"] == pytest.approx(evidence["prefix_cv_loss"])
    assert "No sampled CatBoost setting improved" in evidence["reason"]


def test_manual_recipe_never_imports_or_runs_optuna(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easy_glm.workflow import stack_tuning

    recipe = ComponentRecipe(
        "cb", "CatBoost", "catboost", ("x",), {"tune": False, "iterations": 12}
    )
    project, config = _config(recipe)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Optuna must not run for a manual CatBoost recipe")

    monkeypatch.setattr(stack_tuning, "_tune_component_explicit", forbidden)
    selected, evidence = tune_component(
        project,
        pl.DataFrame({"x": [1, 2], "y": [1.0, 2.0], "w": [1.0, 1.0]}),
        config,
        recipe,
        [],
    )
    assert selected.settings["iterations"] == 12
    assert evidence["kind"] == "manual_catboost_fit"


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"tune": True, "tuning_budget": "trials", "trials": True}, "trials"),
        ({"tune": True, "tuning_budget": "trials", "trials": 1.5}, "trials"),
        ({"tune": True, "tuning_budget": "trials", "cv": True}, "cv"),
        ({"tune": False, "depth": True}, "depth"),
        ({"tune": False, "thread_count": 0}, "thread_count"),
        ({"tune": True, "tuning_budget": "trials", "thread_count": -1}, "thread_count"),
        (
            {"tune": True, "tuning_budget": "trials", "thread_count": True},
            "thread_count",
        ),
        (
            {
                "tune": True,
                "tuning_budget": "time",
                "time_limit_minutes": math.inf,
            },
            "time_limit_minutes",
        ),
        (
            {
                "tune": True,
                "tuning_budget": "time",
                "time_limit_minutes": math.nan,
            },
            "time_limit_minutes",
        ),
    ],
)
def test_explicit_catboost_settings_reject_invalid_values(
    settings: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_catboost_tuning_settings("catboost", settings)


def test_legacy_tune_flag_without_named_budget_is_preserved() -> None:
    # Older saved recipes may have the old opt-out flag but no named budget.
    # They must continue through the legacy combined trials/time path.
    validate_catboost_tuning_settings(
        "catboost", {"tune": True, "trials": 1, "time_limit_minutes": 1}
    )
    # A manual legacy recipe does not tune and can retain an ignored CV null.
    validate_catboost_tuning_settings(
        "catboost", {"tune": False, "cv": None, "depth": 3, "thread_count": 2}
    )
