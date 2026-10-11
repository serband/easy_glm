"""Fold-local selection of one component of an ordered pricing stack.

The folds evaluate a *fixed selected recipe*.  They are tuning evidence, not an
independent estimate of the performance of the whole modelling procedure.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import KFold

from .pair_stages import _pair_loss_sum
from .project import ModelConfig, Project
from .stack_types import ComponentRecipe, StackScorer


def _setting_int(
    settings: dict[str, Any], name: str, default: int, minimum: int
) -> int:
    value = settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _fold_loss(
    config: ModelConfig, rows: pl.DataFrame, prediction: np.ndarray
) -> tuple[float, float]:
    """Return original-response deviance/SSE and exposure or weight total."""
    from .stack import _response

    observed, weight = _response(config, rows)
    power = 1.0 if config.family == "poisson" else float(config.tweedie_power or 0)
    return (
        _pair_loss_sum(observed, prediction, weight, config.family, power),
        float(weight.sum()),
    )


def _fixed_candidate(recipe: ComponentRecipe, config: ModelConfig) -> ComponentRecipe:
    """Make the configured candidate replayable without nested glum CV."""
    settings = dict(recipe.settings)
    if recipe.kind == "glm":
        alpha = settings.get("alpha", config.penalty.alpha)
        # A correction GLM may not run glum CV with an in-sample fitted-prefix
        # offset.  Search fixed alphas in the outer fold loop instead.
        settings["alpha"] = 1e-4 if alpha is None else float(alpha)
        settings["cv"] = None
    elif recipe.kind == "interaction":
        settings.setdefault(
            "candidate",
            {
                "iterations": 120,
                "depth": 3,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,
            },
        )
    return replace(recipe, settings=settings)


def _sampled_candidate(recipe: ComponentRecipe, trial: Any) -> ComponentRecipe:
    settings = dict(recipe.settings)
    # ``no_change`` is an internal candidate marker, never a setting to inherit
    # into an actual Optuna candidate.
    settings.pop("no_change", None)
    if recipe.kind in ("catboost", "catalogue"):
        settings.update(
            iterations=trial.suggest_int("iterations", 40, 160, step=20),
            depth=trial.suggest_int("depth", 2, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.1, 20.0, log=True),
        )
    elif recipe.kind == "interaction":
        settings["candidate"] = {
            "iterations": trial.suggest_int("iterations", 40, 160, step=20),
            "depth": trial.suggest_int("depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 20.0, log=True),
        }
    elif recipe.kind == "glm":
        settings["alpha"] = trial.suggest_float("alpha", 1e-6, 1.0, log=True)
        settings["cv"] = None
    return replace(recipe, settings=settings)


def _fit_fold_candidate(
    project: Project,
    fold_train: pl.DataFrame,
    config: ModelConfig,
    recipe: ComponentRecipe,
    prefix: StackScorer,
) -> tuple[StackScorer, float]:
    """Fit one candidate against the fold prefix and append its accepted scorer."""
    from .stack import _append_component

    result = prefix.clone()
    artifact, _fit = _append_component(project, fold_train, config, recipe, result)
    return result, float(artifact.evidence.get("fit_seconds", 0.0))


def _tune_component_legacy(
    project: Project,
    train: pl.DataFrame,
    config: ModelConfig,
    recipe: ComponentRecipe,
    prefix_recipes: list[ComponentRecipe],
) -> tuple[ComponentRecipe, dict[str, Any]]:
    """Preserve the original combined trial-and-time tuning behaviour.

    ``prefix_recipes`` are the already selected recipes, never full-data fitted
    artifacts.  The same folds and prefix fits are reused for every candidate.
    """
    from .stack import TuningBudget, fit_prefix, use_tuning_budget

    if recipe.kind not in {"glm", "catboost", "catalogue", "interaction"}:
        raise ValueError(f"Unsupported tuned component {recipe.kind!r}")
    settings = recipe.settings
    count = _setting_int(settings, "cv", 3, 2)
    trials = _setting_int(settings, "trials", 8, 0)
    seed = _setting_int(settings, "seed", 42, 0)
    if train.height < count:
        raise ValueError(f"{count}-fold tuning needs at least {count} training rows")
    limit_minutes = settings.get("time_limit_minutes", 5.0)
    if limit_minutes is not None:
        if (
            isinstance(limit_minutes, bool)
            or not math.isfinite(float(limit_minutes))
            or float(limit_minutes) < 0
        ):
            raise ValueError("time_limit_minutes must be nonnegative and finite")
        limit_seconds = 60.0 * float(limit_minutes)
    else:
        limit_seconds = math.inf

    if trials == 0:
        return _fixed_candidate(recipe, config), {
            "kind": "tuning_skipped",
            "reason": "Zero search trials; fitted the configured candidate without CV selection",
            "catboost_tuning_seconds": 0.0,
        }

    if recipe.amendments and (recipe.kind != "glm" and limit_seconds == 0):
        # An identity component cannot carry the saved table edits.  With no
        # fitting allowance, retain the reviewed recipe and report no CV rather
        # than silently omit its amendments or invent validation evidence.
        return _fixed_candidate(recipe, config), {
            "kind": "tuning_unavailable",
            "reason": "Saved amendments require a fitted candidate; tuning allowance is zero",
            "catboost_tuning_seconds": 0.0,
        }

    splitter = KFold(n_splits=count, shuffle=True, random_state=seed)
    folds: list[tuple[pl.DataFrame, pl.DataFrame, StackScorer, float, float]] = []
    for fit_indices, validation_indices in splitter.split(np.arange(train.height)):
        fit_rows, validation_rows = train[fit_indices], train[validation_indices]
        prefix = fit_prefix(project, fit_rows, config, prefix_recipes, recipe)
        baseline = prefix.predict_unit(prefix.enrich(validation_rows))
        loss, weight = _fold_loss(config, validation_rows, baseline)
        folds.append((fit_rows, validation_rows, prefix, loss, weight))

    denominator = sum(item[4] for item in folds)
    prefix_loss = sum(item[3] for item in folds) / denominator
    candidates: list[tuple[float, ComponentRecipe, list[float]]] = []
    failures: list[str] = []
    budget = TuningBudget(limit_seconds)
    consumed = 0.0
    fixed = _fixed_candidate(recipe, config)
    # Identity is an explicit candidate for corrections, including a first
    # CatBoost/pair/catalogue whose conditional intercept was fitted per fold.
    has_identity = (
        bool(prefix_recipes) or recipe.kind != "glm"
    ) and not recipe.amendments
    if has_identity:
        identity = replace(fixed, settings={**fixed.settings, "no_change": True})
        candidates.append(
            (prefix_loss, identity, [item[3] / item[4] for item in folds])
        )

    def evaluate(candidate: ComponentRecipe) -> float:
        nonlocal consumed
        losses: list[float] = []
        weights: list[float] = []
        for fit_rows, validation_rows, prefix, _loss, _weight in folds:
            if is_catboost:
                budget.check()
            previous = budget.consumed
            if is_catboost:
                with use_tuning_budget(budget):
                    scorer, fit_seconds = _fit_fold_candidate(
                        project, fit_rows, config, candidate, prefix
                    )
                # A test or alternate fitter may report elapsed time without
                # using the context timer; real CatBoost fits debit in finally.
                if budget.consumed == previous:
                    budget.debit(fit_seconds)
                consumed = budget.consumed
            else:
                scorer, fit_seconds = _fit_fold_candidate(
                    project, fit_rows, config, candidate, prefix
                )
            prediction = scorer.predict_unit(scorer.enrich(validation_rows))
            loss, weight = _fold_loss(config, validation_rows, prediction)
            losses.append(loss)
            weights.append(weight)
        value = sum(losses) / sum(weights)
        candidates.append(
            (
                value,
                candidate,
                [loss / weight for loss, weight in zip(losses, weights, strict=True)],
            )
        )
        return value

    is_catboost = recipe.kind in {"catboost", "catalogue", "interaction"}
    if trials > 0 and (not is_catboost or limit_seconds > 0):
        try:
            evaluate(fixed)
        except (ValueError, RuntimeError, OverflowError, TimeoutError) as exc:
            failures.append(f"fixed: {exc}")
            consumed = budget.consumed
    if trials > 1 and (not is_catboost or consumed < limit_seconds):
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("Component tuning requires Optuna") from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )

        def objective(trial: Any) -> float:
            candidate = _sampled_candidate(fixed, trial)
            try:
                return evaluate(candidate)
            except (ValueError, RuntimeError, OverflowError, TimeoutError) as exc:
                failures.append(f"trial {trial.number}: {exc}")
                raise optuna.TrialPruned() from exc

        for _ in range(trials - 1):
            if is_catboost and consumed >= limit_seconds:
                break
            study.optimize(objective, n_trials=1, show_progress_bar=False)

    if not candidates:
        raise ValueError(
            "No valid component candidate remained after tuning: " + "; ".join(failures)
        )
    # Retain no-change on a practically tied CV result; a needless correction
    # adds implementation and modelling risk without demonstrated improvement.
    best_loss = min(item[0] for item in candidates)
    allowance = max(1e-8, abs(best_loss) * 1e-8)
    chosen = next(
        (
            item
            for item in candidates
            if item[1].settings.get("no_change") and item[0] <= best_loss + allowance
        ),
        min(candidates, key=lambda item: item[0]),
    )
    if is_catboost:
        consumed = budget.consumed
    return chosen[1], {
        "kind": "fixed_recipe_tuning",
        "folds": count,
        "seed": seed,
        "prefix_cv_loss": prefix_loss,
        "selected_cv_loss": chosen[0],
        "selected_fold_losses": chosen[2],
        "selected_no_change": bool(chosen[1].settings.get("no_change")),
        "valid_candidates": len(candidates),
        "failed_candidates": failures,
        "catboost_tuning_seconds": consumed,
        "time_limit_minutes": limit_minutes,
        "budget_exhausted": is_catboost and consumed >= limit_seconds,
        "evidence_label": "Tuning on selected recipes; not independent selection performance",
    }


def _explicit_settings(recipe: ComponentRecipe) -> tuple[str, int, int, float | None]:
    """Validate the new, deliberately unambiguous CatBoost tuning settings."""
    settings = recipe.settings
    tune = settings.get("tune")
    if not isinstance(tune, bool):
        raise ValueError("CatBoost tune must be true or false")
    mode = settings.get("tuning_budget")
    if mode not in ("trials", "time"):
        raise ValueError("CatBoost tuning_budget must be 'trials' or 'time'")
    count = _setting_int(settings, "cv", 3, 2)
    _setting_int(settings, "seed", 42, 0)
    if mode == "trials":
        return mode, _setting_int(settings, "trials", 8, 1), count, None
    minutes = settings.get("time_limit_minutes", 15.0)
    if (
        isinstance(minutes, bool)
        or not isinstance(minutes, int | float)
        or not math.isfinite(float(minutes))
        or float(minutes) <= 0
    ):
        raise ValueError("CatBoost time_limit_minutes must be positive and finite")
    return mode, 0, count, float(minutes)


def _tune_component_explicit(
    project: Project,
    train: pl.DataFrame,
    config: ModelConfig,
    recipe: ComponentRecipe,
    prefix_recipes: list[ComponentRecipe],
) -> tuple[ComponentRecipe, dict[str, Any]]:
    """Tune direct CatBoost/catalogue components with one explicit budget.

    The time budget contains only candidate CatBoost folds.  Prefix fitting,
    data preparation, scoring and the final full-training fit are intentionally
    outside it.  A candidate can overrun while its current CatBoost fit completes.
    """
    from .stack import TuningBudget, fit_prefix, use_tuning_budget

    if recipe.kind not in {"catboost", "catalogue"}:
        raise ValueError("Explicit CatBoost tuning supports CatBoost components only")
    mode, requested_trials, count, minutes = _explicit_settings(recipe)
    seed = _setting_int(recipe.settings, "seed", 42, 0)
    if train.height < count:
        raise ValueError(f"{count}-fold tuning needs at least {count} training rows")
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("CatBoost automatic tuning requires Optuna") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    splitter = KFold(n_splits=count, shuffle=True, random_state=seed)
    folds: list[tuple[pl.DataFrame, pl.DataFrame, StackScorer, float, float]] = []
    for fit_indices, validation_indices in splitter.split(np.arange(train.height)):
        fit_rows, validation_rows = train[fit_indices], train[validation_indices]
        # Prefix recipes are refitted on this fold's training rows once, then
        # reused by every candidate.  Their time is not this component's budget.
        prefix = fit_prefix(project, fit_rows, config, prefix_recipes, recipe)
        baseline = prefix.predict_unit(prefix.enrich(validation_rows))
        loss, weight = _fold_loss(config, validation_rows, baseline)
        folds.append((fit_rows, validation_rows, prefix, loss, weight))
    prefix_loss = sum(item[3] for item in folds) / sum(item[4] for item in folds)

    if mode == "trials":
        limit_seconds = math.inf
    else:
        if minutes is None:  # satisfies the static contract
            raise ValueError("CatBoost time_limit_minutes must be positive and finite")
        limit_seconds = 60.0 * float(minutes)
    budget = TuningBudget(limit_seconds, raise_on_exhaustion=False)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    candidates: list[tuple[float, ComponentRecipe, list[float]]] = []
    failures: list[str] = []
    attempted = completed = pruned = 0

    def objective(trial: Any) -> float:
        nonlocal attempted, completed, pruned
        attempted += 1
        candidate = _sampled_candidate(recipe, trial)
        losses: list[float] = []
        weights: list[float] = []
        try:
            for fit_rows, validation_rows, prefix, _loss, _weight in folds:
                budget.check()
                previous = budget.consumed
                with use_tuning_budget(budget):
                    scorer, fit_seconds = _fit_fold_candidate(
                        project, fit_rows, config, candidate, prefix
                    )
                if budget.consumed == previous:
                    budget.debit(fit_seconds)
                prediction = scorer.predict_unit(scorer.enrich(validation_rows))
                loss, weight = _fold_loss(config, validation_rows, prediction)
                losses.append(loss)
                weights.append(weight)
        except (ValueError, RuntimeError, OverflowError, TimeoutError) as exc:
            pruned += 1
            failures.append(f"trial {trial.number}: {exc}")
            raise optuna.TrialPruned() from exc
        value = sum(losses) / sum(weights)
        completed += 1
        candidates.append(
            (
                value,
                candidate,
                [loss / weight for loss, weight in zip(losses, weights, strict=True)],
            )
        )
        return value

    stop_reason = "requested trial count reached"
    while mode == "time" or attempted < requested_trials:
        if budget.consumed >= limit_seconds:
            stop_reason = "time limit reached after the current CatBoost fit"
            break
        before = budget.consumed
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        # A custom/non-CatBoost fitter that never reports time must not create an
        # endless time-based search.  This is an error, not a secret trial cap.
        if mode == "time" and budget.consumed <= before:
            raise RuntimeError(
                "CatBoost time tuning could not measure candidate fit time"
            )
    if not candidates:
        detail = "; ".join(failures) or "no completed trial"
        raise ValueError("No completed CatBoost tuning trial: " + detail)
    # A correction that does not beat its frozen prefix is not accepted merely
    # because it was the least poor sampled tree.  This neutral comparison is
    # deliberately outside Optuna's requested trial count.
    candidates_with_identity = list(candidates)
    if not recipe.amendments:
        fixed = _fixed_candidate(recipe, config)
        neutral = replace(fixed, settings={**fixed.settings, "no_change": True})
        candidates_with_identity.append(
            (
                prefix_loss,
                neutral,
                [item[3] / item[4] for item in folds],
            )
        )
    best_loss = min(item[0] for item in candidates_with_identity)
    allowance = max(1e-8, abs(best_loss) * 1e-8)
    chosen = next(
        (
            item
            for item in candidates_with_identity
            if item[1].settings.get("no_change") and item[0] <= best_loss + allowance
        ),
        min(candidates_with_identity, key=lambda item: item[0]),
    )
    selected_no_change = bool(chosen[1].settings.get("no_change"))
    evidence = {
        "kind": "optuna_catboost_tuning",
        "mode": mode,
        "budget": requested_trials if mode == "trials" else minutes,
        "budget_unit": "trials" if mode == "trials" else "minutes",
        "folds": count,
        "seed": seed,
        "prefix_cv_loss": prefix_loss,
        "selected_cv_loss": chosen[0],
        "selected_fold_losses": chosen[2],
        "selected_no_change": selected_no_change,
        "selected_params": (
            None
            if selected_no_change
            else {
                key: chosen[1].settings[key]
                for key in ("iterations", "depth", "learning_rate", "l2_leaf_reg")
            }
        ),
        "attempted_trials": attempted,
        "completed_trials": completed,
        "pruned_trials": pruned,
        "failed_trials": failures,
        "stop_reason": stop_reason,
        "catboost_tuning_seconds": budget.consumed,
        "evidence_label": "Training-fold tuning of this component; not independent performance",
    }
    if selected_no_change:
        evidence["reason"] = (
            "No sampled CatBoost setting improved the preceding model on training folds"
        )
    return chosen[1], evidence


def tune_component(
    project: Project,
    train: pl.DataFrame,
    config: ModelConfig,
    recipe: ComponentRecipe,
    prefix_recipes: list[ComponentRecipe],
) -> tuple[ComponentRecipe, dict[str, Any]]:
    """Tune a component, retaining legacy settings when no explicit mode exists."""
    if recipe.settings.get("tune") is False:
        return _fixed_candidate(recipe, config), {
            "kind": "manual_catboost_fit",
            "reason": "Automatic tuning is off; fitted the configured CatBoost settings once",
            "catboost_tuning_seconds": 0.0,
        }
    if "tuning_budget" in recipe.settings:
        return _tune_component_explicit(project, train, config, recipe, prefix_recipes)
    return _tune_component_legacy(project, train, config, recipe, prefix_recipes)
