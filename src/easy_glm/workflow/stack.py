"""Fit ordered log-link corrections against the accepted preceding scorer."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, cast

import numpy as np
import polars as pl

from easy_glm.core.design import DesignSpec
from easy_glm.core.fit import GLMFit, fit_glm
from easy_glm.core.tables import rate_tables, to_rate_model
from easy_glm.engine.rate_model import RateModel

from .catalogue import (
    align_catalogue_key,
    join_catalogue,
    load_catalogue,
    validate_catalogue,
)
from .pair_distillation import fit_catboost_pair_raw, fit_catboost_raw
from .project import ModelConfig, Project
from .stack_types import (
    CatBoostScorer,
    ComponentArtifact,
    ComponentRecipe,
    ConstantScorer,
    LookupScorer,
    RateModelScorer,
    StackScorer,
    validate_catboost_tuning_settings,
)


class TuningBudget:
    """CatBoost fit time only, shared across candidates and validation folds."""

    def __init__(self, seconds: float, *, raise_on_exhaustion: bool = True) -> None:
        self.limit_seconds = seconds
        self.consumed = 0.0
        self.raise_on_exhaustion = raise_on_exhaustion

    def check(self) -> None:
        if self.consumed >= self.limit_seconds:
            raise TimeoutError("CatBoost component tuning time limit reached")

    def debit(self, seconds: float) -> None:
        self.consumed += max(0.0, seconds)
        if self.raise_on_exhaustion:
            self.check()


_active_tuning_budget: ContextVar[TuningBudget | None] = ContextVar(
    "easy_glm_stack_tuning_budget", default=None
)


@contextmanager
def use_tuning_budget(budget: TuningBudget):
    token = _active_tuning_budget.set(budget)
    try:
        yield
    finally:
        _active_tuning_budget.reset(token)


def raw_features(
    frame: pl.DataFrame,
    factors: tuple[str, ...],
    categorical: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Stable CatBoost array with explicit missing categorical tokens."""
    if not factors:
        raise ValueError("A CatBoost component needs at least one factor")
    missing = [name for name in factors if name not in frame.columns]
    if missing:
        raise ValueError(f"CatBoost factors absent from data: {missing}")
    categorical = (
        tuple(
            i
            for i, name in enumerate(factors)
            if frame[name].dtype in (pl.String, pl.Categorical, pl.Enum)
        )
        if categorical is None
        else tuple(categorical)
    )
    columns: list[np.ndarray] = []
    for index, name in enumerate(factors):
        if index in categorical:
            columns.append(
                frame[name].cast(pl.String).fill_null("__MISSING__").to_numpy()
            )
        else:
            columns.append(
                frame[name].cast(pl.Float64).fill_null(float("nan")).to_numpy()
            )
    return np.column_stack(columns).astype(object), categorical


def _project_for_recipe(project: Project, recipe: ComponentRecipe) -> Project:
    """Return a private project view for one component's saved definitions."""
    if recipe.design is None:
        return project
    raw = project.to_dict()
    raw["design"] = copy.deepcopy(recipe.design)
    return Project.from_dict(raw)


def _recipe_categorical(
    recipe: ComponentRecipe, frame: pl.DataFrame
) -> tuple[int, ...] | None:
    """CatBoost uses local type overrides without applying GLM binning."""
    if recipe.design is None:
        return None
    return tuple(
        index
        for index, factor in enumerate(recipe.factors)
        if (
            recipe.design["variables"].get(factor, {}).get("kind") == "categorical"
            or (
                recipe.design["variables"].get(factor, {}).get("kind") is None
                and frame[factor].dtype in (pl.String, pl.Categorical, pl.Enum)
            )
        )
    )


def _cat_features_for_recipe(
    recipe: ComponentRecipe, frame: pl.DataFrame
) -> tuple[int, ...] | None:
    """Choose one authority for CatBoost's raw categorical columns.

    A component-local definition is an explicit later user decision, so it
    replaces the legacy ``settings.cat_features`` indexes.  Old recipes without
    a definition retain those indexes exactly.
    """
    if recipe.design is not None:
        return _recipe_categorical(recipe, frame)
    if "cat_features" in recipe.settings:
        return tuple(recipe.settings["cat_features"])
    return None


def _response(
    config: ModelConfig, frame: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    if config.target is None or config.target not in frame:
        raise ValueError("Stack needs a target column")
    y = frame[config.target].cast(pl.Float64).to_numpy()
    weight = (
        frame[config.weight].cast(pl.Float64).to_numpy()
        if config.weight
        else np.ones(frame.height)
    )
    if not np.all(np.isfinite(weight) & (weight > 0)):
        raise ValueError("Fitting weights must be positive and finite")
    if config.divide_target_by_weight:
        y = y / weight
    if not np.all(np.isfinite(y)) or (config.family != "gaussian" and np.any(y < 0)):
        raise ValueError("Response is invalid for the stack family")
    return y, weight


def _conditional_level(config: ModelConfig, frame: pl.DataFrame) -> float:
    y, weight = _response(config, frame)
    offset = (
        frame[config.offset].cast(pl.Float64).to_numpy()
        if config.offset
        else np.zeros(frame.height)
    )
    power = (
        0.0
        if config.family == "gaussian"
        else 1.0 if config.family == "poisson" else float(config.tweedie_power)
    )
    with np.errstate(over="ignore", under="ignore"):
        numerator = np.sum(weight * y * np.exp((1 - power) * offset))
        denominator = np.sum(weight * np.exp((2 - power) * offset))
    level = numerator / denominator
    if not math.isfinite(level) or level <= 0:
        raise ValueError(
            "Correction-only first component has no positive, finite conditional intercept baseline"
        )
    return math.log(level)


def _new_stack(config: ModelConfig) -> StackScorer:
    resolved_link = config.link or (
        "identity" if config.family == "gaussian" else "log"
    )
    if (
        config.family not in ("poisson", "tweedie", "gaussian")
        or resolved_link != "log"
    ):
        raise ValueError(
            "A mixed stack supports Poisson, Tweedie or Gaussian with a log link"
        )
    if config.family == "tweedie" and not 1 < config.tweedie_power < 2:
        raise ValueError("Tweedie power must be strictly between 1 and 2")
    return StackScorer(
        config.family,
        "log",
        config.target or "",
        config.weight,
        config.offset,
        config.divide_target_by_weight,
        config.tweedie_power if config.family == "tweedie" else None,
    )


def _check_prefix(config: ModelConfig, prefix: StackScorer) -> None:
    resolved_link = config.link or (
        "identity" if config.family == "gaussian" else "log"
    )
    if (
        prefix.family,
        prefix.link,
        prefix.target,
        prefix.weight,
        prefix.offset_col,
        prefix.divide_target_by_weight,
        prefix.tweedie_power,
    ) != (
        config.family,
        resolved_link,
        config.target,
        config.weight,
        config.offset,
        config.divide_target_by_weight,
        config.tweedie_power if config.family == "tweedie" else None,
    ):
        raise ValueError(
            "Component changes family, target, weight or external offset units"
        )


def _fit_glm(
    project: Project,
    train: pl.DataFrame,
    config: ModelConfig,
    recipe: ComponentRecipe,
    prefix: StackScorer,
    progress: Callable[[str], None] | None,
) -> tuple[RateModelScorer, dict[str, Any], Any]:
    from .run import build_design, monotone_for

    dropped: list[str] = []
    factors = list(recipe.factors)
    component_project = _project_for_recipe(project, recipe)
    spec = build_design(
        component_project, train, factors, weight_col=config.weight, dropped=dropped
    )
    settings = recipe.settings
    alpha = settings.get("alpha", config.penalty.alpha)
    if alpha is not None and (not math.isfinite(float(alpha)) or float(alpha) < 0):
        raise ValueError("GLM alpha must be nonnegative and finite")
    cv = settings.get("cv", config.penalty.cv) if alpha is None else None
    offset = prefix.linear_predictor(train)
    fit = fit_glm(
        train,
        spec,
        config.target or "",
        family=config.family,
        tweedie_power=prefix.tweedie_power,
        weight_col=config.weight,
        divide_target_by_weight=config.divide_target_by_weight,
        offset=offset,
        alpha=alpha,
        cv=cv,
        cv_seed=component_project.data.split.seed,
        n_alphas=settings.get("n_alphas", config.penalty.n_alphas),
        l1_ratio=settings.get("l1_ratio", config.penalty.l1_ratio),
        min_alpha_ratio=settings.get("min_alpha_ratio", config.penalty.min_alpha_ratio),
        monotone=monotone_for(component_project, replace(config, predictors=factors)),
        progress=progress,
        link="log",
    )
    rate_model = to_rate_model(
        fit,
        base=settings.get("base", config.base),
        exposure_col=None,
        train_test_col=None,
        model_type=config.family,
    )
    return (
        RateModelScorer(rate_model),
        {"alpha": fit.alpha, "dropped_predictors": dropped},
        fit,
    )


def _fit_catboost(
    train: pl.DataFrame,
    config: ModelConfig,
    recipe: ComponentRecipe,
    prefix: StackScorer,
) -> tuple[CatBoostScorer, dict[str, Any]]:
    settings = recipe.settings
    raw, categorical = raw_features(
        train,
        recipe.factors,
        _cat_features_for_recipe(recipe, train),
    )
    y, weight = _response(config, train)
    baseline = prefix.predict_unit(train)
    positive_target = y[y > 0]
    if config.family == "gaussian":
        from .pair_stages import _gaussian_training_scale

    target_scale = (
        _gaussian_training_scale(y, baseline)
        if config.family == "gaussian"
        else (
            max(1.0, float(np.max(positive_target)) / 1000.0)
            if config.family == "tweedie" and positive_target.size
            else 1.0
        )
    )
    started = time.monotonic()
    budget = _active_tuning_budget.get()
    if budget is not None:
        budget.check()
    try:
        teacher = fit_catboost_raw(
            raw,
            y,
            baseline,
            sample_weight=weight,
            family=config.family,
            tweedie_power=prefix.tweedie_power,
            target_scale=target_scale,
            iterations=int(settings.get("iterations", 120)),
            depth=int(settings.get("depth", 3)),
            learning_rate=float(settings.get("learning_rate", 0.05)),
            l2_leaf_reg=float(settings.get("l2_leaf_reg", 3.0)),
            thread_count=int(settings.get("thread_count", 2)),
            seed=int(settings.get("seed", 42)),
            cat_features=categorical,
        )
    finally:
        if budget is not None:
            budget.debit(time.monotonic() - started)
    elapsed = time.monotonic() - started
    scorer = CatBoostScorer(recipe.factors, teacher.model, categorical)
    scorer.linear_predictor(train)
    return scorer, {"target_scale": target_scale, "fit_seconds": elapsed}


def _fit_catalogue(
    train: pl.DataFrame,
    config: ModelConfig,
    recipe: ComponentRecipe,
    prefix: StackScorer,
) -> tuple[LookupScorer, dict[str, Any]]:
    settings = recipe.settings
    if settings.get("method", "catboost") != "catboost":
        raise ValueError("First-version catalogue fitting supports method='catboost'")
    key = str(settings["key"])
    master = load_catalogue(settings)
    master = align_catalogue_key(train, master, key)
    validate_catalogue(master, key, recipe.factors)
    enriched = join_catalogue(train, master, key, recipe.factors)
    scorer, evidence = _fit_catboost(enriched, config, recipe, prefix)
    multipliers = np.exp(scorer.linear_predictor(master))
    if not np.all(np.isfinite(multipliers) & (multipliers > 0)):
        raise ValueError("Catalogue scorer produced invalid multipliers")
    table = master.with_columns(pl.Series("multiplier", multipliers))
    support = train.group_by(key).len().rename({"len": "train_rows"})
    table = table.join(support, on=key, how="left").with_columns(
        pl.col("train_rows").fill_null(0)
    )
    # Keep the native teacher separately from the deployed lookup.  Amendments
    # only ever touch the latter; scoring a new complete master needs the
    # original raw-feature CatBoost model and must not reverse-engineer it from
    # amended keys.
    lookup = LookupScorer(
        key,
        table,
        recipe.factors,
        settings.get("fallback"),
        native_scorer=scorer,
    )
    evidence.update(
        {
            "catalogue_rows": table.height,
            "unseen_keys": int((table["train_rows"] == 0).sum()),
        }
    )
    unseen_levels: dict[str, dict[str, Any]] = {}
    for factor in recipe.factors:
        if master[factor].dtype not in (pl.String, pl.Categorical, pl.Enum):
            continue
        fitted = set(enriched[factor].cast(pl.String).drop_nulls().unique().to_list())
        unseen = [
            value
            for value in master[factor].cast(pl.String).drop_nulls().unique().to_list()
            if value not in fitted
        ]
        unseen_levels[factor] = {"count": len(unseen), "examples": unseen[:3]}
    evidence["unseen_attribute_levels"] = unseen_levels
    return lookup, evidence


def _fit_component(
    project: Project,
    train: pl.DataFrame,
    config: ModelConfig,
    recipe: ComponentRecipe,
    prefix: StackScorer,
    progress: Callable[[str], None] | None,
) -> tuple[Any, dict[str, Any], Any | None]:
    if recipe.settings.get("no_change"):
        if recipe.kind == "catalogue":
            key = str(recipe.settings["key"])
            master = load_catalogue(recipe.settings)
            master = align_catalogue_key(train, master, key)
            validate_catalogue(master, key, recipe.factors)
            join_catalogue(train, master, key, recipe.factors)
            table = master.with_columns(pl.lit(1.0).alias("multiplier"))
            support = train.group_by(key).len().rename({"len": "train_rows"})
            table = table.join(support, on=key, how="left").with_columns(
                pl.col("train_rows").fill_null(0)
            )
            return (
                LookupScorer(
                    key, table, recipe.factors, recipe.settings.get("fallback")
                ),
                {"no_change": True},
                None,
            )
        return ConstantScorer(0.0), {"no_change": True}, None
    if recipe.kind == "legacy":
        from easy_glm.core.split import TRAIN_FLAG

        from .run import run_model

        source_name = recipe.settings.get("model_name", recipe.name)
        if source_name not in project.models:
            raise ValueError(
                f"Cannot rebuild legacy prefix {source_name!r}: source model is absent"
            )
        split_name = project.data.split.column
        legacy_train = train.with_columns(pl.lit(TRAIN_FLAG).alias(split_name))
        legacy = run_model(project, legacy_train, source_name)
        return (
            RateModelScorer(legacy.rate_model.clone()),
            {"legacy_rebuilt": True},
            legacy.fit,
        )
    if recipe.kind == "glm":
        return _fit_glm(project, train, config, recipe, prefix, progress)
    if recipe.kind == "catboost":
        validate_catboost_tuning_settings(recipe.kind, recipe.settings)
        scorer, evidence = _fit_catboost(train, config, recipe, prefix)
        return scorer, evidence, None
    if recipe.kind == "catalogue":
        validate_catboost_tuning_settings(recipe.kind, recipe.settings)
        catalogue_scorer, catalogue_evidence = _fit_catalogue(
            train, config, recipe, prefix
        )
        return catalogue_scorer, catalogue_evidence, None
    if recipe.kind == "interaction":
        from .pair_distillation import distill_pair_cells
        from .pair_stages import (
            MAX_CELLS,
            _axis,
            _cell_ids,
            _gaussian_training_scale,
            _raw_features,
            _table,
        )
        from .project import PairStageConfig

        if len(recipe.factors) != 2:
            raise ValueError("Interaction table requires exactly two factors")
        a, b = recipe.factors
        stage = PairStageConfig(
            recipe.component_id,
            a,
            b,
            min_weight_share=float(recipe.settings.get("min_weight_share", 0.001)),
        )
        component_project = _project_for_recipe(project, recipe)
        axes = (
            _axis(component_project, train, a, config.weight),
            _axis(component_project, train, b, config.weight),
        )
        n_cells = len(axes[0].table) * len(axes[1].table)
        if n_cells > MAX_CELLS:
            raise ValueError(
                f"Pair {a} × {b} needs {n_cells:,} cells; limit is {MAX_CELLS:,}. "
                "Reduce configured bins/levels"
            )
        candidate = recipe.settings.get(
            "candidate",
            {"iterations": 120, "depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        )
        y, weight = _response(config, train)
        baseline = prefix.predict_unit(train)
        if candidate is None:
            teacher_mean = baseline
            teacher = None
            fit_seconds = 0.0
        else:
            raw, categorical = _raw_features(train, (a, b), axes)
            positive_target = y[y > 0]
            target_scale = (
                _gaussian_training_scale(y, baseline)
                if config.family == "gaussian"
                else (
                    max(1.0, float(np.max(positive_target)) / 1000.0)
                    if config.family == "tweedie" and positive_target.size
                    else 1.0
                )
            )
            started = time.monotonic()
            budget = _active_tuning_budget.get()
            if budget is not None:
                budget.check()
            try:
                teacher = fit_catboost_pair_raw(
                    raw,
                    y,
                    baseline,
                    sample_weight=weight,
                    family=config.family,
                    tweedie_power=prefix.tweedie_power,
                    target_scale=target_scale,
                    iterations=int(candidate.get("iterations", 120)),
                    depth=int(candidate.get("depth", 3)),
                    learning_rate=float(candidate.get("learning_rate", 0.05)),
                    l2_leaf_reg=float(candidate.get("l2_leaf_reg", 3)),
                    seed=int(recipe.settings.get("seed", 42)),
                    cat_features=categorical,
                )
            finally:
                if budget is not None:
                    budget.debit(time.monotonic() - started)
            fit_seconds = time.monotonic() - started
            teacher_mean = teacher.predict_mean(raw, baseline)
        distilled = distill_pair_cells(
            _cell_ids(train, (a, b), axes),
            baseline,
            teacher_mean,
            sample_weight=weight,
            family=config.family,
            tweedie_power=prefix.tweedie_power,
            n_cells=len(axes[0].table) * len(axes[1].table),
            min_weight_share=stage.min_weight_share,
        )
        pair_table = _table(stage, axes, distilled)
        pair_scorer = RateModelScorer(RateModel(1.0, {}, pair_tables=[pair_table]))
        return (
            pair_scorer,
            {
                "candidate": candidate,
                "cells": len(pair_table.cells),
                "fit_seconds": fit_seconds,
            },
            teacher,
        )
    raise ValueError(f"Unsupported component kind {recipe.kind!r}")


def _apply_amendments(scorer: Any, amendments: tuple[dict[str, Any], ...]) -> None:
    """Apply saved deployment edits in order, including during fold replay."""
    for edit in amendments:
        operation = edit.get("operation") or edit.get("kind")
        if isinstance(scorer, LookupScorer):
            values = scorer.table["multiplier"].cast(pl.Float64).to_numpy()
            if operation == "cap":
                lower, upper = edit.get("lower"), edit.get("upper")
                if lower is not None:
                    values = np.maximum(values, float(lower))
                if upper is not None:
                    values = np.minimum(values, float(upper))
            elif operation == "round":
                values = np.round(values, int(edit["decimals"]))
            elif operation == "set":
                key = edit["key"]
                mask = (scorer.table[scorer.key] == key).to_numpy()
                if not mask.any():
                    raise ValueError(
                        f"Amendment key {key!r} is absent from fold catalogue"
                    )
                values[mask] = float(edit["multiplier"])
            else:
                raise ValueError(f"Unsupported lookup amendment {operation!r}")
            if not np.all(np.isfinite(values) & (values > 0)):
                raise ValueError(
                    "Lookup amendment produced nonpositive/nonfinite multipliers"
                )
            scorer.table = scorer.table.with_columns(pl.Series("multiplier", values))
        elif isinstance(scorer, RateModelScorer) and operation in {"cap", "round"}:
            # An interaction component is represented by a one-table
            # RateModelScorer.  Apply a deployment-wide cap/round to its
            # explicitly fitted cells, leaving unsupported cells at their
            # neutral 1.0 fallback.  Keeping this here makes a saved amendment
            # replayable whenever a later suffix is refitted.
            for table in scorer.rate_model.pair_tables:
                for cell in list(table.cells):
                    value = cell.relativity
                    if operation == "cap":
                        if edit.get("lower") is not None:
                            value = max(value, float(edit["lower"]))
                        if edit.get("upper") is not None:
                            value = min(value, float(edit["upper"]))
                    else:
                        value = round(value, int(edit["decimals"]))
                    scorer.rate_model.update_pair_cell(
                        table.stage_id, cell.axis_a_row, cell.axis_b_row, value
                    )
        elif isinstance(scorer, RateModelScorer) and operation == "set_pair_cell":
            required = ("axis_a_from", "axis_a_to", "axis_b_from", "axis_b_to")
            if any(name not in edit for name in required):
                raise ValueError(
                    "Pair-cell amendment lacks exact axis boundaries; CV replay is unavailable"
                )
            try:
                table = scorer.rate_model.get_pair_table(str(edit["stage_id"]))
                first = table.axes[0].table[int(edit["axis_a_row"])]
                second = table.axes[1].table[int(edit["axis_b_row"])]
            except (KeyError, IndexError) as exc:
                raise ValueError(
                    "Pair-cell amendment cannot map onto fitted fold axes"
                ) from exc
            if (first.from_, first.to_, second.from_, second.to_) != (
                edit["axis_a_from"],
                edit["axis_a_to"],
                edit["axis_b_from"],
                edit["axis_b_to"],
            ):
                raise ValueError(
                    "Pair-cell amendment does not match this fold's exact axes; "
                    "CV is unavailable until cuts/levels are fixed"
                )
            scorer.rate_model.update_pair_cell(
                str(edit["stage_id"]),
                int(edit["axis_a_row"]),
                int(edit["axis_b_row"]),
                float(edit["multiplier"]),
            )
        elif isinstance(scorer, RateModelScorer) and operation == "set_rate_row":
            try:
                scorer.rate_model.update_relativity(
                    str(edit["variable"]),
                    edit.get("from"),
                    edit.get("to"),
                    float(edit["multiplier"]),
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "Rate-row amendment cannot map onto this fold's fitted bands or levels; "
                    "use explicit fixed cuts/levels or review the edit"
                ) from exc
        else:
            raise ValueError(f"Component scorer cannot apply amendment {operation!r}")


def fit_stack(
    project: Project,
    train: pl.DataFrame,
    config: ModelConfig,
    *,
    frozen_prefix: StackScorer | None = None,
    progress: Callable[[str], None] | None = None,
    tuning: bool = True,
) -> Any:
    """Fit recipe suffix in order; return a ModelRun whose stack is the scorer."""
    from .run import ModelRun

    model_name = next(
        (name for name, model in project.models.items() if model is config), None
    )
    config = copy.deepcopy(config)
    # The internal conditional level is an artifact, not a user recipe. Older
    # callers may copy every prefix artifact while constructing an append.
    config.components = [item for item in config.components if item.kind != "baseline"]
    stack = _new_stack(config)
    if not config.components:
        raise ValueError("Stack configuration has no components")
    if frozen_prefix is not None:
        _check_prefix(config, frozen_prefix)
        stack = frozen_prefix.clone()
        prior_ids = [
            item.recipe.component_id
            for item in stack.components
            if item.recipe.kind != "baseline"
        ]
        if prior_ids != [
            item.component_id for item in config.components[: len(prior_ids)]
        ]:
            raise ValueError(
                "Frozen prefix does not match the configured component order"
            )
    else:
        prior_ids = []
    if train.is_empty():
        raise ValueError("No training rows")
    fit: Any = None
    tables: dict[str, pl.DataFrame] = {}
    if not stack.components and config.components[0].kind not in ("glm", "legacy"):
        level = _conditional_level(config, train)
        baseline = ComponentRecipe("__baseline__", "Starting level", "baseline")
        stack.components.append(
            ComponentArtifact(baseline, "", ConstantScorer(level), {"log_level": level})
        )
    for index in range(len(prior_ids), len(config.components)):
        recipe = config.components[index]
        if recipe.component_id in (
            item.recipe.component_id for item in stack.components
        ):
            raise ValueError(f"Duplicate component ID {recipe.component_id!r}")
        if progress:
            progress(f"Fitting {recipe.name}")
        tuning_evidence: dict[str, Any] = {}
        needs_tuning = recipe.kind in ("catboost", "catalogue", "interaction") or (
            recipe.kind == "glm"
            and index > 0
            and recipe.settings.get("alpha", config.penalty.alpha) is None
        )
        if tuning and needs_tuning and recipe.settings.get("tune", True):
            from .stack_tuning import tune_component

            recipe, tuning_evidence = tune_component(
                project, train, config, recipe, config.components[:index]
            )
            config.components[index] = recipe
        elif needs_tuning and recipe.settings.get("tune") is False:
            tuning_evidence = {
                "kind": "manual_catboost_fit",
                "reason": "Automatic tuning is off; fitted the configured CatBoost settings once",
                "catboost_tuning_seconds": 0.0,
            }
        artifact, new_fit = _append_component(
            project, train, config, recipe, stack, progress
        )
        artifact.evidence.update(tuning_evidence)
        stack.predict_unit(train)
        if new_fit is not None and recipe.kind == "glm":
            fit = new_fit
            tables = rate_tables(fit, base=recipe.settings.get("base", config.base))
    # The legacy fields are intentionally empty for mixed stacks; every consumer
    # must dispatch through ``run.stack`` rather than accidentally scoring a stub.
    name = model_name or next(
        (
            name
            for name, model in project.models.items()
            if [
                item.component_id
                for item in model.components
                if item.kind != "baseline"
            ]
            == [item.component_id for item in config.components]
        ),
        "Stack",
    )
    y, weight = _response(config, train)
    expected = stack.predict_unit(train)
    metrics = {
        "train": {
            "actual": float(np.sum(weight * y)),
            "expected": float(np.sum(weight * expected)),
            "ae": float(np.sum(weight * y) / np.sum(weight * expected)),
        }
    }
    return ModelRun(
        name=name,
        config=config,
        spec=cast(DesignSpec, None),
        fit=cast(GLMFit, fit),
        rate_model=cast(RateModel, None),
        tables=tables,
        metrics=metrics,
        project_snapshot=project.to_dict(),
        train_rows=train.height,
        stack=stack,
    )


def _append_component(
    project: Project,
    train: pl.DataFrame,
    config: ModelConfig,
    recipe: ComponentRecipe,
    stack: StackScorer,
    progress: Callable[[str], None] | None = None,
) -> tuple[ComponentArtifact, Any | None]:
    """Fit and append the accepted scorer, with identical fold/full behaviour."""
    fingerprint = stack.fingerprint()
    fit_frame = stack.enrich(train)
    scorer, evidence, new_fit = _fit_component(
        project, fit_frame, config, recipe, stack, progress
    )
    if (
        recipe.kind == "catalogue"
        and len(stack.components) == 1
        and stack.components[0].recipe.kind == "baseline"
    ):
        level = math.exp(stack.components[0].scorer.log_value)
        scorer.table = scorer.table.with_columns(
            (pl.col("multiplier") * level).alias("multiplier")
        )
        stack.components.clear()
        evidence["starting_level_in_lookup"] = level
    _apply_amendments(scorer, recipe.amendments)
    artifact = ComponentArtifact(recipe, fingerprint, scorer, evidence)
    stack.components.append(artifact)
    stack.predict_unit(train)
    return artifact, new_fit


def fit_prefix(
    project: Project,
    train: pl.DataFrame,
    config: ModelConfig,
    prefix_recipes: list[ComponentRecipe],
    upcoming_recipe: ComponentRecipe,
) -> StackScorer:
    """Rebuild a selected deployed prefix on fold training rows only."""
    if prefix_recipes:
        return fit_stack(
            project, train, replace(config, components=prefix_recipes), tuning=False
        ).stack
    stack = _new_stack(config)
    if upcoming_recipe.kind not in ("glm", "legacy"):
        level = _conditional_level(config, train)
        baseline = ComponentRecipe("__baseline__", "Starting level", "baseline")
        stack.components.append(
            ComponentArtifact(baseline, "", ConstantScorer(level), {"log_level": level})
        )
    return stack


def amend_stack(
    project: Project,
    train: pl.DataFrame,
    run: Any,
    component_id: str,
    operation: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    decimals: int | None = None,
    key: Any | None = None,
    multiplier: float | None = None,
    refit_later: bool = False,
) -> Any:
    """Create a new immutable checkpoint with an accepted component edit."""
    if run.stack is None:
        raise ValueError("Only stack runs can be amended with amend_stack")
    edit: dict[str, Any] = {"operation": operation}
    if operation == "cap":
        edit.update(lower=lower, upper=upper)
    elif operation == "round":
        if decimals is None:
            raise ValueError("Rounding requires decimals")
        edit["decimals"] = int(decimals)
    elif operation == "set":
        edit.update(key=key, multiplier=multiplier)
    else:
        raise ValueError(f"Unsupported amendment operation {operation!r}")
    config = copy.deepcopy(run.config)
    index = next(
        (
            i
            for i, recipe in enumerate(config.components)
            if recipe.component_id == component_id
        ),
        None,
    )
    if index is None:
        raise KeyError(component_id)
    recipe = config.components[index]
    config.components[index] = replace(recipe, amendments=(*recipe.amendments, edit))
    stack = run.stack.clone()
    artifact = stack.component(component_id)
    _apply_amendments(artifact.scorer, (edit,))
    artifact.recipe = config.components[index]
    if refit_later and index < len(config.components) - 1:
        stack.components = stack.components[
            : index + (2 if stack.components[0].recipe.kind == "baseline" else 1)
        ]
        rebuilt = fit_stack(project, train, config, frozen_prefix=stack)
        return rebuilt
    for later in stack.components:
        if later.recipe.component_id in [
            item.component_id for item in config.components[index + 1 :]
        ]:
            later.status = "amended_prefix_kept_fixed"
    y, weight = _response(config, train)
    predicted = stack.predict_unit(train)
    metrics = {
        "train": {
            "actual": float(np.sum(weight * y)),
            "expected": float(np.sum(weight * predicted)),
            "ae": float(np.sum(weight * y) / np.sum(weight * predicted)),
        }
    }
    return replace(run, config=config, stack=stack, metrics=metrics)


def stack_from_legacy(project: Project, run: Any) -> StackScorer:
    """Adapt the exact deployed legacy RateModel as a single frozen component."""
    if run.stack is not None:
        return run.stack.clone()
    cfg = run.config
    if run.rate_model.metadata.link != "log":
        raise ValueError(
            "The preceding model uses a non-log link and cannot supply multiplicative corrections"
        )
    if run.rate_model.metadata.offset_col and not run.rate_model.metadata.offset_is_log:
        raise ValueError("The preceding model offset is not already on the log scale")
    if run.rate_model.metadata.offset_col != cfg.offset:
        raise ValueError(
            "The preceding model and recipe disagree about the external offset"
        )
    # ``None`` means the GLM family default.  ``link_for`` resolves a Gaussian
    # GLM default to log, whereas the native stack deliberately treats a
    # missing Gaussian link as identity and refuses it.  An already fitted
    # rate model tells us the actual link, so use that immutable fact when
    # adapting it into a frozen multiplicative prefix.  Do not mutate ``cfg``
    # or rewrite the saved recipe while doing so.
    stack = _new_stack(replace(cfg, link=run.rate_model.metadata.link))
    recipe = ComponentRecipe(
        "legacy",
        "Accepted legacy scorer",
        "legacy",
        tuple(cfg.predictors),
        {"model_name": run.name},
    )
    stack.components.append(
        ComponentArtifact(
            recipe, "", RateModelScorer(run.rate_model.clone()), {"adapted": True}
        )
    )
    stages = [stage.stage_id for stage in cfg.pair_stages]

    def checkpoint_name(prefix_count: int, fallback: str) -> str:
        expected = stages[:prefix_count]
        choices = [
            name
            for name, model in project.models.items()
            if [stage.stage_id for stage in model.pair_stages] == expected
            and model.predictors == cfg.predictors
            and model.target == cfg.target
            and model.family == cfg.family
        ]
        if prefix_count == len(stages) and run.name in choices:
            return run.name
        return choices[0] if choices else fallback

    main = run.rate_model.clone()
    main.pair_tables = []
    main_name = checkpoint_name(0, "Main effects")
    stack.virtual_components.append(
        ComponentArtifact(
            ComponentRecipe("legacy_main", main_name, "glm", tuple(cfg.predictors)),
            "",
            RateModelScorer(main),
            {"from_legacy": run.name, "inspection_only": True},
        )
    )
    for index, table in enumerate(run.rate_model.pair_tables, start=1):
        stage_name = checkpoint_name(index, table.stage_id)
        pair_only = RateModel(1.0, {}, pair_tables=[copy.deepcopy(table)])
        stack.virtual_components.append(
            ComponentArtifact(
                ComponentRecipe(
                    f"legacy_pair_{table.stage_id}",
                    stage_name,
                    "interaction",
                    tuple(table.parents),
                ),
                "",
                RateModelScorer(pair_only),
                {
                    "from_legacy": run.name,
                    "stage_id": table.stage_id,
                    "inspection_only": True,
                },
            )
        )
    return stack
