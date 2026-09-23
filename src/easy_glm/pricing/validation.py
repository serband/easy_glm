"""Validation and explicit model assessment for the pricing facade."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from glum import GeneralizedLinearRegressor

from easy_glm.core.fit import resolve_family
from easy_glm.workflow.diagnostics import family_metrics, gini, totals, unit_values
from easy_glm.workflow.project import FAMILIES, Project
from easy_glm.workflow.run import run_model

if TYPE_CHECKING:
    from .model import PricingModel


def _family_instance(config: Any) -> Any:
    family, _family_name, default_link = resolve_family(
        config.family,
        float(config.tweedie_power) if config.family == "tweedie" else None,
    )
    return GeneralizedLinearRegressor(
        family=family, link=config.link or default_link
    ).family_instance


def require_column(data: pl.DataFrame, column: str | None, label: str) -> str:
    """Return a required column name, with a short user-facing error."""
    if not column:
        raise ValueError(f"{label} column is required")
    if column not in data.columns:
        raise KeyError(f"{label} column {column!r} is not in the data")
    return column


def validate_setup(
    data: pl.DataFrame,
    *,
    family: str,
    target: str,
    weight: str | None,
    offset: str | None,
    identifier: str | None,
    split: str | None,
    ignored: Iterable[str],
    divide_target_by_weight: bool,
) -> None:
    """Validate columns and mutually dependent setup options."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; use one of {FAMILIES}")
    require_column(data, target, "Target")
    for column, label in (
        (weight, "Weight"),
        (offset, "Offset"),
        (identifier, "ID"),
        (split, "Split"),
    ):
        if column is not None:
            require_column(data, column, label)
    for column in ignored:
        require_column(data, column, "Ignored")
    if divide_target_by_weight and weight is None:
        raise ValueError("Claim frequency needs an exposure/weight column")
    assigned = [c for c in (target, weight, offset, identifier, split) if c]
    if len(set(assigned)) != len(assigned):
        raise ValueError(
            "Target, weight, offset, ID and split must be different columns"
        )


def protected_columns(project: Project, model_name: str | None = None) -> set[str]:
    """Columns that cannot be investigated as pricing factors."""
    protected = {
        column
        for column, role in project.data.roles.items()
        if role
        in {
            "target",
            "weight",
            "exposure",
            "offset",
            "current_premium",
            "split",
            "time",
            "id",
            "ignore",
        }
    }
    protected.add(project.data.split.column)
    if model_name and model_name in project.models:
        cfg = project.models[model_name]
        protected.update(c for c in (cfg.target, cfg.weight, cfg.offset) if c)
    return protected


def validate_factors(
    project: Project,
    data: pl.DataFrame,
    factors: Sequence[str],
    *,
    pair_only: bool = False,
    model_name: str | None = None,
) -> list[str]:
    """Validate main factors or raw pair parents and return a detached list."""
    values = [str(value) for value in factors]
    if len(set(values)) != len(values):
        raise ValueError("Factors must be unique")
    protected = protected_columns(project, model_name)
    for factor in values:
        if factor not in data.columns:
            raise KeyError(f"Factor {factor!r} is not in the data")
        if factor in protected:
            raise ValueError(f"{factor!r} is a protected column and cannot be a factor")
        if pair_only and model_name:
            problem = project.pair_parent_problem(project.models[model_name], factor)
            if problem:
                raise ValueError(problem)
    return values


def metric_row(model: PricingModel, subset: str) -> dict[str, Any]:
    """Family-aware assessment for one explicit subset."""
    frame = model._frame(subset)
    if frame.is_empty():
        raise ValueError(f"No {subset} rows are available for validation")
    prediction = model._run.predict(frame)
    actual, expected, exposure = totals(frame, model._run.config, prediction)
    observed_rate, fitting_weight = unit_values(frame, model._run.config)
    family = _family_instance(model._run.config)
    deviance = float(
        family.deviance(observed_rate, prediction, sample_weight=fitting_weight)
    )
    expected_total = float(np.sum(expected))
    weight_total = float(np.sum(fitting_weight))
    if not np.isfinite(expected_total) or weight_total <= 0:
        raise ValueError(
            f"Cannot assess {subset}: totals must be finite with positive weight"
        )
    cfg = model._run.config
    ae = (
        float(np.sum(actual) / expected_total)
        if cfg.family not in {"gaussian", "normal"} and expected_total > 0
        else None
    )
    ranking = (
        float(gini(actual, expected, exposure))
        if cfg.family in {"poisson", "gamma", "tweedie", "inverse_gaussian"}
        and np.all(actual >= 0)
        and np.all(expected >= 0)
        and float(np.sum(actual)) > 0
        and expected_total > 0
        else None
    )
    result = {
        "model": model.name,
        "subset": subset,
        "rows": frame.height,
        "exposure": float(np.sum(exposure)),
        "actual": float(np.sum(actual)),
        "expected": expected_total,
        "ae": ae,
        "mean_deviance": deviance / weight_total,
        # Ranking is deliberately supplementary to calibration and deviance.
        "gini": ranking,
    }
    result.update(family_metrics(cfg.family, observed_rate, prediction, fitting_weight))
    return result


def _cross_validated_row(model: PricingModel, *, folds: int = 5) -> dict[str, Any]:
    """Refit the complete ordered model inside every training fold."""
    frame = model._frame("train")
    if folds < 2 or folds > frame.height:
        raise ValueError("folds must be between 2 and the number of training rows")
    seed = int(model._project.data.split.seed)
    assignment = np.random.default_rng(seed).permutation(frame.height) % folds
    predictions = np.empty(frame.height, dtype=float)
    split_column = "__easy_glm_pricing_cv__"
    while split_column in frame.columns:
        split_column += "_"
    for fold in range(folds):
        fold_project = Project.from_dict(model._project.to_dict())
        old_split = fold_project.data.split.column
        if fold_project.data.roles.get(old_split) == "split":
            fold_project.data.roles.pop(old_split)
        fold_project.data.roles[split_column] = "split"
        fold_project.data.split.mode = "column"
        fold_project.data.split.column = split_column
        fold_project.data.split.train_value = 1
        fold_project.data.split.holdout_value = 0
        marked = frame.with_columns(
            pl.Series(split_column, (assignment != fold).astype(np.int64))
        )
        fold_run = run_model(fold_project, marked, model.name)
        validation = marked.filter(pl.col(split_column) == 0)
        predictions[assignment == fold] = fold_run.predict(validation)
    cfg = model._run.config
    actual, expected, exposure = totals(frame, cfg, predictions)
    observed_rate, fitting_weight = unit_values(frame, cfg)
    family = _family_instance(cfg)
    deviance = float(
        family.deviance(observed_rate, predictions, sample_weight=fitting_weight)
    )
    expected_total = float(np.sum(expected))
    weight_total = float(np.sum(fitting_weight))
    if not np.isfinite(expected_total) or weight_total <= 0:
        raise ValueError(
            "Cannot assess cross-validation: totals must be finite with positive weight"
        )
    ae = (
        float(np.sum(actual) / expected_total)
        if cfg.family not in {"gaussian", "normal"} and expected_total > 0
        else None
    )
    ranking = (
        float(gini(actual, expected, exposure))
        if cfg.family in {"poisson", "gamma", "tweedie", "inverse_gaussian"}
        and np.all(actual >= 0)
        and np.all(expected >= 0)
        and float(np.sum(actual)) > 0
        and expected_total > 0
        else None
    )
    result = {
        "model": model.name,
        "subset": "cross_validation",
        "rows": frame.height,
        "folds": folds,
        "exposure": float(np.sum(exposure)),
        "actual": float(np.sum(actual)),
        "expected": expected_total,
        "ae": ae,
        "mean_deviance": deviance / weight_total,
        "gini": ranking,
    }
    result.update(
        family_metrics(cfg.family, observed_rate, predictions, fitting_weight)
    )
    return result


def compare_models(
    model: PricingModel,
    other: PricingModel | Sequence[PricingModel],
    *,
    subset: str = "train",
    cv: bool = False,
) -> pl.DataFrame:
    """Compare immutable checkpoints on training evidence or explicit CV."""
    others = list(other) if isinstance(other, Sequence) else [other]
    models = [model, *others]
    if not cv and subset not in {"train", "holdout"}:
        raise ValueError("subset must be 'train' or 'holdout'")
    reference_cfg = model._run.config
    reference_frame = model._frame("train" if cv else subset)
    reference_id = model._project.column_with_role("id")
    for candidate in others:
        candidate_cfg = candidate._run.config
        comparable = (
            candidate_cfg.family,
            candidate_cfg.link,
            candidate_cfg.target,
            candidate_cfg.weight,
            candidate_cfg.offset,
            candidate_cfg.divide_target_by_weight,
            candidate_cfg.tweedie_power if candidate_cfg.family == "tweedie" else None,
        )
        expected = (
            reference_cfg.family,
            reference_cfg.link,
            reference_cfg.target,
            reference_cfg.weight,
            reference_cfg.offset,
            reference_cfg.divide_target_by_weight,
            reference_cfg.tweedie_power if reference_cfg.family == "tweedie" else None,
        )
        if comparable != expected:
            raise ValueError(
                "Models must use the same family, response, weight and offset units "
                "before they can be compared"
            )
        candidate_frame = candidate._frame("train" if cv else subset)
        candidate_id = candidate._project.column_with_role("id")
        if reference_frame.height != candidate_frame.height:
            raise ValueError("Models must be assessed on the same policy rows")
        if reference_id and candidate_id:
            same_rows = reference_frame[reference_id].equals(
                candidate_frame[candidate_id]
            )
        else:
            identity = [
                column
                for column in (
                    reference_cfg.target,
                    reference_cfg.weight,
                    reference_cfg.offset,
                    model._project.data.split.column,
                )
                if column
                and column in reference_frame.columns
                and column in candidate_frame.columns
            ]
            same_rows = reference_frame.select(identity).equals(
                candidate_frame.select(identity)
            )
        if not same_rows:
            raise ValueError(
                "Models must be assessed on the same ordered policy rows and split"
            )
    rows = [
        (_cross_validated_row(item) if cv else metric_row(item, subset))
        for item in models
    ]
    if cv:
        for item, row in zip(models, rows, strict=True):
            validation = item._evidence.setdefault("validation", {})
            validation["cross_validation"] = row
    return pl.DataFrame(rows)


def validate_holdout_models(
    model: PricingModel,
    compare_with: PricingModel | Sequence[PricingModel] | None = None,
) -> pl.DataFrame:
    """Open the reserved holdout only through an explicit user call."""
    if compare_with is None:
        others: list[PricingModel] = []
    elif isinstance(compare_with, Sequence):
        others = list(compare_with)
    else:
        others = [compare_with]
    models = [model, *others]
    result = (
        compare_models(model, others, subset="holdout", cv=False)
        if others
        else pl.DataFrame([metric_row(model, "holdout")])
    )
    # Validation is observational evidence rather than part of the fitted run.
    # Keep run.metrics training-only, but make the user's explicit holdout opening
    # durable in JSON/Excel audit outputs for every checkpoint compared.
    for item, row in zip(models, result.to_dicts(), strict=True):
        validation = item._evidence.setdefault("validation", {})
        validation["holdout"] = row
    return result
