"""Residual searches with pricing-model inputs and notebook-friendly outputs."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import polars as pl

from easy_glm.core.design import NUMERIC_DTYPES, frequent_levels, quantile_knots
from easy_glm.workflow.diagnostics import (
    pearson_dispersion,
    residual_factor_search,
    residual_pair_search,
    totals,
)
from easy_glm.workflow.explore import band_expr

from .validation import protected_columns, validate_factors
from .views import DisplayResult


def _candidate_columns(model: Any) -> list[str]:
    frame = model._frame("train")
    protected = protected_columns(model._project, model.name)
    return [
        name
        for name in frame.columns
        if name not in protected
        and frame[name].dtype != pl.Object
        and not frame[name].dtype.is_nested()
    ]


def _residual_inputs(model: Any) -> tuple[pl.DataFrame, Any, Any, Any]:
    frame = model._frame("train")
    prediction = model._run.rate_model.predict(frame, exposure_col=None)
    actual, expected, weight = totals(frame, model._run.config, prediction)
    return frame, actual, expected, weight


def _design_bands(
    model: Any, frame: pl.DataFrame, variables: list[str], default_bins: int
) -> tuple[dict[str, list[float]], dict[str, list[str]]]:
    knots: dict[str, list[float]] = {}
    levels: dict[str, list[str]] = {}
    scorer = model._run.rate_model
    for variable in variables:
        config = scorer.variables.get(variable)
        if config is not None and config.type in ("numeric", "linear"):
            boundaries = {
                float(value)
                for row in config.table
                for value in (row.from_, row.to_)
                if value is not None
            }
            if boundaries:
                knots[variable] = sorted(boundaries)
            continue
        if config is not None and config.type == "categorical":
            levels[variable] = [
                str(row.from_) for row in config.table if row.from_ is not None
            ]
            continue
        design = model._project.design.variables.get(variable)
        kind = design.kind if design is not None else None
        numeric = frame[variable].dtype in NUMERIC_DTYPES
        if kind == "categorical" or not numeric:
            if design is not None and design.levels:
                levels[variable] = [str(value) for value in design.levels]
            else:
                minimum = (
                    design.min_level_share
                    if design is not None and design.min_level_share is not None
                    else model._project.design.defaults.min_level_share
                )
                maximum = design.max_levels if design is not None else None
                weight_col = model._run.config.weight
                levels[variable] = frequent_levels(
                    frame[variable],
                    min_share=minimum,
                    max_levels=maximum,
                    weights=frame[weight_col] if weight_col else None,
                )
        elif design is not None and isinstance(design.knots, (list, tuple)):
            knots[variable] = [float(value) for value in design.knots]
        else:
            count = (
                design.n_bins
                if design is not None and design.n_bins is not None
                else model._project.design.defaults.n_bins or default_bins
            )
            knots[variable] = quantile_knots(frame[variable], count)
    return knots, levels


def _banded_search_frame(
    frame: pl.DataFrame,
    variables: list[str],
    knots: dict[str, list[float]],
    levels: dict[str, list[str]],
) -> pl.DataFrame:
    """Represent saved training bands as labels for the one-way search."""
    out = frame
    for variable in variables:
        if variable in levels:
            kept = levels[variable]
            out = out.with_columns(
                pl.when(pl.col(variable).cast(pl.Utf8).is_in(kept))
                .then(pl.col(variable).cast(pl.Utf8))
                .otherwise(pl.lit("Other / Unknown"))
                .alias(variable)
            )
        elif variable in knots and knots[variable]:
            out = out.with_columns(band_expr(variable, knots[variable]).alias(variable))
    return out


def find_missing_factors(
    model: Any,
    candidates: Sequence[str] | None = None,
    *,
    n_bins: int = 10,
    min_expected: float = 3.0,
) -> DisplayResult:
    """Rank allowed columns omitted from the main GLM using training residuals."""
    frame, actual, expected, weight = _residual_inputs(model)
    available = _candidate_columns(model)
    requested = available if candidates is None else list(candidates)
    requested = validate_factors(
        model._project, frame, requested, model_name=model.name
    )
    fitted = set(model._run.config.predictors)
    variables = [name for name in requested if name not in fitted]
    knots, levels = _design_bands(model, frame, variables, n_bins)
    search_frame = _banded_search_frame(frame, variables, knots, levels)
    table = residual_factor_search(
        search_frame,
        variables,
        actual,
        expected,
        weight,
        n_bins=n_bins,
        min_expected=min_expected,
        dispersion=pearson_dispersion(actual, expected),
    )
    return DisplayResult(
        table,
        title=f"{model.name}: missing-factor search (training only)",
        note="Large positive signal identifies residual structure for review; it does not automatically add a factor.",
    )


def find_interactions(
    model: Any,
    candidates: Sequence[str] | None = None,
    *,
    n_bins: int = 8,
    min_expected: float = 3.0,
    min_cell_share: float = 0.0,
    top: int = 20,
) -> DisplayResult:
    """Rank unused factor pairs by margin-adjusted training residual structure."""
    frame, actual, expected, weight = _residual_inputs(model)
    variables = _candidate_columns(model) if candidates is None else list(candidates)
    variables = validate_factors(
        model._project, frame, variables, pair_only=True, model_name=model.name
    )
    used = {frozenset(table.parents) for table in model._run.rate_model.pair_tables}
    pairs = [pair for pair in combinations(variables, 2) if frozenset(pair) not in used]
    knots, levels = _design_bands(model, frame, variables, n_bins)
    table = residual_pair_search(
        frame,
        variables,
        actual,
        expected,
        weight,
        knots=knots,
        levels=levels,
        n_bins=n_bins,
        min_expected=min_expected,
        min_cell_share=min_cell_share,
        pairs=pairs,
        top=top,
        dispersion=pearson_dispersion(actual, expected),
    )
    return DisplayResult(
        table,
        title=f"{model.name}: interaction search (training only)",
        note="Pairs already deployed in either order are excluded. Review support and A/E before fitting a stage.",
    )
