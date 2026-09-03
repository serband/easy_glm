from __future__ import annotations

import numpy as np
import polars as pl

from easy_glm.engine import RateModel
from easy_glm.engine.models import CellRow, FromToRow, level_labels

FORMULAS: dict[str, str] = {
    "sum_weighted": "sum(target × weight) / sum(weight)",
    "sum_unweighted": "sum(target) / count",
    "sum_over_weight": "sum(target) / sum(weight)",
}


def default_formula(metadata) -> str:
    """A/E formula implied by the model's metadata.

    A count target divided by an exposure weight (``divide_target_by_weight``)
    needs ``sum(target) / sum(weight)`` for both actual and expected; anything
    else (a rate target, or a 0.3 file where the flag is unknown) uses the
    exposure-weighted mean.
    """
    if (
        metadata is not None
        and metadata.divide_target_by_weight
        and metadata.weight_col
    ):
        return "sum_over_weight"
    return "sum_weighted"


def _predictions_on_total_scale(rm: RateModel, data: pl.DataFrame) -> np.ndarray:
    """Expected values on the same scale as the target column: for a count model
    whose RateModel has no exposure column, multiply by the weight so that
    ``sum(expected) / sum(weight)`` is a rate like ``sum(claims) / sum(weight)``."""
    meta = rm.metadata
    if meta.divide_target_by_weight and meta.exposure_col is None and meta.weight_col:
        return rm.predict(data, exposure_col=meta.weight_col)
    return rm.predict(data)


def compute_actual_expected(
    rm: RateModel,
    data: pl.DataFrame,
    variable: str,
    formula: str = "sum_weighted",
) -> dict:
    target = rm.metadata.target
    weight_col = rm.metadata.weight_col
    train_test_col = rm.metadata.train_test_col

    if target is None:
        raise ValueError("Model metadata missing 'target' column")
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    predictions = _predictions_on_total_scale(rm, data)
    data = data.with_columns(pred=pl.Series("pred", predictions))

    subsets = {"all": data}
    if train_test_col and train_test_col in data.columns:
        subsets["train"] = data.filter(pl.col(train_test_col) == 1)
        subsets["test"] = data.filter(pl.col(train_test_col) == 0)

    config = rm.variables[variable]
    rows = config.table
    level_edges = level_labels(rows, config.other_label)

    results: dict[str, list[dict]] = {}
    for subset_name, subset in subsets.items():
        results[subset_name] = []
        for i, row in enumerate(rows):
            if config.type == "interaction":
                a, b = config.parents
                mask = _mask_for_row(
                    subset, a, FromToRow(row.from_a, row.to_a, 1.0), rm.variables[a]
                ) & _mask_for_row(
                    subset, b, FromToRow(row.from_b, row.to_b, 1.0), rm.variables[b]
                )
            else:
                mask = _mask_for_row(subset, variable, row, config)
            matched = subset.filter(mask)
            if matched.is_empty():
                results[subset_name].append(
                    {
                        "level": level_edges[i],
                        "actual": 0.0,
                        "expected": 0.0,
                        "exposure": 0.0,
                    }
                )
                continue

            actual = _compute_actual(matched, target, weight_col, formula)
            expected = _compute_actual(matched, "pred", weight_col, formula)
            exposure = (
                float(matched[weight_col].sum())
                if weight_col and weight_col in matched.columns
                else float(len(matched))
            )
            results[subset_name].append(
                {
                    "level": level_edges[i],
                    "actual": actual,
                    "expected": expected,
                    "exposure": exposure,
                }
            )

    return {"subsets": results, "variable": variable}


def _mask_for_row(data: pl.DataFrame, variable: str, row, config=None) -> pl.Series:
    if isinstance(row, CellRow):
        raise TypeError("pass the parent rows of a cell separately")
    col = data[variable]
    if row.from_ is None and row.to_ is None:
        # Numeric: the null bin. Categorical: Other = unseen levels or null.
        if config is not None and config.type == "categorical":
            known = [str(r.from_) for r in config.table if r.from_ is not None]
            return (~col.cast(pl.Utf8).is_in(known)).fill_null(True) | col.is_null()
        return col.is_null() | col.cast(pl.Float64).is_nan().fill_null(False)
    if row.from_ is None:
        return col < float(row.to_)
    if row.to_ is None:
        return col >= float(row.from_)
    if row.from_ == row.to_:
        return col.cast(pl.Utf8) == str(row.from_)
    return (col >= float(row.from_)) & (col < float(row.to_))


def _compute_actual(
    df: pl.DataFrame, value_col: str, weight_col: str | None, formula: str
) -> float:
    values = df[value_col]
    if formula == "sum_weighted":
        if weight_col and weight_col in df.columns:
            weights = df[weight_col]
            return float((values * weights).sum() / weights.sum())
        return float(values.mean())
    elif formula == "sum_over_weight":
        if weight_col and weight_col in df.columns:
            return float(values.sum() / df[weight_col].sum())
        return float(values.sum())
    else:
        return float(values.mean())
