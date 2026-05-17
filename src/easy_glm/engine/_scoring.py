from __future__ import annotations

import numpy as np
import polars as pl

from .models import VariableConfig


def score_numeric(values: np.ndarray, config: VariableConfig) -> np.ndarray:
    if config.breakpoints is None or config.relativities is None:
        return _score_numeric_fallback(values, config)

    if np.any(np.isnan(values)):
        raise ValueError(
            "Some numeric values did not match any bin. "
            "Check for NaN values in the input data."
        )

    indices = np.searchsorted(config.breakpoints, values, side="right")
    return config.relativities[indices]


def score_categorical(series: pl.Series, config: VariableConfig) -> np.ndarray:
    cat_map = config.cat_map
    if cat_map is None:
        return _score_categorical_fallback(series, config)

    fallback = config.fallback
    arr = series.to_numpy()
    result = np.full(len(arr), fallback, dtype=float)

    if cat_map:
        result[series.is_null().to_numpy()] = fallback
        for level, rel in cat_map.items():
            result[arr == level] = rel

    return result


def _score_numeric_fallback(values: np.ndarray, config: VariableConfig) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    for row in config.table:
        low = -np.inf if row.from_ is None else float(row.from_)
        high = np.inf if row.to_ is None else float(row.to_)
        mask = (values >= low) & (values < high)
        result[mask] = row.relativity
    if np.any(np.isnan(result)):
        raise ValueError(
            "Some numeric values did not match any bin. "
            "Check for NaN values in the input data."
        )
    return result


def _score_categorical_fallback(
    series: pl.Series, config: VariableConfig
) -> np.ndarray:
    known: dict = {}
    fallback = config.fallback
    for row in config.table:
        if row.from_ is not None:
            known[row.from_] = row.relativity
        else:
            fallback = row.relativity

    arr = series.to_numpy()
    result = np.full(len(arr), fallback, dtype=float)

    for level, rel in known.items():
        result[arr == level] = rel

    null_mask = series.is_null().to_numpy()
    result[null_mask] = fallback

    return result
