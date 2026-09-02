from __future__ import annotations

import numpy as np
import polars as pl

from .models import VariableConfig


def score_numeric(values: np.ndarray, config: VariableConfig) -> np.ndarray:
    if config.breakpoints is None or config.relativities is None:
        return _score_numeric_fallback(values, config)

    values = np.asarray(values, dtype=float)
    nan_mask = np.isnan(values)
    if nan_mask.any() and config.null_relativity is None:
        raise ValueError(
            "Some numeric values did not match any bin. "
            "Check for NaN values in the input data."
        )

    indices = np.searchsorted(config.breakpoints, values, side="right")
    result = config.relativities[indices]
    if nan_mask.any():
        result = result.copy()
        result[nan_mask] = config.null_relativity
    return result


def score_categorical(series: pl.Series, config: VariableConfig) -> np.ndarray:
    cat_map = config.cat_map
    if cat_map is None:
        return _score_categorical_fallback(series, config)

    fallback = config.fallback
    # Levels are stored as strings; compare as strings so integer-typed
    # categorical columns match their levels.
    arr = series.cast(pl.Utf8).to_numpy()
    result = np.full(len(arr), fallback, dtype=float)

    if cat_map:
        result[series.is_null().to_numpy()] = fallback
        for level, rel in cat_map.items():
            result[arr == level] = rel

    return result


# --------------------------------------------------------------------------
# rate-table row indices (shared by interactions and diagnostics)
# --------------------------------------------------------------------------
def row_index(series: pl.Series, config: VariableConfig) -> np.ndarray:
    """Rate-table row index (position in ``config.table``) of every value of a
    main-effect variable: numeric → band by ``searchsorted`` (nulls → the null
    row); categorical → level position (unseen / null → the Other row).

    Raises for nulls in a numeric table that has no null row, exactly like
    :func:`score_numeric`.
    """
    if config.type == "numeric":
        if config.breakpoints is None:
            raise ValueError("numeric config not precomputed")
        values = np.asarray(series.cast(pl.Float64).to_numpy(), dtype=float)
        idx = np.searchsorted(config.breakpoints, values, side="right").astype(np.int64)
        nan_mask = np.isnan(values)
        if nan_mask.any():
            if config.null_relativity is None:
                raise ValueError(
                    "Some numeric values did not match any bin. "
                    "Check for NaN values in the input data."
                )
            idx[nan_mask] = len(config.table) - 1  # the (None, None) row
        return idx
    if config.type == "categorical":
        if config.level_index is None:
            raise ValueError("categorical config not precomputed")
        other = len(config.table) - 1
        vals = series.cast(pl.Utf8)
        keys = list(config.level_index)
        idx = vals.replace_strict(
            keys,
            [config.level_index[k] for k in keys],
            default=other,
            return_dtype=pl.Int64,
        )
        return idx.fill_null(other).to_numpy().astype(np.int64)
    raise ValueError(f"row_index is only defined for main effects, not {config.type!r}")


def score_interaction(
    data: pl.DataFrame,
    config: VariableConfig,
    variables: dict[str, VariableConfig],
) -> np.ndarray:
    """Relativity of every row's cell in a two-way interaction table."""
    if config.parents is None or config.cell_matrix is None:
        raise ValueError("interaction config not precomputed")
    a, b = config.parents
    ia = row_index(data[a], variables[a])
    ib = row_index(data[b], variables[b])
    return config.cell_matrix[ia, ib]


def _score_numeric_fallback(values: np.ndarray, config: VariableConfig) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    for row in config.table:
        if row.from_ is None and row.to_ is None:
            result[np.isnan(values)] = row.relativity
            continue
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
            known[str(row.from_)] = row.relativity
        else:
            fallback = row.relativity

    arr = series.cast(pl.Utf8).to_numpy()
    result = np.full(len(arr), fallback, dtype=float)

    for level, rel in known.items():
        result[arr == level] = rel

    null_mask = series.is_null().to_numpy()
    result[null_mask] = fallback

    return result
