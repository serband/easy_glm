"""Descriptive data summaries and bounded numeric correlations for reports."""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
CORRELATION_SAMPLE_ROWS = 10_000
CORRELATION_MATRIX_COLUMNS = 20
CORRELATION_PAIR_LIMIT = 50
CORRELATION_BLOCK_COLUMNS = 64
NUMERIC_STATISTICS = (
    "min",
    "max",
    "range",
    "mean",
    "median",
    "std",
    "skewness",
    "kurtosis",
)


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _label(value: float) -> str:
    return format(value, ".4g")


def _numeric_stats(values: FloatArray) -> dict[str, Any]:
    """Compute moments after scaling, avoiding overflow in intermediate powers."""
    result: dict[str, Any] = dict.fromkeys(NUMERIC_STATISTICS)
    result["finite"] = count = len(values)
    result["histogram"] = []
    if not count:
        return result
    lo, hi = float(values.min()), float(values.max())
    scale = float(np.max(np.abs(values))) or 1.0
    scaled = values / scale
    shifted = scaled - scaled[0]
    shift_mean = float(shifted.mean())
    centre = float(scaled[0]) + shift_mean
    centred = shifted - shift_mean
    second = float(np.mean(centred**2))
    result.update(
        min=lo,
        max=hi,
        range=_finite(hi - lo),
        mean=_finite(centre * scale),
        median=_finite(float(np.median(scaled)) * scale),
    )
    if count > 1:
        result["std"] = _finite(math.sqrt(second * count / (count - 1)) * scale)
    if second > 0:
        # Corrected Fisher-Pearson skewness and unbiased Fisher excess kurtosis.
        normalised = centred / math.sqrt(second)
        if count > 2:
            result["skewness"] = _finite(
                math.sqrt(count * (count - 1))
                / (count - 2)
                * float(np.mean(normalised**3))
            )
        if count > 3:
            fourth = float(np.mean(normalised**4))
            result["kurtosis"] = _finite(
                (count - 1)
                / ((count - 2) * (count - 3))
                * ((count + 1) * (fourth - 3) + 6)
            )
    if lo == hi:
        result["histogram"] = [
            {"label": _label(lo), "count": count, "lower": lo, "upper": hi}
        ]
    else:
        bins = min(16, count)
        edges = np.unique(np.linspace(scaled.min(), scaled.max(), bins + 1))
        counts, edges = np.histogram(scaled, bins=edges)
        result["histogram"] = [
            {
                "label": f"{_label(float(edges[i]) * scale)}–"
                f"{_label(float(edges[i + 1]) * scale)}",
                "count": int(n),
                "lower": _finite(float(edges[i]) * scale),
                "upper": _finite(float(edges[i + 1]) * scale),
            }
            for i, n in enumerate(counts)
        ]
    return result


def _column_summary(series: pl.Series, *, categorical: bool) -> dict[str, Any]:
    numeric = series.dtype.is_numeric() and not categorical
    clean = series.fill_nan(None) if series.dtype.is_float() else series
    missing = clean.null_count()
    result: dict[str, Any] = {
        "name": series.name,
        "dtype": str(series.dtype),
        "kind": "numeric" if numeric else "categorical",
        "rows": len(series),
        "missing": missing,
        "missing_pct": 100 * missing / len(series) if len(series) else 0.0,
        "nonfinite": 0,
        "unique": 0,
        "finite": None,
        **dict.fromkeys(NUMERIC_STATISTICS),
        "histogram": [],
    }
    if series.dtype.is_nested() or series.dtype in (pl.Object, pl.Binary):
        result.update(kind="unsupported", note="This column type is not profiled.")
        result["unique"] = None
        return result
    present = clean.drop_nulls()
    result["unique"] = present.n_unique()
    if series.dtype.is_float():
        result["nonfinite"] = int(present.is_infinite().sum())
    if numeric:
        array = present.cast(pl.Float64).to_numpy()
        result.update(_numeric_stats(array[np.isfinite(array)]))
    elif len(present):
        counts = (
            present.cast(pl.String)
            .value_counts(name="count")
            .rename({series.name: "label"})
            .sort(["count", "label"], descending=[True, False])
        )
        result["histogram"] = counts.head(8).to_dicts()
        if counts.height > 8:
            result["histogram"].append(
                {
                    "label": "Remaining levels",
                    "count": int(counts["count"].slice(8).sum()),
                    "remaining": True,
                }
            )
    return result


def _centred_unit(values: FloatArray) -> FloatArray:
    """Centre before scaling when safe, preserving small represented differences."""
    with np.errstate(over="ignore"):
        shifted = values - values[0]
    if not np.isfinite(shifted).all():
        # Opposite extreme signs can overflow subtraction; scaling first is safe.
        shifted = values / (float(np.max(np.abs(values))) or 1.0)
        shifted -= shifted[0]
    shifted /= float(np.max(np.abs(shifted))) or 1.0
    shifted -= float(shifted.mean())
    spread = float(np.sqrt(np.mean(shifted**2)))
    if spread > 0:
        shifted /= spread
    return shifted


def _stable_pair(left: FloatArray, right: FloatArray) -> float | None:
    """Re-centre on a pair's actual overlap when block subtraction loses precision."""
    valid = np.isfinite(left) & np.isfinite(right)
    x, y = left[valid], right[valid]
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    x, y = _centred_unit(x), _centred_unit(y)
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    return float(np.clip(float(x @ y) / denominator, -1, 1)) if denominator else None


def _correlations(
    frame: pl.DataFrame,
    variables: list[str],
    categorical: set[str],
    sample_rows: int,
) -> dict[str, Any]:
    names = [
        name
        for name in variables
        if name in frame.columns
        and frame.schema[name].is_numeric()
        and name not in categorical
    ]
    count = min(frame.height, sample_rows, CORRELATION_SAMPLE_ROWS)
    result: dict[str, Any] = {
        "method": "Pearson",
        "scope": "training rows",
        "names": names,
        "excluded_names": [name for name in variables if name not in names],
        "sample_rows": count,
        "total_rows": frame.height,
        "sampled": count < frame.height,
        "sample_seed": 0,
        "mode": "matrix" if len(names) <= CORRELATION_MATRIX_COLUMNS else "pairs",
        "pairs": [],
        "valid_pairs": 0,
        "max_pairs": CORRELATION_PAIR_LIMIT,
    }
    size = len(names)
    matrix: list[list[float | None]] = []
    paired_counts: list[list[int]] = []
    if result["mode"] == "matrix":
        matrix = [[None] * size for _ in names]
        paired_counts = [[0] * size for _ in names]
        result.update(matrix=matrix, counts=paired_counts)
    if not count or not names:
        return result
    positions = (
        np.sort(np.random.default_rng(0).choice(frame.height, count, replace=False))
        if count < frame.height
        else None
    )
    values = np.zeros((count, size), dtype=np.float64)
    present = np.zeros((count, size), dtype=np.float64)
    varying = np.zeros(size, dtype=np.bool_)
    for column, name in enumerate(names):
        series = frame[name]
        if positions is not None:
            series = series.gather(positions)
        array = series.cast(pl.Float64).to_numpy()
        valid = np.isfinite(array)
        present[:, column] = valid
        if valid.any():
            observed = array[valid]
            varying[column] = np.any(observed != observed[0])
            values[valid, column] = _centred_unit(observed)
    squares = values * values
    strongest: list[tuple[float, int, int, float, int]] = []
    raw_cache: dict[int, FloatArray] = {}
    epsilon = np.finfo(np.float64).eps
    for start in range(0, size, CORRELATION_BLOCK_COLUMNS):
        stop = min(start + CORRELATION_BLOCK_COLUMNS, size)
        x = values[:, start:stop]
        mask = present[:, start:stop]
        counts = mask.T @ present
        sums_x = x.T @ present
        sums_y = mask.T @ values
        squares_x = squares[:, start:stop].T @ present
        squares_y = mask.T @ squares
        divisor = np.maximum(counts, 1)
        variance_x = squares_x - sums_x * sums_x / divisor
        variance_y = squares_y - sums_y * sums_y / divisor
        covariance = x.T @ values - sums_x * sums_y / divisor
        usable = (
            (counts >= 2)
            & (variance_x > 8 * epsilon * squares_x)
            & (variance_y > 8 * epsilon * squares_y)
        )
        denominator = np.sqrt(np.maximum(variance_x, 0) * np.maximum(variance_y, 0))
        correlations = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=usable,
        ).clip(-1, 1)
        # Near cancellation, n*E[x²] - n*E[x]² can erase overlap variation.
        # Re-read only affected sampled columns; constants never enter this path.
        unstable = (
            (counts >= 2)
            & varying[start:stop, None]
            & varying[None, :]
            & (
                (variance_x <= 64 * math.sqrt(epsilon) * squares_x)
                | (variance_y <= 64 * math.sqrt(epsilon) * squares_y)
            )
        )
        for local_index, right_index in zip(*np.nonzero(unstable), strict=True):
            local, right = int(local_index), int(right_index)
            left = start + local
            if right < left:
                continue
            for column in (left, right):
                if column not in raw_cache:
                    raw = frame[names[column]]
                    if positions is not None:
                        raw = raw.gather(positions)
                    raw_cache[column] = raw.cast(pl.Float64).to_numpy()
            stable = _stable_pair(raw_cache[left], raw_cache[right])
            usable[local, right] = stable is not None
            correlations[local, right] = stable or 0.0
        for local, left in enumerate(range(start, stop)):
            if matrix:
                for right in range(left, size):
                    matrix[left][right] = matrix[right][left] = (
                        float(correlations[local, right])
                        if usable[local, right]
                        else None
                    )
                    paired_counts[left][right] = paired_counts[right][left] = int(
                        round(counts[local, right])
                    )
            for right_index in np.flatnonzero(usable[local, left + 1 :]) + left + 1:
                right = int(right_index)
                score = float(correlations[local, right])
                result["valid_pairs"] += 1
                candidate = (
                    abs(score),
                    -left,
                    -int(right),
                    score,
                    int(round(counts[local, right])),
                )
                if len(strongest) < CORRELATION_PAIR_LIMIT:
                    heapq.heappush(strongest, candidate)
                elif candidate > strongest[0]:
                    heapq.heapreplace(strongest, candidate)
    result["pairs"] = [
        {"left": names[-left], "right": names[-right], "correlation": score, "rows": n}
        for _, left, right, score, n in sorted(strongest, reverse=True)
    ]
    return result


def data_summary(
    frame: pl.DataFrame,
    variables: list[str],
    *,
    categorical: set[str] | None = None,
    correlation_variables: list[str] | None = None,
    sample_rows: int = CORRELATION_SAMPLE_ROWS,
) -> dict[str, Any]:
    """Summarise selected columns without changing data or fitting a model.

    Pass the prepared training frame. Descriptive statistics and histograms use
    every supplied row. Missing means null or NaN; infinities are counted
    separately and excluded from numeric moments, histograms and correlations.
    Standard deviation uses n-1; skewness is the corrected Fisher-Pearson value;
    kurtosis is unbiased excess kurtosis (normal = 0). Constant or insufficient
    samples have undefined shape statistics. Correlations are unweighted,
    pairwise-complete Pearson values on a reproducible sample of at most 10,000
    rows. Numeric columns declared categorical are never assigned numeric codes.
    """
    if (
        isinstance(sample_rows, bool)
        or not isinstance(sample_rows, int)
        or sample_rows < 1
    ):
        raise ValueError("sample_rows must be a positive integer.")
    categorical = categorical or set()
    names = list(dict.fromkeys(variables))
    summaries = [
        _column_summary(frame[name], categorical=name in categorical)
        for name in names
        if name in frame.columns
    ]
    requested = list(
        dict.fromkeys(names if correlation_variables is None else correlation_variables)
    )
    return {
        "rows": frame.height,
        "variables": summaries,
        "unavailable_variables": [name for name in names if name not in frame.columns],
        "columns_with_missing": sum(item["missing"] > 0 for item in summaries),
        "missing_cells": sum(item["missing"] for item in summaries),
        "correlations": _correlations(frame, requested, categorical, sample_rows),
    }
