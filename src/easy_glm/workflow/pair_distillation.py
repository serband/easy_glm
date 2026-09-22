"""Train a two-column CatBoost teacher and distil its means into pair cells.

This module deliberately has no dependency on the deployed scoring model.  Its
output is a table of positive multipliers; CatBoost is only used during training.
All means passed to the distiller are complete response-scale means, including
the upstream deployed prefix and any external offset exactly once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vector(
    name: str, value: ArrayLike, length: int | None = None
) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or (length is not None and len(result) != length):
        raise ValueError(f"{name} must be a one-dimensional vector of matching length")
    return result


def _positive_means(
    name: str, value: ArrayLike, length: int | None = None
) -> NDArray[np.float64]:
    result = _vector(name, value, length)
    if not np.all(np.isfinite(result) & (result > 0)):
        raise ValueError(f"{name} must contain finite, strictly positive means")
    return result


def _weights(value: ArrayLike | None, length: int) -> NDArray[np.float64]:
    result = (
        np.ones(length, dtype=np.float64)
        if value is None
        else _vector("sample_weight", value, length)
    )
    if not np.all(np.isfinite(result) & (result >= 0)):
        raise ValueError("sample_weight must be finite and nonnegative")
    return result


def _power(family: str, tweedie_power: float | None) -> float:
    if family == "poisson":
        if tweedie_power is not None:
            raise ValueError("tweedie_power is only valid for the Tweedie family")
        return 1.0
    if (
        family == "tweedie"
        and tweedie_power is not None
        and math.isfinite(tweedie_power)
        and 1 < tweedie_power < 2
    ):
        return float(tweedie_power)
    raise ValueError(
        "Pair training supports Poisson/log or Tweedie/log with 1 < power < 2"
    )


def teacher_mean_from_raw(
    raw_correction: ArrayLike,
    baseline_mean: ArrayLike,
    *,
    target_scale: float = 1.0,
) -> NDArray[np.float64]:
    """Convert a correction-only CatBoost raw prediction to original-unit means.

    ``baseline_mean`` is already in original response units.  During training
    both target and baseline may have been divided by ``target_scale``; the
    factor cancels on inverse transformation.  Do not pass predictions made on
    a Pool carrying a baseline, because CatBoost adds that baseline to its raw
    prediction.  Predict on the raw two-column features instead.
    """
    if not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("target_scale must be finite and strictly positive")
    baseline = _positive_means("baseline_mean", baseline_mean)
    correction = _vector("raw_correction", raw_correction, len(baseline))
    if not np.all(np.isfinite(correction)):
        raise ValueError("raw_correction must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mean = np.exp(np.log(baseline) + correction)
    return _positive_means("teacher_mean", mean, len(baseline))


@dataclass(frozen=True)
class DistilledPairCells:
    """Row-major pair cells, including neutral unsupported cells."""

    relativities: NDArray[np.float64]
    row_count: NDArray[np.int64]
    fitting_weight: NDArray[np.float64]
    weight_share: NDArray[np.float64]
    fallback_reason: tuple[str | None, ...]


def distill_pair_cells(
    cell_ids: ArrayLike,
    baseline_mean: ArrayLike,
    teacher_mean: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    family: str = "poisson",
    tweedie_power: float | None = None,
    n_cells: int,
    min_weight_share: float = 0.0,
) -> DistilledPairCells:
    """Choose the loss-optimal constant multiplier for each represented cell.

    ``cell_ids`` are zero-based row-major IDs (axis A index * axis B size +
    axis B index).  Weighted Tweedie minimisation gives
    sum(w*t*b**(1-p)) / sum(w*b**(2-p)); Poisson is the p=1 special case.
    Summands are accumulated in log space to handle extreme positive means.
    Empty and below-threshold cells remain neutral, with their reason recorded.
    Zero-weight rows are ignored for support and loss, but all supplied means
    must still be valid so a broken teacher cannot look like a successful fit.
    """
    power = _power(family, tweedie_power)
    if not isinstance(n_cells, int) or n_cells < 1:
        raise ValueError("n_cells must be a positive integer")
    if not math.isfinite(min_weight_share) or not 0 <= min_weight_share < 1:
        raise ValueError("min_weight_share must be in [0, 1)")
    baseline = _positive_means("baseline_mean", baseline_mean)
    teacher = _positive_means("teacher_mean", teacher_mean, len(baseline))
    weight = _weights(sample_weight, len(baseline))
    raw_ids = np.asarray(cell_ids)
    if (
        raw_ids.ndim != 1
        or len(raw_ids) != len(baseline)
        or not np.issubdtype(raw_ids.dtype, np.integer)
    ):
        raise ValueError("cell_ids must be a matching vector of integer IDs")
    if np.any((raw_ids < 0) | (raw_ids >= n_cells)):
        raise ValueError("cell_ids contain an ID outside the pair grid")
    ids = raw_ids.astype(np.int64, copy=False)
    positive = weight > 0
    if not np.any(positive):
        raise ValueError("At least one row must have positive fitting weight")
    active_ids = ids[positive]
    active_weight = weight[positive]
    row_count = np.bincount(active_ids, minlength=n_cells).astype(np.int64)
    fitting_weight = np.bincount(
        active_ids, weights=active_weight, minlength=n_cells
    ).astype(np.float64)
    total_weight = float(np.sum(active_weight, dtype=np.float64))
    if not math.isfinite(total_weight) or not np.all(np.isfinite(fitting_weight)):
        raise ValueError("Fitting-weight totals must be finite")
    share = fitting_weight / total_weight
    relativities = np.ones(n_cells, dtype=np.float64)
    reasons: list[str | None] = [None] * n_cells

    log_weight = np.log(active_weight)
    log_baseline = np.log(baseline[positive])
    log_teacher = np.log(teacher[positive])
    log_num = log_weight + log_teacher + (1 - power) * log_baseline
    log_den = log_weight + (2 - power) * log_baseline
    order = np.argsort(active_ids, kind="stable")
    sorted_ids = active_ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
    ends = np.r_[starts[1:], len(sorted_ids)]
    for start, end in zip(starts, ends, strict=True):
        cell = int(sorted_ids[start])
        if share[cell] < min_weight_share:
            reasons[cell] = "insufficient_support"
            continue
        numerator = float(np.logaddexp.reduce(log_num[order[start:end]]))
        denominator = float(np.logaddexp.reduce(log_den[order[start:end]]))
        if not math.isfinite(denominator):
            raise ValueError(f"Cell {cell} has no positive finite denominator")
        with np.errstate(over="ignore", under="ignore"):
            relativity = float(np.exp(numerator - denominator))
        if not math.isfinite(relativity) or relativity <= 0:
            raise ValueError(f"Cell {cell} has no representable positive relativity")
        relativities[cell] = relativity
    for empty_cell in np.flatnonzero(row_count == 0):
        reasons[int(empty_cell)] = "empty"
    return DistilledPairCells(
        relativities, row_count, fitting_weight, share, tuple(reasons)
    )


@dataclass(frozen=True)
class CatBoostPairTeacher:
    """Training-only teacher; deployed scorers must use distilled tables."""

    model: Any
    family: str
    tweedie_power: float | None
    target_scale: float

    def predict_mean(
        self, two_raw_features: Any, baseline_mean: ArrayLike
    ) -> NDArray[np.float64]:
        """Predict original-unit means without adding CatBoost's baseline twice."""
        correction = self.model.predict(
            two_raw_features, prediction_type="RawFormulaVal"
        )
        return teacher_mean_from_raw(
            correction, baseline_mean, target_scale=self.target_scale
        )


def fit_catboost_pair_raw(
    two_raw_features: Any,
    target: ArrayLike,
    baseline_mean: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    family: str = "poisson",
    tweedie_power: float | None = None,
    target_scale: float = 1.0,
    iterations: int = 120,
    depth: int = 3,
    learning_rate: float = 0.05,
    l2_leaf_reg: float = 3.0,
    thread_count: int = 2,
    seed: int = 0,
    cat_features: Sequence[int] | None = None,
) -> CatBoostPairTeacher:
    """Fit CatBoost on exactly two raw columns against a raw link-scale baseline.

    The target and baseline are scaled together, then the model is trained with
    ``log(baseline / target_scale)`` as its Pool baseline.  The returned teacher
    predicts on feature columns without a Pool baseline; its raw output is only
    the correction.  Validation and early stopping belong to a fold-local caller.
    """
    power = _power(family, tweedie_power)
    if not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("target_scale must be finite and strictly positive")
    baseline = _positive_means("baseline_mean", baseline_mean)
    y = _vector("target", target, len(baseline))
    if not np.all(np.isfinite(y) & (y >= 0)):
        raise ValueError("target must be finite and nonnegative")
    weight = _weights(sample_weight, len(y))
    positive = weight > 0
    if not np.any(positive):
        raise ValueError("At least one row must have positive fitting weight")
    if (
        not isinstance(iterations, int)
        or iterations < 1
        or not isinstance(depth, int)
        or depth < 1
    ):
        raise ValueError("iterations and depth must be positive integers")
    if not isinstance(thread_count, int) or thread_count < 1:
        raise ValueError("thread_count must be a positive integer")
    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not math.isfinite(l2_leaf_reg)
        or l2_leaf_reg < 0
    ):
        raise ValueError("learning_rate must be positive and l2_leaf_reg nonnegative")
    x = np.asarray(two_raw_features)
    if x.ndim != 2 or x.shape != (len(y), 2):
        raise ValueError(
            "two_raw_features must contain exactly two columns and one row per target"
        )
    if cat_features is not None and any(index not in (0, 1) for index in cat_features):
        raise ValueError("cat_features must refer only to the two raw columns")
    scaled_y = y[positive] / target_scale
    scaled_baseline = baseline[positive] / target_scale
    if not np.all(np.isfinite(scaled_y) & (scaled_y >= 0)) or not np.all(
        np.isfinite(scaled_baseline) & (scaled_baseline > 0)
    ):
        raise ValueError("target_scale makes target or baseline nonrepresentable")
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise ImportError(
            "CatBoost is required to train pair corrections. Install with "
            "pip install 'easy-glm[pairs]'. Saved table-only scoring does not require CatBoost"
        ) from exc
    loss = "Poisson" if power == 1 else f"Tweedie:variance_power={power:.16g}"
    pool = Pool(
        x[positive],
        label=scaled_y,
        weight=weight[positive],
        baseline=np.log(scaled_baseline),
        cat_features=list(cat_features or ()),
    )
    model = CatBoostRegressor(
        loss_function=loss,
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        thread_count=thread_count,
        random_seed=seed,
        allow_const_label=True,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(pool)
    return CatBoostPairTeacher(model, family, tweedie_power, target_scale)
