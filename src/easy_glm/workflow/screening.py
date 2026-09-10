"""Bounded, training-only predictor screening without fitting any models.

Associations are review prompts, not feature selection decisions. Numeric pairs
use pairwise-complete weighted Pearson correlation. Categorical comparisons use
bias-corrected Cramer's V; mixed pairs use out-of-fold group predictions.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from easy_glm.workflow.prep import prepare
from easy_glm.workflow.project import Project

FloatArray = NDArray[np.float64]
MIN_ROWS = 30
MIN_EFFECTIVE_ROWS = 20
MAX_LEVELS = 32
PAIR_LIMIT = 2000
BLOCK_SIZE = 96


def _effective_rows(weights: FloatArray) -> float:
    if not np.any(weights > 0):
        return 0.0
    scaled = weights / float(weights.max())
    return float(scaled.sum() ** 2 / (scaled @ scaled))


@dataclass
class _Column:
    name: str
    values: FloatArray | None = None
    codes: NDArray[np.int64] | None = None


def _normalise(values: FloatArray) -> FloatArray:
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if finite.any():
        scale = float(np.max(np.abs(values[finite]))) or 1.0
        centred = values[finite] / scale
        centred -= np.mean(centred)
        spread = float(np.std(centred)) or 1.0
        out[finite] = centred / spread
    return out


def _pearson(
    x: FloatArray, y: FloatArray, weights: FloatArray
) -> tuple[float | None, int]:
    valid = np.isfinite(x) & np.isfinite(y) & (weights > 0)
    count = int(valid.sum())
    if count < MIN_ROWS:
        return None, count
    w = weights[valid] / float(weights[valid].max())
    mass = float(w.sum())
    if mass * mass / float(w @ w) < MIN_EFFECTIVE_ROWS:
        return None, count
    a, b = _normalise(x[valid]), _normalise(y[valid])
    a -= float(w @ a) / mass
    b -= float(w @ b) / mass
    variance_a, variance_b = float(w @ (a * a)), float(w @ (b * b))
    if variance_a <= 1e-12 or variance_b <= 1e-12:
        return None, count
    score = float(w @ (a * b)) / np.sqrt(variance_a * variance_b)
    return float(np.clip(score, -1, 1)), count


def _group_association(
    codes: NDArray[np.int64],
    values: FloatArray,
    weights: FloatArray,
    folds: NDArray[np.int64],
) -> tuple[float | None, int]:
    """Square root of positive out-of-fold R-squared, with rare-level shrinkage."""
    valid = (codes >= 0) & np.isfinite(values) & (weights > 0)
    count = int(valid.sum())
    if count < MIN_ROWS:
        return None, count
    c, y, w, fold = (
        codes[valid],
        _normalise(values[valid]),
        weights[valid],
        folds[valid],
    )
    w = w / float(w.max())
    if w.sum() ** 2 / (w @ w) < MIN_EFFECTIVE_ROWS or np.unique(c).size < 2:
        return None, count
    predictions = np.zeros(count, dtype=np.float64)
    levels = int(c.max()) + 1
    for side in (0, 1):
        train, test = fold != side, fold == side
        if train.sum() < 10 or not test.any():
            return None, count
        base = float(np.average(y[train], weights=w[train]))
        mass = np.bincount(c[train], weights=w[train], minlength=levels)
        number = np.bincount(c[train], minlength=levels)
        sums = np.bincount(c[train], weights=w[train] * y[train], minlength=levels)
        means = np.full(levels, base, dtype=np.float64)
        supported = (number >= 5) & (mass > 0)
        means[supported] = sums[supported] / mass[supported]
        predictions[test] = means[c[test]]
    mean = float(np.average(y, weights=w))
    variance = float(w @ ((y - mean) ** 2))
    if variance <= 1e-12:
        return None, count
    score = 1.0 - float(w @ ((y - predictions) ** 2)) / variance
    return float(np.sqrt(np.clip(score, 0, 1))), count


def _cramers_v(
    a: NDArray[np.int64], b: NDArray[np.int64], weights: FloatArray
) -> tuple[float | None, int]:
    # These are row counts, deliberately not arbitrary exposure sums.
    valid = (a >= 0) & (b >= 0) & (weights > 0)
    n = int(valid.sum())
    if n < MIN_ROWS:
        return None, n
    ac, bc = a[valid], b[valid]
    width = int(bc.max()) + 1
    table = np.bincount(ac * width + bc, minlength=(int(ac.max()) + 1) * width).reshape(
        -1, width
    )
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    r, k = table.shape
    if min(r, k) < 2 or n < 5 * max(r, k):
        return None, n
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    phi = float(np.sum((table - expected) ** 2 / expected)) / n
    corrected = max(0.0, phi - (k - 1) * (r - 1) / (n - 1))
    dimension = min(k - 1 - (k - 1) ** 2 / (n - 1), r - 1 - (r - 1) ** 2 / (n - 1))
    return (
        float(np.sqrt(np.clip(corrected / dimension, 0, 1))) if dimension > 0 else None
    ), n


def _numeric_pairs(
    columns: list[_Column],
    weights: FloatArray,
    threshold: float,
    record: Callable[[str, str, float, str, int], None],
    tick: Callable[[int, int], None],
) -> None:
    if len(columns) < 2:
        return
    original_weights = weights
    weights = weights / float(weights.max())
    x = np.column_stack(
        [
            _normalise(np.where(original_weights > 0, c.values, np.nan))
            for c in columns
            if c.values is not None
        ]
    )
    valid = (np.isfinite(x) & (original_weights[:, None] > 0)).astype(np.float64)
    x = np.nan_to_num(x, copy=False)
    x *= valid
    complete = bool(np.all(valid == 1))
    p = len(columns)
    total = p * (p - 1) // 2
    done = 0
    if complete:
        x -= np.average(x, axis=0, weights=weights)
        norm = np.sqrt(np.sum(weights[:, None] * x * x, axis=0))
        norm[norm <= 1e-12] = np.inf
        x = x * np.sqrt(weights[:, None]) / norm
    for i in range(0, p, BLOCK_SIZE):
        end_i = min(i + BLOCK_SIZE, p)
        for j in range(i, p, BLOCK_SIZE):
            tick(done, total)
            end_j = min(j + BLOCK_SIZE, p)
            a, b = x[:, i:end_i], x[:, j:end_j]
            if complete:
                corr = a.T @ b
                counts = np.full(corr.shape, len(weights), dtype=np.float64)
                support = np.full(
                    corr.shape,
                    weights.sum() ** 2 / (weights @ weights) >= MIN_EFFECTIVE_ROWS,
                )
            else:
                va, vb = valid[:, i:end_i], valid[:, j:end_j]
                mass = (va * weights[:, None]).T @ vb
                safe_mass = np.maximum(mass, 1e-100)
                sa, sb = (a * weights[:, None]).T @ vb, va.T @ (b * weights[:, None])
                covariance = (a * weights[:, None]).T @ b - sa * sb / safe_mass
                vara = ((a * a) * weights[:, None]).T @ vb - sa * sa / safe_mass
                varb = va.T @ ((b * b) * weights[:, None]) - sb * sb / safe_mass
                denominator = np.sqrt(np.maximum(vara, 0) * np.maximum(varb, 0))
                corr = np.divide(
                    covariance,
                    denominator,
                    out=np.zeros_like(covariance),
                    where=denominator > 1e-12,
                )
                counts = va.T @ vb
                mass_squared = (va * (weights * weights)[:, None]).T @ vb
                support = (mass * mass >= MIN_EFFECTIVE_ROWS * mass_squared) & (
                    denominator > 1e-12
                )
            hits = (np.abs(corr) >= threshold) & support & (counts >= MIN_ROWS)
            if not complete:
                # A huge weight outside a pair's overlap must not erase its
                # valid smaller weights. Confirm unusually small-mass pairs
                # with locally scaled weights rather than unstable moments.
                for row, col in zip(
                    *np.nonzero((mass < 1e-6) & (counts >= MIN_ROWS)), strict=True
                ):
                    first, second = i + int(row), j + int(col)
                    if first >= second:
                        continue
                    av, bv = columns[first].values, columns[second].values
                    assert av is not None and bv is not None
                    score, count = _pearson(av, bv, original_weights)
                    hits[row, col] = False
                    if score is not None and abs(score) >= threshold:
                        record(
                            columns[first].name,
                            columns[second].name,
                            score,
                            "Weighted Pearson correlation",
                            count,
                        )
            for row, col in zip(*np.nonzero(hits), strict=True):
                first, second = i + int(row), j + int(col)
                if first < second:
                    record(
                        columns[first].name,
                        columns[second].name,
                        float(np.clip(corr[row, col], -1, 1)),
                        "Weighted Pearson correlation",
                        int(counts[row, col]),
                    )
            done += (
                (end_i - i) * (end_j - j)
                if j != i
                else (end_i - i) * (end_i - i - 1) // 2
            )
    tick(total, total)


def screen_variables(
    project: Project,
    raw: pl.DataFrame,
    *,
    sample_rows: int = 10_000,
    seed: int = 42,
    missing_threshold: float = 0.7,
    correlation_threshold: float = 0.95,
    leakage_threshold: float = 0.9,
    divide_target_by_weight: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Screen selected predictors; never mutate the project or fit a model.

    Missingness uses sampled training rows. Associations exclude nonpositive or
    invalid weights, require 30 observations, and numeric/response comparisons
    require 20 effective observations. Category and mixed pairs use a 2,000-row
    candidate pass followed by confirmation on the full training sample.
    """
    if not 500 <= sample_rows <= 20_000:
        raise ValueError("Use a sample of 500 to 20,000 training rows.")
    if any(
        not 0 <= value <= 1
        for value in (missing_threshold, correlation_threshold, leakage_threshold)
    ):
        raise ValueError("Screening thresholds must be between 0 and 1.")

    def update(phase: str, completed: int = 0, total: int = 0) -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("Predictor check cancelled.")
        if progress:
            progress(
                {
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "message": phase,
                }
            )

    update("Preparing training sample")
    if not project.target:
        raise ValueError("Choose a target on the Variables page first.")
    prepared = prepare(project, raw)
    indices = np.flatnonzero(
        prepared[project.split_column].fill_null(0).to_numpy() == 1
    )
    training_rows = len(indices)
    if training_rows < MIN_ROWS:
        raise ValueError(
            "At least 30 training rows are needed. Check the train/holdout split."
        )
    rng = np.random.default_rng(seed)
    if training_rows > sample_rows:
        indices = np.sort(rng.choice(indices, sample_rows, replace=False))
    names = [name for name in project.predictors if name != project.split_column]
    if not names:
        raise ValueError("Select at least one predictor first.")
    missing_columns = set(names + [project.target]) - set(prepared.columns)
    if missing_columns:
        raise ValueError("Columns not found: " + ", ".join(sorted(missing_columns)))
    frame = prepared.select(
        list(
            dict.fromkeys(
                names + [project.target] + ([project.weight] if project.weight else [])
            )
        )
    )[indices]
    n = frame.height
    del prepared
    weights = np.ones(n, dtype=np.float64)
    if project.weight:
        weights = frame[project.weight].cast(pl.Float64, strict=False).to_numpy().copy()
    good_weights = np.isfinite(weights) & (weights > 0)
    weights[~good_weights] = 0
    if good_weights.sum() < MIN_ROWS:
        raise ValueError("At least 30 training rows need a positive, finite weight.")
    target = frame[project.target].cast(pl.Float64, strict=False).to_numpy()
    response = target.copy()
    divide = bool(divide_target_by_weight and project.weight)
    if divide:
        assert project.weight is not None
        raw_weight = frame[project.weight].cast(pl.Float64, strict=False).to_numpy()
        response = np.divide(
            target, raw_weight, out=np.full(n, np.nan), where=good_weights
        )
    good_target = np.isfinite(response) & np.isfinite(target) & good_weights
    if good_target.sum() < MIN_ROWS:
        raise ValueError(
            "At least 30 training rows need a finite numeric target and valid weight."
        )
    targets = [(target, "target")]
    if divide:
        targets.append((response, "target / weight"))
    folds = np.arange(n, dtype=np.int64) % 2
    rng.shuffle(folds)
    result: dict[str, Any] = {
        "target": project.target,
        "weight": project.weight,
        "divide_target_by_weight": divide,
        "rows": n,
        "training_rows": training_rows,
        "sampled": n < training_rows,
        "predictor_count": len(names),
        "excluded_target_rows": int(n - good_target.sum()),
        "leakage": [],
        "correlated": [],
        "missing": [],
        "unsupported": [],
        "columns": [],
        "notes": [
            "High association is a reason to review a predictor, not proof of leakage.",
            "Checks use training rows only; missingness is a share of rows, not exposure.",
        ],
    }
    if not np.all(good_weights):
        result["notes"].append(
            f"Associations exclude {int((~good_weights).sum()):,} rows with missing, non-finite or nonpositive weights."
        )
    columns: list[_Column] = []
    numeric: list[_Column] = []
    for number, name in enumerate(names):
        update("Checking missingness and target association", number, len(names))
        series = frame[name]
        supported_type = (
            series.dtype.is_numeric()
            or series.dtype in (pl.String, pl.Categorical, pl.Boolean)
            or isinstance(series.dtype, pl.Enum)
        )
        absent = series.is_null().to_numpy()
        if series.dtype.is_float():
            absent = absent | series.is_nan().fill_null(False).to_numpy()
        share, observed = float(absent.mean()), int((~absent).sum())
        result["columns"].append(
            {"variable": name, "missing_share": share, "observations": observed}
        )
        if share >= missing_threshold:
            result["missing"].append(
                {"variable": name, "missing_share": share, "observations": observed}
            )
        if not supported_type:
            result["unsupported"].append(
                {
                    "variable": name,
                    "reason": "Unsupported type; associations were not checked.",
                }
            )
            continue
        values: FloatArray | None = None
        codes: NDArray[np.int64] | None = None
        reason = ""
        if series.dtype.is_numeric():
            values = series.cast(pl.Float64).to_numpy()
            valid_values = np.isfinite(values) & good_weights
            unique = np.unique(values[valid_values])
            if int(valid_values.sum()) < MIN_ROWS:
                reason = "Too few observed values for association checks."
            elif _effective_rows(weights[valid_values]) < MIN_EFFECTIVE_ROWS:
                reason = "Too few effective observations; weight is concentrated in very few rows."
            elif len(unique) < 2:
                reason = "Constant on observed training rows."
            else:
                column = _Column(name, values=values)
                columns.append(column)
                numeric.append(column)
                edges = np.unique(
                    np.quantile(values[valid_values], np.linspace(0, 1, 11)[1:-1])
                )
                codes = np.where(
                    valid_values, np.searchsorted(edges, values, side="right"), -1
                ).astype(np.int64)
        else:
            codes = (
                series.cast(pl.String)
                .cast(pl.Categorical)
                .to_physical()
                .cast(pl.Int64)
                .fill_null(-1)
                .to_numpy()
            )
            # The global string cache may supply sparse physical codes.
            present = (codes >= 0) & good_weights
            _, inverse = np.unique(codes[present], return_inverse=True)
            codes = codes.copy()
            codes[~present] = -1
            codes[present] = inverse
            levels = int(np.unique(codes[present]).size)
            if int((present & good_weights).sum()) < MIN_ROWS:
                reason = "Too few observed values for association checks."
            elif levels < 2:
                reason = "Constant on observed training rows."
            elif levels > MAX_LEVELS or int(present.sum()) < 5 * levels:
                reason = f"Too many or too sparse categories for a quick association check (limit {MAX_LEVELS} levels)."
            else:
                columns.append(_Column(name, codes=codes))
        if reason:
            result["unsupported"].append({"variable": name, "reason": reason})
        associations: list[tuple[float, int, str, str]] = []
        for outcome, label in targets:
            if values is not None and not reason:
                score, count = _pearson(values, outcome, weights)
                if score is not None:
                    associations.append(
                        (
                            abs(score),
                            count,
                            "Weighted Pearson correlation",
                            f"Strong association with {label}.",
                        )
                    )
            if codes is not None and not reason:
                score, count = _group_association(codes, outcome, weights, folds)
                if score is not None:
                    associations.append(
                        (
                            score,
                            count,
                            "Cross-validated group association",
                            f"Groups closely predict {label}.",
                        )
                    )
            if absent.any() and not absent.all():
                score, count = _pearson(absent.astype(np.float64), outcome, weights)
                if score is not None:
                    associations.append(
                        (
                            abs(score),
                            count,
                            "Missingness correlation",
                            f"Missing values strongly track {label}.",
                        )
                    )
        if associations:
            score, count, method, explanation = max(
                associations, key=lambda item: item[0]
            )
            if score >= leakage_threshold:
                result["leakage"].append(
                    {
                        "variable": name,
                        "association": score,
                        "method": method,
                        "reason": explanation,
                        "observations": count,
                    }
                )
        elif not reason:
            result["unsupported"].append(
                {
                    "variable": name,
                    "reason": "Target association could not be checked: too little shared data, weight support or target variation.",
                }
            )
    pairs: list[tuple[float, int, dict[str, Any]]] = []
    pair_count = 0

    def record(first: str, second: str, score: float, method: str, count: int) -> None:
        nonlocal pair_count
        pair_count += 1
        row = {
            "first": first,
            "second": second,
            "association": score,
            "method": method,
            "observations": count,
        }
        item = (abs(score), pair_count, row)
        if len(pairs) < PAIR_LIMIT:
            heapq.heappush(pairs, item)
        elif item[:2] > pairs[0][:2]:
            heapq.heapreplace(pairs, item)

    _numeric_pairs(
        numeric,
        weights,
        correlation_threshold,
        record,
        lambda done, total: update("Checking numeric predictor pairs", done, total),
    )
    categorical = [c for c in columns if c.codes is not None]
    pilot = np.sort(rng.choice(n, min(n, 2000), replace=False))
    category_pairs = len(categorical) * (len(categorical) - 1) // 2 + len(
        categorical
    ) * len(numeric)
    done = 0
    for i, a in enumerate(categorical):
        assert a.codes is not None
        for b in categorical[i + 1 :] + numeric:
            if done % 32 == 0:
                update(
                    "Checking category and mixed predictor pairs", done, category_pairs
                )
            done += 1
            if b.codes is not None:
                score, _ = _cramers_v(a.codes[pilot], b.codes[pilot], weights[pilot])
                method = "Category association (Cramér’s V)"
            else:
                assert b.values is not None
                score, _ = _group_association(
                    a.codes[pilot], b.values[pilot], weights[pilot], folds[pilot]
                )
                method = "Cross-validated group association"
            # A generous candidate threshold reduces sampling misses. The full
            # sample always confirms a displayed flag; the pilot is not evidence.
            if score is None or score < max(0, correlation_threshold - 0.15):
                continue
            if b.codes is not None:
                score, count = _cramers_v(a.codes, b.codes, weights)
            else:
                assert b.values is not None
                score, count = _group_association(a.codes, b.values, weights, folds)
            if score is not None and score >= correlation_threshold:
                record(a.name, b.name, score, method, count)
    result["correlated"] = [item[2] for item in sorted(pairs, reverse=True)]
    result["correlated_count"] = pair_count
    result["leakage"].sort(key=lambda row: (-row["association"], row["variable"]))
    result["missing"].sort(key=lambda row: (-row["missing_share"], row["variable"]))
    if pair_count > PAIR_LIMIT:
        result["notes"].append(
            f"Showing the strongest {PAIR_LIMIT:,} of {pair_count:,} flagged pairs. Remove selected predictors and check again to see further pairs."
        )
    if category_pairs:
        result["notes"].append(
            f"Category and mixed pairs use a {len(pilot):,}-row candidate pass, confirmed on this sample; some relationships may be missed. Group association does not imply interchangeable predictors."
        )
    if result["unsupported"]:
        result["notes"].append(
            "Some predictors could not be checked for association; see coverage details."
        )
    update("Complete", len(names), len(names))
    return result
