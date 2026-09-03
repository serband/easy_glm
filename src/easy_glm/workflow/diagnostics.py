"""Model diagnostics: deviance, lift / Gini, double lift, A/E by any variable,
regularisation path and the residual factor search.

Conventions
-----------
All functions work on *totals* per row so that count and rate targets are
handled uniformly (see :func:`totals`):

* ``actual_total``   — observed amount for the row (claims, cost, ...)
* ``expected_total`` — model prediction for the row on the same scale
* ``weight``         — exposure (or 1)

A/E = ``sum(actual_total) / sum(expected_total)``; rates divide by weight.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from easy_glm.core.design import NUMERIC_DTYPES, quantile_knots
from easy_glm.core.excel import rate_model_tables
from easy_glm.core.fit import GLMFit, TwoStageFit
from easy_glm.engine.models import NULL_LABEL, FromToRow, level_label

from .explore import band_expr
from .project import ModelConfig


# --------------------------------------------------------------------------
# scale helpers
# --------------------------------------------------------------------------
def unit_values(df: pl.DataFrame, cfg: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """``(y_per_unit, weight)`` as the GLM saw them."""
    y = df[cfg.target].cast(pl.Float64).to_numpy()
    w = df[cfg.weight].cast(pl.Float64).to_numpy() if cfg.weight else np.ones(df.height)
    if cfg.divide_target_by_weight:
        y = y / w
    return y, w


def totals(
    df: pl.DataFrame, cfg: ModelConfig, pred_unit: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(actual_total, expected_total, weight)`` for ``df`` given per-unit predictions."""
    y = df[cfg.target].cast(pl.Float64).to_numpy()
    pred_unit = np.asarray(pred_unit, dtype=float)
    if cfg.weight:
        w = df[cfg.weight].cast(pl.Float64).to_numpy()
        actual = y if cfg.divide_target_by_weight else y * w
        expected = pred_unit * w
    else:
        w = np.ones(df.height)
        actual, expected = y, pred_unit
    return actual, expected, w


# --------------------------------------------------------------------------
# deviance
# --------------------------------------------------------------------------
def deviance_stats(
    family: Any, y_unit: np.ndarray, mu_unit: np.ndarray, weight: np.ndarray | None
) -> dict[str, float]:
    """Deviance, null deviance and the share explained (``1 - D / D0``)."""
    w = None if weight is None else np.asarray(weight, dtype=float)
    dev = float(family.deviance(y_unit, mu_unit, sample_weight=w))
    mu0 = np.full_like(y_unit, np.average(y_unit, weights=w), dtype=float)
    null = float(family.deviance(y_unit, mu0, sample_weight=w))
    n = float(len(y_unit)) if w is None else float(w.sum())
    return {
        "deviance": dev,
        "null_deviance": null,
        "deviance_explained": (1.0 - dev / null) if null > 0 else float("nan"),
        "mean_deviance": dev / n if n else float("nan"),
    }


# --------------------------------------------------------------------------
# lift / gini / double lift
# --------------------------------------------------------------------------
def _weighted_bins(order: np.ndarray, weight: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin index per row: equal-weight bins along ``order`` (row indices)."""
    cum = np.cumsum(weight[order])
    total = cum[-1] if cum.size else 1.0
    bin_of_sorted = np.minimum(
        (cum - weight[order] / 2) / total * n_bins, n_bins - 1
    ).astype(int)
    bins = np.empty(len(order), dtype=int)
    bins[order] = bin_of_sorted
    return bins


def lift_table(
    actual_total: np.ndarray,
    expected_total: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    n_bins: int = 10,
) -> pl.DataFrame:
    """Equal-exposure bins ordered by predicted rate (lowest first)."""
    a = np.asarray(actual_total, float)
    e = np.asarray(expected_total, float)
    w = np.ones_like(a) if weight is None else np.asarray(weight, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(w > 0, e / w, 0.0)
    order = np.argsort(rate, kind="stable")
    bins = _weighted_bins(order, w, n_bins)
    frame = pl.DataFrame({"bin": bins, "actual": a, "expected": e, "exposure": w})
    out = (
        frame.group_by("bin")
        .agg(pl.col("exposure").sum(), pl.col("actual").sum(), pl.col("expected").sum())
        .sort("bin")
        .with_columns(
            (pl.col("actual") / pl.col("expected")).alias("ae"),
            (pl.col("actual") / pl.col("exposure")).alias("actual_rate"),
            (pl.col("expected") / pl.col("exposure")).alias("expected_rate"),
        )
    )
    total_a = out["actual"].sum() or 1.0
    total_w = out["exposure"].sum() or 1.0
    return out.with_columns(
        (pl.col("exposure").cum_sum() / total_w).alias("cum_exposure_share"),
        (pl.col("actual").cum_sum() / total_a).alias("cum_actual_share"),
        (pl.col("bin") + 1).alias("bin"),
    )


def _quantise(score: np.ndarray, rel_tol: float = 1e-12) -> np.ndarray:
    """Round ``score`` to ``rel_tol`` of its largest value so that rows whose
    scores differ only by floating-point noise (identical rating cells scored
    through ``e / w``) are treated as tied."""
    scale = float(np.max(np.abs(score))) if score.size else 0.0
    if not np.isfinite(scale) or scale == 0.0:
        return score
    decimals = max(0, int(-np.floor(np.log10(scale * rel_tol))))
    return np.round(score, decimals)


def gini(
    actual_total: np.ndarray,
    expected_total: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    normalize: bool = True,
) -> float:
    """Exposure-weighted Gini of the ordering by predicted rate; ``normalize``
    divides by the Gini of the perfect ordering (by actual rate).

    Ties are handled deterministically: rows with (numerically) equal scores are
    pooled, i.e. the Lorenz curve is linear across a tied group, so the result
    does not depend on row order or on ``e / w`` rounding noise.
    """
    a = np.asarray(actual_total, float)
    e = np.asarray(expected_total, float)
    w = np.ones_like(a) if weight is None else np.asarray(weight, float)
    if a.sum() <= 0 or w.sum() <= 0:
        return float("nan")

    def _g(score: np.ndarray) -> float:
        q = _quantise(score)
        # pool tied scores; order groups from highest score to lowest
        uniq, inverse = np.unique(q, return_inverse=True)
        gw = np.bincount(inverse, weights=w)[::-1]
        ga = np.bincount(inverse, weights=a)[::-1]
        cum_w = np.concatenate([[0.0], np.cumsum(gw) / w.sum()])
        cum_a = np.concatenate([[0.0], np.cumsum(ga) / a.sum()])
        area = np.trapezoid(cum_a, cum_w)
        return float(2.0 * area - 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        g = _g(np.where(w > 0, e / w, 0.0))
        if not normalize:
            return g
        perfect = _g(np.where(w > 0, a / w, 0.0))
    return g / perfect if perfect > 0 else float("nan")


def double_lift(
    actual_total: np.ndarray,
    expected_a: np.ndarray,
    expected_b: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    n_bins: int = 10,
) -> pl.DataFrame:
    """Equal-exposure bins ordered by ``expected_a / expected_b`` (model A cheap
    relative to B first). A wins where its A/E is closer to 1 than B's."""
    a = np.asarray(actual_total, float)
    ea = np.asarray(expected_a, float)
    eb = np.asarray(expected_b, float)
    w = np.ones_like(a) if weight is None else np.asarray(weight, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(eb > 0, ea / eb, np.inf)
    order = np.argsort(ratio, kind="stable")
    bins = _weighted_bins(order, w, n_bins)
    frame = pl.DataFrame(
        {
            "bin": bins,
            "actual": a,
            "expected_a": ea,
            "expected_b": eb,
            "exposure": w,
            "ratio": ratio,
        }
    )
    return (
        frame.group_by("bin")
        .agg(
            pl.col("exposure").sum(),
            pl.col("actual").sum(),
            pl.col("expected_a").sum(),
            pl.col("expected_b").sum(),
            (pl.col("ratio") * pl.col("exposure")).sum().alias("_r"),
        )
        .sort("bin")
        .with_columns(
            (pl.col("actual") / pl.col("expected_a")).alias("ae_a"),
            (pl.col("actual") / pl.col("expected_b")).alias("ae_b"),
            (pl.col("_r") / pl.col("exposure")).alias("mean_ratio"),
            (pl.col("bin") + 1).alias("bin"),
        )
        .drop("_r")
    )


# --------------------------------------------------------------------------
# A/E by any variable
# --------------------------------------------------------------------------
def ae_by_variable(
    df: pl.DataFrame,
    variable: str,
    actual_total: np.ndarray,
    expected_total: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    n_bins: int = 20,
    knots: list[float] | None = None,
    max_levels: int = 30,
) -> pl.DataFrame:
    """Actual, expected and A/E by band (numeric) or level (categorical) of
    ``variable`` — works for variables in or out of the model."""
    w = np.ones(df.height) if weight is None else np.asarray(weight, float)
    frame = df.select(variable).with_columns(
        pl.Series("__a__", np.asarray(actual_total, float)),
        pl.Series("__e__", np.asarray(expected_total, float)),
        pl.Series("__w__", w),
    )
    s = df[variable]
    if s.dtype in NUMERIC_DTYPES:
        ks = knots or quantile_knots(s, n_bins)
        if ks:
            frame = frame.with_columns(band_expr(variable, ks).alias("label"))
        else:
            frame = frame.with_columns(
                pl.col(variable).cast(pl.Utf8).fill_null(NULL_LABEL).alias("label")
            )
        grouped = (
            frame.group_by("label")
            .agg(
                pl.col("__w__").sum().alias("exposure"),
                pl.col("__a__").sum().alias("actual"),
                pl.col("__e__").sum().alias("expected"),
                pl.col(variable).cast(pl.Float64).min().alias("order"),
            )
            .with_columns(
                pl.when(pl.col("label") == NULL_LABEL)
                .then(float("inf"))
                .otherwise(pl.col("order"))
                .alias("order")
            )
        )
    else:
        frame = frame.with_columns(
            pl.col(variable).cast(pl.Utf8).fill_null(NULL_LABEL).alias("label")
        )
        grouped = frame.group_by("label").agg(
            pl.col("__w__").sum().alias("exposure"),
            pl.col("__a__").sum().alias("actual"),
            pl.col("__e__").sum().alias("expected"),
        )
        grouped = grouped.sort("exposure", descending=True)
        if grouped.height > max_levels:
            top = grouped.head(max_levels)
            rest = grouped.slice(max_levels)
            grouped = pl.concat(
                [
                    top,
                    pl.DataFrame(
                        {
                            "label": [f"(other {rest.height} levels)"],
                            "exposure": [rest["exposure"].sum()],
                            "actual": [rest["actual"].sum()],
                            "expected": [rest["expected"].sum()],
                        }
                    ),
                ]
            )
        grouped = grouped.with_columns(
            pl.arange(0, pl.len()).cast(pl.Float64).alias("order")
        )
    return (
        grouped.sort("order")
        .with_columns(
            (pl.col("actual") / pl.col("expected")).alias("ae"),
            (pl.col("actual") / pl.col("exposure")).alias("actual_rate"),
            (pl.col("expected") / pl.col("exposure")).alias("expected_rate"),
        )
        .select(
            "label",
            "exposure",
            "actual",
            "expected",
            "ae",
            "actual_rate",
            "expected_rate",
            "order",
        )
    )


def ae_by_pair(
    df: pl.DataFrame,
    a: str,
    b: str,
    actual_total: np.ndarray,
    expected_total: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    n_bins: int = 10,
    knots_a: list[float] | None = None,
    knots_b: list[float] | None = None,
    levels_a: list[str] | None = None,
    levels_b: list[str] | None = None,
    max_levels: int = 30,
) -> pl.DataFrame:
    """Actual, expected and A/E by **cell** of two variables — the standard way
    to look for an interaction the model is missing (large |log A/E| in a
    cell with real exposure).

    Numerics are banded by ``knots_*`` (default: quantile knots) and
    categoricals by ``levels_*`` (anything else, including null, goes to the
    Other row; default: the top ``max_levels`` levels, the rest ``"(other)"``).
    With the model's knots and levels the labels are **identical to the
    rate-table row labels** of the parents (``rate_tables`` / the Excel matrix),
    so the result joins onto an interaction table by ``(label_a, label_b)``.

    Columns: ``label_a``, ``label_b``, ``exposure``, ``actual``, ``expected``,
    ``ae``, ``actual_rate``, ``expected_rate``, ``order_a``, ``order_b``.
    """
    w = np.ones(df.height) if weight is None else np.asarray(weight, float)
    frame = df.select(a, b).with_columns(
        pl.Series("__a__", np.asarray(actual_total, float)),
        pl.Series("__e__", np.asarray(expected_total, float)),
        pl.Series("__w__", w),
    )

    def _band(
        var: str, knots: list[float] | None, levels: list[str] | None, suffix: str
    ) -> pl.DataFrame:
        nonlocal frame
        s = df[var]
        label = f"label_{suffix}"
        if s.dtype in NUMERIC_DTYPES and levels is None:
            ks = knots or quantile_knots(s, n_bins)
            if ks:
                frame = frame.with_columns(band_expr(var, ks).alias(label))
            else:
                frame = frame.with_columns(
                    pl.col(var).cast(pl.Utf8).fill_null(NULL_LABEL).alias(label)
                )
            order = (
                frame.group_by(label)
                .agg(pl.col(var).cast(pl.Float64).min().alias("o"))
                .with_columns(
                    pl.when(pl.col(label) == NULL_LABEL)
                    .then(float("inf"))
                    .otherwise(pl.col("o"))
                    .alias("o")
                )
            )
        else:
            if levels is not None:
                top = [str(lv) for lv in levels]
                other = NULL_LABEL  # unseen, lumped and null: the table's Other row
            else:
                top = (
                    frame.group_by(pl.col(var).cast(pl.Utf8).alias("lvl"))
                    .agg(pl.col("__w__").sum().alias("w"))
                    .sort("w", descending=True)
                    .drop_nulls("lvl")
                    .head(max_levels)["lvl"]
                    .to_list()
                )
                other = "(other)"
            frame = frame.with_columns(
                pl.when(pl.col(var).is_null())
                .then(pl.lit(NULL_LABEL))
                .when(pl.col(var).cast(pl.Utf8).is_in(top))
                .then(pl.col(var).cast(pl.Utf8))
                .otherwise(pl.lit(other))
                .alias(label)
            )
            names = [*top, other] + ([NULL_LABEL] if other != NULL_LABEL else [])
            order = pl.DataFrame(
                {label: names, "o": [float(i) for i in range(len(names))]}
            )
        return order.rename({"o": f"order_{suffix}"})

    order_a = _band(a, knots_a, levels_a, "a")
    order_b = _band(b, knots_b, levels_b, "b")
    out = (
        frame.group_by("label_a", "label_b")
        .agg(
            pl.col("__w__").sum().alias("exposure"),
            pl.col("__a__").sum().alias("actual"),
            pl.col("__e__").sum().alias("expected"),
        )
        .join(order_a, on="label_a", how="left")
        .join(order_b, on="label_b", how="left")
        .with_columns(
            (pl.col("actual") / pl.col("expected")).alias("ae"),
            (pl.col("actual") / pl.col("exposure")).alias("actual_rate"),
            (pl.col("expected") / pl.col("exposure")).alias("expected_rate"),
        )
        .sort(["order_a", "order_b"])
    )
    return out.select(
        "label_a",
        "label_b",
        "exposure",
        "actual",
        "expected",
        "ae",
        "actual_rate",
        "expected_rate",
        "order_a",
        "order_b",
    )


def residual_factor_search(
    df: pl.DataFrame,
    variables: list[str],
    actual_total: np.ndarray,
    expected_total: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    n_bins: int = 10,
) -> pl.DataFrame:
    """Rank variables by how much unexplained structure they show: the
    exposure-weighted standard deviation of ``log(A/E)`` across their bands.
    Large values on a variable *not* in the model suggest a missing factor."""
    rows = []
    for var in variables:
        try:
            tbl = ae_by_variable(
                df, var, actual_total, expected_total, weight, n_bins=n_bins
            )
        except Exception:  # noqa: BLE001
            continue
        tbl = tbl.filter((pl.col("expected") > 0) & (pl.col("actual") > 0))
        if tbl.height < 2:
            continue
        log_ae = np.log(tbl["ae"].to_numpy())
        w = tbl["exposure"].to_numpy()
        mean = np.average(log_ae, weights=w)
        signal = float(np.sqrt(np.average((log_ae - mean) ** 2, weights=w)))
        rows.append(
            {
                "variable": var,
                "signal": signal,
                "max_abs_log_ae": float(np.abs(log_ae).max()),
                "n_bands": tbl.height,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "variable": pl.Utf8,
            "signal": pl.Float64,
            "max_abs_log_ae": pl.Float64,
            "n_bands": pl.Int64,
        },
    ).sort("signal", descending=True)


def pearson_dispersion(
    actual_total: np.ndarray, expected_total: np.ndarray, n_params: int = 0
) -> float:
    """Pearson dispersion ``φ̂ = Σ (A − E)² / E / (n − p)`` on totals, the scale
    that makes ``(A − E)² / (φ E)`` a unit-variance quantity. About 1 for
    Poisson claim counts; far above 1 for amounts (severity, Tweedie). Used to
    make :func:`residual_pair_search` comparable across families."""
    a = np.asarray(actual_total, float)
    e = np.asarray(expected_total, float)
    ok = e > 0
    n = int(ok.sum())
    if n <= max(n_params, 0) + 1:
        return 1.0
    return float(np.sum((a[ok] - e[ok]) ** 2 / e[ok]) / (n - n_params))


def _margin_adjusted(cells: pl.DataFrame, iterations: int = 5) -> np.ndarray:
    """Expected counts per cell after re-fitting the two margins (iterative
    proportional fitting on the A/E of each row and each column), so that what
    remains is the *interaction* signal, not misfit of the two main effects."""
    act = cells["actual"].to_numpy()
    exp = cells["expected"].to_numpy().copy()
    la = cells["label_a"].to_numpy()
    lb = cells["label_b"].to_numpy()
    for _ in range(iterations):
        for labels in (la, lb):
            for lab in np.unique(labels):
                m = labels == lab
                e = exp[m].sum()
                if e > 0:
                    exp[m] *= act[m].sum() / e
    return exp


def residual_pair_search(
    df: pl.DataFrame,
    variables: list[str],
    actual_total: np.ndarray,
    expected_total: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    knots: dict[str, list[float]] | None = None,
    levels: dict[str, list[str]] | None = None,
    n_bins: int = 8,
    min_expected: float = 3.0,
    min_cell_share: float = 0.0,
    pairs: list[tuple[str, str]] | None = None,
    top: int = 20,
    dispersion: float = 1.0,
) -> pl.DataFrame:
    """Rank variable **pairs** by the interaction structure left in their cells.

    For each pair the cells (numeric variables in ``n_bins`` quantile bands
    unless ``knots`` gives the bands; categoricals by ``levels`` or their top
    levels) are kept when their expected count is at least ``min_expected``
    (and their exposure share at least ``min_cell_share``). The two margins are
    then re-fitted by iterative proportional fitting so misfit of the main
    effects does not count, and ``signal`` is the Pearson excess as a z-score:
    ``(Σ (A − E')² / (φ E') − d) / sqrt(2d)`` with ``d = k − rows − cols + 1`` the
    degrees of freedom left after the margin refit and ``φ = dispersion`` — so a pair
    with many small noisy cells does not outrank one with a single large real
    effect. Large values point at an interaction worth adding.

    ``min_expected`` is on the count scale: a cell is kept when
    ``expected >= min_expected × φ``. ``Var(A) = φ E'`` holds for claim counts
    with ``φ = 1``; for amounts
    (severity, Tweedie) pass the model's Pearson dispersion from
    :func:`pearson_dispersion`, otherwise the z-scores are meaningless.

    Columns: ``pair``, ``a``, ``b``, ``signal``, ``sd_log_ae`` (exposure-weighted
    sd of log A/E' over kept cells), ``max_abs_log_ae``, ``n_cells``,
    ``worst_cell`` (largest |A − E'| / sqrt(E')). Sorted by ``signal``, at most
    ``top`` rows."""
    knots = knots or {}
    levels = levels or {}
    if pairs is None:
        pairs = [
            (variables[i], variables[j])
            for i in range(len(variables))
            for j in range(i + 1, len(variables))
        ]
    rows = []
    for a, b in pairs:
        try:
            tbl = ae_by_pair(
                df,
                a,
                b,
                actual_total,
                expected_total,
                weight,
                n_bins=n_bins,
                knots_a=knots.get(a),
                knots_b=knots.get(b),
                levels_a=levels.get(a),
                levels_b=levels.get(b),
            )
        except Exception:  # noqa: BLE001 - a pair that cannot be banded is skipped
            continue
        total = float(tbl["exposure"].sum()) or 1.0
        cells = tbl.filter(
            (pl.col("expected") >= min_expected * float(dispersion))
            & (pl.col("exposure") / total >= min_cell_share)
        )
        if cells.height < 2:
            continue
        act = cells["actual"].to_numpy()
        exp = _margin_adjusted(cells)
        ok = exp > 0
        if ok.sum() < 2:
            continue
        act, exp = act[ok], exp[ok]
        k = int(ok.sum())
        pearson = float(np.sum((act - exp) ** 2 / exp)) / float(dispersion)
        # the margin refit uses (rows + cols - 1) degrees of freedom
        n_rows = len(set(cells["label_a"].to_numpy()[ok]))
        n_cols = len(set(cells["label_b"].to_numpy()[ok]))
        dof = max(k - n_rows - n_cols + 1, 1)
        signal = (pearson - dof) / np.sqrt(2.0 * dof)
        with np.errstate(divide="ignore"):
            ratio = np.where(act > 0, act / exp, np.nan)
        pos = ~np.isnan(ratio)
        log_ae = np.log(ratio[pos]) if pos.any() else np.zeros(1)
        w = cells["exposure"].to_numpy()[ok][pos] if pos.any() else np.ones(1)
        mean = np.average(log_ae, weights=w)
        sd = float(np.sqrt(np.average((log_ae - mean) ** 2, weights=w)))
        stand = (act - exp) / np.sqrt(exp)
        worst = int(np.argmax(np.abs(stand)))
        la = cells["label_a"].to_numpy()[ok]
        lb = cells["label_b"].to_numpy()[ok]
        rows.append(
            {
                "pair": f"{a} × {b}",
                "a": a,
                "b": b,
                "signal": float(signal),
                "sd_log_ae": sd,
                "max_abs_log_ae": float(np.abs(log_ae).max()),
                "n_cells": k,
                "worst_cell": f"{la[worst]} | {lb[worst]}",
            }
        )
    out = pl.DataFrame(
        rows,
        schema={
            "pair": pl.Utf8,
            "a": pl.Utf8,
            "b": pl.Utf8,
            "signal": pl.Float64,
            "sd_log_ae": pl.Float64,
            "max_abs_log_ae": pl.Float64,
            "n_cells": pl.Int64,
            "worst_cell": pl.Utf8,
        },
    ).sort("signal", descending=True)
    return out.head(top)


# --------------------------------------------------------------------------
# regularisation path
# --------------------------------------------------------------------------
def alpha_path(fit: GLMFit) -> pl.DataFrame:
    """One row per (stage, l1_ratio, alpha) of the fitted path with CV deviance
    (mean/std over folds), training deviance where available, the number of
    non-zero coefficients and the selected point.

    A two-stage interaction fit has **two** paths — the mains' and the cells' —
    and both are returned, told apart by the ``stage`` column (1 = main effects,
    2 = interaction cells). A one-stage fit is all ``stage = 1``."""
    if isinstance(fit, TwoStageFit):
        return pl.concat(
            [
                _stage_alpha_path(f).with_columns(
                    pl.lit(k, dtype=pl.Int64).alias("stage")
                )
                for k, f in ((1, fit.stage1), (2, fit.stage2))
            ]
        ).select(["stage", pl.exclude("stage")])
    return (
        _stage_alpha_path(fit)
        .with_columns(pl.lit(1, dtype=pl.Int64).alias("stage"))
        .select(["stage", pl.exclude("stage")])
    )


def _stage_alpha_path(fit: GLMFit) -> pl.DataFrame:
    """The path of one glum estimator (see :func:`alpha_path`)."""
    m = fit.model
    rows: list[dict[str, Any]] = []
    if hasattr(m, "alphas_") and hasattr(m, "deviance_path_"):
        alphas = np.atleast_2d(np.asarray(m.alphas_, dtype=float))
        dev = np.asarray(m.deviance_path_, dtype=float)  # folds x l1 x alphas
        coef = np.asarray(m.coef_path_, dtype=float)  # folds x l1 x alphas x p
        train_dev = getattr(m, "train_deviance_path_", None)
        l1s = np.atleast_1d(
            np.asarray(
                (
                    m.l1_ratio
                    if isinstance(m.l1_ratio, list | tuple | np.ndarray)
                    else [m.l1_ratio]
                ),
                dtype=float,
            )
        )
        for i, l1 in enumerate(l1s):
            for j, alpha in enumerate(alphas[i]):
                rows.append(
                    {
                        "l1_ratio": float(l1),
                        "alpha": float(alpha),
                        "cv_deviance": float(dev[:, i, j].mean()),
                        "cv_deviance_std": float(dev[:, i, j].std()),
                        "train_deviance": (
                            float(np.asarray(train_dev)[:, i, j].mean())
                            if train_dev is not None
                            else None
                        ),
                        "n_nonzero": float((coef[:, i, j, :] != 0).sum(axis=1).mean()),
                        "selected": bool(
                            np.isclose(alpha, m.alpha_) and np.isclose(l1, m.l1_ratio_)
                        ),
                    }
                )
    elif hasattr(m, "coef_path_") and getattr(m, "_alphas", None) is not None:
        coef = np.asarray(m.coef_path_, dtype=float)
        for j, alpha in enumerate(np.asarray(m._alphas, dtype=float)):
            rows.append(
                {
                    "l1_ratio": float(m.l1_ratio),
                    "alpha": float(alpha),
                    "cv_deviance": None,
                    "cv_deviance_std": None,
                    "train_deviance": None,
                    "n_nonzero": float((coef[j] != 0).sum()),
                    "selected": j == len(coef) - 1,
                }
            )
    else:
        rows.append(
            {
                "l1_ratio": float(m.l1_ratio),
                "alpha": fit.alpha,
                "cv_deviance": None,
                "cv_deviance_std": None,
                "train_deviance": None,
                "n_nonzero": float((fit.coef != 0).sum()),
                "selected": True,
            }
        )
    schema = {
        "l1_ratio": pl.Float64,
        "alpha": pl.Float64,
        "cv_deviance": pl.Float64,
        "cv_deviance_std": pl.Float64,
        "train_deviance": pl.Float64,
        "n_nonzero": pl.Float64,
        "selected": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema).sort(
        ["l1_ratio", "alpha"], descending=[False, True]
    )


# --------------------------------------------------------------------------
# headline metrics
# --------------------------------------------------------------------------
def model_metrics(
    fit: GLMFit,
    pred_unit_by_subset: dict[str, np.ndarray],
    frames: dict[str, pl.DataFrame],
    cfg: ModelConfig,
) -> dict[str, dict[str, float]]:
    """Per subset (e.g. ``train`` / ``holdout``): exposure, A/E, Gini, deviance."""
    out: dict[str, dict[str, float]] = {}
    fam = fit.model.family_instance
    for name, frame in frames.items():
        if frame.is_empty():
            continue
        pred = np.asarray(pred_unit_by_subset[name], dtype=float)
        actual, expected, w = totals(frame, cfg, pred)
        y_unit, w_unit = unit_values(frame, cfg)
        dev = deviance_stats(fam, y_unit, pred, w_unit if cfg.weight else None)
        out[name] = {
            "rows": float(frame.height),
            "exposure": float(w.sum()),
            "actual": float(actual.sum()),
            "expected": float(expected.sum()),
            "ae": (
                float(actual.sum() / expected.sum())
                if expected.sum() > 0
                else float("nan")
            ),
            "gini": gini(actual, expected, w),
            **dev,
        }
    return out


# --------------------------------------------------------------------------
# champion vs challenger: which relativities differ
# --------------------------------------------------------------------------
#: schema of :func:`relativity_diff` (kept explicit so an empty diff still has
#: the right columns and dtypes)
DIFF_SCHEMA = {
    "variable": pl.Utf8,
    "kind": pl.Utf8,
    "band": pl.Utf8,
    "status": pl.Utf8,
    "relativity_a": pl.Float64,
    "relativity_b": pl.Float64,
    "log_diff": pl.Float64,
    "abs_log_diff": pl.Float64,
}
#: ``band`` of a row that is about a whole variable rather than one of its bands
WHOLE_VARIABLE = "(all bands)"
#: ``variable`` / ``band`` of the base-rate row
BASE_RATE = "(base rate)"
#: ``band`` of a numeric variable whose table has no band edge at all
ALL_VALUES = "(all values)"
#: kinds whose relativity is a function of a *number*: two models can be put on
#: a common grid of band edges even when their own bands do not line up
NUMERIC_KINDS = ("numeric", "linear")

#: one non-null table row as ``(from, to, relativity at from, log slope)``
_Band = tuple[float | None, float | None, float, float]


def _model_tables(run: Any) -> dict[str, tuple[str, pl.DataFrame]]:
    """``{variable: (kind, rate table)}`` with the *current* relativities
    (manual adjustments included)."""
    tables = rate_model_tables(run.rate_model)
    return {
        var: (run.rate_model.variables[var].type, table)
        for var, table in tables.items()
    }


def _label_values(table: pl.DataFrame) -> dict[str, float]:
    """``{rate-table row label: current relativity}`` — how categoricals and
    interaction cells are matched between two models."""
    return {
        str(label): float(rel)
        for label, rel in zip(
            table["label"].to_list(), table["relativity"].to_list(), strict=True
        )
    }


def _curve(table: pl.DataFrame) -> tuple[list[_Band], float | None]:
    """``([(from, to, relativity, slope)], null relativity)`` of a numeric or
    piecewise-linear table, in band order. ``slope`` is 0 for a step table, so
    both kinds are evaluated by the same rule."""
    bands: list[_Band] = []
    null_relativity: float | None = None
    has_slope = "slope" in table.columns
    for row in table.iter_rows(named=True):
        lo, hi = row["from"], row["to"]
        relativity = float(row["relativity"])
        if lo is None and hi is None:
            null_relativity = relativity
            continue
        slope = float(row["slope"]) if has_slope and row["slope"] is not None else 0.0
        bands.append(
            (
                None if lo is None else float(lo),
                None if hi is None else float(hi),
                relativity,
                slope,
            )
        )
    bands.sort(key=lambda b: (b[0] is not None, b[0] if b[0] is not None else 0.0))
    return bands, null_relativity


def _value_at(bands: list[_Band], x: float) -> float | None:
    """The curve's relativity at ``x``: the band containing ``x`` (bands are
    ``[from, to)``), with the piecewise-linear rule
    ``relativity * exp(slope * (x - from))`` inside it."""
    for lo, hi, relativity, slope in bands:
        if (lo is None or x >= lo) and (hi is None or x < hi):
            if slope and lo is not None:
                return float(relativity * np.exp(slope * (x - lo)))
            return relativity
    return None


def _common_grid(
    a: list[_Band], b: list[_Band]
) -> list[tuple[float | None, float | None]]:
    """The union of both curves' band edges, as ``[from, to)`` intervals. Every
    interval lies inside one band of *each* model, so the two curves can be
    compared on it whatever their own bands are."""
    edges = sorted(
        {
            edge
            for bands in (a, b)
            for lo, hi, _r, _s in bands
            for edge in (lo, hi)
            if edge is not None
        }
    )
    if not edges:
        return [(None, None)]
    return [
        (None, edges[0]),
        *((edges[i], edges[i + 1]) for i in range(len(edges) - 1)),
        (edges[-1], None),
    ]


def _grid_point(lo: float | None, hi: float | None) -> float:
    """Where a common-grid interval is evaluated: its **start**, and one unit
    below the lowest edge for the open bottom interval (both curves are flat
    there, so any point below it gives the same answer)."""
    if lo is not None:
        return lo
    return (hi - 1.0) if hi is not None else 0.0


def _log_ratio(a: float, b: float) -> float | None:
    if a > 0 and b > 0:
        return float(np.log(b / a))
    return None


def _difference(rel_a: float, rel_b: float, tol: float) -> tuple[bool, float | None]:
    """``(list this band?, log(b / a))``.

    Two identical values are never a change — including two zeros or two
    negatives, whose log ratio does not exist. A value that changed but whose
    log ratio does not exist (a relativity crossing zero) is always listed.
    """
    if rel_a == rel_b:
        return False, 0.0
    log_diff = _log_ratio(rel_a, rel_b)
    if log_diff is None:
        return True, None
    return abs(log_diff) > tol, log_diff


def base_rate_change(run_a: Any, run_b: Any) -> float | None:
    """``base_rate_b / base_rate_a - 1`` — the change in the **overall level**
    between two runs, which multiplies every band's relativity change into a
    premium change. ``None`` when either base rate is not positive."""
    base_a = float(run_a.rate_model.base_rate)
    base_b = float(run_b.rate_model.base_rate)
    if base_a <= 0 or base_b <= 0:
        return None
    return base_b / base_a - 1.0


def relativity_diff(run_a: Any, run_b: Any, tol: float = 0.01) -> pl.DataFrame:
    """Which relativities differ between two fitted runs (champion ``run_a``
    vs challenger ``run_b``), as a long table.

    One row per **band that differs**: a band whose ``|log(rel_b / rel_a)|``
    exceeds ``tol`` (0.01 ≈ a 1 % change in the relativity), a band only one of
    them has, and a variable only one of them has (one row,
    ``band = "(all bands)"``). The base rates are compared too, as the row
    ``variable = "(base rate)"`` — two models with the same relativities but a
    different base rate charge different premiums; :func:`base_rate_change`
    gives that overall level change on its own.

    How two models are lined up depends on the kind of the factor:

    * **numeric and piecewise-linear** terms are compared on a **common grid**:
      the union of both models' band edges. Every interval of that grid lies
      inside one band of each model, so a model whose knots moved — or that
      represents the factor as a straight line while the other bands it — is
      compared like for like instead of by band name. Each interval is
      evaluated at its start (for a linear term that is the node the editor
      edits; a step term is flat across the interval anyway).
    * **categorical** terms and **interaction cells** are matched by their
      rate-table label, the same string the tables, the Excel workbook and
      every A/E diagnostic use. A level or cell only one model has becomes a
      ``band_only_in_*`` row.

    When the two models represent a factor differently the ``kind`` column
    reads ``"numeric → linear"``, so no row can be mistaken for a like-for-like
    comparison of the same design.

    Columns: ``variable``, ``kind``, ``band``, ``status`` (``changed``,
    ``only_in_a`` / ``only_in_b`` for a whole variable, ``band_only_in_a`` /
    ``band_only_in_b``), ``relativity_a``, ``relativity_b``, ``log_diff``
    (``log(b / a)``) and ``abs_log_diff``. Sorted by ``abs_log_diff``
    descending, the *only in* rows last. Two identical runs give an empty
    table, and the result is exactly symmetric: swapping the runs swaps the
    statuses and negates every ``log_diff``.
    """
    tol = abs(float(tol))
    a_tables = _model_tables(run_a)
    b_tables = _model_tables(run_b)
    rows: list[dict[str, Any]] = []

    def _row(var: str, kind: str, band: str, status: str, **rest: Any) -> None:
        rows.append(
            {
                "variable": var,
                "kind": kind,
                "band": band,
                "status": status,
                "relativity_a": rest.get("relativity_a"),
                "relativity_b": rest.get("relativity_b"),
                "log_diff": rest.get("log_diff"),
                "abs_log_diff": (
                    None
                    if rest.get("log_diff") is None
                    else abs(float(rest["log_diff"]))
                ),
            }
        )

    base_a = float(run_a.rate_model.base_rate)
    base_b = float(run_b.rate_model.base_rate)
    listed, base_log = _difference(base_a, base_b, tol)
    if listed:
        _row(
            BASE_RATE,
            "base rate",
            BASE_RATE,
            "changed",
            relativity_a=base_a,
            relativity_b=base_b,
            log_diff=base_log,
        )

    for var, (kind_a, table_a) in a_tables.items():
        if var not in b_tables:
            _row(var, kind_a, WHOLE_VARIABLE, "only_in_a")
            continue
        kind_b, table_b = b_tables[var]
        kind = kind_a if kind_a == kind_b else f"{kind_a} → {kind_b}"
        if kind_a in NUMERIC_KINDS and kind_b in NUMERIC_KINDS:
            _numeric_diff(_row, var, kind, table_a, table_b, tol)
        else:
            _label_diff(_row, var, kind, table_a, table_b, tol)

    for var, (kind_b, _table_b) in b_tables.items():
        if var not in a_tables:
            _row(var, kind_b, WHOLE_VARIABLE, "only_in_b")

    return pl.DataFrame(rows, schema=DIFF_SCHEMA).sort(
        ["abs_log_diff", "variable", "band"],
        descending=[True, False, False],
        nulls_last=True,
    )


def _numeric_diff(
    emit: Any,
    var: str,
    kind: str,
    table_a: pl.DataFrame,
    table_b: pl.DataFrame,
    tol: float,
) -> None:
    """Compare two numeric / piecewise-linear tables on the union of their band
    edges (see :func:`relativity_diff`)."""
    bands_a, null_a = _curve(table_a)
    bands_b, null_b = _curve(table_b)
    for lo, hi in _common_grid(bands_a, bands_b):
        band = (
            ALL_VALUES
            if lo is None and hi is None
            else level_label(FromToRow(lo, hi, 0.0))
        )
        x = _grid_point(lo, hi)
        rel_a = _value_at(bands_a, x)
        rel_b = _value_at(bands_b, x)
        if rel_a is None or rel_b is None:  # pragma: no cover - grid covers both
            continue
        listed, log_diff = _difference(rel_a, rel_b, tol)
        if listed:
            emit(
                var,
                kind,
                band,
                "changed",
                relativity_a=rel_a,
                relativity_b=rel_b,
                log_diff=log_diff,
            )
    _compare_pair(emit, var, kind, NULL_LABEL, null_a, null_b, tol)


def _label_diff(
    emit: Any,
    var: str,
    kind: str,
    table_a: pl.DataFrame,
    table_b: pl.DataFrame,
    tol: float,
) -> None:
    """Compare two tables row by row on their rate-table labels (categoricals
    and interaction cells)."""
    values_a = _label_values(table_a)
    values_b = _label_values(table_b)
    for label, rel_a in values_a.items():
        _compare_pair(emit, var, kind, label, rel_a, values_b.get(label), tol)
    for label, rel_b in values_b.items():
        if label not in values_a:
            _compare_pair(emit, var, kind, label, None, rel_b, tol)


def _compare_pair(
    emit: Any,
    var: str,
    kind: str,
    band: str,
    rel_a: float | None,
    rel_b: float | None,
    tol: float,
) -> None:
    """One band that either model may lack."""
    if rel_a is None and rel_b is None:
        return
    if rel_b is None:
        emit(var, kind, band, "band_only_in_a", relativity_a=rel_a)
        return
    if rel_a is None:
        emit(var, kind, band, "band_only_in_b", relativity_b=rel_b)
        return
    listed, log_diff = _difference(rel_a, rel_b, tol)
    if listed:
        emit(
            var,
            kind,
            band,
            "changed",
            relativity_a=rel_a,
            relativity_b=rel_b,
            log_diff=log_diff,
        )


def describe_diff(diff: pl.DataFrame, name_a: str, name_b: str) -> pl.DataFrame:
    """:func:`relativity_diff` with the statuses in words and the two relativity
    columns named after the models — what a page or a report shows.

    ``diff`` is not modified. Columns that would collide with an existing name
    are left alone.
    """
    labels = {
        "changed": "changed",
        "only_in_a": f"only in {name_a}",
        "only_in_b": f"only in {name_b}",
        "band_only_in_a": f"band only in {name_a}",
        "band_only_in_b": f"band only in {name_b}",
    }
    out = diff.with_columns(pl.col("status").replace(labels))
    rename = {
        "relativity_a": f"{name_a} relativity",
        "relativity_b": f"{name_b} relativity",
    }
    if len(set(rename.values())) == 2 and not (set(rename.values()) & set(out.columns)):
        out = out.rename(rename)
    return out
