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
from easy_glm.core.fit import GLMFit

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


def gini(
    actual_total: np.ndarray,
    expected_total: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    normalize: bool = True,
) -> float:
    """Exposure-weighted Gini of the ordering by predicted rate; ``normalize``
    divides by the Gini of the perfect ordering (by actual rate)."""
    a = np.asarray(actual_total, float)
    e = np.asarray(expected_total, float)
    w = np.ones_like(a) if weight is None else np.asarray(weight, float)
    if a.sum() <= 0 or w.sum() <= 0:
        return float("nan")

    def _g(score: np.ndarray) -> float:
        order = np.argsort(-score, kind="stable")
        cum_w = np.concatenate([[0.0], np.cumsum(w[order]) / w.sum()])
        cum_a = np.concatenate([[0.0], np.cumsum(a[order]) / a.sum()])
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
                pl.col(variable).cast(pl.Utf8).fill_null("null").alias("label")
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
                pl.when(pl.col("label") == "null")
                .then(float("inf"))
                .otherwise(pl.col("order"))
                .alias("order")
            )
        )
    else:
        frame = frame.with_columns(
            pl.col(variable).cast(pl.Utf8).fill_null("null").alias("label")
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


# --------------------------------------------------------------------------
# regularisation path
# --------------------------------------------------------------------------
def alpha_path(fit: GLMFit) -> pl.DataFrame:
    """One row per (l1_ratio, alpha) of the fitted path with CV deviance
    (mean/std over folds), training deviance where available, the number of
    non-zero coefficients and the selected point."""
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
