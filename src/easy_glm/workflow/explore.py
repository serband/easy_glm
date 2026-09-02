"""Exploration: univariate summaries and the leakage report."""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import polars as pl

from easy_glm.core.design import NUMERIC_DTYPES, DesignSpec, quantile_knots, row_label
from easy_glm.core.fit import fit_glm
from easy_glm.engine.models import NULL_LABEL

from .project import ModelConfig, Project

POST_OUTCOME_PATTERN = re.compile(
    r"claim|incur|paid|loss|settl|recover|reserv|cost|severity|indemn|_dt$|date",
    re.IGNORECASE,
)


def _rate(
    frame: pl.DataFrame, target: str | None, weight: str | None, divide: bool
) -> pl.Expr:
    """Expression for the target rate of a group (see ``ModelConfig``)."""
    if target is None:
        return pl.lit(None, dtype=pl.Float64)
    if weight is None:
        return pl.col(target).mean()
    if divide:
        return pl.col(target).sum() / pl.col(weight).sum()
    return (pl.col(target) * pl.col(weight)).sum() / pl.col(weight).sum()


def band_labels(knots: list[float]) -> list[str]:
    """Labels of the bands ``(-inf, k0), [k0, k1), ..., [k_last, inf)`` — identical
    to the rate-table row labels of a step encoder with the same knots."""
    ks = [float(k) for k in knots]
    edges: list[float | None] = [None, *ks, None]
    return [row_label((edges[i], edges[i + 1])) for i in range(len(edges) - 1)]


def band_expr(series_name: str, knots: list[float]) -> pl.Expr:
    """Band a numeric column by ``knots`` into the rate-table row labels; nulls
    (and NaN) get :data:`NULL_LABEL`, the label of the table's null row."""
    knots = [float(k) for k in knots]
    labels = band_labels(knots)
    col = pl.col(series_name).cast(pl.Float64)
    expr = pl.when(col.is_null() | col.is_nan()).then(pl.lit(NULL_LABEL))
    expr = expr.when(col < knots[0]).then(pl.lit(labels[0]))
    for i in range(1, len(knots)):
        expr = expr.when(col < knots[i]).then(pl.lit(labels[i]))
    return expr.otherwise(pl.lit(labels[-1]))


def univariate(
    df: pl.DataFrame,
    variable: str,
    *,
    target: str | None = None,
    weight: str | None = None,
    divide_target_by_weight: bool = False,
    n_bins: int = 20,
    knots: list[float] | None = None,
    max_levels: int = 30,
) -> dict[str, Any]:
    """Exposure and target rate by band/level for one variable.

    Returns ``{"variable", "kind", "n", "null_share", "n_unique", "table"}``
    where ``table`` has ``label``, ``exposure``, ``share``, ``rate``, ``order``.
    """
    s = df[variable]
    numeric = s.dtype in NUMERIC_DTYPES
    w = pl.col(weight).sum() if weight else pl.len().cast(pl.Float64)
    if numeric:
        ks = knots or quantile_knots(s, n_bins)
        if ks:
            banded = df.with_columns(band_expr(variable, ks).alias("__band__"))
            order_expr = pl.col(variable).cast(pl.Float64).min().alias("order")
        else:  # constant column
            banded = df.with_columns(
                pl.col(variable).cast(pl.Utf8).fill_null(NULL_LABEL).alias("__band__")
            )
            order_expr = pl.lit(0.0).alias("order")
        table = (
            banded.group_by("__band__")
            .agg(
                w.alias("exposure"),
                _rate(df, target, weight, divide_target_by_weight).alias("rate"),
                order_expr,
            )
            .rename({"__band__": "label"})
            .with_columns(
                pl.when(pl.col("label") == NULL_LABEL)
                .then(pl.lit(float("inf")))
                .otherwise(pl.col("order"))
                .alias("order")
            )
            .sort("order")
        )
    else:
        lv = df.with_columns(pl.col(variable).cast(pl.Utf8).alias("__lvl__"))
        table = (
            lv.group_by("__lvl__")
            .agg(
                w.alias("exposure"),
                _rate(df, target, weight, divide_target_by_weight).alias("rate"),
            )
            .rename({"__lvl__": "label"})
            .sort("exposure", descending=True)
        )
        if table.height > max_levels:
            top = table.head(max_levels)
            rest = table.slice(max_levels)
            other_rate = None
            if target is not None:
                rest_levels = rest["label"].to_list()
                rest_rows = lv.filter(
                    pl.col("__lvl__").is_in(rest_levels) | pl.col("__lvl__").is_null()
                )
                other_rate = rest_rows.select(
                    _rate(df, target, weight, divide_target_by_weight)
                ).item()
            table = pl.concat(
                [
                    top,
                    pl.DataFrame(
                        {
                            "label": [f"(other {rest.height} levels)"],
                            "exposure": [rest["exposure"].sum()],
                            "rate": [other_rate],
                        },
                        schema={
                            "label": pl.Utf8,
                            "exposure": pl.Float64,
                            "rate": pl.Float64,
                        },
                    ),
                ]
            )
        table = table.with_columns(
            pl.col("label").fill_null(NULL_LABEL),
            pl.arange(0, pl.len()).cast(pl.Float64).alias("order"),
        )
    total = table["exposure"].sum() or 1.0
    table = table.with_columns((pl.col("exposure") / total).alias("share")).select(
        "label", "exposure", "share", "rate", "order"
    )
    return {
        "variable": variable,
        "kind": "numeric" if numeric else "categorical",
        "n": df.height,
        "null_share": s.null_count() / max(df.height, 1),
        "n_unique": s.n_unique(),
        "table": table,
    }


# --------------------------------------------------------------------------
# Leakage report
# --------------------------------------------------------------------------
def _null_deviance(family, y: np.ndarray, w: np.ndarray | None) -> float:
    mu = np.full_like(y, np.average(y, weights=w), dtype=float)
    return float(family.deviance(y, mu, sample_weight=w))


def single_factor_strength(
    train: pl.DataFrame,
    variable: str,
    cfg: ModelConfig,
    *,
    n_bins: int = 20,
    min_level_share: float = 0.002,
) -> float | None:
    """Share of the null deviance explained by a one-variable ridge GLM
    (``1 - deviance / null_deviance``). ``None`` if the variable cannot be encoded."""
    try:
        spec = DesignSpec.from_data(
            train,
            [variable],
            n_bins=n_bins,
            min_level_share=min_level_share,
            max_levels=60,
        )
    except (ValueError, KeyError):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = fit_glm(
                train,
                spec,
                cfg.target,
                family=cfg.family,
                weight_col=cfg.weight,
                offset_col=cfg.offset,
                divide_target_by_weight=cfg.divide_target_by_weight,
                alpha=1e-6,
                l1_ratio=0.0,
                max_iter=50,
            )
        except Exception:  # noqa: BLE001 - a failing single-factor fit is not fatal
            return None
    y = train[cfg.target].cast(pl.Float64).to_numpy()
    w = train[cfg.weight].cast(pl.Float64).to_numpy() if cfg.weight else None
    if cfg.divide_target_by_weight and w is not None:
        y = y / w
    mu = fit.predict(train)
    fam = fit.model.family_instance
    dev = float(fam.deviance(y, mu, sample_weight=w))
    null = _null_deviance(fam, y, w)
    if null <= 0:
        return None
    return max(0.0, 1.0 - dev / null)


def leakage_report(
    df: pl.DataFrame,
    project: Project,
    *,
    model: str | None = None,
    candidates: list[str] | None = None,
    sample_rows: int = 50_000,
    seed: int = 42,
    strength_check: float = 0.4,
    strength_flag: float = 0.8,
) -> pl.DataFrame:
    """Rank candidate predictors by how likely they are to leak the outcome.

    ``df`` should be the prepared frame (with the split column); only training
    rows (a sample of at most ``sample_rows``) are used. Returns one row per
    candidate with ``score`` (0-100), ``recommendation`` (``ignore`` / ``check``
    / ``ok``), ``flags`` and the underlying statistics.
    """
    cfg = project.models.get(
        model or project.champion or "", None
    ) or project.new_model("__probe__")
    if "__probe__" in project.models:
        cfg = project.models.pop("__probe__")
        if project.champion == "__probe__":
            project.champion = None
    target, weight = cfg.target, cfg.weight
    if target is None or target not in df.columns:
        raise ValueError(
            "A target column (role 'target') is required for the leakage report"
        )

    split_col = project.data.split.column
    train = df.filter(pl.col(split_col) == 1) if split_col in df.columns else df
    if train.height > sample_rows:
        train = train.sample(n=sample_rows, seed=seed)

    reserved = {target, weight, cfg.offset, split_col} - {None}
    if candidates is None:
        candidates = [
            c
            for c in df.columns
            if c not in reserved
            and project.data.roles.get(c, "predictor") in ("predictor", "id")
        ]
    y = train[target].cast(pl.Float64).to_numpy()
    w = train[weight].cast(pl.Float64).to_numpy() if weight else np.ones(train.height)
    y_rate = y / w if cfg.divide_target_by_weight else y

    rows: list[dict[str, Any]] = []
    for var in candidates:
        s = train[var]
        n = max(train.height, 1)
        null_share = s.null_count() / n
        n_unique = s.n_unique()
        unique_ratio = n_unique / n
        numeric = s.dtype in NUMERIC_DTYPES
        flags: list[str] = []
        score = 0.0

        # degenerate
        top_share = 0.0
        if n_unique <= 1:
            flags.append("constant")
        else:
            counts = (
                s.drop_nulls()
                .cast(pl.Utf8)
                .value_counts()
                .sort("count", descending=True)
            )
            if counts.height:
                top_share = counts["count"][0] / max(s.drop_nulls().len(), 1)
                if top_share > 0.995:
                    flags.append("one level > 99.5%")

        # identifier-like
        if unique_ratio > 0.9 or (not numeric and unique_ratio > 0.5):
            flags.append("identifier-like")
            score = max(score, 90.0 if unique_ratio > 0.9 else 60.0)

        # correlation / proxy
        corr = None
        if numeric and n_unique > 1:
            x = s.cast(pl.Float64).to_numpy()
            mask = ~np.isnan(x)
            if mask.sum() > 10 and np.nanstd(x[mask]) > 0 and np.std(y_rate[mask]) > 0:
                xr = pl.Series(x[mask]).rank().to_numpy()
                yr = pl.Series(y_rate[mask]).rank().to_numpy()
                corr = float(np.corrcoef(xr, yr)[0, 1])
                if abs(corr) > 0.9:
                    flags.append(f"target proxy (|ρ|={abs(corr):.2f})")
                    score = max(score, 95.0)

        # post-outcome naming
        if POST_OUTCOME_PATTERN.search(var):
            flags.append("post-outcome name")
            score = max(score, 40.0)

        # single-factor strength
        strength = None
        if "constant" not in flags and not (unique_ratio > 0.9 and not numeric):
            strength = single_factor_strength(train, var, cfg)
            if strength is not None:
                if strength > strength_flag:
                    flags.append(f"explains {strength:.0%} of deviance")
                    score = max(score, 98.0)
                elif strength > strength_check:
                    flags.append(f"explains {strength:.0%} of deviance")
                    score = max(score, 70.0)
                else:
                    score = max(score, 100.0 * strength * 0.5)

        if null_share > 0.98:
            flags.append("almost all null")

        if score >= 80 or "constant" in flags:
            rec = "ignore"
        elif score >= 40:
            rec = "check"
        else:
            rec = "ok"
        prior = project.exploration.get("leakage", {})
        if var in prior.get("ignored", []):
            rec = "ignored"
        elif var in prior.get("acknowledged", []):
            rec = "acknowledged"

        rows.append(
            {
                "variable": var,
                "role": project.data.roles.get(var, "predictor"),
                "recommendation": rec,
                "score": round(score, 1),
                "flags": ", ".join(flags),
                "deviance_explained": strength,
                "rank_corr": corr,
                "unique_ratio": unique_ratio,
                "top_level_share": top_share,
                "null_share": null_share,
                "n_unique": n_unique,
            }
        )
    schema = {
        "variable": pl.Utf8,
        "role": pl.Utf8,
        "recommendation": pl.Utf8,
        "score": pl.Float64,
        "flags": pl.Utf8,
        "deviance_explained": pl.Float64,
        "rank_corr": pl.Float64,
        "unique_ratio": pl.Float64,
        "top_level_share": pl.Float64,
        "null_share": pl.Float64,
        "n_unique": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema).sort("score", descending=True)
