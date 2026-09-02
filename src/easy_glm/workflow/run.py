"""Execute a model of a :class:`Project`: build the design on training rows,
fit, compile the RateModel, apply manual adjustments, compute metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl

from easy_glm.core.design import (
    NUMERIC_DTYPES,
    CategoricalEncoder,
    DesignSpec,
    Encoder,
    InteractionEncoder,
    StepEncoder,
    frequent_levels,
    linear_encoder_from_data,
    quantile_knots,
)
from easy_glm.core.fit import GLMFit, fit_glm
from easy_glm.core.tables import rate_tables, to_rate_model
from easy_glm.engine.rate_model import RateModel

from .diagnostics import model_metrics
from .prep import train_holdout
from .project import Interaction, ModelConfig, Project, VariableDesign


# --------------------------------------------------------------------------
# design from the project
# --------------------------------------------------------------------------
def integer_knots(series: pl.Series, max_knots: int) -> list[float] | None:
    """One knot per integer strictly above the minimum, or ``None`` if the
    range is too wide (caller falls back to quantiles)."""
    s = series.drop_nulls().cast(pl.Float64)
    if s.is_empty():
        return None
    lo, hi = math.floor(s.min()), math.floor(s.max())
    if hi - lo > max_knots or hi <= lo:
        return None
    return [float(k) for k in range(lo + 1, hi + 1)]


def encoder_for(
    variable: str,
    series: pl.Series,
    vd: VariableDesign,
    project: Project,
    *,
    weights: pl.Series | None = None,
) -> Encoder:
    d = project.design.defaults
    numeric = series.dtype in NUMERIC_DTYPES
    kind = vd.kind or ("step" if numeric else "categorical")
    if kind in ("step", "linear"):
        if not numeric:
            raise ValueError(f"{variable!r} is not numeric; cannot use a {kind} design")
        n_bins = vd.n_bins or d.n_bins
        if isinstance(vd.knots, list | tuple):
            knots: list[float] | None = [float(k) for k in vd.knots]
        elif vd.knots == "integer":
            knots = integer_knots(series, d.max_integer_knots) or quantile_knots(
                series, n_bins
            )
        else:
            knots = quantile_knots(series, n_bins)
        null_ind = d.null_indicator if vd.null_indicator is None else vd.null_indicator
        if kind == "linear":
            clamp = (float(vd.clamp[0]), float(vd.clamp[1])) if vd.clamp else None
            return linear_encoder_from_data(
                variable,
                series,
                knots=knots or [],
                n_bins=n_bins,
                clamp=clamp,
                null_indicator=null_ind,
            )
        if not knots:
            raise ValueError(
                f"Cannot derive knots for {variable!r} (constant or all-null on train)"
            )
        return StepEncoder(variable, knots, null_indicator=null_ind)
    levels = vd.levels or frequent_levels(
        series,
        min_share=(
            vd.min_level_share if vd.min_level_share is not None else d.min_level_share
        ),
        max_levels=vd.max_levels,
        weights=weights,
    )
    if not levels:
        raise ValueError(f"Cannot derive levels for {variable!r} (all null on train)")
    return CategoricalEncoder(variable, levels)


def build_design(
    project: Project,
    train: pl.DataFrame,
    predictors: list[str],
    *,
    weight_col: str | None = None,
    interactions: list[Interaction] | None = None,
) -> DesignSpec:
    """A :class:`DesignSpec` for ``predictors`` (and ``interactions`` on top of
    them) from the project's design config; kept cells are decided on ``train``."""
    weights = train[weight_col] if weight_col and weight_col in train.columns else None
    encoders: dict[str, Encoder] = {}
    for var in predictors:
        if var not in train.columns:
            raise KeyError(f"Predictor {var!r} not in the prepared data")
        vd = project.design.variables.get(var, VariableDesign())
        encoders[var] = encoder_for(var, train[var], vd, project, weights=weights)
    spec = DesignSpec(encoders)
    for it in interactions or []:
        spec.add_interaction(
            InteractionEncoder.from_data(
                spec[it.a],
                spec[it.b],
                train,
                weights=weights,
                min_cell_exposure=it.min_cell_exposure,
                penalty_weight=it.penalty_weight,
            )
        )
    return spec


def monotone_for(project: Project, cfg: ModelConfig) -> dict[str, str]:
    """Design-level monotone directions, overridden by the model's."""
    out = {
        v: vd.monotone
        for v, vd in project.design.variables.items()
        if vd.monotone and v in cfg.predictors
    }
    out.update({v: d for v, d in cfg.monotone.items() if v in cfg.predictors})
    return out


# --------------------------------------------------------------------------
# a run
# --------------------------------------------------------------------------
@dataclass
class ModelRun:
    name: str
    config: ModelConfig
    spec: DesignSpec
    fit: GLMFit
    rate_model: RateModel
    tables: dict[str, pl.DataFrame]
    metrics: dict[str, dict[str, float]]
    project_snapshot: dict[str, Any]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    train_rows: int = 0
    holdout_rows: int = 0

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """Per-unit predictions from the (possibly adjusted) rate model."""
        return self.rate_model.predict(df, exposure_col=None)

    @property
    def alpha(self) -> float:
        return self.fit.alpha

    def summary(self) -> dict[str, Any]:
        h = self.metrics.get("holdout", {})
        t = self.metrics.get("train", {})
        return {
            "name": self.name,
            "family": self.fit.family,
            "alpha": self.fit.alpha,
            "features": len(self.fit.coef),
            "non_zero": int((self.fit.coef != 0).sum()),
            "train_ae": t.get("ae"),
            "holdout_ae": h.get("ae"),
            "train_gini": t.get("gini"),
            "holdout_gini": h.get("gini"),
            "train_dev_explained": t.get("deviance_explained"),
            "holdout_dev_explained": h.get("deviance_explained"),
            "adjustments": len(self.config.adjustments),
            "created_at": self.created_at,
        }


def apply_adjustments(rm: RateModel, cfg: ModelConfig) -> None:
    for adj in cfg.adjustments:
        config = rm.variables.get(adj.variable)
        if config is None:
            raise KeyError(
                f"Adjustment refers to {adj.variable!r}, which is not a variable of "
                f"the model (known: {list(rm.variables)})"
            )
        is_cell = config.type == "interaction"
        if is_cell != bool(adj.cell):
            raise ValueError(
                f"Adjustment on {adj.variable!r}: "
                + (
                    "it is an interaction, so the adjustment needs cell=True with "
                    "from_b / to_b"
                    if is_cell
                    else "it is a main effect, but the adjustment is marked cell=True"
                )
            )
        if is_cell:
            rm.update_relativity(
                adj.variable,
                adj.from_,
                adj.to_,
                float(adj.relativity),
                from_b=adj.from_b,
                to_b=adj.to_b,
            )
        else:
            rm.update_relativity(
                adj.variable, adj.from_, adj.to_, float(adj.relativity)
            )
    if cfg.adjustments:
        rm.create_snapshot(f"{len(cfg.adjustments)} manual adjustment(s)")


def run_model(project: Project, df: pl.DataFrame, model_name: str) -> ModelRun:
    """Fit ``project.models[model_name]`` on the training rows of the prepared
    frame ``df`` (must contain the split column) and return a :class:`ModelRun`."""
    problems = project.validate(model_name)
    if problems:
        raise ValueError("Project is not valid:\n- " + "\n- ".join(problems))
    cfg = project.models[model_name]
    train, holdout = train_holdout(df, project.data.split)
    if train.is_empty():
        raise ValueError("No training rows after the split")

    spec = build_design(
        project,
        train,
        cfg.predictors,
        weight_col=cfg.weight,
        interactions=cfg.interactions,
    )
    pen = cfg.penalty
    kwargs: dict[str, Any] = {}
    if cfg.link:
        kwargs["link"] = cfg.link
    fit = fit_glm(
        train,
        spec,
        cfg.target,
        family=cfg.family,
        weight_col=cfg.weight,
        offset_col=cfg.offset,
        divide_target_by_weight=cfg.divide_target_by_weight,
        alpha=pen.alpha,
        cv=None if pen.alpha is not None else pen.cv,
        n_alphas=pen.n_alphas,
        l1_ratio=pen.l1_ratio,
        min_alpha_ratio=pen.min_alpha_ratio,
        monotone=monotone_for(project, cfg),
        **kwargs,
    )
    exposure = exposure_for(project, cfg)
    rm = to_rate_model(
        fit,
        base=cfg.base,  # type: ignore[arg-type]
        base_rate_override=cfg.base_rate_override,
        exposure_col=exposure,
        train_test_col=project.data.split.column,
        model_type=cfg.family,
    )
    apply_adjustments(rm, cfg)
    frames = {"train": train, "holdout": holdout}
    preds = {
        k: rm.predict(v, exposure_col=None)
        for k, v in frames.items()
        if not v.is_empty()
    }
    metrics = model_metrics(
        fit, preds, {k: v for k, v in frames.items() if not v.is_empty()}, cfg
    )
    rm.set_snapshot_metrics(metrics)
    return ModelRun(
        name=model_name,
        config=project.models[model_name],
        spec=spec,
        fit=fit,
        rate_model=rm,
        tables=rate_tables(fit, base=cfg.base),  # type: ignore[arg-type]
        metrics=metrics,
        project_snapshot=project.to_dict(),
        train_rows=train.height,
        holdout_rows=holdout.height,
    )


def exposure_for(project: Project, cfg: ModelConfig) -> str | None:
    """Column the RateModel multiplies by when scoring."""
    return project.exposure or (cfg.weight if cfg.divide_target_by_weight else None)


def rebuild_rate_model(project: Project, run: ModelRun, df: pl.DataFrame) -> ModelRun:
    """Recompile the run's RateModel from its fit (no refit) — used after the
    manual adjustments or base-rate override of its model config change —
    and refresh tables and metrics in place."""
    cfg = project.models[run.name]
    rm = to_rate_model(
        run.fit,
        base=cfg.base,  # type: ignore[arg-type]
        base_rate_override=cfg.base_rate_override,
        exposure_col=exposure_for(project, cfg),
        train_test_col=project.data.split.column,
        model_type=cfg.family,
    )
    apply_adjustments(rm, cfg)
    train, holdout = train_holdout(df, project.data.split)
    frames = {
        k: v
        for k, v in {"train": train, "holdout": holdout}.items()
        if not v.is_empty()
    }
    preds = {k: rm.predict(v, exposure_col=None) for k, v in frames.items()}
    run.rate_model = rm
    run.config = cfg
    run.metrics = model_metrics(run.fit, preds, frames, cfg)
    rm.set_snapshot_metrics(run.metrics)
    run.tables = rate_tables(run.fit, base=cfg.base)  # type: ignore[arg-type]
    return run
