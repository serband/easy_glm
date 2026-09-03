"""Execute a model of a :class:`Project`: build the design on training rows,
fit, compile the RateModel, apply manual adjustments, compute metrics."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
from easy_glm.core.fit import GLMFit, TwoStageFit, fit_glm, fit_two_stage
from easy_glm.core.tables import rate_tables, to_rate_model
from easy_glm.engine.rate_model import RateModel

from .diagnostics import expected_claims, model_metrics
from .prep import train_holdout
from .project import Adjustment, Interaction, ModelConfig, Project, VariableDesign

#: sentinel for "keep the model config's own base-rate override"
_KEEP: Any = object()


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


class UnusableColumnError(ValueError):
    """A predictor that cannot become GLM features at all because of what the
    *training rows* hold: a constant column, or one that is entirely null.

    It is told apart from every other design error because nothing the user can
    set on the Design page would rescue it — the only sensible outcomes are to
    drop the column from the design or to refuse the whole fit. The workbench
    drops it and names it, so one useless column cannot block a whole model.
    """


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
    if kind in ("step", "linear", "continuous"):
        if not numeric:
            raise ValueError(f"{variable!r} is not numeric; cannot use a {kind} design")
        n_bins = vd.n_bins or d.n_bins
        null_ind = d.null_indicator if vd.null_indicator is None else vd.null_indicator
        clamp = (float(vd.clamp[0]), float(vd.clamp[1])) if vd.clamp else None
        if kind == "continuous":
            # one band, a single slope on the clamped value: no knots to derive
            return linear_encoder_from_data(
                variable,
                series,
                knots=[],
                n_bins=n_bins,
                clamp=clamp,
                null_indicator=null_ind,
            )
        if isinstance(vd.knots, list | tuple):
            knots: list[float] | None = [float(k) for k in vd.knots]
        elif vd.knots == "integer":
            knots = integer_knots(series, d.max_integer_knots) or quantile_knots(
                series, n_bins
            )
        else:
            knots = quantile_knots(series, n_bins)
        if kind == "linear":
            return linear_encoder_from_data(
                variable,
                series,
                knots=knots or [],
                n_bins=n_bins,
                clamp=clamp,
                null_indicator=null_ind,
            )
        if not knots:
            raise UnusableColumnError(
                f"Cannot derive knots for {variable!r} (constant or all-null on train)"
            )
        return StepEncoder(variable, knots, null_indicator=null_ind)
    share = vd.min_level_share if vd.min_level_share is not None else d.min_level_share
    levels = vd.levels or frequent_levels(
        series, min_share=share, max_levels=vd.max_levels, weights=weights
    )
    if not levels:
        if series.null_count() == series.len():
            raise UnusableColumnError(
                f"Cannot derive levels for {variable!r}: all null on train"
            )
        raise ValueError(
            f"No level of {variable!r} reaches the minimum level share ({share:.2%} of "
            f"training rows; {series.n_unique()} distinct values). Lower the share "
            "on the Design page or treat the column differently"
        )
    return CategoricalEncoder(variable, levels, other_label=other_label_for(levels))


def other_label_for(levels: list[str]) -> str:
    """The lumped-bucket label: ``"Other"`` unless a real level is called that
    (e.g. after a recode with default "Other"), then ``"Other (lumped)"``."""
    label = "Other"
    while label in levels:
        label += " (lumped)"
    return label


def build_design(
    project: Project,
    train: pl.DataFrame,
    predictors: list[str],
    *,
    weight_col: str | None = None,
    interactions: list[Interaction] | None = None,
    dropped: list[str] | None = None,
) -> DesignSpec:
    """A :class:`DesignSpec` for ``predictors`` (and ``interactions`` on top of
    them) from the project's design config; kept cells are decided on ``train``.

    Pass ``dropped`` (an empty list) to skip predictors that cannot be encoded
    from the training rows at all — a constant or all-null column — instead of
    raising: their names are appended to it. Every other design problem is
    still an error, because the user can act on it.

    ``kind="continuous"`` builds the same :class:`LinearEncoder` as ``"linear"``
    with no interior knots, so a continuous term shares the linear term's rate
    table, editor, Excel sheet and exported script."""
    weights = train[weight_col] if weight_col and weight_col in train.columns else None
    encoders: dict[str, Encoder] = {}
    for var in predictors:
        if var not in train.columns:
            raise KeyError(f"Predictor {var!r} not in the prepared data")
        vd = project.design.variables.get(var, VariableDesign())
        try:
            encoders[var] = encoder_for(var, train[var], vd, project, weights=weights)
        except UnusableColumnError:
            if dropped is None:
                raise
            dropped.append(var)
    if predictors and not encoders:
        raise ValueError(
            "Every predictor is constant or all-null on the training rows: "
            + ", ".join(dropped or predictors)
        )
    spec = DesignSpec(encoders)
    for it in interactions or []:
        if dropped is not None and (it.a not in encoders or it.b not in encoders):
            continue  # a parent was dropped: the interaction goes with it
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
    #: predictors left out of the design because they are constant or all-null
    #: on the training rows (the fit ran without them; the page names them)
    dropped_predictors: list[str] = field(default_factory=list)

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """Per-unit predictions from the (possibly adjusted) rate model."""
        return self.rate_model.predict(df, exposure_col=None)

    @property
    def alpha(self) -> float:
        """Stage 1's penalty — the one that fitted the main effects."""
        return self.fit.alpha

    @property
    def alpha_stage2(self) -> float | None:
        """The penalty the interaction cells were fitted at, or ``None`` when
        the model has no interactions and so only one stage."""
        return self.fit.alpha_stage2 if isinstance(self.fit, TwoStageFit) else None

    @property
    def cells_kept(self) -> int:
        """Cell columns the second stage fitted (cells with enough exposure)."""
        return sum(len(e.cells) for e in self.spec.interactions)

    def summary(self) -> dict[str, Any]:
        h = self.metrics.get("holdout", {})
        t = self.metrics.get("train", {})
        return {
            "name": self.name,
            "family": self.fit.family,
            "alpha": self.fit.alpha,
            "alpha_stage2": self.alpha_stage2,
            "cells_kept": self.cells_kept,
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


class AdjustmentError(ValueError):
    """A manual adjustment the RateModel refuses (e.g. a non-positive value on a
    piecewise-linear band); ``adjustment`` is the offending entry."""

    def __init__(self, adjustment: Adjustment, message: str) -> None:
        super().__init__(message)
        self.adjustment = adjustment


def apply_adjustments(rm: RateModel, cfg: ModelConfig) -> None:
    """Apply ``cfg.adjustments`` to ``rm`` in order. Raises
    :class:`AdjustmentError` for an adjustment the model refuses."""
    for adj in cfg.adjustments:
        config = rm.variables.get(adj.variable)
        if config is None:
            # an AdjustmentError, not a KeyError: the caller's job is to drop it
            # and say so (a predictor left the model, or a snapshot is older
            # than the design), never to show a traceback
            raise AdjustmentError(
                adj,
                f"Adjustment refers to {adj.variable!r}, which is not a variable of "
                f"the model (known: {list(rm.variables)})",
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
        try:
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
        except ValueError as exc:
            raise AdjustmentError(adj, str(exc)) from exc
    if cfg.adjustments:
        rm.create_snapshot(f"{len(cfg.adjustments)} manual adjustment(s)")


def stage2_alpha(cfg: ModelConfig) -> float | None:
    """The penalty strength for the interaction cells, or ``None`` for "the
    same as the mains".

    The second stage is one fit with one alpha, so when several interactions
    ask for one the largest (the most cautious) wins; per-interaction
    differences belong in ``Interaction.penalty_weight``."""
    asked = [it.alpha for it in cfg.interactions if it.alpha is not None]
    return max(asked) if asked else None


def snapshot_metrics(
    fit: GLMFit, metrics: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """``metrics`` (one entry per data subset) plus a ``model`` entry naming the
    penalty of each stage, so a saved ``.easyglm`` says how it was fitted."""
    stage2 = fit.alpha_stage2 if isinstance(fit, TwoStageFit) else None
    return {
        **metrics,
        "model": {
            "alpha": fit.alpha,
            "alpha_stage2": stage2,
            "stages": 2 if stage2 is not None else 1,
        },
    }


def run_model(
    project: Project,
    df: pl.DataFrame,
    model_name: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> ModelRun:
    """Fit ``project.models[model_name]`` on the training rows of the prepared
    frame ``df`` (must contain the split column) and return a :class:`ModelRun`.

    ``progress`` is called with a short status string about once a second while
    the fit runs (see :func:`easy_glm.core.fit.fit_glm`); it is how the
    workbench shows elapsed time on a long fit. The design matrix is chosen by
    row count — a book past
    :data:`~easy_glm.core.design.SPARSE_ROW_THRESHOLD` rows is fitted on the
    compact one, which is the same fit in a fraction of the memory.

    A model **with interactions is fitted in two stages** (Q5): stage 1 is the
    main-effect model — bit for bit the fit the same model without the
    interaction would give — and stage 2 fits the interaction cells on top of
    it with no intercept and stage 1's linear predictor as the offset. Main
    rate tables and the base rate therefore never move when an interaction is
    added, and every cell is a pure adjustment (1.00 = none)."""
    problems = project.validate(model_name)
    if problems:
        raise ValueError("Project is not valid:\n- " + "\n- ".join(problems))
    cfg = project.models[model_name]
    train, holdout = train_holdout(df, project.data.split)
    if train.is_empty():
        raise ValueError("No training rows after the split")

    dropped: list[str] = []
    spec = build_design(
        project,
        train,
        cfg.predictors,
        weight_col=cfg.weight,
        interactions=cfg.interactions,
        dropped=dropped,
    )
    pen = cfg.penalty
    kwargs: dict[str, Any] = {}
    if cfg.link:
        kwargs["link"] = cfg.link
    fit_kwargs: dict[str, Any] = dict(
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
        progress=progress,
        **kwargs,
    )
    fit: GLMFit
    if spec.interactions:
        # two stages (the actuary's answer to Q5): the mains are fitted exactly
        # as they would be without the interaction and are then frozen, and the
        # cells are fitted on top of them as pure adjustments
        fit = fit_two_stage(
            train,
            spec,
            cfg.target,
            stage2_alpha=stage2_alpha(cfg),
            **fit_kwargs,
        )
    else:
        fit = fit_glm(train, spec, cfg.target, **fit_kwargs)
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
    rm.set_snapshot_metrics(snapshot_metrics(fit, metrics))
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
        dropped_predictors=dropped,
    )


def exposure_for(project: Project, cfg: ModelConfig) -> str | None:
    """Column the RateModel multiplies by when scoring."""
    return project.exposure or (cfg.weight if cfg.divide_target_by_weight else None)


def rate_model_for(
    project: Project,
    run: ModelRun,
    adjustments: list[Adjustment] | None = None,
    *,
    base_rate_override: float | None = _KEEP,
) -> RateModel:
    """The RateModel the run's fit gives with ``adjustments`` applied — a fresh
    model, with nothing on the run touched and no refit.

    ``adjustments`` defaults to the model's current ones; pass a snapshot's list
    to see (or compare against) the tables as they stood then. This is the one
    place that turns "a fit plus a list of adjustments" into tables, so the
    editor, a snapshot and a snapshot diff can never disagree about what a set
    of adjustments means.
    """
    cfg = project.models[run.name]
    rm = to_rate_model(
        run.fit,
        base=cfg.base,  # type: ignore[arg-type]
        base_rate_override=(
            cfg.base_rate_override
            if base_rate_override is _KEEP
            else base_rate_override
        ),
        exposure_col=exposure_for(project, cfg),
        train_test_col=project.data.split.column,
        model_type=cfg.family,
    )
    if adjustments is None:
        apply_adjustments(rm, cfg)
    else:
        apply_adjustments(rm, replace(cfg, adjustments=list(adjustments)))
    return rm


def missing_variables(rm: RateModel, adjustments: list[Adjustment]) -> list[str]:
    """Variables the adjustments name that ``rm`` does not have.

    A set of adjustments that names one of these cannot be applied at all — the
    caller (a snapshot restore, a project loaded against a changed design)
    should say which factors are missing and leave the model alone rather than
    apply half of them.
    """
    return sorted({a.variable for a in adjustments if a.variable not in rm.variables})


def rebalance_override(
    project: Project, run: ModelRun, df: pl.DataFrame
) -> float | None:
    """The base rate that puts **total expected claims back where the fitted
    model had them** on the training rows — the off-balance correction after
    editing rate tables.

    Predictions are linear in the base rate, so this is one ratio:
    ``base_rate x (expected claims as fitted) / (expected claims now)``. Returns
    ``None`` when it cannot be computed (no training rows, or a model that
    expects nothing).
    """
    cfg = project.models[run.name]
    train, _holdout = train_holdout(df, project.data.split)
    if train.is_empty():
        return None
    fitted = rate_model_for(project, run, [], base_rate_override=None)
    target = expected_claims(fitted, train, cfg)
    current = expected_claims(run.rate_model, train, cfg)
    if not current > 0 or not target > 0:
        return None
    return float(run.rate_model.base_rate * target / current)


def rebuild_rate_model(project: Project, run: ModelRun, df: pl.DataFrame) -> ModelRun:
    """Recompile the run's RateModel from its fit (no refit) — used after the
    manual adjustments or base-rate override of its model config change —
    and refresh tables and metrics in place."""
    cfg = project.models[run.name]
    rm = rate_model_for(project, run)
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
    rm.set_snapshot_metrics(snapshot_metrics(run.fit, run.metrics))
    run.tables = rate_tables(run.fit, base=cfg.base)  # type: ignore[arg-type]
    return run
