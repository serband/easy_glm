"""Sequential, fold-local fitting of raw-column CatBoost pair corrections.

Only frozen pair tables enter the returned RateModel.  The teacher and its
training Pool never enter deployed scoring, serialization, or exports.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from importlib import metadata
from itertools import product
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from easy_glm.core.design import SPARSE_ROW_THRESHOLD, frequent_levels, quantile_knots
from easy_glm.core.fit import _fit_main_effects
from easy_glm.core.tables import to_rate_model
from easy_glm.engine._scoring import row_index
from easy_glm.engine.models import (
    FromToRow,
    PairCellRow,
    PairTableConfig,
    VariableConfig,
)
from easy_glm.engine.rate_model import RateModel

from .pair_distillation import (
    DistilledPairCells,
    distill_pair_cells,
    fit_catboost_pair_raw,
)
from .project import (
    ModelConfig,
    PairCandidateConfig,
    PairStageConfig,
    Project,
    VariableDesign,
)

MAX_STAGES = 8
MAX_CELLS = 10_000
MAX_CANDIDATES = 3
MAX_PREFIX_CONFIGS = 8
MAX_TEACHER_FITS = 5_000
MAX_SECONDS = 900
MAX_ESTIMATED_PEAK_BYTES = 3 * 1024**3
CV_TIE_TOLERANCE = 1e-8
ALGORITHM_VERSION = "pair-stages-2-optuna"
SEARCH_SPACE_VERSION = "shallow-v1"
MAIN_TPE_STARTUP_TRIALS = 3
PREFIX_TPE_STARTUP_TRIALS = 2


@dataclass(frozen=True)
class FoldLoss:
    fold: int
    loss_sum: float
    weight_sum: float
    prefix_loss_sum: float
    teacher_loss_sum: float | None
    target_scale: float | None = None
    approximation_loss_sum: float | None = None


@dataclass(frozen=True)
class CandidateCV:
    candidate: PairCandidateConfig | None
    folds: tuple[FoldLoss, ...]

    @property
    def table_loss(self) -> float:
        return sum(f.loss_sum for f in self.folds) / sum(
            f.weight_sum for f in self.folds
        )

    @property
    def prefix_loss(self) -> float:
        return sum(f.prefix_loss_sum for f in self.folds) / sum(
            f.weight_sum for f in self.folds
        )

    @property
    def teacher_loss(self) -> float | None:
        total = 0.0
        for fold in self.folds:
            if fold.teacher_loss_sum is None:
                return None
            total += fold.teacher_loss_sum
        return total / sum(f.weight_sum for f in self.folds)

    @property
    def approximation_loss(self) -> float | None:
        total = 0.0
        for fold in self.folds:
            if fold.approximation_loss_sum is None:
                return None
            total += fold.approximation_loss_sum
        return total / sum(f.weight_sum for f in self.folds)


@dataclass(frozen=True)
class SearchTrial:
    """Serializable evidence for a completed Optuna proposal."""

    number: int
    state: str
    candidates: tuple[PairCandidateConfig | None, ...]
    folds: tuple[FoldLoss, ...]
    value: float
    outer_fold: int | None = None


@dataclass
class PairStageArtifact:
    stage_id: str
    parents: tuple[str, str]
    table: PairTableConfig
    chosen_candidate: PairCandidateConfig | None
    prefix_fingerprint: str
    cv_candidates: tuple[CandidateCV, ...] = ()
    prefix_cv_loss: float | None = None
    table_cv_loss: float | None = None
    teacher_cv_loss: float | None = None
    approximation_loss: float | None = None
    observed_table_minus_teacher_cv_loss: float | None = None
    training_teacher_loss: float | None = None
    training_table_loss: float | None = None
    target_scale: float = 1.0
    scaling_policy: str = "tweedie_fold_train_max_over_1000_else_one"
    fit_seconds: float = 0.0
    reused: bool = False
    status: str = "up_to_date"
    baseline_stage_ids: tuple[str, ...] = ()
    selected_prefix_configs: tuple[tuple[int, ...], ...] = ()
    selected_prefix_parameters: tuple[tuple[PairCandidateConfig | None, ...], ...] = ()
    search_trials: tuple[SearchTrial, ...] = ()
    prefix_search_trials: tuple[SearchTrial, ...] = ()


def _response(cfg: ModelConfig, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not cfg.target:
        raise ValueError("A pair stage needs a target column")
    target = frame[cfg.target].cast(pl.Float64).to_numpy()
    weight = (
        frame[cfg.weight].cast(pl.Float64).to_numpy()
        if cfg.weight
        else np.ones(frame.height, dtype=np.float64)
    )
    if not np.all(np.isfinite(weight) & (weight > 0)):
        raise ValueError("Model fitting weights must be finite and strictly positive")
    if cfg.divide_target_by_weight:
        target = target / weight
    if not np.all(np.isfinite(target) & (target >= 0)):
        raise ValueError("Poisson/Tweedie pair targets must be finite and nonnegative")
    return target.astype(np.float64), weight.astype(np.float64)


def _deviance_sum(
    observed: np.ndarray, prediction: np.ndarray, weight: np.ndarray, power: float
) -> float:
    if not np.all(np.isfinite(prediction) & (prediction > 0)):
        raise ValueError("Deployed pair predictions must be finite and positive")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if power == 1:
            terms = np.where(
                observed > 0, observed * np.log(observed / prediction), 0.0
            )
            dev = 2 * (terms - observed + prediction)
        else:
            dev = 2 * (
                observed ** (2 - power) / ((1 - power) * (2 - power))
                - observed * prediction ** (1 - power) / (1 - power)
                + prediction ** (2 - power) / (2 - power)
            )
    result = float(np.sum(weight * dev, dtype=np.float64))
    if not math.isfinite(result):
        raise ValueError("Validation loss is not finite")
    return result


def _axis(
    project: Project, train: pl.DataFrame, variable: str, weight_col: str | None
) -> VariableConfig:
    series = train[variable]
    vd = project.design.variables.get(variable, VariableDesign())
    defaults = project.design.defaults
    if series.dtype.is_numeric() and vd.kind != "categorical":
        if isinstance(vd.knots, (list, tuple)):
            cuts = [float(v) for v in vd.knots]
        elif vd.knots == "integer":
            from .run import integer_knots

            cuts = integer_knots(series, defaults.max_integer_knots) or quantile_knots(
                series, vd.n_bins or defaults.n_bins
            )
        else:
            cuts = quantile_knots(series, vd.n_bins or defaults.n_bins)
        cuts = sorted(set(cuts or []))
        if not cuts:
            present = series.cast(pl.Float64).drop_nulls()
            median = present.median()
            cuts = [float(cast(float, median)) if median is not None else 0.0]
        if vd.clamp:
            lo, hi = map(float, vd.clamp)
            cuts = sorted({lo, hi, *(cut for cut in cuts if lo < cut < hi)})
        if not all(math.isfinite(cut) for cut in cuts):
            raise ValueError(f"Pair axis {variable!r} has non-finite boundaries")
        rows = [FromToRow(None, cuts[0] if cuts else None, 1.0)]
        rows.extend(FromToRow(a, b, 1.0) for a, b in zip(cuts, cuts[1:], strict=False))
        if cuts:
            rows.append(FromToRow(cuts[-1], None, 1.0))
        rows.append(FromToRow(None, None, 1.0))
        # With no finite cut there is one all-range row.  The scorer needs its
        # open endpoints and the explicit null row.
        axis = VariableConfig(type="numeric", table=rows)
    else:
        share = (
            vd.min_level_share
            if vd.min_level_share is not None
            else defaults.min_level_share
        )
        levels = vd.levels or frequent_levels(
            series,
            min_share=share,
            max_levels=vd.max_levels,
            weights=train[weight_col] if weight_col else None,
        )
        from .run import other_label_for

        label = other_label_for(levels)
        axis = VariableConfig(
            type="categorical",
            table=[FromToRow(level, level, 1.0) for level in levels]
            + [FromToRow(None, None, 1.0)],
            other_label=label,
        )
    RateModel._precompute_variables({variable: axis})
    return axis


def _raw_features(
    frame: pl.DataFrame,
    parents: tuple[str, str],
    axes: tuple[VariableConfig, VariableConfig],
) -> tuple[np.ndarray, list[int]]:
    columns: list[np.ndarray] = []
    cats: list[int] = []
    for position, (name, axis) in enumerate(zip(parents, axes, strict=True)):
        if axis.type == "categorical":
            # Prefix ensures a real string cannot collide with the null token.
            raw = frame[name].cast(pl.Utf8).to_list()
            columns.append(
                np.asarray(
                    ["N" if value is None else "S" + value for value in raw],
                    dtype=object,
                )
            )
            cats.append(position)
        else:
            numeric = frame[name].cast(pl.Float64).to_numpy()
            if np.any(np.isinf(numeric)):
                raise ValueError(f"Pair feature {name!r} contains infinity")
            columns.append(numeric)
    return np.column_stack(columns), cats


def _cell_ids(
    frame: pl.DataFrame,
    parents: tuple[str, str],
    axes: tuple[VariableConfig, VariableConfig],
) -> np.ndarray:
    a, b = parents
    return row_index(frame[a], axes[0]) * len(axes[1].table) + row_index(
        frame[b], axes[1]
    )


def _table(
    stage: PairStageConfig,
    axes: tuple[VariableConfig, VariableConfig],
    result: DistilledPairCells,
) -> PairTableConfig:
    width = len(axes[1].table)
    cells = [
        PairCellRow(
            axis_a_row=index // width,
            axis_b_row=index % width,
            relativity=float(result.relativities[index]),
            row_count=int(result.row_count[index]),
            fitting_weight=float(result.fitting_weight[index]),
            weight_share=float(result.weight_share[index]),
            fallback_reason=result.fallback_reason[index],
        )
        for index in range(len(result.relativities))
    ]
    return PairTableConfig(
        stage_id=stage.stage_id, parents=(stage.a, stage.b), axes=axes, cells=cells
    )


def _fit_table(
    project: Project,
    cfg: ModelConfig,
    stage: PairStageConfig,
    candidate: PairCandidateConfig | None,
    train: pl.DataFrame,
    prefix: RateModel,
    *,
    axes: tuple[VariableConfig, VariableConfig] | None = None,
    deadline: float,
) -> tuple[PairTableConfig, Any | None]:
    if time.monotonic() > deadline:
        raise TimeoutError("Pair fitting exceeded the 900-second budget")
    parents = (stage.a, stage.b)
    axes = axes or (
        _axis(project, train, stage.a, cfg.weight),
        _axis(project, train, stage.b, cfg.weight),
    )
    n_cells = len(axes[0].table) * len(axes[1].table)
    if n_cells > MAX_CELLS:
        raise ValueError(
            f"Pair {stage.a} × {stage.b} needs {n_cells:,} cells; limit is {MAX_CELLS:,}. Reduce the configured bins/levels"
        )
    with np.errstate(over="ignore", under="ignore"):
        baseline = np.exp(prefix.linear_predictor(train))
    if not np.all(np.isfinite(baseline) & (baseline > 0)):
        raise ValueError(
            "The deployed prefix baseline is not finite and strictly positive"
        )
    y, weight = _response(cfg, train)
    teacher = None
    teacher_mean = baseline
    if candidate is not None:
        raw, categorical = _raw_features(train, parents, axes)
        positive_target = y[y > 0]
        target_scale = (
            max(1.0, float(np.max(positive_target)) / 1000.0)
            if cfg.family == "tweedie" and positive_target.size
            else 1.0
        )
        teacher = fit_catboost_pair_raw(
            raw,
            y,
            baseline,
            sample_weight=weight,
            family=cfg.family,
            tweedie_power=cfg.tweedie_power if cfg.family == "tweedie" else None,
            target_scale=target_scale,
            iterations=candidate.iterations,
            depth=candidate.depth,
            learning_rate=candidate.learning_rate,
            l2_leaf_reg=candidate.l2_leaf_reg,
            thread_count=1,
            seed=stage.seed,
            cat_features=categorical,
        )
        teacher_mean = teacher.predict_mean(raw, baseline)
    result = distill_pair_cells(
        _cell_ids(train, parents, axes),
        baseline,
        teacher_mean,
        sample_weight=weight,
        family=cfg.family,
        tweedie_power=cfg.tweedie_power if cfg.family == "tweedie" else None,
        n_cells=n_cells,
        min_weight_share=stage.min_weight_share,
    )
    return _table(stage, axes, result), teacher


def _append_table(prefix: RateModel, table: PairTableConfig) -> RateModel:
    scorer = copy.deepcopy(prefix)
    scorer.add_pair_table(copy.deepcopy(table))
    return scorer


def _apply_stage_edits(
    project: Project,
    cfg: ModelConfig,
    stage: PairStageConfig,
    scorer: RateModel,
    *,
    fold_local: bool,
) -> None:
    edits = [adj for adj in cfg.adjustments if adj.stage_id == stage.stage_id]
    if not edits:
        return
    table = scorer.get_pair_table(stage.stage_id)
    if fold_local:
        for parent in (stage.a, stage.b):
            axis = table.axes[0 if parent == stage.a else 1]
            design = project.design.variables.get(parent, VariableDesign())
            if axis.type == "numeric" and not isinstance(design.knots, (list, tuple)):
                raise ValueError(
                    f"Pair edit on {stage.stage_id!r} cannot replay across CV folds: "
                    f"{parent!r} has data-derived cuts. Specify fixed cuts or reset the edit"
                )
    from .run import apply_adjustments

    mapped = []
    for edit in edits:
        row_keys = ((edit.from_, edit.to_), (edit.from_b, edit.to_b))
        positions = []
        for axis, key in zip(table.axes, row_keys, strict=True):
            matches = [
                index
                for index, row in enumerate(axis.table)
                if (row.from_, row.to_) == key
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Pair edit on {stage.stage_id!r} has no exact cell in this fold; "
                    "specify fixed cuts/levels or reset the edit"
                )
            positions.append(matches[0])
        mapped.append(replace(edit, axis_a_row=positions[0], axis_b_row=positions[1]))
    apply_adjustments(scorer, replace(cfg, adjustments=mapped))


def _folds(length: int, count: int, seed: int) -> list[np.ndarray]:
    if length < count:
        raise ValueError(
            f"Five-fold pair validation needs at least {count} rows; got {length}"
        )
    permutation = np.random.default_rng(seed).permutation(length)
    return [np.sort(part) for part in np.array_split(permutation, count)]


def _candidate_options(stage: PairStageConfig) -> list[PairCandidateConfig | None]:
    return [None, *stage.candidates]


def _prefix_configs(stages: list[PairStageConfig]) -> list[tuple[int, ...]]:
    if not stages:
        return [()]
    options = [range(len(_candidate_options(stage))) for stage in stages]
    configs = list(product(*options))
    configs.sort(key=lambda item: (sum(value != 0 for value in item), sum(item), item))
    # Include complete trained-prefix alternatives even when a shallow sort
    # would fill the budget with single-stage corrections only.
    homogeneous = [
        tuple([candidate_index] * len(stages))
        for candidate_index in range(1, max(len(values) for values in options))
        if all(candidate_index in values for values in options)
    ]
    priority = [tuple([0] * len(stages)), *homogeneous]
    return (priority + [item for item in configs if item not in priority])[
        :MAX_PREFIX_CONFIGS
    ]


def _optuna_module() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Automatic pair tuning requires Optuna. Install it with "
            "pip install 'easy-glm[pairs]'."
        ) from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def _suggest_candidate(trial: Any, prefix: str) -> PairCandidateConfig:
    return PairCandidateConfig(
        depth=trial.suggest_int(f"{prefix}depth", 2, 5),
        iterations=trial.suggest_int(f"{prefix}iterations", 40, 160, step=20),
        learning_rate=trial.suggest_float(
            f"{prefix}learning_rate", 0.03, 0.15, log=True
        ),
        l2_leaf_reg=trial.suggest_float(f"{prefix}l2_leaf_reg", 0.1, 20.0, log=True),
    )


def _suggest_prefix(
    trial: Any, previous: list[PairStageConfig]
) -> tuple[PairCandidateConfig | None, ...]:
    result: list[PairCandidateConfig | None] = []
    for index, stage in enumerate(previous):
        if stage.search is None:
            choice = trial.suggest_int(
                f"stage_{index}_choice", 0, len(stage.candidates)
            )
            result.append(_candidate_options(stage)[choice])
        elif trial.suggest_categorical(f"stage_{index}_active", [False, True]):
            result.append(_suggest_candidate(trial, f"stage_{index}_"))
        else:
            result.append(None)
    return tuple(result)


def _safe_prefix_params(previous: list[PairStageConfig]) -> dict[str, Any]:
    """One data-independent full-prefix starting point for a small TPE study."""
    params: dict[str, Any] = {}
    for index, stage in enumerate(previous):
        if stage.search is None:
            params[f"stage_{index}_choice"] = 1 if stage.candidates else 0
        else:
            params.update(
                {
                    f"stage_{index}_active": True,
                    f"stage_{index}_depth": 2,
                    f"stage_{index}_iterations": 60,
                    f"stage_{index}_learning_rate": 0.08,
                    f"stage_{index}_l2_leaf_reg": 3.0,
                }
            )
    return params


def preflight_pair_stages(
    stages: list[PairStageConfig],
    project: Project,
    train: pl.DataFrame,
    cfg: ModelConfig,
) -> dict[str, Any]:
    """Bound the nested search and pair grids before starting any model fit."""
    if len(stages) > MAX_STAGES:
        raise ValueError(f"At most {MAX_STAGES} pair stages are supported")
    if train.height - math.ceil(train.height / 5) < 5:
        raise ValueError(
            "Nested five-by-five pair CV needs at least seven training rows"
        )
    expected_fits = 0
    grid_cells: list[int] = []
    for index, stage in enumerate(stages):
        if stage.search is None and len(stage.candidates) > MAX_CANDIDATES:
            raise ValueError(
                f"Stage {stage.stage_id} has too many teacher candidates (limit {MAX_CANDIDATES})"
            )
        if stage.cv_folds != 5:
            raise ValueError("V1 pair stages require five deterministic row folds")
        axes = (
            _axis(project, train, stage.a, cfg.weight),
            _axis(project, train, stage.b, cfg.weight),
        )
        cells = len(axes[0].table) * len(axes[1].table)
        grid_cells.append(cells)
        if cells > MAX_CELLS:
            raise ValueError(
                f"Stage {stage.stage_id} needs {cells:,} cells, above the {MAX_CELLS:,} limit. Reduce bins/levels before fitting"
            )
        previous = stages[:index]
        if any(item.search is not None for item in previous):
            prefix_trials = (
                stage.search.prefix_trials
                if stage.search is not None
                else max(
                    item.search.prefix_trials
                    for item in previous
                    if item.search is not None
                )
            )
            expected_fits += 25 * prefix_trials * index
        else:
            configurations = _prefix_configs(previous)
            expected_fits += 25 * sum(
                sum(choice != 0 for choice in config) for config in configurations
            )
        candidate_trials = (
            stage.search.trials if stage.search is not None else len(stage.candidates)
        )
        expected_fits += 5 * index + 5 * candidate_trials + 1
    if expected_fits > MAX_TEACHER_FITS:
        raise ValueError(
            f"The requested nested validation may require {expected_fits:,} CatBoost fits (limit {MAX_TEACHER_FITS:,}). Reduce stages/candidates"
        )
    from .run import build_design

    design = build_design(
        project, train, cfg.predictors, weight_col=cfg.weight, dropped=[]
    )
    estimated_design_bytes = (
        design.expected_design_bytes(train.height)
        if train.height >= SPARSE_ROW_THRESHOLD
        else train.height * design.n_features * 8
    )
    # Conservative workload estimate, calibrated above the 371 MiB observed
    # on 4k training rows in the Phase 0 CPU spike.  Native library overhead
    # can vary by platform, so this is a refusal gate, not a peak-RSS guarantee.
    estimated_peak_bytes = (
        512 * 1024**2
        + train.height * 512
        + 4 * estimated_design_bytes
        + 128 * sum(grid_cells)
    )
    if estimated_peak_bytes > MAX_ESTIMATED_PEAK_BYTES:
        raise ValueError(
            f"Estimated pair-fit peak memory is {estimated_peak_bytes / 1024**3:.1f} GiB "
            f"(limit {MAX_ESTIMATED_PEAK_BYTES / 1024**3:.0f} GiB). Reduce the "
            "training data or pair-grid size; rows are never silently sampled"
        )
    return {
        "teacher_fits_upper_bound": expected_fits,
        "fold_main_fits_upper_bound": 30 if len(stages) > 1 else 5,
        "main_cv_folds": cfg.penalty.cv if cfg.penalty.alpha is None else 0,
        "main_n_alphas": cfg.penalty.n_alphas if cfg.penalty.alpha is None else 0,
        "grid_cells": grid_cells,
        "estimated_table_bytes": 128 * sum(grid_cells),
        "estimated_main_design_bytes": estimated_design_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
        "deadline_seconds": MAX_SECONDS,
    }


def _frame_bytes(frame: pl.DataFrame, columns: set[str]) -> bytes:
    selected = frame.select(sorted(columns))
    digest = hashlib.sha256()
    digest.update(str(selected.schema).encode())
    digest.update(str(selected.height).encode())
    for seed in (42, 2147483647):
        digest.update(selected.hash_rows(seed=seed).to_numpy().tobytes())
    return digest.digest()


@lru_cache(maxsize=1)
def _dependency_bytes() -> bytes:
    digest = hashlib.sha256(ALGORITHM_VERSION.encode())
    for package in (
        "easy-glm",
        "catboost",
        "optuna",
        "numpy",
        "polars",
        "glum",
        "scipy",
        "scikit-learn",
        "tabmat",
    ):
        try:
            digest.update(metadata.version(package).encode())
        except metadata.PackageNotFoundError:
            digest.update(b"missing")
    package_root = Path(__file__).resolve().parents[1]
    for relative in (
        "workflow/pair_stages.py",
        "workflow/pair_distillation.py",
        "workflow/project.py",
        "engine/rate_model.py",
        "core/fit.py",
        "core/design.py",
    ):
        digest.update((package_root / relative).read_bytes())
    return digest.digest()


def _scorer_signature(scorer: RateModel) -> dict[str, Any]:
    raw = scorer._to_dict()
    raw.pop("snapshots", None)
    raw.pop("current_version", None)
    return raw


def _main_settings(cfg: ModelConfig) -> dict[str, Any]:
    settings = asdict(cfg)
    for irrelevant in ("pair_stages", "pair_method", "snapshots"):
        settings.pop(irrelevant, None)
    settings["adjustments"] = [
        asdict(adj) for adj in cfg.adjustments if adj.stage_id is None
    ]
    return settings


def _fingerprint(
    project: Project,
    cfg: ModelConfig,
    train: pl.DataFrame,
    prefix: RateModel,
    stage: PairStageConfig,
    prior_stages: list[PairStageConfig],
) -> str:
    digest = hashlib.sha256()
    digest.update(_dependency_bytes())
    relevant = {
        cfg.target,
        cfg.weight,
        cfg.offset,
        *cfg.predictors,
        *(parent for table in prefix.pair_tables for parent in table.parents),
        stage.a,
        stage.b,
    } - {None}
    digest.update(_frame_bytes(train, {name for name in relevant if name is not None}))
    config = _main_settings(cfg)
    design_variables = (
        set(cfg.predictors)
        | {stage.a, stage.b}
        | {parent for table in prefix.pair_tables for parent in table.parents}
    )
    design = {
        "defaults": asdict(project.design.defaults),
        "variables": {
            name: asdict(project.design.variables[name])
            for name in sorted(design_variables)
            if name in project.design.variables
        },
    }
    stable = {
        "config": config,
        "design": design,
        "split": asdict(project.data.split),
        "preparation": asdict(project.data),
        "prefix": _scorer_signature(prefix),
        "prior_stages": [asdict(previous) for previous in prior_stages],
        "prior_edits": [
            asdict(adj)
            for adj in cfg.adjustments
            if adj.stage_id in {previous.stage_id for previous in prior_stages}
        ],
        "stage": asdict(stage),
    }
    digest.update(json.dumps(stable, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def fit_pair_stages(
    project: Project,
    train: pl.DataFrame,
    model_config: ModelConfig,
    main_rate_model: RateModel,
    *,
    cache: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    deadline_monotonic: float | None = None,
    replay_pair_adjustments: bool = False,
) -> tuple[RateModel, list[PairStageArtifact]]:
    """Fit the ordered pair chain, selecting each deployed table by nested CV.

    No holdout frame is accepted.  Every outer fold builds its own main GLM,
    previous teacher tables and candidate table from its training partition.
    Inner selection evaluates complete, fixed prefix configurations.  A cache
    can reuse full-training prefix artifacts; it never supplies fold artefacts.
    """
    stages = model_config.pair_stages
    if not stages:
        return copy.deepcopy(main_rate_model), []
    # Inner and outer folds retain multiple frames.  Keep only fitting and
    # scoring columns so unrelated prepared columns do not multiply in memory.
    from .run import exposure_for

    required = {
        model_config.target,
        model_config.weight,
        model_config.offset,
        exposure_for(project, model_config),
        *model_config.predictors,
        *(parent for stage in stages for parent in (stage.a, stage.b)),
    }
    train = train.select([name for name in train.columns if name in required])
    if model_config.family not in ("poisson", "tweedie") or (
        model_config.link and model_config.link != "log"
    ):
        raise ValueError("Pair stages support only Poisson/log or Tweedie/log")
    estimate = preflight_pair_stages(stages, project, train, model_config)
    if progress:
        progress(
            f"Pair preflight: up to {estimate['teacher_fits_upper_bound']:,} teacher fits, "
            f"{estimate['fold_main_fits_upper_bound']} fold-local main GLM fits "
            f"(CV={estimate['main_cv_folds']}, n_alphas={estimate['main_n_alphas']}), "
            f"grids {estimate['grid_cells']} cells, about "
            f"{estimate['estimated_table_bytes'] / 1024**2:.1f} MiB tables, "
            f"{estimate['estimated_main_design_bytes'] / 1024**2:.1f} MiB main design; "
            f"five outer and five inner folds"
        )
    deadline = deadline_monotonic or time.monotonic() + MAX_SECONDS
    if time.monotonic() > deadline:
        raise TimeoutError(
            "Pair fitting exceeded the 900-second budget before teacher training"
        )
    cache = cache if cache is not None else {}
    full_cache: dict[str, PairStageArtifact] = cache.setdefault("full_prefix", {})
    inner_main_cache: dict[bytes, RateModel] = cache.setdefault("fold_main", {})
    fold_prefix_cache: dict[bytes, RateModel] = cache.setdefault("fold_prefix", {})
    prefix_selection_cache: dict[
        bytes,
        tuple[
            tuple[PairCandidateConfig | None, ...],
            tuple[int, ...],
            tuple[SearchTrial, ...],
        ],
    ] = cache.setdefault("prefix_selection", {})

    def main_for(partition: pl.DataFrame) -> RateModel:
        if time.monotonic() > deadline:
            raise TimeoutError("Pair fitting exceeded the 900-second budget")
        main_columns = {
            name
            for name in (
                model_config.target,
                model_config.weight,
                model_config.offset,
                *model_config.predictors,
            )
            if name is not None
        }
        design_settings = {
            name: asdict(project.design.variables[name])
            for name in model_config.predictors
            if name in project.design.variables
        }
        key = hashlib.sha256(
            _frame_bytes(partition, main_columns)
            + json.dumps(
                _main_settings(model_config), sort_keys=True, default=str
            ).encode()
            + json.dumps(design_settings, sort_keys=True, default=str).encode()
            + json.dumps(asdict(project.design.defaults), sort_keys=True).encode()
            + json.dumps(
                asdict(project.data.split), sort_keys=True, default=str
            ).encode()
            + json.dumps(asdict(project.data), sort_keys=True, default=str).encode()
            + _dependency_bytes()
        ).digest()
        if key in inner_main_cache:
            return copy.deepcopy(inner_main_cache[key])
        from .run import apply_adjustments, build_design, exposure_for, monotone_for

        dropped: list[str] = []
        spec = build_design(
            project,
            partition,
            model_config.predictors,
            weight_col=model_config.weight,
            dropped=dropped,
        )
        penalty = model_config.penalty
        if model_config.target is None:
            raise ValueError("A pair stage needs a target column")
        fit_kwargs: dict[str, Any] = {
            "family": model_config.family,
            "weight_col": model_config.weight,
            "offset_col": model_config.offset,
            "divide_target_by_weight": model_config.divide_target_by_weight,
            "alpha": penalty.alpha,
            "cv": None if penalty.alpha is not None else penalty.cv,
            "cv_seed": project.data.split.seed,
            "n_alphas": penalty.n_alphas,
            "l1_ratio": penalty.l1_ratio,
            "min_alpha_ratio": penalty.min_alpha_ratio,
            "monotone": monotone_for(project, model_config),
        }
        if model_config.family == "tweedie":
            fit_kwargs["tweedie_power"] = model_config.tweedie_power
        if model_config.link:
            fit_kwargs["link"] = model_config.link
        fit = _fit_main_effects(partition, spec, model_config.target, **fit_kwargs)
        rm = to_rate_model(
            fit,
            base=model_config.base,  # type: ignore[arg-type]
            base_rate_override=model_config.base_rate_override,
            exposure_col=exposure_for(project, model_config),
            offset_is_premium=False,
        )
        try:
            main_edits = [
                adj
                for adj in model_config.adjustments
                if getattr(adj, "stage_id", None) is None
            ]
            apply_adjustments(rm, replace(model_config, adjustments=main_edits))
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Fold-local main adjustment cannot map to this fold's axes; specify fixed cuts or reset that edit before pair tuning"
            ) from exc
        inner_main_cache[key] = copy.deepcopy(rm)
        return rm

    def fixed_prefix(
        partition: pl.DataFrame,
        parameters: tuple[PairCandidateConfig | None, ...],
        base: RateModel,
    ) -> RateModel:
        prior = stages[: len(parameters)]
        columns = {
            name
            for name in (
                model_config.target,
                model_config.weight,
                model_config.offset,
                *model_config.predictors,
                *(parent for previous in prior for parent in (previous.a, previous.b)),
            )
            if name is not None
        }
        prior_ids = {previous.stage_id for previous in prior}
        edits = [
            asdict(adj) for adj in model_config.adjustments if adj.stage_id in prior_ids
        ]
        pair_design = {
            name: asdict(project.design.variables[name])
            for name in {
                parent for previous in prior for parent in (previous.a, previous.b)
            }
            if name in project.design.variables
        }
        key = hashlib.sha256(
            _frame_bytes(partition, columns)
            + json.dumps(
                [asdict(previous) for previous in prior], sort_keys=True
            ).encode()
            + json.dumps(edits, sort_keys=True, default=str).encode()
            + json.dumps(pair_design, sort_keys=True, default=str).encode()
            + json.dumps(asdict(project.design.defaults), sort_keys=True).encode()
            + json.dumps(asdict(project.data), sort_keys=True, default=str).encode()
            + json.dumps(
                [asdict(item) if item is not None else None for item in parameters],
                sort_keys=True,
            ).encode()
            + json.dumps(_scorer_signature(base), sort_keys=True, default=str).encode()
            + _dependency_bytes()
        ).digest()
        if key in fold_prefix_cache:
            return copy.deepcopy(fold_prefix_cache[key])
        scorer = base
        for previous, candidate in zip(prior, parameters, strict=True):
            table, _ = _fit_table(
                project,
                model_config,
                previous,
                candidate,
                partition,
                scorer,
                deadline=deadline,
            )
            scorer = _append_table(scorer, table)
            _apply_stage_edits(project, model_config, previous, scorer, fold_local=True)
        fold_prefix_cache[key] = copy.deepcopy(scorer)
        return scorer

    def select_prefix(
        outer_train: pl.DataFrame, previous: list[PairStageConfig], outer_fold: int
    ) -> tuple[
        tuple[PairCandidateConfig | None, ...],
        tuple[int, ...],
        tuple[SearchTrial, ...],
    ]:
        if not previous:
            return (), (), ()
        current = stages[len(previous)]
        adaptive = any(item.search is not None for item in previous)
        configurations = _prefix_configs(previous) if not adaptive else []
        if len(configurations) == 1:
            choices = configurations[0]
            return (
                tuple(
                    _candidate_options(item)[choice]
                    for item, choice in zip(previous, choices, strict=True)
                ),
                choices,
                (),
            )
        relevant = {
            name
            for name in (
                model_config.target,
                model_config.weight,
                model_config.offset,
                *model_config.predictors,
                *(parent for item in previous for parent in (item.a, item.b)),
            )
            if name is not None
        }
        prior_ids = {item.stage_id for item in previous}
        design_names = set(model_config.predictors) | {
            parent for item in previous for parent in (item.a, item.b)
        }
        search_identity = {
            "main": _main_settings(model_config),
            "previous": [asdict(item) for item in previous],
            "previous_edits": [
                asdict(adj)
                for adj in model_config.adjustments
                if adj.stage_id in prior_ids
            ],
            "design": {
                "defaults": asdict(project.design.defaults),
                "variables": {
                    name: asdict(project.design.variables[name])
                    for name in sorted(design_names)
                    if name in project.design.variables
                },
            },
            "data": asdict(project.data),
            "outer_fold": outer_fold,
            "sampler_seed": current.seed + outer_fold,
            "sampler_multivariate": False,
            "prefix_trials": (
                current.search.prefix_trials
                if current.search is not None
                else (
                    max(
                        item.search.prefix_trials
                        for item in previous
                        if item.search is not None
                    )
                    if adaptive
                    else None
                )
            ),
            "space": SEARCH_SPACE_VERSION,
            "startup": PREFIX_TPE_STARTUP_TRIALS,
        }
        selection_key = hashlib.sha256(
            _dependency_bytes()
            + _frame_bytes(outer_train, relevant)
            + json.dumps(search_identity, sort_keys=True, default=str).encode()
        ).digest()
        if selection_key in prefix_selection_cache:
            return copy.deepcopy(prefix_selection_cache[selection_key])
        inner_folds = _folds(
            outer_train.height, 5, project.data.split.seed + 10_000 + outer_fold
        )
        contexts: list[
            tuple[pl.DataFrame, pl.DataFrame, RateModel, np.ndarray, np.ndarray]
        ] = []
        for val_idx in inner_folds:
            train_idx = np.setdiff1d(
                np.arange(outer_train.height), val_idx, assume_unique=True
            )
            fit_frame = outer_train[train_idx]
            val_frame = outer_train[val_idx]
            y, weight = _response(model_config, val_frame)
            contexts.append((fit_frame, val_frame, main_for(fit_frame), y, weight))
        power = model_config.tweedie_power if model_config.family == "tweedie" else 1.0

        def evaluate_prefix(
            params: tuple[PairCandidateConfig | None, ...],
            number: int,
            total: int,
        ) -> SearchTrial:
            folds: list[FoldLoss] = []
            for inner_fold, (fit_frame, val_frame, base, y, weight) in enumerate(
                contexts
            ):
                scorer = fixed_prefix(fit_frame, params, base)
                loss = _deviance_sum(
                    y, scorer.predict(val_frame, exposure_col=None), weight, power
                )
                folds.append(
                    FoldLoss(inner_fold, loss, float(weight.sum()), loss, None)
                )
                if progress:
                    label = (
                        "no-correction check"
                        if number < 0
                        else f"prefix trial {number + 1}/{total}"
                    )
                    progress(
                        f"Pair stage {len(previous) + 1}/{len(stages)}: "
                        f"fold {outer_fold + 1}/5, {label}, "
                        f"inner {inner_fold + 1}/5"
                    )
            value = sum(item.loss_sum for item in folds) / sum(
                item.weight_sum for item in folds
            )
            return SearchTrial(
                number, "COMPLETE", params, tuple(folds), value, outer_fold
            )

        outcome: tuple[
            tuple[PairCandidateConfig | None, ...],
            tuple[int, ...],
            tuple[SearchTrial, ...],
        ]
        if adaptive:
            trial_budget = (
                current.search.prefix_trials
                if current.search is not None
                else max(
                    item.search.prefix_trials
                    for item in previous
                    if item.search is not None
                )
            )
            optuna = _optuna_module()
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(
                    seed=current.seed + outer_fold,
                    n_startup_trials=PREFIX_TPE_STARTUP_TRIALS,
                    multivariate=False,
                ),
                pruner=optuna.pruners.NopPruner(),
            )
            study.enqueue_trial(_safe_prefix_params(previous))
            trials = [evaluate_prefix((None,) * len(previous), -1, trial_budget)]

            def objective(trial: Any) -> float:
                params = _suggest_prefix(trial, previous)
                record = evaluate_prefix(params, trial.number, trial_budget)
                trials.append(record)
                return record.value

            study.optimize(objective, n_trials=trial_budget, n_jobs=1)
            best_loss = min(record.value for record in trials)
            close = [
                record
                for record in trials
                if record.value <= best_loss + CV_TIE_TOLERANCE * max(1, abs(best_loss))
            ]
            selected = min(
                close,
                key=lambda item: (
                    sum(candidate is not None for candidate in item.candidates),
                    sum(
                        candidate.depth
                        for candidate in item.candidates
                        if candidate is not None
                    ),
                    sum(
                        candidate.iterations
                        for candidate in item.candidates
                        if candidate is not None
                    ),
                    item.number,
                ),
            )
            outcome = (selected.candidates, (), tuple(trials))
        else:
            fixed_records: list[tuple[SearchTrial, tuple[int, ...]]] = []
            for number, choices in enumerate(configurations):
                params = tuple(
                    _candidate_options(item)[choice]
                    for item, choice in zip(previous, choices, strict=True)
                )
                fixed_records.append(
                    (evaluate_prefix(params, number, len(configurations)), choices)
                )
            best_loss = min(record.value for record, _ in fixed_records)
            close_fixed = [
                item
                for item in fixed_records
                if item[0].value
                <= best_loss + CV_TIE_TOLERANCE * max(1, abs(best_loss))
            ]
            selected, choices = min(
                close_fixed,
                key=lambda item: (
                    sum(choice != 0 for choice in item[1]),
                    sum(item[1]),
                    item[1],
                ),
            )
            outcome = (selected.candidates, choices, ())
        prefix_selection_cache[selection_key] = copy.deepcopy(outcome)
        return outcome

    scorer = copy.deepcopy(main_rate_model)
    artifacts: list[PairStageArtifact] = []
    outer_folds = _folds(train.height, 5, project.data.split.seed)
    power = model_config.tweedie_power if model_config.family == "tweedie" else 1.0
    for stage_index, stage in enumerate(stages):
        started = time.perf_counter()
        fingerprint = _fingerprint(
            project, model_config, train, scorer, stage, stages[:stage_index]
        )
        if fingerprint in full_cache:
            artifact = copy.deepcopy(full_cache[fingerprint])
            artifact.reused = True
            scorer = _append_table(scorer, artifact.table)
            _apply_stage_edits(project, model_config, stage, scorer, fold_local=False)
            artifacts.append(artifact)
            if progress:
                progress(
                    f"Pair stage {stage_index + 1}/{len(stages)}: reused full-training prefix"
                )
            continue
        if not replay_pair_adjustments and any(
            adj.stage_id == stage.stage_id for adj in model_config.adjustments
        ):
            raise ValueError(
                f"Refitting pair stage {stage.stage_id!r} replaces its manual cell edits. "
                "Clear those edits after reviewing them, then refit; edits on earlier "
                "frozen stages remain in the baseline"
            )
        chosen_prefix_configs: list[tuple[int, ...]] = []
        chosen_prefix_parameters: list[tuple[PairCandidateConfig | None, ...]] = []
        prefix_trials: list[SearchTrial] = []
        outer_contexts: list[
            tuple[
                pl.DataFrame,
                pl.DataFrame,
                RateModel,
                tuple[VariableConfig, VariableConfig],
                np.ndarray,
                np.ndarray,
                np.ndarray,
                float,
            ]
        ] = []
        for fold_index, val_idx in enumerate(outer_folds):
            train_idx = np.setdiff1d(
                np.arange(train.height), val_idx, assume_unique=True
            )
            fit_frame = train[train_idx]
            val_frame = train[val_idx]
            params, choices, trial_records = select_prefix(
                fit_frame, stages[:stage_index], fold_index
            )
            chosen_prefix_configs.append(choices)
            chosen_prefix_parameters.append(params)
            prefix_trials.extend(trial_records)
            fold_base = main_for(fit_frame)
            fold_prefix = fixed_prefix(fit_frame, params, fold_base)
            y_val, w_val = _response(model_config, val_frame)
            prefix_pred = fold_prefix.predict(val_frame, exposure_col=None)
            prefix_sum = _deviance_sum(y_val, prefix_pred, w_val, power)
            axes = (
                _axis(project, fit_frame, stage.a, model_config.weight),
                _axis(project, fit_frame, stage.b, model_config.weight),
            )
            outer_contexts.append(
                (
                    fit_frame,
                    val_frame,
                    fold_prefix,
                    axes,
                    y_val,
                    w_val,
                    prefix_pred,
                    prefix_sum,
                )
            )

        stage_contexts = tuple(outer_contexts)

        def evaluate_stage_candidate(
            candidate: PairCandidateConfig | None,
            number: int,
            total: int,
            stage_spec: PairStageConfig = stage,
            contexts: tuple[Any, ...] = stage_contexts,
            current_stage_index: int = stage_index,
        ) -> CandidateCV:
            records: list[FoldLoss] = []
            for fold_index, (
                fit_frame,
                val_frame,
                fold_prefix,
                axes,
                y_val,
                w_val,
                prefix_pred,
                prefix_sum,
            ) in enumerate(contexts):
                table, teacher = _fit_table(
                    project,
                    model_config,
                    stage_spec,
                    candidate,
                    fit_frame,
                    fold_prefix,
                    axes=axes,
                    deadline=deadline,
                )
                deployed = _append_table(fold_prefix, table)
                table_pred = deployed.predict(val_frame, exposure_col=None)
                teacher_sum = None
                approximation_sum = None
                if teacher is not None:
                    raw_val, _ = _raw_features(
                        val_frame, (stage_spec.a, stage_spec.b), axes
                    )
                    teacher_mean = teacher.predict_mean(raw_val, prefix_pred)
                    teacher_sum = _deviance_sum(y_val, teacher_mean, w_val, power)
                    approximation_sum = _deviance_sum(
                        teacher_mean, table_pred, w_val, power
                    )
                records.append(
                    FoldLoss(
                        fold_index,
                        _deviance_sum(y_val, table_pred, w_val, power),
                        float(w_val.sum()),
                        prefix_sum,
                        teacher_sum,
                        teacher.target_scale if teacher is not None else None,
                        approximation_sum,
                    )
                )
                if progress:
                    label = (
                        "no-correction check"
                        if number < 0
                        else f"trial {number + 1}/{total}"
                    )
                    progress(
                        f"Pair stage {current_stage_index + 1}/{len(stages)}: "
                        f"fold {fold_index + 1}/5, {label}"
                    )
            return CandidateCV(candidate, tuple(records))

        search_records: list[SearchTrial] = []
        if stage.search is None:
            options = _candidate_options(stage)
            candidates_cv = tuple(
                evaluate_stage_candidate(candidate, index, len(options))
                for index, candidate in enumerate(options)
            )
        else:
            optuna = _optuna_module()
            search_config = stage.search
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(
                    seed=stage.seed,
                    n_startup_trials=MAIN_TPE_STARTUP_TRIALS,
                    multivariate=False,
                ),
                pruner=optuna.pruners.NopPruner(),
            )
            evaluated = [evaluate_stage_candidate(None, -1, search_config.trials)]

            def objective(
                trial: Any, trial_budget: int = search_config.trials
            ) -> float:
                candidate = _suggest_candidate(trial, "")
                result = evaluate_stage_candidate(candidate, trial.number, trial_budget)
                evaluated.append(result)  # noqa: B023 - study completes in this loop
                search_records.append(  # noqa: B023 - study completes in this loop
                    SearchTrial(
                        trial.number,
                        "COMPLETE",
                        (candidate,),
                        result.folds,
                        result.table_loss,
                    )
                )
                return result.table_loss

            study.optimize(objective, n_trials=stage.search.trials, n_jobs=1)
            candidates_cv = tuple(evaluated)
        best = min(item.table_loss for item in candidates_cv)
        close = [
            item
            for item in candidates_cv
            if item.table_loss <= best + CV_TIE_TOLERANCE * max(1, abs(best))
        ]
        selected = min(
            close,
            key=lambda item: (
                item.candidate is not None,
                item.candidate.depth if item.candidate else 0,
                item.candidate.iterations if item.candidate else 0,
            ),
        )
        # Full-data fit sees the frozen full-training prefix; no earlier teacher
        # is refitted when a stage is appended.
        table, teacher = _fit_table(
            project,
            model_config,
            stage,
            selected.candidate,
            train,
            scorer,
            deadline=deadline,
        )
        table.provenance = {
            "input_prefix_fingerprint": fingerprint,
            "chosen_candidate": (
                asdict(selected.candidate) if selected.candidate is not None else None
            ),
            "algorithm_version": ALGORITHM_VERSION,
            "search_space_version": SEARCH_SPACE_VERSION,
            "search": asdict(stage.search) if stage.search is not None else None,
            "dependency_digest": _dependency_bytes().hex(),
            "training_rows": train.height,
            "target_scale": teacher.target_scale if teacher is not None else 1.0,
        }
        next_scorer = _append_table(scorer, table)
        y_train, w_train = _response(model_config, train)
        teacher_train_loss = None
        if teacher is not None:
            raw_train, _ = _raw_features(train, (stage.a, stage.b), table.axes)
            teacher_mean = teacher.predict_mean(
                raw_train, scorer.predict(train, exposure_col=None)
            )
            teacher_train_loss = _deviance_sum(
                y_train, teacher_mean, w_train, power
            ) / float(w_train.sum())
        table_train_loss = _deviance_sum(
            y_train, next_scorer.predict(train, exposure_col=None), w_train, power
        ) / float(w_train.sum())
        artifact = PairStageArtifact(
            stage_id=stage.stage_id,
            parents=(stage.a, stage.b),
            table=table,
            chosen_candidate=selected.candidate,
            prefix_fingerprint=fingerprint,
            cv_candidates=candidates_cv,
            prefix_cv_loss=selected.prefix_loss,
            table_cv_loss=selected.table_loss,
            teacher_cv_loss=selected.teacher_loss,
            approximation_loss=selected.approximation_loss,
            observed_table_minus_teacher_cv_loss=(
                selected.table_loss - selected.teacher_loss
                if selected.teacher_loss is not None
                else None
            ),
            training_teacher_loss=teacher_train_loss,
            training_table_loss=table_train_loss,
            target_scale=teacher.target_scale if teacher is not None else 1.0,
            fit_seconds=time.perf_counter() - started,
            status="no_improvement" if selected.candidate is None else "up_to_date",
            baseline_stage_ids=tuple(
                previous.stage_id for previous in stages[:stage_index]
            ),
            selected_prefix_configs=tuple(chosen_prefix_configs),
            selected_prefix_parameters=tuple(chosen_prefix_parameters),
            search_trials=tuple(search_records),
            prefix_search_trials=tuple(prefix_trials),
        )
        full_cache[fingerprint] = copy.deepcopy(artifact)
        scorer = next_scorer
        _apply_stage_edits(project, model_config, stage, scorer, fold_local=False)
        artifacts.append(artifact)
    return scorer, artifacts
