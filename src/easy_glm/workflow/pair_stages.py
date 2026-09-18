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
ALGORITHM_VERSION = "pair-stages-1"


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
        if len(stage.candidates) > MAX_CANDIDATES:
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
        configurations = _prefix_configs(stages[:index])
        expected_fits += 25 * sum(
            sum(choice != 0 for choice in config) for config in configurations
        )
        expected_fits += 5 * index + 5 * len(stage.candidates) + 1
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
        partition: pl.DataFrame, choices: tuple[int, ...], base: RateModel
    ) -> RateModel:
        prior = stages[: len(choices)]
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
            + json.dumps(choices).encode()
            + json.dumps(_scorer_signature(base), sort_keys=True, default=str).encode()
            + _dependency_bytes()
        ).digest()
        if key in fold_prefix_cache:
            return copy.deepcopy(fold_prefix_cache[key])
        scorer = base
        for previous, option in zip(stages[: len(choices)], choices, strict=True):
            candidate = _candidate_options(previous)[option]
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
    ) -> tuple[int, ...]:
        configurations = _prefix_configs(previous)
        if len(configurations) == 1:
            return configurations[0]
        inner_folds = _folds(
            outer_train.height, 5, project.data.split.seed + 10_000 + outer_fold
        )
        scores: list[tuple[float, tuple[int, ...]]] = []
        for configuration in configurations:
            loss_sum = 0.0
            weight_sum = 0.0
            for inner_fold, val_idx in enumerate(inner_folds):
                train_idx = np.setdiff1d(
                    np.arange(outer_train.height), val_idx, assume_unique=True
                )
                fit_frame = outer_train[train_idx]
                val_frame = outer_train[val_idx]
                base = main_for(fit_frame)
                scorer = fixed_prefix(fit_frame, configuration, base)
                y, weight = _response(model_config, val_frame)
                power = (
                    model_config.tweedie_power
                    if model_config.family == "tweedie"
                    else 1.0
                )
                loss_sum += _deviance_sum(
                    y, scorer.predict(val_frame, exposure_col=None), weight, power
                )
                weight_sum += float(weight.sum())
                if progress:
                    progress(
                        f"Pair stage {len(previous) + 1}/{len(stages)}: "
                        f"fold {outer_fold + 1}/5 prefix selection, "
                        f"inner {inner_fold + 1}/5"
                    )
            scores.append((loss_sum / weight_sum, configuration))
        best_loss = min(score for score, _ in scores)
        close = [
            item
            for item in scores
            if item[0] <= best_loss + CV_TIE_TOLERANCE * max(1, abs(best_loss))
        ]
        return min(
            close,
            key=lambda item: (
                sum(choice != 0 for choice in item[1]),
                sum(item[1]),
                item[1],
            ),
        )[1]

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
        options = _candidate_options(stage)
        fold_records: list[list[FoldLoss]] = [[] for _ in options]
        chosen_prefix_configs: list[tuple[int, ...]] = []
        for fold_index, val_idx in enumerate(outer_folds):
            train_idx = np.setdiff1d(
                np.arange(train.height), val_idx, assume_unique=True
            )
            fit_frame = train[train_idx]
            val_frame = train[val_idx]
            prefix_choices = select_prefix(fit_frame, stages[:stage_index], fold_index)
            chosen_prefix_configs.append(prefix_choices)
            fold_base = main_for(fit_frame)
            fold_prefix = fixed_prefix(fit_frame, prefix_choices, fold_base)
            y_val, w_val = _response(model_config, val_frame)
            prefix_pred = fold_prefix.predict(val_frame, exposure_col=None)
            prefix_sum = _deviance_sum(y_val, prefix_pred, w_val, power)
            weight_sum = float(w_val.sum())
            for candidate_index, candidate in enumerate(options):
                axes = (
                    _axis(project, fit_frame, stage.a, model_config.weight),
                    _axis(project, fit_frame, stage.b, model_config.weight),
                )
                table, teacher = _fit_table(
                    project,
                    model_config,
                    stage,
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
                    raw_val, _ = _raw_features(val_frame, (stage.a, stage.b), axes)
                    teacher_mean = teacher.predict_mean(raw_val, prefix_pred)
                    teacher_sum = _deviance_sum(y_val, teacher_mean, w_val, power)
                    approximation_sum = _deviance_sum(
                        teacher_mean, table_pred, w_val, power
                    )
                fold_records[candidate_index].append(
                    FoldLoss(
                        fold_index,
                        _deviance_sum(y_val, table_pred, w_val, power),
                        weight_sum,
                        prefix_sum,
                        teacher_sum,
                        teacher.target_scale if teacher is not None else None,
                        approximation_sum,
                    )
                )
                if progress:
                    progress(
                        f"Pair stage {stage_index + 1}/{len(stages)}: fold {fold_index + 1}/5, candidate {candidate_index + 1}/{len(options)}"
                    )
        candidates_cv = tuple(
            CandidateCV(candidate, tuple(records))
            for candidate, records in zip(options, fold_records, strict=True)
        )
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
        )
        full_cache[fingerprint] = copy.deepcopy(artifact)
        scorer = next_scorer
        _apply_stage_edits(project, model_config, stage, scorer, fold_local=False)
        artifacts.append(artifact)
    return scorer, artifacts
