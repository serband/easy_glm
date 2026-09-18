"""Training-only, one-way shadow screening of candidate predictors."""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
from sklearn.exceptions import ConvergenceWarning

from easy_glm.core.design import (
    DesignSpec,
    StepEncoder,
    encoder_from_dict,
    quantile_knots,
)
from easy_glm.core.fit import _validate_target, fit_glm, resolve_family

from .diagnostics import permutation_importance
from .prep import prepare
from .project import Project
from .run import UnusableColumnError, build_design

Progress = Callable[[dict[str, Any]], None]
Cancelled = Callable[[], bool]
_MAX_DESIGN_BYTES = 512 * 1024 * 1024
_MAX_DESIGN_COLUMNS = 1024


def _seed(seed: int, variable: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{variable}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _finite(value: float) -> float | None:
    return value if np.isfinite(value) else None


def _empty_row(
    variable: str,
    role: str,
    status: str,
    reason: str,
    monotone: str | None = None,
) -> dict[str, Any]:
    return {
        "variable": variable,
        "role": role,
        "status": status,
        "importance": None,
        "std": None,
        "shadows": None,
        "shadow_stds": None,
        "random_importance": None,
        "random_std": None,
        "threshold": None,
        "margin": None,
        "alpha": None,
        "design_columns": None,
        "monotone": monotone,
        "reason": reason,
    }


def _check_cancelled(cancelled: Cancelled | None) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("Variable selection cancelled")


def _validate_options(
    family: str,
    link: str | None,
    tweedie_power: float,
    l1_ratio: float,
    n_alphas: int,
    repeats: int,
    seed: int,
) -> tuple[str, str]:
    if not isinstance(family, str):
        raise ValueError("family must be a supported family name.")
    allowed_links = {
        "poisson": {"log", "identity"},
        "gamma": {"log", "identity"},
        "normal": {"log", "identity"},
        "binomial": {"logit"},
        "tweedie": {"log", "identity"},
    }
    if (
        not isinstance(tweedie_power, (int, float))
        or isinstance(tweedie_power, bool)
        or not (math.isfinite(tweedie_power) and 1 < tweedie_power < 2)
    ):
        raise ValueError("tweedie_power must be a finite number in (1, 2).")
    if (
        not isinstance(l1_ratio, (int, float))
        or isinstance(l1_ratio, bool)
        or not (math.isfinite(l1_ratio) and 0 < l1_ratio <= 1)
    ):
        raise ValueError("l1_ratio must be in (0, 1].")
    if (
        isinstance(n_alphas, bool)
        or not isinstance(n_alphas, int)
        or not (2 <= n_alphas <= 100)
    ):
        raise ValueError("n_alphas must be an integer in [2, 100].")
    if (
        isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or not (1 <= repeats <= 20)
    ):
        raise ValueError("repeats must be an integer in [1, 20].")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not (0 <= seed <= 2**32 - 1)
    ):
        raise ValueError("seed must be an integer in [0, 2**32 - 1].")
    if family.strip().lower() != "tweedie" and tweedie_power != 1.5:
        raise ValueError("Tweedie power applies only to the Tweedie family.")
    power = tweedie_power if family.strip().lower() == "tweedie" else None
    _, family_name, default_link = resolve_family(family, power)
    if family_name not in allowed_links:
        raise ValueError(f"Unsupported selection family {family!r}.")
    if link is not None and (
        not isinstance(link, str) or link not in allowed_links[family_name]
    ):
        raise ValueError(f"{link!r} link is not compatible with {family_name}.")
    return family_name, link or default_link


def _validate_inputs(
    train: pl.DataFrame,
    project: Project,
    family: str,
    divide_target_by_weight: bool,
) -> None:
    target = project.target
    if target is None or target not in train.columns:
        raise ValueError("Choose a target column present in the prepared data.")
    if train.height < 5:
        raise ValueError("Variable selection needs at least five training rows for CV.")
    weight = project.weight
    if divide_target_by_weight and not weight:
        raise ValueError("divide_target_by_weight=True needs a weight column.")
    weights: np.ndarray | None = None
    if weight:
        if weight not in train.columns:
            raise ValueError(f"Weight column {weight!r} is missing.")
        weights = train[weight].cast(pl.Float64).to_numpy()
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Weights must be finite and strictly positive.")
    offset = project.offset_column
    if offset:
        if offset not in train.columns:
            raise ValueError(f"Offset column {offset!r} is missing.")
        offset_values = train[offset].cast(pl.Float64).to_numpy()
        if not np.all(np.isfinite(offset_values)):
            raise ValueError("Offset must contain only finite values.")
    y = train[target].cast(pl.Float64).to_numpy()
    if weights is not None and divide_target_by_weight:
        y = y / weights
    _validate_target(y, family)


def select_variables(
    project: Project,
    raw: pl.DataFrame,
    *,
    family: str = "poisson",
    link: str | None = None,
    tweedie_power: float = 1.5,
    divide_target_by_weight: bool = False,
    l1_ratio: float = 1.0,
    n_alphas: int = 20,
    repeats: int = 5,
    seed: int = 42,
    include_unassigned: bool = True,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> dict[str, Any]:
    """Screen each candidate against four own shuffles and uniform numeric noise.

    Each raw source column (after any configured rename) with a predictor or
    unassigned role gets its own five-fold GLM, fitted and measured on training
    rows only. Generated derived-only columns are outside the Variables draft
    and are not screened. ``signal`` means its mean permutation importance
    strictly exceeds zero and all five controls; it is a screening heuristic,
    not a significance test. Candidate errors remain visible in the rows.
    """
    family_name, selected_link = _validate_options(
        family, link, tweedie_power, l1_ratio, n_alphas, repeats, seed
    )
    _check_cancelled(cancelled)
    prepared = prepare(project, raw)
    train = prepared.filter(pl.col(project.split_column) == 1)
    _validate_inputs(train, project, family_name, divide_target_by_weight)
    protected = {
        project.target,
        project.weight,
        project.exposure,
        project.offset_column,
        project.current_premium,
        project.split_column,
        *project.columns_with_role("id"),
        *project.columns_with_role("time"),
    }
    source_columns = {project.data.renames.get(name, name) for name in raw.columns}
    candidates = [
        name
        for name in prepared.columns
        if name in source_columns
        and name not in protected
        and (
            project.data.roles.get(name) == "predictor"
            or (
                include_unassigned
                and project.data.roles.get(name) in (None, "unassigned")
            )
        )
    ]
    rows: list[dict[str, Any]] = []
    total = len(candidates)

    def report(phase: str, completed: int, message: str, current: str | None) -> None:
        if progress is not None:
            progress(
                {
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "message": message,
                    "current_variable": current,
                }
            )

    report("preparing", 0, f"Screening {total} candidates", None)
    for index, variable in enumerate(candidates):
        _check_cancelled(cancelled)
        role = project.data.roles.get(variable, "unassigned")
        variable_design = project.design.variables.get(variable)
        direction = variable_design.monotone if variable_design else None
        report("fitting", index, f"Fitting {variable}", variable)
        try:
            series = train[variable]
            if series.is_null().all() or series.n_unique() == 1:
                raise UnusableColumnError("Constant or all-null on training rows")
            real_spec = build_design(
                project, train, [variable], weight_col=project.weight
            )
            real_encoder = real_spec[variable]
            rng = np.random.default_rng(_seed(seed, variable))
            controls: list[str] = []
            additions: list[pl.Series] = []
            for shadow_index in range(4):
                name = f"__selection_shadow_{shadow_index}__"
                while name in train.columns or name in controls:
                    name += "_"
                controls.append(name)
                additions.append(
                    series.gather(rng.permutation(train.height)).alias(name)
                )
            random_name = "__selection_random__"
            while random_name in train.columns or random_name in controls:
                random_name += "_"
            random_series = pl.Series(random_name, rng.random(train.height))
            additions.append(random_series)
            screen = train.with_columns(additions)
            encoders = {variable: real_encoder}
            for name in controls:
                encoded = real_encoder.to_dict()
                encoded["variable"] = name
                encoders[name] = encoder_from_dict(encoded)
            random_knots = quantile_knots(
                random_series, n_bins=project.design.defaults.n_bins
            )
            encoders[random_name] = StepEncoder(
                random_name,
                random_knots,
                null_indicator=project.design.defaults.null_indicator,
            )
            spec = DesignSpec(encoders)
            if spec.n_features > _MAX_DESIGN_COLUMNS:
                raise MemoryError(
                    f"Design has {spec.n_features} columns "
                    f"(limit {_MAX_DESIGN_COLUMNS})"
                )
            design_bytes = spec.expected_design_bytes(screen.height)
            if design_bytes > _MAX_DESIGN_BYTES:
                raise MemoryError(
                    f"Compact design would need {design_bytes / 2**20:.0f} MiB "
                    f"(limit {_MAX_DESIGN_BYTES / 2**20:.0f} MiB)"
                )
            monotone = (
                dict.fromkeys([variable, *controls], direction) if direction else None
            )
            _check_cancelled(cancelled)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                fit = fit_glm(
                    screen,
                    spec,
                    project.target or "",
                    family=family_name,
                    tweedie_power=tweedie_power if family_name == "tweedie" else None,
                    weight_col=project.weight,
                    offset_col=project.offset_column,
                    divide_target_by_weight=divide_target_by_weight,
                    cv=5,
                    alpha=None,
                    n_alphas=n_alphas,
                    cv_seed=seed,
                    l1_ratio=l1_ratio,
                    monotone=monotone,
                    sparse=True,
                    link=selected_link,
                )
            _check_cancelled(cancelled)
            report("importance", index, f"Measuring {variable}", variable)
            importance_rows = permutation_importance(
                fit,
                screen,
                repeats=repeats,
                seed=seed,
                protected_columns=tuple(c for c in protected if c is not None),
            )
            _check_cancelled(cancelled)
            by_name = {item["variable"]: item for item in importance_rows.to_dicts()}
            values = [float(by_name[name]["importance"]) for name in controls]
            random_importance = float(by_name[random_name]["importance"])
            actual = float(by_name[variable]["importance"])
            threshold = max(0.0, *values, random_importance)
            margin = actual - threshold
            tolerance = max(1e-12, 1e-9 * abs(threshold))
            row = _empty_row(
                variable,
                role,
                "signal" if margin > tolerance else "no_signal",
                "",
                direction,
            )
            row.update(
                importance=_finite(actual),
                std=_finite(float(by_name[variable]["std"])),
                shadows=[_finite(v) for v in values],
                shadow_stds=[_finite(float(by_name[name]["std"])) for name in controls],
                random_importance=_finite(random_importance),
                random_std=_finite(float(by_name[random_name]["std"])),
                threshold=_finite(threshold),
                margin=_finite(margin),
                alpha=_finite(float(fit.alpha)),
                design_columns=spec.n_features,
            )
            rows.append(row)
        except UnusableColumnError as exc:
            rows.append(_empty_row(variable, role, "skipped", str(exc), direction))
        except InterruptedError:
            raise
        except Exception as exc:
            rows.append(
                _empty_row(
                    variable, role, "failed", f"{type(exc).__name__}: {exc}", direction
                )
            )
        report("screening", index + 1, f"Completed {variable}", variable)
        _check_cancelled(cancelled)
    report("complete", total, "Variable selection complete", None)
    _check_cancelled(cancelled)
    return {
        "method": "one_way_shadow",
        "training_rows": train.height,
        "candidate_count": total,
        "tested_count": sum(r["status"] in ("signal", "no_signal") for r in rows),
        "family": family_name,
        "link": selected_link,
        "target": project.target,
        "weight": project.weight,
        "offset": project.offset_column,
        "divide_target_by_weight": divide_target_by_weight,
        "cv_folds": 5,
        "n_alphas": n_alphas,
        "l1_ratio": l1_ratio,
        "seed": seed,
        "repeats": repeats,
        "random_definition": "independent uniform[0,1) numeric noise, default step bins",
        "decision_rule": "importance > max(0, four shadows, random) + max(1e-12, 1e-9 * abs(threshold))",
        "rows": rows,
    }
