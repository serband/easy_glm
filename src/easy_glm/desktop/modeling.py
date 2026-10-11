"""Model setup adapter; the modelling algorithms remain in easy_glm.workflow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from easy_glm.workflow.prep import add_split_column, apply_variables, train_holdout
from easy_glm.workflow.project import (
    FAMILIES,
    Interaction,
    ModelConfig,
    PairCandidateConfig,
    PairSearchConfig,
    PairStageConfig,
    Project,
    VariableDesign,
    premium_offset_column,
    validate_model_name,
)
from easy_glm.workflow.stack_types import ComponentRecipe


class Revision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    revision: int = Field(ge=0)


class FitRequest(Revision):
    #: Refit this saved branch from its complete recipe.  This is deliberately
    #: distinct from the guarded in-session frozen-prefix optimisation.
    full_refit: bool = False


class ModelEdit(Revision):
    name: str
    create: bool = False
    source_name: str | None = None
    source_fit_id: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    designs: dict[
        str,
        Literal["step", "linear", "continuous", "categorical", "aglm", "aglm_l"] | None,
    ] = Field(default_factory=dict)
    n_bins: int | None = Field(default=None, ge=2, le=200)
    min_level_share: float | None = Field(default=None, ge=0, lt=1)


class ReducedModelEdit(Revision):
    name: str
    fit_id: str
    predictors: list[str]


class SplitEdit(Revision):
    mode: Literal["random", "column"]
    column: str = Field(min_length=1)
    train_value: str | int | float | bool | None = 1
    holdout_value: str | int | float | bool | None = None
    fraction: float = Field(default=0.7, gt=0, lt=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


def pair_candidate_groups(
    project: Project, raw: pl.DataFrame, prepared_columns: set[str]
) -> dict[str, list[str]]:
    """Eligible sequential-pair parents, grouped without changing main roles.

    Candidate identity starts from the source schema and follows the saved
    rename.  Derived columns and generated split/offset columns therefore do
    not become pair inputs merely because they happen to exist after prep.
    """
    groups: dict[str, list[str]] = {"predictors": [], "unassigned": []}
    split_column = project.data.split.column
    premium = project.current_premium
    premium_offset = premium_offset_column(premium) if premium is not None else None
    for source in raw.columns:
        name = project.data.renames.get(source, source)
        if (
            name not in prepared_columns
            or name == split_column
            or name == premium_offset
        ):
            continue
        role = project.data.roles.get(name)
        if role == "predictor":
            groups["predictors"].append(name)
        elif role is None:
            groups["unassigned"].append(name)
    return groups


def stack_factor_stages(
    project: Project,
    raw: pl.DataFrame,
    prepared_columns: set[str],
    config: ModelConfig,
) -> list[list[str]]:
    """Canonical factor scope before every component in a stack.

    Only raw predictor/unassigned columns enter at the root.  Model controls,
    protected project roles, derived columns and generated premium/split fields
    cannot leak in through a hand-edited component recipe.  A catalogue's key
    is an input at its own stage; its pinned attributes become available only
    to following components.
    """
    groups = pair_candidate_groups(project, raw, prepared_columns)
    protected = _stack_protected_columns(project, config)
    available = {
        name
        for name in [*groups["predictors"], *groups["unassigned"]]
        if name not in protected
    }
    stages: list[list[str]] = []
    for component in config.components:
        stages.append(sorted(available, key=str.casefold))
        if component.kind == "catalogue":
            available.update(set(component.factors) - protected)
    stages.append(sorted(available, key=str.casefold))
    return stages


def _stack_protected_columns(project: Project, config: ModelConfig) -> set[str]:
    """All columns that cannot become raw or lookup-derived stack factors."""
    protected_roles = {
        name
        for name, role in project.data.roles.items()
        if role not in (None, "predictor")
    }
    protected = protected_roles | {
        value
        for value in (
            config.target,
            config.weight,
            config.offset,
            project.data.split.column,
        )
        if isinstance(value, str)
    }
    premium = project.current_premium
    if premium is not None:
        protected.update((premium, premium_offset_column(premium)))
    protected.update(item.name for item in project.data.derived)
    return protected


def validate_stack_factor_scope(
    project: Project,
    raw: pl.DataFrame,
    prepared_columns: set[str],
    config: ModelConfig,
    *,
    start_index: int = 0,
) -> None:
    """Refuse unsafe/new stack factors while allowing repeated corrections."""
    stages = stack_factor_stages(project, raw, prepared_columns, config)
    protected = _stack_protected_columns(project, config)
    catalogue_id_keys = {
        project.data.renames.get(source, source)
        for source in raw.columns
        if project.data.roles.get(project.data.renames.get(source, source)) == "id"
        and project.data.renames.get(source, source) in prepared_columns
    }
    for index, component in enumerate(config.components):
        if index < start_index:
            continue
        available = set(stages[index])
        if component.kind == "catalogue":
            key = component.settings.get("key")
            if not isinstance(key, str) or key not in available | catalogue_id_keys:
                raise ValueError(
                    f"Catalogue component {component.name!r} needs an eligible source key or ID."
                )
            if not component.factors or not all(
                isinstance(value, str) and value.strip() for value in component.factors
            ):
                raise ValueError(
                    f"Catalogue component {component.name!r} needs pinned attribute names."
                )
            collisions = sorted(set(component.factors) & protected)
            if collisions:
                raise ValueError(
                    f"Catalogue component {component.name!r} cannot introduce protected columns: {collisions}"
                )
            continue
        if component.kind == "legacy":
            continue
        invalid = [factor for factor in component.factors if factor not in available]
        if invalid:
            raise ValueError(
                f"Component {component.name!r} uses factors unavailable at this stage: {invalid}"
            )


def setup_info(project: Project, raw: pl.DataFrame) -> dict[str, Any]:
    """Readiness uses prepared data, including a generated random split."""
    problems: list[str] = []
    columns: list[dict[str, Any]] = []
    prepared_columns: set[str] = set()
    counts = {"train": 0, "holdout": 0}
    try:
        prepared = apply_variables(raw, project.data)
        prepared_columns = set(prepared.columns)
        columns = [
            {
                "name": name,
                "dtype": str(dtype),
                "numeric": dtype.is_numeric() or dtype == pl.Boolean,
            }
            for name, dtype in prepared.schema.items()
        ]
        frame = add_split_column(prepared, project.data.split)
        train, holdout = train_holdout(frame, project.data.split)
        counts = {"train": train.height, "holdout": holdout.height}
        if not train.height:
            problems.append("The split has no training rows.")
        if not holdout.height:
            problems.append(
                "The split has no holdout rows. Change its settings to evaluate on holdout data."
            )
    except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError) as exc:
        problems.append(str(exc))
    if not project.target:
        problems.append("Assign a target role in Project & data and apply it.")
    if not project.predictors and not any(
        config.pair_stages or getattr(config, "components", ())
        for config in project.models.values()
    ):
        problems.append(
            "Assign at least one predictor role in Project & data and apply it."
        )
    pair_search_preflight = {
        name: [
            {
                "stage_id": stage.stage_id,
                "mode": "automatic" if stage.search is not None else "fixed",
                "folds": stage.cv_folds,
                "trials": stage.search.trials if stage.search is not None else None,
                "prefix_trials": (
                    stage.search.prefix_trials if stage.search is not None else None
                ),
                "prefix_stages": index,
                "fixed_candidates": (
                    len(stage.candidates) if stage.search is None else None
                ),
            }
            for index, stage in enumerate(config.pair_stages)
        ]
        for name, config in project.models.items()
        if config.pair_stages
    }
    eligible_groups = pair_candidate_groups(project, raw, prepared_columns)
    base_stack_factors = stack_factor_stages(
        project,
        raw,
        prepared_columns,
        ModelConfig(
            target=project.target,
            weight=project.weight,
            offset=project.offset_column,
        ),
    )[0]
    stack_candidates: dict[str, list[list[str]]] = {}
    for model_name, config in project.models.items():
        stack_candidates[model_name] = stack_factor_stages(
            project, raw, prepared_columns, config
        )
    return {
        "columns": columns,
        "split": asdict(project.data.split),
        "counts": counts,
        "problems": problems,
        "models": project.to_dict()["models"],
        "predictors": project.predictors,
        "pair_candidates": eligible_groups,
        "stack_candidates": stack_candidates,
        "stack_base_candidates": base_stack_factors,
        "target": project.target,
        "weight": project.weight,
        "offset": project.offset_column,
        "design": project.to_dict()["design"],
        "families": list(FAMILIES),
        "pair_stage_defaults": {
            "min_weight_share": 0.001,
            "seed": 42,
            "cv_folds": 5,
            "search": asdict(PairSearchConfig()),
            "search_limits": {"trials": [1, 16], "prefix_trials": [1, 8]},
            "candidates": [
                {
                    "depth": 2,
                    "iterations": 60,
                    "learning_rate": 0.08,
                    "l2_leaf_reg": 3.0,
                },
                {
                    "depth": 3,
                    "iterations": 120,
                    "learning_rate": 0.06,
                    "l2_leaf_reg": 5.0,
                },
            ],
        },
        "pair_search_preflight": pair_search_preflight,
    }


def _edit_interactions(cfg: ModelConfig, values: Any) -> None:
    """Replace the pair list while preserving settings not exposed by the editor."""
    if not isinstance(values, list):
        raise ValueError("Interactions must be a list of variable pairs.")
    existing = {frozenset((item.a, item.b)): item for item in cfg.interactions}
    interactions: list[Interaction] = []
    seen: set[frozenset[str]] = set()
    for value in values:
        if not isinstance(value, dict) or set(value) - {
            "a",
            "b",
            "min_cell_exposure",
            "penalty_weight",
            "alpha",
        }:
            raise ValueError(
                "Each interaction needs a, b, min_cell_exposure and penalty_weight."
            )
        if any(
            not isinstance(value.get(parent), str) or not value[parent].strip()
            for parent in ("a", "b")
        ):
            raise ValueError("Each interaction needs two column names.")
        pair = frozenset((value["a"], value["b"]))
        if len(pair) != 2:
            raise ValueError("Choose two different variables for each interaction.")
        if pair in seen:
            raise ValueError(
                f"Interaction {value['a']} × {value['b']} is listed twice."
            )
        seen.add(pair)
        original = existing.get(pair)
        # Keep the original orientation: cell edits use the ordered pair name
        # and carry the first and second parent's band coordinates separately.
        item = deepcopy(original) if original else Interaction(value["a"], value["b"])
        if "alpha" in value and value["alpha"] != item.alpha:
            raise ValueError(
                "Legacy interaction alpha cannot be changed here; use penalty weight."
            )
        for field in ("min_cell_exposure", "penalty_weight"):
            if field in value:
                setattr(item, field, value[field])
        interactions.append(item)
    for pair, original in existing.items():
        if pair not in seen:
            cfg.drop_adjustments_for(original.name)
    cfg.interactions = interactions


def _edit_pair_stages(cfg: ModelConfig, values: Any) -> None:
    """Replace the ordered pair recipe, retaining stable stage identities."""
    if not isinstance(values, list):
        raise ValueError("Pair stages must be an ordered list.")
    allowed = {
        "stage_id",
        "a",
        "b",
        "candidates",
        "search",
        "min_weight_share",
        "seed",
        "cv_folds",
    }
    candidate_fields = {"depth", "iterations", "learning_rate", "l2_leaf_reg"}
    stages: list[PairStageConfig] = []
    for value in values:
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError(
                "Each pair stage needs an ID, two parents and candidate settings."
            )
        if not all(
            isinstance(value.get(key), str) and value[key].strip()
            for key in ("stage_id", "a", "b")
        ):
            raise ValueError("Each pair stage needs an ID and two column names.")
        candidates = value.get("candidates", [])
        if not isinstance(candidates, list):
            raise ValueError("Pair stage candidates must be a list.")
        parsed = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or set(candidate) != candidate_fields:
                raise ValueError(
                    "A candidate needs depth, iterations, learning_rate and l2_leaf_reg."
                )
            parsed.append(PairCandidateConfig(**candidate))
        raw_search = value.get("search")
        if raw_search is None:
            search = None
        elif not isinstance(raw_search, dict) or set(raw_search) - {
            "method",
            "trials",
            "prefix_trials",
        }:
            raise ValueError("Pair search needs only method, trials and prefix_trials.")
        else:
            search = PairSearchConfig(**raw_search)
        stages.append(
            PairStageConfig(
                stage_id=value["stage_id"],
                a=value["a"],
                b=value["b"],
                candidates=parsed,
                min_weight_share=value.get("min_weight_share", 0.001),
                seed=value.get("seed", 42),
                cv_folds=value.get("cv_folds", 5),
                search=search,
            )
        )
    removed = {stage.stage_id for stage in cfg.pair_stages} - {
        stage.stage_id for stage in stages
    }
    for stage_id in removed:
        cfg.drop_adjustments_for(stage_id)
    cfg.pair_stages = stages


def edit_model(project: Project, edit: ModelEdit) -> Project:
    candidate = deepcopy(project)
    if edit.create:
        problem = validate_model_name(edit.name, candidate.models)
        if problem:
            raise ValueError(problem)
        candidate.models[edit.name] = ModelConfig(
            target=candidate.target,
            weight=candidate.weight,
            offset=candidate.offset_column,
        )
    if edit.name not in candidate.models:
        raise ValueError("Choose an existing model or create a new one.")
    cfg = candidate.models[edit.name]
    if ("components" in edit.fields or cfg.components) and (
        edit.designs or edit.n_bins is not None or edit.min_level_share is not None
    ):
        raise ValueError(
            "Stack model definitions carry their own variables and bands; "
            "change them on the model component instead."
        )
    allowed = {
        "family",
        "link",
        "target",
        "weight",
        "offset",
        "divide_target_by_weight",
        "predictors",
        "interactions",
        "pair_method",
        "pair_time_limit_minutes",
        "pair_stages",
        "components",
        "penalty",
        "tweedie_power",
        "base",
    }
    if set(edit.fields) - allowed:
        raise ValueError(
            "Unsupported model fields: " + ", ".join(sorted(set(edit.fields) - allowed))
        )
    for key, value in edit.fields.items():
        if key == "interactions":
            _edit_interactions(cfg, value)
        elif key == "pair_stages":
            _edit_pair_stages(cfg, value)
        elif key == "components":
            if not isinstance(value, list) or not all(
                isinstance(item, dict) for item in value
            ):
                raise ValueError("Components must be an ordered list of recipes.")
            cfg.components = [ComponentRecipe.from_dict(item) for item in value]
        elif key == "penalty":
            if not isinstance(value, dict) or set(value) - {
                "alpha",
                "cv",
                "n_alphas",
                "l1_ratio",
            }:
                raise ValueError(
                    "Only alpha, cv, n_alphas and l1_ratio can be edited here."
                )
            for field, setting in value.items():
                setattr(cfg.penalty, field, setting)
        else:
            setattr(cfg, key, value)
    if (
        not isinstance(cfg.predictors, list)
        or not all(isinstance(p, str) for p in cfg.predictors)
        or len(set(cfg.predictors)) != len(cfg.predictors)
    ):
        raise ValueError("Predictors must be a list of distinct column names.")
    if cfg.link not in (None, "log", "logit"):
        raise ValueError("This rate-table workbench supports log and logit links.")
    if not isinstance(cfg.divide_target_by_weight, bool):
        raise ValueError("Divide target by weight must be true or false.")
    for value in (cfg.target, cfg.weight, cfg.offset):
        if value is not None and not isinstance(value, str):
            raise ValueError(
                "Target, weight and offset must name a column or be empty."
            )
    for name, kind in edit.designs.items():
        if name not in candidate.predictors:
            raise ValueError(f"{name!r} is not a predictor-role column.")
        existing = candidate.design.variables.get(name, VariableDesign())
        if kind in ("aglm", "aglm_l") and existing.kind not in ("aglm", "aglm_l"):
            existing.knots, existing.n_bins, existing.clamp, existing.monotone = (
                "quantile",
                None,
                None,
                None,
            )
        elif kind not in ("aglm", "aglm_l"):
            existing.breaks, existing.n_breaks, existing.extrapolation = (
                None,
                None,
                None,
            )
        existing.kind = kind
        candidate.design.variables[name] = existing
    if edit.n_bins is not None:
        candidate.design.defaults.n_bins = edit.n_bins
    if edit.min_level_share is not None:
        candidate.design.defaults.min_level_share = edit.min_level_share
    # Monotone rules and expert design fields stay intact. Interactions are
    # changed only when explicitly included; invalid parent selections are refused.
    problems = candidate.validate(edit.name)
    if problems:
        raise ValueError("; ".join(problems))
    return candidate
