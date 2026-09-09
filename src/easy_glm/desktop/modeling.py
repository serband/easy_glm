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
    Project,
    VariableDesign,
    validate_model_name,
)


class Revision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    revision: int = Field(ge=0)


class ModelEdit(Revision):
    name: str
    create: bool = False
    fields: dict[str, Any] = Field(default_factory=dict)
    designs: dict[
        str, Literal["step", "linear", "continuous", "categorical"] | None
    ] = Field(default_factory=dict)
    n_bins: int | None = Field(default=None, ge=2, le=200)
    min_level_share: float | None = Field(default=None, ge=0, lt=1)


class SplitEdit(Revision):
    mode: Literal["random", "column"]
    column: str = Field(min_length=1)
    train_value: str | int | float | bool = 1
    fraction: float = Field(default=0.7, gt=0, lt=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


def setup_info(project: Project, raw: pl.DataFrame) -> dict[str, Any]:
    """Readiness uses prepared data, including a generated random split."""
    problems: list[str] = []
    columns: list[dict[str, Any]] = []
    counts = {"train": 0, "holdout": 0}
    try:
        prepared = apply_variables(raw, project.data)
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
        problems.append("Assign a target role on Variables and apply it.")
    if not project.predictors:
        problems.append("Assign at least one predictor role on Variables and apply it.")
    return {
        "columns": columns,
        "split": asdict(project.data.split),
        "counts": counts,
        "problems": problems,
        "models": project.to_dict()["models"],
        "predictors": project.predictors,
        "target": project.target,
        "weight": project.weight,
        "offset": project.offset_column,
        "design": project.to_dict()["design"],
        "families": list(FAMILIES),
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
    allowed = {
        "family",
        "link",
        "target",
        "weight",
        "offset",
        "divide_target_by_weight",
        "predictors",
        "interactions",
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
