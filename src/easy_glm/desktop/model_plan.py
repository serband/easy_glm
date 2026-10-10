"""Validation and materialisation of the active workbench model plan."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl
from pydantic import Field

from easy_glm.desktop.modeling import Revision, stack_factor_stages
from easy_glm.workflow.model_plan import ModelPlan, structural_problems
from easy_glm.workflow.project import (
    FAMILIES,
    ModelConfig,
    Penalty,
    Project,
    validate_model_name,
)
from easy_glm.workflow.stack_types import ComponentRecipe


class ModelPlanEdit(Revision):
    plan: dict[str, Any] | None = None
    operation: Literal["save", "start", "revise", "adopt"] = "save"
    adopt_model_name: str | None = Field(default=None, min_length=1)


@dataclass(frozen=True)
class PlanFitRef:
    plan_id: str
    stage_id: str
    model_name: str
    fit_id: str


def _plan_problems(project: Project, plan: ModelPlan) -> list[str]:
    return structural_problems(plan, FAMILIES)


def validate_model_plan(
    project: Project,
    plan: ModelPlan,
    raw: pl.DataFrame,
    prepared_columns: set[str],
    *,
    through_count: int = 0,
) -> None:
    """Validate plan shape now and fit readiness through one requested prefix.

    A future blank stage must never block fitting an earlier reviewed stage.
    """
    problems = _plan_problems(project, plan)
    if through_count < 0 or through_count > len(plan.stages):
        problems.append("Choose a declared stage to fit")
    if through_count:
        if not project.target:
            problems.append("Assign a target in Project & data before fitting a model")
        config = ModelConfig(
            family=plan.family,
            link=plan.link,
            target=project.target,
            weight=project.weight,
            offset=project.offset_column,
            divide_target_by_weight=plan.divide_target_by_weight,
            components=deepcopy(plan.stages),
            tweedie_power=plan.tweedie_power,
        )
        candidates = stack_factor_stages(project, raw, prepared_columns, config)
        for index, stage in enumerate(plan.stages[:through_count]):
            # An adopted root is an already fitted exact scorer; it has no new
            # factors to validate or refit.
            if index == 0 and plan.adopted_model_name is not None:
                continue
            allowed = set(candidates[index]) if index < len(candidates) else set()
            if stage.kind == "catalogue":
                key = stage.settings.get("key")
                id_keys = {
                    project.data.renames.get(source, source)
                    for source in raw.columns
                    if project.data.roles.get(project.data.renames.get(source, source))
                    == "id"
                    and project.data.renames.get(source, source) in prepared_columns
                }
                if not isinstance(key, str) or key not in allowed | id_keys:
                    problems.append(
                        f"{stage.name}: choose an available source key for the catalogue"
                    )
                if stage.settings.get("catalogue") is None and not stage.settings.get(
                    "source_path"
                ):
                    problems.append(
                        f"{stage.name}: upload a catalogue in Project & data before fitting"
                    )
                # Its factors are pinned attributes from the uploaded lookup,
                # introduced by this stage rather than raw columns available before it.
                continue
            invalid = sorted(set(stage.factors) - allowed)
            if invalid:
                problems.append(
                    f"{stage.name}: factors are not available for this stage: "
                    + ", ".join(invalid)
                )
            if stage.kind == "interaction" and len(stage.factors) != 2:
                problems.append(
                    f"{stage.name}: an interaction needs exactly two factors"
                )
            elif stage.kind != "catalogue" and not stage.factors:
                problems.append(f"{stage.name}: choose at least one factor")
    if problems:
        raise ValueError("; ".join(problems))


def _single_glm_config(project: Project, plan: ModelPlan, stage: Any) -> ModelConfig:
    settings = stage.settings
    alpha = settings.get("alpha")
    cv = None if alpha is not None else settings.get("cv", 5)
    penalty = Penalty(
        alpha=alpha,
        cv=cv,
        n_alphas=settings.get("n_alphas", 20),
        l1_ratio=settings.get("l1_ratio", 1.0),
        min_alpha_ratio=settings.get("min_alpha_ratio", 0.0001),
    )
    return ModelConfig(
        family=plan.family,
        link=plan.link,
        target=project.target,
        weight=project.weight,
        offset=project.offset_column,
        divide_target_by_weight=plan.divide_target_by_weight,
        tweedie_power=plan.tweedie_power,
        predictors=list(stage.factors),
        penalty=penalty,
        base=settings.get("base", "modal"),
        design=deepcopy(stage.design),
        allow_unassigned_factors=True,
    )


def materialise_plan_prefix(
    project: Project, plan: ModelPlan, count: int, name: str | None
) -> ModelConfig:
    """Turn one declared prefix into one immutable saved fit recipe."""
    if count < 1 or count > len(plan.stages):
        raise ValueError("Choose a declared stage to fit")
    if name is not None:
        problem = validate_model_name(name, project.models)
        if problem:
            raise ValueError(problem)
    if plan.adopted_model_name is not None:
        source = project.models.get(plan.adopted_model_name)
        if source is None:
            raise ValueError("The adopted saved model is unavailable")
        if count == 1:
            return deepcopy(source)
        inherited = deepcopy(source.components)
        if not inherited:
            inherited = [
                ComponentRecipe(
                    "legacy",
                    "Accepted saved scorer",
                    "legacy",
                    tuple(source.predictors),
                    {"model_name": plan.adopted_model_name},
                )
            ]
        return ModelConfig(
            family=plan.family,
            link=plan.link,
            target=project.target,
            weight=project.weight,
            offset=project.offset_column,
            divide_target_by_weight=plan.divide_target_by_weight,
            tweedie_power=plan.tweedie_power,
            components=[*inherited, *deepcopy(plan.stages[1:count])],
        )
    if plan.mode == "single" and plan.stages[0].kind == "glm":
        return _single_glm_config(project, plan, plan.stages[0])
    return ModelConfig(
        family=plan.family,
        link=plan.link,
        target=project.target,
        weight=project.weight,
        offset=project.offset_column,
        divide_target_by_weight=plan.divide_target_by_weight,
        tweedie_power=plan.tweedie_power,
        components=deepcopy(plan.stages[:count]),
    )


def plan_model_name(project: Project, plan: ModelPlan, count: int) -> str:
    base = project.name.strip() or "Model"
    suffix = plan.stages[count - 1].name.strip()
    candidate = f"{base} — {suffix}"
    if candidate not in project.models:
        return candidate
    number = 2
    while f"{candidate} {number}" in project.models:
        number += 1
    return f"{candidate} {number}"
