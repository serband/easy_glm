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
    model_id: str | None = None
    prefix_fingerprint: str | None = None


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
    logical_count = plan.logical_model_count
    if through_count < 0 or through_count > logical_count:
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
            components=deepcopy(plan.physical_stages),
            tweedie_power=plan.tweedie_power,
        )
        candidates = stack_factor_stages(project, raw, prepared_columns, config)
        physical = (
            [
                item
                for model in plan.models[:through_count]
                for item in model.physical_components
            ]
            if plan.is_logical_models_plan
            else plan.stages[:through_count]
        )
        for index, stage in enumerate(physical):
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
        link=_materialised_plan_link(plan),
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


def _materialised_plan_link(plan: ModelPlan) -> str | None:
    """Resolve the v2 Gaussian family default before creating a fit recipe.

    A direct GLM treats a missing Gaussian link as log.  A multiplicative
    stack must make that log link explicit, because an absent link there means
    identity and is deliberately refused.  V2 plans therefore use the same
    effective link whether a core is fitted directly or later extended with a
    correction.  Legacy plan serialisation keeps its historical behaviour.
    """
    if plan.is_logical_models_plan and plan.family == "gaussian" and plan.link is None:
        return "log"
    return plan.link


def materialise_plan_prefix(
    project: Project, plan: ModelPlan, count: int, name: str | None
) -> ModelConfig:
    """Turn one declared prefix into one immutable saved fit recipe."""
    if count < 1 or count > plan.logical_model_count:
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
    if plan.is_logical_models_plan:
        selected = plan.models[:count]
        components = [item for model in selected for item in model.physical_components]
        if plan.adopted_root is not None:
            # An adopted root is a frozen scorer, not an editable first stage.
            # Its physical recipe is included only so the stack configuration
            # agrees exactly with the frozen fit supplied to the worker.
            components = [*plan.adopted_root.physical_prefix, *components]
        # Retain the ordinary, exact single-GLM path until it gains an accepted
        # interaction table.  A table then needs the native stack scorer.
        if (
            len(selected) == 1
            and selected[0].core.kind == "glm"
            and not selected[0].pair_corrections
            and plan.adopted_root is None
        ):
            return _single_glm_config(project, plan, selected[0].core)
        return ModelConfig(
            family=plan.family,
            link=_materialised_plan_link(plan),
            target=project.target,
            weight=project.weight,
            offset=project.offset_column,
            divide_target_by_weight=plan.divide_target_by_weight,
            tweedie_power=plan.tweedie_power,
            components=deepcopy(components),
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


def materialise_plan_through_component(
    project: Project,
    plan: ModelPlan,
    model_index: int,
    through_component_id: str,
    name: str | None = None,
) -> ModelConfig:
    """Compile an exact v2 checkpoint ending at one owned physical component.

    Logical models may have a fitted core and several accepted pair tables.  A
    revision can therefore retain a checkpoint through the core or an earlier
    table while the remainder is explicitly waiting to be refitted.
    """
    if not plan.is_logical_models_plan:
        raise ValueError(
            "Partial checkpoints are available only for logical model plans"
        )
    if not 0 <= model_index < len(plan.models):
        raise ValueError("Choose a declared model")
    model = plan.models[model_index]
    owned = list(model.physical_components)
    position = next(
        (
            index
            for index, component in enumerate(owned)
            if component.component_id == through_component_id
        ),
        None,
    )
    if position is None:
        raise ValueError("The checkpoint does not belong to this logical model")
    if name is not None:
        problem = validate_model_name(name, project.models)
        if problem:
            raise ValueError(problem)
    components = [
        component
        for prior in plan.models[:model_index]
        for component in prior.physical_components
    ]
    if plan.adopted_root is not None:
        components = [*plan.adopted_root.physical_prefix, *components]
    components.extend(owned[: position + 1])
    if (
        model_index == 0
        and position == 0
        and model.core.kind == "glm"
        and plan.adopted_root is None
    ):
        return _single_glm_config(project, plan, model.core)
    return ModelConfig(
        family=plan.family,
        link=_materialised_plan_link(plan),
        target=project.target,
        weight=project.weight,
        offset=project.offset_column,
        divide_target_by_weight=plan.divide_target_by_weight,
        tweedie_power=plan.tweedie_power,
        components=deepcopy(components),
    )


def plan_model_name(project: Project, plan: ModelPlan, count: int) -> str:
    base = project.name.strip() or "Model"
    suffix = (
        plan.models[count - 1].name.strip()
        if plan.is_logical_models_plan
        else plan.stages[count - 1].name.strip()
    )
    candidate = f"{base} — {suffix}"
    if candidate not in project.models:
        return candidate
    number = 2
    while f"{candidate} {number}" in project.models:
        number += 1
    return f"{candidate} {number}"
