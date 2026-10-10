"""An editable, ordered recipe for the workbench's current model chain.

``Project.models`` remains immutable history.  ``ModelPlan`` is the one
working chain a user is assembling; its opaque ID binds its exact checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from .stack_types import ComponentRecipe, validate_catboost_tuning_settings

PlanMode = Literal["single", "sequential"]


def new_plan_id() -> str:
    return uuid4().hex


@dataclass
class ModelPlan:
    """Current ordered workbench recipe, separate from immutable fit history."""

    plan_id: str = field(default_factory=new_plan_id)
    mode: PlanMode = "single"
    family: str = "poisson"
    link: str | None = None
    tweedie_power: float = 1.5
    divide_target_by_weight: bool = False
    stages: list[ComponentRecipe] = field(default_factory=list)
    #: Server-created reference used only when explicitly adopting one saved root.
    adopted_model_name: str | None = None
    #: Server-issued immutable checkpoint identities for this plan.
    fit_refs: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "mode": self.mode,
            "family": self.family,
            "link": self.link,
            "tweedie_power": self.tweedie_power,
            "divide_target_by_weight": self.divide_target_by_weight,
            "stages": [stage.to_dict() for stage in self.stages],
            "adopted_model_name": self.adopted_model_name,
            "fit_refs": self.fit_refs,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelPlan:
        if not isinstance(value, dict):
            raise ValueError("model_plan must be an object")
        allowed = {
            "plan_id",
            "mode",
            "family",
            "link",
            "tweedie_power",
            "divide_target_by_weight",
            "stages",
            "adopted_model_name",
            "fit_refs",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError("model_plan has unknown fields: " + ", ".join(unknown))
        for key in ("mode", "family"):
            if key in value and not isinstance(value[key], str):
                raise ValueError(f"model_plan.{key} must be a string")
        if value.get("link") is not None and not isinstance(value["link"], str):
            raise ValueError("model_plan.link must be a string or null")
        stages = value.get("stages", [])
        if not isinstance(stages, list) or not all(
            isinstance(item, dict) for item in stages
        ):
            raise ValueError("model_plan.stages must be a list of stage objects")
        refs = value.get("fit_refs", [])
        if not isinstance(refs, list) or not all(
            isinstance(item, dict) for item in refs
        ):
            raise ValueError("model_plan.fit_refs must be a list of objects")
        if not all(
            all(
                isinstance(item.get(key), str) and item[key]
                for key in ("stage_id", "model_name", "fit_id")
            )
            for item in refs
        ):
            raise ValueError("model_plan.fit_refs entries are invalid")
        plan_id = value.get("plan_id") or new_plan_id()
        if not isinstance(plan_id, str) or not plan_id.strip():
            raise ValueError("model_plan.plan_id must be a non-empty string")
        adopted = value.get("adopted_model_name")
        if adopted is not None and (
            not isinstance(adopted, str) or not adopted.strip()
        ):
            raise ValueError("model_plan.adopted_model_name must be a non-empty string")
        return cls(
            plan_id=plan_id,
            mode=value.get("mode", "single"),
            family=value.get("family", "poisson"),
            link=value.get("link"),
            tweedie_power=value.get("tweedie_power", 1.5),
            divide_target_by_weight=value.get("divide_target_by_weight", False),
            stages=[ComponentRecipe.from_dict(item) for item in stages],
            adopted_model_name=adopted,
            fit_refs=[dict(item) for item in refs],
        )


def structural_problems(
    plan: ModelPlan, families: set[str] | tuple[str, ...]
) -> list[str]:
    """Validation safe during project import; does not require data or factors."""
    problems: list[str] = []
    if plan.mode not in {"single", "sequential"}:
        problems.append("Model plan mode must be single or sequential")
    if plan.family not in families:
        problems.append("Choose a supported model family")
    if plan.link not in {None, "log", "logit"}:
        problems.append("Rate-table models require a log or logit link")
    if not isinstance(plan.divide_target_by_weight, bool):
        problems.append("Divide target by weight must be true or false")
    if plan.mode == "single" and len(plan.stages) > 1:
        problems.append("A single-model plan can contain one stage")
    if plan.mode == "sequential" and plan.family not in {
        "poisson",
        "tweedie",
        "gaussian",
    }:
        problems.append("Models in stages require Poisson, Tweedie or Gaussian")
    if plan.mode == "sequential" and plan.family == "gaussian" and plan.link != "log":
        problems.append("Gaussian models in stages require a log link")
    if (
        plan.mode == "sequential"
        and plan.family != "gaussian"
        and plan.link not in {None, "log"}
    ):
        problems.append("Models in stages require a log link")
    ids = [stage.component_id for stage in plan.stages]
    names = [stage.name.strip() for stage in plan.stages]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        problems.append("Stage IDs must be nonblank and unique")
    if any(not value for value in names) or len(set(names)) != len(names):
        problems.append("Stage names must be nonblank and unique")
    for stage in plan.stages:
        if stage.kind not in {"glm", "catboost", "interaction", "catalogue"}:
            problems.append(f"{stage.name}: unsupported model method {stage.kind!r}")
        try:
            validate_catboost_tuning_settings(stage.kind, stage.settings)
        except ValueError as exc:
            problems.append(f"{stage.name}: {exc}")
    return problems
