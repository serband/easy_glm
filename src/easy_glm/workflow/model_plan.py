"""The editable workbench recipe for a single model or ordered model chain.

Version 1 represents every physical scorer as a peer stage.  Version 2 keeps
that scoring order but makes ownership explicit: a named model has one core
recipe and a GLM may own ordered interaction-table corrections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from .stack_types import ComponentRecipe, validate_catboost_tuning_settings

PlanMode = Literal["single", "sequential"]
PLAN_FORMAT_LEGACY = 1
PLAN_FORMAT_MODELS = 2


def new_plan_id() -> str:
    return uuid4().hex


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class PlannedModel:
    """A named model and the interaction tables owned by its GLM core."""

    model_id: str
    name: str
    core: ComponentRecipe
    pair_corrections: tuple[ComponentRecipe, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.model_id, "Planned model ID")
        _nonempty(self.name, "Planned model name")
        if self.core.component_id != self.model_id:
            raise ValueError(
                "A planned model core component ID must equal its model ID"
            )
        if self.core.kind not in {"glm", "catboost", "catalogue"}:
            raise ValueError("Planned model core must be GLM, CatBoost or Catalogue")
        if self.core.kind != "glm" and self.pair_corrections:
            raise ValueError("Only a GLM can own interaction tables")
        seen_pairs: set[frozenset[str]] = set()
        for correction in self.pair_corrections:
            if correction.kind != "interaction":
                raise ValueError("A pair correction must be an interaction table")
            if len(correction.factors) != 2:
                raise ValueError("An interaction table needs exactly two factors")
            if correction.factors[0] == correction.factors[1]:
                raise ValueError("An interaction table needs two different factors")
            pair = frozenset(correction.factors)
            if pair in seen_pairs:
                raise ValueError("A GLM cannot contain the same interaction twice")
            seen_pairs.add(pair)

    @property
    def physical_components(self) -> tuple[ComponentRecipe, ...]:
        """Exact scorer order for fitting, scoring, diagnostics and export."""
        return (self.core, *self.pair_corrections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "core": self.core.to_dict(),
            "pair_corrections": [pair.to_dict() for pair in self.pair_corrections],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PlannedModel:
        if not isinstance(raw, dict):
            raise ValueError("Planned model must be an object")
        allowed = {"model_id", "name", "core", "pair_corrections"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError("Planned model has unknown fields: " + ", ".join(unknown))
        _nonempty(raw.get("model_id"), "Planned model ID")
        _nonempty(raw.get("name"), "Planned model name")
        core = raw.get("core")
        if not isinstance(core, dict):
            raise ValueError("Planned model core must be an object")
        pairs = raw.get("pair_corrections", [])
        if not isinstance(pairs, list) or not all(
            isinstance(pair, dict) for pair in pairs
        ):
            raise ValueError("Planned model pair_corrections must be a list of objects")
        return cls(
            model_id=raw["model_id"],
            name=raw["name"],
            core=ComponentRecipe.from_dict(core),
            pair_corrections=tuple(ComponentRecipe.from_dict(pair) for pair in pairs),
        )


@dataclass(frozen=True)
class AdoptedRoot:
    """An exact frozen starting scorer, never a synthetic editable GLM."""

    model_name: str
    fit_id: str
    scorer_fingerprint: str
    physical_prefix: tuple[ComponentRecipe, ...]

    def __post_init__(self) -> None:
        _nonempty(self.model_name, "Adopted root model_name")
        _nonempty(self.fit_id, "Adopted root fit_id")
        _nonempty(self.scorer_fingerprint, "Adopted root scorer_fingerprint")
        if not self.physical_prefix:
            raise ValueError("Adopted root physical_prefix must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "fit_id": self.fit_id,
            "scorer_fingerprint": self.scorer_fingerprint,
            "physical_prefix": [
                component.to_dict() for component in self.physical_prefix
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AdoptedRoot:
        if not isinstance(raw, dict):
            raise ValueError("adopted_root must be an object")
        required = {
            "model_name",
            "fit_id",
            "scorer_fingerprint",
            "physical_prefix",
        }
        if set(raw) != required:
            raise ValueError("adopted_root has unknown or missing fields")
        prefix = raw["physical_prefix"]
        if not isinstance(prefix, list) or not all(
            isinstance(item, dict) for item in prefix
        ):
            raise ValueError("adopted_root.physical_prefix must be a list of recipes")
        return cls(
            model_name=raw["model_name"],
            fit_id=raw["fit_id"],
            scorer_fingerprint=raw["scorer_fingerprint"],
            physical_prefix=tuple(ComponentRecipe.from_dict(item) for item in prefix),
        )


@dataclass
class ModelPlan:
    """Current workbench recipe, separated from immutable fitted history.

    ``stages`` remains the v1 physical representation.  For a v2 plan it is a
    read-compatible projection of ``models`` and is deliberately not written
    into canonical v2 JSON.
    """

    plan_id: str = field(default_factory=new_plan_id)
    mode: PlanMode = "single"
    family: str = "poisson"
    link: str | None = None
    tweedie_power: float = 1.5
    divide_target_by_weight: bool = False
    stages: list[ComponentRecipe] = field(default_factory=list)
    format_version: int = PLAN_FORMAT_LEGACY
    models: list[PlannedModel] = field(default_factory=list)
    adopted_root: AdoptedRoot | None = None
    adopted_model_name: str | None = None
    fit_refs: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.format_version not in {PLAN_FORMAT_LEGACY, PLAN_FORMAT_MODELS}:
            raise ValueError("model_plan.format_version must be 1 or 2")
        if self.format_version == PLAN_FORMAT_MODELS:
            expected = self._flatten_models()
            if self.stages and self.stages != expected:
                raise ValueError(
                    "model_plan v2 stages must be the models' physical projection"
                )
            self.stages = expected

    def _flatten_models(self) -> list[ComponentRecipe]:
        return [
            component
            for model in self.models
            for component in model.physical_components
        ]

    @property
    def physical_stages(self) -> tuple[ComponentRecipe, ...]:
        return tuple(self.stages)

    @property
    def is_logical_models_plan(self) -> bool:
        return self.format_version == PLAN_FORMAT_MODELS

    @property
    def logical_model_count(self) -> int:
        return len(self.models) if self.is_logical_models_plan else len(self.stages)

    def core_for_model(self, model_id: str) -> ComponentRecipe:
        for model in self.models:
            if model.model_id == model_id:
                return model.core
        raise KeyError(model_id)

    def logical_model_for_component(self, component_id: str) -> PlannedModel | None:
        if not self.is_logical_models_plan:
            return None
        for model in self.models:
            if any(
                component.component_id == component_id
                for component in model.physical_components
            ):
                return model
        return None

    def components_through(self, component_id: str) -> tuple[ComponentRecipe, ...]:
        for index, component in enumerate(self.stages):
            if component.component_id == component_id:
                return tuple(self.stages[: index + 1])
        raise KeyError(component_id)

    def to_dict(self) -> dict[str, Any]:
        common = {
            "plan_id": self.plan_id,
            "mode": self.mode,
            "family": self.family,
            "link": self.link,
            "tweedie_power": self.tweedie_power,
            "divide_target_by_weight": self.divide_target_by_weight,
            "fit_refs": self.fit_refs,
        }
        if self.is_logical_models_plan:
            return {
                "format_version": PLAN_FORMAT_MODELS,
                **common,
                "models": [model.to_dict() for model in self.models],
                "adopted_root": (
                    self.adopted_root.to_dict()
                    if self.adopted_root is not None
                    else None
                ),
            }
        return {
            **common,
            "stages": [stage.to_dict() for stage in self.stages],
            "adopted_model_name": self.adopted_model_name,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelPlan:
        if not isinstance(value, dict):
            raise ValueError("model_plan must be an object")
        raw_format = value.get("format_version", PLAN_FORMAT_LEGACY)
        if raw_format not in {PLAN_FORMAT_LEGACY, PLAN_FORMAT_MODELS}:
            raise ValueError("model_plan.format_version must be 1 or 2")
        common = {
            "plan_id",
            "mode",
            "family",
            "link",
            "tweedie_power",
            "divide_target_by_weight",
            "adopted_model_name",
            "adopted_root",
            "fit_refs",
            "format_version",
        }
        content_key = "models" if raw_format == PLAN_FORMAT_MODELS else "stages"
        unknown = sorted(set(value) - (common | {content_key}))
        if unknown:
            raise ValueError("model_plan has unknown fields: " + ", ".join(unknown))
        for key in ("mode", "family"):
            if key in value and not isinstance(value[key], str):
                raise ValueError(f"model_plan.{key} must be a string")
        if value.get("link") is not None and not isinstance(value["link"], str):
            raise ValueError("model_plan.link must be a string or null")
        plan_id = value.get("plan_id") or new_plan_id()
        _nonempty(plan_id, "model_plan.plan_id")
        adopted = value.get("adopted_model_name")
        if adopted is not None:
            _nonempty(adopted, "model_plan.adopted_model_name")
        refs = value.get("fit_refs", [])
        if not isinstance(refs, list) or not all(
            isinstance(item, dict) for item in refs
        ):
            raise ValueError("model_plan.fit_refs must be a list of objects")

        common_values = {
            "plan_id": plan_id,
            "mode": value.get("mode", "single"),
            "family": value.get("family", "poisson"),
            "link": value.get("link"),
            "tweedie_power": value.get("tweedie_power", 1.5),
            "divide_target_by_weight": value.get("divide_target_by_weight", False),
            "fit_refs": [dict(item) for item in refs],
        }
        if raw_format == PLAN_FORMAT_MODELS:
            if adopted is not None:
                raise ValueError(
                    "model_plan v2 uses adopted_root, not adopted_model_name"
                )
            raw_models = value.get("models", [])
            if not isinstance(raw_models, list) or not all(
                isinstance(item, dict) for item in raw_models
            ):
                raise ValueError("model_plan.models must be a list of model objects")
            plan = cls(
                **common_values,
                format_version=PLAN_FORMAT_MODELS,
                models=[PlannedModel.from_dict(item) for item in raw_models],
                adopted_root=(
                    AdoptedRoot.from_dict(value["adopted_root"])
                    if value.get("adopted_root") is not None
                    else None
                ),
            )
            _validate_v2_fit_refs(plan)
            return plan

        raw_stages = value.get("stages", [])
        if value.get("adopted_root") is not None:
            raise ValueError("model_plan v1 uses adopted_model_name, not adopted_root")
        if not isinstance(raw_stages, list) or not all(
            isinstance(item, dict) for item in raw_stages
        ):
            raise ValueError("model_plan.stages must be a list of stage objects")
        if not all(
            all(
                isinstance(item.get(key), str) and item[key]
                for key in ("stage_id", "model_name", "fit_id")
            )
            for item in refs
        ):
            raise ValueError("model_plan.fit_refs entries are invalid")
        return cls(
            **common_values,
            stages=[ComponentRecipe.from_dict(item) for item in raw_stages],
            adopted_model_name=adopted,
        )


def _validate_v2_fit_refs(plan: ModelPlan) -> None:
    """Check server-issued immutable identities rather than client guesses."""
    components = {component.component_id for component in plan.physical_stages}
    if plan.adopted_root is not None:
        adopted_ids = [
            component.component_id for component in plan.adopted_root.physical_prefix
        ]
        if len(set(adopted_ids)) != len(adopted_ids) or not all(adopted_ids):
            raise ValueError("adopted_root.physical_prefix component IDs are invalid")
        if components.intersection(adopted_ids):
            raise ValueError(
                "adopted_root.physical_prefix component IDs overlap the working plan"
            )
    model_ids = {model.model_id for model in plan.models}
    required = {
        "model_id",
        "through_component_id",
        "model_name",
        "fit_id",
        "prefix_fingerprint",
    }
    seen_models: set[str] = set()
    for item in plan.fit_refs:
        if set(item) != required or not all(
            isinstance(item.get(key), str) and item[key] for key in required
        ):
            raise ValueError("model_plan v2 fit_refs entries are invalid")
        if item["model_id"] not in model_ids:
            raise ValueError("model_plan v2 fit_refs references an unknown model")
        if item["model_id"] in seen_models:
            raise ValueError(
                "model_plan v2 fit_refs has more than one active reference per model"
            )
        seen_models.add(item["model_id"])
        if item["through_component_id"] not in components:
            raise ValueError("model_plan v2 fit_refs references an unknown component")
        owner = plan.logical_model_for_component(item["through_component_id"])
        if owner is None or owner.model_id != item["model_id"]:
            raise ValueError(
                "model_plan v2 fit_refs checkpoint belongs to another model"
            )


def structural_problems(
    plan: ModelPlan, families: set[str] | tuple[str, ...]
) -> list[str]:
    """Validation safe during import; no data or fitted artifacts are needed."""
    problems: list[str] = []
    if plan.mode not in {"single", "sequential"}:
        problems.append("Model plan mode must be single or sequential")
    if plan.family not in families:
        problems.append("Choose a supported model family")
    if plan.link not in {None, "log", "logit"}:
        problems.append("Rate-table models require a log or logit link")
    if not isinstance(plan.divide_target_by_weight, bool):
        problems.append("Divide target by weight must be true or false")
    if plan.mode == "single" and plan.logical_model_count > 1:
        problems.append("A single-model plan can contain one model")
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

    if plan.is_logical_models_plan:
        model_ids = [model.model_id for model in plan.models]
        names = [model.name.strip() for model in plan.models]
        if any(not value for value in model_ids) or len(set(model_ids)) != len(
            model_ids
        ):
            problems.append("Model IDs must be nonblank and unique")
        if any(not value for value in names) or len(set(names)) != len(names):
            problems.append("Model names must be nonblank and unique")
    else:
        names = [stage.name.strip() for stage in plan.stages]
        if any(not value for value in names) or len(set(names)) != len(names):
            problems.append("Stage names must be nonblank and unique")
    ids = [stage.component_id for stage in plan.physical_stages]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        problems.append("Component IDs must be nonblank and unique")
    for stage in plan.physical_stages:
        if stage.kind not in {"glm", "catboost", "interaction", "catalogue"}:
            problems.append(f"{stage.name}: unsupported model method {stage.kind!r}")
        try:
            validate_catboost_tuning_settings(stage.kind, stage.settings)
        except ValueError as exc:
            problems.append(f"{stage.name}: {exc}")
    return problems
