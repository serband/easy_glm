"""Pure presentation choices for scoring an immutable fitted artifact."""

from __future__ import annotations

from typing import Any

from easy_glm.engine.rate_model import RateModel
from easy_glm.workflow.scoring import _requirements
from easy_glm.workflow.stack_types import LookupScorer, StackScorer


def _scorer(fitted: Any) -> RateModel | StackScorer:
    stack = getattr(fitted, "stack", None)
    return stack if isinstance(stack, StackScorer) else fitted.rate_model


def _raw_columns(project: Any, required: set[str]) -> list[str]:
    data = project.data
    derived = {item.name: item for item in data.derived}
    expanded = set(required)
    pending = list(required)
    while pending:
        name = pending.pop()
        item = derived.get(name)
        if item is not None:
            expanded.discard(name)
            from easy_glm.workflow.prep import eval_expr

            for root in eval_expr(item.expr).meta.root_names():
                if root not in expanded:
                    expanded.add(root)
                    pending.append(root)
    premium = next(
        (name for name, role in data.roles.items() if role == "current_premium"), None
    )
    if premium and f"log_{premium}" in expanded:
        expanded.remove(f"log_{premium}")
        expanded.add(premium)
    inverse_renames = {new: old for old, new in data.renames.items()}
    return sorted(inverse_renames.get(name, name) for name in expanded)


def _kind(
    scorer: RateModel | StackScorer, *, root: bool, candidate: bool = False
) -> str:
    if candidate:
        return "candidate_multiplier"
    link = scorer.link if isinstance(scorer, StackScorer) else scorer.metadata.link
    if link == "logit":
        return "probability"
    family = (
        scorer.family
        if isinstance(scorer, StackScorer)
        else getattr(scorer.metadata, "family", "")
    )
    divided = (
        getattr(scorer, "divide_target_by_weight", False)
        if isinstance(scorer, StackScorer)
        else scorer.metadata.divide_target_by_weight
    )
    if root and family == "poisson" and divided:
        return "frequency"
    return "rate" if root else "multiplier"


def _labels(kind: str) -> tuple[str, str]:
    return {
        "frequency": ("Predicted frequency", "Expected claims"),
        "probability": ("Predicted probability", "Expected probability"),
        "rate": ("Predicted rate", "Expected amount"),
        "multiplier": ("Rating multiplier", "Expected multiplier"),
        "candidate_multiplier": (
            "Catalogue candidate multiplier",
            "Candidate multiplier",
        ),
    }[kind]


def _display_name(recipe: Any) -> str:
    """Use the saved public model name when an adapted legacy recipe has one."""
    return str(recipe.settings.get("model_name", recipe.name))


def _target(
    *,
    identifier: str,
    purpose: str,
    label: str,
    model_name: str,
    saved_version: str,
    created_at: str,
    included: list[str],
    scorer: RateModel | StackScorer,
    project: Any,
    scope: str,
    components: list[str] | None = None,
    through: str | None = None,
    root: bool = False,
    candidate_for: str | None = None,
    native_available: bool = False,
    request_model_id: str | None = None,
) -> dict[str, Any]:
    component_ids = components or []
    target_component = None
    if isinstance(scorer, StackScorer) and (through or component_ids):
        target_component = scorer.component(through or component_ids[-1])
    kind = _kind(scorer, root=root, candidate=purpose == "candidate")
    output_label, expected_label = _labels(kind)
    exposure = (
        None
        if kind == "probability" or purpose in {"rating", "candidate"}
        else (
            scorer.weight
            if isinstance(scorer, StackScorer)
            else scorer.metadata.weight_col
        )
    )
    if isinstance(scorer, StackScorer) and scope == "model":
        required: set[str] = set()
        for component_id in component_ids:
            required.update(
                _requirements(scorer, "model", scorer.component(component_id))
            )
    else:
        required = _requirements(scorer, scope, target_component)
    return {
        "id": identifier,
        "purpose": purpose,
        "label": label,
        "model_name": model_name,
        "saved_version": saved_version,
        "created_at": created_at,
        "included_models": included,
        "output_kind": kind,
        "output_label": output_label,
        "expected_label": expected_label,
        "required_columns": _raw_columns(project, required),
        "request": {
            "model_id": request_model_id or identifier.split("/", 1)[0],
            "scope": scope,
            "component_ids": component_ids,
            "through": through,
        },
        "exposure_column": exposure,
        "allowed_exposure_columns": [exposure] if exposure else [],
        "candidate_for": candidate_for,
        "native_available": native_available,
        "external_offset_column": (
            scorer.offset_col
            if isinstance(scorer, StackScorer)
            else scorer.metadata.offset_col
        ),
    }


def scoring_targets(
    project: Any, fitted: Any, record: dict[str, Any]
) -> list[dict[str, Any]]:
    """Project a frozen fit to UI choices without changing it or its recipe."""
    scorer = _scorer(fitted)
    fit_id = str(record["fit_id"])
    name = str(getattr(fitted, "name", record.get("name", "Saved model")))
    created = str(getattr(fitted, "created_at", ""))
    version = f"{name} · {created}" if created else name
    overall_id = fit_id + "/overall"
    real = scorer.diagnostic_components() if isinstance(scorer, StackScorer) else []
    real_names = [_display_name(item.recipe) for item in real]
    plan = getattr(project, "model_plan", None)
    artifacts_by_id = {item.recipe.component_id: item for item in real}
    presentation_groups: list[tuple[str, str, list[Any]]] = []
    if plan is not None and plan.is_logical_models_plan:
        # The fitted stack is the source of the components.  The plan merely
        # groups those immutable physical artifacts for display; fit_refs only
        # decide which logical groups are selectable below.
        claimed: set[str] = set()
        for model in plan.models:
            owned = [
                artifacts_by_id[recipe.component_id]
                for recipe in model.physical_components
                if recipe.component_id in artifacts_by_id
            ]
            if owned:
                claimed.update(item.recipe.component_id for item in owned)
                presentation_groups.append((model.model_id, model.name, owned))
        presentation_groups.extend(
            (
                item.recipe.component_id,
                _display_name(item.recipe),
                [item],
            )
            for item in real
            if item.recipe.component_id not in claimed
        )
    result = [
        _target(
            identifier=overall_id,
            purpose="overall",
            label="Complete accepted score",
            model_name=name,
            saved_version=version,
            created_at=created,
            included=real_names or [name],
            scorer=scorer,
            project=project,
            scope="full",
            root=True,
            request_model_id=fit_id,
        )
    ]
    if not isinstance(scorer, StackScorer):
        return result

    groups: list[tuple[str, str, list[Any]]] = []
    if plan is not None and plan.is_logical_models_plan:
        for ref in plan.fit_refs:
            if ref.get("fit_id") != fit_id or ref.get("model_name") != name:
                continue
            model = next(
                (item for item in plan.models if item.model_id == ref.get("model_id")),
                None,
            )
            if model is None:
                continue
            owned = [
                artifacts_by_id[recipe.component_id]
                for recipe in model.physical_components
                if recipe.component_id in artifacts_by_id
            ]
            end = next(
                (
                    i
                    for i, item in enumerate(owned)
                    if item.recipe.component_id == ref.get("through_component_id")
                ),
                None,
            )
            if end is not None:
                groups.append((model.model_id, model.name, owned[: end + 1]))
    if not groups and not (plan is not None and plan.is_logical_models_plan):
        groups = [
            (
                item.recipe.component_id,
                _display_name(item.recipe),
                [item],
            )
            for item in real
            if item.recipe.kind != "interaction"
        ]
    result[0]["included_models"] = (
        [
            label
            + (
                f" + {len(recipes) - 1} accepted pair table(s)"
                if len(recipes) > 1
                else ""
            )
            for _group_id, label, recipes in (presentation_groups or groups)
        ]
        or real_names
        or [name]
    )
    for group_id, label, recipes in groups:
        recipes = [getattr(item, "recipe", item) for item in recipes]
        ids = [item.component_id for item in recipes]
        root = bool(real and ids[0] == real[0].recipe.component_id)
        rating_id = f"{fit_id}/{group_id}"
        request_id = (
            f"{fit_id}:{group_id}"
            if plan is not None and plan.is_logical_models_plan
            else fit_id
        )
        result.append(
            _target(
                identifier=rating_id,
                purpose="rating",
                label=label,
                model_name=name,
                saved_version=version,
                created_at=created,
                included=[_display_name(item) for item in recipes],
                scorer=scorer,
                project=project,
                scope="model",
                components=ids,
                root=root,
                request_model_id=request_id,
            )
        )
        for recipe in recipes:
            if recipe.component_id == real[-1].recipe.component_id:
                continue
            prefix = [
                item.recipe
                for item in real[
                    : next(
                        i
                        for i, item in enumerate(real)
                        if item.recipe.component_id == recipe.component_id
                    )
                    + 1
                ]
            ]
            result.append(
                _target(
                    identifier=f"{fit_id}/{recipe.component_id}/checkpoint",
                    purpose="checkpoint",
                    label=f"Through {_display_name(recipe)}",
                    model_name=name,
                    saved_version=version,
                    created_at=created,
                    included=[_display_name(item) for item in prefix],
                    scorer=scorer,
                    project=project,
                    scope="through",
                    through=recipe.component_id,
                    root=True,
                    request_model_id=request_id,
                )
            )
        for recipe in recipes:
            artifact = scorer.component(recipe.component_id)
            if (
                recipe.kind == "catalogue"
                and isinstance(artifact.scorer, LookupScorer)
                and artifact.scorer.native_scorer is not None
            ):
                result.append(
                    _target(
                        identifier=f"{fit_id}/{recipe.component_id}/candidate",
                        purpose="candidate",
                        label=f"{_display_name(recipe)} catalogue candidate",
                        model_name=name,
                        saved_version=version,
                        created_at=created,
                        included=[_display_name(recipe)],
                        scorer=scorer,
                        project=project,
                        scope="catalogue",
                        components=[recipe.component_id],
                        candidate_for=rating_id,
                        native_available=True,
                        request_model_id=request_id,
                    )
                )
    return result
