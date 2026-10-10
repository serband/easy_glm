"""Desktop read models for ordered component stacks.

Every diagnostic here scores the accepted cumulative stack. Holdout is only
computed through an explicit request, never while fitting or opening a result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from easy_glm.workflow.diagnostics import (
    ae_by_variable,
    family_metrics,
    gini,
    lift_table,
    metric_definitions,
    totals,
    unit_values,
)
from easy_glm.workflow.model_plan import ModelPlan
from easy_glm.workflow.prep import train_holdout
from easy_glm.workflow.project import Project
from easy_glm.workflow.stack_types import LookupScorer, StackScorer


def lookup_coverage(scorer: LookupScorer, frame: pl.DataFrame) -> dict[str, Any]:
    """Older live sessions may still hold the pre-coverage scorer class."""
    method = getattr(scorer, "coverage", None)
    if callable(method):
        return method(frame)
    return {
        "rows": frame.height,
        "missing_key_rows": None,
        "fallback_rows": None,
        "unknown_examples": [],
        "unavailable": True,
    }


def _metrics(
    run: Any, frame: pl.DataFrame, scorer: StackScorer | None = None
) -> dict[str, Any]:
    from easy_glm.pricing.validation import _family_instance

    prediction = (scorer or run.stack).predict_unit(frame)
    actual, expected, weight = totals(frame, run.config, prediction)
    unit_target, unit_weight = unit_values(frame, run.config)
    family = run.config.family
    gini_value = (
        gini(actual, expected, weight)
        if family in {"poisson", "gamma", "tweedie", "inverse_gaussian"}
        and np.all(actual >= 0)
        and np.all(expected >= 0)
        and float(actual.sum()) > 0
        else float("nan")
    )
    deviance = _family_instance(run.config).deviance(
        unit_target, prediction, sample_weight=unit_weight
    )
    return {
        "rows": float(frame.height),
        "exposure": float(weight.sum()),
        "actual": float(actual.sum()),
        "expected": float(expected.sum()),
        "ae": (
            float(actual.sum() / expected.sum())
            if family not in {"gaussian", "normal"} and expected.sum() > 0
            else None
        ),
        "gini": float(gini_value) if np.isfinite(gini_value) else None,
        "mean_deviance": float(deviance / unit_weight.sum()),
        **family_metrics(run.config.family, unit_target, prediction, unit_weight),
    }


def _saved_plan(run: Any, saved_plan: ModelPlan | None = None) -> ModelPlan | None:
    """Read the plan snapshot attached to this fitted run, never live state."""
    if saved_plan is not None:
        return saved_plan
    snapshot = getattr(run, "project_snapshot", None)
    if isinstance(snapshot, ModelPlan):
        return snapshot
    if isinstance(snapshot, Project):
        return snapshot.model_plan
    if isinstance(snapshot, dict):
        try:
            return Project.from_dict(snapshot).model_plan
        except (TypeError, ValueError):
            # An old or partial saved artifact still has exact physical
            # components.  Its safe presentation is the existing legacy list.
            return None
    return None


def _component_detail(component: Any) -> dict[str, Any]:
    return {
        "id": component.recipe.component_id,
        "name": component.name,
        "kind": component.kind,
        "factors": list(component.factors),
    }


def _logical_diagnostic_scopes(
    stack: StackScorer, plan: ModelPlan
) -> list[dict[str, Any]] | None:
    """Group a v2 saved physical stack by its saved named-model ownership.

    A plan can be edited after fitting.  We use it only when its saved IDs are
    a prefix-exact match for this stack; otherwise the physical saved recipes
    below are the only honest labels available.
    """
    if not plan.is_logical_models_plan or plan.adopted_root is not None:
        return None
    components = stack.diagnostic_components()
    expected = list(plan.physical_stages)
    actual_ids = [component.recipe.component_id for component in components]
    expected_ids = [component.component_id for component in expected]
    if not actual_ids or actual_ids != expected_ids[: len(actual_ids)]:
        return None

    by_id = {component.recipe.component_id: component for component in components}
    scopes: list[dict[str, Any]] = []
    included: list[str] = []
    cursor = 0
    for model in plan.models:
        physical = model.physical_components
        model_ids = [component.component_id for component in physical]
        if expected_ids[cursor : cursor + len(model_ids)] != model_ids:
            return None
        if cursor + len(model_ids) > len(actual_ids):
            break
        for index, recipe in enumerate(physical):
            component = by_id[recipe.component_id]
            included.append(component.name)
            final = cursor + index == len(actual_ids) - 1
            if index == 0 and model.core.kind == "glm":
                label = f"Main effects — {model.name}"
                checkpoint = "main_effects"
            elif index == 0:
                label = f"After {model.name}"
                checkpoint = "model"
            elif index == len(physical) - 1:
                label = f"After {model.name} ({len(model.pair_corrections)} tables)"
                checkpoint = "model"
            else:
                label = f"{model.name} + {component.name}"
                checkpoint = "interaction"
            scopes.append(
                {
                    "id": "" if final else recipe.component_id,
                    "label": label,
                    "included_models": list(included),
                    "component": _component_detail(component),
                    "logical_model_id": model.model_id,
                    "logical_model_name": model.name,
                    "checkpoint": checkpoint,
                }
            )
        cursor += len(physical)
    return scopes if scopes else None


def diagnostic_scopes(
    stack: StackScorer, saved_plan: ModelPlan | None = None
) -> list[dict[str, Any]]:
    """Describe the saved physical checkpoints available for diagnostics."""
    if saved_plan is not None:
        logical = _logical_diagnostic_scopes(stack, saved_plan)
        if logical is not None:
            return logical
    components = stack.diagnostic_components()
    if not components:
        return []
    identifiers = [item.recipe.component_id for item in components]
    if any(not identifier.strip() for identifier in identifiers):
        raise ValueError("Saved diagnostic stages need non-empty component IDs.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Saved diagnostic stages have duplicate component IDs.")
    scopes = []
    for index, component in enumerate(components):
        detail = _component_detail(component)
        included = [item.name for item in components[: index + 1]]
        if index == len(components) - 1:
            # The final score is selected by omitting ``through``.  Giving it
            # a separate empty ID keeps the menu unambiguous and avoids a
            # duplicate last-stage result.
            scopes.append(
                {
                    "id": "",
                    "label": f"Final fit — after {component.name}",
                    "included_models": included,
                    "component": detail,
                }
            )
        else:
            scopes.append(
                {
                    "id": component.recipe.component_id,
                    "label": f"After {component.name}",
                    "included_models": included,
                    "component": detail,
                }
            )
    return scopes


def _diagnostic_scope(
    stack: StackScorer, through: str | None, saved_plan: ModelPlan | None = None
) -> tuple[dict[str, Any], StackScorer]:
    """Resolve one public scope to its exact saved scorer prefix."""
    scopes = diagnostic_scopes(stack, saved_plan)
    if not scopes:
        raise ValueError("This fitted stack has no diagnostic stages.")
    selected = "" if through is None else through
    scope = next((item for item in scopes if item["id"] == selected), None)
    if scope is None:
        # The final physical ID is accepted as an alias for the final score.
        # It is deliberately not emitted separately in ``diagnostic_scopes``.
        components = stack.diagnostic_components()
        if components and selected == components[-1].recipe.component_id:
            scope = scopes[-1]
        else:
            raise ValueError(f"Diagnostic stage {selected!r} is unavailable.")
    return scope, stack.prefix_through(scope["id"] or None)


def _review_variables(
    project: Any,
    config: Any,
    raw: pl.DataFrame,
    enriched: pl.DataFrame,
) -> list[str]:
    """Return all factors usable for grouped A/E on this saved checkpoint."""
    protected_roles = {
        "target",
        "weight",
        "exposure",
        "offset",
        "id",
        "split",
        "time",
        "ignore",
        "current_premium",
    }
    protected = {
        value
        for value in (config.target, config.weight, config.offset)
        if value is not None
    }
    protected.update(
        name for name, role in project.data.roles.items() if role in protected_roles
    )
    # Start with every raw source column, then add only catalogue attributes
    # available at this prefix.  This lets a reviewer inspect an unmodelled
    # source variable while not offering an attribute that has not yet joined.
    candidates = set(raw.columns) | (set(enriched.columns) - set(raw.columns))
    return sorted(name for name in candidates if name not in protected)


def stack_result(
    project: Any,
    frame: pl.DataFrame,
    run: Any,
    notices: list[str] | None = None,
    *,
    saved_plan: ModelPlan | None = None,
) -> dict[str, Any]:
    """Describe a fitted stack using only the training portfolio."""
    train, _holdout = train_holdout(frame, project.data.split)
    metrics = _metrics(run, train)
    actual, expected, weight = totals(train, run.config, run.stack.predict_unit(train))
    components = []
    prefix: list[str] = []
    visible_components = [
        *run.stack.virtual_components,
        *(
            item
            for item in run.stack.components
            if item.kind != "legacy" or not run.stack.virtual_components
        ),
    ]
    for index, item in enumerate(visible_components):
        unit_label = (
            "Complete unit prediction"
            if index == 0 and item.kind == "catalogue" and not run.stack.offset_col
            else (
                "Multiplier on external offset"
                if index == 0 and item.kind == "catalogue"
                else "Incremental multiplier"
            )
        )
        components.append(
            {
                "id": item.recipe.component_id,
                "name": item.name,
                "kind": item.kind,
                "factors": list(item.factors),
                "key": item.recipe.settings.get("key"),
                "baseline": list(prefix),
                "status": item.status,
                "unit_label": unit_label,
                "evidence": item.evidence,
                "coverage": (
                    lookup_coverage(item.scorer, train)
                    if isinstance(item.scorer, LookupScorer)
                    else None
                ),
            }
        )
        prefix.append(item.name)
    plan = _saved_plan(run, saved_plan)
    return {
        "stack": True,
        "components": components,
        "summary": {
            "name": run.name,
            "family": run.config.family,
            "link": run.stack.link,
            "train_ae": metrics["ae"],
            "components": len(components),
            "train_rows": train.height,
            "created_at": run.created_at,
        },
        "metrics": {"train": metrics},
        "lift": {"train": lift_table(actual, expected, weight).to_dicts()},
        "diagnostic_scopes": diagnostic_scopes(run.stack, plan),
        "review_variables": _review_variables(
            project, run.config, train, run.stack.enrich(train)
        ),
        "metric_definitions": {
            "train": metric_definitions(
                run.config.family, run.config.tweedie_power, metrics
            )
        },
        "warnings": list(dict.fromkeys(notices or [])),
    }


def stack_diagnostics(
    project: Any,
    frame: pl.DataFrame,
    run: Any,
    *,
    subset: str,
    variable: str | None = None,
    through: str | None = None,
    saved_plan: ModelPlan | None = None,
) -> dict[str, Any]:
    train, holdout = train_holdout(frame, project.data.split)
    if subset not in {"train", "holdout"}:
        raise ValueError("Choose training or holdout rows.")
    selected = train if subset == "train" else holdout
    if selected.is_empty():
        raise ValueError(f"No {subset} rows are available.")
    plan = _saved_plan(run, saved_plan)
    scope, scorer = _diagnostic_scope(run.stack, through, plan)
    enriched = scorer.enrich(selected)
    prediction = scorer.predict_unit(selected)
    actual, expected, weight = totals(selected, run.config, prediction)
    result = {
        "subset": subset,
        "saved_model": run.name,
        "scope": scope,
        "diagnostic_scopes": diagnostic_scopes(run.stack, plan),
        "review_variables": _review_variables(project, run.config, selected, enriched),
        "metrics": _metrics(run, selected, scorer),
        "lift": lift_table(actual, expected, weight).to_dicts(),
        "catalogue_coverage": {
            item.recipe.component_id: lookup_coverage(item.scorer, selected)
            for item in scorer.components
            if isinstance(item.scorer, LookupScorer)
        },
    }
    result["metric_definitions"] = {
        subset: metric_definitions(
            run.config.family, run.config.tweedie_power, result["metrics"]
        )
    }
    if variable is not None:
        if variable not in enriched.columns:
            raise ValueError(f"Variable {variable!r} is unavailable for A/E.")
        result["variable"] = variable
        result["ae"] = ae_by_variable(
            enriched, variable, actual, expected, weight
        ).to_dicts()
    return result
