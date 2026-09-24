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
from easy_glm.workflow.prep import train_holdout
from easy_glm.workflow.stack_types import LookupScorer


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


def _metrics(run: Any, frame: pl.DataFrame) -> dict[str, Any]:
    from easy_glm.pricing.validation import _family_instance

    prediction = run.stack.predict_unit(frame)
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


def stack_result(
    project: Any, frame: pl.DataFrame, run: Any, notices: list[str] | None = None
) -> dict[str, Any]:
    """Describe a fitted stack using only the training portfolio."""
    train, _holdout = train_holdout(frame, project.data.split)
    metrics = _metrics(run, train)
    actual, expected, weight = totals(train, run.config, run.stack.predict_unit(train))
    components = []
    prefix = []
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
        "review_variables": sorted(
            name
            for name in run.stack.enrich(train).columns
            if name not in {run.config.target, run.config.weight, run.config.offset}
            and project.data.roles.get(name)
            not in {
                "target",
                "weight",
                "exposure",
                "offset",
                "id",
                "split",
                "time",
                "ignore",
            }
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
) -> dict[str, Any]:
    train, holdout = train_holdout(frame, project.data.split)
    if subset not in {"train", "holdout"}:
        raise ValueError("Choose training or holdout rows.")
    selected = train if subset == "train" else holdout
    if selected.is_empty():
        raise ValueError(f"No {subset} rows are available.")
    enriched = run.stack.enrich(selected)
    prediction = run.stack.predict_unit(selected)
    actual, expected, weight = totals(selected, run.config, prediction)
    result = {
        "subset": subset,
        "metrics": _metrics(run, selected),
        "lift": lift_table(actual, expected, weight).to_dicts(),
        "catalogue_coverage": {
            item.recipe.component_id: lookup_coverage(item.scorer, selected)
            for item in run.stack.components
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
