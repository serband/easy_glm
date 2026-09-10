"""Cached observed rates on applied training data; never fitted predictions."""

from __future__ import annotations

import threading
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import polars as pl

from easy_glm.core.design import NUMERIC_DTYPES
from easy_glm.desktop.fit_worker import json_safe
from easy_glm.workflow.explore import univariate
from easy_glm.workflow.prep import prepare, train_holdout
from easy_glm.workflow.project import Project

Generation = tuple[str, str, int]


def _kind(dtype: pl.DataType) -> str:
    return "numeric" if dtype in NUMERIC_DTYPES else "categorical"


def _available_numeric(frame: pl.DataFrame, column: str | None) -> bool:
    return bool(
        column in frame.columns
        and (
            frame.schema[column].is_numeric()
            or frame.schema[column] in (pl.Boolean, pl.Null)
        )
    )


def _context(
    project: Project, frame: pl.DataFrame, model: str | None
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str | None]]:
    models = [
        {
            "name": name,
            "target": cfg.target,
            "weight": cfg.weight,
            "divide_target_by_weight": cfg.divide_target_by_weight,
        }
        for name, cfg in project.models.items()
        if _available_numeric(frame, cfg.target)
        and (cfg.weight is None or _available_numeric(frame, cfg.weight))
        and (not cfg.divide_target_by_weight or cfg.weight is not None)
    ]
    by_name = {cfg["name"]: cfg for cfg in models}
    if model and model not in by_name:
        raise ValueError("Choose a model with an available numeric target and weight.")
    chosen = model or (
        project.champion if project.champion in by_name else next(iter(by_name), None)
    )
    if chosen:
        context = {"model": chosen, **by_name[chosen]}
        context.pop("name")
        offset = project.models[chosen].offset
    else:
        target = project.target if _available_numeric(frame, project.target) else None
        weight = project.weight
        if weight and not _available_numeric(frame, weight):
            raise ValueError(f"Weight column {weight!r} must contain numeric values.")
        context = {
            "model": None,
            "target": target,
            "weight": weight,
            "divide_target_by_weight": bool(weight),
        }
        offset = project.offset_column
    reserved = {
        context["target"],
        context["weight"],
        offset,
        project.target,
        project.weight,
        project.offset_column,
        project.data.split.column,
    }
    return models, context, reserved


class ExplorationCache:
    """One prepared training sample and a bounded set of charts per revision."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.generation: Generation | None = None
        self.frame: pl.DataFrame | None = None
        self.training_rows = 0
        self.charts: OrderedDict[tuple[str | None, str | None, int], dict[str, Any]] = (
            OrderedDict()
        )

    def view(
        self,
        project: Project,
        raw: pl.DataFrame,
        generation: Generation,
        *,
        column: str | None = None,
        model: str | None = None,
        n_bins: int = 20,
    ) -> dict[str, Any]:
        if not 5 <= n_bins <= 50:
            raise ValueError("Choose between 5 and 50 bands.")
        with self.lock:
            if generation != self.generation:
                if not raw.width:
                    raise ValueError("Open data before exploring variables.")
                frame, _ = train_holdout(prepare(project, raw), project.data.split)
                training_rows = frame.height
                sample_rows = project.data.sample_rows
                if sample_rows is not None:
                    if isinstance(sample_rows, bool) or sample_rows < 1:
                        raise ValueError(
                            "The exploration sample size must be positive."
                        )
                    if sample_rows < training_rows:
                        frame = frame.sample(
                            n=sample_rows, seed=project.data.sample_seed
                        )
                self.frame = frame
                self.training_rows = training_rows
                self.generation = generation
                self.charts.clear()
            assert self.frame is not None
            frame = self.frame
            models, context, reserved = _context(project, frame, model)
            columns = [
                {"name": name, "kind": _kind(dtype)}
                for name, dtype in frame.schema.items()
                if name not in reserved
            ]
            available = {item["name"] for item in columns}
            if column not in available:
                selected = context["model"]
                preferred = (
                    project.models[selected].predictors if selected else []
                ) + project.predictors
                column = next(
                    (name for name in preferred if name in available),
                    columns[0]["name"] if columns else None,
                )
            key = context["model"], column, n_bins
            if key in self.charts:
                self.charts.move_to_end(key)
                return deepcopy(self.charts[key])
            weight, target = context["weight"], context["target"]
            if weight:
                values = frame[weight].cast(pl.Float64)
                invalid = (~values.is_finite() | (values < 0)).fill_null(True).sum()
                if invalid:
                    raise ValueError(
                        f"Weight column {weight!r} has {invalid:,} missing, negative or "
                        "non-finite values. Filter or correct those training rows."
                    )
            excluded = (
                int((~frame[target].cast(pl.Float64).is_finite()).fill_null(True).sum())
                if target
                else 0
            )
            summary = (
                univariate(
                    frame,
                    column,
                    target=target,
                    weight=weight,
                    divide_target_by_weight=context["divide_target_by_weight"],
                    n_bins=n_bins,
                )
                if column
                else None
            )
            rate_label = "Observed rate"
            if target:
                rate_label = (
                    f"{target} / {weight}"
                    if context["divide_target_by_weight"] and weight
                    else f"{'Weighted mean' if weight else 'Mean'} {target}"
                )
            result = json_safe(
                {
                    "session_id": generation[0],
                    "project_id": generation[1],
                    "revision": generation[2],
                    "models": models,
                    **context,
                    "columns": columns,
                    "column": column,
                    "n_bins": n_bins,
                    "rows": frame.height,
                    "training_rows": self.training_rows,
                    "sampled": frame.height < self.training_rows,
                    "kind": summary["kind"] if summary else None,
                    "null_share": summary["null_share"] if summary else 0,
                    "n_unique": summary["n_unique"] if summary else 0,
                    "rate_label": rate_label,
                    "exposure_label": weight or "Rows",
                    "rate_excluded_rows": excluded,
                    "table": summary["table"].to_dicts() if summary else [],
                }
            )
            self.charts[key] = result
            if len(self.charts) > 64:
                self.charts.popitem(last=False)
            return deepcopy(result)
