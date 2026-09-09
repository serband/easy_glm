"""Canonical fitted-model diagnostics shared by the local review worker."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from easy_glm.workflow.diagnostics import (
    alpha_path,
    base_rate_change,
    describe_diff,
    double_lift,
    gini,
    lift_table,
    predictions_effectively_equal,
    relativity_diff,
    totals,
)
from easy_glm.workflow.prep import train_holdout
from easy_glm.workflow.project import Project
from easy_glm.workflow.run import ModelRun, null_model_predict


def compatible(a: ModelRun, b: ModelRun) -> None:
    """Comparisons must describe the same response and exposure basis."""
    for field in ("target", "weight", "divide_target_by_weight"):
        if getattr(a.config, field) != getattr(b.config, field):
            raise ValueError(
                f"Cannot compare these models: {field.replace('_', ' ')} differs."
            )


def safe_gini(actual: Any, expected: Any, weight: Any) -> float | None:
    if np.any(np.asarray(actual) < 0) or np.any(np.asarray(expected) < 0):
        return None
    value = gini(actual, expected, weight)
    return value if np.isfinite(value) else None


def metadata(run: ModelRun, frame: pl.DataFrame) -> dict[str, Any]:
    actual, expected, weight = totals(frame, run.config, run.predict(frame))
    nonnegative = bool(np.all(actual >= 0) and np.all(expected >= 0))
    note = (
        "Exposure-weighted normalised Gini measures ordering by predicted rate; it is not ROC AUC."
        if nonnegative
        else "Gini and ratio-based double lift are unavailable for signed actuals or predictions."
    )
    return {
        "saved_versions": [
            {
                "model": run.name,
                "version": snap.version,
                "description": snap.description,
                "subset": subset,
                "ae": metrics.get("ae"),
                "gini": metrics.get("gini"),
            }
            for snap in run.rate_model.snapshots
            if snap.metrics
            for subset, metrics in snap.metrics.items()
            if isinstance(metrics, dict)
        ],
        "gini_note": note,
        "ratio_suitable": nonnegative,
        "variables": [
            {
                "name": name,
                "numeric": frame.schema[name].is_numeric(),
                "kind": run.spec[name].kind if name in run.spec.main_effects else None,
            }
            for name in frame.columns
        ],
        "facts": {
            "family": run.config.family,
            "link": run.fit.link,
            "alpha": run.alpha,
            "nonzero": int(np.count_nonzero(run.fit.coef)),
            "features": len(run.fit.coef),
            "base_rate": run.rate_model.base_rate,
            "interactions": ", ".join(f"{i.a} × {i.b}" for i in run.config.interactions)
            or "none",
            "adjustments": len(run.config.adjustments),
        },
    }


def view(
    project: Project,
    run: ModelRun,
    frame: pl.DataFrame,
    request: dict[str, Any],
    challenger: ModelRun | None = None,
) -> dict[str, Any]:
    train, holdout = train_holdout(frame, project.data.split)
    subset = request.get("subset", "holdout")
    part = {"train": train, "holdout": holdout, "all": frame}[subset]
    action = request["action"]
    if action == "path":
        return {
            "tables": [
                {"title": "Regularisation path", "rows": alpha_path(run.fit).to_dicts()}
            ],
            "path": alpha_path(run.fit).to_dicts(),
            "note": "Stage 1 is main effects; stage 2 is interaction cells. Selected marks each fitted penalty. A fixed alpha has one point.",
        }
    if action == "coefficients":
        rows = run.fit.coef_table(
            drop_zero=bool(request.get("options", {}).get("kept", True))
        ).to_dicts()
        return {
            "tables": [{"title": "Coefficients", "rows": rows}],
            "note": "Original fitted coefficients; post-fit table adjustments do not change these coefficients.",
        }
    if part.is_empty():
        raise ValueError("The selected subset has no rows.")
    prediction = run.predict(part)
    actual, expected, weight = totals(part, run.config, prediction)
    other = None
    if challenger is not None:
        compatible(run, challenger)
        other = totals(part, challenger.config, challenger.predict(part))[1]
    n_bins = request.get("n_bins", 10)
    if action == "lift":
        charts = [
            {
                "title": f"{run.name} · {subset}",
                "rows": lift_table(actual, expected, weight, n_bins=n_bins).to_dicts(),
            }
        ]
        scores = [
            {"model": run.name, "normalised_gini": safe_gini(actual, expected, weight)}
        ]
        if other is not None:
            charts.append(
                {
                    "title": f"{challenger.name} · {subset}",
                    "rows": lift_table(actual, other, weight, n_bins=n_bins).to_dicts(),
                }
            )
            scores.append(
                {
                    "model": challenger.name,
                    "normalised_gini": safe_gini(actual, other, weight),
                }
            )
        return {
            "charts": charts,
            "tables": [{"title": "Gini", "rows": scores}],
            "note": "Equal-exposure bins ordered by each model's predicted rate. Gini is unavailable for signed amounts or a constant actual rate.",
        }
    if action == "double_lift":
        benchmark = challenger.name if challenger else "Training-calibrated null model"
        if other is None:
            other = totals(
                part, run.config, null_model_predict(project, run.config, train, part)
            )[1]
        if np.any(actual < 0) or np.any(expected < 0) or np.any(other <= 0):
            raise ValueError(
                "Double lift requires nonnegative actuals/predictions and a strictly positive benchmark on every row."
            )
        same = predictions_effectively_equal(expected, other)
        rows = double_lift(actual, expected, other, weight, n_bins=n_bins).to_dicts()
        return {
            "charts": [
                {
                    "title": f"Double lift · {subset}",
                    "rows": rows,
                    "series": [
                        {"key": "ae_a", "label": run.name},
                        {"key": "ae_b", "label": benchmark},
                    ],
                }
            ],
            "note": (
                "The predictions are equal within numerical precision; the A/E lines overlap. "
                if same
                else ""
            )
            + "Bins order selected-model expected / benchmark expected, cheapest first. A/E nearer 1 indicates better calibration. "
            + (
                "No challenger selected: the null benchmark is calibrated on training rows."
                if challenger is None
                else ""
            ),
        }
    if action == "compare":
        if challenger is None:
            raise ValueError("Select a fitted challenger to compare relativities.")
        diff = relativity_diff(run, challenger, tol=request.get("tolerance", 0.01))
        return {
            "tables": [
                {
                    "title": "Relativities that differ",
                    "rows": describe_diff(diff, run.name, challenger.name).to_dicts(),
                }
            ],
            "base_rate_change": base_rate_change(run, challenger),
            "note": "Numeric factors use the union of both models' band edges. Categories and interaction cells match by label. Band changes combine with the base-rate change. Tolerance is absolute log difference.",
        }
    raise ValueError("Unknown diagnostic view.")
