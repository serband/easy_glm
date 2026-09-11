"""Chronological calibration diagnostics over the full prepared portfolio."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .diagnostics import ae_by_variable, totals
from .project import Project
from .run import ModelRun


def time_values(series: pl.Series) -> pl.Series:
    """Validate numeric time; zero and negative values are valid.

    Nulls are retained for an explicit excluded-row count. Source row order is
    irrelevant. Non-finite numbers, dates, text and booleans are refused.
    """
    if not series.dtype.is_numeric():
        raise ValueError(
            "Time must be numeric, for example a year or period number. Zero is valid."
        )
    parsed = series
    numeric = parsed.cast(pl.Float64)
    valid = numeric.drop_nulls()
    if valid.is_empty():
        raise ValueError("The Time column has no numeric values.")
    if not valid.is_finite().all():
        raise ValueError("The Time column contains non-finite numbers.")
    return parsed


def time_diagnostics(
    project: Project,
    run: ModelRun,
    frame: pl.DataFrame,
    *,
    n_bins: int = 5,
    variable: str | None = None,
    grouping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A/E by ordered time bands, with optional common-band factor comparisons.

    Score the original fitted model once, over both train and holdout. Quantile
    boundaries group adjacent time values without separating ties. This is a
    descriptive calibration review, not a forward-validation score.
    """
    column = project.column_with_role("time")
    if not column or column not in frame.columns:
        raise ValueError("Assign a Time role on Variables, then apply the settings.")
    if not 2 <= n_bins <= 20:
        raise ValueError("Choose between 2 and 20 time bands.")
    if variable and variable not in run.spec.main_effects:
        raise ValueError("Choose a fitted main factor for the time comparison.")
    times = time_values(frame[column])
    mask = times.is_not_null().to_numpy()
    excluded = int((~mask).sum())
    data = frame.filter(pl.Series(mask))
    times = times.filter(pl.Series(mask))
    numeric = times.cast(pl.Float64).to_numpy()
    # Observed quantiles avoid artificial boundaries and keep repeated dates together.
    distinct = np.unique(numeric)
    boundaries = (
        distinct[1:]
        if len(distinct) <= n_bins
        else np.unique(
            np.quantile(numeric, np.arange(1, n_bins) / n_bins, method="higher")
        )
    )
    boundaries = boundaries[boundaries > numeric.min()]
    codes = np.searchsorted(boundaries, numeric, side="right")
    actual, expected, weight = totals(data, run.config, run.fit.predict(data))
    overall: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    series: list[dict[str, str]] = []
    factor_rows: dict[str, dict[str, Any]] = {}
    if variable:
        template = ae_by_variable(
            data, variable, actual, expected, weight, **(grouping or {})
        )
        factor_rows = {
            r["label"]: {"label": r["label"], "exposure": r["exposure"]}
            for r in template.to_dicts()
        }
    for index, code in enumerate(np.unique(codes)):
        selected = codes == code
        time_slice = times.filter(pl.Series(selected))
        lo, hi = str(time_slice.min()), str(time_slice.max())
        label = lo if lo == hi else f"{lo} – {hi}"
        a, e, w = (
            float(actual[selected].sum()),
            float(expected[selected].sum()),
            float(weight[selected].sum()),
        )
        ratio = a / e if e > 0 else None
        overall.append(
            {
                "label": label,
                "rows": int(selected.sum()),
                "exposure": w,
                "actual": a,
                "expected": e,
                "ae": ratio,
            }
        )
        if variable:
            key = f"period_{index}"
            series.append({"key": key, "label": label})
            cells = ae_by_variable(
                data.filter(pl.Series(selected)),
                variable,
                actual[selected],
                expected[selected],
                weight[selected],
                **(grouping or {}),
            )
            for cell in cells.to_dicts():
                cell_ratio = (
                    cell["actual"] / cell["expected"]
                    if cell["expected"] > 0 and cell["exposure"] > 0
                    else None
                )
                normalised = (
                    cell_ratio / ratio if cell_ratio is not None and ratio else None
                )
                comparisons.append(
                    {
                        **cell,
                        "period": label,
                        "ae": cell_ratio,
                        "relative_ae": normalised,
                    }
                )
                factor_rows[cell["label"]][key] = cell_ratio
                factor_rows[cell["label"]][key + "_relative"] = normalised
    return {
        "time_column": column,
        "synthetic_time": project.exploration.get("example", {}).get(
            "synthetic_time_column"
        )
        == column,
        "requested_bands": n_bins,
        "bands": len(overall),
        "rows": data.height,
        "excluded_rows": excluded,
        "overall": overall,
        "variable": variable,
        "factors": list(run.spec.main_effects),
        "factor_rows": list(factor_rows.values()),
        "cells": comparisons,
        "series": series,
    }
