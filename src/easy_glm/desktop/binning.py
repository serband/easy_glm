"""Variables-page projection of the project's canonical numeric design settings."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from easy_glm.core.design import LinearEncoder, StepEncoder, format_knot
from easy_glm.workflow.prep import prepare, train_holdout
from easy_glm.workflow.project import Project, VariableDesign
from easy_glm.workflow.run import encoder_for, integer_knots


def _count(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= 200:
        raise ValueError(f"{where} must be a whole number from 2 to 200.")
    return value


def _override(vd: VariableDesign | None) -> dict[str, Any] | None:
    if vd is None or (vd.knots == "quantile" and vd.n_bins is None):
        return None
    if isinstance(vd.knots, list):
        return {"method": "cuts", "cuts": list(vd.knots)}
    if vd.knots == "integer":
        result: dict[str, Any] = {"method": "integer"}
        if vd.n_bins is not None:
            result["fallback_bins"] = vd.n_bins
        return result
    return {"method": "quantile", "bins": vd.n_bins}


def binning_setup(project: Project, raw_columns: list[str]) -> dict[str, Any]:
    """Represent every managed override by raw source name, including inactive ones."""
    overrides: dict[str, dict[str, Any]] = {}
    for raw_name in raw_columns:
        final = project.data.renames.get(raw_name, raw_name)
        setting = _override(project.design.variables.get(final))
        if setting is not None:
            overrides[raw_name] = setting
    return {"default_bins": project.design.defaults.n_bins, "overrides": overrides}


def binning_columns(project: Project, raw: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Sparse explicit design metadata; ordinary kinds come from the column dtype."""
    result: dict[str, dict[str, Any]] = {}
    for source, dtype in raw.schema.items():
        name = project.data.renames.get(source, source)
        design = project.design.variables.get(name, VariableDesign())
        if design.kind is None and design.clamp is None:
            continue
        numeric = project.data.types.get(name, "auto") == "numeric" or (
            project.data.types.get(name, "auto") == "auto" and dtype.is_numeric()
        )
        kind = design.kind or ("step" if numeric else "categorical")
        result[source] = {
            "kind": kind,
            "clamp": list(design.clamp) if design.clamp is not None else None,
            "active": kind in ("step", "linear"),
        }
    return result


def _parse_override(
    raw_name: str, value: Any
) -> tuple[str, int | None, list[float] | None] | None:
    if value is None:
        return None
    where = f"binning.overrides.{raw_name}"
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object or null.")
    method = value.get("method")
    allowed = {
        "quantile": {"method", "bins"},
        "integer": {"method", "fallback_bins"},
        "cuts": {"method", "cuts"},
    }
    if method not in allowed:
        raise ValueError(f"{where}.method must be 'quantile', 'integer' or 'cuts'.")
    extra = set(value) - allowed[method]
    if extra:
        raise ValueError(
            f"{where} has conflicting or unknown field(s): {', '.join(sorted(extra))}."
        )
    if method == "quantile":
        if "bins" not in value:
            raise ValueError(f"{where}.bins is required.")
        return "quantile", _count(value["bins"], f"{where}.bins"), None
    if method == "integer":
        bins = value.get("fallback_bins")
        return (
            "integer",
            _count(bins, f"{where}.fallback_bins") if bins is not None else None,
            None,
        )
    cuts = value.get("cuts")
    if not isinstance(cuts, list):
        raise ValueError(f"{where}.cuts must be a list of numbers.")
    if any(
        isinstance(c, bool) or not isinstance(c, int | float) or not math.isfinite(c)
        for c in cuts
    ):
        raise ValueError(f"{where}.cuts must contain only finite numbers.")
    numbers = [float(c) for c in cuts]
    if any(left >= right for left, right in zip(numbers, numbers[1:], strict=False)):
        raise ValueError(
            f"{where}.cuts must be strictly increasing without duplicates."
        )
    return "cuts", None, numbers


def apply_binning_setup(project: Project, raw_columns: list[str], setup: Any) -> None:
    """Replace only managed binning fields after role and rename edits."""
    if not isinstance(setup, dict) or set(setup) != {"default_bins", "overrides"}:
        raise ValueError("binning must contain default_bins and overrides objects.")
    default = _count(setup["default_bins"], "binning.default_bins")
    overrides = setup["overrides"]
    if not isinstance(overrides, dict):
        raise ValueError("binning.overrides must be an object.")
    unknown = set(overrides) - set(raw_columns)
    if unknown:
        raise ValueError(
            f"Unknown raw binning column(s): {', '.join(sorted(unknown))}."
        )
    parsed = {raw: _parse_override(raw, entry) for raw, entry in overrides.items()}
    project.design.defaults.n_bins = default
    for raw in raw_columns:
        final = project.data.renames.get(raw, raw)
        vd = project.design.variables.get(final)
        setting = parsed.get(raw)
        if vd is None and setting is None:
            continue
        if vd is None:
            vd = project.design.variables[final] = VariableDesign()
        if setting is None:
            vd.knots, vd.n_bins = "quantile", None
        else:
            method, count, cuts = setting
            vd.knots = cuts if method == "cuts" else method
            vd.n_bins = count


def binning_changes(
    before: Project, after: Project, raw_columns: list[str]
) -> list[dict[str, str]]:
    old = binning_setup(before, raw_columns)
    new = binning_setup(after, raw_columns)
    changes: list[dict[str, str]] = []
    if old["default_bins"] != new["default_bins"]:
        changes.append(
            {
                "raw column": "All numeric predictors",
                "name": "Default bins",
                "role": "binning",
                "type": f"{old['default_bins']} → {new['default_bins']}",
            }
        )
    for raw in raw_columns:
        former = old["overrides"].get(raw)
        latter = new["overrides"].get(raw)
        if former != latter:
            changes.append(
                {
                    "raw column": raw,
                    "name": raw,
                    "role": "binning",
                    "type": f"{former or 'default'} → {latter or 'default'}",
                }
            )
    return changes


def validate_changed_linear_binning(
    before: Project, after: Project, raw: pl.DataFrame
) -> None:
    """Reject newly entered linear cuts that fitting would silently discard."""
    changed_raw = {
        item["raw column"] for item in binning_changes(before, after, raw.columns)
    }
    training_needed: list[tuple[str, VariableDesign]] = []
    for source in changed_raw & set(raw.columns):
        name = after.data.renames.get(source, source)
        design = after.design.variables.get(name)
        if design is None or not isinstance(design.knots, list):
            continue
        numeric = after.data.types.get(name, "auto") == "numeric" or (
            after.data.types.get(name, "auto") == "auto"
            and raw.schema[source].is_numeric()
        )
        kind = design.kind or ("step" if numeric else "categorical")
        if kind == "step" and not design.knots:
            raise ValueError(f"Custom cuts for step factor {name!r} cannot be empty.")
        if kind != "linear":
            continue
        if design.clamp is not None:
            lo, hi = design.clamp
            outside = [cut for cut in design.knots if not lo < cut < hi]
            if outside:
                raise ValueError(
                    f"Linear cuts {outside} for {name!r} must lie strictly "
                    f"inside the effective clamp ({lo}, {hi})."
                )
        elif design.knots:
            training_needed.append((name, design))
    if not training_needed:
        return
    try:
        training, _ = train_holdout(prepare(after, raw), after.data.split)
    except (ValueError, KeyError, pl.exceptions.PolarsError):
        # Saving a draft does not require a usable existing training split.
        # The data preview verifies it before showing boundaries.
        return
    from easy_glm.core.design import round_range_outward

    for name, design in training_needed:
        if name not in training:
            continue
        values = training[name].cast(pl.Float64).to_numpy()
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        lo, hi = round_range_outward(float(finite.min()), float(finite.max()))
        outside = [cut for cut in design.knots if not lo < cut < hi]
        if outside:
            raise ValueError(
                f"Linear cuts {outside} for {name!r} must lie strictly "
                f"inside the effective clamp ({lo}, {hi})."
            )


def binning_preview(project: Project, raw: pl.DataFrame, column: str) -> dict[str, Any]:
    """Compute one draft column's bands from the same prepared training data as fit."""
    if column not in raw.columns:
        raise ValueError(f"Unknown raw source column {column!r}.")
    name = project.data.renames.get(column, column)
    frame = prepare(project, raw)
    training, _ = train_holdout(frame, project.data.split)
    if not training.height:
        raise ValueError("There are no training rows for a binning preview.")
    if name not in training:
        raise ValueError(f"Prepared column {name!r} does not exist.")
    series = training[name]
    vd = project.design.variables.get(name, VariableDesign())
    kind = vd.kind or ("step" if series.dtype.is_numeric() else "categorical")
    active = kind in ("step", "linear")
    requested = vd.n_bins or project.design.defaults.n_bins
    if isinstance(vd.knots, list) and active:
        requested = len(vd.knots) + (3 if kind == "linear" else 1)
    result: dict[str, Any] = {
        "column": column,
        "name": name,
        "kind": kind,
        "requested_bins": requested if active else None,
        "actual_bins": None,
        "rows": [],
        "missing_rows": series.null_count(),
        "nonfinite_rows": 0,
        "training_rows": training.height,
        "warnings": [],
        "active": active,
    }
    if not active:
        result["warnings"].append(
            f"Saved numeric binning is inactive for a {kind} factor."
        )
        return result
    if not series.dtype.is_numeric():
        raise ValueError(f"{name!r} is not numeric after variable preparation.")
    values = series.cast(pl.Float64).to_numpy()
    nan = np.isnan(values)
    result["missing_rows"] = int(nan.sum())
    result["nonfinite_rows"] = int(
        (~np.isfinite(values) & ~series.is_null().to_numpy()).sum()
    )
    if result["nonfinite_rows"]:
        result["warnings"].append(
            f"{result['nonfinite_rows']} non-finite training value(s) are shown separately; the encoder still assigns them."
        )
    if not np.isfinite(values).any() and not isinstance(vd.knots, list):
        result["warnings"].append(
            "No finite training values; no numeric bands can be derived."
        )
        return result
    if not np.isfinite(values).any():
        result["warnings"].append(
            "No finite training values; custom cuts have no observed finite rows."
        )
        if kind == "linear":
            return result
    if np.unique(values[np.isfinite(values)]).size == 1:
        result["warnings"].append(
            "The training column is constant; automatic cuts cannot separate its rows."
        )
    integer_fallback = False
    if vd.knots == "integer":
        if integer_knots(series, project.design.defaults.max_integer_knots) is None:
            integer_fallback = True
            result["warnings"].append(
                f"Integer cuts exceed the configured range; using {requested} quantile bins instead."
            )
    if isinstance(vd.knots, list) and kind == "linear":
        # encoder_for's constructor otherwise drops out-of-clamp knots.
        from easy_glm.core.design import round_range_outward

        finite = values[np.isfinite(values)]
        lo, hi = vd.clamp or round_range_outward(
            float(finite.min()), float(finite.max())
        )
        bad = [cut for cut in vd.knots if not lo < cut < hi]
        if bad:
            raise ValueError(
                f"Linear cuts {bad} must lie strictly inside the effective clamp ({lo}, {hi})."
            )
    try:
        encoder = encoder_for(name, series, vd, project)
    except ValueError as exc:
        result["warnings"].append(str(exc))
        return result
    assert isinstance(encoder, StepEncoder | LinearEncoder)
    bins = encoder.bins()
    indices = encoder.row_index(series)
    exposure_col = project.exposure
    weights = (
        training[exposure_col].cast(pl.Float64).to_numpy()
        if exposure_col and exposure_col in training
        else None
    )
    for index, (lo, hi) in enumerate(bins):
        mask = indices == index
        label = (
            f"[{format_knot(lo) if lo is not None else '-∞'}, "
            f"{format_knot(hi) if hi is not None else '∞'})"
        )
        count = int(mask.sum())
        result["rows"].append(
            {
                "label": label,
                "lower": lo,
                "upper": hi,
                "rows": count,
                "exposure": (
                    float(np.sum(weights[mask])) if weights is not None else None
                ),
            }
        )
    result["actual_bins"] = len(bins)
    if any(row["rows"] == 0 for row in result["rows"]):
        result["warnings"].append("Some bands contain no training rows.")
    if (
        active
        and result["actual_bins"] < requested
        and (vd.knots == "quantile" or integer_fallback)
    ):
        result["warnings"].append("Tied values reduced the actual number of bins.")
    return result
