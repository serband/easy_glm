"""Small fitted-factor aggregates, keyed by immutable fit and applied scoring basis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FORMAT = 1


def eligible_variables(run: Any) -> list[str]:
    """Bound background work by fitted groups, independent of raw schema width."""
    names: list[str] = []
    groups = 0
    for name in run.spec.main_effects:
        count = run.tables[name].height
        if count > 500 or groups + count > 10000 or len(names) >= 128:
            continue
        names.append(name)
        groups += count
    return names


def _basis(project: Any, source: Path, name: str, *, original: bool = False) -> dict:
    spec = project.to_dict()
    model = dict(spec["models"][name])
    for field in ("notes", "snapshots"):
        model.pop(field, None)
    if original:
        model.pop("adjustments", None)
        model.pop("base_rate_override", None)
    files = {}
    for filename in ("fit.pkl", "raw.parquet"):
        stat = (source / filename).stat()
        files[filename] = [stat.st_size, stat.st_mtime_ns]
    return {
        "format": FORMAT,
        "fit_id": source.name,
        "files": files,
        "data": spec["data"],
        "design": spec["design"],
        "model": model,
    }


def cache_path(
    project: Any, source: Path, request: dict, *, original: bool = False
) -> Path:
    basis = {"selected": _basis(project, source, request["model"], original=original)}
    if not original and request.get("challenger"):
        basis["challenger"] = _basis(
            project, Path(request["_challenger_source"]), request["challenger"]
        )
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()
    return (
        source / "ae-cache" / f"{'original' if original else 'adjusted'}-{digest}.json"
    )


def read_packet(project: Any, source: Path, request: dict) -> dict | None:
    if request.get("action") != "variable":
        return None
    try:
        packet = json.loads(cache_path(project, source, request).read_text())
        return packet if request.get("variable") in packet["variables"] else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def variable_view(packet: dict, request: dict) -> dict:
    variable = request["variable"]
    item = packet["variables"][variable]
    subset = request.get("subset", "train")
    subsets = item["subsets"]
    if subset not in subsets:
        raise ValueError("The selected subset has no rows.")
    both = request.get("options", {}).get("both_subsets", False)
    return {
        "rows": subsets[subset],
        "kind": item["kind"],
        "subset": subset,
        "book_impact": packet["book_impact"],
        "ae_sets": [
            {"title": f"{variable} · {s}", "rows": subsets[s]}
            for s in ("train", "holdout")
            if both and s != subset and s in subsets
        ],
        "ae_cache": packet,
    }


def build_packet(
    project: Any,
    run: Any,
    frame: Any,
    source: Path,
    request: dict,
    challenger: Any = None,
) -> dict:
    """Score each model once; group only fitted main effects, never raw columns/pairs."""
    import numpy as np
    import polars as pl

    from easy_glm.desktop.fit_worker import json_safe, write_json
    from easy_glm.desktop.review_worker import grouping
    from easy_glm.workflow.diagnostics import ae_by_variable, totals
    from easy_glm.workflow.run import rate_model_for

    names = [name for name in eligible_variables(run) if name in frame.columns]
    masks = {
        "train": frame[project.data.split.column].to_numpy() == 1,
        "holdout": frame[project.data.split.column].to_numpy() == 0,
        "all": np.ones(frame.height, dtype=bool),
    }
    actual, expected, weight = totals(frame, run.config, run.predict(frame))
    original_path = cache_path(project, source, request, original=True)
    try:
        original = json.loads(original_path.read_text())
    except (OSError, ValueError):
        fitted = rate_model_for(project, run, [], base_rate_override=None)
        fitted_expected = totals(
            frame, run.config, fitted.predict(frame, exposure_col=None)
        )[1]
        original = {
            "variables": {},
            "expected": float(np.sum(fitted_expected[masks["train"]])),
        }
        for name in names:
            original["variables"][name] = {}
            for subset, mask in masks.items():
                if not mask.any():
                    continue
                original["variables"][name][subset] = ae_by_variable(
                    frame.select(name).filter(pl.Series(mask)),
                    name,
                    actual[mask],
                    fitted_expected[mask],
                    weight[mask],
                    **grouping(run, name),
                ).to_dicts()
        original_path.parent.mkdir(exist_ok=True)
        write_json(original_path, original)
    other = None
    if challenger is not None:
        from easy_glm.desktop.diagnostic_views import compatible

        compatible(run, challenger)
        other = totals(frame, challenger.config, challenger.predict(frame))[1]
    packet: dict[str, Any] = {
        "variables": {},
        "book_impact": {
            "current": float(np.sum(expected[masks["train"]])),
            "fitted": original["expected"],
        },
    }
    for name in names:
        packet["variables"][name] = {
            "kind": run.rate_model.variables[name].type,
            "subsets": {},
        }
        for subset, mask in masks.items():
            if not mask.any():
                continue
            part = frame.select(name).filter(pl.Series(mask))
            table = ae_by_variable(
                part,
                name,
                actual[mask],
                expected[mask],
                weight[mask],
                **grouping(run, name),
            )
            fitted_rows = original["variables"][name][subset]
            table = table.with_columns(
                *[
                    pl.Series(dst, [r[src] for r in fitted_rows])
                    for dst, src in [
                        ("fitted_rate", "expected_rate"),
                        ("fitted_ae", "ae"),
                        ("fitted_expected", "expected"),
                    ]
                ]
            )
            if other is not None:
                grouped = ae_by_variable(
                    part,
                    name,
                    actual[mask],
                    other[mask],
                    weight[mask],
                    **grouping(run, name),
                )
                table = table.with_columns(
                    pl.Series("challenger_rate", grouped["expected_rate"])
                )
            packet["variables"][name]["subsets"][subset] = table.to_dicts()
    path = cache_path(project, source, request)
    path.parent.mkdir(exist_ok=True)
    write_json(path, packet)
    # Bound adjusted versions; the immutable original is reused after edits.
    for old in sorted(
        path.parent.glob("adjusted-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[8:]:
        old.unlink(missing_ok=True)
    return json_safe(packet)
