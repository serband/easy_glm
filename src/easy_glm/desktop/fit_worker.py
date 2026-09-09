"""Subprocess entry: call existing preparation, fitting and diagnostics functions."""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Represent undefined diagnostics as JSON null, never NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def fit_result(
    project: Any, raw: Any, name: str, progress: Any, artifact: Path | None = None
) -> dict[str, Any]:
    """A JSON result, including current adjusted tables, from the canonical engine."""
    from easy_glm.workflow.prep import prepare
    from easy_glm.workflow.run import run_model

    progress("Preparing full data and train/holdout split…")
    frame = prepare(project, raw)
    problems = project.validate(name, columns=frame.columns)
    if problems:
        raise ValueError("; ".join(problems))
    progress("Building design and fitting model…")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = run_model(project, frame, name, progress=progress)
    if artifact is not None:
        import pickle

        with (artifact / "fit.pkl").open("wb") as handle:
            pickle.dump(run, handle)
    return result_for(project, frame, run, [str(w.message) for w in caught])


def result_for(
    project: Any, frame: Any, run: Any, notices: list[str] | None = None
) -> dict[str, Any]:
    from easy_glm.core.excel import rate_model_tables
    from easy_glm.desktop.diagnostic_views import metadata, safe_gini
    from easy_glm.workflow.diagnostics import lift_table, totals
    from easy_glm.workflow.prep import train_holdout

    train, holdout = train_holdout(frame, project.data.split)
    charts = {}
    base_metrics = [run.metrics[k] for k in ("train", "holdout") if k in run.metrics]
    combined = {
        k: sum(m[k] for m in base_metrics)
        for k in ("rows", "exposure", "actual", "expected", "deviance", "null_deviance")
    }
    combined["ae"] = (
        combined["actual"] / combined["expected"] if combined["expected"] > 0 else None
    )
    combined["deviance_explained"] = (
        1 - combined["deviance"] / combined["null_deviance"]
        if combined["null_deviance"] > 0
        else None
    )
    denominator = combined["exposure"] if run.config.weight else combined["rows"]
    combined["mean_deviance"] = (
        combined["deviance"] / denominator if denominator else None
    )
    run.metrics["all"] = combined
    for subset, part in (("train", train), ("holdout", holdout), ("all", frame)):
        if part.height:
            actual, expected, weight = totals(part, run.config, run.predict(part))
            run.metrics[subset]["gini"] = safe_gini(actual, expected, weight)
            charts[subset] = lift_table(actual, expected, weight).to_dicts()
    tables = {
        key: {"columns": table.columns, "rows": table.to_dicts()}
        for key, table in rate_model_tables(run.rate_model).items()
    }
    return json_safe(
        {
            "review_variables": [
                c
                for c in frame.columns
                if project.data.roles.get(c)
                not in (
                    "target",
                    "weight",
                    "exposure",
                    "offset",
                    "current_premium",
                    "id",
                    "split",
                    "ignore",
                )
                and c
                not in (
                    project.data.split.column,
                    run.config.target,
                    run.config.weight,
                    run.config.offset,
                )
            ],
            "diagnostic_info": metadata(run, frame),
            "summary": run.summary(),
            "metrics": run.metrics,
            "lift": charts,
            "tables": tables,
            "base_rate": run.rate_model.base_rate,
            "link": run.fit.link,
            "relativity_label": run.rate_model.relativity_label,
            "relativity_note": run.rate_model.relativity_note,
            "dropped_predictors": run.dropped_predictors,
            "warnings": list(dict.fromkeys(notices or [])),
        }
    )


def main() -> None:
    import polars as pl

    from easy_glm.workflow.project import Project

    folder, name = Path(sys.argv[1]), sys.argv[2]

    def progress(message: str) -> None:
        write_json(folder / "progress.json", {"message": message})

    try:
        result = fit_result(
            Project.from_json(folder / "project.json"),
            pl.read_parquet(folder / "raw.parquet"),
            name,
            progress,
            folder,
        )
    except (
        Exception
    ) as exc:  # process boundary; actionable message is the public result
        result = {"error": str(exc)}
    write_json(folder / "result.json", result)


if __name__ == "__main__":
    main()
