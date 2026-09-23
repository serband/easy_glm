"""Subprocess entry: call existing preparation, fitting and diagnostics functions."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Represent undefined diagnostics as JSON null, never NaN/Infinity."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
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


def _write_optional_pickle(path: Path, value: Any) -> None:
    """Replace a private acceleration cache without masking a fit outcome."""
    import pickle

    temporary = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=5)
        os.replace(temporary, path)
    except OSError as exc:
        try:
            print(
                f"Could not update optional fit cache {path}: {exc}",
                file=sys.stderr,
            )
        except (OSError, ValueError):
            pass
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def fit_result(
    project: Any,
    raw: Any,
    name: str,
    progress: Any,
    artifact: Path | None = None,
    main_cache_path: Path | None = None,
    pair_cache_path: Path | None = None,
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
    import pickle

    reuse = {"main_effects": False, "fold_predictions": False}

    def fit_progress(message: str) -> None:
        if "Reusing unchanged main-effects fit" in message:
            reuse["main_effects"] = True
        if "reusing main-effects fold predictions" in message:
            reuse["fold_predictions"] = True
        stage = re.search(r"Pair stage (\d+)/(\d+)", message)
        fold = re.search(r"fold (\d+)/(\d+)", message)
        trial = re.search(r"(?<!prefix )trial (\d+)/(\d+)", message)
        prefix_trial = re.search(r"prefix trial (\d+)/(\d+)", message)
        inner_fold = re.search(r"inner (\d+)/(\d+)", message)
        if stage:
            progress(
                {
                    "message": message,
                    "stage_number": int(stage.group(1)) + 1,
                    "stage_total": int(stage.group(2)) + 1,
                    "fold": int(fold.group(1)) if fold else None,
                    "folds": int(fold.group(2)) if fold else None,
                    "trial": int(trial.group(1)) if trial else None,
                    "trials": int(trial.group(2)) if trial else None,
                    "prefix_trial": (
                        int(prefix_trial.group(1)) if prefix_trial else None
                    ),
                    "prefix_trials": (
                        int(prefix_trial.group(2)) if prefix_trial else None
                    ),
                    "inner_fold": int(inner_fold.group(1)) if inner_fold else None,
                    "inner_folds": int(inner_fold.group(2)) if inner_fold else None,
                    "prefix_reused": "reused full-training prefix" in message,
                    "cv_pending": bool(fold),
                }
            )
        else:
            progress(message)

    main_cache: dict[str, Any] | None = None
    if main_cache_path is not None:
        main_cache = {}
        try:
            with main_cache_path.open("rb") as handle:
                saved_cache = pickle.load(handle)  # Private, local worker cache only.
            if saved_cache.get("format") == 2 and isinstance(
                saved_cache.get("cache"), dict
            ):
                main_cache = saved_cache["cache"]
        except (
            OSError,
            EOFError,
            pickle.UnpicklingError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            ImportError,
        ):
            pass
    pair_cache: dict[str, Any] = {"full_prefix": {}}
    if pair_cache_path is not None and project.models[name].pair_stages:
        try:
            with pair_cache_path.open("rb") as handle:
                saved_pair_cache = pickle.load(handle)
            if saved_pair_cache.get("format") == 3 and isinstance(
                saved_pair_cache.get("cache", {}).get("full_prefix"), dict
            ):
                pair_cache = saved_pair_cache["cache"]
        except (
            OSError,
            EOFError,
            pickle.UnpicklingError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            ImportError,
        ):
            pass
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            run = run_model(
                project,
                frame,
                name,
                progress=fit_progress,
                main_effects_cache=main_cache,
                pair_stages_cache=pair_cache,
            )
        finally:
            if pair_cache_path is not None and project.models[name].pair_stages:
                _write_optional_pickle(
                    pair_cache_path, {"format": 3, "cache": pair_cache}
                )
    if main_cache_path is not None:
        _write_optional_pickle(main_cache_path, {"format": 2, "cache": main_cache})
    if artifact is not None:
        with (artifact / "fit.pkl").open("wb") as handle:
            pickle.dump(run, handle)
        from easy_glm.desktop.ae_cache import build_packet

        progress("Preparing fitted-variable diagnostics…")
        try:
            build_packet(project, run, frame, artifact, {"model": name})
        except Exception:
            # Optional acceleration must never turn a successful fit into failure.
            progress("Fitted diagnostics will be prepared on demand.")
        from easy_glm.desktop.importance_cache import build_packet as build_importance

        progress("Preparing variable importance…")
        try:
            build_importance(project, run, frame, artifact)
        except Exception:
            progress("Variable importance will be prepared on demand.")
    return {
        **result_for(project, frame, run, [str(w.message) for w in caught]),
        "reuse": reuse,
    }


def table_payload(rate_model: Any, variable: str, frame: Any) -> dict[str, Any]:
    """Keep canonical row order and expose interaction parent labels explicitly."""
    from easy_glm.engine.models import level_label

    rows = frame.to_dicts()
    cfg = rate_model.variables[variable]
    if cfg.type == "interaction":
        a, b = cfg.parents
        labels = [
            {
                (r.from_, r.to_): level_label(
                    r, rate_model.variables[parent].other_label
                )
                for r in rate_model.variables[parent].table
            }
            for parent in (a, b)
        ]
        for row, cell in zip(rows, cfg.table, strict=True):
            row["label_a"] = labels[0][(cell.from_a, cell.to_a)]
            row["label_b"] = labels[1][(cell.from_b, cell.to_b)]
    return {"columns": list(rows[0]) if rows else frame.columns, "rows": rows}


def result_for(
    project: Any, frame: Any, run: Any, notices: list[str] | None = None
) -> dict[str, Any]:
    from easy_glm.core.excel import rate_model_tables
    from easy_glm.desktop.diagnostic_views import metadata
    from easy_glm.workflow.diagnostics import lift_table, model_metrics, totals
    from easy_glm.workflow.prep import train_holdout
    from easy_glm.workflow.run import null_model_predict

    train, holdout = train_holdout(frame, project.data.split)
    charts = {}
    frames = {
        name: part
        for name, part in (("train", train), ("holdout", holdout), ("all", frame))
        if part.height
    }
    predictions = {name: run.predict(part) for name, part in frames.items()}
    null_predictions = {
        name: null_model_predict(project, run.config, train, part)
        for name, part in frames.items()
    }
    run.metrics = model_metrics(
        run.fit, predictions, frames, run.config, null_predictions
    )
    for subset, part in frames.items():
        actual, expected, weight = totals(part, run.config, predictions[subset])
        charts[subset] = lift_table(actual, expected, weight).to_dicts()
    tables = {
        key: table_payload(run.rate_model, key, table)
        for key, table in rate_model_tables(run.rate_model).items()
    }
    pair_tables: dict[str, Any] = {}
    if getattr(run.rate_model, "pair_tables", None):
        from easy_glm.core.excel import pair_table_frames

        fitted_by_id = {
            stage.stage_id: stage.table for stage in getattr(run, "pair_stages", [])
        }
        for stage_id, table in pair_table_frames(run.rate_model).items():
            rows = table.to_dicts()
            fitted = fitted_by_id.get(stage_id)
            if fitted is not None:
                values = {
                    (cell.axis_a_row, cell.axis_b_row): cell.relativity
                    for cell in fitted.cells
                }
                for row in rows:
                    row["fitted"] = values.get(
                        (row["axis_a_row"], row["axis_b_row"]), 1.0
                    )
            pair_tables[stage_id] = {
                "columns": list(rows[0]) if rows else list(table.columns),
                "rows": rows,
            }
    pair_stages = []
    configured_stages = {
        configured.stage_id: configured
        for configured in getattr(run.config, "pair_stages", [])
    }
    for stage in getattr(run, "pair_stages", []):
        table = stage.table
        cells = table.cells
        pair_stages.append(
            {
                "stage_id": stage.stage_id,
                "search": (
                    asdict(configured_stages[stage.stage_id].search)
                    if stage.stage_id in configured_stages
                    and configured_stages[stage.stage_id].search is not None
                    else None
                ),
                "parents": list(stage.parents),
                "baseline_stage_ids": list(stage.baseline_stage_ids),
                "chosen_candidate": (
                    asdict(stage.chosen_candidate)
                    if is_dataclass(stage.chosen_candidate)
                    else stage.chosen_candidate
                ),
                "prefix_fingerprint": stage.prefix_fingerprint,
                "cv_candidates": [asdict(item) for item in stage.cv_candidates],
                "search_trials": [
                    asdict(item) for item in getattr(stage, "search_trials", ())
                ],
                "prefix_search_trials": [
                    asdict(item) for item in getattr(stage, "prefix_search_trials", ())
                ],
                "prefix_cv_loss": stage.prefix_cv_loss,
                "table_cv_loss": stage.table_cv_loss,
                "teacher_cv_loss": stage.teacher_cv_loss,
                "approximation_loss": stage.approximation_loss,
                "observed_table_minus_teacher_cv_loss": getattr(
                    stage, "observed_table_minus_teacher_cv_loss", None
                ),
                "training_teacher_loss": stage.training_teacher_loss,
                "training_table_loss": stage.training_table_loss,
                "fit_seconds": stage.fit_seconds,
                "reused": stage.reused,
                "status": stage.status,
                "dimensions": [len(axis.table) for axis in table.axes],
                "support": {
                    "supported_cells": sum(
                        cell.fallback_reason is None for cell in cells
                    ),
                    "total_cells": len(cells),
                    "training_weight_share": sum(
                        cell.weight_share
                        for cell in cells
                        if cell.fallback_reason is None
                    ),
                },
            }
        )
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
                    "time",
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
            "pair_tables": pair_tables,
            "pair_stages": pair_stages,
            "base_rate": run.rate_model.base_rate,
            "link": run.fit.link,
            "relativity_label": run.rate_model.relativity_label,
            "relativity_note": run.rate_model.relativity_note,
            "dropped_predictors": run.dropped_predictors,
            "warnings": list(dict.fromkeys(notices or [])),
        }
    )


def main() -> None:
    from easy_glm.desktop.fit_progress import reserve_progress_stdout

    progress = reserve_progress_stdout()
    import polars as pl

    from easy_glm.workflow.project import Project

    folder, name = Path(sys.argv[1]), sys.argv[2]

    try:
        result = fit_result(
            Project.from_json(folder / "project.json"),
            pl.read_parquet(folder / "raw.parquet"),
            name,
            progress,
            folder,
            Path(sys.argv[3]) if len(sys.argv) > 3 else None,
            Path(sys.argv[4]) if len(sys.argv) > 4 else None,
        )
    except (
        Exception
    ) as exc:  # process boundary; actionable message is the public result
        result = {"error": str(exc)}
    try:
        write_json(folder / "result.json", result)
    finally:
        progress.close()


if __name__ == "__main__":
    main()
