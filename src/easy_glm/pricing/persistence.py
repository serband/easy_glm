"""Portable, JSON-only persistence for pricing workflow checkpoints."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl

from easy_glm.core.excel import rate_model_tables
from easy_glm.engine import RateModel
from easy_glm.workflow import ModelRun, Project

BUNDLE_FORMAT = "easy-glm-pricing-model"
BUNDLE_VERSION = 1


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_model(model: Any, path: str | Path) -> Path:
    """Save settings, evidence and the exact deployed scorer without policy data."""
    destination = Path(path)
    payload = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_VERSION,
        "name": model.name,
        "project": model._project.to_dict(),
        "rate_model": model._run.rate_model.to_dict(),
        "run": {
            "metrics": model._run.metrics,
            "created_at": model._run.created_at,
            "train_rows": model._run.train_rows,
            "holdout_rows": model._run.holdout_rows,
            "dropped_predictors": model._run.dropped_predictors,
            "pair_stages": model._run.pair_stages,
        },
        "history": model._history,
        "evidence": model._evidence,
        "frozen_stages": model._frozen_stages,
        "data_fingerprint": model._evidence.get("data_fingerprint")
        or getattr(model._session, "_data_fingerprint", None),
    }
    destination.write_text(
        json.dumps(_json_value(payload), indent=2, ensure_ascii=False, allow_nan=False)
    )
    return destination


def _read_data(data: pl.DataFrame | str | Path | None) -> pl.DataFrame | None:
    if data is None or isinstance(data, pl.DataFrame):
        return data
    path = Path(data)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(path)
    raise ValueError(
        "Saved-model analysis data must be a Polars frame, Parquet, or CSV"
    )


class _LoadedSession:
    """Minimum session state used when a full PricingSession is unavailable."""

    def __init__(self, data: pl.DataFrame | None, project: Project) -> None:
        self._data = data
        self._project = project
        self._main_cache: dict[str, Any] = {}
        self._pair_cache: dict[str, Any] = {}


def load_model(
    path: str | Path, *, data: pl.DataFrame | str | Path | None = None
) -> Any:
    """Load a frozen checkpoint; attach policy data only when analysis is needed."""
    raw = json.loads(Path(path).read_text())
    if raw.get("format") != BUNDLE_FORMAT:
        raise ValueError("This is not an EasyGLM pricing-model JSON file")
    if raw.get("format_version") != BUNDLE_VERSION:
        raise ValueError(
            f"Unsupported pricing-model format {raw.get('format_version')!r}; expected {BUNDLE_VERSION}"
        )
    project = Project.from_dict(raw["project"])
    name = str(raw["name"])
    if name not in project.models:
        raise ValueError(f"Saved project does not contain model {name!r}")
    scorer = RateModel.from_dict(raw["rate_model"])
    run_data = raw.get("run", {})
    tables_by_stage = {table.stage_id: table for table in scorer.pair_tables}
    pair_stages = []
    for saved in run_data.get("pair_stages", []):
        values = dict(saved)
        stage_id = str(values.get("stage_id", ""))
        if stage_id in tables_by_stage:
            values["table"] = tables_by_stage[stage_id]
        pair_stages.append(SimpleNamespace(**values))
    run = ModelRun(
        name=name,
        config=project.models[name],
        # A frozen scorer has no fit matrix; refitting starts from saved settings.
        spec=None,  # type: ignore[arg-type]
        fit=None,  # type: ignore[arg-type]
        rate_model=scorer,
        tables=rate_model_tables(scorer),
        metrics=run_data.get("metrics", {}),
        project_snapshot=project.to_dict(),
        created_at=run_data.get("created_at", ""),
        train_rows=int(run_data.get("train_rows", 0)),
        holdout_rows=int(run_data.get("holdout_rows", 0)),
        dropped_predictors=list(run_data.get("dropped_predictors", [])),
        pair_stages=pair_stages,
    )
    attached = _read_data(data)
    from .session import PricingSession

    session: Any
    if attached is not None:
        session = PricingSession._from_saved(attached, project)
        saved_identity = raw.get("data_fingerprint") or raw.get("evidence", {}).get(
            "data_fingerprint"
        )
        if saved_identity is not None and session._data_fingerprint != saved_identity:
            raise ValueError(
                "The supplied analysis data does not match the modelling data saved with this model. "
                "Load without data to score new policies, or supply the original modelling data."
            )
    else:
        session = _LoadedSession(attached, project)
    from .model import PricingModel

    return PricingModel(
        session,
        project,
        run,
        history=list(raw.get("history", [])),
        evidence=dict(raw.get("evidence", {})),
        frozen_stages=list(raw.get("frozen_stages", [])),
    )
