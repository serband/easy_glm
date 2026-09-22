"""Immutable fitted-model permutation importance, independent of table edits."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

FORMAT = 3
REPEATS = 5
SEED = 42


def is_importance(request: dict) -> bool:
    """The named action and a read-only bridge for already-running prototypes."""
    return request.get("action") == "importance" or (
        request.get("action") == "coefficients"
        and request.get("options", {}).get("view") == "importance"
    )


def options_from_request(request: dict[str, Any]) -> dict[str, Any]:
    """Validate the only settings that change an importance cache identity."""
    options = request.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError("Importance options must be an object.")
    view = options.get("view")
    if view is not None and view != "importance":
        raise ValueError("Unsupported coefficient view.")
    percentage = options.get("importance_sample_pct", 30.0)
    if (
        isinstance(percentage, bool)
        or not isinstance(percentage, (int, float))
        or not math.isfinite(float(percentage))
        or not 0 < float(percentage) <= 100
    ):
        raise ValueError("importance_sample_pct must be a finite number in (0, 100].")
    seed = options.get("seed", SEED)
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("seed must be an integer in [0, 2**32 - 1].")
    return {"importance_sample_pct": float(percentage), "seed": seed}


def cache_path(
    source: Path, *, importance_sample_pct: float = 30.0, seed: int = SEED
) -> Path:
    files = {}
    for filename in ("fit.pkl", "raw.parquet", "project.json"):
        path = source / filename
        if path.exists():
            stat = path.stat()
            files[filename] = [stat.st_size, stat.st_mtime_ns]
    basis = {
        "format": FORMAT,
        "fit_id": source.name,
        "files": files,
        "repeats": REPEATS,
        "seed": seed,
        "importance_sample_pct": importance_sample_pct,
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()
    return source / "importance-cache" / f"{digest}.json"


def read_packet(
    source: Path, *, importance_sample_pct: float = 30.0, seed: int = SEED
) -> dict | None:
    try:
        packet = json.loads(
            cache_path(
                source, importance_sample_pct=importance_sample_pct, seed=seed
            ).read_text()
        )
        if (
            packet["format"] == FORMAT
            and packet["fit_id"] == source.name
            and isinstance(packet["rows"], list)
        ):
            return packet
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def build_packet(
    project: Any,
    run: Any,
    frame: Any,
    source: Path | None = None,
    *,
    importance_sample_pct: float = 30.0,
    seed: int = SEED,
) -> dict:
    """Score the original fit on a declared training sample and optionally cache it."""
    from easy_glm.desktop.fit_worker import json_safe, write_json
    from easy_glm.workflow.diagnostics import (
        permutation_importance_with_metadata,
        unit_values,
    )
    from easy_glm.workflow.importance_sampling import importance_sample_indices
    from easy_glm.workflow.prep import train_holdout
    from easy_glm.workflow.run import rate_model_for

    train, _ = train_holdout(frame, project.data.split)
    protected = tuple(
        name
        for name, role in project.data.roles.items()
        if role
        in (
            "target",
            "weight",
            "exposure",
            "offset",
            "current_premium",
            "id",
            "split",
            "time",
        )
    ) + (project.data.split.column,)
    pair_parents = tuple(
        dict.fromkeys(
            parent
            for stage in getattr(run.config, "pair_stages", [])
            for parent in (stage.a, stage.b)
        )
    )
    frozen = rate_model_for(project, run, [], base_rate_override=None)

    def frozen_predict(frame: Any) -> Any:
        return frozen.predict(frame, exposure_col=None)

    importance, sample_metadata = permutation_importance_with_metadata(
        run.fit,
        train,
        repeats=REPEATS,
        seed=seed,
        protected_columns=protected,
        scorer=frozen_predict,
        additional_variables=pair_parents,
        importance_sample_pct=importance_sample_pct,
    )
    rows = importance.to_dicts()
    baseline = rows[0]["baseline_deviance"] if rows else None
    if baseline is None:
        full_y, full_w = unit_values(train, run.fit)
        indices, _ = importance_sample_indices(
            train,
            importance_sample_pct=importance_sample_pct,
            seed=seed,
            outcome=full_y,
            weights=full_w,
            family=run.fit.family,
        )
        sampled = train[indices]
        y, w = unit_values(sampled, run.fit)
        baseline = float(
            run.fit.model.family_instance.deviance(
                y,
                frozen_predict(sampled),
                sample_weight=w if run.fit.weight_col else None,
            )
        ) / float(w.sum())
    packet = json_safe(
        {
            "format": FORMAT,
            "fit_id": source.name if source is not None else None,
            "metric": "Mean deviance increase",
            "subset": "train",
            "basis": "original",
            "scoring_basis": "complete frozen rate tables",
            "repeats": REPEATS,
            "seed": seed,
            "training_rows": train.height,
            **sample_metadata,
            "baseline_deviance": baseline,
            "rows": rows,
        }
    )
    if source is not None:
        try:
            path = cache_path(
                source, importance_sample_pct=importance_sample_pct, seed=seed
            )
            path.parent.mkdir(exist_ok=True)
            write_json(path, packet)
        except OSError:
            pass  # A read-only artifact must not suppress a completed diagnostic.
    return packet
