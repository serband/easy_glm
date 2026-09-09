"""Immutable fitted-model permutation importance, independent of table edits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FORMAT = 1
REPEATS = 5
SEED = 42


def is_importance(request: dict) -> bool:
    """The named action and a read-only bridge for already-running prototypes."""
    return request.get("action") == "importance" or (
        request.get("action") == "coefficients"
        and request.get("options", {}).get("view") == "importance"
    )


def cache_path(source: Path) -> Path:
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
        "seed": SEED,
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()
    return source / "importance-cache" / f"{digest}.json"


def read_packet(source: Path) -> dict | None:
    try:
        packet = json.loads(cache_path(source).read_text())
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
    project: Any, run: Any, frame: Any, source: Path | None = None
) -> dict:
    """Compute from original fit and full training data; cache writing is optional."""
    from easy_glm.desktop.fit_worker import json_safe, write_json
    from easy_glm.workflow.diagnostics import permutation_importance, unit_values
    from easy_glm.workflow.prep import train_holdout

    train, _ = train_holdout(frame, project.data.split)
    protected = tuple(
        name
        for name, role in project.data.roles.items()
        if role
        in ("target", "weight", "exposure", "offset", "current_premium", "id", "split")
    ) + (project.data.split.column,)
    rows = permutation_importance(
        run.fit, train, repeats=REPEATS, seed=SEED, protected_columns=protected
    ).to_dicts()
    baseline = rows[0]["baseline_deviance"] if rows else None
    if baseline is None:
        y, w = unit_values(train, run.fit)
        baseline = float(
            run.fit.model.family_instance.deviance(
                y,
                run.fit.predict(train),
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
            "repeats": REPEATS,
            "seed": SEED,
            "training_rows": train.height,
            "baseline_deviance": baseline,
            "rows": rows,
        }
    )
    if source is not None:
        try:
            path = cache_path(source)
            path.parent.mkdir(exist_ok=True)
            write_json(path, packet)
        except OSError:
            pass  # A read-only artifact must not suppress a completed diagnostic.
    return packet
