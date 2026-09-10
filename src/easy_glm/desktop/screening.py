"""Cancellable, read-only screening of an immutable Variables draft."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from easy_glm.desktop.fit_worker import json_safe
from easy_glm.desktop.modeling import Revision
from easy_glm.workflow.project import Project

Generation = tuple[str, str, int]


class ScreeningOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sample_rows: int = Field(default=10_000, ge=500, le=20_000, strict=True)
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    missing_threshold: float = Field(default=0.7, ge=0, le=1)
    correlation_threshold: float = Field(default=0.95, ge=0, le=1)
    leakage_threshold: float = Field(default=0.9, ge=0, le=1)
    divide_target_by_weight: bool = True


class ScreeningRequest(Revision):
    setup: dict[str, Any]
    options: ScreeningOptions = Field(default_factory=ScreeningOptions)


def draft_fingerprint(setup: dict[str, Any], options: dict[str, Any]) -> str:
    """Bind a scan to every draft setting and numerical option."""
    return hashlib.sha256(
        json.dumps(
            {"setup": setup, "options": options}, sort_keys=True, allow_nan=False
        ).encode()
    ).hexdigest()


def with_raw_names(
    result: dict[str, Any], project: Project, raw_columns: list[str]
) -> dict[str, Any]:
    """Keep display names, adding stable raw IDs for editing the Variables grid."""
    result = deepcopy(result)
    sources = {project.data.renames.get(name, name): name for name in raw_columns}
    for group in ("leakage", "missing", "unsupported", "columns"):
        for row in result.get(group, []):
            row["raw_name"] = sources.get(row.get("variable", row.get("name")))
    for row in result.get("correlated", []):
        row["first_raw"] = sources.get(row["first"])
        row["second_raw"] = sources.get(row["second"])
    result["target_raw_name"] = sources.get(result.get("target"))
    result["weight_raw_name"] = sources.get(result.get("weight"))
    return json_safe(result)


class ScreeningJobs:
    """One cooperative worker, with at most two completed result packets."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.tasks: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.generation: Generation | None = None
        self.closed = False

    def invalidate(self, generation: Generation) -> None:
        with self.lock:
            self.generation = generation
            for task in self.tasks.values():
                if task["generation"] != generation:
                    task["cancelled"].set()
                    task.update(
                        status="stale", message="Settings changed. Run the scan again."
                    )
                    task.pop("result", None)

    def start(
        self,
        project: Project,
        raw: pl.DataFrame,
        generation: Generation,
        setup: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        fingerprint = draft_fingerprint(setup, options)
        with self.lock:
            self.invalidate(generation)
            if self.closed:
                raise ValueError("The screening session has closed.")
            for key, task in self.tasks.items():
                if (
                    task["generation"] == generation
                    and task["fingerprint"] == fingerprint
                    and task["status"] in ("queued", "running", "complete")
                ):
                    self.tasks.move_to_end(key)
                    return self._packet(task)
            if any(task["thread"].is_alive() for task in self.tasks.values()):
                raise ValueError(
                    "A scan is still running. Cancel it or wait for it to finish."
                )
            key = secrets.token_hex(12)
            task = {
                "id": key,
                "generation": generation,
                "fingerprint": fingerprint,
                "status": "queued",
                "message": "Preparing training data…",
                "progress": {"phase": "preparing", "completed": 0, "total": 0},
                "started": time.monotonic(),
                "cancelled": threading.Event(),
            }
            thread = threading.Thread(
                target=self._run,
                args=(task, project, raw, options),
                daemon=True,
                name="easyglm-screening",
            )
            task["thread"] = thread
            self.tasks[key] = task
            self._prune()
            thread.start()
            return self._packet(task)

    def _run(
        self,
        task: dict[str, Any],
        project: Project,
        raw: pl.DataFrame,
        options: dict[str, Any],
    ) -> None:
        def cancelled() -> bool:
            return task["cancelled"].is_set()

        def progress(value: dict[str, Any]) -> None:
            with self.lock:
                if not cancelled():
                    task["progress"] = json_safe(dict(value))
                    task["message"] = str(value.get("message", task["message"]))

        try:
            from easy_glm.workflow.screening import screen_variables

            with self.lock:
                if cancelled():
                    return
                task["status"] = "running"
            result = screen_variables(
                project, raw, **options, progress=progress, cancelled=cancelled
            )
            result = with_raw_names(result, project, raw.columns)
            with self.lock:
                if (
                    not cancelled()
                    and task["generation"] == self.generation
                    and not self.closed
                ):
                    task.update(
                        status="complete", message="Scan complete.", result=result
                    )
        except Exception as exc:  # Thread boundary: a message, never a traceback.
            with self.lock:
                if not cancelled():
                    task.update(status="failed", message=str(exc))
        finally:
            with self.lock:
                task["elapsed"] = time.monotonic() - task["started"]
                self._prune()

    def _prune(self) -> None:
        completed = [
            key for key, task in self.tasks.items() if task["status"] == "complete"
        ]
        for key in completed[:-2]:
            self.tasks.pop(key)
        while len(self.tasks) > 8:
            removable = next(
                (
                    key
                    for key, task in self.tasks.items()
                    if not task["thread"].is_alive()
                ),
                None,
            )
            if removable is None:
                break
            self.tasks.pop(removable)

    def _packet(self, task: dict[str, Any]) -> dict[str, Any]:
        session, project, revision = task["generation"]
        return deepcopy(
            {
                "id": task["id"],
                "status": task["status"],
                "session_id": session,
                "project_id": project,
                "revision": revision,
                "fingerprint": task["fingerprint"],
                "message": task["message"],
                "progress": task["progress"],
                "elapsed": task.get("elapsed", time.monotonic() - task["started"]),
                **({"result": task["result"]} if task["status"] == "complete" else {}),
            }
        )

    def status(self, key: str, generation: Generation) -> dict[str, Any]:
        with self.lock:
            self.invalidate(generation)
            if key not in self.tasks:
                raise KeyError("Scan expired. Run it again.")
            return self._packet(self.tasks[key])

    def cancel(self, key: str) -> None:
        with self.lock:
            if key not in self.tasks:
                raise KeyError("Scan expired. Run it again.")
            task = self.tasks[key]
            if task["status"] in ("queued", "running"):
                task["cancelled"].set()
                task.update(status="cancelled", message="Scan cancelled.")

    def close(self) -> None:
        with self.lock:
            self.closed = True
            tasks = list(self.tasks.values())
            for task in tasks:
                task["cancelled"].set()
                if task["status"] in ("queued", "running"):
                    task.update(status="cancelled", message="Session closed.")
        for task in tasks:
            task["thread"].join(timeout=2)
