"""Revision-bound, cancellable subprocess jobs for GLM feature selection."""

from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from easy_glm.app import _launcher_env
from easy_glm.desktop.modeling import Revision
from easy_glm.desktop.screening import draft_fingerprint
from easy_glm.workflow.project import Project

Generation = tuple[str, str, int]


class FeatureSelectionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    family: Literal["poisson", "gamma", "gaussian", "binomial", "tweedie"] = "poisson"
    link: Literal["log", "identity", "logit"] | None = None
    tweedie_power: float = Field(default=1.5, gt=1, lt=2)
    divide_target_by_weight: bool = False
    l1_ratio: float = Field(default=1.0, gt=0, le=1)
    n_alphas: int = Field(default=20, ge=2, le=100)
    repeats: int = Field(default=5, ge=1, le=20)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    include_unassigned: bool = True

    @model_validator(mode="after")
    def compatible_link(self) -> FeatureSelectionOptions:
        allowed = {
            "poisson": {"log", "identity"},
            "gamma": {"log", "identity"},
            "gaussian": {"identity", "log"},
            "binomial": {"logit"},
            "tweedie": {"log", "identity"},
        }
        if self.link is not None and self.link not in allowed[self.family]:
            raise ValueError(f"{self.link} link is not compatible with {self.family}.")
        if self.family != "tweedie" and self.tweedie_power != 1.5:
            raise ValueError("Tweedie power applies only to the Tweedie family.")
        return self


class FeatureSelectionRequest(Revision):
    setup: dict[str, Any]
    options: FeatureSelectionOptions = Field(default_factory=FeatureSelectionOptions)


def with_raw_names(
    result: dict[str, Any], project: Project, columns: list[str]
) -> dict[str, Any]:
    """Add stable raw column IDs for the existing Variables draft editor."""
    result = deepcopy(result)
    sources = {project.data.renames.get(name, name): name for name in columns}
    for row in result.get("rows", []):
        row["raw_name"] = sources.get(row.get("variable"))
    for key in ("target", "weight", "offset"):
        result[key + "_raw_name"] = sources.get(result.get(key))
    return result


def _read_json(path: Path, limit: int) -> Any:
    if path.stat().st_size > limit:
        raise ValueError("Feature selection output exceeded its size limit.")
    return json.loads(path.read_text(encoding="utf-8"))


class FeatureSelectionJobs:
    """One process at a time, with bounded packets and promptly removed inputs."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.tasks: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.generation: Generation | None = None
        self.active: dict[str, Any] | None = None
        self.closed = False
        self.folder = tempfile.TemporaryDirectory(prefix="easyglm_selection_")

    def invalidate(self, generation: Generation) -> None:
        with self.lock:
            self.generation = generation
            for task in self.tasks.values():
                if task["generation"] != generation and task["status"] != "stale":
                    self._stop(
                        task, "stale", "Settings changed. Run feature selection again."
                    )

    def _stop(self, task: dict[str, Any], status: str, message: str) -> None:
        if task["status"] in ("queued", "running"):
            task["cancel"] = True
            process = task.get("process")
            if process is not None and process.poll() is None:
                task.setdefault("terminate_started", time.monotonic())
                process.terminate()
        task.update(status=status, message=message)
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
                raise ValueError("The feature selection session has closed.")
            for key, task in self.tasks.items():
                if (
                    task["generation"] == generation
                    and task["fingerprint"] == fingerprint
                    and task["status"] in ("queued", "running", "complete")
                ):
                    self.tasks.move_to_end(key)
                    return self._packet(task)
            if self.active is not None and self.active["thread"].is_alive():
                raise ValueError(
                    "Feature selection is still finishing. Wait for it before starting again."
                )
            key = secrets.token_hex(12)
            task: dict[str, Any] = {
                "id": key,
                "generation": generation,
                "fingerprint": fingerprint,
                "status": "queued",
                "message": "Preparing training data…",
                "progress": {"phase": "preparing", "completed": 0, "total": 0},
                "started": time.monotonic(),
                "cancel": False,
                "process": None,
            }
            thread = threading.Thread(
                target=self._run,
                args=(task, deepcopy(project), raw.clone(), dict(options)),
                daemon=True,
                name="easyglm-feature-selection",
            )
            task["thread"] = thread
            self.tasks[key] = self.active = task
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
        folder = Path(self.folder.name) / task["id"]
        try:
            folder.mkdir()
            project.to_json(folder / "project.json")
            raw.write_parquet(folder / "raw.parquet")
            (folder / "options.json").write_text(json.dumps(options), encoding="utf-8")
            with self.lock:
                if task["cancel"]:
                    return
                with (folder / "worker.log").open("w") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "easy_glm.desktop.feature_selection_worker",
                            str(folder),
                        ],
                        env={
                            **_launcher_env(),
                            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                            "POLARS_MAX_THREADS": "2",
                        },
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                task.update(
                    process=process, status="running", message="Worker starting…"
                )
            while process.poll() is None:
                with self.lock:
                    if task["cancel"]:
                        if process.poll() is None:
                            if (
                                time.monotonic()
                                - task.setdefault("terminate_started", time.monotonic())
                                > 1
                            ):
                                process.kill()
                            else:
                                process.terminate()
                    else:
                        try:
                            progress = _read_json(folder / "progress.json", 16_384)
                            if isinstance(progress, dict):
                                task["progress"] = progress
                                task["message"] = str(
                                    progress.get("message", task["message"])
                                )
                        except (OSError, ValueError, json.JSONDecodeError):
                            pass
                time.sleep(0.1)
            with self.lock:
                if (
                    task["cancel"]
                    or task["generation"] != self.generation
                    or self.closed
                ):
                    return
                result = _read_json(folder / "result.json", 8_000_000)
                if "error" in result:
                    task.update(status="failed", message=str(result["error"])[:1000])
                elif process.returncode:
                    task.update(
                        status="failed",
                        message="Feature selection worker stopped unexpectedly.",
                    )
                else:
                    task.update(
                        status="complete",
                        message="Feature selection complete.",
                        result=with_raw_names(result, project, raw.columns),
                        progress={
                            "phase": "complete",
                            "completed": result.get("candidate_count", 0),
                            "total": result.get("candidate_count", 0),
                            "message": "Feature selection complete.",
                            "current_variable": None,
                        },
                    )
        except (
            Exception
        ) as exc:  # Process boundary: return a message, never a traceback.
            with self.lock:
                if not task["cancel"] and task["status"] in ("queued", "running"):
                    task.update(
                        status="failed", message=f"Feature selection failed: {exc}"
                    )
        finally:
            with self.lock:
                task["elapsed"] = time.monotonic() - task["started"]
                task["process"] = None
            shutil.rmtree(folder, ignore_errors=True)

    def _prune(self) -> None:
        complete = [
            key for key, task in self.tasks.items() if task["status"] == "complete"
        ]
        for key in complete[:-2]:
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
                raise KeyError("Feature selection expired. Run it again.")
            return self._packet(self.tasks[key])

    def cancel(self, key: str) -> None:
        with self.lock:
            if key not in self.tasks:
                raise KeyError("Feature selection expired. Run it again.")
            task = self.tasks[key]
            if task["status"] in ("queued", "running"):
                self._stop(task, "cancelled", "Feature selection cancelled.")

    def close(self) -> None:
        with self.lock:
            self.closed = True
            tasks = list(self.tasks.values())
            for task in tasks:
                if task["status"] in ("queued", "running"):
                    self._stop(task, "cancelled", "Session closed.")
        for task in tasks:
            task["thread"].join(timeout=5)
        # Never remove a worker's input directory while it may still write to it.
        if all(not task["thread"].is_alive() for task in tasks):
            self.folder.cleanup()
