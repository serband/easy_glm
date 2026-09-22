"""Revision-bound, cancellable subprocess jobs for GLM feature selection."""

from __future__ import annotations

import secrets
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from easy_glm.app import _launcher_env
from easy_glm.desktop.feature_selection_ipc import read_message, write_request
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
    importance_sample_pct: float = Field(default=30.0, gt=0, le=100)
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


class FeatureSelectionJobs:
    """One process at a time, exchanging data and bounded messages through pipes."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.tasks: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.generation: Generation | None = None
        self.active: dict[str, Any] | None = None
        self.closed = False
        # The server installs this. It must run after releasing ``lock``
        # because server requests take their project lock first.
        self.on_complete: Callable[[dict[str, Any], dict[str, Any]], None] | None = None

    def invalidate(self, generation: Generation) -> None:
        with self.lock:
            self.generation = generation
            for task in self.tasks.values():
                if task["generation"] != generation and task["status"] != "stale":
                    self._stop(
                        task, "stale", "Settings changed. Run feature selection again."
                    )

    def _stop(self, task: dict[str, Any], status: str, message: str) -> None:
        if task["status"] in ("queued", "running", "completing"):
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
            # Selection needs only the data/design draft. Keeping previous
            # models or audit metadata here would recursively carry an older
            # selection report into every new worker and recipe.
            screening_project = deepcopy(project)
            screening_project.models = {}
            screening_project.champion = None
            screening_project.exploration = {}
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
                # Private immutable source for the exported recipe. It is not
                # returned in API packets.
                "recipe_project": screening_project.to_dict(),
                "recipe_options": deepcopy(options),
            }
            thread = threading.Thread(
                target=self._run,
                args=(task, screening_project, raw.clone(), dict(options)),
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
        process = None
        transfers: list[threading.Thread] = []
        errors: list[Exception] = []
        transfer_failed = threading.Event()
        result: dict[str, Any] | None = None
        try:
            completing = False
            completed_result: dict[str, Any] | None = None
            with self.lock:
                if task["cancel"]:
                    return
                process = subprocess.Popen(
                    [sys.executable, "-m", "easy_glm.desktop.feature_selection_worker"],
                    env={
                        **_launcher_env(),
                        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "POLARS_MAX_THREADS": "2",
                    },
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                task.update(
                    process=process, status="running", message="Worker starting…"
                )
            assert process.stdin is not None and process.stdout is not None

            def send_input() -> None:
                try:
                    with process.stdin:
                        write_request(process.stdin, project, raw, options)
                except Exception as exc:
                    errors.append(exc)
                    transfer_failed.set()

            def receive_output() -> None:
                nonlocal result
                try:
                    with process.stdout:
                        while (packet := read_message(process.stdout)) is not None:
                            value = packet.get("value")
                            if not isinstance(value, dict) or result is not None:
                                raise ValueError("Invalid feature selection response.")
                            if packet.get("type") == "result":
                                result = value
                            elif packet.get("type") == "progress":
                                with self.lock:
                                    if (
                                        not task["cancel"]
                                        and task["status"] == "running"
                                    ):
                                        task["progress"] = value
                                        task["message"] = str(
                                            value.get("message", task["message"])
                                        )
                            else:
                                raise ValueError("Unknown feature selection response.")
                except Exception as exc:
                    errors.append(exc)
                    transfer_failed.set()

            # Neither pipe may block cancellation, including while a large
            # input is being sent. Drain output concurrently to avoid deadlocks.
            for target in (send_input, receive_output):
                thread = threading.Thread(
                    target=target,
                    daemon=True,
                    name=f"easyglm-selection-{target.__name__}",
                )
                thread.start()
                transfers.append(thread)
            while process.poll() is None:
                with self.lock:
                    if task["cancel"]:
                        if (
                            time.monotonic()
                            - task.setdefault("terminate_started", time.monotonic())
                            > 1
                        ):
                            process.kill()
                        else:
                            process.terminate()
                    elif transfer_failed.is_set():
                        raise errors[0]
                time.sleep(0.1)
            process.wait()
            for thread in transfers:
                thread.join()
            with self.lock:
                if (
                    task["cancel"]
                    or task["generation"] != self.generation
                    or self.closed
                ):
                    return
                if result is not None and "error" in result:
                    task.update(status="failed", message=str(result["error"])[:1000])
                elif process.returncode or result is None:
                    task.update(
                        status="failed",
                        message="Feature selection worker stopped unexpectedly.",
                    )
                elif errors:
                    raise errors[0]
                else:
                    completed_result = with_raw_names(result, project, raw.columns)
                    task.update(
                        # Do not expose completion before the server has made
                        # the recipe durable. Otherwise an Apply could advance
                        # the generation between a complete packet and publish.
                        status="completing",
                        message="Saving feature-selection recipe…",
                        progress={
                            "phase": "complete",
                            "completed": result.get("candidate_count", 0),
                            "total": result.get("candidate_count", 0),
                            "message": "Feature selection complete.",
                            "current_variable": None,
                        },
                    )
                    completing = True
            # Do not call the server while holding the jobs lock: its callback
            # verifies the task while holding the project lock first.
            if completing and self.on_complete is not None:
                assert completed_result is not None
                self.on_complete(task, completed_result)
            elif completing:
                with self.lock:
                    if task["status"] == "completing":
                        task.update(
                            status="complete",
                            message="Feature selection complete.",
                            result=completed_result,
                        )
        except (
            Exception
        ) as exc:  # Process boundary: return a message, never a traceback.
            with self.lock:
                if not task["cancel"] and task["status"] in (
                    "queued",
                    "running",
                    "completing",
                ):
                    task.update(
                        status="failed", message=f"Feature selection failed: {exc}"
                    )
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
                for thread in transfers:
                    thread.join()
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
            with self.lock:
                task["elapsed"] = time.monotonic() - task["started"]
                task["process"] = None

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
        status = "running" if task["status"] == "completing" else task["status"]
        return deepcopy(
            {
                "id": task["id"],
                "status": status,
                "session_id": session,
                "project_id": project,
                "revision": revision,
                "fingerprint": task["fingerprint"],
                "message": task["message"],
                "progress": task["progress"],
                "elapsed": task.get("elapsed", time.monotonic() - task["started"]),
                **({"result": task["result"]} if status == "complete" else {}),
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
            if task["status"] in ("queued", "running", "completing"):
                self._stop(task, "cancelled", "Feature selection cancelled.")

    def close(self) -> None:
        with self.lock:
            self.closed = True
            tasks = list(self.tasks.values())
            for task in tasks:
                if task["status"] in ("queued", "running", "completing"):
                    self._stop(task, "cancelled", "Session closed.")
        for task in tasks:
            task["thread"].join(timeout=5)
