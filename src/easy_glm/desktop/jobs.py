"""One cancellable worker process at a time; immutable inputs and stale results."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import polars as pl

from easy_glm.app import _launcher_env
from easy_glm.workflow.project import Project


def model_key(project: Project, name: str) -> str:
    """Fingerprint the settings affecting this result, not unrelated models."""
    spec = project.to_dict()
    data = dict(spec["data"])
    data.pop("sample_rows", None)
    data.pop("sample_seed", None)
    return hashlib.sha256(
        json.dumps(
            {"data": data, "design": spec["design"], "model": spec["models"].get(name)},
            sort_keys=True,
        ).encode()
    ).hexdigest()


class FitJobs:
    """Own isolated temporary inputs/results; never read or write workbench caches."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.active: dict[str, Any] | None = None
        self.folder = tempfile.TemporaryDirectory(prefix="easyglm_fits_")
        self.thread: threading.Thread | None = None
        self.on_complete: Callable[[dict[str, Any], dict[str, Any]], None] | None = None

    def restore(self, project: Project, raw: pl.DataFrame, folder: Path) -> None:
        """Restore application-owned upgrade artifacts without fitting any model.

        This is a local launcher operation, never an HTTP-provided path.
        """
        import pickle
        import shutil

        from easy_glm.desktop.fit_worker import result_for, write_json
        from easy_glm.workflow.prep import prepare
        from easy_glm.workflow.run import rebuild_rate_model

        records = json.loads((folder / "jobs.json").read_text())
        prepared = None
        for name, record in records.items():
            if name not in project.models:
                continue
            saved = {
                k: record[k]
                for k in ("id", "name", "status", "message", "elapsed", "key")
            }
            if record.get("applicable"):
                if record["key"] != model_key(project, name):
                    raise ValueError(
                        "Restored fit does not match applied settings: " + name
                    )
                source = folder / "fits" / record["id"]
                if not raw.equals(pl.read_parquet(source / "raw.parquet")):
                    raise ValueError(
                        "Restored fit data differs from the loaded source."
                    )
                with (source / "fit.pkl").open("rb") as handle:
                    run = pickle.load(
                        handle
                    )  # only private artifacts created by this application
                if run.name != name:
                    raise ValueError("Restored fit identity does not match.")
                if prepared is None:
                    prepared = prepare(project, raw)
                rebuild_rate_model(project, run, prepared)
                warnings = json.loads((source / "result.json").read_text()).get(
                    "warnings", []
                )
                result = result_for(project, prepared, run, warnings)
                destination = Path(self.folder.name) / record["id"]
                shutil.copytree(source, destination)
                write_json(destination / "result.json", result)
                saved["result"] = result
            else:
                saved["status"] = "stale"
            self.jobs[name] = saved

    def status(self, project: Project) -> dict[str, Any]:
        with self.lock:
            return {
                name: {
                    k: v
                    for k, v in job.items()
                    if k in ("id", "name", "status", "message", "elapsed", "key")
                }
                | {
                    "applicable": job["key"] == model_key(project, name)
                    and job["status"] == "complete"
                }
                for name, job in self.jobs.items()
            }

    def start(self, project: Project, raw: pl.DataFrame, name: str) -> dict[str, Any]:
        with self.lock:
            if self.active and self.active["status"] in ("queued", "running"):
                raise ValueError(
                    "A fit is already running. Wait for it or cancel it first."
                )
            job = {
                "id": str(time.time_ns()),
                "name": name,
                "key": model_key(project, name),
                "status": "queued",
                "message": "Preparing an isolated worker…",
                "elapsed": 0.0,
                "cancel": False,
                "process": None,
            }
            self.jobs[name] = self.active = job
            self.thread = threading.Thread(
                target=self._run, args=(job, project, raw), daemon=True
            )
            self.thread.start()
            return {k: job[k] for k in ("id", "name", "status", "message")}

    def _run(self, job: dict[str, Any], project: Project, raw: pl.DataFrame) -> None:
        started = time.monotonic()
        folder = Path(self.folder.name) / job["id"]
        folder.mkdir()
        try:
            # Keep the supplied applied state for private recovery. An explicit
            # new fit starts from its coefficients, never inherited table edits.
            project.to_json(folder / "project-before-refit.json")
            clean_project = deepcopy(project)
            clean_project.models[job["name"]].adjustments = []
            clean_project.models[job["name"]].base_rate_override = None
            clean_project.to_json(folder / "project.json")
            raw.write_parquet(folder / "raw.parquet")
            with self.lock:
                if job["cancel"]:
                    return
                with (folder / "worker.log").open("w") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "easy_glm.desktop.fit_worker",
                            str(folder),
                            job["name"],
                            str(
                                Path(self.folder.name)
                                / "main-effects"
                                / (
                                    hashlib.sha256(job["name"].encode()).hexdigest()
                                    + ".pkl"
                                )
                            ),
                        ],
                        # Use this server's package, not another editable install.
                        # The path is derived from code; never inherit kernel paths.
                        env={
                            **_launcher_env(),
                            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                        },
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                job.update(
                    process=process, status="running", message="Worker starting…"
                )
            while process.poll() is None:
                with self.lock:
                    job["elapsed"] = time.monotonic() - started
                    if job["cancel"]:
                        process.terminate()
                    try:
                        job["message"] = json.loads(
                            (folder / "progress.json").read_text()
                        )["message"]
                    except (OSError, json.JSONDecodeError):
                        pass
                time.sleep(0.1)
            with self.lock:
                job["elapsed"] = time.monotonic() - started
                if job["cancel"]:
                    return
                result = json.loads((folder / "result.json").read_text())
                if "error" in result:
                    job.update(status="failed", message=result["error"])
                    return
            # The server takes its project lock before the jobs lock. Do not
            # call it while holding this lock: requests acquire them in that order.
            if self.on_complete is not None:
                self.on_complete(job, result)
            else:
                with self.lock:
                    if not job["cancel"]:
                        job.update(
                            status="complete",
                            message="Diagnostics and rate tables are ready.",
                            result=result,
                        )
        except (
            Exception
        ) as exc:  # worker/process boundary: return a UI error, never a traceback
            with self.lock:
                if not job["cancel"]:
                    job.update(status="failed", message=f"Fit could not finish: {exc}")
        finally:
            with self.lock:
                job["process"] = None

    def cancel(self, name: str, message: str = "Fit cancelled") -> None:
        with self.lock:
            job = self.jobs.get(name)
            if job and job["status"] in ("queued", "running"):
                job.update(cancel=True, status="cancelled", message=message)
                if job["process"] is not None:
                    job["process"].terminate()

    def invalidate(self, project: Project) -> None:
        with self.lock:
            for name, job in self.jobs.items():
                if job["key"] != model_key(project, name):
                    self.cancel(name, "Inputs changed; the old fit was cancelled.")
                    job.update(
                        status="stale",
                        message="Settings changed. Fit this model again.",
                    )
                    job.pop("result", None)

    def result(self, project: Project, name: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(name)
            if (
                not job
                or job["status"] != "complete"
                or job["key"] != model_key(project, name)
            ):
                raise ValueError(
                    "No completed fit matches the applied settings. Fit this model first."
                )
            return job["result"]

    def artifact(self, project: Project, name: str) -> Path:
        with self.lock:
            self.result(project, name)
            return Path(self.folder.name) / self.jobs[name]["id"]

    def edited(self, project: Project, name: str, result: dict[str, Any]) -> None:
        with self.lock:
            self.jobs[name].update(key=model_key(project, name), result=result)

    def close(self) -> None:
        with self.lock:
            for name in self.jobs:
                self.cancel(name)
        if self.thread:
            self.thread.join(timeout=5)
        self.folder.cleanup()
