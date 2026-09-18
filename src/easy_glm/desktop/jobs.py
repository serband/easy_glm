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
from dataclasses import asdict
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


def _main_key(project: Project, name: str) -> str:
    """Fingerprint the main fit without post-fit edits or pair configuration."""
    spec = project.to_dict()
    data = dict(spec["data"])
    data.pop("sample_rows", None)
    data.pop("sample_seed", None)
    model = dict(spec["models"][name])
    for field in (
        "pair_stages",
        "pair_method",
        "adjustments",
        "base_rate_override",
        "snapshots",
        "notes",
    ):
        model.pop(field, None)
    design = _design_for(project, model["predictors"])
    return hashlib.sha256(
        json.dumps(
            {"data": data, "design": design, "model": model}, sort_keys=True
        ).encode()
    ).hexdigest()


def _design_for(
    project: Project, parents: list[str] | tuple[str, str]
) -> dict[str, Any]:
    """Design defaults and only the variables used by one fit component."""
    design = project.to_dict()["design"]
    return {
        "defaults": design["defaults"],
        "variables": {
            name: design["variables"].get(name) for name in sorted(set(parents))
        },
    }


def _stage_design_changed(
    current: Project, previous: Project, current_stage: Any, old_stage: Any
) -> bool:
    return _design_for(current, (current_stage.a, current_stage.b)) != _design_for(
        previous, (old_stage.a, old_stage.b)
    )


def _stage_edits(project: Project, name: str, stage_id: str) -> list[dict[str, Any]]:
    return [
        item.__dict__
        for item in project.models[name].adjustments
        if item.stage_id == stage_id
    ]


def _completed_record(job: dict[str, Any]) -> dict[str, Any]:
    """Detach a published result from the mutable active-job lifecycle."""
    return {
        key: deepcopy(job[key])
        for key in (
            "id",
            "name",
            "key",
            "status",
            "message",
            "elapsed",
            "result",
            "project",
        )
        if key in job
    }


def _first_pair_refit(project: Project, previous: Project | None, name: str) -> int:
    """First stage needing a new fitted table; edits on that suffix are replaced."""
    stages = project.models[name].pair_stages
    if previous is None or name not in previous.models:
        return 0
    if _main_key(project, name) != _main_key(previous, name):
        return 0
    current = project.models[name]
    old = previous.models[name]
    if current.base_rate_override != old.base_rate_override or [
        a.__dict__ for a in current.adjustments if a.stage_id is None
    ] != [a.__dict__ for a in old.adjustments if a.stage_id is None]:
        return 0
    for index, stage in enumerate(stages):
        if index >= len(old.pair_stages) or stage != old.pair_stages[index]:
            return index
        if _stage_design_changed(project, previous, stage, old.pair_stages[index]):
            return index
        if _stage_edits(project, name, stage.stage_id) != _stage_edits(
            previous, name, stage.stage_id
        ):
            return index + 1
    return len(stages)


class FitJobs:
    """Own isolated temporary inputs/results; never read or write workbench caches."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.last_complete: dict[str, dict[str, Any]] = {}
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
                saved["project"] = (
                    Project.from_json(source / "project.json")
                    if (source / "project.json").exists()
                    else deepcopy(project)
                )
                self.last_complete[name] = _completed_record(saved)
                if getattr(run, "pair_stages", None):
                    pair_cache_path = (
                        Path(self.folder.name)
                        / "pair-prefix"
                        / (hashlib.sha256(name.encode()).hexdigest() + ".pkl")
                    )
                    pair_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with pair_cache_path.open("wb") as handle:
                        pickle.dump(
                            {
                                "format": 3,
                                "cache": {
                                    "full_prefix": {
                                        stage.prefix_fingerprint: deepcopy(stage)
                                        for stage in run.pair_stages
                                    }
                                },
                            },
                            handle,
                            protocol=5,
                        )
            else:
                saved["status"] = "stale"
            self.jobs[name] = saved

    def status(self, project: Project) -> dict[str, Any]:
        with self.lock:
            return {
                name: {
                    k: v
                    for k, v in job.items()
                    if k
                    in ("id", "name", "status", "message", "elapsed", "key", "progress")
                }
                | {
                    "applicable": job["key"] == model_key(project, name)
                    and job["status"] == "complete",
                }
                for name, job in self.jobs.items()
            }

    def stage_statuses(self, project: Project) -> dict[str, list[dict[str, Any]]]:
        """Describe the current ordered recipe against the last published fit."""
        with self.lock:
            result: dict[str, list[dict[str, Any]]] = {}
            for name, config in project.models.items():
                if not config.pair_stages:
                    continue
                job = self.jobs.get(name)
                complete = (
                    job
                    if job and job["status"] == "complete"
                    else self.last_complete.get(name)
                )
                fitted_project = complete.get("project") if complete else None
                fitted_result = complete.get("result", {}) if complete else {}
                fitted_stages = {
                    item["stage_id"]: item
                    for item in fitted_result.get("pair_stages", [])
                }
                old_stages = (
                    fitted_project.models[name].pair_stages
                    if fitted_project is not None and name in fitted_project.models
                    else []
                )
                main_current = fitted_project is not None and _main_key(
                    project, name
                ) == _main_key(fitted_project, name)
                fitting = job is not None and job["status"] in ("queued", "running")
                failed = job is not None and job["status"] == "failed"
                stage_number = (job or {}).get("progress", {}).get("stage_number", 1)
                main_status = (
                    "up_to_date"
                    if main_current
                    else "needs_refitting" if complete else "not_fitted"
                )
                if fitting and stage_number == 1 and not main_current:
                    main_status = "fitting"
                if failed and stage_number == 1 and not main_current:
                    main_status = "failed"
                cards: list[dict[str, Any]] = [
                    {
                        "stage_id": "main",
                        "stage_number": 1,
                        "status": main_status,
                        "baseline": [],
                        "dimensions": None,
                        "support": None,
                        "chosen_candidate": None,
                        "message": (
                            (job or {}).get("message", "")
                            if main_status in ("fitting", "failed")
                            else ""
                        ),
                    }
                ]
                prefix_changed = not main_current
                if fitted_project is not None and main_current:
                    old_config = fitted_project.models[name]
                    if config.base_rate_override != old_config.base_rate_override or [
                        a.__dict__ for a in config.adjustments if a.stage_id is None
                    ] != [
                        a.__dict__ for a in old_config.adjustments if a.stage_id is None
                    ]:
                        prefix_changed = True
                for index, stage in enumerate(config.pair_stages):
                    saved = fitted_stages.get(stage.stage_id, {})
                    same_position = (
                        index < len(old_stages) and stage == old_stages[index]
                    )
                    same_design = (
                        fitted_project is not None
                        and same_position
                        and not _stage_design_changed(
                            project, fitted_project, stage, old_stages[index]
                        )
                    )
                    current = bool(
                        complete
                        and main_current
                        and same_position
                        and same_design
                        and not prefix_changed
                        and saved
                    )
                    state = (
                        saved.get("status", "up_to_date")
                        if current
                        else ("needs_refitting" if complete else "not_fitted")
                    )
                    if fitting and index + 2 == stage_number and not current:
                        state = "fitting"
                    if failed and index + 2 == stage_number and not current:
                        state = "failed"
                    cards.append(
                        {
                            "stage_id": stage.stage_id,
                            "search": asdict(stage.search) if stage.search else None,
                            "stage_number": index + 2,
                            "status": state,
                            "baseline": ["main"]
                            + [s.stage_id for s in config.pair_stages[:index]],
                            "dimensions": saved.get("dimensions"),
                            "support": saved.get("support"),
                            "chosen_candidate": saved.get("chosen_candidate"),
                            "loss": (
                                {
                                    "prefix": saved.get("prefix_cv_loss"),
                                    "table": saved.get("table_cv_loss"),
                                    "teacher": saved.get("teacher_cv_loss"),
                                    "approximation": saved.get("approximation_loss"),
                                }
                                if saved
                                else None
                            ),
                            "reused": saved.get("reused", False),
                            "message": (
                                (job or {}).get("message", "")
                                if state in ("fitting", "failed")
                                else ""
                            ),
                        }
                    )
                    if not current:
                        prefix_changed = True
                    elif fitted_project is not None and _stage_edits(
                        project, name, stage.stage_id
                    ) != _stage_edits(fitted_project, name, stage.stage_id):
                        prefix_changed = True
                result[name] = cards
            return result

    def refit_previews(self, project: Project) -> dict[str, dict[str, Any]]:
        """Edits that an explicit Fit remaining stages will replace on success."""
        with self.lock:
            previews: dict[str, dict[str, Any]] = {}
            for name, cfg in project.models.items():
                if not cfg.pair_stages:
                    continue
                previous = self.last_complete.get(name, {}).get("project")
                first = _first_pair_refit(project, previous, name)
                ids = [stage.stage_id for stage in cfg.pair_stages[first:]]
                previews[name] = {
                    "first_refit_stage": (
                        first + 2 if first < len(cfg.pair_stages) else None
                    ),
                    "stage_ids": ids,
                    "cleared_adjustments": [
                        adjustment.__dict__
                        for adjustment in cfg.adjustments
                        if adjustment.stage_id in ids
                    ],
                }
            return previews

    def start(self, project: Project, raw: pl.DataFrame, name: str) -> dict[str, Any]:
        with self.lock:
            if self.active and self.active["status"] in ("queued", "running"):
                raise ValueError(
                    "A fit is already running. Wait for it or cancel it first."
                )
            previous = self.jobs.get(name)
            if previous and previous["status"] == "complete":
                self.last_complete[name] = _completed_record(previous)
            first_refit = (
                _first_pair_refit(
                    project,
                    self.last_complete.get(name, {}).get("project"),
                    name,
                )
                if project.models[name].pair_stages
                else 0
            )
            clear_stage_ids = [
                stage.stage_id
                for stage in project.models[name].pair_stages[first_refit:]
            ]
            job = {
                "id": str(time.time_ns()),
                "name": name,
                "key": model_key(project, name),
                "status": "queued",
                "message": "Preparing an isolated worker…",
                "elapsed": 0.0,
                "cancel": False,
                "process": None,
                "project": deepcopy(project),
                "previous": deepcopy(self.last_complete.get(name)),
                "clear_stage_ids": clear_stage_ids,
                "first_refit_stage": first_refit,
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
            # A staged suffix must consume the exact deployed upstream edits.
            # Legacy fits still start from fresh coefficients and clear edits.
            if not clean_project.models[job["name"]].pair_stages:
                clean_project.models[job["name"]].adjustments = []
                clean_project.models[job["name"]].base_rate_override = None
            else:
                clear = set(job["clear_stage_ids"])
                clean_project.models[job["name"]].adjustments = [
                    adjustment
                    for adjustment in clean_project.models[job["name"]].adjustments
                    if adjustment.stage_id not in clear
                ]
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
                            str(
                                Path(self.folder.name)
                                / "pair-prefix"
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
                        packet = json.loads((folder / "progress.json").read_text())
                        job["message"] = packet.get("message", job["message"])
                        if "stage_number" in packet:
                            job["progress"] = packet
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
                        self.last_complete[job["name"]] = _completed_record(job)
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
                    if job["status"] == "complete":
                        self.last_complete[name] = _completed_record(job)
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

    def remember_complete(self, name: str) -> None:
        """Record a newly published scorer without exposing an in-flight result."""
        with self.lock:
            job = self.jobs.get(name)
            if job is not None and job["status"] == "complete":
                self.last_complete[name] = _completed_record(job)

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
