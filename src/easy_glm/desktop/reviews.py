"""Short-lived background review processes and revision-bound previews."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from easy_glm.app import _launcher_env
from easy_glm.desktop.modeling import Revision
from easy_glm.workflow.project import Project


class ReviewEdit(Revision):
    action: Literal[
        "include_factor",
        "include_pair",
        "variable",
        "pair",
        "factors",
        "interactions",
        "edit",
        "moving",
        "isotonic",
        "cap",
        "round",
        "rebalance",
        "undo",
        "redo",
        "snapshot",
        "restore_snapshot",
        "reset",
    ]
    variable: str | None = None
    subset: Literal["train", "holdout"] = "train"
    a: str | None = None
    b: str | None = None
    edits: dict[str, float] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    snapshot: str = ""


class ReviewJobs:
    def __init__(self) -> None:
        self.folder = tempfile.TemporaryDirectory(prefix="easyglm_reviews_")
        self.lock = threading.RLock()
        self.tasks: dict[str, dict[str, Any]] = {}

    def start(
        self, project: Project, source: Path, request: dict[str, Any], revision: int
    ) -> dict[str, str]:
        with self.lock:
            if any(t["status"] in ("queued", "running") for t in self.tasks.values()):
                raise ValueError("A review is running. Wait for it or cancel it first.")
            if len(self.tasks) >= 64:
                self.tasks.pop(next(iter(self.tasks)))
            key = str(time.time_ns())
            task = {
                "id": key,
                "revision": revision,
                "request": request,
                "status": "queued",
                "cancel": False,
                "process": None,
            }
            self.tasks[key] = task
            thread = threading.Thread(
                target=self._run, args=(task, project, source), daemon=True
            )
            task["thread"] = thread
            thread.start()
            return {"id": key}

    def _run(self, task: dict[str, Any], project: Project, source: Path) -> None:
        folder = Path(self.folder.name) / task["id"]
        try:
            folder.mkdir()
            project.to_json(folder / "project.json")
            (folder / "request.json").write_text(json.dumps(task["request"]))
            with self.lock:
                if task["cancel"]:
                    return
                with (folder / "worker.log").open("w") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "easy_glm.desktop.review_worker",
                            str(folder),
                            str(source),
                        ],
                        env=_launcher_env(),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                task.update(process=process, status="running")
            process.wait()
            with self.lock:
                if task["cancel"]:
                    return
                data = json.loads((folder / "result.json").read_text())
                task.update(
                    status="failed" if "error" in data else "complete", data=data
                )
        except Exception as exc:
            with self.lock:
                if not task["cancel"]:
                    task.update(status="failed", data={"error": str(exc)})

    def get(self, key: str) -> dict[str, Any]:
        with self.lock:
            if key not in self.tasks:
                raise ValueError("Review expired. Run the preview again.")
            return self.tasks[key]

    def cancel(self, key: str) -> None:
        with self.lock:
            task = self.get(key)
            if task["status"] in ("queued", "running"):
                task.update(cancel=True, status="cancelled")
                if task["process"]:
                    task["process"].terminate()

    def close(self) -> None:
        for key in list(self.tasks):
            self.cancel(key)
        for task in list(self.tasks.values()):
            task["thread"].join(timeout=5)
            process = task.get("process")
            if process and process.poll() is None:
                process.kill()
                process.wait()
        self.folder.cleanup()
