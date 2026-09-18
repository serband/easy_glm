"""Feature selection runs against a Variables draft without changing the project."""

from __future__ import annotations

import inspect
import os
import sys
import tempfile
import threading
import time

import polars as pl
import pytest
from fastapi.testclient import TestClient

from easy_glm.desktop import feature_selection as selection
from easy_glm.desktop.feature_selection_ipc import read_request, write_message
from easy_glm.desktop.server import create_app
from easy_glm.workflow import Project


@pytest.fixture
def client():
    project = Project(name="Selection")
    project.data.roles = {"Claims": "target", "Age": "predictor", "Region": "predictor"}
    raw = pl.DataFrame(
        {"Claims": [0, 1, 0], "Age": [20, 40, 60], "Region": ["A", "B", "A"]}
    )
    with TestClient(
        create_app(project, raw, port=8781), base_url="http://127.0.0.1:8781"
    ) as api:
        api.headers["X-EasyGLM-Token"] = api.get("/api/session").json()["token"]
        yield api


def draft(client):
    value = client.get("/api/variables").json()
    return {key: value[key] for key in ("session_id", "revision", "setup")}


def revision(client):
    return {key: value for key, value in draft(client).items() if key != "setup"}


def selection_jobs(client):
    endpoint = next(
        route.endpoint
        for route in client.app.routes
        if getattr(route, "path", "") == "/api/variables/feature-selection"
    )
    return inspect.getclosurevars(endpoint).nonlocals["feature_selections"]


def wait_for_worker_exit(client, key):
    # A result can be published before the job thread finishes releasing its resources.
    # Tests starting another search must wait for that worker to leave first.
    thread = selection_jobs(client).tasks[key]["thread"]
    thread.join(timeout=5)
    assert not thread.is_alive(), "Feature-selection worker did not finish cleanup"


def wait_status(client, key):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = client.get("/api/feature-selections/" + key)
        assert response.status_code == 200
        packet = response.json()
        if packet["status"] not in ("queued", "running"):
            return packet
        time.sleep(0.01)
    pytest.fail("Feature selection did not finish")


class PipeWorker:
    """A fake process with real pipes, so transport and cleanup are exercised."""

    def __init__(self, command, **kwargs):
        input_read, input_write = os.pipe()
        output_read, output_write = os.pipe()
        self.stdin = os.fdopen(input_write, "wb")
        self.stdout = os.fdopen(output_read, "rb")
        self.input = os.fdopen(input_read, "rb")
        self.output = os.fdopen(output_write, "wb")
        self.returncode = None
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            with self.input, self.output:
                self.project, self.raw, self.options = read_request(self.input)
                self.respond()
        finally:
            if self.returncode is None:
                self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self):
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self.finished.set()

    def kill(self):
        self.returncode = -9
        self.finished.set()


class FinishedWorker(PipeWorker):
    def respond(self):
        write_message(
            self.output,
            {
                "type": "result",
                "value": {
                    "target": "Claims",
                    "weight": None,
                    "offset": None,
                    "rows": [
                        {"variable": "Region", "role": "predictor", "status": "signal"},
                        {"variable": "Age", "role": "predictor", "status": "no_signal"},
                    ],
                },
            },
        )


def test_draft_rename_and_options_fingerprint_do_not_apply(client, monkeypatch):
    workers = []

    def launch(*args, **kwargs):
        worker = FinishedWorker(*args, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(selection.subprocess, "Popen", launch)
    before = client.get("/api/project").json()
    body = draft(client)
    body["setup"]["renames"] = {"Age": "Region", "Region": "Age"}
    first = client.post("/api/variables/feature-selection", json=body)
    assert first.status_code == 202, first.text
    packet = wait_status(client, first.json()["id"])
    assert packet["status"] == "complete"
    assert packet["result"]["rows"][0]["raw_name"] == "Age"
    assert packet["result"]["rows"][1]["raw_name"] == "Region"
    assert packet["result"]["target_raw_name"] == "Claims"
    assert workers[0].project.data.renames == body["setup"]["renames"]
    assert workers[0].options["repeats"] == 5
    saved = client.get("/api/project").json()
    recipe = saved["exploration"].pop("feature_selection")
    assert recipe["version"] == 1
    assert recipe["project"]["data"]["renames"] == body["setup"]["renames"]
    assert recipe["project"]["models"] == {}
    assert recipe["project"]["champion"] is None
    assert recipe["project"]["exploration"] == {}
    assert recipe["options"] == workers[0].options
    assert recipe["result"] == packet["result"]
    assert saved == before
    assert client.get("/api/jobs").json() == {}
    again = client.post("/api/variables/feature-selection", json=body).json()
    assert again["id"] == first.json()["id"]
    wait_for_worker_exit(client, first.json()["id"])
    different = client.post(
        "/api/variables/feature-selection", json=body | {"options": {"seed": 7}}
    )
    assert different.status_code == 202, different.text
    assert different.json()["fingerprint"] != packet["fingerprint"]


@pytest.mark.parametrize(
    "options",
    [
        {"family": "negative_binomial"},
        {"family": "binomial", "link": "log"},
        {"family": "poisson", "tweedie_power": 1.7},
        {"family": "tweedie", "tweedie_power": 2},
        {"l1_ratio": 0},
        {"l1_ratio": 1.1},
        {"n_alphas": 1},
        {"n_alphas": 101},
        {"repeats": 0},
        {"repeats": 21},
        {"seed": -1},
        {"seed": True},
        {"include_unassigned": "yes"},
        {"unknown": 1},
    ],
)
def test_invalid_options_are_rejected_before_worker(client, monkeypatch, options):
    calls = []
    monkeypatch.setattr(selection.subprocess, "Popen", lambda *a, **k: calls.append(1))
    response = client.post(
        "/api/variables/feature-selection", json=draft(client) | {"options": options}
    )
    assert response.status_code == 422
    assert calls == []


class WaitingWorker(PipeWorker):
    entered = threading.Event()
    stopped = threading.Event()

    def respond(self):
        write_message(
            self.output,
            {
                "type": "progress",
                "value": {
                    "phase": "fit",
                    "completed": 1,
                    "total": 3,
                    "current_variable": "Age",
                    "message": "Fitting Age",
                },
            },
        )
        self.entered.set()
        assert self.finished.wait(10)

    def terminate(self):
        super().terminate()
        self.stopped.set()


def test_cancel_terminates_worker_and_revision_stales_result(client, monkeypatch):
    WaitingWorker.entered.clear()
    WaitingWorker.stopped.clear()
    monkeypatch.setattr(selection.subprocess, "Popen", WaitingWorker)
    body = draft(client)
    response = client.post("/api/variables/feature-selection", json=body)
    assert response.status_code == 202
    key = response.json()["id"]
    assert WaitingWorker.entered.wait(2)
    assert client.get("/health").status_code == 200
    cancelled = client.post(
        f"/api/feature-selections/{key}/cancel", json=revision(client)
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert WaitingWorker.stopped.wait(2)
    assert "result" not in wait_status(client, key)

    wait_for_worker_exit(client, key)
    WaitingWorker.entered.clear()
    second = client.post("/api/variables/feature-selection", json=body)
    assert second.status_code == 202, second.text
    assert WaitingWorker.entered.wait(2)
    edit = draft(client)
    edit["setup"]["roles"]["predictor"].remove("Region")
    edit["setup"]["roles"]["ignore"].append("Region")
    assert client.post("/api/variables/apply", json=edit).status_code == 200
    stale = wait_status(client, second.json()["id"])
    assert stale["status"] == "stale" and "result" not in stale
    assert client.post("/api/variables/feature-selection", json=body).status_code == 409


def test_real_worker_completes_through_pipes():
    project = Project(name="Worker smoke")
    project.data.roles = {"Claims": "target", "Constant": "predictor"}
    project.data.split.mode = "random"
    raw = pl.DataFrame({"Claims": [0, 1] * 15, "Constant": [7] * 30})
    with TestClient(
        create_app(project, raw, port=8782), base_url="http://127.0.0.1:8782"
    ) as api:
        api.headers["X-EasyGLM-Token"] = api.get("/api/session").json()["token"]
        body = draft(api)
        response = api.post("/api/variables/feature-selection", json=body)
        assert response.status_code == 202, response.text
        packet = wait_status(api, response.json()["id"])
        assert packet["status"] == "complete", packet["message"]
        assert packet["result"]["rows"][0]["status"] == "skipped"
        assert packet["result"]["rows"][0]["raw_name"] == "Constant"


def test_cancel_escalates_when_worker_ignores_terminate(client, monkeypatch):
    killed = threading.Event()

    class StubbornWorker(WaitingWorker):
        entered = threading.Event()

        def terminate(self):
            pass

        def kill(self):
            super().kill()
            killed.set()

    monkeypatch.setattr(selection.subprocess, "Popen", StubbornWorker)
    response = client.post("/api/variables/feature-selection", json=draft(client))
    assert response.status_code == 202
    key = response.json()["id"]
    assert StubbornWorker.entered.wait(5)
    assert (
        client.post(
            f"/api/feature-selections/{key}/cancel", json=revision(client)
        ).status_code
        == 200
    )
    assert killed.wait(3)
    assert wait_status(client, key)["status"] == "cancelled"


def test_feature_selection_requires_token_and_target(client, monkeypatch):
    body = draft(client)
    token = client.headers.pop("X-EasyGLM-Token")
    assert client.post("/api/variables/feature-selection", json=body).status_code == 401
    client.headers["X-EasyGLM-Token"] = token
    body["setup"]["roles"]["target"] = None
    assert client.post("/api/variables/feature-selection", json=body).status_code == 422


def test_recipe_preserves_original_candidates_across_later_apply_and_json_roundtrip(
    client, monkeypatch
):
    monkeypatch.setattr(selection.subprocess, "Popen", FinishedWorker)
    original = draft(client)
    packet = client.post("/api/variables/feature-selection", json=original).json()
    assert wait_status(client, packet["id"])["status"] == "complete"
    assert revision(client)["revision"] == original["revision"]

    edit = draft(client)
    edit["setup"]["roles"]["predictor"].remove("Age")
    edit["setup"]["roles"]["ignore"].append("Age")
    assert client.post("/api/variables/apply", json=edit).status_code == 200
    saved = client.get("/api/project").json()
    assert saved["data"]["roles"]["Age"] == "ignore"
    recipe = saved["exploration"]["feature_selection"]
    assert recipe["project"]["data"]["roles"]["Age"] == "predictor"
    assert (
        Project.from_dict(saved).to_dict()["exploration"]["feature_selection"] == recipe
    )


def test_incomplete_and_stale_selection_cannot_replace_saved_recipe(
    client, monkeypatch
):
    monkeypatch.setattr(selection.subprocess, "Popen", FinishedWorker)
    first = client.post("/api/variables/feature-selection", json=draft(client)).json()
    assert wait_status(client, first["id"])["status"] == "complete"
    expected = client.get("/api/project").json()["exploration"]["feature_selection"]
    wait_for_worker_exit(client, first["id"])

    WaitingWorker.entered.clear()
    monkeypatch.setattr(selection.subprocess, "Popen", WaitingWorker)
    second = client.post(
        "/api/variables/feature-selection",
        json=draft(client) | {"options": {"seed": 7}},
    )
    assert second.status_code == 202, second.text
    second = second.json()
    assert WaitingWorker.entered.wait(2)
    edit = draft(client)
    edit["setup"]["roles"]["predictor"].remove("Age")
    edit["setup"]["roles"]["ignore"].append("Age")
    assert client.post("/api/variables/apply", json=edit).status_code == 200
    assert wait_status(client, second["id"])["status"] == "stale"
    assert (
        client.get("/api/project").json()["exploration"]["feature_selection"]
        == expected
    )


def test_selection_is_not_publicly_complete_until_recipe_is_published(
    client, monkeypatch
):
    monkeypatch.setattr(selection.subprocess, "Popen", FinishedWorker)
    jobs = selection_jobs(client)
    original = jobs.on_complete
    entered, release = threading.Event(), threading.Event()

    def hold_publish(task, result):
        entered.set()
        assert release.wait(2)
        assert original is not None
        original(task, result)

    jobs.on_complete = hold_publish
    started = client.post("/api/variables/feature-selection", json=draft(client)).json()
    assert entered.wait(2)
    pending = client.get("/api/feature-selections/" + started["id"]).json()
    assert pending["status"] == "running" and "result" not in pending

    edit = draft(client)
    edit["setup"]["roles"]["predictor"].remove("Age")
    edit["setup"]["roles"]["ignore"].append("Age")
    assert client.post("/api/variables/apply", json=edit).status_code == 200
    release.set()
    assert wait_status(client, started["id"])["status"] == "stale"
    assert "feature_selection" not in client.get("/api/project").json()["exploration"]


@pytest.mark.parametrize("family", ["poisson", "tweedie"])
def test_real_fit_works_when_filesystem_writes_are_denied(monkeypatch, family):
    from easy_glm.workflow.feature_selection import select_variables
    from easy_glm.workflow.project import VariableDesign

    project = Project(name="No files")
    project.data.roles = {
        "y": "target",
        "x": "predictor",
        "split": "split",
        "w": "weight",
        "o": "offset",
    }
    project.data.split.mode = "column"
    project.data.split.column = "split"
    project.design.defaults.n_bins = 3
    project.design.variables["x"] = VariableDesign(knots=[0.5])
    raw = pl.DataFrame(
        {
            "y": [0.0, 3.0, 0.0, 2.0] * 30 + [9999.0] * 20,
            "x": [0, 1, 0, 1] * 35,
            "split": [1] * 120 + [0] * 20,
            "w": [1.0, 2.0] * 70,
            "o": [0.1] * 140,
        }
    )
    options = {"n_alphas": 3, "repeats": 1, "family": family}
    expected = select_variables(project, raw, **options)
    assert expected["rows"][0]["status"] == "signal"
    real_popen = selection.subprocess.Popen

    def launch(command, **kwargs):
        # The hook runs in the real child before importing the worker. Deny
        # Python filesystem writes everywhere, not just one possible temp root.
        command = [
            sys.executable,
            "-c",
            """
import os, sys

def deny_writes(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            raise PermissionError('File writes are blocked in this test')
    if event in {'os.mkdir', 'os.rename', 'os.remove', 'tempfile.mkdtemp', 'tempfile.mkstemp'}:
        raise PermissionError('File changes are blocked in this test')

sys.addaudithook(deny_writes)
from easy_glm.desktop.feature_selection_worker import main
main()
""",
        ]
        return real_popen(command, **kwargs)

    def deny_temp(*args, **kwargs):
        raise PermissionError("Feature selection must not allocate temporary files")

    monkeypatch.setattr(selection.subprocess, "Popen", launch)
    # Also catch attempts to reintroduce the parent's former job directory.
    monkeypatch.setattr(tempfile, "TemporaryDirectory", deny_temp)
    jobs = selection.FeatureSelectionJobs()
    try:
        packet = jobs.start(project, raw, ("session", "project", 0), {}, options)
        jobs.tasks[packet["id"]]["thread"].join(timeout=15)
        assert not jobs.tasks[packet["id"]]["thread"].is_alive()
        packet = jobs.status(packet["id"], ("session", "project", 0))
        assert packet["status"] == "complete", packet["message"]
        result = packet["result"]
        assert result["training_rows"] == 120
        assert result["rows"][0]["status"] == expected["rows"][0]["status"]
        assert result["rows"][0]["importance"] == pytest.approx(
            expected["rows"][0]["importance"], abs=1e-10
        )
    finally:
        jobs.close()


def test_progress_arrives_before_result_and_close_reaps_worker(client, monkeypatch):
    WaitingWorker.entered.clear()
    monkeypatch.setattr(selection.subprocess, "Popen", WaitingWorker)
    response = client.post("/api/variables/feature-selection", json=draft(client))
    key = response.json()["id"]
    assert WaitingWorker.entered.wait(5)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        packet = client.get("/api/feature-selections/" + key).json()
        if packet["progress"]["phase"] == "fit":
            break
        time.sleep(0.01)
    assert packet["status"] == "running"
    assert packet["message"] == "Fitting Age"
    assert packet["progress"]["completed"] == 1
    jobs = selection_jobs(client)
    process = jobs.tasks[key]["process"]
    jobs.close()
    wait_for_worker_exit(client, key)
    assert process.poll() is not None
    assert process.stdin.closed and process.stdout.closed


@pytest.mark.parametrize("mode", ["error", "truncated", "oversized", "crash"])
def test_worker_failure_does_not_publish_and_can_restart(client, monkeypatch, mode):
    from easy_glm.desktop.feature_selection_ipc import MESSAGE_LIMIT

    class BrokenWorker(PipeWorker):
        def respond(self):
            if mode == "error":
                write_message(
                    self.output,
                    {"type": "result", "value": {"error": "Invalid target"}},
                )
            elif mode == "truncated":
                self.output.write(b'{"type": "result"')
            elif mode == "oversized":
                self.output.write(b"x" * (MESSAGE_LIMIT + 1))
                self.output.flush()
            else:
                self.returncode = 1

    monkeypatch.setattr(selection.subprocess, "Popen", BrokenWorker)
    response = client.post("/api/variables/feature-selection", json=draft(client))
    key = response.json()["id"]
    packet = wait_status(client, key)
    assert packet["status"] == "failed"
    assert "result" not in packet
    if mode == "error":
        assert packet["message"] == "Invalid target"
    assert "feature_selection" not in client.get("/api/project").json()["exploration"]
    wait_for_worker_exit(client, key)
    monkeypatch.setattr(selection.subprocess, "Popen", FinishedWorker)
    restarted = client.post("/api/variables/feature-selection", json=draft(client))
    assert restarted.status_code == 202
    assert wait_status(client, restarted.json()["id"])["status"] == "complete"


def test_pipe_request_preserves_column_types_and_values():
    from datetime import date

    from polars.testing import assert_frame_equal

    from easy_glm.desktop.feature_selection_ipc import write_request

    raw = pl.DataFrame(
        {
            "category": pl.Series(["A", None, "B"], dtype=pl.Categorical),
            "number": [1.0, float("nan"), None],
            "date": [date(2025, 1, 1), None, date(2025, 2, 1)],
            "unicode": ["café", "東京", "\n"],
        }
    )
    project = Project(name="Unicode é")
    worker = FinishedWorker([])
    try:
        with worker.stdin:
            write_request(worker.stdin, project, raw, {"seed": 7})
        worker.wait()
        assert worker.project.to_dict() == project.to_dict()
        assert_frame_equal(worker.raw, raw)
        assert worker.options == {"seed": 7}
    finally:
        worker.stdout.close()


def test_cancel_during_large_input_transfer_reaps_real_worker(monkeypatch):
    real_popen = selection.subprocess.Popen
    processes = []

    def launch(command, **kwargs):
        process = real_popen(
            [
                sys.executable,
                "-c",
                """
import sys, threading
from easy_glm.desktop.feature_selection_ipc import write_message
write_message(sys.stdout.buffer, {'type': 'progress', 'value': {'phase': 'waiting'}})
threading.Event().wait(30)
""",
            ],
            **kwargs,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(selection.subprocess, "Popen", launch)
    jobs = selection.FeatureSelectionJobs()
    generation = ("session", "project", 0)
    try:
        # Larger than the OS pipe buffer; the child deliberately never reads it.
        packet = jobs.start(
            Project(), pl.DataFrame({"x": range(500_000)}), generation, {}, {}
        )
        key = packet["id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if jobs.status(key, generation)["progress"]["phase"] == "waiting":
                break
            time.sleep(0.01)
        assert jobs.status(key, generation)["progress"]["phase"] == "waiting"
        jobs.cancel(key)
        jobs.tasks[key]["thread"].join(timeout=5)
        assert not jobs.tasks[key]["thread"].is_alive()
        assert jobs.status(key, generation)["status"] == "cancelled"
        assert processes[0].poll() is not None
        assert processes[0].stdin.closed and processes[0].stdout.closed
    finally:
        jobs.close()
