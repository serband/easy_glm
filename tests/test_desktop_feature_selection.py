"""Feature selection runs against a Variables draft without changing the project."""

from __future__ import annotations

import inspect
import json
import threading
import time
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from easy_glm.desktop import feature_selection as selection
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
    # A result can be published before the worker finishes cleaning up its files.
    # Tests starting another search must wait for that worker to leave first.
    thread = selection_jobs(client).tasks[key]["thread"]
    thread.join(timeout=5)
    assert not thread.is_alive(), "Feature-selection worker did not finish cleanup"


def wait_status(client, key):
    for _ in range(200):
        response = client.get("/api/feature-selections/" + key)
        assert response.status_code == 200
        packet = response.json()
        if packet["status"] not in ("queued", "running"):
            return packet
        time.sleep(0.01)
    pytest.fail("Feature selection did not finish")


class FinishedWorker:
    def __init__(self, command, **kwargs):
        folder = Path(command[-1])
        self.project = Project.from_json(folder / "project.json")
        self.raw = pl.read_parquet(folder / "raw.parquet")
        self.options = json.loads((folder / "options.json").read_text())
        self.returncode = 0
        (folder / "result.json").write_text(
            json.dumps(
                {
                    "target": "Claims",
                    "weight": None,
                    "offset": None,
                    "rows": [
                        {"variable": "Region", "role": "predictor", "status": "signal"},
                        {"variable": "Age", "role": "predictor", "status": "no_signal"},
                    ],
                }
            )
        )

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15


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


class WaitingWorker:
    entered = threading.Event()
    stopped = threading.Event()

    def __init__(self, command, **kwargs):
        self.returncode = None
        self.folder = Path(command[-1])
        (self.folder / "progress.json").write_text(
            json.dumps(
                {
                    "phase": "fit",
                    "completed": 1,
                    "total": 3,
                    "current_variable": "Age",
                    "message": "Fitting Age",
                }
            )
        )
        self.entered.set()

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15
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


def test_real_worker_completes_and_cleans_its_inputs():
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
            self.returncode = -9
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
