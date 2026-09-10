"""Local source onboarding is validated and replaces session state atomically."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import polars as pl
import pytest
from fastapi.testclient import TestClient
from test_desktop_models import model_session as model_session
from test_desktop_models import revision, save_model
from test_desktop_refit import internals
from test_desktop_reviews import apply, fitted, review

from easy_glm.desktop import server
from easy_glm.desktop.loading import load_project_input
from easy_glm.workflow.project import Project


def token(client):
    client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]


@pytest.mark.parametrize("kind", ["csv", "parquet", "xlsx", "ipc"])
def test_load_local_data_types_without_roles_or_overwriting_split(tmp_path, kind):
    frame = pl.DataFrame({"claim": [0, 1, 2], "traintest": [7, 7, 8]})
    path = tmp_path / f"portfolio.{kind}"
    if kind == "xlsx":
        frame.write_excel(path)
    else:
        getattr(frame, f"write_{kind}")(path)
    before = path.read_bytes()
    project, raw = load_project_input("data", path)
    assert raw.equals(frame)
    assert not project.data.roles and not project.models
    assert project.data.split.mode == "random"
    assert project.data.split.column == "traintest_2"
    assert path.read_bytes() == before


def test_empty_app_then_open_project_with_relative_data_path(tmp_path):
    pl.DataFrame({"claim": [0, 1]}).write_parquet(tmp_path / "raw.parquet")
    project = Project(name="Saved portfolio")
    project.data.source.path = "raw.parquet"
    project.data.split.mode = "random"
    path = tmp_path / "project.json"
    project.to_json(path)
    before = path.read_bytes()
    with TestClient(
        server.create_app(Project(), pl.DataFrame(), port=8780),
        base_url="http://127.0.0.1:8780",
    ) as client:
        token(client)
        old = client.get("/api/variables").json()
        response = client.post(
            "/api/project/open",
            json={**revision(client), "kind": "project", "path": str(path)},
        )
        assert response.status_code == 200, response.text
        new = response.json()
        assert new["row_count"] == 2 and new["name"] == "Saved portfolio"
        assert new["session_id"] != old["session_id"]
        assert new["project_id"] != old["project_id"] and new["revision"] == 0
        assert client.get("/api/project").status_code == 401
        token(client)
        assert client.get("/api/project").json()["data"]["source"]["path"] == str(
            tmp_path / "raw.parquet"
        )
        assert client.get("/api/jobs").json() == {}
    assert path.read_bytes() == before


def test_open_clears_applied_fit_history_and_cancels_active_work(
    model_session, tmp_path
):
    client, _, _ = model_session
    fitted(client)
    apply(client, review(client, "edit", variable="Age", edits={"1": 2.1}))
    old_jobs, undo, redo = internals(client)
    assert undo
    response = client.post("/api/models/Frequency/fit", json=revision(client))
    assert response.status_code == 202
    path = tmp_path / "replacement.csv"
    pl.DataFrame({"Amount": [2, 3, 4]}).write_csv(path)
    response = client.post(
        "/api/project/open",
        json={**revision(client), "kind": "data", "path": str(path)},
    )
    assert response.status_code == 200, response.text
    assert not undo and not redo
    assert old_jobs.active is None or old_jobs.active["cancel"]
    if old_jobs.thread:
        assert not old_jobs.thread.is_alive()
    token(client)
    assert client.get("/api/jobs").json() == {}
    assert client.get("/api/project").json()["models"] == {}
    assert client.get("/api/variables").json()["row_count"] == 3


def test_bad_source_and_project_preserve_project_fit_history_and_session(
    model_session, tmp_path
):
    client, _, _ = model_session
    fitted(client)
    apply(client, review(client, "edit", variable="Age", edits={"1": 2.1}))
    before = client.get("/api/project").json()
    snapshot = client.get("/api/variables").json()
    jobs = client.get("/api/jobs").json()
    _, undo, redo = internals(client)
    history = deepcopy((undo, redo))
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    for kind, path in [
        ("project", bad),
        ("data", tmp_path / "missing.csv"),
        ("data", bad),
    ]:
        response = client.post(
            "/api/project/open",
            json={**revision(client), "kind": kind, "path": str(path)},
        )
        assert response.status_code == 422, response.text
        assert client.get("/api/project").json() == before
        assert client.get("/api/variables").json() == snapshot
        assert client.get("/api/jobs").json() == jobs
        assert (undo, redo) == history


def test_loading_rechecks_revision_and_does_not_hold_project_lock(
    model_session, tmp_path, monkeypatch
):
    client, _, _ = model_session
    path = tmp_path / "new.csv"
    pl.DataFrame({"x": [1, 2]}).write_csv(path)
    started, release = threading.Event(), threading.Event()
    real_load = server.load_project_input

    def held_load(*args):
        loaded = real_load(*args)
        started.set()
        assert release.wait(10)
        return loaded

    monkeypatch.setattr(server, "load_project_input", held_load)
    old = revision(client)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            client.post,
            "/api/project/open",
            json={**old, "kind": "data", "path": str(path)},
        )
        try:
            assert started.wait(5)
            assert client.get("/api/project").status_code == 200
            save_model(client)
            before = client.get("/api/project").json()
        finally:
            release.set()
        response = future.result(timeout=5)
        assert response.status_code == 409, response.text
    assert client.get("/api/project").json() == before
    assert revision(client)["session_id"] == old["session_id"]
