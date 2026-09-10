"""Explicit new fits reset overlays only when the clean result can be committed."""

from __future__ import annotations

import inspect
import pickle
import threading
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_desktop_models import model_session as model_session
from test_desktop_models import revision, wait_fit
from test_desktop_reviews import apply, fitted, review


def internals(client):
    endpoint = next(
        route.endpoint
        for route in client.app.routes
        if getattr(route, "path", "") == "/api/reviews/{key}/apply"
    )
    state = inspect.getclosurevars(endpoint).nonlocals
    return state["jobs"], state["undo_steps"], state["redo_steps"]


def applied_state(client):
    fitted(client)
    original = client.get("/api/results/Frequency").json()
    apply(client, review(client, "edit", variable="Age", edits={"1": 2.1}))
    apply(client, review(client, "rebalance", variable="Age"))
    review(client, "snapshot", snapshot="Saved rates")
    cfg = client.get("/api/project").json()["models"]["Frequency"]
    assert cfg["adjustments"] and cfg["base_rate_override"] is not None
    return original


def hold_completion(jobs):
    arrived = threading.Event()
    release = threading.Event()
    complete = jobs.on_complete

    def held(job, result):
        arrived.set()
        if not release.wait(30):
            raise TimeoutError("Test did not release the completed fit")
        complete(job, result)

    jobs.on_complete = held
    return arrived, release


def test_successful_refit_commits_clean_rates_and_keeps_private_recovery(model_session):
    client, _, _ = model_session
    original = applied_state(client)
    before = client.get("/api/project").json()
    prior_revision = revision(client)
    jobs, undo, redo = internals(client)
    history = deepcopy((undo, redo))
    old_id = client.get("/api/jobs").json()["Frequency"]["id"]
    old_fit_path = Path(jobs.folder.name) / old_id / "fit.pkl"
    old_fit_bytes = old_fit_path.read_bytes()
    arrived, release = hold_completion(jobs)
    try:
        response = client.post("/api/models/Frequency/fit", json=prior_revision)
        assert response.status_code == 202, response.text
        assert arrived.wait(30)
        assert client.get("/health").status_code == 200
        assert client.get("/api/jobs").json()["Frequency"]["status"] == "running"
        assert client.get("/api/project").json() == before
        assert revision(client) == prior_revision
        assert (undo, redo) == history
    finally:
        release.set()
    status = wait_fit(client)
    assert status["status"] == "complete" and status["applicable"]
    assert status["id"] != old_id
    current = client.get("/api/project").json()
    expected = deepcopy(before)
    expected["models"]["Frequency"]["adjustments"] = []
    expected["models"]["Frequency"]["base_rate_override"] = None
    assert current == expected  # Includes the named snapshot unchanged.
    assert revision(client)["revision"] == prior_revision["revision"] + 1
    assert "Frequency" not in undo and "Frequency" not in redo
    result = client.get("/api/results/Frequency").json()
    assert result["base_rate"] == pytest.approx(original["base_rate"], rel=1e-12)
    assert result["metrics"]["train"]["expected"] == pytest.approx(
        original["metrics"]["train"]["expected"], rel=1e-12
    )
    for variable in ("Age", "Region"):
        table = client.get(
            "/api/results/Frequency/table", params={"variable": variable}
        ).json()
        assert all(row["fitted"] == row["relativity"] for row in table["rows"])
    assert old_fit_path.read_bytes() == old_fit_bytes
    folder = Path(jobs.folder.name) / status["id"]
    from easy_glm.workflow.project import Project

    assert Project.from_json(folder / "project-before-refit.json").to_dict() == before
    assert Project.from_json(folder / "project.json").to_dict() == current
    with (folder / "fit.pkl").open("rb") as handle:
        new_fit = pickle.load(handle)
    old_fit = pickle.loads(old_fit_bytes)
    np.testing.assert_allclose(
        new_fit.fit.coef, old_fit.fit.coef, rtol=1e-12, atol=1e-13
    )
    assert not new_fit.config.adjustments and new_fit.config.base_rate_override is None
    # A read and a dynamic tool preview after fitting still cannot apply anything.
    review(client, "variable", variable="Age")
    review(client, "moving", variable="Age", options={"window": 3})
    assert client.get("/api/project").json() == current
    assert client.get("/api/jobs").json()["Frequency"]["id"] == status["id"]


@pytest.mark.parametrize("outcome", ["cancelled", "stale"])
def test_obsolete_completed_refit_preserves_overlays_and_history(
    model_session, outcome
):
    client, _, _ = model_session
    applied_state(client)
    jobs, undo, redo = internals(client)
    before = client.get("/api/project").json()
    history = deepcopy((undo, redo))
    arrived, release = hold_completion(jobs)
    try:
        response = client.post("/api/models/Frequency/fit", json=revision(client))
        assert response.status_code == 202
        assert arrived.wait(30)
        if outcome == "cancelled":
            response = client.post(
                "/api/models/Frequency/cancel", json=revision(client)
            )
        else:
            response = client.post(
                "/api/models/save",
                json={
                    **revision(client),
                    "name": "Frequency",
                    "fields": {"penalty": {"alpha": 0.02}},
                },
            )
            before["models"]["Frequency"]["penalty"]["alpha"] = 0.02
        assert response.status_code == 200, response.text
        assert client.get("/api/project").json() == before
        assert (undo, redo) == history
    finally:
        release.set()
        jobs.thread.join(timeout=5)
    status = client.get("/api/jobs").json()["Frequency"]
    assert status["status"] == outcome and not status["applicable"]
    assert client.get("/api/project").json() == before
    assert (undo, redo) == history


def test_failed_worker_refit_preserves_overlays_and_history(model_session):
    client, _, _ = model_session
    applied_state(client)
    # A valid model definition whose string response cannot be fitted by Poisson.
    response = client.post(
        "/api/models/save",
        json={
            **revision(client),
            "name": "Frequency",
            "fields": {"target": "Region", "predictors": ["Age"]},
        },
    )
    assert response.status_code == 200, response.text
    before = client.get("/api/project").json()
    prior_revision = revision(client)
    _, undo, redo = internals(client)
    history = deepcopy((undo, redo))
    response = client.post("/api/models/Frequency/fit", json=prior_revision)
    assert response.status_code == 202, response.text
    status = wait_fit(client)
    assert status["status"] == "failed" and not status["applicable"]
    assert client.get("/api/project").json() == before
    assert revision(client) == prior_revision
    assert (undo, redo) == history
