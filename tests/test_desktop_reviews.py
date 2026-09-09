"""Real subprocess diagnostics and edits must preserve the fitted coefficients."""

import time

import test_desktop_models as fixtures
from test_desktop_models import revision, save_model, wait_fit

review_session = fixtures.model_session


def review(client, action, **kwargs):
    response = client.post(
        "/api/review/Frequency", json={**revision(client), "action": action, **kwargs}
    )
    assert response.status_code == 202, response.text
    task = response.json()
    if "snapshot" in task:
        return task
    for _ in range(400):
        result = client.get("/api/reviews/" + task["id"]).json()
        if result["status"] not in ("queued", "running"):
            assert result["status"] == "complete", result
            return result
        time.sleep(0.025)
    raise AssertionError("Review timed out")


def apply(client, task):
    result = client.post("/api/reviews/" + task["id"] + "/apply", json=revision(client))
    assert result.status_code == 200, result.text


def fitted(client):
    save_model(client)
    client.post("/api/models/Frequency/fit", json=revision(client))
    assert wait_fit(client)["status"] == "complete"


def test_edit_preview_apply_undo_redo_and_rebalance(review_session):
    client, _, _ = review_session
    fitted(client)
    original = client.get("/api/results/Frequency").json()
    task = review(client, "edit", variable="Age", edits={"1": 2.1})
    assert task["can_apply"]
    assert task["data"]["after_expected"] != task["data"]["before_expected"]
    assert client.get("/api/results/Frequency").json() == original
    apply(client, task)
    adjusted = client.get("/api/results/Frequency").json()
    for key in ("alpha", "features", "non_zero", "alpha_stage2"):
        assert adjusted["summary"][key] == original["summary"][key]
    assert (
        adjusted["metrics"]["train"]["expected"]
        != original["metrics"]["train"]["expected"]
    )
    assert client.get("/api/jobs").json()["Frequency"]["applicable"]
    apply(client, review(client, "undo", variable="Age"))
    assert (
        abs(
            client.get("/api/results/Frequency").json()["metrics"]["train"]["expected"]
            - original["metrics"]["train"]["expected"]
        )
        < 1e-8
    )
    apply(client, review(client, "redo", variable="Age"))
    review(client, "snapshot", snapshot="Adjusted")
    balanced = review(client, "rebalance", variable="Age")
    assert (
        abs(balanced["data"]["after_expected"] - balanced["data"]["fitted_expected"])
        < 1e-8
    )
    apply(client, balanced)
    apply(
        client, review(client, "restore_snapshot", snapshot="Adjusted", variable="Age")
    )
    assert (
        abs(
            client.get("/api/results/Frequency").json()["metrics"]["train"]["expected"]
            - adjusted["metrics"]["train"]["expected"]
        )
        < 1e-8
    )


def test_variable_pair_searches_and_smoothers(review_session):
    client, _, _ = review_session
    fitted(client)
    for subset in ("train", "holdout"):
        data = review(client, "variable", variable="Age", subset=subset)["data"]
        assert data["rows"] and "fitted_rate" in data["rows"][0]
        assert review(client, "pair", a="Age", b="Region", subset=subset)["data"][
            "rows"
        ]
    for action in ("factors", "interactions"):
        assert review(client, action)["data"]["subset"] == "train"
    for action, options in [
        ("moving", {"window": 3}),
        ("isotonic", {"direction": "increasing"}),
        ("cap", {"cap": 1.1}),
        ("round", {"decimals": 1}),
    ]:
        task = review(client, action, variable="Age", options=options)
        assert task["can_apply"] and task["data"]["rows"]


def test_stale_preview_cannot_overwrite_new_settings(review_session):
    client, _, _ = review_session
    fitted(client)
    task = review(client, "cap", variable="Age", options={"cap": 1.1})
    client.post("/api/models/Frequency/fit", json=revision(client))
    assert wait_fit(client)["status"] == "complete"
    assert client.get("/api/reviews/" + task["id"]).json()["status"] == "stale"
    assert (
        client.post(
            "/api/reviews/" + task["id"] + "/apply", json=revision(client)
        ).status_code
        == 409
    )
    task = review(client, "cap", variable="Age", options={"cap": 1.1})
    client.post(
        "/api/models/save",
        json={
            **revision(client),
            "name": "Frequency",
            "fields": {"penalty": {"alpha": 0.02}},
        },
    )
    assert client.get("/api/reviews/" + task["id"]).json()["status"] == "stale"
    assert (
        client.post(
            "/api/reviews/" + task["id"] + "/apply", json=revision(client)
        ).status_code
        == 409
    )


def test_include_interaction_and_edit_cell_without_refitting(review_session):
    client, _, _ = review_session
    fitted(client)
    response = client.post(
        "/api/review/Frequency",
        json={**revision(client), "action": "include_pair", "a": "Age", "b": "Region"},
    )
    assert response.status_code == 202, response.text
    assert not client.get("/api/jobs").json()["Frequency"]["applicable"]
    client.post("/api/models/Frequency/fit", json=revision(client))
    assert wait_fit(client)["status"] == "complete"
    table = client.get(
        "/api/results/Frequency/table", params={"variable": "Age×Region", "limit": 500}
    ).json()
    index = next(i for i, r in enumerate(table["rows"]) if r["exposure"] > 0)
    job_id = client.get("/api/jobs").json()["Frequency"]["id"]
    task = review(client, "edit", variable="Age×Region", edits={str(index): 1.8})
    apply(client, task)
    assert client.get("/api/jobs").json()["Frequency"]["id"] == job_id
    edited = client.get(
        "/api/results/Frequency/table", params={"variable": "Age×Region", "limit": 500}
    ).json()
    assert edited["rows"][index]["relativity"] == 1.8


def test_invalid_candidate_cannot_change_edit_history(review_session):
    import copy
    import inspect

    client, _, _ = review_session
    fitted(client)
    before = client.get("/api/project").json()
    task = review(client, "edit", variable="Age", edits={"1": 2.1})
    endpoint = next(
        route.endpoint
        for route in client.app.routes
        if getattr(route, "path", "") == "/api/reviews/{key}/apply"
    )
    jobs = inspect.getclosurevars(endpoint).nonlocals["reviews"]
    invalid = copy.deepcopy(jobs.get(task["id"])["data"]["project"])
    invalid["models"] = {}
    jobs.get(task["id"])["data"]["project"] = invalid
    response = client.post(
        "/api/reviews/" + task["id"] + "/apply", json=revision(client)
    )
    assert response.status_code == 409
    assert client.get("/api/project").json() == before
    assert not client.get("/api/review-info/Frequency").json()["undo"]
