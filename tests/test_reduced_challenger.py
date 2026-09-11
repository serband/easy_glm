import time
from copy import deepcopy
from unittest.mock import patch

import pytest
from test_desktop_models import model_session as model_session
from test_desktop_models import revision
from test_desktop_reviews import apply, fitted, review

from easy_glm.workflow.project import Interaction, Project, TableSnapshot
from easy_glm.workflow.reduction import reduced_challenger


def test_reduction_preserves_original_and_fitting_settings():
    project = Project()
    project.data.roles = {
        "a": "predictor",
        "b": "predictor",
        "c": "predictor",
        "y": "target",
        "w": "weight",
    }
    project.data.split.mode = "random"
    cfg = project.new_model("source", family="tweedie")
    cfg.interactions = [Interaction("a", "b"), Interaction("b", "c")]
    cfg.monotone = {"a": "increasing", "c": "decreasing"}
    cfg.base_rate_override = 3
    cfg.snapshots = [TableSnapshot("old")]
    project.champion = "source"
    before = project.to_dict()
    candidate = reduced_challenger(project, "source", "small", ["b", "a"])
    assert project.to_dict() == before
    assert candidate.models["source"] == cfg
    small = candidate.models["small"]
    assert small.predictors == ["a", "b"]
    assert [(i.a, i.b) for i in small.interactions] == [("a", "b")]
    assert small.monotone == {"a": "increasing"}
    assert (
        not small.adjustments
        and not small.snapshots
        and small.base_rate_override is None
    )
    assert small.penalty == cfg.penalty and small.penalty is not cfg.penalty
    assert small.family == cfg.family
    assert small.tweedie_power == cfg.tweedie_power
    assert candidate.data == project.data and candidate.design == project.design
    assert candidate.champion == "source"


@pytest.mark.parametrize(
    "name,keep",
    [
        ("source", ["a"]),
        ("small", []),
        ("small", ["a", "a"]),
        ("small", ["unknown"]),
        ("small", ["a", "b"]),
    ],
)
def test_invalid_reductions_do_not_change_project(name, keep):
    project = Project()
    project.data.roles = {"a": "predictor", "b": "predictor", "y": "target"}
    project.new_model("source")
    before = project.to_dict()
    with pytest.raises(ValueError):
        reduced_challenger(project, "source", name, keep)
    assert project.to_dict() == before


def test_real_challenger_fit_preserves_source_and_split(model_session):
    client, _, _ = model_session
    fitted(client)
    apply(client, review(client, "edit", variable="Age", edits={"1": 2.1}))
    source = client.get("/api/project").json()
    result = client.get("/api/results/Frequency").json()
    source_job = client.get("/api/jobs").json()["Frequency"]
    response = client.post(
        "/api/models/Frequency/reduce",
        json={
            **revision(client),
            "name": "Smaller",
            "fit_id": source_job["id"],
            "predictors": ["Age"],
        },
    )
    assert response.status_code == 202, response.text
    for _ in range(600):
        jobs = client.get("/api/jobs").json()
        if jobs["Smaller"]["status"] not in ("queued", "running"):
            break
        time.sleep(0.025)
    assert jobs["Smaller"]["applicable"], jobs
    after = client.get("/api/project").json()
    assert after["models"]["Frequency"] == source["models"]["Frequency"]
    assert after["data"] == source["data"]
    assert (
        jobs["Frequency"]["id"] == source_job["id"] and jobs["Frequency"]["applicable"]
    )
    assert client.get("/api/results/Frequency").json() == result
    small = client.get("/api/results/Smaller").json()
    for subset in ["train", "holdout", "all"]:
        for key in ["rows", "actual", "exposure"]:
            assert small["metrics"][subset][key] == pytest.approx(
                result["metrics"][subset][key]
            )
    assert after["models"]["Smaller"]["adjustments"] == []
    old = deepcopy(after)
    for payload in [
        {"name": "Smaller", "fit_id": source_job["id"]},
        {"name": "Other", "fit_id": "old-fit"},
    ]:
        response = client.post(
            "/api/models/Frequency/reduce",
            json={**revision(client), **payload, "predictors": ["Age"]},
        )
        assert response.status_code == 409
        assert client.get("/api/project").json() == old


def test_busy_fit_does_not_create_orphan_model(model_session):
    client, _, _ = model_session
    fitted(client)
    before = client.get("/api/project").json()
    job = client.get("/api/jobs").json()["Frequency"]
    with patch(
        "easy_glm.desktop.jobs.FitJobs.start",
        side_effect=ValueError("A fit is already running."),
    ):
        response = client.post(
            "/api/models/Frequency/reduce",
            json={
                **revision(client),
                "name": "Small",
                "fit_id": job["id"],
                "predictors": ["Age"],
            },
        )
    assert response.status_code == 409
    assert client.get("/api/project").json() == before
