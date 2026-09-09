"""Canonical comparisons, applicability and upgrade restoration."""

import json
import pickle
import time

import numpy as np
import pytest
from test_desktop_models import model_session as model_session
from test_desktop_models import revision
from test_desktop_reviews import fitted, review

from easy_glm.desktop.diagnostic_views import safe_gini
from easy_glm.desktop.fit_worker import result_for, write_json
from easy_glm.desktop.jobs import FitJobs, model_key
from easy_glm.workflow.prep import prepare
from easy_glm.workflow.run import run_model


def fit_other(client, name="Challenger", **fields):
    response = client.post(
        "/api/models/save",
        json={
            **revision(client),
            "name": name,
            "create": True,
            "fields": {
                "predictors": ["Age"],
                "divide_target_by_weight": True,
                "penalty": {"alpha": 0.05, "cv": None},
                **fields,
            },
        },
    )
    assert response.status_code == 200, response.text
    client.post(f"/api/models/{name}/fit", json=revision(client))
    for _ in range(600):
        job = client.get("/api/jobs").json()[name]
        if job["status"] not in ("queued", "running"):
            assert job["status"] == "complete", job
            return
        time.sleep(0.025)
    pytest.fail("fit timed out")


def test_two_models_diagnostics_and_champion(model_session):
    client, _, _ = model_session
    fitted(client)
    fit_other(client)
    before = client.get("/api/jobs").json()
    for action in ["lift", "double_lift", "path", "coefficients", "compare"]:
        data = review(client, action, challenger="Challenger", subset="all")["data"]
        assert data.get("charts") or data.get("tables")
    variable = review(
        client,
        "variable",
        challenger="Challenger",
        variable="Age",
        options={"both_subsets": True},
    )["data"]
    assert "challenger_rate" in variable["rows"][0]
    assert len(variable["ae_sets"]) == 1
    pair = review(client, "pair", challenger="Challenger", a="Age", b="Region")["data"]
    assert "challenger_ae" in pair["rows"][0]
    null = review(client, "double_lift", subset="holdout")["data"]
    assert "training" in null["note"].lower()
    promoted = client.post(
        "/api/review/Challenger", json={**revision(client), "action": "champion"}
    )
    assert promoted.status_code == 202, promoted.text
    assert client.get("/api/project").json()["champion"] == "Challenger"
    assert client.get("/api/jobs").json() == before
    result = client.get("/api/results/Frequency").json()
    assert result["metrics"]["all"]["rows"] == 1200
    assert result["metrics"]["all"]["deviance"] == pytest.approx(
        sum(result["metrics"][s]["deviance"] for s in ("train", "holdout"))
    )


def test_incompatible_comparison_is_a_message(model_session):
    client, _, _ = model_session
    fitted(client)
    fit_other(client, divide_target_by_weight=False)
    response = client.post(
        "/api/review/Frequency",
        json={**revision(client), "action": "double_lift", "challenger": "Challenger"},
    )
    assert response.status_code == 202
    for _ in range(300):
        data = client.get("/api/reviews/" + response.json()["id"]).json()
        if data["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)
    assert data["status"] == "failed"
    assert "divide target by weight differs" in data["data"]["error"]


def test_signed_gini_and_upgrade_without_solver(model_session, tmp_path, monkeypatch):
    _, project, raw = model_session
    from easy_glm.workflow.project import ModelConfig

    project.models["Frequency"] = ModelConfig(
        target="Claims",
        weight="Exposure",
        predictors=["Age"],
        divide_target_by_weight=True,
    )
    project.models["Frequency"].penalty.alpha = 0.01
    frame = prepare(project, raw)
    run = run_model(project, frame, "Frequency")
    values = run.predict(frame).copy()
    coefs = run.fit.coef.copy()
    assert safe_gini(np.array([-1.0, 2.0]), np.ones(2), np.ones(2)) is None
    assert safe_gini(np.ones(2), np.ones(2), np.ones(2)) is None
    folder = tmp_path / "fits" / "123"
    folder.mkdir(parents=True)
    with (folder / "fit.pkl").open("wb") as f:
        pickle.dump(run, f)
    raw.write_parquet(folder / "raw.parquet")
    write_json(folder / "result.json", result_for(project, frame, run))
    record = {
        "id": "123",
        "name": "Frequency",
        "status": "complete",
        "message": "Fit complete",
        "elapsed": 1.25,
        "key": model_key(project, "Frequency"),
        "applicable": True,
    }
    (tmp_path / "jobs.json").write_text(json.dumps({"Frequency": record}))
    import easy_glm.workflow.run as workflow_run

    monkeypatch.setattr(
        workflow_run,
        "fit_glm",
        lambda *a, **k: pytest.fail("Restoration must not refit"),
    )
    jobs = FitJobs()
    try:
        jobs.restore(project, raw, tmp_path)
        assert jobs.status(project)["Frequency"] == record
        with (jobs.artifact(project, "Frequency") / "fit.pkl").open("rb") as f:
            restored = pickle.load(f)
        np.testing.assert_array_equal(restored.fit.coef, coefs)
        np.testing.assert_array_equal(restored.predict(frame), values)
        assert (
            jobs.result(project, "Frequency")["diagnostic_info"]["facts"]["family"]
            == "poisson"
        )
    finally:
        jobs.close()


@pytest.mark.parametrize(
    "family,link", [("gaussian", "log"), ("binomial", "logit"), ("gamma", "log")]
)
def test_family_applicability_uses_real_fits(family, link):
    import polars as pl

    from easy_glm.desktop.diagnostic_views import metadata, view
    from easy_glm.workflow.project import ModelConfig, Project

    rng = np.random.default_rng(21)
    x = rng.uniform(-1, 1, 400)
    y = (
        0.5 + x + rng.normal(0, 0.1, 400)
        if family == "gaussian"
        else (
            rng.binomial(1, 0.5 + 0.2 * x)
            if family == "binomial"
            else rng.gamma(2, np.exp(0.3 * x) / 2)
        )
    )
    raw = pl.DataFrame({"y": y, "x": x})
    p = Project(name="Family applicability")
    p.data.roles = {"y": "target", "x": "predictor"}
    p.data.split.mode = "random"
    p.models["Model"] = ModelConfig(
        target="y", family=family, link=link, predictors=["x"]
    )
    p.models["Model"].penalty.alpha = 0.01
    frame = prepare(p, raw)
    run = run_model(p, frame, "Model")
    result = result_for(p, frame, run)
    assert result["metrics"]["all"]["rows"] == 400
    assert view(p, run, frame, {"action": "lift", "subset": "holdout"})["charts"]
    if family == "gaussian":
        assert not metadata(run, frame)["ratio_suitable"]
        assert result["metrics"]["all"]["gini"] is None
        with pytest.raises(ValueError, match="nonnegative"):
            view(p, run, frame, {"action": "double_lift", "subset": "holdout"})
    else:
        assert metadata(run, frame)["ratio_suitable"]
        assert view(p, run, frame, {"action": "double_lift", "subset": "holdout"})[
            "charts"
        ]


def test_challenger_refit_invalidates_comparison_preview(model_session):
    client, _, _ = model_session
    fitted(client)
    fit_other(client)
    task = review(
        client, "edit", challenger="Challenger", variable="Age", edits={"1": 2.1}
    )
    assert task["can_apply"]
    before = client.get("/api/jobs").json()["Challenger"]["id"]
    client.post("/api/models/Challenger/fit", json=revision(client))
    for _ in range(600):
        job = client.get("/api/jobs").json()["Challenger"]
        if job["id"] != before and job["status"] == "complete":
            break
        time.sleep(0.025)
    assert job["id"] != before
    assert client.get("/api/reviews/" + task["id"]).json()["status"] == "stale"
    response = client.post(
        "/api/reviews/" + task["id"] + "/apply", json=revision(client)
    )
    assert response.status_code == 409
