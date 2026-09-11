"""Draft-only scans must remain responsive, cancellable and revision-bound."""

from __future__ import annotations

import sys
import threading
import time
from copy import deepcopy
from types import ModuleType

import polars as pl
import pytest
from fastapi.testclient import TestClient

from easy_glm.desktop.screening import ScreeningJobs, ScreeningOptions
from easy_glm.desktop.server import create_app
from easy_glm.workflow import Project


@pytest.fixture
def scanner(monkeypatch):
    module = ModuleType("easy_glm.workflow.screening")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.screen_variables = lambda *args, **kwargs: packet()
    return module


def packet(name="Age", other="Region"):
    return {
        "target": "Claims",
        "weight": None,
        "leakage": [{"variable": name, "association": 0.99}],
        "missing": [{"variable": name, "missing_share": 0.8}],
        "unsupported": [{"variable": other, "reason": "constant"}],
        "correlated": [{"first": name, "second": other}],
        "columns": [{"variable": name}, {"name": other}],
    }


@pytest.fixture
def client():
    project = Project(name="Screening")
    project.data.roles = {"Claims": "target", "Age": "predictor", "Region": "predictor"}
    project.data.split.mode = "random"
    raw = pl.DataFrame(
        {"Claims": [0, 1, 0], "Age": [20, 40, 60], "Region": ["A", "B", "A"]}
    )
    with TestClient(
        create_app(project, raw, port=8780), base_url="http://127.0.0.1:8780"
    ) as api:
        api.headers["X-EasyGLM-Token"] = api.get("/api/session").json()["token"]
        yield api


def draft(client):
    value = client.get("/api/variables").json()
    return {key: value[key] for key in ("session_id", "revision", "setup")}


def revision(client):
    return {key: value for key, value in draft(client).items() if key != "setup"}


def start(client, body=None):
    response = client.post("/api/variables/screen", json=body or draft(client))
    assert response.status_code == 202, response.text
    return response.json()


def done(client, job):
    for _ in range(200):
        response = client.get("/api/screenings/" + job["id"])
        assert response.status_code == 200
        value = response.json()
        if value["status"] not in ("queued", "running"):
            return value
        time.sleep(0.01)
    pytest.fail("Scan did not finish")


def test_scan_uses_draft_renames_and_maps_raw_ids_without_applying(client, scanner):
    before = client.get("/api/project").json()
    body = draft(client)
    body["setup"]["renames"] = {"Age": "Region", "Region": "Age"}
    captured = []

    def scan(project, raw, **kwargs):
        captured.append((project.to_dict(), raw.columns, kwargs))
        return packet("Region", "Age")

    scanner.screen_variables = scan
    job = done(client, start(client, body))
    assert job["status"] == "complete"
    result = job["result"]
    for group in ("leakage", "missing", "columns"):
        assert result[group][0]["raw_name"] == "Age"
    assert result["unsupported"][0]["raw_name"] == "Region"
    assert result["columns"][1]["raw_name"] == "Region"
    assert result["correlated"][0]["first_raw"] == "Age"
    assert result["correlated"][0]["second_raw"] == "Region"
    assert result["target_raw_name"] == "Claims"
    assert captured[0][0]["data"]["renames"] == body["setup"]["renames"]
    assert captured[0][2]["sample_rows"] == 10_000
    assert client.get("/api/project").json() == before
    assert client.get("/api/jobs").json() == {}
    assert draft(client)["revision"] == body["revision"]
    again = start(client, body)
    assert again["id"] == job["id"] and len(captured) == 1
    changed = deepcopy(body)
    changed["options"] = {"seed": 7}
    assert start(client, changed)["fingerprint"] != job["fingerprint"]


def test_running_scan_reports_progress_allows_reads_and_cancels(client, scanner):
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()

    def scan(project, raw, *, progress, cancelled, **options):
        progress(
            {
                "phase": "correlation",
                "completed": 3,
                "total": 1_000,
                "message": "Checking pairs",
            }
        )
        entered.set()
        try:
            while not release.wait(0.01) and not cancelled():
                pass
            return packet()
        finally:
            exited.set()

    scanner.screen_variables = scan
    job = start(client)
    try:
        assert entered.wait(2)
        assert client.get("/health").status_code == 200
        assert client.get("/api/variables").status_code == 200
        assert client.get("/api/jobs").json() == {}
        status = client.get("/api/screenings/" + job["id"]).json()
        assert status["progress"]["completed"] == 3
        assert status["message"] == "Checking pairs"
        assert start(client)["id"] == job["id"]
        other = draft(client) | {"options": {"seed": 7}}
        assert client.post("/api/variables/screen", json=other).status_code == 422
        cancelled = client.post(
            "/api/screenings/" + job["id"] + "/cancel", json=revision(client)
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert exited.wait(2)
        assert "result" not in done(client, job)
    finally:
        release.set()


def test_applied_revision_cancels_old_scan_and_hides_late_result(client, scanner):
    entered, release, saw_cancel = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def scan(project, raw, *, progress, cancelled, **options):
        entered.set()
        assert release.wait(3)
        if cancelled():
            saw_cancel.set()
        progress({"phase": "complete", "completed": 2, "total": 2})
        return packet()

    scanner.screen_variables = scan
    old = draft(client)
    job = start(client, old)
    try:
        assert entered.wait(2)
        edit = draft(client)
        edit["setup"]["roles"]["predictor"].remove("Region")
        edit["setup"]["roles"]["ignore"].append("Region")
        assert client.post("/api/variables/apply", json=edit).status_code == 200
        assert done(client, job)["status"] == "stale"
        assert client.post("/api/variables/screen", json=old).status_code == 409
        release.set()
        assert saw_cancel.wait(2)
        assert "result" not in done(client, job)
    finally:
        release.set()


def test_replacing_project_invalidates_scan_without_touching_replacement(
    client, scanner, tmp_path
):
    first = done(client, start(client))
    source = tmp_path / "new.csv"
    source.write_text("Loss,Vehicle\n1,A\n0,B\n")
    response = client.post(
        "/api/project/open",
        json=revision(client) | {"kind": "data", "path": str(source)},
    )
    assert response.status_code == 200
    client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
    stale = done(client, first)
    assert stale["status"] == "stale" and "result" not in stale
    assert client.get("/api/variables").json()["columns"][0]["name"] == "Loss"


@pytest.mark.parametrize(
    "options",
    [
        {"sample_rows": 499},
        {"sample_rows": 20_001},
        {"sample_rows": True},
        {"seed": -1},
        {"missing_threshold": 1.1},
        {"correlation_threshold": -0.1},
        {"leakage_threshold": "nan"},
        {"unknown": 1},
    ],
)
def test_invalid_options_are_rejected_before_worker(client, scanner, options):
    calls = []
    scanner.screen_variables = lambda *args, **kwargs: calls.append(1)
    assert (
        client.post(
            "/api/variables/screen", json=draft(client) | {"options": options}
        ).status_code
        == 422
    )
    assert calls == []


def test_invalid_draft_and_stale_session_never_start_worker(client, scanner):
    calls = []
    scanner.screen_variables = lambda *args, **kwargs: calls.append(1)
    body = draft(client)
    body["setup"]["renames"] = {"Age": "Region"}
    assert client.post("/api/variables/screen", json=body).status_code == 422
    body = draft(client) | {"session_id": "old"}
    assert client.post("/api/variables/screen", json=body).status_code == 409
    assert calls == []


def test_worker_failure_is_a_message_and_cache_holds_only_two_completed(scanner):
    project = Project()
    raw = pl.DataFrame({f"x{i}": [1, 2] for i in range(1_000)})
    calls = []

    def scan(saved, snapshot, **kwargs):
        assert snapshot.width == 1_000
        calls.append(kwargs["seed"])
        if kwargs["seed"] == 9:
            raise ValueError("Choose a target first.")
        return packet()

    scanner.screen_variables = scan
    manager = ScreeningJobs()
    try:
        for seed in (1, 2, 3, 9):
            job = manager.start(
                project,
                raw.clone(),
                ("session", "project", 0),
                {},
                ScreeningOptions(seed=seed).model_dump(),
            )
            manager.tasks[job["id"]]["thread"].join(2)
        assert sum(task["status"] == "complete" for task in manager.tasks.values()) == 2
        status = manager.status(job["id"], ("session", "project", 0))
        assert status["status"] == "failed"
        assert status["message"] == "Choose a target first."
        assert calls == [1, 2, 3, 9]
    finally:
        manager.close()


def test_actual_scan_is_training_only_maps_draft_names_and_does_not_fit():
    project = Project(name="Actual screening")
    project.data.split.holdout_value = 0
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "Copy": "predictor",
        "Twin": "predictor",
        "Sparse": "predictor",
        "traintest": "split",
    }
    training = [float(i % 11) for i in range(60)]
    packets = []
    for holdout in (0.0, 1e9):
        raw = pl.DataFrame(
            {
                "Claims": training + [holdout] * 10,
                "Exposure": [1.0] * 70,
                "Copy": training + [-holdout] * 10,
                "Twin": [3 * value + 1 for value in training] + [holdout] * 10,
                "Sparse": [None] * 50 + [1.0] * 20,
                "traintest": [1] * 60 + [0] * 10,
            }
        )
        with TestClient(
            create_app(project, raw, port=8780), base_url="http://127.0.0.1:8780"
        ) as api:
            api.headers["X-EasyGLM-Token"] = api.get("/api/session").json()["token"]
            before = api.get("/api/project").json()
            body = draft(api)
            body["setup"]["renames"] = {"Copy": "Target copy"}
            status = done(api, start(api, body))
            assert status["status"] == "complete", status
            result = status["result"]
            assert result["rows"] == result["training_rows"] == 60
            assert result["predictor_count"] == 3
            assert result["missing"][0]["raw_name"] == "Sparse"
            assert result["missing"][0]["missing_share"] == pytest.approx(50 / 60)
            assert any(
                row["variable"] == "Target copy" and row["raw_name"] == "Copy"
                for row in result["leakage"]
            )
            assert any(
                {row["first_raw"], row["second_raw"]} == {"Copy", "Twin"}
                for row in result["correlated"]
            )
            assert api.get("/api/project").json() == before
            assert api.get("/api/jobs").json() == {}
            packets.append(result)
    assert packets[0] == packets[1]


def test_screen_remove_predictor_then_fit_and_export_use_only_remaining_design(
    tmp_path,
):
    import pickle

    import numpy as np
    from test_desktop_models import wait_fit
    from test_desktop_refit import internals

    from easy_glm.core.fit import TwoStageFit
    from easy_glm.engine import RateModel
    from easy_glm.workflow.prep import prepare

    rng = np.random.default_rng(781)
    count = 900
    age = rng.integers(18, 80, count)
    region = rng.choice(["A", "B", "C"], count)
    exposure = rng.uniform(0.4, 1.5, count)
    claims = rng.poisson(exposure * np.exp(-2 + age / 60 + 0.5 * (region == "B")))
    raw = pl.DataFrame(
        {
            "Claims": claims,
            "Exposure": exposure,
            "Age": age,
            "Region": region,
            "Leak": claims.copy(),
        }
    )
    project = Project(name="Screen and prune")
    project.data.split.mode = "random"
    with TestClient(
        create_app(project, raw, port=8780), base_url="http://127.0.0.1:8780"
    ) as api:
        api.headers["X-EasyGLM-Token"] = api.get("/api/session").json()["token"]
        setup = draft(api)
        setup["setup"]["assignments"].update(target="Claims", weight="Exposure")
        setup["setup"]["roles"]["unassigned"] = []
        setup["setup"]["roles"]["predictor"] = ["Age", "Region", "Leak"]
        response = api.post("/api/variables/apply", json=setup)
        assert response.status_code == 200, response.text
        response = api.post(
            "/api/models/save",
            json={
                **revision(api),
                "name": "Frequency",
                "create": True,
                "n_bins": 4,
                "fields": {
                    "target": "Claims",
                    "weight": "Exposure",
                    "divide_target_by_weight": True,
                    "predictors": ["Age", "Region", "Leak"],
                    "interactions": [
                        {"a": "Leak", "b": "Region"},
                        {"a": "Age", "b": "Region"},
                    ],
                    "penalty": {"alpha": 0.01, "cv": None},
                },
            },
        )
        assert response.status_code == 200, response.text
        before_scan = api.get("/api/project").json()
        scan = done(api, start(api))
        assert scan["status"] == "complete", scan
        assert any(row["raw_name"] == "Leak" for row in scan["result"]["leakage"])
        assert api.get("/api/project").json() == before_scan
        assert api.get("/api/jobs").json() == {}

        removal = draft(api)
        removal["setup"]["roles"]["predictor"].remove("Leak")
        removal["setup"]["roles"]["ignore"].append("Leak")
        assert api.post("/api/variables/preview", json=removal).status_code == 200
        assert api.get("/api/project").json() == before_scan
        applied = api.post("/api/variables/apply", json=removal)
        assert applied.status_code == 200, applied.text
        assert applied.json()["setup"]["roles"]["ignore"] == ["Leak"]
        saved = api.get("/api/project").json()
        config = saved["models"]["Frequency"]
        assert config["predictors"] == ["Age", "Region"]
        assert [(pair["a"], pair["b"]) for pair in config["interactions"]] == [
            ("Age", "Region")
        ]
        assert saved["data"]["roles"]["Leak"] == "ignore"
        assert done(api, scan)["status"] == "stale"

        response = api.post("/api/models/Frequency/fit", json=revision(api))
        assert response.status_code == 202, response.text
        job = wait_fit(api)
        assert job["status"] == "complete", job
        result = api.get("/api/results/Frequency").json()
        assert {item["name"] for item in result["table_index"]} == {
            "Age",
            "Region",
            "Age×Region",
        }
        assert (
            api.get(
                "/api/results/Frequency/table", params={"variable": "Leak"}
            ).status_code
            == 404
        )
        jobs, _, _ = internals(api)
        fit_path = jobs.artifact(Project.from_dict(saved), "Frequency") / "fit.pkl"
        with fit_path.open("rb") as handle:
            run = pickle.load(handle)
        assert isinstance(run.fit, TwoStageFit)
        assert set(run.fit.spec.main_effects) == {"Age", "Region"}
        assert all(
            "Leak" not in (pair.a.variable, pair.b.variable)
            for pair in run.fit.spec.interactions
        )

        exported = api.post(
            "/api/exports/Frequency", json=revision(api) | {"format": "easyglm"}
        )
        assert exported.status_code == 200, exported.text
        path = tmp_path / "pruned.easyglm"
        path.write_bytes(exported.content)
        scorer = RateModel.from_json(path)
        assert set(scorer.variables) == {"Age", "Region", "Age×Region"}
        frame = prepare(Project.from_dict(saved), raw)
        expected = run.rate_model.predict(frame)
        np.testing.assert_allclose(
            scorer.predict(frame.drop("Leak")), expected, rtol=1e-13
        )
        changed = frame.with_columns(pl.lit(1e12).alias("Leak"))
        np.testing.assert_array_equal(scorer.predict(changed), scorer.predict(frame))
        assert api.get("/api/jobs").json()["Frequency"]["id"] == job["id"]
