"""Contract checks for the local experiment, including existing model semantics."""

from __future__ import annotations

import json
from copy import deepcopy

import polars as pl
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import Interaction, ModelConfig, Project


@pytest.fixture
def session():
    project = Project(name="API test")
    project.data.roles = {
        "claims": "target",
        "age": "predictor",
        "area": "predictor",
        "id": "ignore",
    }
    project.models["Frequency"] = ModelConfig(
        target="claims",
        predictors=["age", "area"],
        interactions=[Interaction(a="age", b="area")],
    )
    raw = pl.DataFrame(
        {
            "claims": [0, 1, 0],
            "age": [20, 40, 60],
            "area": ["A", "B", "A"],
            "id": [1, 2, 3],
            "spare": [4, 5, 6],
        }
    )
    with TestClient(
        create_app(project, raw, port=8765), base_url="http://127.0.0.1:8765"
    ) as client:
        token = client.get("/api/session").json()["token"]
        client.headers["x-easyglm-token"] = token
        yield client, project


def test_role_snapshot_and_roundtrip(session):
    client, project = session
    state = client.get("/api/variables").json()
    assert len(state["setup"]["assignments"]) == 6
    assert state["setup"]["assignments"]["weight"] is None
    assert state["setup"]["roles"]["ignore"] == ["id"]
    assert state["setup"]["roles"]["unassigned"] == ["spare"]
    before = deepcopy(project.to_dict())
    response = client.post(
        "/api/variables/apply", json={"revision": 0, "setup": state["setup"]}
    )
    assert response.status_code == 200
    assert response.json()["revision"] == 0
    assert project.to_dict() == before


def test_atomic_preview_apply_rename_and_role_model_sync(session):
    client, original = session
    setup = client.get("/api/variables").json()["setup"]
    setup["renames"] = {"age": "area", "area": "age"}
    setup["roles"]["predictor"] = ["area"]
    setup["roles"]["ignore"].append("age")
    setup["types"] = {"categorical": ["area"]}
    body = {"revision": 0, "setup": setup}
    preview = client.post("/api/variables/preview", json=body)
    assert preview.status_code == 200
    assert preview.json()["notices"]
    assert client.get("/api/project").json() == original.to_dict()
    applied = client.post("/api/variables/apply", json=body)
    assert applied.status_code == 200
    assert applied.json()["revision"] == 1
    saved = client.get("/api/project").json()
    assert saved["models"]["Frequency"]["predictors"] == ["age"]
    assert saved["models"]["Frequency"]["interactions"] == []
    assert saved["data"]["roles"]["area"] == "ignore"
    assert saved["data"]["types"]["age"] == "categorical"
    assert original.models["Frequency"].predictors == ["age", "area"]
    assert client.post("/api/variables/apply", json=body).status_code == 409


@pytest.mark.parametrize("invalid", ["collision", "duplicate", "unknown", "bad_type"])
def test_invalid_drafts_leave_project_untouched(session, invalid):
    client, original = session
    setup = client.get("/api/variables").json()["setup"]
    if invalid == "collision":
        setup["renames"] = {"age": "area"}
    elif invalid == "duplicate":
        setup["roles"]["ignore"].append("age")
    elif invalid == "unknown":
        setup["assignments"]["target"] = "missing"
    else:
        setup["types"] = {"date": ["age"]}
    response = client.post("/api/variables/apply", json={"revision": 0, "setup": setup})
    assert response.status_code == 422
    assert client.get("/api/project").json() == original.to_dict()


def test_plot_and_no_fit_on_edit(session, monkeypatch):
    from easy_glm.workflow import run

    def no_fit(*args, **kwargs):
        pytest.fail("Variable edits must never fit a model")

    monkeypatch.setattr(run, "run_model", no_fit)
    client, _ = session
    setup = client.get("/api/variables").json()["setup"]
    setup["renames"] = {"age": "DriverAge"}
    assert (
        client.post(
            "/api/variables/apply", json={"revision": 0, "setup": setup}
        ).status_code
        == 200
    )
    plot = client.get("/api/plot?column=age").json()
    assert plot["column"] == "DriverAge"
    assert sum(r["exposure"] for r in plot["table"]) == 3
    assert client.get("/api/plot?column=missing").status_code == 404


def test_loopback_origin_token_and_static_assets(session):
    client, _ = session
    assert (
        client.get("/api/variables", headers={"host": "attacker.example"}).status_code
        == 403
    )
    assert (
        client.get(
            "/api/session", headers={"origin": "https://attacker.example"}
        ).status_code
        == 403
    )
    assert (
        client.get("/api/session", headers={"sec-fetch-site": "cross-site"}).status_code
        == 403
    )
    assert (
        client.get("/api/variables", headers={"x-easyglm-token": "wrong"}).status_code
        == 403
    )
    assert (
        client.post(
            "/api/variables/apply", content="{}", headers={"content-type": "text/plain"}
        ).status_code
        == 415
    )
    assert client.get("/").status_code == 200
    assert (
        "frame-ancestors 'none'" in client.get("/").headers["content-security-policy"]
    )
    assert (
        client.post(
            "/api/variables/apply",
            content='"' + "x" * 8_000_000 + '"',
            headers={"content-type": "application/json"},
        ).status_code
        == 413
    )


def test_wide_schema_does_not_scan_each_column(monkeypatch):
    raw = pl.DataFrame({f"x{i}": [i, i + 1] for i in range(2000)})

    def no_scan(*args, **kwargs):
        pytest.fail("Schema load scanned column values")

    monkeypatch.setattr(pl.Series, "n_unique", no_scan)
    with TestClient(
        create_app(Project(), raw, port=8765), base_url="http://127.0.0.1:8765"
    ) as client:
        client.headers["x-easyglm-token"] = client.get("/api/session").json()["token"]
        result = client.get("/api/variables")
        assert len(result.json()["columns"]) == 2000
        assert len(json.dumps(result.json())) < 200_000


def test_explicit_browser_navigation_is_allowed_but_cross_site_api_is_not(session):
    client, _ = session
    headers = {
        "sec-fetch-site": "cross-site",
        "sec-fetch-mode": "navigate",
        "sec-fetch-dest": "document",
    }
    assert client.get("/", headers=headers).status_code == 200
    assert client.get("/api/session", headers=headers).status_code == 403


def test_slow_plot_does_not_block_health_or_variables(session, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from easy_glm.desktop import server

    client, _ = session
    entered, release = threading.Event(), threading.Event()
    original = server.univariate

    def slow_plot(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(server, "univariate", slow_plot)
    with ThreadPoolExecutor() as worker:
        pending = worker.submit(client.get, "/api/plot?column=age")
        assert entered.wait(5)
        try:
            assert client.get("/health").status_code == 200
            assert client.get("/api/variables").status_code == 200
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200
