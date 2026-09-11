"""Export the applied two-stage scorer without changing or refitting the session."""

from __future__ import annotations

import ast
import io
import pickle
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from test_desktop_models import revision
from test_desktop_refit import internals
from test_desktop_reviews import apply, review

from easy_glm.core.fit import TwoStageFit
from easy_glm.desktop import server
from easy_glm.engine import RateModel
from easy_glm.workflow.prep import prepare
from easy_glm.workflow.project import Interaction, ModelConfig, Project
from easy_glm.workflow.run import rebuild_rate_model


def wait_model(client, name):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get("/api/jobs").json()[name]
        if job["status"] not in ("queued", "running"):
            assert job["status"] == "complete", job
            return
        time.sleep(0.03)
    pytest.fail("Fit did not complete")


@pytest.fixture(scope="module")
def export_session(tmp_path_factory):
    folder = tmp_path_factory.mktemp("desktop_exports")
    rng = np.random.default_rng(917)
    n = 700
    age = rng.integers(18, 80, n)
    region = rng.choice(["A", "B", "C"], n)
    exposure = rng.uniform(0.2, 1.8, n)
    mean = exposure * np.exp(-1 + age / 100 + (region == "B") * (age > 45))
    raw = pl.DataFrame(
        {
            "Claims": rng.poisson(mean),
            "Exposure": exposure,
            "Age": age,
            "Region": region,
        }
    )
    source = folder / "source.parquet"
    raw.write_parquet(source)
    project = Project(name='Report / "North"\nRésumé: 2026')
    project.data.source.path = str(source)
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "Age": "predictor",
        "Region": "predictor",
    }
    project.data.split.mode = "random"
    project.design.defaults.n_bins = 4
    for name in ("Frequency", "Challenger", "Different basis", "Unfitted"):
        cfg = ModelConfig(
            target="Claims",
            weight="Exposure",
            divide_target_by_weight=name != "Different basis",
            predictors=["Age", "Region"],
            interactions=[Interaction("Age", "Region", min_cell_exposure=0.01)],
        )
        cfg.penalty.alpha = 0.01
        cfg.penalty.cv = None
        project.models[name] = cfg
    with TestClient(
        server.create_app(project, raw, port=8780), base_url="http://127.0.0.1:8780"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        for name in ("Frequency", "Challenger", "Different basis"):
            response = client.post(f"/api/models/{name}/fit", json=revision(client))
            assert response.status_code == 202, response.text
            wait_model(client, name)
        apply(client, review(client, "edit", variable="Age", edits={"1": 2.3456}))
        table = client.get(
            "/api/results/Frequency/table",
            params={"variable": "Age×Region", "limit": 500},
        ).json()
        cell = next(i for i, row in enumerate(table["rows"]) if row["exposure"] > 0)
        apply(
            client,
            review(client, "edit", variable="Age×Region", edits={str(cell): 1.6789}),
        )
        apply(client, review(client, "rebalance", variable="Age"))
        review(client, "snapshot", snapshot="Applied rates")
        # An outstanding preview must never enter any downloaded artifact.
        draft = review(client, "cap", variable="Age", options={"cap": 0.1})
        assert draft["can_apply"]
        yield client, raw, folder, cell


def download(client, format, name="Frequency", **kwargs):
    return client.post(
        f"/api/exports/{name}", json={**revision(client), "format": format, **kwargs}
    )


def fitted_source(client, name="Frequency"):
    jobs, _, _ = internals(client)
    job = client.get("/api/jobs").json()[name]
    return Path(jobs.folder.name) / job["id"] / "fit.pkl"


def test_scorer_excel_script_and_report_include_applied_state_only(
    export_session, monkeypatch
):
    client, raw, folder, cell = export_session
    project_before = client.get("/api/project").json()
    rev_before = revision(client)
    jobs_before = client.get("/api/jobs").json()
    _, undo, redo = internals(client)
    history_before = deepcopy((undo, redo))
    source = fitted_source(client)
    frozen = source.read_bytes()
    run = pickle.loads(frozen)
    assert isinstance(run.fit, TwoStageFit)
    assert not run.config.adjustments
    project = Project.from_dict(project_before)
    frame = prepare(project, raw)
    expected = rebuild_rate_model(project, run, frame).rate_model
    assert expected.variables["Age"].table[1].relativity == 2.3456
    assert expected.variables["Age×Region"].table[cell].relativity == 1.6789
    assert project.models["Frequency"].base_rate_override is not None

    def no_fit(*args, **kwargs):
        raise AssertionError("Export must not fit a model")

    monkeypatch.setattr("easy_glm.workflow.run.run_model", no_fit)
    monkeypatch.setattr("easy_glm.core.fit.fit_glm", no_fit)
    monkeypatch.setattr("easy_glm.core.fit.fit_two_stage", no_fit)
    monkeypatch.setattr(type(internals(client)[0]), "start", no_fit)
    scorer = download(client, "easyglm")
    assert scorer.status_code == 200, scorer.text
    assert scorer.headers["content-type"] == "application/json"
    assert scorer.headers["cache-control"] == "no-store"
    disposition = scorer.headers["content-disposition"]
    assert (
        "\n" not in disposition and "\r" not in disposition and "/" not in disposition
    )
    assert "filename*=UTF-8''" in disposition and "%C3%A9" in disposition
    saved_path = folder / "download.easyglm"
    saved_path.write_bytes(scorer.content)
    saved = RateModel.from_json(saved_path)
    np.testing.assert_allclose(
        saved.predict(frame), expected.predict(frame), rtol=1e-13
    )
    assert saved.base_rate == expected.base_rate
    assert saved.variables["Age"].table[1].relativity == 2.3456
    assert saved.variables["Age×Region"].table[cell].relativity == 1.6789
    train = frame.filter(
        pl.col(project.data.split.column) == project.data.split.train_value
    )
    assert saved.predict(train).sum() == pytest.approx(
        client.get("/api/results/Frequency").json()["metrics"]["train"]["expected"]
    )

    excel = download(client, "xlsx")
    assert excel.status_code == 200, excel.text
    assert "spreadsheetml.sheet" in excel.headers["content-type"]
    sheets = pl.read_excel(io.BytesIO(excel.content), sheet_id=0)
    assert sheets["Age"]["relativity"][1] == 2.3456
    assert sheets["Age×Region"]["relativity"][cell] == 1.6789
    assert "Age×Region (matrix)" in sheets
    summary = pl.read_excel(
        io.BytesIO(excel.content), sheet_name="Summary", has_header=False
    )
    base = next(row[1] for row in summary.iter_rows() if row[0] == "base_rate")
    assert float(base) == pytest.approx(expected.base_rate)

    script = download(client, "python")
    assert script.status_code == 200, script.text
    ast.parse(script.text)
    assert str(project.data.source.path) in script.text
    assert "2.3456" in script.text and "1.6789" in script.text
    assert repr(expected.base_rate) in script.text
    assert "0.1, cell=" not in script.text
    script_path = folder / "reproduce.py"
    script_path.write_text(script.text)
    process = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=folder,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr[-2000:]
    reproduced = next(path for path in folder.glob("*.easyglm") if path != saved_path)
    np.testing.assert_allclose(
        RateModel.from_json(reproduced).predict(frame),
        expected.predict(frame),
        rtol=1e-10,
        atol=1e-12,
    )

    report = download(client, "html", challenger="Challenger")
    assert report.status_code == 200, report.text[:1000]
    assert report.headers["content-type"].startswith("text/html")
    assert "Challenger" in report.text and "Frequency" in report.text
    assert "Double lift" in report.text
    assert "<h2>2. Data summary</h2>" in report.text
    assert "Age: training distribution" in report.text
    assert "Excess kurtosis" in report.text
    assert "2.3456" in report.text
    assert client.get("/api/project").json() == project_before
    assert revision(client) == rev_before
    assert client.get("/api/jobs").json() == jobs_before
    assert (undo, redo) == history_before
    assert source.read_bytes() == frozen


@pytest.mark.parametrize(
    "name,payload,status",
    [
        ("Missing", {"format": "xlsx"}, 404),
        ("Unfitted", {"format": "xlsx"}, 409),
        ("Frequency", {"format": "csv"}, 422),
        ("Frequency", {"format": "html", "challenger": "Missing"}, 404),
        ("Frequency", {"format": "html", "challenger": "Frequency"}, 422),
        ("Frequency", {"format": "xlsx", "challenger": "Challenger"}, 422),
        ("Frequency", {"format": "html", "challenger": "Unfitted"}, 409),
        ("Frequency", {"format": "html", "challenger": "Different basis"}, 422),
        ("Frequency", {"format": "xlsx", "session_id": "wrong"}, 409),
        ("Frequency", {"format": "xlsx", "revision": 0}, 409),
    ],
)
def test_export_invalid_or_inapplicable_requests_are_json_errors(
    export_session, name, payload, status
):
    client, _, _, _ = export_session
    before = client.get("/api/project").json()
    response = client.post(f"/api/exports/{name}", json={**revision(client), **payload})
    assert response.status_code == status, response.text
    assert response.json()["detail"]
    assert client.get("/api/project").json() == before


def test_export_generation_releases_project_lock(export_session, monkeypatch):
    client, _, _, _ = export_session
    started, release = threading.Event(), threading.Event()
    real_export = server.export_attachment
    snapshot = {}

    def held_export(project, *args, **kwargs):
        snapshot["project"] = project
        started.set()
        assert release.wait(10)
        return real_export(project, *args, **kwargs)

    monkeypatch.setattr(server, "export_attachment", held_export)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(download, client, "easyglm")
        try:
            assert started.wait(5)
            response = pool.submit(client.get, "/api/project").result(timeout=2)
            assert response.status_code == 200
            assert snapshot["project"].to_dict() == response.json()
        finally:
            release.set()
        assert future.result(timeout=10).status_code == 200


def test_changed_model_cannot_export_old_fit(export_session):
    client, _, _, _ = export_session
    response = client.post(
        "/api/models/save",
        json={
            **revision(client),
            "name": "Different basis",
            "fields": {"penalty": {"alpha": 0.02}},
        },
    )
    assert response.status_code == 200, response.text
    response = download(client, "easyglm", name="Different basis")
    assert response.status_code == 409


def test_pathless_script_reports_error_but_html_remains_available(
    export_session, monkeypatch
):
    client, _, _, _ = export_session
    before = client.get("/api/project").json()
    real_export = server.export_attachment

    def pathless_export(project, *args, **kwargs):
        # Simulate the in-memory starter dataset on the private export snapshot.
        project.data.source.path = ""
        return real_export(project, *args, **kwargs)

    monkeypatch.setattr(server, "export_attachment", pathless_export)
    response = download(client, "python")
    assert response.status_code == 422
    assert "source data" in response.json()["detail"]
    report = download(client, "html")
    assert report.status_code == 200, report.text[:500]
    assert "The script could not be rendered" in report.text
    assert "Save the source data" in report.text
    assert client.get("/api/project").json() == before
