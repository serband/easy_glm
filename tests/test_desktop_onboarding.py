"""Upload/example onboarding uses real parsers and preserves sessions on failure."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import polars as pl
import pytest
from desktop_onboarding_fixtures import example_frames
from fastapi.testclient import TestClient
from test_desktop_loading import token
from test_desktop_models import model_session as model_session
from test_desktop_models import revision, save_model
from test_desktop_refit import internals
from test_desktop_reviews import apply, fitted, review

import easy_glm
from easy_glm.desktop import server
from easy_glm.desktop.loading import load_project_input, upload_basename
from easy_glm.workflow import prepare
from easy_glm.workflow.project import Project


@pytest.fixture
def input_folders(monkeypatch, tmp_path):
    folders = []

    def new_folder():
        folder = tmp_path / f"input-{len(folders)}"
        folder.mkdir()
        folders.append(folder)
        return folder

    monkeypatch.setattr(server, "new_input_folder", new_folder)
    return folders


@pytest.fixture
def empty_session(input_folders):
    with TestClient(
        server.create_app(Project(), pl.DataFrame(), port=8780),
        base_url="http://127.0.0.1:8780",
    ) as client:
        token(client)
        yield client


def upload(client, content, *, kind="data", filename="book.csv", **options):
    params = {**revision(client), "kind": kind, "filename": filename, **options}
    return client.post(
        "/api/project/upload",
        params=params,
        headers={"content-type": "application/octet-stream"},
        content=content,
    )


@pytest.mark.parametrize("source_type", ["csv", "parquet", "excel", "ipc"])
def test_upload_real_formats_retains_source_for_project_reopen(
    empty_session, input_folders, source_type, tmp_path
):
    client = empty_session
    raw = pl.DataFrame({"claim": [0, 1, 2], "traintest": [8, 8, 9]})
    source = tmp_path / "original.data"
    writer = "excel" if source_type == "excel" else source_type
    getattr(raw, f"write_{writer}")(source)
    original = source.read_bytes()
    old = revision(client)
    response = upload(
        client,
        original,
        filename=r"C:\fakepath\portfolio.data",
        source_type=source_type,
    )
    assert response.status_code == 200, response.text
    assert response.json()["session_id"] != old["session_id"]
    assert response.json()["row_count"] == 3
    assert client.get("/api/project").status_code == 401
    token(client)
    project = Project.from_dict(client.get("/api/project").json())
    assert not project.models and not project.data.roles
    assert project.data.split.column == "traintest_2"
    retained = Path(project.data.source.path)
    assert retained == input_folders[0] / "portfolio.data"
    assert retained.read_bytes() == original and source.read_bytes() == original
    saved = tmp_path / "exported-project.json"
    project.to_json(saved)
    reopened, frame = load_project_input("project", saved)
    assert frame.equals(raw)
    assert reopened.data.source.path == str(retained)
    assert client.get("/api/jobs").json() == {}


def test_local_explicit_type_and_sas_dispatch(tmp_path, monkeypatch):
    path = tmp_path / "unusual.extension"
    raw = pl.DataFrame({"claim": [0, 1], "age": [20, 50]})
    raw.write_csv(path)
    project, parsed = load_project_input("data", path, "csv")
    assert parsed.equals(raw) and project.data.source.type == "csv"
    with pytest.raises(ValueError, match="Choose a CSV"):
        load_project_input("data", path)
    import pandas as pd

    seen = {}

    def read_sas(source, **kwargs):
        seen.update(path=source, **kwargs)
        return raw.to_pandas()

    monkeypatch.setattr(pd, "read_sas", read_sas)
    project, parsed = load_project_input("data", path, "sas7bdat")
    assert parsed.equals(raw) and project.data.source.type == "sas7bdat"
    assert seen == {"path": path, "encoding": "latin-1"}


def test_upload_project_resolves_absolute_source_without_applying_adjustments(
    empty_session, input_folders, tmp_path, monkeypatch
):
    raw = pl.DataFrame({"claims": [0, 1], "age": [20, 50]})
    source = tmp_path / "source.parquet"
    raw.write_parquet(source)
    project = Project(name="Imported project")
    project.data.source.path = str(source)
    project.data.split.mode = "random"
    project.data.roles = {"claims": "target", "age": "predictor"}
    project.new_model("saved")
    from easy_glm.workflow import run

    def no_fit_or_adjust(*args, **kwargs):
        pytest.fail("Loading a project must not fit or apply adjustments")

    monkeypatch.setattr(run, "run_model", no_fit_or_adjust)
    monkeypatch.setattr(run, "apply_adjustments", no_fit_or_adjust)
    response = upload(
        empty_session,
        json.dumps(project.to_dict()).encode(),
        kind="project",
        filename="project.easyglm-project.json",
    )
    assert response.status_code == 200, response.text
    token(empty_session)
    assert empty_session.get("/api/project").json() == project.to_dict()
    assert empty_session.get("/api/jobs").json() == {}
    assert input_folders[0].is_dir()


@pytest.mark.parametrize("example", ["french_motor", "swedish_motorcycle"])
def test_examples_preserve_roles_clear_models_and_never_fit(
    model_session, input_folders, monkeypatch, example
):
    client, _, _ = model_session
    save_model(client)
    frames = example_frames()
    monkeypatch.setattr(
        easy_glm, "load_external_dataframe", lambda: frames["french_motor"]
    )
    monkeypatch.setattr(
        easy_glm, "load_swedish_motorcycle_data", lambda: frames["swedish_motorcycle"]
    )
    from easy_glm.workflow import run

    monkeypatch.setattr(
        run, "run_model", lambda *a, **kw: pytest.fail("Examples must not fit on load")
    )
    response = client.post(
        "/api/project/open",
        json={**revision(client), "kind": "example", "example": example},
    )
    assert response.status_code == 200, response.text
    token(client)
    project = Project.from_dict(client.get("/api/project").json())
    assert "Claims" not in project.data.roles and "Frequency" not in project.models
    assert project.data.split.mode == "random"
    assert project.data.split.fraction == 0.7 and project.data.split.seed == 42
    raw = pl.read_parquet(project.data.source.path)
    assert raw.height == 240
    assert client.get("/api/jobs").json() == {}
    assert project.models == {}
    assert project.champion is None
    assert project.data.roles["SyntheticYear"] == "time"
    assert raw["SyntheticYear"].n_unique() == 5
    assert project.data.roles["Exposure"] == "weight"
    if example == "french_motor":
        assert project.data.roles["ClaimNb"] == "target"
        assert project.data.roles["IDpol"] == "id"
    else:
        assert project.data.roles["ClaimAmount"] == "target"
        assert project.data.roles["ClaimNb"] == "ignore"
        prepared = prepare(project, raw)
        assert prepared.height == 238 and (prepared["Exposure"] > 0).all()
        assert (prepared["ClaimAmount"] == 0).any()


def test_french_example_caps_the_saved_sample_at_50000(
    empty_session, input_folders, monkeypatch
):
    raw = example_frames()["french_motor"]
    raw = pl.concat([raw] * 210)
    monkeypatch.setattr(easy_glm, "load_external_dataframe", lambda: raw)
    response = empty_session.post(
        "/api/project/open",
        json={**revision(empty_session), "kind": "example", "example": "french_motor"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["row_count"] == 50_000
    assert (
        pl.read_parquet(input_folders[0] / "french_motor_sample.parquet")
        .drop("SyntheticYear")
        .equals(raw.sample(n=50_000, seed=42))
    )


def test_invalid_uploads_examples_and_stale_requests_keep_fit_raw_and_history(
    model_session, input_folders, monkeypatch
):
    client, _, _ = model_session
    fitted(client)
    apply(client, review(client, "edit", variable="Age", edits={"1": 2.1}))
    before = client.get("/api/project").json()
    snapshot = client.get("/api/variables").json()
    job_state = client.get("/api/jobs").json()
    _, undo, redo = internals(client)
    history = deepcopy((undo, redo))
    relative = Project()
    relative.data.source.path = "relative.parquet"
    cases = [
        (b"not JSON", "project", "project.json", None),
        (b"not parquet", "data", "data.parquet", None),
        (b"", "data", "empty.csv", None),
        (
            json.dumps(relative.to_dict()).encode(),
            "project",
            "project.json",
            "local file path",
        ),
    ]
    for content, kind, filename, message in cases:
        response = upload(client, content, kind=kind, filename=filename)
        assert response.status_code == 422, response.text
        if message:
            assert message in response.json()["detail"]

    def failed_network():
        raise OSError("Example download unavailable")

    monkeypatch.setattr(easy_glm, "load_external_dataframe", failed_network)
    response = client.post(
        "/api/project/open",
        json={**revision(client), "kind": "example", "example": "french_motor"},
    )
    assert (
        response.status_code == 422 and "Example download unavailable" in response.text
    )
    response = upload(client, b"x\n1\n", session_id="old-session")
    assert response.status_code == 409
    assert client.get("/api/project").json() == before
    assert client.get("/api/variables").json() == snapshot
    assert client.get("/api/jobs").json() == job_state
    assert (undo, redo) == history
    assert all(not folder.exists() for folder in input_folders)


def test_upload_rechecks_revision_after_parsing_and_cleans_stale_source(
    model_session, input_folders, monkeypatch
):
    client, _, _ = model_session
    started, release = threading.Event(), threading.Event()
    real_load = server.load_project_input

    def held_load(*args, **kwargs):
        loaded = real_load(*args, **kwargs)
        started.set()
        assert release.wait(10)
        return loaded

    monkeypatch.setattr(server, "load_project_input", held_load)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(upload, client, b"x\n1\n2\n")
        try:
            assert started.wait(5)
            assert client.get("/health").status_code == 200
            save_model(client)
            before = client.get("/api/project").json()
        finally:
            release.set()
        response = future.result(timeout=5)
    assert response.status_code == 409, response.text
    assert client.get("/api/project").json() == before
    assert all(not folder.exists() for folder in input_folders)


def test_stream_limit_and_loopback_protections(
    empty_session, input_folders, monkeypatch
):
    client = empty_session
    before = client.get("/api/variables").json()
    params = {**revision(client), "kind": "data", "filename": "book.csv"}
    for headers, status in [
        ({"origin": "https://other.example"}, 403),
        ({"host": "other.example"}, 403),
        ({"x-easyglm-token": "wrong"}, 401),
        ({"content-type": "application/json"}, 415),
    ]:
        response = client.post(
            "/api/project/upload",
            params=params,
            content=b"x\n1\n",
            headers={"content-type": "application/octet-stream", **headers},
        )
        assert response.status_code == status
    assert not input_folders
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 16)
    # A generator has no Content-Length; the actual stream must still be bounded.
    response = upload(client, iter([b"x\n", b"1\n" * 20]))
    assert response.status_code == 413
    assert all(not folder.exists() for folder in input_folders)
    assert client.get("/api/variables").json() == before


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("../../book.csv", "book.csv"),
        (r"C:\fakepath\book.csv", "book.csv"),
        ("CON.csv", "_CON.csv"),
    ],
)
def test_upload_names_cannot_escape_owned_folder(filename, expected):
    assert upload_basename(filename) == expected
