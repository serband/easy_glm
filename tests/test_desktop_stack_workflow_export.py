"""The stack workflow download is a replayable training artefact, not a scorer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from easy_glm.desktop.server import create_app
from easy_glm.pricing import PricingModel
from easy_glm.workflow.project import Project, safe_filename
from easy_glm.workflow.stack_types import StackScorer


def _revision(client: TestClient) -> dict[str, object]:
    snapshot = client.get("/api/variables").json()
    return {key: snapshot[key] for key in ("session_id", "revision")}


def _wait_for_fit(client: TestClient, name: str) -> dict[str, object]:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        status = client.get("/api/jobs").json()[name]
        if status["status"] not in {"queued", "running"}:
            return status
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for stack fit")


def test_stack_workflow_download_replays_in_fresh_process(tmp_path: Path) -> None:
    """The API's Python workflow refits main, pair, then tree corrections."""
    rng = np.random.default_rng(731)
    rows = 100
    age = rng.integers(18, 70, rows)
    region = rng.choice(["north", "south"], rows)
    vehicle = rng.choice(["A", "B"], rows)
    raw = pl.DataFrame(
        {
            "Claims": rng.poisson(
                np.exp(-1.2 + 0.35 * (age >= 45) + 0.2 * (region == "north"))
            ),
            "Exposure": np.ones(rows),
            "Age": age,
            "Region": region,
            "Vehicle": vehicle,
            "Split": np.where(np.arange(rows) < 80, 1, 0),
        }
    )
    source = tmp_path / "experience.parquet"
    raw.write_parquet(source)
    project = Project(name='Workflow "export"\nname')
    project.data.source.path = str(source)
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "Age": "predictor",
        "Region": "predictor",
        "Vehicle": "predictor",
        "Split": "split",
    }
    project.data.split.column = "Split"
    project.data.split.train_value = 1

    with TestClient(
        create_app(project, raw, port=8864), base_url="http://127.0.0.1:8864"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        saved = client.post(
            "/api/models/save",
            json={
                **_revision(client),
                "name": "Stack replay",
                "create": True,
                "fields": {
                    "family": "poisson",
                    "link": "log",
                    "target": "Claims",
                    "weight": "Exposure",
                    "divide_target_by_weight": True,
                    "components": [
                        {
                            "component_id": "main",
                            "name": "Main effects",
                            "kind": "glm",
                            "factors": ["Age", "Region"],
                            "settings": {"alpha": 0.01},
                        },
                        {
                            "component_id": "pair",
                            "name": "Age by region",
                            "kind": "interaction",
                            "factors": ["Age", "Region"],
                            "settings": {
                                "tune": False,
                                "candidate": {
                                    "iterations": 2,
                                    "depth": 2,
                                    "learning_rate": 0.1,
                                    "l2_leaf_reg": 3.0,
                                },
                            },
                        },
                        {
                            "component_id": "vehicle",
                            "name": "Vehicle correction",
                            "kind": "catboost",
                            "factors": ["Vehicle"],
                            "settings": {
                                "tune": False,
                                "iterations": 2,
                                "depth": 2,
                                "learning_rate": 0.1,
                                "l2_leaf_reg": 3.0,
                            },
                        },
                    ],
                },
            },
        )
        assert saved.status_code == 200, saved.text
        assert (
            client.post(
                "/api/models/Stack replay/fit", json=_revision(client)
            ).status_code
            == 202
        )
        job = _wait_for_fit(client, "Stack replay")
        assert job["status"] == "complete", job

        amendment = {
            **_revision(client),
            "operation": "cap",
            "upper": 1.0,
            "fit_id": job["id"],
        }
        preview = client.post(
            "/api/stack/Stack replay/component/pair/preview", json=amendment
        )
        assert preview.status_code == 200, preview.text
        export_name = "Stack replay amended"
        amended = client.post(
            "/api/stack/Stack replay/component/pair/apply",
            json={
                **amendment,
                "new_name": export_name,
                "refit_later": True,
            },
        )
        assert amended.status_code == 200, amended.text
        amended_job = amended.json()["job"]
        assert amended_job["status"] in {"queued", "running"}, amended_job
        amended_job = _wait_for_fit(client, export_name)
        assert amended_job["status"] == "complete", amended_job

        before_project = client.get("/api/project").json()
        before_jobs = client.get("/api/jobs").json()
        artifact = next(
            Path(tempfile.gettempdir()).glob(
                f"easyglm_fits_*/{amended_job['id']}/fit.pkl"
            )
        )
        before_artifact = artifact.read_bytes()
        frozen = client.post(
            f"/api/exports/{export_name}",
            json={**_revision(client), "format": "easyglm"},
        )
        assert frozen.status_code == 200, frozen.text
        expected = StackScorer.from_dict(frozen.json()).predict_unit(raw)

        downloaded = client.post(
            f"/api/exports/{export_name}",
            json={**_revision(client), "format": "python"},
        )
        assert downloaded.status_code == 200, downloaded.text
        assert "_workflow.py" in downloaded.headers["content-disposition"]
        source_text = downloaded.text
        assert "SOURCE_PATH =" in source_text
        assert "project.data.source.path = SOURCE_PATH" in source_text
        assert "Age by region" in source_text
        assert "Vehicle correction" in source_text
        assert "holdout A/E" not in source_text

        workflow = tmp_path / "workflow.py"
        workflow.write_text(source_text)
        subprocess.run(
            [sys.executable, str(workflow)],
            cwd=tmp_path,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            },
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        replay_path = (
            tmp_path
            / f"{safe_filename(project.name, 'project')}_{safe_filename(export_name)}.easyglm"
        )
        replayed = PricingModel.load(replay_path)
        replay_bundle = json.loads(replay_path.read_text())
        replay_pair = next(
            component
            for component in replay_bundle["project"]["models"][export_name][
                "components"
            ]
            if component["component_id"] == "pair"
        )
        assert replay_pair["amendments"][-1]["operation"] == "cap"
        assert replay_pair["amendments"][-1]["upper"] == 1.0
        # Both accepted and replayed stacks refit the same downstream recipe
        # after the capped pair table, so they retain the same complete scorer.
        np.testing.assert_allclose(replayed.predict(raw), expected, rtol=1e-6)
        assert client.get("/api/project").json() == before_project
        assert client.get("/api/jobs").json() == before_jobs
        assert artifact.read_bytes() == before_artifact
