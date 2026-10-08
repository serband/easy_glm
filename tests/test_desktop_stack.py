"""The desktop uses the cumulative accepted scorer for mixed-model views."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import Project


def _revision(client: TestClient) -> dict[str, object]:
    snapshot = client.get("/api/variables").json()
    return {key: snapshot[key] for key in ("session_id", "revision")}


def test_stack_fit_diagnostics_and_catalogue_upload(tmp_path):
    rng = np.random.default_rng(882)
    raw = pl.DataFrame(
        {
            "Claims": rng.poisson(0.25, 500),
            "Exposure": np.ones(500),
            "Age": rng.integers(18, 80, 500),
            "Region": rng.choice(["A", "B"], 500),
        }
    )
    project = Project(name="Desktop stack")
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "Age": "predictor",
        "Region": "predictor",
    }
    project.data.split.mode = "random"
    with TestClient(
        create_app(project, raw, port=8798), base_url="http://127.0.0.1:8798"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        catalogue = tmp_path / "vehicle.csv"
        catalogue.write_text("Code,Make\n001,A\n002,B\n")
        uploaded = client.post(
            "/api/catalogues/upload",
            params={"filename": catalogue.name, **_revision(client)},
            headers={"Content-Type": "application/octet-stream"},
            content=catalogue.read_bytes(),
        )
        assert uploaded.status_code == 200, uploaded.text
        assert uploaded.json()["rows"] == 2
        assert pl.read_csv(uploaded.json()["source_path"], infer_schema_length=0)[
            "Code"
        ].to_list() == ["001", "002"]
        saved = client.post(
            "/api/models/save",
            json={
                **_revision(client),
                "name": "Frequency",
                "create": True,
                "fields": {
                    "family": "poisson",
                    "link": "log",
                    "target": "Claims",
                    "weight": "Exposure",
                    "components": [
                        {
                            "component_id": "main",
                            "name": "Main GLM",
                            "kind": "glm",
                            "factors": ["Age", "Region"],
                            "settings": {"alpha": 0.01},
                        }
                    ],
                },
            },
        )
        assert saved.status_code == 200, saved.text
        started = client.post("/api/models/Frequency/fit", json=_revision(client))
        assert started.status_code == 202, started.text
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            job = client.get("/api/jobs").json()["Frequency"]
            if job["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert job["status"] == "complete", job
        result = client.get("/api/results/Frequency").json()
        assert result["stack"] is True
        assert set(result["metrics"]) == {"train"}
        assert result["components"][0]["name"] == "Main GLM"
        diagnostics = client.get(
            "/api/stack/Frequency/diagnostics", params={"variable": "Age"}
        )
        assert diagnostics.status_code == 200, diagnostics.text
        assert diagnostics.json()["ae"]
        holdout = client.get(
            "/api/stack/Frequency/diagnostics", params={"subset": "holdout"}
        )
        assert holdout.status_code == 200, holdout.text
        table = client.get("/api/stack/Frequency/component/main/table")
        assert table.status_code == 200, table.text
        assert table.json()["tables"]
        bundle = client.post(
            "/api/exports/Frequency",
            json={**_revision(client), "format": "easyglm"},
        )
        assert bundle.status_code == 200, bundle.text
        assert bundle.json()["format_version"] == 1


def test_catalogue_review_creates_named_checkpoint(tmp_path):
    rng = np.random.default_rng(94)
    keys = rng.choice(["001", "002", "003"], 360)
    raw = pl.DataFrame(
        {
            "Claims": rng.poisson(np.where(keys == "003", 0.5, 0.2)),
            "Exposure": np.ones(360),
            "VehicleCode": keys,
            "Age": rng.integers(18, 75, 360),
        }
    )
    project = Project(name="Catalogue review")
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "VehicleCode": "predictor",
        "Age": "predictor",
    }
    project.data.split.mode = "random"
    with TestClient(
        create_app(project, raw, port=8799), base_url="http://127.0.0.1:8799"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        uploaded = client.post(
            "/api/catalogues/upload",
            params={"filename": "vehicles.csv", **_revision(client)},
            headers={"Content-Type": "application/octet-stream"},
            content=b"VehicleCode,Make\n001,A\n002,B\n003,C\n004,D\n",
        )
        assert uploaded.status_code == 200, uploaded.text
        location = uploaded.json()
        saved = client.post(
            "/api/models/save",
            json={
                **_revision(client),
                "name": "Vehicle",
                "create": True,
                "fields": {
                    "family": "poisson",
                    "link": "log",
                    "target": "Claims",
                    "weight": "Exposure",
                    "components": [
                        {
                            "component_id": "main",
                            "name": "Main GLM",
                            "kind": "glm",
                            "factors": ["Age"],
                            "settings": {"alpha": 0.01},
                        },
                        {
                            "component_id": "vehicle",
                            "name": "Vehicle file",
                            "kind": "catalogue",
                            "factors": ["Make"],
                            "settings": {
                                "key": "VehicleCode",
                                "method": "catboost",
                                "source_path": location["source_path"],
                                "source_sha256": location["source_sha256"],
                                "iterations": 15,
                                "depth": 2,
                            },
                        },
                        {
                            "component_id": "age",
                            "name": "Age correction",
                            "kind": "glm",
                            "factors": ["Age"],
                            "settings": {"alpha": 0.01},
                        },
                    ],
                },
            },
        )
        assert saved.status_code == 200, saved.text
        assert (
            client.post("/api/models/Vehicle/fit", json=_revision(client)).status_code
            == 202
        )
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            job = client.get("/api/jobs").json()["Vehicle"]
            if job["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert job["status"] == "complete", job
        table = client.get("/api/stack/Vehicle/component/vehicle/table")
        assert table.status_code == 200, table.text
        assert len(table.json()["tables"][0]["rows"]) == 4
        assert table.json()["unit_label"] == "Incremental multiplier"
        assert table.json()["coverage"]["fallback_rows"] == 0
        assert "unseen_attribute_levels" in table.json()["evidence"]
        edit = {
            **_revision(client),
            "operation": "cap",
            "upper": 0.5,
            "fit_id": job["id"],
        }
        preview = client.post("/api/stack/Vehicle/component/vehicle/preview", json=edit)
        assert preview.status_code == 200, preview.text
        applied = client.post(
            "/api/stack/Vehicle/component/vehicle/apply",
            json={**edit, "new_name": "Vehicle reviewed"},
        )
        assert applied.status_code == 200, applied.text
        assert applied.json()["job"]["status"] == "complete"
        assert client.get("/api/results/Vehicle").json()["stack"] is True
        reviewed = client.get("/api/results/Vehicle reviewed").json()
        assert reviewed["stack"] is True
        csv = client.get("/api/stack/Vehicle reviewed/export/vehicle")
        assert csv.status_code == 200, csv.text
        assert "004" in csv.text
        refit = client.post(
            "/api/stack/Vehicle/component/vehicle/apply",
            json={
                **edit,
                **_revision(client),
                "new_name": "Vehicle refitted",
                "refit_later": True,
            },
        )
        assert refit.status_code == 200, refit.text
        assert refit.json()["job"]["status"] in {"queued", "running"}
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            job = client.get("/api/jobs").json()["Vehicle refitted"]
            if job["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert job["status"] == "complete", job
        original_bundle = client.post(
            "/api/exports/Vehicle",
            json={**_revision(client), "format": "easyglm"},
        ).json()
        refitted_bundle = client.post(
            "/api/exports/Vehicle refitted",
            json={**_revision(client), "format": "easyglm"},
        ).json()
        assert original_bundle["components"][0] == refitted_bundle["components"][0]
        assert (
            original_bundle["components"][1]["scorer"]
            != refitted_bundle["components"][1]["scorer"]
        )
        assert (
            original_bundle["components"][2]["prefix_fingerprint"]
            != refitted_bundle["components"][2]["prefix_fingerprint"]
        )
        workbook_response = client.post(
            "/api/exports/Vehicle refitted",
            json={**_revision(client), "format": "xlsx"},
        )
        assert workbook_response.status_code == 200, workbook_response.text
        with zipfile.ZipFile(BytesIO(workbook_response.content)) as workbook:
            manifest = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
            sheets = [
                node.attrib["name"]
                for node in manifest.iter()
                if node.tag.endswith("sheet")
            ]
        assert any("Vehicle file" in sheet for sheet in sheets)
        assert any("Main GLM" in sheet for sheet in sheets)
        assert "Components" in sheets
        lookup_sheet = next(sheet for sheet in sheets if "Vehicle file" in sheet)
        lookup_frame = pl.read_excel(
            BytesIO(workbook_response.content), sheet_name=lookup_sheet
        )
        accepted_lookup = refitted_bundle["components"][1]["scorer"]["table"]
        assert (
            lookup_frame["VehicleCode"].cast(pl.String).to_list()
            == accepted_lookup["VehicleCode"]
        )
        np.testing.assert_allclose(
            lookup_frame["multiplier"].to_numpy(),
            accepted_lookup["multiplier"],
            rtol=1e-12,
        )
        from easy_glm.workflow.stack_types import StackScorer

        expected = StackScorer.from_dict(refitted_bundle).predict_unit(raw)
        generated = client.post(
            "/api/exports/Vehicle refitted",
            json={**_revision(client), "format": "python_score"},
        )
        assert generated.status_code == 200, generated.text
        scorer_script = tmp_path / "scorer.py"
        scorer_script.write_bytes(generated.content)
        input_path = tmp_path / "rows.parquet"
        raw.write_parquet(input_path)
        # A deployed scorer must read its embedded accepted lookup and model
        # assets, without reopening the source master catalogue.
        Path(location["source_path"]).unlink()
        executed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, runpy, sys, polars as pl; "
                "module = runpy.run_path(sys.argv[1]); "
                "print(json.dumps(module['predict'](pl.read_parquet(sys.argv[2])).tolist()))",
                str(scorer_script),
                str(input_path),
            ],
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            },
            check=True,
            capture_output=True,
            text=True,
        )
        np.testing.assert_allclose(json.loads(executed.stdout), expected, rtol=1e-12)


def test_append_reuses_exact_stack_and_legacy_prefix(tmp_path):
    rng = np.random.default_rng(117)
    raw = pl.DataFrame(
        {
            "Claims": rng.poisson(0.35, 450),
            "Exposure": np.ones(450),
            "Age": rng.integers(20, 75, 450),
            "Region": rng.choice(["A", "B"], 450),
            "PolicyId": np.arange(450),
        }
    )
    project = Project(name="Append source")
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "Age": "predictor",
        "Region": "predictor",
        "PolicyId": "id",
    }
    project.data.split.mode = "random"

    def wait_fit(client: TestClient, name: str) -> dict:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            status = client.get("/api/jobs").json()[name]
            if status["status"] not in ("queued", "running"):
                return status
            time.sleep(0.05)
        raise AssertionError("Fit did not finish")

    with TestClient(
        create_app(project, raw, port=8800), base_url="http://127.0.0.1:8800"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        source = client.post(
            "/api/models/save",
            json={
                **_revision(client),
                "name": "Base",
                "create": True,
                "fields": {
                    "family": "poisson",
                    "link": "log",
                    "target": "Claims",
                    "weight": "Exposure",
                    "components": [
                        {
                            "component_id": "main",
                            "name": "Main",
                            "kind": "glm",
                            "factors": ["Age"],
                            "settings": {"alpha": 0.01},
                        }
                    ],
                },
            },
        )
        assert source.status_code == 200, source.text
        assert (
            client.post("/api/models/Base/fit", json=_revision(client)).status_code
            == 202
        )
        assert wait_fit(client, "Base")["status"] == "complete"
        forged = client.post(
            "/api/models/save",
            json={
                **_revision(client),
                "name": "Base",
                "fields": {
                    "components": [
                        {
                            "component_id": "forged",
                            "name": "Forged ID",
                            "kind": "glm",
                            "factors": ["PolicyId"],
                            "settings": {"alpha": 0.01},
                        }
                    ]
                },
            },
        )
        assert forged.status_code == 409
        assert "read-only" in forged.json()["detail"]
        wb = client.get("/api/workbench").json()
        pin = wb["stack_sources"]["Base"]
        appended = client.post(
            "/api/models/save",
            json={
                **_revision(client),
                "name": "Extended",
                "create": True,
                "source_name": "Base",
                "source_fit_id": pin["source_fit_id"],
                "fields": {
                    "family": "poisson",
                    "link": "log",
                    "target": "Claims",
                    "weight": "Exposure",
                    "components": [
                        *pin["components"],
                        {
                            "component_id": "region",
                            "name": "Region correction",
                            "kind": "glm",
                            "factors": ["Region"],
                            "settings": {"alpha": 0.01},
                        },
                    ],
                },
            },
        )
        assert appended.status_code == 200, appended.text
        flow = client.get("/api/workbench").json()["model_flow"]
        assert {
            "from": "Base",
            "to": "Extended",
            "kind": "builds_on",
            "parent_fit_id": pin["source_fit_id"],
            "prefix_length": 1,
            "edge_status": "live",
            "historical": False,
        } in flow["edges"]
        assert (
            client.get("/api/workbench").json()["models"]["Extended"]["lineage"][
                "parent_name"
            ]
            == "Base"
        )
        draft_node = next(
            node
            for node in client.get("/api/workbench").json()["model_flow"]["nodes"]
            if node["name"] == "Extended"
        )
        assert draft_node["own_fit_available"] is False
        assert draft_node["parent_fit_available"] is True
        assert draft_node["full_refit_required"] is False
        # A saved, unfitted append remains tied to its original fitted prefix.
        # Editing its last component must not silently turn the later fit into
        # a full-recipe fit.
        amended = client.post(
            "/api/models/save",
            json={
                **_revision(client),
                "name": "Extended",
                "fields": {
                    "components": [
                        *pin["components"],
                        {
                            "component_id": "region",
                            "name": "Region correction",
                            "kind": "glm",
                            "factors": ["Region"],
                            "settings": {"alpha": 0.02},
                        },
                    ]
                },
            },
        )
        assert amended.status_code == 200, amended.text
        amended_project = client.get("/api/project").json()
        assert (
            amended_project["models"]["Extended"]["lineage"]
            == client.get("/api/workbench").json()["models"]["Extended"]["lineage"]
        )
        assert (
            amended_project["models"]["Extended"]["components"][-1]["settings"]["alpha"]
            == 0.02
        )
        # A rejected edit is atomic: the saved draft still has the amended
        # suffix and unchanged frozen parent prefix.
        bad_edit = client.post(
            "/api/models/save",
            json={
                **_revision(client),
                "name": "Extended",
                "fields": {
                    "components": [
                        {
                            "component_id": "replacement",
                            "name": "Replacement",
                            "kind": "glm",
                            "factors": ["Region"],
                            "settings": {"alpha": 0.01},
                        },
                        *pin["components"],
                    ]
                },
            },
        )
        assert bad_edit.status_code == 422
        assert client.get("/api/project").json() == amended_project
        assert (
            client.post("/api/models/Extended/fit", json=_revision(client)).status_code
            == 202
        )
        assert wait_fit(client, "Extended")["status"] == "complete"
        base_bundle = client.post(
            "/api/exports/Base", json={**_revision(client), "format": "easyglm"}
        ).json()
        extended_bundle = client.post(
            "/api/exports/Extended", json={**_revision(client), "format": "easyglm"}
        ).json()
        assert base_bundle["components"][0] == extended_bundle["components"][0]
        assert len(extended_bundle["components"]) == 2
        assert (
            client.post(
                "/api/models/Extended/fit",
                json={**_revision(client), "full_refit": True},
            ).status_code
            == 409
        )
        saved_project = client.get("/api/project").json()
        saved_job = client.get("/api/jobs").json()["Extended"]
        source = next(
            Path(tempfile.gettempdir()).glob("easyglm_fits_*/" + saved_job["id"])
        )
        shutil.copytree(source, tmp_path / "fits" / saved_job["id"])
        (tmp_path / "jobs.json").write_text(json.dumps({"Extended": saved_job}))
        with TestClient(
            create_app(
                Project.from_dict(saved_project),
                raw,
                port=8801,
                restore_folder=tmp_path,
            ),
            base_url="http://127.0.0.1:8801",
        ) as restored:
            restored.headers["X-EasyGLM-Token"] = restored.get("/api/session").json()[
                "token"
            ]
            restored_edge = next(
                edge
                for edge in restored.get("/api/workbench").json()["model_flow"]["edges"]
                if edge["to"] == "Extended"
            )
            assert restored_edge["edge_status"] == "historical"
            restored_workbench = restored.get("/api/workbench").json()
            extended_node = next(
                node
                for node in restored_workbench["model_flow"]["nodes"]
                if node["name"] == "Extended"
            )
            # The original Base artifact was not restored, but Extended's own
            # complete scorer was.  It remains usable as the next offset.
            assert extended_node["own_fit_available"] is True
            assert extended_node["parent_fit_available"] is False
            assert extended_node["append_available"] is True
            assert extended_node["revise_available"] is False
            source_components = restored_workbench["stack_sources"]["Extended"][
                "components"
            ]
            grandchild = restored.post(
                "/api/models/save",
                json={
                    **_revision(restored),
                    "name": "Grandchild",
                    "create": True,
                    "source_name": "Extended",
                    "source_fit_id": extended_node["fit_id"],
                    "fields": {
                        "family": "poisson",
                        "link": "log",
                        "target": "Claims",
                        "weight": "Exposure",
                        "components": [
                            *source_components,
                            {
                                "component_id": "age-again",
                                "name": "Age again",
                                "kind": "glm",
                                "factors": ["Age"],
                                "settings": {"alpha": 0.01},
                            },
                        ],
                    },
                },
            )
            assert grandchild.status_code == 200, grandchild.text
            assert (
                restored.post(
                    "/api/models/Grandchild/fit", json=_revision(restored)
                ).status_code
                == 202
            )
            assert wait_fit(restored, "Grandchild")["status"] == "complete"
            # Losing a child's own artifact is different from losing its
            # incoming parent.  Its parent remains live, but this named
            # checkpoint cannot be advertised as scoreable or overwritten.
            grandchild_job = restored.get("/api/jobs").json()["Grandchild"]
            grandchild_source = next(
                Path(tempfile.gettempdir()).glob(
                    "easyglm_fits_*/" + grandchild_job["id"]
                )
            )
            shutil.rmtree(grandchild_source)
            missing_workbench = restored.get("/api/workbench").json()
            missing_node = next(
                node
                for node in missing_workbench["model_flow"]["nodes"]
                if node["name"] == "Grandchild"
            )
            assert missing_node["status"] == "missing_fit"
            assert missing_node["own_fit_available"] is False
            assert missing_node["parent_fit_available"] is True
            assert missing_node["full_refit_required"] is True
            assert restored.get("/api/jobs").json()["Grandchild"]["applicable"] is False
            missing_result = restored.get("/api/results/Grandchild")
            assert missing_result.status_code == 409
            assert "files are no longer available" in missing_result.json()["detail"]
            assert (
                restored.post(
                    "/api/models/Grandchild/fit", json=_revision(restored)
                ).status_code
                == 409
            )

        legacy = client.post(
            "/api/models/save",
            json={
                **_revision(client),
                "name": "Legacy",
                "create": True,
                "fields": {
                    "family": "poisson",
                    "link": "log",
                    "target": "Claims",
                    "weight": "Exposure",
                    "predictors": ["Age"],
                    "penalty": {"alpha": 0.01, "cv": None},
                },
            },
        )
        assert legacy.status_code == 200, legacy.text
        assert (
            client.post("/api/models/Legacy/fit", json=_revision(client)).status_code
            == 202
        )
        assert wait_fit(client, "Legacy")["status"] == "complete"
        pin = client.get("/api/workbench").json()["stack_sources"]["Legacy"]
        assert pin["components"][0]["kind"] == "legacy"
        adapted = client.post(
            "/api/models/save",
            json={
                **_revision(client),
                "name": "After legacy",
                "create": True,
                "source_name": "Legacy",
                "source_fit_id": pin["source_fit_id"],
                "fields": {
                    "family": "poisson",
                    "link": "log",
                    "target": "Claims",
                    "weight": "Exposure",
                    "components": [
                        *pin["components"],
                        {
                            "component_id": "region",
                            "name": "Region correction",
                            "kind": "glm",
                            "factors": ["Region"],
                            "settings": {"alpha": 0.01},
                        },
                    ],
                },
            },
        )
        assert adapted.status_code == 200, adapted.text
        assert (
            client.post(
                "/api/models/After legacy/fit", json=_revision(client)
            ).status_code
            == 202
        )
        assert wait_fit(client, "After legacy")["status"] == "complete"
        result = client.get("/api/results/After legacy").json()
        assert result["stack"] is True
        assert result["components"][0]["id"] == "legacy_main"
        table = client.get("/api/stack/After legacy/component/legacy_main/table")
        assert table.status_code == 200, table.text
        assert table.json()["tables"]
