"""Project-level model-plan contracts for the staged workbench."""

from __future__ import annotations

import io
import time
from copy import deepcopy

import numpy as np
import polars as pl
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from easy_glm.desktop import server
from easy_glm.desktop.jobs import model_key
from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import Project


@pytest.fixture
def plan_client():
    rng = np.random.default_rng(19)
    raw = pl.DataFrame(
        {
            "Claims": rng.binomial(1, 0.4, 500),
            "Exposure": np.ones(500),
            "Age": rng.integers(18, 80, 500),
            "Region": rng.choice(["A", "B"], 500),
            "VehicleCode": rng.choice(["A", "B"], 500),
        }
    )
    project = Project(name="Frequency plan")
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "Age": "predictor",
        "Region": "predictor",
    }
    project.data.split.mode = "random"
    with TestClient(
        create_app(project, raw, port=8789), base_url="http://127.0.0.1:8789"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        yield client


def revision(client):
    view = client.get("/api/model-plan").json()
    return {key: view[key] for key in ("session_id", "revision")}


def wait(client, name):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get("/api/jobs").json().get(name, {})
        if job.get("status") not in {"queued", "running"}:
            return job
        time.sleep(0.05)
    pytest.fail("plan stage did not finish")


def stage(stage_id, name, *factors):
    return {
        "component_id": stage_id,
        "name": name,
        "kind": "glm",
        "factors": list(factors),
        "settings": {"alpha": 0.01, "cv": None},
        "amendments": [],
    }


def test_legacy_project_serialisation_and_key_stay_unchanged_without_plan():
    project = Project()
    project.data.roles = {"Y": "target", "x": "predictor"}
    project.new_model("Existing", predictors=["x"])
    before = project.to_dict()
    assert "model_plan" not in before
    assert before["version"] == 2
    assert model_key(project, "Existing") == model_key(
        Project.from_dict(before), "Existing"
    )


def test_plan_can_be_saved_before_variables_are_finished(plan_client):
    response = plan_client.post(
        "/api/model-plan",
        json={
            **revision(plan_client),
            "operation": "start",
            "plan": {"mode": "sequential", "stages": [stage("main", "Main GLM")]},
        },
    )
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved["plan"]["stages"][0]["name"] == "Main GLM"
    assert saved["next_stage_id"] == "main"


def test_future_blank_stage_does_not_block_first_fit(plan_client):
    plan = {
        "mode": "sequential",
        "family": "poisson",
        "link": "log",
        "stages": [stage("main", "Main GLM", "Age"), stage("later", "Later GLM")],
    }
    assert (
        plan_client.post(
            "/api/model-plan", json={**revision(plan_client), "plan": plan}
        ).status_code
        == 200
    )
    started = plan_client.post("/api/model-plan/fit-next", json=revision(plan_client))
    assert started.status_code == 202, started.text
    assert wait(plan_client, started.json()["model_name"])["status"] == "complete"
    assert wait(plan_client, started.json()["model_name"])["status"] == "complete"
    assert plan_client.get("/api/model-plan").json()["next_stage_id"] == "later"


def test_catalogue_can_be_planned_before_upload_but_cannot_fit_without_it(plan_client):
    plan = {
        "mode": "sequential",
        "family": "poisson",
        "link": "log",
        "stages": [
            stage("main", "Main GLM", "Age"),
            {
                "component_id": "vehicle",
                "name": "Vehicle file",
                "kind": "catalogue",
                "factors": [],
                "settings": {"key": "VehicleCode"},
            },
        ],
    }
    saved = plan_client.post(
        "/api/model-plan", json={**revision(plan_client), "plan": plan}
    )
    assert saved.status_code == 200, saved.text
    first = plan_client.post("/api/model-plan/fit-next", json=revision(plan_client))
    assert first.status_code == 202, first.text
    assert wait(plan_client, first.json()["model_name"])["status"] == "complete"
    blocked = plan_client.post("/api/model-plan/fit-next", json=revision(plan_client))
    assert blocked.status_code == 422
    assert "upload a catalogue in Project & data" in blocked.text
    assert len(plan_client.get("/api/model-plan").json()["fit_refs"]) == 1

    # An arbitrary path or a half-specified upload must not become a saved source.
    plan["stages"][1]["settings"]["source_path"] = "/tmp/not-an-upload.parquet"
    denied = plan_client.post(
        "/api/model-plan", json={**revision(plan_client), "plan": plan}
    )
    assert denied.status_code == 422
    plan["stages"][1]["settings"]["source_sha256"] = "forged"
    denied = plan_client.post(
        "/api/model-plan", json={**revision(plan_client), "plan": plan}
    )
    assert denied.status_code == 422
    assert "not a pinned upload" in denied.text


def test_opening_project_replaces_plan_references_without_assuming_fits_exist(
    plan_client, monkeypatch
):
    plan = {
        "mode": "sequential",
        "family": "poisson",
        "link": "log",
        "stages": [stage("main", "Main GLM", "Age"), stage("later", "Later GLM")],
    }
    saved = plan_client.post(
        "/api/model-plan", json={**revision(plan_client), "plan": plan}
    )
    assert saved.status_code == 200
    first = plan_client.post("/api/model-plan/fit-next", json=revision(plan_client))
    assert first.status_code == 202
    assert wait(plan_client, first.json()["model_name"])["status"] == "complete"
    loaded = Project.from_dict(plan_client.get("/api/project").json())
    loaded.model_plan.plan_id = "imported-plan"
    expected_refs = deepcopy(loaded.model_plan.fit_refs)
    loaded_raw = pl.DataFrame(
        {"Claims": [0, 1], "Exposure": [1.0, 1.0], "Age": [20, 50]}
    )
    monkeypatch.setattr(
        server, "load_project_input", lambda *args: (loaded, loaded_raw)
    )
    opened = plan_client.post(
        "/api/project/open",
        json={**revision(plan_client), "kind": "project", "path": "saved-project.json"},
    )
    assert opened.status_code == 200, opened.text
    plan_client.headers["X-EasyGLM-Token"] = plan_client.get("/api/session").json()[
        "token"
    ]
    view = plan_client.get("/api/model-plan").json()
    assert view["plan"]["fit_refs"] == expected_refs
    assert view["fit_refs"] == []  # Project JSON alone does not restore fit artifacts.
    assert view["next_stage_id"] == "main"
    assert plan_client.get("/api/jobs").json() == {}
    view["plan"]["stages"][1]["name"] = "Renamed future model"
    updated = plan_client.post(
        "/api/model-plan", json={**revision(plan_client), "plan": view["plan"]}
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["plan"]["fit_refs"] == expected_refs
    assert updated.json()["fit_refs"] == []


def test_single_plan_fit_has_opt_in_stack_inspection_without_changing_single_result(
    plan_client,
):
    plan = {
        "mode": "single",
        "family": "poisson",
        "link": "log",
        "stages": [stage("main", "Main GLM", "Age")],
    }
    saved = plan_client.post(
        "/api/model-plan", json={**revision(plan_client), "plan": plan}
    )
    assert saved.status_code == 200, saved.text
    first = plan_client.post("/api/model-plan/fit-next", json=revision(plan_client))
    assert first.status_code == 202, first.text
    name = first.json()["model_name"]
    assert wait(plan_client, name)["status"] == "complete"

    ordinary = plan_client.get(f"/api/results/{name}")
    assert ordinary.status_code == 200
    assert not ordinary.json().get("stack")
    assert ordinary.json()["table_index"]
    inspection = plan_client.get(f"/api/results/{name}", params={"stack_view": True})
    assert inspection.status_code == 200, inspection.text
    viewed = inspection.json()
    assert viewed["stack"] is True
    assert len(viewed["components"]) == 1
    assert viewed["summary"]["name"] == name
    assert viewed["summary"]["train_ae"] == pytest.approx(
        ordinary.json()["metrics"]["train"]["ae"]
    )
    assert viewed["diagnostic_scopes"][-1]["id"] == ""

    diagnostics = plan_client.get(
        f"/api/stack/{name}/diagnostics",
        params={"subset": "train", "variable": "Age"},
    )
    assert diagnostics.status_code == 200, diagnostics.text
    assert diagnostics.json()["saved_model"] == name
    assert diagnostics.json()["ae"]
    component_id = viewed["components"][0]["id"]
    table = plan_client.get(f"/api/stack/{name}/component/{component_id}/table")
    assert table.status_code == 200, table.text
    assert table.json()["tables"]

    another = {**plan, "stages": [stage("region", "Region GLM", "Region")]}
    restarted = plan_client.post(
        "/api/model-plan",
        json={**revision(plan_client), "operation": "start", "plan": another},
    )
    assert restarted.status_code == 200, restarted.text
    second = plan_client.post("/api/model-plan/fit-next", json=revision(plan_client))
    assert second.status_code == 202, second.text
    second_name = second.json()["model_name"]
    assert wait(plan_client, second_name)["status"] == "complete"
    comparison = plan_client.get(
        f"/api/stack/{name}/compare",
        params={"challenger": second_name, "subset": "train"},
    )
    assert comparison.status_code == 200, comparison.text
    assert comparison.json()["model"] == name
    for format_name in ("easyglm", "xlsx", "python_score"):
        exported = plan_client.post(
            f"/api/exports/{name}",
            json={**revision(plan_client), "format": format_name},
        )
        assert exported.status_code == 200, (format_name, exported.text[:300])
        assert exported.content
    assert not plan_client.get(f"/api/results/{name}").json().get("stack")


def test_fit_next_uses_declared_prefix_and_frozen_saved_predecessor(plan_client):
    plan = {
        "mode": "sequential",
        "family": "poisson",
        "link": "log",
        "divide_target_by_weight": True,
        "stages": [
            stage("main", "Main GLM", "Age"),
            stage("region", "Region", "Region"),
        ],
    }
    assert (
        plan_client.post(
            "/api/model-plan", json={**revision(plan_client), "plan": plan}
        ).status_code
        == 200
    )
    first = plan_client.post("/api/model-plan/fit-next", json=revision(plan_client))
    assert first.status_code == 202, first.text
    first_name = first.json()["model_name"]
    assert wait(plan_client, first_name)["status"] == "complete"
    view = plan_client.get("/api/model-plan").json()
    assert view["stage_statuses"][0]["state"] == "fitted"
    assert view["next_stage_id"] == "region"
    second = plan_client.post("/api/model-plan/fit-next", json=revision(plan_client))
    assert second.status_code == 202, second.text
    second_name = second.json()["model_name"]
    assert wait(plan_client, second_name)["status"] == "complete"
    final = plan_client.get("/api/project").json()
    config = final["models"][second_name]
    assert [item["component_id"] for item in config["components"]] == ["main", "region"]
    assert config["lineage"]["parent_name"] == first_name


def test_save_cannot_reorder_a_fitted_prefix_without_explicit_revision(plan_client):
    plan = {
        "mode": "sequential",
        "family": "poisson",
        "link": "log",
        "divide_target_by_weight": True,
        "stages": [stage("main", "Main GLM", "Age")],
    }
    assert (
        plan_client.post(
            "/api/model-plan", json={**revision(plan_client), "plan": plan}
        ).status_code
        == 200
    )
    started = plan_client.post(
        "/api/model-plan/fit-next", json=revision(plan_client)
    ).json()
    assert wait(plan_client, started["model_name"])["status"] == "complete"
    changed = {**plan, "stages": [stage("main", "Main GLM", "Region")]}
    denied = plan_client.post(
        "/api/model-plan", json={**revision(plan_client), "plan": changed}
    )
    assert denied.status_code == 422
    allowed = plan_client.post(
        "/api/model-plan",
        json={**revision(plan_client), "operation": "revise", "plan": changed},
    )
    assert allowed.status_code == 200, allowed.text


def test_revise_keeps_unchanged_fitted_prefix(plan_client):
    plan = {
        "mode": "sequential",
        "family": "poisson",
        "link": "log",
        "stages": [stage("first", "First", "Age"), stage("second", "Second", "Region")],
    }
    assert (
        plan_client.post(
            "/api/model-plan", json={**revision(plan_client), "plan": plan}
        ).status_code
        == 200
    )
    fit = plan_client.post(
        "/api/model-plan/fit-next", json=revision(plan_client)
    ).json()
    assert wait(plan_client, fit["model_name"])["status"] == "complete"
    revised = {
        **plan,
        "stages": [
            stage("first", "First", "Age"),
            stage("second", "Second revised", "Region"),
        ],
    }
    response = plan_client.post(
        "/api/model-plan",
        json={**revision(plan_client), "operation": "revise", "plan": revised},
    )
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["stage_statuses"][0]["state"] == "fitted"
    assert view["next_stage_id"] == "second"


def test_single_glm_plan_keeps_binomial_logit_available(plan_client):
    plan = {
        "mode": "single",
        "family": "binomial",
        "link": "logit",
        "stages": [stage("glm", "Risk GLM", "Age")],
    }
    response = plan_client.post(
        "/api/model-plan", json={**revision(plan_client), "plan": plan}
    )
    assert response.status_code == 200, response.text
    started = plan_client.post("/api/model-plan/fit-next", json=revision(plan_client))
    assert started.status_code == 202, started.text
    name = started.json()["model_name"]
    assert wait(plan_client, name)["status"] == "complete"
    ordinary = plan_client.get(f"/api/results/{name}")
    assert ordinary.status_code == 200
    assert not ordinary.json().get("stack")
    assert (
        plan_client.get(f"/api/results/{name}", params={"stack_view": True}).status_code
        == 404
    )


@pytest.mark.parametrize("link", ["identity", {}, ["log"]])
def test_plan_rejects_a_link_that_cannot_produce_rate_tables(plan_client, link):
    response = plan_client.post(
        "/api/model-plan",
        json={
            **revision(plan_client),
            "plan": {
                "mode": "single",
                "family": "gaussian",
                "link": link,
                "stages": [stage("main", "Gaussian GLM", "Age")],
            },
        },
    )
    assert response.status_code == 422
    assert "link" in response.text
    assert plan_client.get("/api/model-plan").json()["plan"] is None


def test_starting_identical_plan_does_not_reuse_old_fit(plan_client):
    plan = {
        "mode": "sequential",
        "family": "poisson",
        "link": "log",
        "stages": [stage("root", "Root", "Age")],
    }
    assert (
        plan_client.post(
            "/api/model-plan", json={**revision(plan_client), "plan": plan}
        ).status_code
        == 200
    )
    started = plan_client.post(
        "/api/model-plan/fit-next", json=revision(plan_client)
    ).json()
    assert wait(plan_client, started["model_name"])["status"] == "complete"
    fresh = plan_client.post(
        "/api/model-plan",
        json={**revision(plan_client), "operation": "start", "plan": plan},
    )
    assert fresh.status_code == 200, fresh.text
    assert fresh.json()["fit_refs"] == []
    assert fresh.json()["stage_statuses"][0]["state"] == "ready"


def test_revising_shared_assignments_keeps_recipes_but_requires_a_new_root_fit(
    plan_client,
):
    plan = {
        "mode": "sequential",
        "family": "poisson",
        "link": "log",
        "stages": [stage("root", "Core GLM", "Age"), stage("later", "Vehicle", "Age")],
    }
    saved = plan_client.post(
        "/api/model-plan", json={**revision(plan_client), "plan": plan}
    )
    assert saved.status_code == 200, saved.text
    started = plan_client.post("/api/model-plan/fit-next", json=revision(plan_client))
    assert started.status_code == 202, started.text
    model_name = started.json()["model_name"]
    assert wait(plan_client, model_name)["status"] == "complete"
    fitted = plan_client.get("/api/model-plan").json()
    assert len(fitted["fit_refs"]) == 1

    revised = plan_client.post(
        "/api/model-plan",
        json={
            **revision(plan_client),
            "operation": "start",
            "plan": fitted["plan"],
        },
    )
    assert revised.status_code == 200, revised.text
    assert revised.json()["plan"]["plan_id"] != fitted["plan"]["plan_id"]
    assert revised.json()["plan"]["stages"] == fitted["plan"]["stages"]
    assert revised.json()["fit_refs"] == []

    variables = plan_client.get("/api/variables").json()
    setup = deepcopy(variables["setup"])
    setup["roles"]["predictor"].remove("Region")
    setup["roles"].setdefault("ignore", []).append("Region")
    edit = {**revision(plan_client), "setup": setup}
    preview = plan_client.post("/api/variables/preview", json=edit)
    assert preview.status_code == 200, preview.text
    applied = plan_client.post("/api/variables/apply", json=edit)
    assert applied.status_code == 200, applied.text
    current_plan = plan_client.get("/api/model-plan").json()
    assert current_plan["fit_refs"] == []
    assert current_plan["next_stage_id"] == "root"
    assert current_plan["stage_statuses"][0]["state"] == "ready"
    assert "Region" not in current_plan["safe_columns"][0]
    assert model_name in plan_client.get("/api/project").json()["models"]


def test_single_plan_can_use_an_unassigned_raw_factor():
    raw = pl.DataFrame(
        {
            "Claims": [0, 1] * 150,
            "Exposure": [1.0] * 300,
            "Age": list(range(20, 80)) * 5,
        }
    )
    project = Project(name="unassigned")
    project.data.roles = {"Claims": "target", "Exposure": "weight"}
    project.data.split.mode = "random"
    with TestClient(
        create_app(project, raw, port=8790), base_url="http://127.0.0.1:8790"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        plan = {
            "mode": "single",
            "family": "binomial",
            "link": "logit",
            "stages": [stage("root", "Root", "Age")],
        }
        assert (
            client.post(
                "/api/model-plan", json={**revision(client), "plan": plan}
            ).status_code
            == 200
        )
        response = client.post("/api/model-plan/fit-next", json=revision(client))
        assert response.status_code == 202, response.text
        assert wait(client, response.json()["model_name"])["status"] == "complete"


def test_adopted_complete_stack_remains_exact_frozen_prefix(plan_client):
    original = {
        "mode": "sequential",
        "family": "poisson",
        "link": "log",
        "stages": [stage("one", "One", "Age"), stage("two", "Two", "Region")],
    }
    assert (
        plan_client.post(
            "/api/model-plan", json={**revision(plan_client), "plan": original}
        ).status_code
        == 200
    )
    first = plan_client.post(
        "/api/model-plan/fit-next", json=revision(plan_client)
    ).json()
    assert wait(plan_client, first["model_name"])["status"] == "complete"
    second = plan_client.post(
        "/api/model-plan/fit-next", json=revision(plan_client)
    ).json()
    assert wait(plan_client, second["model_name"])["status"] == "complete"

    adopted = plan_client.post(
        "/api/model-plan",
        json={
            **revision(plan_client),
            "operation": "adopt",
            "adopt_model_name": second["model_name"],
        },
    )
    assert adopted.status_code == 200, adopted.text
    assert adopted.json()["stage_statuses"][0]["state"] == "fitted"
    adopted_plan = adopted.json()["plan"]
    continued = {
        **adopted_plan,
        "stages": [
            *adopted_plan["stages"],
            stage("three", "Three", "Age"),
        ],
    }
    revised = plan_client.post(
        "/api/model-plan",
        json={**revision(plan_client), "operation": "revise", "plan": continued},
    )
    assert revised.status_code == 200, revised.text
    third = plan_client.post("/api/model-plan/fit-next", json=revision(plan_client))
    assert third.status_code == 202, third.text
    assert wait(plan_client, third.json()["model_name"])["status"] == "complete"


def test_client_cannot_inject_an_adopted_source(plan_client):
    injected = {
        "mode": "sequential",
        "family": "poisson",
        "link": "log",
        "adopted_model_name": "anything",
        "stages": [stage("root", "Root", "Age")],
    }
    response = plan_client.post(
        "/api/model-plan",
        json={**revision(plan_client), "operation": "start", "plan": injected},
    )
    assert response.status_code == 422


def test_adopted_root_cannot_be_revised_into_a_different_root(plan_client):
    original = {
        "mode": "sequential",
        "family": "poisson",
        "link": "log",
        "stages": [stage("root", "Root", "Age")],
    }
    assert (
        plan_client.post(
            "/api/model-plan", json={**revision(plan_client), "plan": original}
        ).status_code
        == 200
    )
    fitted = plan_client.post(
        "/api/model-plan/fit-next", json=revision(plan_client)
    ).json()
    assert wait(plan_client, fitted["model_name"])["status"] == "complete"
    adopted = plan_client.post(
        "/api/model-plan",
        json={
            **revision(plan_client),
            "operation": "adopt",
            "adopt_model_name": fitted["model_name"],
        },
    )
    assert adopted.status_code == 200
    forged = {
        **adopted.json()["plan"],
        "stages": [stage("adopted-root", fitted["model_name"], "Region")],
    }
    response = plan_client.post(
        "/api/model-plan",
        json={**revision(plan_client), "operation": "revise", "plan": forged},
    )
    assert response.status_code == 422


def test_catalogue_attributes_retain_schema_in_reopened_plan_view(plan_client):
    payload = io.BytesIO()
    pl.DataFrame({"VehicleCode": ["A", "B"], "EngineSize": [1200, 1800]}).write_parquet(
        payload
    )
    upload = plan_client.post(
        "/api/catalogues/upload",
        params={
            "filename": "vehicle.parquet",
            **revision(plan_client),
        },
        content=payload.getvalue(),
        headers={"content-type": "application/octet-stream"},
    )
    assert upload.status_code == 200, upload.text
    pinned = upload.json()
    plan = {
        "mode": "sequential",
        "family": "poisson",
        "link": "log",
        "stages": [
            stage("root", "Root", "Age"),
            {
                "component_id": "vehicle-file",
                "name": "Vehicle file",
                "kind": "catalogue",
                "factors": ["EngineSize"],
                "settings": {
                    "key": "VehicleCode",
                    "source_path": pinned["source_path"],
                    "source_sha256": pinned["source_sha256"],
                    "method": "catboost",
                    "tune": False,
                    "iterations": 5,
                    "depth": 2,
                },
                "amendments": [],
            },
            stage("after", "After vehicle", "EngineSize"),
        ],
    }
    saved = plan_client.post(
        "/api/model-plan", json={**revision(plan_client), "plan": plan}
    )
    assert saved.status_code == 200, saved.text
    view = saved.json()
    descriptor = view["catalogue_descriptors"]["vehicle-file"]
    assert descriptor["filename"] == "vehicle.parquet"
    assert descriptor["rows"] is None
    assert descriptor["columns"] == [
        {"name": "VehicleCode", "dtype": "String"},
        {"name": "EngineSize", "dtype": "Int64"},
    ]
    assert descriptor["key_columns"] == descriptor["columns"]
    assert descriptor["attribute_columns"] == descriptor["columns"]
    # The attribute becomes a raw factor only after its catalogue stage, while
    # retaining its source dtype so its own numeric setup is available.
    assert view["safe_column_details"][2] == [
        {"name": "Age", "dtype": "Int64", "role": "predictor"},
        {"name": "EngineSize", "dtype": "Int64", "role": "unassigned"},
        {"name": "Region", "dtype": "String", "role": "predictor"},
        {"name": "VehicleCode", "dtype": "String", "role": "unassigned"},
    ]

    root = plan_client.post(
        "/api/model-plan/fit-next", json=revision(plan_client)
    ).json()
    assert wait(plan_client, root["model_name"])["status"] == "complete"
    # Its numeric attribute is valid at the catalogue stage; it is introduced
    # by the pinned lookup rather than required in the raw project dataset.
    catalogue_fit = plan_client.post(
        "/api/model-plan/fit-next", json=revision(plan_client)
    )
    assert catalogue_fit.status_code == 202, catalogue_fit.text
    assert wait(plan_client, catalogue_fit.json()["model_name"])["status"] == "complete"
    after = plan_client.post("/api/model-plan/fit-next", json=revision(plan_client))
    assert after.status_code == 202, after.text
    assert wait(plan_client, after.json()["model_name"])["status"] == "complete"


def test_pending_catalogue_plan_restores_pinned_source_after_restart(tmp_path):
    raw = pl.DataFrame(
        {
            "Claims": [0, 1] * 100,
            "Exposure": [1.0] * 200,
            "Age": list(range(20, 70)) * 4,
            "VehicleCode": ["A", "B"] * 100,
        }
    )
    project = Project(name="restart catalogue")
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "Age": "predictor",
    }
    project.data.split.mode = "random"
    with TestClient(
        create_app(project, raw, port=8791), base_url="http://127.0.0.1:8791"
    ) as first:
        first.headers["X-EasyGLM-Token"] = first.get("/api/session").json()["token"]
        payload = io.BytesIO()
        pl.DataFrame(
            {"VehicleCode": ["A", "B"], "EngineSize": [1200, 1800]}
        ).write_parquet(payload)
        upload = first.post(
            "/api/catalogues/upload",
            params={"filename": "vehicle.parquet", **revision(first)},
            content=payload.getvalue(),
            headers={"content-type": "application/octet-stream"},
        )
        assert upload.status_code == 200, upload.text
        pinned = upload.json()
        plan = {
            "mode": "sequential",
            "family": "poisson",
            "link": "log",
            "stages": [
                stage("root", "Root", "Age"),
                {
                    "component_id": "vehicle-file",
                    "name": "Vehicle file",
                    "kind": "catalogue",
                    "factors": ["EngineSize"],
                    "settings": {
                        "key": "VehicleCode",
                        "source_path": pinned["source_path"],
                        "source_sha256": pinned["source_sha256"],
                        "method": "catboost",
                        "tune": False,
                        "iterations": 5,
                        "depth": 2,
                    },
                    "amendments": [],
                },
            ],
        }
        assert (
            first.post(
                "/api/model-plan", json={**revision(first), "plan": plan}
            ).status_code
            == 200
        )
        saved = Project.from_dict(first.get("/api/project").json())

    with TestClient(
        create_app(saved, raw, port=8792), base_url="http://127.0.0.1:8792"
    ) as restarted:
        restarted.headers["X-EasyGLM-Token"] = restarted.get("/api/session").json()[
            "token"
        ]
        view = restarted.get("/api/model-plan").json()
        assert view["catalogue_descriptors"]["vehicle-file"]["columns"][-1] == {
            "name": "EngineSize",
            "dtype": "Int64",
        }
        root = restarted.post("/api/model-plan/fit-next", json=revision(restarted))
        assert root.status_code == 202, root.text
        assert wait(restarted, root.json()["model_name"])["status"] == "complete"
        catalogue = restarted.post("/api/model-plan/fit-next", json=revision(restarted))
        assert catalogue.status_code == 202, catalogue.text
        assert wait(restarted, catalogue.json()["model_name"])["status"] == "complete"
