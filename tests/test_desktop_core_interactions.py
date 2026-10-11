"""Lifecycle checks for interaction candidates nested inside a v2 GLM core."""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from easy_glm.desktop.jobs import FitJobs, model_key
from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import Project


def _core(component_id: str, name: str, *factors: str) -> dict[str, object]:
    return {
        "component_id": component_id,
        "name": name,
        "kind": "glm",
        "factors": list(factors),
        "settings": {"alpha": 0.02, "cv": None},
        "amendments": [],
    }


def _plan(*, sequential: bool) -> dict[str, object]:
    models = [
        {
            "model_id": "main",
            "name": "Main effects",
            "core": _core("main", "Main effects", "Age"),
            "pair_corrections": [],
        }
    ]
    if sequential:
        models.append(
            {
                "model_id": "vehicle",
                "name": "Vehicle correction",
                "core": _core("vehicle", "Vehicle correction", "Region"),
                "pair_corrections": [],
            }
        )
    return {
        "format_version": 2,
        "mode": "sequential" if sequential else "single",
        "family": "poisson",
        "link": "log",
        "tweedie_power": 1.5,
        "divide_target_by_weight": False,
        "models": models,
        "adopted_root": None,
        "fit_refs": [],
    }


@pytest.fixture
def core_client():
    rng = np.random.default_rng(903)
    rows = 360
    age = rng.integers(18, 80, rows)
    region = rng.choice(["A", "B"], rows)
    # Deliberately make the Age x Region signal large enough for a small,
    # manual CatBoost table to have a reviewable non-neutral correction.
    rate = np.where((age > 50) & (region == "B"), 0.8, 0.12)
    raw = pl.DataFrame(
        {
            "Claims": rng.poisson(rate),
            "Exposure": np.ones(rows),
            "Age": age,
            "Region": region,
            "Vehicle": rng.choice(["small", "large"], rows),
        }
    )
    project = Project(name="Core interactions")
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "Age": "predictor",
        "Region": "predictor",
    }
    project.data.split.mode = "random"
    with TestClient(
        create_app(project, raw, port=8813), base_url="http://127.0.0.1:8813"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        yield client


@pytest.fixture
def binomial_core_client():
    rng = np.random.default_rng(904)
    rows = 360
    raw = pl.DataFrame(
        {
            "Claims": rng.binomial(1, 0.22, rows),
            "Exposure": np.ones(rows),
            "Age": rng.integers(18, 80, rows),
            "Region": rng.choice(["A", "B"], rows),
        }
    )
    project = Project(name="Binomial core")
    project.data.roles = {
        "Claims": "target",
        "Age": "predictor",
        "Region": "predictor",
    }
    project.data.split.mode = "random"
    with TestClient(
        create_app(project, raw, port=8814), base_url="http://127.0.0.1:8814"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        yield client


def _revision(client: TestClient) -> dict[str, object]:
    view = client.get("/api/model-plan").json()
    return {key: view[key] for key in ("session_id", "revision")}


def _wait(client: TestClient, name: str) -> dict[str, object]:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        job = client.get("/api/jobs").json().get(name, {})
        if job.get("status") not in {"queued", "running"}:
            return job
        time.sleep(0.05)
    pytest.fail(f"{name} did not finish")


def _fit_main(client: TestClient, *, sequential: bool) -> str:
    saved = client.post(
        "/api/model-plan",
        json={
            **_revision(client),
            "operation": "start",
            "plan": _plan(sequential=sequential),
        },
    )
    assert saved.status_code == 200, saved.text
    started = client.post("/api/model-plan/fit-next", json=_revision(client))
    assert started.status_code == 202, started.text
    name = started.json()["model_name"]
    assert _wait(client, name)["status"] == "complete"
    return name


def _new_candidate(
    client: TestClient, *, a: str = "Age", b: str = "Region"
) -> dict[str, object]:
    response = client.post(
        "/api/core-models/main/interactions",
        json={
            **_revision(client),
            "model_id": "main",
            "a": a,
            "b": b,
            "settings": {
                "tune": False,
                "candidate": {
                    "iterations": 8,
                    "depth": 2,
                    "learning_rate": 0.1,
                    "l2_leaf_reg": 3.0,
                },
                "min_weight_share": 0.001,
            },
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["candidate"]


def _review(client: TestClient, candidate: dict[str, object]) -> None:
    candidate_id = str(candidate["candidate_id"])
    started = client.post(
        f"/api/core-models/main/interactions/{candidate_id}/fit",
        json={**_revision(client), "candidate_id": candidate_id},
    )
    assert started.status_code == 202, started.text
    assert _wait(client, str(started.json()["job"]["name"]))["status"] == "complete"
    view = client.get("/api/core-models/main/interactions").json()
    current = next(
        item for item in view["candidates"] if item["candidate_id"] == candidate_id
    )
    assert current["status"] == "review"
    tables = current["review"]["pair_tables"]
    assert tables
    table = next(iter(tables.values()))
    assert table["columns"]
    assert table["rows"]


def _keep(client: TestClient, candidate: dict[str, object]) -> dict[str, object]:
    _review(client, candidate)
    candidate_id = str(candidate["candidate_id"])
    response = client.post(
        f"/api/core-models/main/interactions/{candidate_id}/accept",
        json={**_revision(client), "candidate_id": candidate_id},
    )
    assert response.status_code == 200, response.text
    return response.json()["accepted"]


def test_candidate_isolated_then_keep_updates_main_checkpoint_and_refits_suffix(
    core_client,
):
    source_name = _fit_main(core_client, sequential=True)
    before = core_client.get("/api/model-plan").json()
    candidate = _new_candidate(core_client)
    after_create = core_client.get("/api/model-plan").json()
    assert after_create["fit_refs"] == before["fit_refs"]
    assert all(
        "__interaction_candidate__" not in name for name in after_create["history"]
    )

    _review(core_client, candidate)
    candidate_id = str(candidate["candidate_id"])
    kept = core_client.post(
        f"/api/core-models/main/interactions/{candidate_id}/accept",
        json={**_revision(core_client), "candidate_id": candidate_id},
    )
    assert kept.status_code == 200, kept.text
    accepted = kept.json()["accepted"]
    interaction_view = core_client.get("/api/core-models/main/interactions").json()
    assert accepted["through_component_id"] in interaction_view["accepted_tables"]
    assert interaction_view["accepted_tables"][accepted["through_component_id"]]["rows"]
    view = core_client.get("/api/model-plan").json()
    assert view["stage_statuses"][0]["state"] == "fitted"
    assert (
        view["fit_refs"][0]["through_component_id"] == accepted["through_component_id"]
    )
    assert view["stage_statuses"][1]["state"] == "ready"
    assert source_name in {item["model_name"] for item in before["fit_refs"]}

    downstream = core_client.post(
        "/api/model-plan/fit-next", json=_revision(core_client)
    )
    assert downstream.status_code == 202, downstream.text
    assert _wait(core_client, downstream.json()["model_name"])["status"] == "complete"


@pytest.mark.parametrize(
    ("mode", "link"),
    [("single", "log"), ("single", None), ("sequential", "log")],
)
def test_sole_gaussian_glm_keeps_the_direct_glm_fit_path(core_client, mode, link):
    plan = _plan(sequential=False)
    plan["mode"] = mode
    plan["family"] = "gaussian"
    plan["link"] = link
    saved = core_client.post(
        "/api/model-plan",
        json={**_revision(core_client), "operation": "start", "plan": plan},
    )
    assert saved.status_code == 200, saved.text
    started = core_client.post("/api/model-plan/fit-next", json=_revision(core_client))
    assert started.status_code == 202, started.text
    job = _wait(core_client, started.json()["model_name"])
    assert job["status"] == "complete", job["message"]
    assert (
        not core_client.get(f"/api/results/{started.json()['model_name']}")
        .json()
        .get("stack")
    )
    view = core_client.get("/api/model-plan").json()
    assert view["stage_statuses"][0]["state"] == "fitted"
    assert view["fit_refs"][0]["model_id"] == "main"
    if link is None:
        # A family-default Gaussian GLM is fitted with a log link.  Its first
        # correction must therefore use that fitted log scorer as an offset,
        # rather than treating the absent plan field as Gaussian identity.
        _review(core_client, _new_candidate(core_client))


def test_sequential_gaussian_direct_root_is_retained_for_the_next_model(core_client):
    plan = _plan(sequential=True)
    plan["family"] = "gaussian"
    saved = core_client.post(
        "/api/model-plan",
        json={**_revision(core_client), "operation": "start", "plan": plan},
    )
    assert saved.status_code == 200, saved.text
    main = core_client.post("/api/model-plan/fit-next", json=_revision(core_client))
    assert main.status_code == 202, main.text
    assert _wait(core_client, main.json()["model_name"])["status"] == "complete"

    view = core_client.get("/api/model-plan").json()
    assert view["stage_statuses"][0]["state"] == "fitted"
    assert view["stage_statuses"][1]["state"] == "ready"
    assert view["fit_refs"][0]["model_id"] == "main"

    downstream = core_client.post(
        "/api/model-plan/fit-next", json=_revision(core_client)
    )
    assert downstream.status_code == 202, downstream.text
    assert _wait(core_client, downstream.json()["model_name"])["status"] == "complete"


def test_single_binomial_logit_glm_is_retained_without_a_stack_adapter(
    binomial_core_client,
):
    plan = _plan(sequential=False)
    plan.update({"family": "binomial", "link": "logit"})
    saved = binomial_core_client.post(
        "/api/model-plan",
        json={
            **_revision(binomial_core_client),
            "operation": "start",
            "plan": plan,
        },
    )
    assert saved.status_code == 200, saved.text
    started = binomial_core_client.post(
        "/api/model-plan/fit-next", json=_revision(binomial_core_client)
    )
    assert started.status_code == 202, started.text
    assert (
        _wait(binomial_core_client, started.json()["model_name"])["status"]
        == "complete"
    )
    view = binomial_core_client.get("/api/model-plan").json()
    assert view["stage_statuses"][0]["state"] == "fitted"
    assert view["fit_refs"][0]["model_id"] == "main"


def test_reviewed_candidate_cannot_be_kept_after_other_candidate_changes_source(
    core_client,
):
    source_name = _fit_main(core_client, sequential=False)
    assert not core_client.get(f"/api/results/{source_name}").json().get("stack")
    first = _new_candidate(core_client)
    second = _new_candidate(core_client)
    _review(core_client, first)
    _review(core_client, second)
    first_id = str(first["candidate_id"])
    accepted = core_client.post(
        f"/api/core-models/main/interactions/{first_id}/accept",
        json={**_revision(core_client), "candidate_id": first_id},
    )
    assert accepted.status_code == 200, accepted.text
    second_id = str(second["candidate_id"])
    stale = core_client.post(
        f"/api/core-models/main/interactions/{second_id}/accept",
        json={**_revision(core_client), "candidate_id": second_id},
    )
    assert stale.status_code == 422
    assert "source changed" in stale.text.lower()


def test_remove_last_pair_keeps_the_earlier_immutable_checkpoint(core_client):
    _fit_main(core_client, sequential=False)
    first = _keep(core_client, _new_candidate(core_client))
    second = _keep(core_client, _new_candidate(core_client, b="Vehicle"))
    before = core_client.get("/api/model-plan").json()
    assert (
        before["fit_refs"][0]["through_component_id"] == second["through_component_id"]
    )

    removed = core_client.post(
        f"/api/core-models/main/interactions/{second['through_component_id']}/remove",
        json=_revision(core_client),
    )
    assert removed.status_code == 200, removed.text
    after = core_client.get("/api/model-plan").json()
    assert after["fit_refs"][0]["through_component_id"] == first["through_component_id"]
    assert after["fit_refs"][0]["model_name"] == first["model_name"]
    # The surviving P1 checkpoint is a complete model, so it is immediately
    # usable; EasyGLM must not refit its main effects merely because P2 went.
    assert after["stage_statuses"][0]["state"] == "fitted"


def test_reorder_pairs_reuses_frozen_core_then_refits_only_pair_suffix(core_client):
    main_name = _fit_main(core_client, sequential=False)
    first = _keep(core_client, _new_candidate(core_client))
    second = _keep(core_client, _new_candidate(core_client, b="Vehicle"))
    moved = core_client.post(
        f"/api/core-models/main/interactions/{second['through_component_id']}/move",
        json={
            **_revision(core_client),
            "before_component_id": first["through_component_id"],
        },
    )
    assert moved.status_code == 200, moved.text
    plan = core_client.get("/api/model-plan").json()["plan"]
    # The checkpoint through main effects survives; only pair ordering is new.
    assert plan["fit_refs"][0]["through_component_id"] == "main"
    assert plan["fit_refs"][0]["model_name"] == main_name
    started = core_client.post("/api/model-plan/fit-next", json=_revision(core_client))
    assert started.status_code == 202, started.text
    assert _wait(core_client, started.json()["model_name"])["status"] == "complete"
    accepted = core_client.get("/api/model-plan").json()
    assert accepted["stage_statuses"][0]["state"] == "fitted"


def test_pair_amendment_preview_is_read_only_and_apply_keeps_partial_prefix(
    core_client,
):
    _fit_main(core_client, sequential=False)
    first = _keep(core_client, _new_candidate(core_client))
    second = _keep(core_client, _new_candidate(core_client, b="Vehicle"))
    before = core_client.get("/api/model-plan").json()
    pair_id = str(first["through_component_id"])
    preview = core_client.post(
        f"/api/core-models/main/interactions/{pair_id}/preview",
        json={
            **_revision(core_client),
            "operation": "round",
            "decimals": 2,
            "fit_id": second["fit_id"],
        },
    )
    assert preview.status_code == 200, preview.text
    assert core_client.get("/api/model-plan").json()["fit_refs"] == before["fit_refs"]

    applied = core_client.post(
        f"/api/core-models/main/interactions/{pair_id}/apply",
        json={
            **_revision(core_client),
            "operation": "round",
            "decimals": 2,
            "fit_id": second["fit_id"],
            "new_name": "Main effects — amended P1",
        },
    )
    assert applied.status_code == 200, applied.text
    view = core_client.get("/api/model-plan").json()
    assert view["plan"]["fit_refs"][0]["through_component_id"] == pair_id
    assert view["plan"]["fit_refs"][0]["model_name"] == "Main effects — amended P1"
    assert view["stage_statuses"][0]["state"] == "needs_refit"
    assert view["stage_statuses"][0]["fit"]["through_component_id"] == pair_id
    assert view["next_stage_id"] == "main"
    interaction = core_client.get("/api/core-models/main/interactions").json()
    states = {
        item["component_id"]: item["state"] for item in interaction["pair_statuses"]
    }
    assert states[pair_id] == "fitted"
    assert states[str(second["through_component_id"])] == "needs_refit"
    stored_pair = view["plan"]["models"][0]["pair_corrections"][0]
    assert stored_pair["amendments"][-1]["operation"] == "round"
    # P2 has been invalidated. Its next fit starts from the amended P1 scorer,
    # rather than rebuilding the preceding main effect or using old P1 values.
    started = core_client.post("/api/model-plan/fit-next", json=_revision(core_client))
    assert started.status_code == 202, started.text
    assert _wait(core_client, started.json()["model_name"])["status"] == "complete"


def test_partial_suffix_rejects_a_duplicate_fit_and_allows_retry_after_failure(
    core_client, monkeypatch
):
    _fit_main(core_client, sequential=False)
    first = _keep(core_client, _new_candidate(core_client))
    second = _keep(core_client, _new_candidate(core_client, b="Vehicle"))
    pair_id = str(first["through_component_id"])
    applied = core_client.post(
        f"/api/core-models/main/interactions/{pair_id}/apply",
        json={
            **_revision(core_client),
            "operation": "round",
            "decimals": 2,
            "fit_id": second["fit_id"],
            "new_name": "Main effects — held P1",
        },
    )
    assert applied.status_code == 200, applied.text

    held: list[dict[str, object]] = []

    def hold_start(self, project, raw, name, **kwargs):
        job = {
            "id": f"held-{len(held)}",
            "name": name,
            "status": "queued",
            "message": "Held by lifecycle test",
            "cancel": False,
            "applicable": False,
            "key": model_key(project, name),
            "elapsed": 0.0,
            "process": None,
        }
        self.jobs[name] = job
        self.active = job
        held.append(job)
        return job

    monkeypatch.setattr(FitJobs, "start", hold_start)
    started = core_client.post("/api/model-plan/fit-next", json=_revision(core_client))
    assert started.status_code == 202, started.text
    view = core_client.get("/api/model-plan").json()
    assert view["stage_statuses"][0]["state"] == "fitting"
    assert view["stage_statuses"][0]["fit"]["through_component_id"] == pair_id
    duplicate = core_client.post(
        "/api/model-plan/fit-next", json=_revision(core_client)
    )
    assert duplicate.status_code == 422
    assert "already fitting" in duplicate.text.lower()

    # A failed/cancelled suffix leaves the accepted P1 reference intact and
    # makes the same one remaining table retryable.
    held[0]["status"] = "failed"
    held[0]["message"] = "Stopped"
    view = core_client.get("/api/model-plan").json()
    assert view["stage_statuses"][0]["state"] == "needs_refit"
    retry = core_client.post("/api/model-plan/fit-next", json=_revision(core_client))
    assert retry.status_code == 202, retry.text


def test_v2_adopted_root_is_server_minted_and_fits_the_first_new_model(core_client):
    source_name = _fit_main(core_client, sequential=False)
    adopted_plan = _plan(sequential=False)
    adopted_plan["models"] = [
        {
            "model_id": "after-adopt",
            "name": "After adopted scorer",
            "core": _core("after-adopt", "After adopted scorer", "Region"),
            "pair_corrections": [],
        }
    ]
    adopted = core_client.post(
        "/api/model-plan",
        json={
            **_revision(core_client),
            "operation": "adopt",
            "adopt_model_name": source_name,
            "plan": adopted_plan,
        },
    )
    assert adopted.status_code == 200, adopted.text
    view = adopted.json()
    assert view["plan"]["adopted_root"]["model_name"] == source_name
    assert view["fit_refs"] == []
    child = core_client.post("/api/model-plan/fit-next", json=_revision(core_client))
    assert child.status_code == 202, child.text
    assert _wait(core_client, child.json()["model_name"])["status"] == "complete"


def test_v2_adopted_root_cannot_change_the_source_family_contract(core_client):
    source_name = _fit_main(core_client, sequential=False)
    incompatible = _plan(sequential=False)
    incompatible["family"] = "gaussian"
    response = core_client.post(
        "/api/model-plan",
        json={
            **_revision(core_client),
            "operation": "adopt",
            "adopt_model_name": source_name,
            "plan": incompatible,
        },
    )
    assert response.status_code == 422
    assert "family" in response.text.lower()


def test_core_residual_search_uses_the_accepted_training_checkpoint(core_client):
    source_name = _fit_main(core_client, sequential=False)
    response = core_client.post(
        "/api/core-models/main/search",
        json={**_revision(core_client), "action": "interactions", "top": 5},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["subset"] == "train"
    assert result["source_model_name"] == source_name
    assert result["source"] == "Complete accepted scorer on training rows"
