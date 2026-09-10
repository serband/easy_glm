"""Pruning draft predictors cleans only their applied and saved factor edits."""

from copy import deepcopy

import pytest

from easy_glm.workflow.project import (
    Adjustment,
    Interaction,
    ModelConfig,
    Project,
    TableSnapshot,
)
from easy_glm.workflow.variables import apply_roles_grid


def configured():
    project = Project()
    project.data.roles = dict.fromkeys(("x", "y", "z", "w"), "predictor")
    removed, kept = Interaction("x", "y"), Interaction("z", "w")
    adjustments = [
        Adjustment(name, None, 2, 1.3) for name in ("x", "y", removed.name, kept.name)
    ]
    project.models["Model"] = ModelConfig(
        predictors=["x", "y", "z", "w"],
        interactions=[removed, kept],
        monotone={"x": "increasing", "z": "decreasing"},
        adjustments=adjustments,
        base_rate_override=0.4,
        snapshots=[
            TableSnapshot(
                "Saved", adjustments=deepcopy(adjustments), base_rate_override=0.5
            )
        ],
    )
    return project


@pytest.mark.parametrize("role", ["ignore", "unassigned"])
def test_role_removal_cleans_factor_and_interaction_snapshots_after_rename(role):
    project = configured()
    rows = [
        {"column": name, "role": "predictor", "type": "auto"}
        for name in ("x", "y", "z", "w")
    ]
    row = next(row for row in rows if row["column"] == "x")
    row.update({"rename to": "Renamed", "role": role})
    changed, notices = apply_roles_grid(project, ["x", "y", "z", "w"], rows)
    assert changed and notices
    model = project.models["Model"]
    assert model.predictors == ["y", "z", "w"]
    assert [pair.name for pair in model.interactions] == ["z×w"]
    assert model.monotone == {"z": "decreasing"}
    assert [a.variable for a in model.adjustments] == ["y", "z×w"]
    assert [a.variable for a in model.snapshots[0].adjustments] == ["y", "z×w"]
    assert model.base_rate_override == 0.4
    assert model.snapshots[0].base_rate_override == 0.5
    project.apply_role_change("Renamed", "predictor")
    assert model.predictors == ["y", "z", "w"]
    assert [a.variable for a in model.adjustments] == ["y", "z×w"]


def test_preview_keeps_saved_adjustments_until_explicit_apply():
    import polars as pl
    from fastapi.testclient import TestClient

    from easy_glm.desktop.server import create_app

    project = configured()
    before = deepcopy(project.to_dict())
    raw = pl.DataFrame({name: [1, 2] for name in ("x", "y", "z", "w")})
    with TestClient(
        create_app(project, raw, port=8780), base_url="http://127.0.0.1:8780"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        state = client.get("/api/variables").json()
        setup = state["setup"]
        setup["roles"]["predictor"].remove("x")
        setup["roles"]["ignore"].append("x")
        body = {
            "session_id": state["session_id"],
            "revision": state["revision"],
            "setup": setup,
        }
        preview = client.post("/api/variables/preview", json=body)
        assert preview.status_code == 200
        assert client.get("/api/project").json() == before
        assert project.to_dict() == before
        assert client.post("/api/variables/apply", json=body).status_code == 200
        saved = client.get("/api/project").json()["models"]["Model"]
        assert [a["variable"] for a in saved["adjustments"]] == ["y", "z×w"]
        assert [a["variable"] for a in saved["snapshots"][0]["adjustments"]] == [
            "y",
            "z×w",
        ]
        assert saved["base_rate_override"] == 0.4
        assert saved["snapshots"][0]["base_rate_override"] == 0.5
