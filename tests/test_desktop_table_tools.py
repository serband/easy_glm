"""User-facing table previews agree with canonical tools and preserve the fit."""

import copy

import numpy as np
import polars as pl
import pytest
from test_desktop_models import model_session as model_session
from test_desktop_models import revision
from test_desktop_reviews import apply, fitted, review

from easy_glm.desktop.review_worker import review as worker_review
from easy_glm.engine import tooling
from easy_glm.workflow.diagnostics import expected_claims
from easy_glm.workflow.prep import prepare, train_holdout
from easy_glm.workflow.project import Adjustment, ModelConfig, Project, VariableDesign
from easy_glm.workflow.run import rebuild_rate_model, run_model


@pytest.mark.parametrize("kind", ["step", "linear", "categorical"])
@pytest.mark.parametrize(
    "action,options",
    [
        ("moving", {"window": 3, "ordered": True}),
        ("isotonic", {"direction": "increasing", "ordered": True}),
        ("isotonic", {"direction": "decreasing", "ordered": True}),
        ("cap", {"floor": 1.1}),
        ("cap", {"cap": 1.4}),
        ("cap", {"floor": 1.1, "cap": 1.4}),
        ("round", {"decimals": 1}),
        ("round", {"step": 0.25}),
    ],
)
def test_preview_modes_match_canonical_values_and_money(kind, action, options):
    rng = np.random.default_rng(9)
    x = rng.integers(1, 7, 400)
    raw = pl.DataFrame(
        {
            "x": x if kind != "categorical" else [str(v) for v in x],
            "y": rng.poisson(0.3 * x),
            "w": rng.uniform(0.3, 1.2, 400),
        }
    )
    p = Project(name="Tool preview")
    p.data.roles = {"x": "predictor", "y": "target", "w": "weight"}
    p.data.split.mode = "random"
    p.design.variables["x"] = VariableDesign(kind=kind, n_bins=5)
    p.models["Model"] = ModelConfig(
        target="y", weight="w", divide_target_by_weight=True, predictors=["x"]
    )
    p.models["Model"].penalty.alpha = 0.001
    frame = prepare(p, raw)
    run = run_model(p, frame, "Model")
    row = run.rate_model.variables["x"].table[1]
    p.models["Model"].adjustments = [Adjustment("x", row.from_, row.to_, 2.321)]
    rebuild_rate_model(p, run, frame)
    before = run.rate_model.clone()
    coefs = run.fit.coef.copy()
    functions = {
        "moving": tooling.smooth_moving_average,
        "isotonic": tooling.smooth_isotonic,
        "cap": tooling.cap_floor,
        "round": tooling.round_relativities,
    }
    expected = functions[action](before.variables["x"], "x", **options)
    predicted = tooling.preview_model(before, "x", expected.values)
    train, _ = train_holdout(frame, p.data.split)
    data = worker_review(
        copy.deepcopy(p),
        run,
        raw,
        {"action": action, "variable": "x", "subset": "holdout", "options": options},
    )
    np.testing.assert_allclose(
        [r["relativity"] for r in data["preview_table"]["rows"]],
        expected.values,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        [r["fitted"] for r in data["preview_table"]["rows"]],
        [r.relativity for r in before.variables["x"].table],
        rtol=1e-12,
    )
    assert data["after_expected"] == pytest.approx(
        expected_claims(predicted, train, p.models["Model"]), rel=1e-12
    )
    assert data["before_expected"] == pytest.approx(
        expected_claims(before, train, p.models["Model"]), rel=1e-12
    )
    np.testing.assert_array_equal(coefs, run.fit.coef)
    if kind == "linear":
        np.testing.assert_allclose(
            run.rate_model.predict(frame, exposure_col=None),
            predicted.predict(frame, exposure_col=None),
            rtol=1e-12,
        )
    if action in ("moving", "isotonic"):
        assert data["tool_details"]["log_mean_before"] == pytest.approx(
            data["tool_details"]["log_mean_after"], abs=1e-12
        )


def test_reset_one_factor_and_snapshot_lifecycle(model_session):
    client, _, _ = model_session
    fitted(client)
    before = client.get("/api/jobs").json()["Frequency"]["id"]
    apply(client, review(client, "edit", variable="Age", edits={"1": 2.1}))
    apply(client, review(client, "edit", variable="Region", edits={"1": 1.7}))
    review(client, "snapshot", snapshot="Edited")
    compared = review(
        client, "compare_snapshots", options={"left": "__fitted__", "right": "Edited"}
    )["data"]
    assert compared["tables"][0]["rows"]
    apply(client, review(client, "reset_variable", variable="Age"))
    config = client.get("/api/project").json()["models"]["Frequency"]
    assert config["adjustments"] and all(
        a["variable"] == "Region" for a in config["adjustments"]
    )
    apply(client, review(client, "restore_snapshot", variable="Age", snapshot="Edited"))
    config = client.get("/api/project").json()["models"]["Frequency"]
    assert {a["variable"] for a in config["adjustments"]} == {"Age", "Region"}
    refused = client.post(
        "/api/review/Frequency",
        json={**revision(client), "action": "delete_snapshot", "snapshot": "Edited"},
    )
    assert refused.status_code == 422
    review(client, "delete_snapshot", snapshot="Edited", options={"confirmed": True})
    assert client.get("/api/review-info/Frequency").json()["snapshots"] == []
    assert client.get("/api/jobs").json()["Frequency"]["id"] == before


def test_upgrade_restores_undo_and_redo_without_touching_fit(model_session, tmp_path):
    import json
    import shutil
    import tempfile
    from pathlib import Path

    from fastapi.testclient import TestClient

    from easy_glm.desktop.server import create_app

    client, _, raw = model_session
    fitted(client)

    def edit_state():
        cfg = client.get("/api/project").json()["models"]["Frequency"]
        return {k: cfg[k] for k in ("adjustments", "base_rate_override")}

    first = edit_state()
    apply(client, review(client, "edit", variable="Age", edits={"1": 2.1}))
    second = edit_state()
    apply(client, review(client, "edit", variable="Age", edits={"2": 1.6}))
    saved = client.get("/api/project").json()
    jobs = client.get("/api/jobs").json()
    job = jobs["Frequency"]
    source = next(Path(tempfile.gettempdir()).glob("easyglm_fits_*/" + job["id"]))
    shutil.copytree(source, tmp_path / "fits" / job["id"])
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    (tmp_path / "history.json").write_text(
        json.dumps({"Frequency": {"undo": [first, second], "redo": []}})
    )
    with TestClient(
        create_app(Project.from_dict(saved), raw, port=8781, restore_folder=tmp_path),
        base_url="http://127.0.0.1:8781",
    ) as restored:
        restored.headers["X-EasyGLM-Token"] = restored.get("/api/session").json()[
            "token"
        ]
        assert restored.get("/api/review-info/Frequency").json()["undo"]
        apply(restored, review(restored, "undo", variable="Age"))
        cfg = restored.get("/api/project").json()["models"]["Frequency"]
        assert cfg["adjustments"] == second["adjustments"]
        apply(restored, review(restored, "undo", variable="Age"))
        assert (
            restored.get("/api/project").json()["models"]["Frequency"]["adjustments"]
            == first["adjustments"]
        )
        apply(restored, review(restored, "redo", variable="Age"))
        apply(restored, review(restored, "redo", variable="Age"))
        assert restored.get("/api/project").json() == saved
        assert restored.get("/api/jobs").json()["Frequency"]["id"] == job["id"]
