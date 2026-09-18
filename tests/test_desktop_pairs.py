"""Desktop contracts for ordered pair stages and complete table comparisons."""

from __future__ import annotations

import time
from copy import deepcopy

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from easy_glm.desktop.jobs import FitJobs, _first_pair_refit, model_key
from easy_glm.desktop.modeling import ModelEdit, edit_model
from easy_glm.desktop.server import create_app
from easy_glm.engine.models import (
    FromToRow,
    PairCellRow,
    PairTableConfig,
    VariableConfig,
)
from easy_glm.engine.rate_model import RateModel
from easy_glm.workflow.diagnostics import rate_model_diff
from easy_glm.workflow.project import (
    Adjustment,
    ModelConfig,
    PairCandidateConfig,
    PairStageConfig,
    Penalty,
    Project,
    TableSnapshot,
    VariableDesign,
)
from easy_glm.workflow.variables import apply_roles_grid


def _project() -> Project:
    project = Project(name="pairs")
    project.data.roles = {"Claims": "target", "Age": "predictor", "Region": "predictor"}
    project.models["Frequency"] = ModelConfig(
        target="Claims",
        predictors=["Age"],
        penalty=project.models.get("Frequency", ModelConfig()).penalty,
        pair_stages=[
            PairStageConfig("first", "Age", "Region"),
            PairStageConfig("second", "Region", "Age"),
        ],
    )
    return project


def test_model_editor_accepts_ordered_pair_stage_config_and_rejects_malformed_candidates() -> (
    None
):
    project = _project()
    project.models["Frequency"].pair_stages = []
    fields = {
        "pair_stages": [
            {
                "stage_id": "pair-1",
                "a": "Age",
                "b": "Region",
                "candidates": [
                    {
                        "depth": 2,
                        "iterations": 60,
                        "learning_rate": 0.08,
                        "l2_leaf_reg": 3.0,
                    }
                ],
            }
        ]
    }
    edited = edit_model(
        project, ModelEdit(session_id="s", revision=0, name="Frequency", fields=fields)
    )
    stage = edited.models["Frequency"].pair_stages[0]
    assert stage.stage_id == "pair-1"
    assert stage.candidates[0].depth == 2
    assert stage.min_weight_share == 0.001
    assert project.models["Frequency"].pair_stages == []


def test_empty_sequential_model_keeps_method_after_roundtrip() -> None:
    project = _project()
    project.models["Frequency"].pair_stages.clear()
    edited = edit_model(
        project,
        ModelEdit(
            session_id="s",
            revision=0,
            name="Frequency",
            fields={"pair_method": "sequential_catboost"},
        ),
    )
    saved = edited.to_dict()
    assert saved["version"] == 3
    assert saved["models"]["Frequency"]["pair_method"] == "sequential_catboost"
    reloaded = Project.from_dict(saved)
    assert reloaded.models["Frequency"].pair_method == "sequential_catboost"
    assert not reloaded.models["Frequency"].pair_stages


def test_pair_fold_count_requires_integer_five() -> None:
    project = _project()
    project.models["Frequency"].pair_stages = [
        PairStageConfig(
            "first",
            "Age",
            "Region",
            candidates=[PairCandidateConfig(2, 10, 0.08, 3.0)],
            cv_folds=5.0,
        )
    ]
    assert any(
        "cv_folds must be 5" in problem for problem in project.validate("Frequency")
    )


def test_main_stage_identity_is_reserved() -> None:
    project = _project()
    project.models["Frequency"].pair_stages = [
        PairStageConfig(
            "main",
            "Age",
            "Region",
            candidates=[PairCandidateConfig(2, 10, 0.08, 3.0)],
        )
    ]
    assert any("reserved" in problem for problem in project.validate("Frequency"))


def test_unassign_pair_only_predictor_cleans_stage_edits_and_snapshot() -> None:
    project = _project()
    config = project.models["Frequency"]
    config.pair_stages = [PairStageConfig("first", "Age", "Region")]
    config.predictors = ["Age"]
    edit = Adjustment(
        "first",
        "low",
        "low",
        1.2,
        from_b="N",
        to_b="N",
        cell=True,
        stage_id="first",
        axis_a_row=0,
        axis_b_row=0,
    )
    config.adjustments = [edit]
    config.snapshots = [TableSnapshot("saved", adjustments=[deepcopy(edit)])]
    changed, notices = apply_roles_grid(
        project,
        ["Claims", "Age", "Region"],
        [{"column": "Region", "role": "unassigned", "rename to": "", "type": "auto"}],
    )
    assert changed
    assert not config.pair_stages
    assert not config.adjustments
    assert not config.snapshots[0].adjustments
    assert any("Pair stage(s) first removed" in message for _, message in notices)
    assert not project.validate("Frequency")


def test_rename_pair_only_parent_keeps_stage_id_and_cell_edit() -> None:
    project = _project()
    config = project.models["Frequency"]
    config.pair_stages = [PairStageConfig("first", "Age", "Region")]
    config.adjustments = [
        Adjustment(
            "first",
            "low",
            "low",
            1.2,
            from_b="N",
            to_b="N",
            cell=True,
            stage_id="first",
            axis_a_row=0,
            axis_b_row=0,
        )
    ]
    changed, _ = apply_roles_grid(
        project,
        ["Claims", "Age", "Region"],
        [
            {
                "column": "Region",
                "role": "predictor",
                "rename to": "Area",
                "type": "auto",
            }
        ],
    )
    assert changed
    assert config.pair_stages[0].b == "Area"
    assert config.pair_stages[0].stage_id == "first"
    assert config.adjustments[0].stage_id == "first"


def test_variables_api_unassign_pair_only_parent_drops_stage_with_notice() -> None:
    project = _project()
    project.models["Frequency"].pair_stages = [
        PairStageConfig("first", "Age", "Region")
    ]
    raw = pl.DataFrame(
        {
            "Claims": [0, 1, 0, 2, 1, 0],
            "Age": [20, 30, 40, 50, 60, 70],
            "Region": ["N", "S", "N", "S", "N", "S"],
        }
    )
    with TestClient(
        create_app(project, raw, port=8789), base_url="http://127.0.0.1:8789"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        snapshot = client.get("/api/variables").json()
        setup = snapshot["setup"]
        setup["roles"]["predictor"].remove("Region")
        setup["roles"]["unassigned"].append("Region")
        revision = {key: snapshot[key] for key in ("session_id", "revision")}
        preview = client.post(
            "/api/variables/preview", json={**revision, "setup": setup}
        )
        assert preview.status_code == 200, preview.text
        assert any(
            "Pair stage(s) first removed" in message
            for _, message in preview.json()["notices"]
        )
        applied = client.post("/api/variables/apply", json={**revision, "setup": setup})
        assert applied.status_code == 200, applied.text
        saved = client.get("/api/project").json()
        assert not saved["models"]["Frequency"]["pair_stages"]
        assert "Region" not in saved["data"]["roles"]


def test_stage_cards_mark_only_downstream_stage_stale_after_cell_edit() -> None:
    project = _project()
    # The status layer accepts the stored recipe and artifacts; model validation
    # independently rejects a reversed duplicate pair before fitting.
    jobs = FitJobs()
    try:
        published = {
            "status": "complete",
            "project": deepcopy(project),
            "result": {
                "pair_stages": [
                    {"stage_id": "first", "status": "up_to_date", "dimensions": [2, 2]},
                    {
                        "stage_id": "second",
                        "status": "up_to_date",
                        "dimensions": [2, 2],
                    },
                ]
            },
        }
        jobs.last_complete["Frequency"] = published
        project.models["Frequency"].adjustments.append(
            Adjustment(
                variable="first",
                from_="A",
                to_="A",
                relativity=1.2,
                from_b="R",
                to_b="R",
                cell=True,
                stage_id="first",
                axis_a_row=0,
                axis_b_row=0,
            )
        )
        cards = jobs.stage_statuses(project)["Frequency"]
        assert [card["status"] for card in cards] == [
            "up_to_date",
            "up_to_date",
            "needs_refitting",
        ]
        assert cards[2]["baseline"] == ["main", "first"]
    finally:
        jobs.close()


def test_completed_prefix_survives_append_invalidation_and_failed_suffix() -> None:
    project = _project()
    project.models["Frequency"].pair_stages.pop()
    jobs = FitJobs()
    try:
        published = {
            "id": "old",
            "name": "Frequency",
            "key": model_key(project, "Frequency"),
            "status": "complete",
            "message": "Complete",
            "elapsed": 1.0,
            "project": deepcopy(project),
            "result": {"pair_stages": [{"stage_id": "first", "status": "up_to_date"}]},
        }
        jobs.jobs["Frequency"] = published
        jobs.remember_complete("Frequency")
        project.models["Frequency"].pair_stages.append(
            PairStageConfig("new", "Age", "Other")
        )
        jobs.invalidate(project)
        assert jobs.jobs["Frequency"]["status"] == "stale"
        assert jobs.last_complete["Frequency"]["status"] == "complete"
        assert (
            jobs.last_complete["Frequency"]["result"]["pair_stages"][0]["stage_id"]
            == "first"
        )
        assert [
            card["status"] for card in jobs.stage_statuses(project)["Frequency"]
        ] == ["up_to_date", "up_to_date", "needs_refitting"]
        jobs.jobs["Frequency"] = {
            "status": "failed",
            "message": "Later stage failed",
            "progress": {"stage_number": 3},
        }
        assert [
            card["status"] for card in jobs.stage_statuses(project)["Frequency"]
        ] == ["up_to_date", "up_to_date", "failed"]
    finally:
        jobs.close()


def test_pair_only_design_change_refits_its_suffix_and_preserves_upstream_edit() -> (
    None
):
    project = Project(name="independent pair parents")
    project.data.roles = {
        "Claims": "target",
        "Main": "predictor",
        "A": "predictor",
        "B": "predictor",
        "C": "predictor",
    }
    config = ModelConfig(
        target="Claims",
        predictors=["Main"],
        penalty=Penalty(alpha=0.1),
        pair_stages=[
            PairStageConfig("ab", "A", "B"),
            PairStageConfig("bc", "B", "C"),
        ],
    )
    config.adjustments = [
        Adjustment("ab", "A", "A", 1.2, "B", "B", True, "ab", 0, 0),
        Adjustment("bc", "B", "B", 1.3, "C", "C", True, "bc", 0, 0),
    ]
    project.models["m"] = config
    previous = deepcopy(project)
    project.design.variables["C"] = VariableDesign(kind="step", knots=[25.0])
    assert _first_pair_refit(project, previous, "m") == 1
    jobs = FitJobs()
    try:
        jobs.last_complete["m"] = {
            "status": "complete",
            "project": previous,
            "result": {
                "pair_stages": [
                    {"stage_id": "ab", "status": "up_to_date"},
                    {"stage_id": "bc", "status": "up_to_date"},
                ]
            },
        }
        assert [card["status"] for card in jobs.stage_statuses(project)["m"]] == [
            "up_to_date",
            "up_to_date",
            "needs_refitting",
        ]
        preview = jobs.refit_previews(project)["m"]
        assert preview["stage_ids"] == ["bc"]
        assert [a["stage_id"] for a in preview["cleared_adjustments"]] == ["bc"]
    finally:
        jobs.close()


def _pair_model(stage_id: str, parents: tuple[str, str], value: float) -> RateModel:
    axes = tuple(
        VariableConfig(
            "categorical",
            [FromToRow(level, level, 1.0), FromToRow(None, None, 1.0)],
        )
        for level in (
            "adult" if parents[0] == "Age" else "north",
            "adult" if parents[1] == "Age" else "north",
        )
    )
    cells = [
        PairCellRow(i, j, value if (i, j) == (0, 0) else 1.0)
        for i in range(2)
        for j in range(2)
    ]
    return RateModel(
        1.0, {}, pair_tables=[PairTableConfig(stage_id, parents, axes, cells)]
    )


def test_pair_compare_matches_parent_identity_across_stage_ids() -> None:
    first = _pair_model("old-id", ("Age", "Region"), 1.2)
    second = _pair_model("new-id", ("Region", "Age"), 1.3)
    diff = rate_model_diff(first, second, tol=0)
    assert diff.height == 1
    row = diff.row(0, named=True)
    assert row["variable"] == "Age × Region"
    assert row["kind"] == "pair stage old-id → new-id"
    assert row["status"] == "changed"


def test_fitted_pair_cell_preview_apply_and_snapshot_identity() -> None:
    rng = np.random.default_rng(21)
    age = rng.integers(18, 71, 120)
    region = rng.choice(["N", "S"], 120)
    mean = np.exp(-1.2 + 0.02 * (age - 40) + 0.3 * (region == "S"))
    raw = pl.DataFrame({"Claims": rng.poisson(mean), "Age": age, "Region": region})
    project = Project(name="pair review")
    project.data.roles = {
        "Claims": "target",
        "Age": "predictor",
        "Region": "predictor",
    }
    project.data.split.mode = "random"
    project.models["Frequency"] = ModelConfig(
        target="Claims",
        predictors=["Age"],
        penalty=Penalty(alpha=0.01, cv=None),
        pair_stages=[
            PairStageConfig(
                "pair-1",
                "Age",
                "Region",
                candidates=[PairCandidateConfig(2, 10, 0.08, 3.0)],
            )
        ],
    )
    with TestClient(
        create_app(project, raw, port=8788), base_url="http://127.0.0.1:8788"
    ) as client:
        client.headers["X-EasyGLM-Token"] = client.get("/api/session").json()["token"]
        identity = client.get("/api/workbench").json()
        revision = {key: identity[key] for key in ("session_id", "revision")}
        queued = client.post("/api/models/Frequency/fit", json=revision)
        assert queued.status_code == 202, queued.text
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            job = client.get("/api/jobs").json()["Frequency"]
            if job["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert job["status"] == "complete", job
        results = client.get("/api/results/Frequency").json()
        frozen_script = client.post(
            "/api/exports/Frequency", json={**revision, "format": "python_score"}
        )
        assert frozen_script.status_code == 200, frozen_script.text
        assert "rate_model" in frozen_script.text
        assert results["pair_stages"][0]["stage_id"] == "pair-1"
        assert results["table_index"][-1]["kind"] == "pair"
        table = client.get(
            "/api/results/Frequency/table", params={"stage_id": "pair-1"}
        ).json()
        assert table["rows"] and "fitted" in table["columns"]
        new_relativity = table["rows"][0]["relativity"] * 1.2
        preview_response = client.post(
            "/api/review/Frequency",
            json={
                **revision,
                "action": "edit",
                "stage_id": "pair-1",
                "edits": {"0": new_relativity},
            },
        )
        assert preview_response.status_code == 202, preview_response.text
        review_id = preview_response.json()["id"]
        while time.monotonic() < deadline:
            review = client.get(f"/api/reviews/{review_id}").json()
            if review["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert review["status"] == "complete", review
        assert review["can_apply"]
        assert (
            review["data"]["preview_table"]["rows"][0]["relativity"] == new_relativity
        )
        applied = client.post(f"/api/reviews/{review_id}/apply", json=revision)
        assert applied.status_code == 200, applied.text
        current = client.get("/api/project").json()["models"]["Frequency"]
        assert current["adjustments"][0]["stage_id"] == "pair-1"
        assert current["adjustments"][0]["axis_a_row"] == 0
        revision = {
            key: client.get("/api/workbench").json()[key]
            for key in ("session_id", "revision")
        }
        saved = client.post(
            "/api/review/Frequency",
            json={**revision, "action": "snapshot", "snapshot": "first pair edit"},
        )
        assert saved.status_code == 202, saved.text
        snapshot = client.get("/api/project").json()["models"]["Frequency"][
            "snapshots"
        ][0]
        assert snapshot["pair_stage_fingerprint"]

        revision = {
            key: client.get("/api/workbench").json()[key]
            for key in ("session_id", "revision")
        }
        next_preview = client.post(
            "/api/review/Frequency",
            json={
                **revision,
                "action": "edit",
                "stage_id": "pair-1",
                "edits": {"0": new_relativity * 1.2},
            },
        )
        assert next_preview.status_code == 202, next_preview.text
        next_id = next_preview.json()["id"]
        while time.monotonic() < deadline:
            next_review = client.get(f"/api/reviews/{next_id}").json()
            if next_review["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert next_review["can_apply"]
        assert (
            client.post(f"/api/reviews/{next_id}/apply", json=revision).status_code
            == 200
        )

        revision = {
            key: client.get("/api/workbench").json()[key]
            for key in ("session_id", "revision")
        }
        restore = client.post(
            "/api/review/Frequency",
            json={
                **revision,
                "action": "restore_snapshot",
                "snapshot": "first pair edit",
            },
        )
        assert restore.status_code == 202, restore.text
        restore_id = restore.json()["id"]
        while time.monotonic() < deadline:
            restore_review = client.get(f"/api/reviews/{restore_id}").json()
            if restore_review["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert restore_review["can_apply"], restore_review
        assert (
            client.post(f"/api/reviews/{restore_id}/apply", json=revision).status_code
            == 200
        )
        restored = client.get(
            "/api/results/Frequency/table", params={"stage_id": "pair-1"}
        ).json()["rows"][0]
        assert restored["relativity"] == new_relativity

        revision = {
            key: client.get("/api/workbench").json()[key]
            for key in ("session_id", "revision")
        }
        changed_recipe = client.post(
            "/api/models/save",
            json={
                **revision,
                "name": "Frequency",
                "fields": {
                    "pair_stages": [
                        {
                            "stage_id": "pair-1",
                            "a": "Age",
                            "b": "Region",
                            "candidates": [
                                {
                                    "depth": 2,
                                    "iterations": 11,
                                    "learning_rate": 0.08,
                                    "l2_leaf_reg": 3.0,
                                }
                            ],
                        }
                    ]
                },
            },
        )
        assert changed_recipe.status_code == 200, changed_recipe.text
        workbench = client.get("/api/workbench").json()
        preview = workbench["stage_refit_preview"]["Frequency"]
        assert preview["stage_ids"] == ["pair-1"]
        assert len(preview["cleared_adjustments"]) == 1
        revision = {key: workbench[key] for key in ("session_id", "revision")}
        fit_again = client.post("/api/models/Frequency/fit", json=revision)
        assert fit_again.status_code == 202, fit_again.text
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            job = client.get("/api/jobs").json()["Frequency"]
            if job["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert job["status"] == "complete", job
        assert (
            client.get("/api/project").json()["models"]["Frequency"]["adjustments"]
            == []
        )
        revision = {
            key: client.get("/api/workbench").json()[key]
            for key in ("session_id", "revision")
        }
        old_restore = client.post(
            "/api/review/Frequency",
            json={
                **revision,
                "action": "restore_snapshot",
                "snapshot": "first pair edit",
            },
        )
        assert old_restore.status_code == 422
        assert "different pair-stage axes or fitted prefix" in old_restore.text
