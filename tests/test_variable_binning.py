"""Variables binning draft, preview, and persistence contracts."""

from __future__ import annotations

from copy import deepcopy

import polars as pl
import pytest
from fastapi.testclient import TestClient

from easy_glm.desktop.binning import apply_binning_setup, binning_preview, binning_setup
from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import (
    Adjustment,
    ModelConfig,
    Project,
    TableSnapshot,
    VariableDesign,
)


def _project() -> Project:
    project = Project(name="Binning")
    project.data.roles = {"age": "predictor", "claims": "target"}
    project.models["Frequency"] = ModelConfig(target="claims", predictors=["age"])
    return project


def _raw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "age": [-1, 0, 0, 1, 2, 3, None, 9],
            "claims": [0, 1, 0, 1, 0, 1, 0, 0],
            "traintest": [1, 1, 1, 1, 1, 1, 1, 0],
        }
    )


def test_snapshot_preserves_integer_custom_and_inactive_settings() -> None:
    p = _project()
    p.design.variables["age"] = VariableDesign(
        kind="categorical", knots="integer", n_bins=12, penalty_weight=2.0
    )
    before = deepcopy(p.design.variables["age"])
    setup = binning_setup(p, ["age", "claims"])
    assert setup["overrides"]["age"] == {"method": "integer", "fallback_bins": 12}
    apply_binning_setup(p, ["age", "claims"], setup)
    assert p.design.variables["age"] == before
    setup["overrides"] = {"claims": {"method": "cuts", "cuts": [-1, 0, 2]}}
    apply_binning_setup(p, ["age", "claims"], setup)
    assert p.design.variables["age"].kind == "categorical"
    assert p.design.variables["age"].penalty_weight == 2.0
    assert p.design.variables["age"].knots == "quantile"
    assert p.design.variables["age"].n_bins is None


@pytest.mark.parametrize(
    "override",
    [
        {"method": "quantile", "bins": 1},
        {"method": "quantile", "bins": True},
        {"method": "quantile", "bins": 10, "cuts": [0]},
        {"method": "cuts", "cuts": [0, 0]},
        {"method": "cuts", "cuts": [2, 1]},
        {"method": "cuts", "cuts": [float("inf")]},
        {"method": "integer", "bins": 10},
    ],
)
def test_invalid_overrides_are_atomic(override: dict) -> None:
    p = _project()
    original = deepcopy(p.design)
    with pytest.raises(ValueError):
        apply_binning_setup(
            p, ["age", "claims"], {"default_bins": 12, "overrides": {"age": override}}
        )
    assert p.design == original


def test_step_preview_uses_exact_right_hand_boundaries_and_missing() -> None:
    p = _project()
    apply_binning_setup(
        p,
        ["age", "claims"],
        {
            "default_bins": 20,
            "overrides": {"age": {"method": "cuts", "cuts": [0, 1, 2]}},
        },
    )
    result = binning_preview(p, _raw(), "age")
    assert result["active"] is True
    assert result["actual_bins"] == 4
    assert [row["rows"] for row in result["rows"]] == [1, 2, 1, 2]
    assert result["rows"][1]["label"] == "[0, 1)"
    assert result["missing_rows"] == 1


def test_quantile_ties_reduce_actual_bins() -> None:
    p = _project()
    p.design.defaults.n_bins = 10
    result = binning_preview(p, _raw(), "age")
    assert result["requested_bins"] == 10
    assert result["actual_bins"] < 10
    assert any("Tied" in warning for warning in result["warnings"])


def test_numeric_string_cast_and_exposure_use_prepared_training_rows() -> None:
    p = _project()
    p.data.types["age"] = "numeric"
    p.data.roles["exposure"] = "exposure"
    p.design.variables["age"] = VariableDesign(knots=[0.0])
    raw = pl.DataFrame(
        {
            "age": ["-1", "0", "bad", "2", "7"],
            "claims": [0, 0, 0, 0, 0],
            "exposure": [1.0, 2.0, 3.0, 4.0, 999.0],
            "traintest": [1, 1, 1, 1, 0],
        }
    )
    result = binning_preview(p, raw, "age")
    assert [row["rows"] for row in result["rows"]] == [1, 2]
    assert [row["exposure"] for row in result["rows"]] == [1.0, 6.0]
    assert result["missing_rows"] == 1
    assert result["training_rows"] == 4


def test_integer_fallback_is_visible() -> None:
    p = _project()
    p.design.variables["age"] = VariableDesign(knots="integer", n_bins=4)
    raw = pl.DataFrame(
        {
            "age": [0, 1, 1000, 1001, 2],
            "claims": [0, 0, 0, 0, 0],
            "traintest": [1, 1, 1, 1, 0],
        }
    )
    result = binning_preview(p, raw, "age")
    assert result["requested_bins"] == 4
    assert any("Integer cuts exceed" in warning for warning in result["warnings"])


def test_nonfinite_values_are_counted_and_encoder_assigns_infinity() -> None:
    p = _project()
    p.design.variables["age"] = VariableDesign(knots=[0.0])
    raw = pl.DataFrame(
        {
            "age": [-1.0, 0.0, float("inf"), float("-inf"), float("nan"), 2.0],
            "claims": [0, 0, 0, 0, 0, 0],
            "traintest": [1, 1, 1, 1, 1, 0],
        }
    )
    result = binning_preview(p, raw, "age")
    assert result["nonfinite_rows"] == 3
    assert result["missing_rows"] == 1
    assert [row["rows"] for row in result["rows"]] == [2, 2]


def test_linear_cuts_outside_clamp_refused_by_preview_but_legacy_fit_unchanged() -> (
    None
):
    from easy_glm.workflow.run import encoder_for

    p = _project()
    vd = VariableDesign(kind="linear", knots=[0.0, 5.0], clamp=[-1.0, 3.0])
    p.design.variables["age"] = vd
    with pytest.raises(ValueError, match="effective clamp"):
        binning_preview(p, _raw(), "age")
    assert encoder_for("age", _raw()["age"], vd, p).knots == [0.0]


def test_explicit_step_cuts_show_empty_bands_on_all_missing_training() -> None:
    p = _project()
    p.design.variables["age"] = VariableDesign(knots=[0.0, 1.0])
    raw = _raw().with_columns(pl.col("age").fill_null(0).cast(pl.Float64))
    raw = raw.with_columns(
        pl.when(pl.col("traintest") == 1)
        .then(None)
        .otherwise(pl.col("age"))
        .alias("age")
    )
    result = binning_preview(p, raw, "age")
    assert result["actual_bins"] == 3
    assert [row["rows"] for row in result["rows"]] == [0, 0, 0]
    assert result["missing_rows"] == 7


def test_empty_linear_cuts_roundtrip() -> None:
    p = _project()
    p.design.variables["age"] = VariableDesign(kind="linear", knots=[], clamp=[-1, 3])
    setting = binning_setup(p, ["age"])
    assert setting["overrides"]["age"] == {"method": "cuts", "cuts": []}
    apply_binning_setup(p, ["age"], setting)
    assert p.design.variables["age"].knots == []


def test_new_empty_step_cuts_are_refused() -> None:
    p = _project()
    with TestClient(
        create_app(p, _raw(), port=8776), base_url="http://127.0.0.1:8776"
    ) as client:
        session = client.get("/api/session").json()
        client.headers["x-easyglm-token"] = session["token"]
        state = client.get("/api/variables").json()
        setup = deepcopy(state["setup"])
        setup["binning"]["overrides"]["age"] = {"method": "cuts", "cuts": []}
        body = {"session_id": state["session_id"], "revision": 0, "setup": setup}
        response = client.post("/api/variables/preview", json=body)
        assert response.status_code == 422
        assert "cannot be empty" in response.json()["detail"]


def test_api_binning_only_preview_apply_and_legacy_preservation() -> None:
    p = _project()
    p.design.variables["age"] = VariableDesign(knots="integer", n_bins=12)
    with TestClient(
        create_app(p, _raw(), port=8773), base_url="http://127.0.0.1:8773"
    ) as client:
        session = client.get("/api/session").json()
        client.headers["x-easyglm-token"] = session["token"]
        state = client.get("/api/variables").json()
        body = {
            "session_id": state["session_id"],
            "revision": 0,
            "setup": deepcopy(state["setup"]),
        }
        body["setup"].pop("binning")
        assert client.post("/api/variables/apply", json=body).json()["revision"] == 0
        assert (
            client.get("/api/project").json()["design"]["variables"]["age"]["knots"]
            == "integer"
        )
        body["setup"]["binning"] = {
            "default_bins": 10,
            "overrides": {"age": {"method": "cuts", "cuts": [0, 1]}},
        }
        preview = client.post("/api/variables/preview", json=body)
        assert preview.status_code == 200
        assert any(row["role"] == "binning" for row in preview.json()["changes"])
        assert client.get("/api/project").json()["design"]["defaults"]["n_bins"] == 20
        data_preview = client.post(
            "/api/variables/binning-preview", json={**body, "column": "age"}
        )
        assert data_preview.status_code == 200
        assert [row["rows"] for row in data_preview.json()["rows"]] == [1, 2, 3]
        applied = client.post("/api/variables/apply", json=body)
        assert applied.status_code == 200
        assert applied.json()["revision"] == 1
        assert applied.json()["setup"]["binning"]["overrides"]["age"] == {
            "method": "cuts",
            "cuts": [0.0, 1.0],
        }
        assert (
            client.post(
                "/api/variables/binning-preview", json={**body, "column": "age"}
            ).status_code
            == 409
        )


def test_api_atomic_rename_swap_and_old_table_work_warning() -> None:
    p = _project()
    old_edit = Adjustment("age", None, 0, 1.2)
    p.models["Frequency"].adjustments = [old_edit]
    p.models["Frequency"].snapshots = [
        TableSnapshot(name="Before cuts", adjustments=[deepcopy(old_edit)])
    ]
    with TestClient(
        create_app(p, _raw(), port=8774), base_url="http://127.0.0.1:8774"
    ) as client:
        session = client.get("/api/session").json()
        client.headers["x-easyglm-token"] = session["token"]
        state = client.get("/api/variables").json()
        setup = deepcopy(state["setup"])
        setup["renames"] = {"age": "claims", "claims": "age"}
        setup["binning"] = {
            "default_bins": 20,
            "overrides": {"age": {"method": "cuts", "cuts": [-1, 0, 2]}},
        }
        body = {"session_id": state["session_id"], "revision": 0, "setup": setup}
        preview = client.post("/api/variables/preview", json=body)
        assert preview.status_code == 200
        assert any(
            "adjustments or snapshots" in text for _, text in preview.json()["notices"]
        )
        applied = client.post("/api/variables/apply", json=body)
        assert applied.status_code == 200
        saved = client.get("/api/project").json()
        assert saved["design"]["variables"]["claims"]["knots"] == [-1.0, 0.0, 2.0]
        assert saved["models"]["Frequency"]["snapshots"][0]["name"] == "Before cuts"


def test_api_refuses_new_linear_cuts_outside_explicit_clamp() -> None:
    p = _project()
    p.design.variables["age"] = VariableDesign(kind="linear", clamp=[0.0, 10.0])
    with TestClient(
        create_app(p, _raw(), port=8775), base_url="http://127.0.0.1:8775"
    ) as client:
        session = client.get("/api/session").json()
        client.headers["x-easyglm-token"] = session["token"]
        state = client.get("/api/variables").json()
        setup = deepcopy(state["setup"])
        setup["binning"]["overrides"]["age"] = {"method": "cuts", "cuts": [-20, 2]}
        body = {"session_id": state["session_id"], "revision": 0, "setup": setup}
        assert client.post("/api/variables/preview", json=body).status_code == 422
        assert client.post("/api/variables/apply", json=body).status_code == 422
        assert (
            client.get("/api/project").json()["design"]["variables"]["age"]["knots"]
            == "quantile"
        )
