"""Explore uses applied training data, coherent model context and cached rates."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from easy_glm.desktop import exploration
from easy_glm.desktop.server import create_app
from easy_glm.workflow import Project, univariate
from easy_glm.workflow.project import Derived, ModelConfig, Recode


@contextmanager
def session(project, raw):
    with TestClient(
        create_app(project, raw, port=8780), base_url="http://127.0.0.1:8780"
    ) as client:
        client.headers["x-easyglm-token"] = client.get("/api/session").json()["token"]
        yield client


def read(client, **params):
    response = client.get("/api/explore", params=params)
    assert response.status_code == 200, response.text
    result = response.json()
    json.dumps(result, allow_nan=False)
    return result


def project_for(*, weighted=True):
    project = Project(name="Training exploration")
    project.data.roles = {"target": "target", "x": "predictor"}
    if weighted:
        project.data.roles["weight"] = "weight"
    return project


@pytest.mark.parametrize(
    "weighted,divide,target,expected,label",
    [
        (True, True, [2.0, 9.0, 1e9], 2.75, "target / weight"),
        (True, False, [2.0, 3.0, 1e9], 2.75, "Weighted mean target"),
        (False, False, [2.0, 3.0, 1e9], 2.5, "Mean target"),
    ],
)
def test_observed_rates_match_model_scale_and_exclude_holdout(
    weighted, divide, target, expected, label
):
    project = project_for(weighted=weighted)
    project.new_model("chosen", divide_target_by_weight=divide)
    frame = pl.DataFrame(
        {
            "x": ["A", "A", "HOLDOUT"],
            "target": target,
            "weight": [1, 3, 500],
            "traintest": [1, 1, 0],
        }
    )
    before = deepcopy(project.to_dict())
    with session(project, frame) as client:
        result = read(client, column="x")
        assert result["model"] == "chosen"
        assert result["target"] == "target"
        assert result["weight"] == ("weight" if weighted else None)
        assert result["divide_target_by_weight"] == divide
        assert result["rows"] == result["training_rows"] == 2
        assert result["table"] == [
            {
                "label": "A",
                "rate": expected,
                "exposure": 4 if weighted else 2,
                "share": 1,
                "order": 0,
            }
        ]
        assert result["rate_label"] == label
        assert result["exposure_label"] == ("weight" if weighted else "Rows")
        assert client.get("/api/project").json() == before
        assert client.get("/api/jobs").json() == {}
    assert project.to_dict() == before


def test_champion_and_explicit_model_use_their_own_target_weight_and_divide():
    project = project_for()
    project.models = {
        "unfinished": ModelConfig(),
        "Count": ModelConfig(
            target="count", weight="expo", divide_target_by_weight=True
        ),
        "Probability": ModelConfig(
            family="binomial", target="proportion", weight="trials"
        ),
    }
    project.champion = "Probability"
    frame = pl.DataFrame(
        {
            "x": ["A", "A"],
            "target": [999, 999],
            "weight": [999, 999],
            "count": [2, 9],
            "expo": [1, 3],
            "proportion": [0.2, 0.6],
            "trials": [10, 30],
            "traintest": [1, 1],
        }
    )
    with session(project, frame) as client:
        default = read(client, column="x")
        assert default["model"] == "Probability"
        assert default["target"] == "proportion" and default["weight"] == "trials"
        assert default["table"][0]["rate"] == 0.5
        assert default["table"][0]["exposure"] == 40
        chosen = read(client, column="x", model="Count")
        assert chosen["table"][0]["rate"] == 2.75
        assert [item["name"] for item in chosen["models"]] == ["Count", "Probability"]
    project.champion = "unfinished"
    with session(project, frame) as client:
        assert read(client, column="x")["model"] == "Count"


def test_project_roles_work_without_a_model_and_missing_target_keeps_exposure():
    frame = pl.DataFrame(
        {"x": ["A", "A"], "target": [2, 9], "weight": [1, 3], "traintest": [1, 1]}
    )
    project = project_for()
    with session(project, frame) as client:
        result = read(client, column="x")
        assert result["model"] is None and result["models"] == []
        assert result["divide_target_by_weight"]
        assert result["table"][0]["rate"] == 2.75
    project.data.roles.pop("target")
    with session(project, frame) as client:
        result = read(client, column="x")
        assert result["target"] is None
        assert result["table"][0]["rate"] is None
        assert result["table"][0]["exposure"] == 4


def test_default_prefers_model_then_project_predictors_but_keeps_identifiers():
    project = project_for()
    project.data.roles["ID"] = "id"
    project.new_model("selected", predictors=["removed", "age", "x"])
    raw = pl.DataFrame(
        {
            "ID": [11, 12],
            "x": ["A", "B"],
            "age": [20, 40],
            "target": [2, 9],
            "weight": [1, 3],
            "traintest": [1, 1],
        }
    )
    with session(project, raw) as client:
        assert read(client)["column"] == "age"
        assert read(client, column="removed")["column"] == "age"
        assert read(client, column="ID")["column"] == "ID"
    project.models.clear()
    with session(project, raw) as client:
        assert read(client)["column"] == "x"
    project.data.roles["x"] = "ignore"
    with session(project, raw) as client:
        assert read(client)["column"] == "ID"


def test_applied_renames_recodes_derived_types_filters_and_custom_split():
    project = project_for()
    project.data.roles["unused"] = "ignore"
    project.data.renames = {"raw_age": "Age"}
    project.data.types = {"Age": "numeric", "code": "categorical"}
    project.data.recodes = {"x": Recode(mapping={"old": "new"})}
    project.data.derived = [Derived("TwiceAge", "pl.col('Age') * 2")]
    project.data.filters = ["pl.col('Age') < 50"]
    project.data.split.train_value = "learn"
    raw = pl.DataFrame(
        {
            "raw_age": ["20", "30", "60", "40"],
            "x": ["old"] * 4,
            "code": [1, 2, 3, 4],
            "unused": [1, 2, 3, 4],
            "target": [2, 9, 1000, 10000],
            "weight": [1, 3, 1, 1],
            "traintest": ["learn", "learn", "learn", "test"],
        }
    )
    with session(project, raw) as client:
        result = read(client, column="x")
        assert result["training_rows"] == 2
        assert result["table"][0]["label"] == "new"
        assert result["table"][0]["rate"] == 2.75
        columns = {c["name"]: c["kind"] for c in result["columns"]}
        assert columns == {
            "Age": "numeric",
            "x": "categorical",
            "code": "categorical",
            "unused": "numeric",
            "TwiceAge": "numeric",
        }
        assert read(client, column="TwiceAge")["kind"] == "numeric"
        assert read(client, column="code")["kind"] == "categorical"
        assert read(client, column="target")["column"] == "x"
        assert read(client, column="removed")["column"] == "x"


def test_sampling_uses_training_rows_seed_and_full_preparation():
    project = project_for()
    project.data.sample_rows = 17
    project.data.sample_seed = 19
    raw = pl.DataFrame(
        {
            "x": np.arange(140, dtype=float),
            "target": np.arange(140, dtype=float) ** 2,
            "weight": np.arange(140, dtype=float) + 1,
            "traintest": [1] * 100 + [0] * 40,
        }
    )
    sampled = raw.head(100).sample(n=17, seed=19)
    expected = univariate(
        sampled,
        "x",
        target="target",
        weight="weight",
        divide_target_by_weight=True,
        n_bins=5,
    )
    with session(project, raw) as client:
        result = read(client, column="x", n_bins=5)
        assert result["rows"] == 17 and result["training_rows"] == 100
        assert result["sampled"]
        assert result["table"] == expected["table"].to_dicts()
    changed_holdout = raw.with_columns(
        pl.when(pl.col("traintest") == 0).then(1e12).otherwise(pl.col("x")).alias("x"),
        pl.when(pl.col("traintest") == 0)
        .then(1e12)
        .otherwise(pl.col("target"))
        .alias("target"),
    )
    with session(project, changed_holdout) as client:
        assert read(client, column="x", n_bins=5)["table"] == result["table"]


def test_null_variables_outcomes_and_zero_exposure_are_json_safe():
    project = project_for()
    raw = pl.DataFrame(
        {
            "x": [None] * 4,
            "numeric_null": pl.Series([None, np.nan, None, np.nan], dtype=pl.Float64),
            "target": [None, None, 3.0, np.inf],
            "weight": [1, 1, 0, 1],
            "traintest": [1] * 4,
        }
    )
    with session(project, raw) as client:
        for name, kind in [("x", "categorical"), ("numeric_null", "numeric")]:
            result = read(client, column=name)
            assert result["kind"] == kind and result["null_share"] == 1
            assert result["n_unique"] == 1 and len(result["table"]) == 1
            assert result["table"][0]["rate"] is None
            assert result["table"][0]["exposure"] == 3
            assert result["rate_excluded_rows"] == 3


def test_cache_reuses_training_sample_and_refreshes_after_apply_and_replacement(
    monkeypatch, tmp_path
):
    calls = {"prepare": 0, "univariate": 0}
    for name in calls:
        original = getattr(exploration, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(exploration, name, counted)
    raw = pl.DataFrame(
        {
            "x": [1, 2],
            "other": [3, 4],
            "target": [2, 9],
            "weight": [1, 3],
            "traintest": [1, 1],
        }
    )
    with session(project_for(), raw) as client:
        assert read(client, column="x") == read(client, column="x")
        assert calls == {"prepare": 1, "univariate": 1}
        read(client, column="other")
        read(client, column="x", n_bins=5)
        assert calls == {"prepare": 1, "univariate": 3}
        state = client.get("/api/variables").json()
        state["setup"]["types"] = {"categorical": ["x"]}
        response = client.post(
            "/api/variables/apply",
            json={k: state[k] for k in ("session_id", "revision", "setup")},
        )
        assert response.status_code == 200, response.text
        result = read(client, column="x")
        assert result["revision"] == 1 and result["kind"] == "categorical"
        assert calls == {"prepare": 2, "univariate": 4}
        source = tmp_path / "new.csv"
        pl.DataFrame({"fresh": [1, 2, 3, 4]}).write_csv(source)
        response = client.post(
            "/api/project/open",
            json={
                "session_id": result["session_id"],
                "revision": result["revision"],
                "kind": "data",
                "path": str(source),
            },
        )
        assert response.status_code == 200, response.text
        client.headers["x-easyglm-token"] = client.get("/api/session").json()["token"]
        fresh = read(client)
        assert fresh["project_id"] != result["project_id"]
        assert fresh["column"] == "fresh" and fresh["target"] is None
        assert calls == {"prepare": 3, "univariate": 5}


def test_inflight_chart_never_returns_old_revision_as_current(monkeypatch):
    started, release = threading.Event(), threading.Event()
    original = exploration.univariate

    def held(*args, **kwargs):
        started.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(exploration, "univariate", held)
    raw = pl.DataFrame(
        {"x": [1, 2], "target": [2, 9], "weight": [1, 3], "traintest": [1, 1]}
    )
    with (
        session(project_for(), raw) as client,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        future = pool.submit(client.get, "/api/explore?column=x")
        try:
            assert started.wait(5)
            assert client.get("/health").status_code == 200
            state = client.get("/api/variables").json()
            state["setup"]["types"] = {"categorical": ["x"]}
            response = client.post(
                "/api/variables/apply",
                json={k: state[k] for k in ("session_id", "revision", "setup")},
            )
            assert response.status_code == 200
        finally:
            release.set()
        assert future.result(timeout=5).status_code == 409
        assert read(client, column="x")["revision"] == 1


@pytest.mark.parametrize("invalid", [None, -1.0, np.nan, np.inf])
def test_invalid_weights_are_actionable_without_mutating_data(invalid):
    raw = pl.DataFrame(
        {"x": [1, 2], "target": [2, 9], "weight": [1.0, invalid], "traintest": [1, 1]}
    )
    with session(project_for(), raw) as client:
        before = client.get("/api/project").json()
        response = client.get("/api/explore?column=x")
        assert response.status_code == 422
        assert "Weight column" in response.text
        assert client.get("/api/project").json() == before


def test_bad_parameters_and_old_plot_endpoint():
    raw = pl.DataFrame(
        {"x": [1, 2], "target": [2, 9], "weight": [1, 3], "traintest": [1, 1]}
    )
    with session(project_for(), raw) as client:
        for params in [{"n_bins": 4}, {"n_bins": 51}, {"model": "missing"}]:
            assert client.get("/api/explore", params=params).status_code == 422
        assert client.get("/api/plot?column=x").status_code == 200
