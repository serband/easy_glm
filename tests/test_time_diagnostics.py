from datetime import date
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from easy_glm.desktop.review_worker import grouping, review
from easy_glm.desktop.server import create_app
from easy_glm.workflow.diagnostics import totals
from easy_glm.workflow.prep import prepare
from easy_glm.workflow.project import Project
from easy_glm.workflow.run import run_model
from easy_glm.workflow.time_diagnostics import time_diagnostics, time_values


@pytest.fixture
def problem():
    rng = np.random.default_rng(4)
    n = 600
    raw = pl.DataFrame(
        {
            "year": rng.choice([2020, 2021, 2022, 2023, 2024], n),
            "region": rng.choice(["North", "South"], n),
            "claims": rng.poisson(0.7, n),
            "weight": rng.uniform(0.2, 1.5, n),
        }
    )
    project = Project()
    project.data.roles = {
        "year": "time",
        "region": "predictor",
        "claims": "target",
        "weight": "weight",
    }
    project.data.split.mode = "random"
    config = project.new_model("m", family="poisson")
    config.penalty.alpha = 0.01
    config.penalty.cv = None
    frame = prepare(project, raw)
    run = run_model(project, frame, "m")
    return project, raw, frame, run


def test_all_rows_ordered_and_no_refit(problem):
    project, raw, frame, run = problem
    with patch("easy_glm.core.fit.fit_glm", side_effect=AssertionError("refit")):
        result = review(
            project, run, raw, {"action": "time", "n_bins": 5, "subset": "holdout"}
        )
    actual, expected, weights = totals(frame, run.config, run.fit.predict(frame))
    assert result["rows"] == frame.height == run.train_rows + run.holdout_rows
    assert [r["label"] for r in result["overall"]] == list(map(str, range(2020, 2025)))
    assert sum(r["actual"] for r in result["overall"]) == pytest.approx(actual.sum())
    assert sum(r["expected"] for r in result["overall"]) == pytest.approx(
        expected.sum()
    )
    assert sum(r["exposure"] for r in result["overall"]) == pytest.approx(weights.sum())
    assert all(
        r["ae"] == pytest.approx(r["actual"] / r["expected"]) for r in result["overall"]
    )


def test_common_factor_bands_and_period_normalisation(problem):
    project, _, frame, run = problem
    result = time_diagnostics(
        project, run, frame, variable="region", grouping=grouping(run, "region")
    )
    assert len(result["series"]) == 5
    for period in result["overall"]:
        cells = [r for r in result["cells"] if r["period"] == period["label"]]
        assert sum(r["expected"] for r in cells) == pytest.approx(period["expected"])
        assert sum(r["exposure"] for r in cells) == pytest.approx(period["exposure"])
        for cell in cells:
            if cell["ae"] is not None:
                assert cell["relative_ae"] == pytest.approx(cell["ae"] / period["ae"])


def test_numeric_ties_missing_and_short_period_count(problem):
    project, _, frame, run = problem
    times = [-1] * 300 + [0] * 299 + [None]
    frame = frame.with_columns(pl.Series("year", times))
    result = time_diagnostics(project, run, frame, n_bins=10)
    assert result["bands"] == 2
    assert result["rows"] == 599
    assert result["excluded_rows"] == 1
    assert [r["rows"] for r in result["overall"]] == [300, 299]


@pytest.mark.parametrize(
    "values", [[True, False], ["early", "late"], [1.0, float("inf")], [None, None]]
)
def test_bad_time_values(values):
    with pytest.raises(ValueError):
        time_values(pl.Series("time", values))


@pytest.mark.parametrize(
    "values", [["2025-01-31", "2024-01-31"], [date(2024, 1, 1), date(2025, 1, 1)]]
)
def test_time_refuses_text_and_dates(values):
    with pytest.raises(ValueError, match="Time must be numeric"):
        time_values(pl.Series("time", values))


def test_role_json_round_trip_and_validation(problem):
    project, raw, _, _ = problem
    project.data.roles["year"] = "ignore"
    with TestClient(
        create_app(project, raw, port=8765), base_url="http://127.0.0.1:8765"
    ) as client:
        client.headers["x-easyglm-token"] = client.get("/api/session").json()["token"]
        state = client.get("/api/variables").json()
        setup = state["setup"]
        setup["assignments"]["time"] = "year"
        setup["roles"]["ignore"].remove("year")
        body = {k: state[k] for k in ("session_id", "revision")}
        body["setup"] = setup
        response = client.post("/api/variables/apply", json=body)
        assert response.status_code == 200, response.text
        saved = Project.from_dict(client.get("/api/project").json())
        assert saved.column_with_role("time") == "year"
        state = client.get("/api/variables").json()
        state["setup"]["assignments"]["time"] = "region"
        state["setup"]["roles"]["predictor"].remove("region")
        response = client.post(
            "/api/variables/preview",
            json={k: state[k] for k in ("session_id", "revision", "setup")},
        )
        assert response.status_code == 422
        assert "Time must be numeric" in response.text


def test_uneven_year_counts_keep_each_year_when_five_are_available(problem):
    project, _, frame, run = problem
    years = [2020] * 300 + [2021] * 100 + [2022] * 100 + [2023] * 50 + [2024] * 50
    result = time_diagnostics(
        project, run, frame.with_columns(pl.Series("year", years))
    )
    assert result["bands"] == 5
    assert [r["rows"] for r in result["overall"]] == [300, 100, 100, 50, 50]


def test_original_fit_diagnostics_ignore_post_fit_table_changes(problem):
    project, _, frame, run = problem
    before = time_diagnostics(project, run, frame)
    run.rate_model.base_rate *= 2
    after = time_diagnostics(project, run, frame)
    assert before == after
