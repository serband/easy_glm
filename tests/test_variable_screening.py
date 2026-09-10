"""Screening flags reproducible data problems without changing model selections."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from easy_glm.workflow.project import Project
from easy_glm.workflow.screening import screen_variables


def project_for(predictors: list[str], *, weighted: bool = False) -> Project:
    project = Project(name="Predictor screening")
    project.data.roles = {"target": "target", **dict.fromkeys(predictors, "predictor")}
    if weighted:
        project.data.roles["weight"] = "weight"
    return project


def check(project: Project, frame: pl.DataFrame, **options) -> dict:
    result = screen_variables(project, frame, **options)
    # API results must remain serialisable even when a source includes NaN/null.
    json.dumps(result, allow_nan=False)
    return result


def pair(result: dict, first: str, second: str) -> dict:
    matches = [
        item
        for item in result["correlated"]
        if {item["first"], item["second"]} == {first, second}
    ]
    assert len(matches) == 1, result["correlated"]
    return matches[0]


def test_custom_training_value_excludes_holdout_from_every_check():
    rng = np.random.default_rng(18)
    count = 120
    target = rng.normal(size=count)
    train = pl.DataFrame(
        {
            "target": target,
            "copy": target,
            "inverse": -target,
            "available": rng.normal(size=count),
            "partition": ["learn"] * count,
        }
    )
    holdout = pl.DataFrame(
        {
            "target": rng.normal(size=500),
            "copy": rng.normal(size=500),
            "inverse": rng.normal(size=500),
            "available": [None] * 500,
            "partition": ["test"] * 500,
        },
        schema_overrides={"available": pl.Float64},
    )
    project = project_for(["copy", "inverse", "available"])
    project.data.split.column = "partition"
    project.data.split.train_value = "learn"

    expected = check(project, train)
    actual = check(project, pl.concat([train, holdout]))

    assert actual["training_rows"] == actual["rows"] == count
    for section in ("leakage", "correlated", "missing", "unsupported", "columns"):
        assert actual[section] == expected[section]
    assert {item["variable"] for item in actual["leakage"]} >= {"copy", "inverse"}
    assert "available" not in {item["variable"] for item in actual["missing"]}
    assert abs(pair(actual, "copy", "inverse")["association"]) == pytest.approx(1)


def test_seeded_sample_is_repeatable_and_does_not_mutate_frame_or_project():
    rng = np.random.default_rng(823)
    frame = pl.DataFrame(
        {
            "target": rng.poisson(2, 2_000),
            "x": rng.normal(size=2_000),
            "z": rng.normal(size=2_000),
            "traintest": [1] * 1_600 + [0] * 400,
        }
    )
    project = project_for(["x", "z"])
    project.new_model("Keep selection", predictors=["x"])
    before_project = deepcopy(project.to_dict())
    before_frame = frame.clone()

    first = check(project, frame, sample_rows=500, seed=71)
    second = check(project, frame, sample_rows=500, seed=71)

    assert first == second
    assert first["rows"] == 500
    assert first["training_rows"] == 1_600
    assert first["sampled"]
    assert project.to_dict() == before_project
    assert_frame_equal(frame, before_frame)


def test_raw_target_copy_is_leakage_even_when_modelling_target_per_exposure():
    rng = np.random.default_rng(522)
    target = rng.poisson(3, 600).astype(float)
    frame = pl.DataFrame(
        {
            "target": target,
            "weight": np.exp(rng.normal(0, 2, 600)),
            "copied_count": target * 7 + 11,
            "unselected_copy": target,
            "traintest": [1] * 600,
        }
    )
    project = project_for(["copied_count"], weighted=True)
    project.data.roles["unselected_copy"] = "ignore"

    result = check(project, frame, divide_target_by_weight=True)

    leakage = {item["variable"]: item for item in result["leakage"]}
    assert set(leakage) == {"copied_count"}
    assert abs(leakage["copied_count"]["association"]) == pytest.approx(1)
    assert leakage["copied_count"]["method"]
    assert leakage["copied_count"]["reason"]


def test_numeric_pairs_use_weighted_pearson_on_joint_valid_rows():
    rng = np.random.default_rng(14)
    count = 160
    x = rng.normal(size=count)
    z = 2 * x + rng.normal(scale=0.3, size=count)
    weight = rng.uniform(0.5, 3, count)
    x[::13] = np.nan
    z[::17] = np.nan
    weight[:5] = [0, -3, np.nan, 0, -1]
    # These invalid-weight outliers must not dominate the association.
    x[:5], z[:5] = 1e8, -1e8
    frame = pl.DataFrame(
        {
            "target": rng.poisson(2, count),
            "x": x,
            "z": z,
            "weight": weight,
            "traintest": [1] * count,
        }
    )
    valid = np.isfinite(x) & np.isfinite(z) & np.isfinite(weight) & (weight > 0)
    weights = weight[valid]
    centred_x = x[valid] - np.average(x[valid], weights=weights)
    centred_z = z[valid] - np.average(z[valid], weights=weights)
    expected = np.sum(weights * centred_x * centred_z) / np.sqrt(
        np.sum(weights * centred_x**2) * np.sum(weights * centred_z**2)
    )

    result = check(
        project_for(["x", "z"], weighted=True),
        frame,
        correlation_threshold=0.95,
        divide_target_by_weight=False,
    )
    actual = pair(result, "x", "z")

    assert abs(actual["association"]) == pytest.approx(abs(expected), abs=1e-12)
    assert actual["observations"] == int(valid.sum())
    assert "pearson" in actual["method"].lower()


def test_missing_share_counts_null_and_nan_with_the_threshold_inclusive():
    rng = np.random.default_rng(79)
    project = project_for(["sparse", "usable"])
    frame = pl.DataFrame(
        {
            "target": rng.normal(size=200),
            "sparse": [None] * 70 + [float("nan")] * 70 + list(range(60)),
            "usable": rng.normal(size=200),
            "traintest": [1] * 200,
        },
        schema_overrides={"sparse": pl.Float64},
    )

    result = check(project, frame, missing_threshold=0.7)

    missing = {item["variable"]: item for item in result["missing"]}
    assert set(missing) == {"sparse"}
    assert missing["sparse"]["missing_share"] == pytest.approx(0.7)
    assert missing["sparse"]["observations"] > 0


def test_categorical_associations_do_not_depend_on_level_names_or_sort_order():
    rng = np.random.default_rng(29)
    groups = np.tile(np.arange(3), 200)
    target = np.array([1.0, 8.0, 20.0])[groups] + rng.normal(scale=0.02, size=600)
    frame = pl.DataFrame(
        {
            "target": target,
            "category": np.array(["A", "B", "C"])[groups],
            "duplicate": np.array(["north", "south", "west"])[groups],
            "numeric_proxy": np.array([1.0, 8.0, 20.0])[groups],
            "traintest": [1] * 600,
        }
    )
    renamed = frame.with_columns(
        pl.col("category").replace({"A": "zebra", "B": "apple", "C": "middle"}),
        pl.col("duplicate").replace(
            {"north": "third", "south": "first", "west": "second"}
        ),
    )
    project = project_for(["category", "duplicate", "numeric_proxy"])

    original = check(project, frame)
    relabelled = check(project, renamed)

    for first, second in (("category", "duplicate"), ("category", "numeric_proxy")):
        a, b = pair(original, first, second), pair(relabelled, first, second)
        assert a["association"] == pytest.approx(b["association"], abs=1e-12)
        assert a["association"] > 0.95
        assert a["method"] and b["method"]
    for variable in ("category", "duplicate"):
        a = next(item for item in original["leakage"] if item["variable"] == variable)
        b = next(item for item in relabelled["leakage"] if item["variable"] == variable)
        assert a["association"] == pytest.approx(b["association"], abs=1e-12)


def test_unique_categorical_identifiers_are_not_reported_as_leakage():
    rng = np.random.default_rng(42)
    frame = pl.DataFrame(
        {
            "target": rng.normal(size=400),
            "policy_reference": [f"policy-{index}" for index in range(400)],
            "traintest": [1] * 400,
        }
    )

    result = check(project_for(["policy_reference"]), frame)

    assert result["leakage"] == []
    assert any(
        item["variable"] == "policy_reference" and item["reason"]
        for item in result["unsupported"]
    )


def test_constant_unsupported_and_low_support_predictors_are_explained():
    rng = np.random.default_rng(75)
    frame = pl.DataFrame(
        {
            "target": rng.normal(size=120),
            "constant": [5.0] * 120,
            "nested": [[1, 2]] * 120,
            "few_observations": [None] * 100 + list(range(20)),
            "traintest": [1] * 120,
        },
        schema_overrides={"few_observations": pl.Float64},
    )

    result = check(project_for(["constant", "nested", "few_observations"]), frame)

    unsupported = {item["variable"]: item["reason"] for item in result["unsupported"]}
    assert set(unsupported) >= {"constant", "nested", "few_observations"}
    assert all(unsupported[name] for name in ("constant", "nested", "few_observations"))
    assert result["leakage"] == result["correlated"] == []


def test_one_dominant_weight_does_not_count_as_many_independent_observations():
    rng = np.random.default_rng(483)
    target = rng.normal(size=120)
    frame = pl.DataFrame(
        {
            "target": target,
            "copy": target,
            "weight": [1e9] + [1.0] * 119,
            "traintest": [1] * 120,
        }
    )

    result = check(
        project_for(["copy"], weighted=True),
        frame,
        divide_target_by_weight=False,
    )

    assert result["leakage"] == []
    assert any(
        item["variable"] == "copy" and item["reason"] for item in result["unsupported"]
    )


def test_cancellation_stops_screening_without_mutation():
    project = project_for(["x"])
    frame = pl.DataFrame(
        {"target": range(120), "x": range(120), "traintest": [1] * 120}
    )
    before = deepcopy(project.to_dict())

    with pytest.raises(InterruptedError):
        screen_variables(project, frame, cancelled=lambda: True)

    assert project.to_dict() == before


def test_invalid_weight_categories_cannot_make_valid_categories_unscreenable():
    """Rows excluded from associations cannot increase their level cardinality."""
    project = project_for(["category", "duplicate"], weighted=True)
    frame = pl.DataFrame(
        {
            "target": [1.0, 10.0] * 150 + [1.0] * 100,
            "weight": [1.0] * 300 + [0.0] * 100,
            "category": ["a", "b"] * 150 + [f"invalid-{index}" for index in range(100)],
            "duplicate": ["x", "y"] * 150
            + [f"excluded-{index}" for index in range(100)],
            "traintest": [1] * 400,
        }
    )

    result = check(project, frame, divide_target_by_weight=False)

    assert {item["variable"] for item in result["leakage"]} == {
        "category",
        "duplicate",
    }
    association = pair(result, "category", "duplicate")
    assert association["association"] == pytest.approx(1)
    assert association["observations"] == 300
    assert result["unsupported"] == []


def test_huge_weight_on_missing_predictors_does_not_underflow_valid_pair_weights():
    """A weight outside a pair must not destroy that pair's finite moments."""
    project = project_for(["x", "z"], weighted=True)
    frame = pl.DataFrame(
        {
            "target": list(range(120)) + [0],
            "weight": [1.0] * 120 + [1e300],
            "x": list(range(120)) + [None],
            "z": list(range(120)) + [None],
            "traintest": [1] * 121,
        }
    )

    result = check(project, frame, divide_target_by_weight=False)

    association = pair(result, "x", "z")
    assert association["association"] == pytest.approx(1)
    assert association["observations"] == 120
    assert {item["variable"] for item in result["leakage"]} == {"x", "z"}
    assert result["unsupported"] == []
