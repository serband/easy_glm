"""Report summaries describe training data without affecting the model."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from easy_glm.workflow.data_summary import data_summary


def summary(frame: pl.DataFrame, **kwargs):
    result = data_summary(frame, list(frame.columns), **kwargs)
    json.dumps(result, allow_nan=False)
    return result


def test_descriptive_statistics_match_known_sample_and_histogram_covers_rows():
    frame = pl.DataFrame({"age": [1.0, 2.0, 3.0, 4.0, 5.0, None, float("nan")]})
    before = frame.clone()
    result = summary(frame)
    column = result["variables"][0]
    assert column["rows"] == 7
    assert column["missing"] == 2
    assert column["missing_pct"] == pytest.approx(200 / 7)
    assert column["unique"] == 5
    assert column["finite"] == 5
    assert column["min"] == 1
    assert column["max"] == 5
    assert column["range"] == 4
    assert column["mean"] == pytest.approx(3)
    assert column["median"] == pytest.approx(3)
    assert column["std"] == pytest.approx(np.sqrt(2.5))
    assert column["skewness"] == pytest.approx(0, abs=1e-14)
    assert column["kurtosis"] == pytest.approx(-1.2)
    assert sum(item["count"] for item in column["histogram"]) == 5
    assert result["columns_with_missing"] == 1
    assert result["missing_cells"] == 2
    assert_frame_equal(frame, before)


def test_sample_shape_statistics_match_independent_scipy_reference():
    from scipy.stats import kurtosis, skew

    values = np.array([0.0, 0.0, 0.3, 1.0, 2.0, 9.0, 15.0])
    column = summary(pl.DataFrame({"x": values}))["variables"][0]
    assert column["skewness"] == pytest.approx(skew(values, bias=False))
    assert column["kurtosis"] == pytest.approx(
        kurtosis(values, fisher=True, bias=False)
    )


def test_numeric_categorical_override_preserves_levels_and_excludes_correlation():
    frame = pl.DataFrame(
        {
            "code": [1, 2, 1, 3, 2, 1, None, 1],
            "label": ["<risk>", "b", "<risk>", "b", "c", "c", "d", None],
            "x": list(range(8)),
        }
    )
    result = summary(frame, categorical={"code"})
    code, label, _ = result["variables"]
    assert code["kind"] == "categorical"
    assert code["unique"] == 3
    assert code["histogram"] == [
        {"label": "1", "count": 4},
        {"label": "2", "count": 2},
        {"label": "3", "count": 1},
    ]
    assert label["histogram"][0] == {"label": "<risk>", "count": 2}
    assert result["correlations"]["names"] == ["x"]
    assert result["correlations"]["excluded_names"] == ["code", "label"]


def test_categorical_histogram_bounds_levels_without_losing_observations():
    values = ["popular"] * 20 + [f"level{i}" for i in range(25)] + [None] * 3
    result = summary(pl.DataFrame({"category": values}))["variables"][0]
    assert result["unique"] == 26
    assert len(result["histogram"]) == 9
    assert result["histogram"][0] == {"label": "popular", "count": 20}
    assert result["histogram"][-1] == {
        "label": "Remaining levels",
        "count": 18,
        "remaining": True,
    }
    assert sum(item["count"] for item in result["histogram"]) == 45


def test_pairwise_complete_signed_correlations_and_observation_counts():
    frame = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, None, 6.0],
            "negative": [10.0, 8.0, 6.0, None, 2.0, float("inf")],
            "partial": [3.0, None, 2.0, 5.0, 8.0, 1.0],
            "constant": [2.0] * 6,
        }
    )
    result = summary(frame)["correlations"]
    assert result["mode"] == "matrix"
    assert result["counts"][0] == [5, 3, 4, 5]
    assert result["matrix"][0][1] == pytest.approx(-1)
    expected = np.corrcoef([1, 3, 4, 6], [3, 2, 5, 1])[0, 1]
    assert result["matrix"][0][2] == pytest.approx(expected)
    assert result["matrix"][3] == [None, None, None, None]
    assert result["valid_pairs"] == 3
    assert len(result["pairs"]) == 3
    assert result["pairs"][0]["correlation"] == pytest.approx(-1)
    for left in range(4):
        for right in range(4):
            assert result["matrix"][left][right] == pytest.approx(
                result["matrix"][right][left]
            )


def test_pair_that_is_constant_only_on_overlap_has_no_correlation():
    result = summary(
        pl.DataFrame({"x": [1.0, 1.0, 9.0, None], "y": [1.0, 3.0, None, 8.0]})
    )["correlations"]
    assert result["counts"][0][1] == 2
    assert result["matrix"][0][1] is None
    assert result["valid_pairs"] == 0


@pytest.mark.parametrize("outlier", [1e6, 1e12, 1e100, 1e308])
def test_outlier_outside_pair_does_not_erase_overlap_variation(outlier):
    result = summary(
        pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, outlier],
                "same": [1.0, 2.0, 3.0, None],
                "inverse": [3.0, 2.0, 1.0, None],
            }
        )
    )["correlations"]
    assert result["matrix"][0][1] == pytest.approx(1, abs=1e-12)
    assert result["matrix"][0][2] == pytest.approx(-1, abs=1e-12)
    assert result["matrix"][1][0] == result["matrix"][0][1]
    assert result["matrix"][2][0] == result["matrix"][0][2]
    assert result["counts"][0][1:] == [3, 3]
    assert result["valid_pairs"] == 3


def test_large_offset_small_representable_differences_and_missing_values():
    result = summary(
        pl.DataFrame(
            {
                "x": [1e16, 1e16 + 2, 1e16 + 6, 1e16 + 10, None],
                "y": [1.0, 2.0, 3.0, None, 20.0],
                "outlier": [1e16, 1e16 + 2, 1e16 + 6, 1e30, None],
            }
        )
    )["correlations"]
    expected = np.corrcoef([0.0, 2.0, 6.0], [1.0, 2.0, 3.0])[0, 1]
    assert result["matrix"][0][1] == pytest.approx(expected, abs=1e-12)
    assert result["matrix"][2][1] == pytest.approx(expected, abs=1e-12)
    assert result["counts"][0][1] == 3
    assert result["counts"][2][1] == 3


def test_wide_constant_columns_never_use_pairwise_fallback(monkeypatch):
    import importlib

    module = importlib.import_module("easy_glm.workflow.data_summary")

    def unexpected_fallback(*args):
        raise AssertionError("Constant columns must stay on the block path.")

    monkeypatch.setattr(module, "_stable_pair", unexpected_fallback)
    frame = pl.DataFrame({f"x{i}": [float(i)] * 50 for i in range(1000)})
    result = summary(frame)["correlations"]
    assert result["valid_pairs"] == 0
    assert result["pairs"] == []


def test_small_empty_constant_and_nonfinite_columns_are_json_safe():
    frame = pl.DataFrame(
        {
            "empty": pl.Series([None] * 5, dtype=pl.Float64),
            "constant": [4.0] * 5,
            "one": [1.0, None, float("nan"), float("inf"), -float("inf")],
            "nested": [[1], [2], None, [3], [4]],
            "binary": [b"\xff", None, b"a", b"b", b"c"],
        }
    )
    empty, constant, one, nested, binary = summary(frame)["variables"]
    assert empty["histogram"] == []
    assert empty["min"] is None
    assert empty["missing"] == 5
    assert constant["skewness"] is None
    assert constant["kurtosis"] is None
    assert constant["std"] == 0
    assert len(constant["histogram"]) == 1
    assert one["missing"] == 2
    assert one["nonfinite"] == 2
    assert one["unique"] == 3
    assert one["std"] is None
    assert nested["kind"] == "unsupported"
    assert nested["missing"] == 1
    assert binary["kind"] == "unsupported"
    assert binary["missing"] == 1
    assert summary(frame.head(0))["rows"] == 0


@pytest.mark.parametrize("value", [1.0, 1e308, 1e-300])
def test_adjacent_float_values_have_valid_histograms_and_stable_shape(value):
    values = [value, np.nextafter(value, np.inf)] * 10
    result = summary(pl.DataFrame({"x": values, "same": values}))
    column = result["variables"][0]
    assert sum(item["count"] for item in column["histogram"]) == len(values)
    assert column["skewness"] == pytest.approx(0, abs=1e-14)
    assert result["correlations"]["matrix"][0][1] == pytest.approx(1)


def test_all_column_kinds_have_the_same_descriptive_statistic_keys():
    frame = pl.DataFrame(
        {"numeric": [1], "category": ["A"], "nested": [[1]], "null": [None]}
    )
    required = {
        "finite",
        "min",
        "max",
        "range",
        "mean",
        "median",
        "std",
        "skewness",
        "kurtosis",
    }
    for column in summary(frame)["variables"]:
        assert required <= column.keys()
        if column["kind"] != "numeric":
            assert all(column[key] is None for key in required)


def test_extreme_magnitudes_do_not_overflow_moments_or_correlations():
    frame = pl.DataFrame(
        {
            "large": [-1e308, -5e307, 0.0, 5e307, 1e308],
            "small": [-1e-300, -5e-301, 0.0, 5e-301, 1e-300],
        }
    )
    result = summary(frame)
    large = result["variables"][0]
    assert large["range"] is None
    assert large["mean"] == pytest.approx(0)
    assert large["std"] == pytest.approx(np.sqrt(0.625) * 1e308)
    assert large["kurtosis"] == pytest.approx(-1.2)
    assert result["correlations"]["matrix"][0][1] == pytest.approx(1)
    assert sum(item["count"] for item in large["histogram"]) == 5


def test_sampling_is_reproducible_but_descriptions_use_every_row():
    rng = np.random.default_rng(27)
    x = rng.normal(size=12000)
    frame = pl.DataFrame({"x": x, "y": x * 0.5 + rng.normal(size=len(x))})
    first = summary(frame, sample_rows=200)
    assert first == summary(frame, sample_rows=200)
    correlations = first["correlations"]
    positions = np.sort(np.random.default_rng(0).choice(len(x), 200, replace=False))
    expected = np.corrcoef(
        frame["x"].to_numpy()[positions], frame["y"].to_numpy()[positions]
    )
    assert correlations["matrix"][0][1] == pytest.approx(expected[0, 1])
    assert correlations["sample_rows"] == 200
    assert correlations["total_rows"] == 12000
    assert correlations["sampled"] is True
    assert first["variables"][0]["rows"] == 12000
    assert first["variables"][0]["mean"] == pytest.approx(x.mean())
    assert summary(frame, sample_rows=20000)["correlations"]["sample_rows"] == 10000


def test_wide_input_checks_all_columns_but_only_returns_strongest_fifty_pairs():
    rng = np.random.default_rng(13)
    columns = {f"x{i}": rng.normal(size=100) for i in range(1000)}
    columns["x999"] = -columns["x0"]
    result = summary(pl.DataFrame(columns), sample_rows=100)["correlations"]
    assert result["mode"] == "pairs"
    assert "matrix" not in result
    assert "counts" not in result
    assert len(result["names"]) == 1000
    assert result["valid_pairs"] == 1000 * 999 // 2
    assert len(result["pairs"]) == 50
    assert result["pairs"][0]["left"] == "x0"
    assert result["pairs"][0]["right"] == "x999"
    assert result["pairs"][0]["correlation"] == pytest.approx(-1)


def test_only_requested_predictors_enter_correlations_and_unavailable_is_reported():
    frame = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "target": [1, 1, 2]})
    result = data_summary(
        frame,
        ["x", "target", "absent", "x"],
        correlation_variables=["x", "y", "absent"],
    )
    assert [item["name"] for item in result["variables"]] == ["x", "target"]
    assert result["unavailable_variables"] == ["absent"]
    assert result["correlations"]["names"] == ["x", "y"]
    assert result["correlations"]["excluded_names"] == ["absent"]


@pytest.mark.parametrize("sample_rows", [0, -1, 1.5, True])
def test_invalid_sample_size_is_rejected(sample_rows):
    with pytest.raises(ValueError, match="positive integer"):
        summary(pl.DataFrame({"x": [1]}), sample_rows=sample_rows)
