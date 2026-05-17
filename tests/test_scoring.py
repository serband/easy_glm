from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from easy_glm.engine._scoring import score_categorical, score_numeric
from easy_glm.engine.models import FromToRow, VariableConfig


def _make_numeric_config():
    table = [
        FromToRow(from_=None, to_=18, relativity=1.45),
        FromToRow(from_=18, to_=23, relativity=1.30),
        FromToRow(from_=23, to_=28, relativity=1.15),
        FromToRow(from_=28, to_=33, relativity=1.00),
        FromToRow(from_=33, to_=None, relativity=0.90),
    ]
    bp = np.array([18, 23, 28, 33], dtype=float)
    rels = np.array([1.45, 1.30, 1.15, 1.00, 0.90], dtype=float)
    return VariableConfig(
        type="numeric", table=table, breakpoints=bp, relativities=rels
    )


def _make_categorical_config():
    table = [
        FromToRow(from_="North", to_="North", relativity=0.95),
        FromToRow(from_="South", to_="South", relativity=1.05),
        FromToRow(from_="Urban", to_="Urban", relativity=1.00),
        FromToRow(from_=None, to_=None, relativity=1.0),
    ]
    return VariableConfig(
        type="categorical",
        table=table,
        cat_map={"North": 0.95, "South": 1.05, "Urban": 1.00},
        fallback=1.0,
    )


class TestScoreNumeric:
    def test_exact_boundaries(self):
        config = _make_numeric_config()
        values = np.array([18.0, 23.0, 28.0, 33.0])
        result = score_numeric(values, config)
        expected = np.array([1.30, 1.15, 1.00, 0.90])
        np.testing.assert_array_equal(result, expected)

    def test_below_first(self):
        config = _make_numeric_config()
        values = np.array([10.0, 17.9])
        result = score_numeric(values, config)
        expected = np.array([1.45, 1.45])
        np.testing.assert_array_equal(result, expected)

    def test_above_last(self):
        config = _make_numeric_config()
        values = np.array([40.0, 99.0])
        result = score_numeric(values, config)
        expected = np.array([0.90, 0.90])
        np.testing.assert_array_equal(result, expected)

    def test_between_boundaries(self):
        config = _make_numeric_config()
        values = np.array([20.0, 25.0, 30.0])
        result = score_numeric(values, config)
        expected = np.array([1.30, 1.15, 1.00])
        np.testing.assert_array_equal(result, expected)

    def test_just_below_edge(self):
        config = _make_numeric_config()
        values = np.array([22.999, 17.0])
        result = score_numeric(values, config)
        expected = np.array([1.30, 1.45])
        np.testing.assert_array_equal(result, expected)

    def test_single_value(self):
        config = _make_numeric_config()
        values = np.array([25.0])
        result = score_numeric(values, config)
        assert result[0] == 1.15

    def test_empty_array(self):
        config = _make_numeric_config()
        values = np.array([], dtype=float)
        result = score_numeric(values, config)
        assert len(result) == 0

    def test_nan_raises(self):
        config = _make_numeric_config()
        values = np.array([20.0, np.nan, 30.0])
        with pytest.raises(ValueError, match="NaN"):
            score_numeric(values, config)

    def test_large_array(self):
        config = _make_numeric_config()
        rng = np.random.default_rng(42)
        values = rng.uniform(10, 50, size=1_000_000).astype(float)
        result = score_numeric(values, config)
        assert len(result) == 1_000_000
        assert not np.any(np.isnan(result))
        assert np.all(result >= 0.90)
        assert np.all(result <= 1.45)

    def test_fallback_when_no_precompute(self):
        table = [
            FromToRow(from_=None, to_=18, relativity=1.45),
            FromToRow(from_=18, to_=None, relativity=1.30),
        ]
        config = VariableConfig(type="numeric", table=table)
        values = np.array([10.0, 25.0])
        result = score_numeric(values, config)
        expected = np.array([1.45, 1.30])
        np.testing.assert_array_equal(result, expected)


class TestScoreCategorical:
    def test_known_levels(self):
        config = _make_categorical_config()
        series = pl.Series("Region", ["North", "South", "Urban"])
        result = score_categorical(series, config)
        expected = np.array([0.95, 1.05, 1.00])
        np.testing.assert_array_equal(result, expected)

    def test_unknown_level(self):
        config = _make_categorical_config()
        series = pl.Series("Region", ["Rural", "West"])
        result = score_categorical(series, config)
        expected = np.array([1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_null_value(self):
        config = _make_categorical_config()
        series = pl.Series("Region", ["North", None, "Urban"])
        result = score_categorical(series, config)
        expected = np.array([0.95, 1.0, 1.00])
        np.testing.assert_array_equal(result, expected)

    def test_empty_series(self):
        config = _make_categorical_config()
        series = pl.Series("Region", [], dtype=pl.Utf8)
        result = score_categorical(series, config)
        assert len(result) == 0

    def test_all_unknown(self):
        config = _make_categorical_config()
        series = pl.Series("Region", ["X", "Y", "Z"])
        result = score_categorical(series, config)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_no_known_map(self):
        table = [
            FromToRow(from_=None, to_=None, relativity=0.88),
        ]
        config = VariableConfig(type="categorical", table=table, fallback=0.88)
        series = pl.Series("Region", ["A", "B"])
        result = score_categorical(series, config)
        expected = np.array([0.88, 0.88])
        np.testing.assert_array_equal(result, expected)
