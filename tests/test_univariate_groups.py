"""One-way groups must not duplicate null outcomes or lose exposure types."""

import numpy as np
import polars as pl
import pytest

from easy_glm.engine.models import NULL_LABEL
from easy_glm.workflow.explore import univariate


@pytest.mark.parametrize("integer_weight", [False, True])
@pytest.mark.parametrize("null_in_top", [False, True])
def test_lumped_categories_aggregate_exactly_the_remaining_rows(
    integer_weight, null_in_top
):
    weights = [100, 1, 1] if null_in_top else [1, 100, 1]
    frame = pl.DataFrame(
        {
            "x": [None, "A", "B"],
            "y": [100, 0, 0],
            "w": weights if integer_weight else [float(w) for w in weights],
        }
    )
    result = univariate(
        frame, "x", target="y", weight="w", divide_target_by_weight=True, max_levels=1
    )
    rows = result["table"].to_dicts()
    assert rows[0]["label"] == (NULL_LABEL if null_in_top else "A")
    assert rows[1]["exposure"] == 2
    assert rows[1]["rate"] == (0 if null_in_top else 50)
    assert sum(row["exposure"] for row in rows) == 102
    assert sum(row["share"] for row in rows) == pytest.approx(1)


def test_constant_numeric_nan_and_null_are_one_missing_group():
    frame = pl.DataFrame({"x": [2.0, 2.0, np.nan, None], "y": [1, 3, 5, 7]})
    result = univariate(frame, "x", target="y")
    assert result["kind"] == "numeric"
    assert result["null_share"] == 0.5 and result["n_unique"] == 2
    assert result["table"]["label"].to_list() == ["2.0", NULL_LABEL]
    assert result["table"]["rate"].to_list() == [2.0, 6.0]


@pytest.mark.parametrize("divide,expected", [(True, 1.0), (False, 1.0)])
def test_missing_target_is_excluded_from_both_rate_numerator_and_denominator(
    divide, expected
):
    frame = pl.DataFrame(
        {"x": ["A", "A", "B", "C"], "y": [None, 1.0, None, 3.0], "w": [10, 1, 5, 0]}
    )
    result = univariate(
        frame, "x", target="y", weight="w", divide_target_by_weight=divide
    )
    rows = {row["label"]: row for row in result["table"].to_dicts()}
    assert rows["A"]["rate"] == expected and rows["A"]["exposure"] == 11
    assert rows["B"]["rate"] is None and rows["C"]["rate"] is None


def test_nonfinite_target_values_are_not_observed_outcomes():
    frame = pl.DataFrame(
        {"x": ["A"] * 4, "y": [2.0, np.nan, np.inf, -np.inf], "w": [2, 10, 10, 10]}
    )
    assert univariate(frame, "x", target="y")["table"]["rate"][0] == 2
    assert univariate(frame, "x", target="y", weight="w")["table"]["rate"][0] == 2
