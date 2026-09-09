"""Trailing arithmetic means follow point order, independent of exposure."""

import copy

import numpy as np
import pytest
from test_d5_tooling import categorical_table, linear_table

from easy_glm.engine import tooling
from easy_glm.engine.models import FromToRow, VariableConfig


def test_trailing_example_endpoints_exposure_and_null():
    values = [1, 1, 1, 4, 4, 4]
    cfg = VariableConfig(
        type="numeric",
        table=[
            FromToRow(None if i == 0 else i, None if i == 5 else i + 1, value, 10**i)
            for i, value in enumerate(values)
        ]
        + [FromToRow(None, None, 7, 999)],
    )
    before = copy.deepcopy(cfg)
    assert tooling.smooth_trailing_average(cfg, "x", window=3).values == [
        1,
        1,
        1,
        2,
        3,
        4,
        7,
    ]
    assert cfg == before
    future = copy.deepcopy(cfg)
    future.table[5].relativity = 100
    assert tooling.smooth_trailing_average(future, "x", window=3).values[:5] == [
        1,
        1,
        1,
        2,
        3,
    ]
    for row in cfg.table:
        row.exposure = 0
    assert tooling.smooth_trailing_average(cfg, "x", window=3).values == [
        1,
        1,
        1,
        2,
        3,
        4,
        7,
    ]
    assert tooling.smooth_trailing_average(cfg, "x", window=2).values == [
        1,
        1,
        1,
        2.5,
        4,
        4,
        7,
    ]
    assert tooling.smooth_trailing_average(cfg, "x", window=1).values == values + [7]


def test_linear_nodes_once_and_slopes_remain_continuous():
    cfg = linear_table()
    current = tooling.group_values(cfg)
    result = tooling.smooth_trailing_average(cfg, "x", window=3)
    expected = [
        sum(current[max(0, i - 2) : i + 1]) / len(current[max(0, i - 2) : i + 1])
        for i in range(len(current))
    ]
    changed = tooling.apply_values(cfg, result.values)
    np.testing.assert_allclose(tooling.group_values(changed), expected)
    assert changed.table[0].relativity == changed.table[1].relativity
    for left, right in zip(changed.table[1:-2], changed.table[2:-1], strict=True):
        assert left.relativity_to == pytest.approx(right.relativity)
    assert changed.table[-1].relativity == cfg.table[-1].relativity


def test_categorical_requires_order_and_window_validation():
    cfg = categorical_table()
    with pytest.raises(tooling.ToolingError):
        tooling.smooth_trailing_average(cfg, "x")
    out = tooling.smooth_trailing_average(cfg, "x", ordered=True)
    np.testing.assert_allclose(out.values, [0.9, 1.1, (0.9 + 1.3 + 1.05) / 3, 1.75])
    for invalid in [0, -1, 2.5, True]:
        with pytest.raises(tooling.ToolingError):
            tooling.smooth_trailing_average(cfg, "x", ordered=True, window=invalid)
