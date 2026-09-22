"""Permutation-importance samples are deterministic and coverage-aware."""

import numpy as np
import polars as pl
import pytest

from easy_glm.workflow.importance_sampling import importance_sample_indices


def test_percentage_uses_sorted_seeded_floor_sample_and_changes_with_seed():
    frame = pl.DataFrame({"x": np.arange(5_003), "y": np.arange(5_003) % 7})
    first, metadata = importance_sample_indices(
        frame,
        importance_sample_pct=30,
        seed=7,
        outcome=frame["y"].to_numpy(),
        family="normal",
        variables=("x",),
    )
    again, _ = importance_sample_indices(
        frame,
        importance_sample_pct=30,
        seed=7,
        outcome=frame["y"].to_numpy(),
        family="normal",
        variables=("x",),
    )
    other, _ = importance_sample_indices(
        frame,
        importance_sample_pct=30,
        seed=8,
        outcome=frame["y"].to_numpy(),
        family="normal",
        variables=("x",),
    )
    assert len(first) == 1_500 == int(np.floor(0.3 * frame.height))
    assert np.all(first[:-1] < first[1:])
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert metadata["importance_rows"] == 1_500
    assert metadata["actual_pct"] == pytest.approx(100 * 1_500 / 5_003)
    assert metadata["fallback_reasons"] == []


@pytest.mark.parametrize("value", [True, 0, -1, 101, float("nan"), float("inf")])
def test_invalid_percentages_are_rejected(value):
    with pytest.raises(ValueError, match="importance_sample_pct"):
        importance_sample_indices(pl.DataFrame({"x": [1]}), importance_sample_pct=value)


def test_support_and_constant_sample_checks_expand_once_to_full_training():
    n = 5_000
    frame = pl.DataFrame({"x": np.arange(n), "rare": np.zeros(n)})
    proposed, _ = importance_sample_indices(
        frame,
        importance_sample_pct=30,
        seed=13,
        outcome=np.arange(n, dtype=float),
        family="normal",
    )
    omitted = next(index for index in range(n) if index not in set(proposed))
    rare = np.zeros(n)
    rare[omitted] = 1
    frame = frame.with_columns(pl.Series("rare", rare))
    indices, metadata = importance_sample_indices(
        frame,
        importance_sample_pct=30,
        seed=13,
        outcome=np.r_[np.ones(10), np.zeros(n - 10)],
        family="poisson",
        variables=("rare",),
    )
    assert np.array_equal(indices, np.arange(n))
    assert any("positive-outcome" in reason for reason in metadata["fallback_reasons"])
    assert any(
        "made a variable constant" in reason for reason in metadata["fallback_reasons"]
    )
    assert metadata["positive_outcome_rows"] == 10
    assert metadata["support_warnings"]


def test_explicit_full_sample_is_exact_and_reports_weight_share():
    frame = pl.DataFrame({"x": np.arange(1_200)})
    weights = np.linspace(0.1, 2, frame.height)
    indices, metadata = importance_sample_indices(
        frame,
        importance_sample_pct=100,
        seed=99,
        weights=weights,
        variables=("x",),
    )
    assert np.array_equal(indices, np.arange(frame.height))
    assert metadata["actual_pct"] == 100
    assert metadata["weight_share"] == pytest.approx(1)
    assert metadata["fallback_reasons"] == []
