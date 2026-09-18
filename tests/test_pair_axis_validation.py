"""Pair scoring refuses malformed numeric cuts before searchsorted can use them."""

import pytest

from easy_glm.engine.models import FromToRow, PairTableConfig, VariableConfig
from easy_glm.engine.rate_model import RateModel


@pytest.mark.parametrize("cuts", [[1.0, -1.0], [1.0, 1.0], [1.0, float("inf")]])
def test_pair_numeric_axis_cuts_are_finite_and_increasing(cuts):
    numeric = VariableConfig(
        "numeric",
        [
            FromToRow(None, cuts[0], 1.0),
            FromToRow(cuts[0], cuts[1], 1.0),
            FromToRow(cuts[1], None, 1.0),
            FromToRow(None, None, 1.0),
        ],
    )
    categorical = VariableConfig(
        "categorical",
        [FromToRow("A", "A", 1.0), FromToRow(None, None, 1.0)],
    )
    table = PairTableConfig("stage", ("Age", "Region"), (numeric, categorical), [])
    with pytest.raises(ValueError, match="Numeric pair axes need tiled open tails"):
        RateModel(1.0, {}, pair_tables=[table])
