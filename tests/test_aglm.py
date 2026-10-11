import numpy as np
import polars as pl
import pytest

from easy_glm.core.design import (
    AglmLEncoder,
    AglmOEncoder,
    DesignSpec,
    aglm_encoder_from_data,
    quantile_knots,
    round_range_outward,
)


def test_aglm_o_and_l_match_documented_bases() -> None:
    values = pl.Series("x", [7.0])
    o = AglmOEncoder("x", [0.0, 2.0, 5.0, 10.0], null_indicator=False)
    linear = AglmLEncoder("x", [0.0, 2.0, 5.0, 10.0], null_indicator=False)
    np.testing.assert_allclose(o.transform(values), [[7.0, 1.0, 1.0, 0.4]])
    np.testing.assert_allclose(linear.transform(values), [[7.0, 5.0, 2.0]])


def test_aglm_flat_clamps_raw_and_basis_but_extend_does_not() -> None:
    values = pl.Series("x", [-2.0, 3.0])
    extend = AglmOEncoder("x", [0.0, 1.0, 2.0], null_indicator=False)
    flat = AglmOEncoder(
        "x", [0.0, 1.0, 2.0], extrapolation="flat", null_indicator=False
    )
    np.testing.assert_allclose(extend.transform(values)[:, 0], [-2.0, 3.0])
    np.testing.assert_allclose(flat.transform(values)[:, 0], [0.0, 2.0])


def test_aglm_missing_needs_indicator_and_infinity_is_refused() -> None:
    with pytest.raises(ValueError, match="null_indicator=False"):
        AglmOEncoder("x", [0.0, 1.0], null_indicator=False).transform(
            pl.Series("x", [None, 1.0])
        )
    with pytest.raises(ValueError, match="infinite"):
        aglm_encoder_from_data("x", pl.Series("x", [0.0, float("inf")]))
    manual = aglm_encoder_from_data("x", pl.Series("x", [1.0, 1.0]), breaks=[0.0, 2.0])
    assert isinstance(manual, AglmOEncoder)
    with pytest.raises(ValueError, match="integer"):
        aglm_encoder_from_data(
            "x", pl.Series("x", [0.0, 1.0]), breaks=[0.0, 1.0], n_breaks=True
        )


def test_aglm_design_dense_sparse_and_chunked_prediction_agree() -> None:
    data = pl.DataFrame({"x": np.linspace(-2.0, 3.0, 32)})
    spec = DesignSpec.from_data(
        data, ["x"], numeric_kinds={"x": "aglm"}, breaks={"x": [0.0, 1.0, 2.0]}
    )
    dense = spec.build(data, sparse=False)
    sparse = spec.build(data, sparse=True).toarray()
    np.testing.assert_allclose(dense, sparse)
    coef = np.arange(spec.n_features, dtype=float) / 3
    np.testing.assert_allclose(spec.linear_predictor(data, coef), dense @ coef)


def test_rare_binary_quantile_and_tiny_outward_range_remain_usable() -> None:
    assert quantile_knots(pl.Series("x", [0.0] * 99 + [1.0]), 20) == [1.0]
    assert round_range_outward(1e-12, 9e-12) == (1e-12, 9e-12)
