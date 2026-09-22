import numpy as np
import polars as pl
import pytest

from easy_glm.core.design import DesignSpec
from easy_glm.core.fit import fit_glm
from easy_glm.core.tables import to_rate_model


@pytest.fixture
def intercept_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "claims": [0.0, 2.0, 1.0, 4.0, 1.0, 3.0],
            "exposure": [1.0, 2.0, 1.5, 2.5, 1.0, 2.0],
            "offset": [-0.2, 0.1, 0.0, 0.3, -0.1, 0.2],
        }
    )


@pytest.mark.parametrize("aggregate", [False, True])
def test_intercept_only_fit_preserves_weights_offset_and_empty_spec(
    intercept_data: pl.DataFrame, aggregate: bool
):
    fit = fit_glm(
        intercept_data,
        DesignSpec(),
        "claims",
        family="poisson",
        weight_col="exposure",
        offset_col="offset",
        divide_target_by_weight=True,
        alpha=0.05,
        aggregate=aggregate,
    )
    predicted = fit.predict(intercept_data)
    exposure = intercept_data["exposure"].to_numpy()
    observed_unit = intercept_data["claims"].to_numpy() / exposure

    assert fit.coef.shape == (0,)
    assert fit.feature_names == []
    assert fit.spec.encoders == {}
    assert np.dot(exposure, observed_unit - predicted) == pytest.approx(0.0, abs=1e-7)
    rate_model = to_rate_model(fit)
    assert rate_model.variables == {}
    np.testing.assert_allclose(
        rate_model.predict(intercept_data), predicted, rtol=1e-12
    )


def test_intercept_only_cv_has_no_dummy_coefficient(intercept_data: pl.DataFrame):
    fit = fit_glm(
        intercept_data,
        DesignSpec(),
        "claims",
        family="poisson",
        weight_col="exposure",
        divide_target_by_weight=True,
        cv=3,
        n_alphas=4,
    )

    assert fit.coef.shape == (0,)
    assert fit.model.coef_path_.shape[-1] == 0
    assert np.all(np.isfinite(fit.predict(intercept_data)))
