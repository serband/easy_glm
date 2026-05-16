import numpy as np
import polars as pl
import pytest

import easy_glm.core.model as model_module
from easy_glm import (
    fit_lasso_glm,
    generate_blueprint,
    predict_with_model,
    prepare_data,
    ratetable,
)


@pytest.mark.parametrize(
    ("divide_target_by_weight", "expected_y"),
    [(True, [1.0, 2.0]), (False, [2.0, 4.0])],
)
def test_fit_lasso_glm_divides_target_only_when_requested(
    monkeypatch, divide_target_by_weight, expected_y
):
    class CapturingRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x_data, y, sample_weight=None):
            self.x_data = x_data
            self.y = y
            self.sample_weight = sample_weight
            return self

    monkeypatch.setattr(model_module, "GeneralizedLinearRegressor", CapturingRegressor)
    df = pl.DataFrame(
        {
            "feature": [0.0, 1.0],
            "ClaimNb": [2.0, 4.0],
            "Exposure": [2.0, 2.0],
            "traintest": [1, 1],
        }
    )

    fitted = fit_lasso_glm(
        dataframe=df,
        target="ClaimNb",
        model_type="Poisson",
        weight_col="Exposure",
        train_test_col="traintest",
        divide_target_by_weight=divide_target_by_weight,
    )

    assert fitted.y.tolist() == expected_y
    assert fitted.sample_weight.tolist() == [2.0, 2.0]


def test_fit_lasso_glm_and_predict(synthetic_insurance_data):
    df = synthetic_insurance_data
    predictors = ["VehAge", "Region", "DrivAge"]
    bp = generate_blueprint(df)
    prepped = prepare_data(
        df=df,
        modelling_variables=predictors,
        additional_columns=["Exposure", "ClaimNb", "traintest"],
        formats=bp,
        traintest_column="traintest",
        table_name="cars",
    )
    model = fit_lasso_glm(
        dataframe=prepped,
        target="ClaimNb",
        model_type="Poisson",
        weight_col="Exposure",
        train_test_col="traintest",
        divide_target_by_weight=True,
    )
    new_slice_raw = df.head(5)
    new_slice_prepped = prepare_data(
        df=new_slice_raw,
        modelling_variables=predictors,
        formats=bp,
        table_name="line_prepped",
    )
    preds = predict_with_model(model, new_slice_prepped)
    assert len(preds) == 5
    assert np.all(np.isfinite(preds))


def test_ratetable_basic(synthetic_insurance_data):
    df = synthetic_insurance_data
    predictors = ["VehAge", "Region", "DrivAge"]
    bp = generate_blueprint(df)
    prepped = prepare_data(
        df=df,
        modelling_variables=predictors,
        additional_columns=["Exposure", "ClaimNb", "traintest"],
        formats=bp,
        traintest_column="traintest",
        table_name="cars",
    )
    model = fit_lasso_glm(
        dataframe=prepped,
        target="ClaimNb",
        model_type="Poisson",
        weight_col="Exposure",
        train_test_col="traintest",
        divide_target_by_weight=True,
    )
    levels = bp["VehAge"]
    tbl = ratetable(
        model=model,
        dataset=df,
        col_name="VehAge",
        levels=levels,
        prepare=lambda d: prepare_data(
            df=d,
            modelling_variables=predictors,
            formats=bp,
            table_name="line_prepped",
        ),
        random_seed=123,
    )
    assert tbl.height == len(levels)
    assert {"VehAge", "relativity", "prediction"}.issubset(set(tbl.columns))


def test_transforms_helpers_roundtrip():
    from easy_glm.core.transforms import lump_fun, o_matrix

    breaks = [1, 2, 3]
    cases = o_matrix("NumCol", breaks)
    assert len(cases) == 3
    assert all("NumCol" in c for c in cases)

    sql_expr = lump_fun("CatCol", ["A", "B", "C"])
    assert "CatCol_lumped" in sql_expr
