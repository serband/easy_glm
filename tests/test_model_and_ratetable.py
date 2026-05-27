import numpy as np
import polars as pl
import pytest

import easy_glm.core.model as model_module
from easy_glm import (
    EasyGLM,
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
        use_cv=False,
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
        use_cv=False,
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
        use_cv=False,
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


def test_fit_lasso_glm_cv(synthetic_insurance_data):
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
        use_cv=True,
        cv_params={"l1_ratio": [1.0], "n_alphas": 5},
    )
    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert hasattr(model, "alpha_")
    preds = predict_with_model(
        model, prepped.drop(["Exposure", "ClaimNb", "traintest"])
    )
    assert len(preds) == len(prepped)
    assert np.all(np.isfinite(preds))


def test_fit_lasso_glm_no_cv(synthetic_insurance_data):
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
        use_cv=False,
    )
    assert model.coef_ is not None
    assert model.intercept_ is not None
    preds = predict_with_model(
        model, prepped.drop(["Exposure", "ClaimNb", "traintest"])
    )
    assert len(preds) == len(prepped)
    assert np.all(np.isfinite(preds))


def test_easyglm_fit_and_predict(synthetic_insurance_data):
    df = synthetic_insurance_data
    predictors = ["VehAge", "Region", "DrivAge"]

    eglm = EasyGLM.fit(
        data=df,
        target="ClaimNb",
        model_type="Poisson",
        predictors=predictors,
        weight_col="Exposure",
        divide_target_by_weight=True,
        use_cv=False,
        base_rate=0.05,
    )

    assert eglm.model is not None
    assert eglm.blueprint is not None
    assert len(eglm.blueprint) >= len(predictors)
    assert eglm.rate_model is not None
    assert eglm.base_rate == 0.05
    assert set(eglm.predictors) == set(predictors)

    preds = eglm.predict(df.head(10))
    assert isinstance(preds, pl.Series)
    assert len(preds) == 10

    tables = eglm.relativities
    assert set(tables.keys()) == set(predictors)

    s = eglm.summary()
    assert s["model_type"] == "Poisson"
    assert s["target"] == "ClaimNb"
    assert s["weight_col"] == "Exposure"


def test_easyglm_matches_manual_pipeline(synthetic_insurance_data):
    """EasyGLM.fit must match the documented step-by-step workflow."""
    from easy_glm import generate_all_ratetables
    from easy_glm.engine import RateModel

    df = synthetic_insurance_data
    predictors = ["VehAge", "Region", "DrivAge"]
    base_rate = 0.05
    random_seed = 99

    eglm = EasyGLM.fit(
        data=df,
        target="ClaimNb",
        model_type="Poisson",
        predictors=predictors,
        weight_col="Exposure",
        divide_target_by_weight=True,
        use_cv=False,
        base_rate=base_rate,
        random_seed=random_seed,
    )

    train_df = df.filter(pl.col("traintest") == 1)
    bp = generate_blueprint(train_df)
    prepped = prepare_data(
        df=df,
        modelling_variables=predictors,
        additional_columns=["Exposure", "ClaimNb", "traintest"],
        formats=bp,
        traintest_column=None,
        table_name="line_prepped",
    )
    model = fit_lasso_glm(
        dataframe=prepped,
        target="ClaimNb",
        model_type="Poisson",
        weight_col="Exposure",
        train_test_col="traintest",
        divide_target_by_weight=True,
        use_cv=False,
    )
    tables = generate_all_ratetables(
        model=model,
        dataset=df,
        predictor_variables=predictors,
        blueprint=bp,
        random_seed=random_seed,
    )
    manual_rm = RateModel.from_rate_tables(
        all_tables=tables,
        blueprint=bp,
        base_rate=base_rate,
        model_type="Poisson",
        target="ClaimNb",
        weight_col="Exposure",
        exposure_col="Exposure",
        train_test_col="traintest",
        predictor_variables=predictors,
    )

    sample = df.head(20)
    np.testing.assert_array_almost_equal(
        eglm.rate_model.predict(sample),
        manual_rm.predict(sample),
    )
    assert set(eglm.relativities.keys()) == set(tables.keys())


def test_easyglm_blueprint_uses_training_rows_only(synthetic_insurance_data):
    df = synthetic_insurance_data
    predictors = ["VehAge", "Region", "DrivAge"]

    eglm = EasyGLM.fit(
        data=df,
        target="ClaimNb",
        model_type="Poisson",
        predictors=predictors,
        weight_col="Exposure",
        train_test_col="traintest",
        divide_target_by_weight=True,
        use_cv=False,
    )

    train_bp = generate_blueprint(df.filter(pl.col("traintest") == 1))
    for var in predictors:
        assert eglm.blueprint[var] == train_bp[var]


def test_easyglm_requires_train_test_col(synthetic_insurance_data):
    df = synthetic_insurance_data.drop("traintest")
    with pytest.raises(ValueError, match="traintest"):
        EasyGLM.fit(
            data=df,
            target="ClaimNb",
            model_type="Poisson",
            predictors=["VehAge"],
            train_test_col="traintest",
            use_cv=False,
        )


def test_easyglm_custom_train_test_col_name(synthetic_insurance_data):
    df = synthetic_insurance_data.rename({"traintest": "is_train"})
    eglm = EasyGLM.fit(
        data=df,
        target="ClaimNb",
        model_type="Poisson",
        predictors=["VehAge", "Region"],
        weight_col="Exposure",
        train_test_col="is_train",
        divide_target_by_weight=True,
        use_cv=False,
    )
    assert eglm.rate_model.metadata.train_test_col == "is_train"
    holdout = df.filter(pl.col("is_train") == 0)
    preds = eglm.rate_model.predict(holdout)
    assert len(preds) == holdout.height


def test_validate_train_test_column_rejects_invalid_values():
    from easy_glm.core.model import validate_train_test_column

    df = pl.DataFrame({"split": [1, 2, 0]})
    with pytest.raises(ValueError, match="only 1"):
        validate_train_test_column(df, "split")


def test_easyglm_serialization(synthetic_insurance_data, tmp_path):
    df = synthetic_insurance_data
    predictors = ["VehAge", "Region", "DrivAge"]

    eglm = EasyGLM.fit(
        data=df,
        target="ClaimNb",
        model_type="Poisson",
        predictors=predictors,
        weight_col="Exposure",
        divide_target_by_weight=True,
        use_cv=False,
        base_rate=0.05,
    )

    model_dir = tmp_path / "test_model"
    eglm.save(model_dir)

    assert (model_dir / "glm_model.joblib").exists()
    assert (model_dir / "blueprint.json").exists()
    assert (model_dir / "rate_model.json").exists()
    assert (model_dir / "config.json").exists()
    assert (model_dir / "rate_tables").is_dir()

    loaded = EasyGLM.load(model_dir)
    assert loaded.base_rate == eglm.base_rate
    assert loaded.predictors == eglm.predictors
    assert loaded.model is not None

    original_preds = eglm.predict(df.head(5)).to_list()
    loaded_preds = loaded.predict(df.head(5)).to_list()
    assert original_preds == loaded_preds
