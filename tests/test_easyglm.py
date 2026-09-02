"""EasyGLM front door: fit, predict, building-block equivalence, serialization."""

import numpy as np
import polars as pl
import pytest

from easy_glm import DesignSpec, EasyGLM, fit_glm, to_rate_model


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
        alpha=0.01,
        base_rate=0.05,
    )

    assert eglm.model is not None
    assert eglm.blueprint is not None
    assert len(eglm.blueprint) == len(predictors)
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
    df = synthetic_insurance_data
    predictors = ["VehAge", "Region", "DrivAge"]

    eglm = EasyGLM.fit(
        data=df,
        target="ClaimNb",
        model_type="Poisson",
        predictors=predictors,
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=0.01,
    )

    train_df = df.filter(pl.col("traintest") == 1)
    spec = DesignSpec.from_data(train_df, predictors, weight_col="Exposure")
    fit = fit_glm(
        train_df,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=0.01,
    )
    manual_rm = to_rate_model(fit, exposure_col="Exposure", train_test_col="traintest")

    sample = df.head(20)
    np.testing.assert_allclose(
        eglm.rate_model.predict(sample), manual_rm.predict(sample), rtol=1e-10
    )
    np.testing.assert_allclose(
        eglm.rate_model.predict(sample, exposure_col=None),
        fit.predict(sample),
        rtol=1e-10,
    )
    assert set(eglm.relativities.keys()) == set(predictors)


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
        alpha=0.01,
    )

    train_spec = DesignSpec.from_data(
        df.filter(pl.col("traintest") == 1), predictors, weight_col="Exposure"
    )
    assert eglm.spec.to_dict() == train_spec.to_dict()
    full_spec = DesignSpec.from_data(df, predictors, weight_col="Exposure")
    assert eglm.spec.to_dict() != full_spec.to_dict()


def test_easyglm_requires_train_test_col(synthetic_insurance_data):
    df = synthetic_insurance_data.drop("traintest")
    with pytest.raises(ValueError, match="traintest"):
        EasyGLM.fit(
            data=df,
            target="ClaimNb",
            model_type="Poisson",
            predictors=["VehAge"],
            train_test_col="traintest",
            alpha=0.01,
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
        alpha=0.01,
    )
    assert eglm.rate_model.metadata.train_test_col == "is_train"
    holdout = df.filter(pl.col("is_train") == 0)
    preds = eglm.rate_model.predict(holdout)
    assert len(preds) == holdout.height


def test_validate_train_test_column_rejects_invalid_values():
    from easy_glm.core.split import validate_train_test_column

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
        alpha=0.01,
        base_rate=0.05,
    )

    model_dir = tmp_path / "test_model"
    eglm.save(model_dir)

    assert (model_dir / "glm_model.joblib").exists()
    assert (model_dir / "spec.json").exists()
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
