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


def test_easyglm_actual_expected_plots_follow_fitted_tables(synthetic_insurance_data):
    """The first-use diagnostic is rate-scale, split, and table-aligned."""
    df = synthetic_insurance_data
    eglm = EasyGLM.fit(
        data=df,
        target="ClaimNb",
        model_type="Poisson",
        predictors=["VehAge", "Region"],
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=0.01,
    )

    figures = eglm.plot_actual_vs_expected(df, show=False)
    assert set(figures) == {"VehAge", "Region"}
    assert all(set(by_split) == {"Training", "Test"} for by_split in figures.values())

    for factor, by_split in figures.items():
        labels = eglm.relativities[factor]["label"].to_list()
        for split, figure in by_split.items():
            volume, actual, expected = figure.data[:3]
            assert list(actual.x) == labels
            assert list(expected.x) == labels
            assert volume.name == "Exposure"
            assert actual.name == "actual" and actual.line.color == "#c0392b"
            assert expected.name == "expected" and expected.line.color == "#1f5f99"

            frame = df.filter(pl.col("traintest") == (1 if split == "Training" else 0))
            assert np.nansum(
                np.asarray(actual.y) * np.asarray(volume.y)
            ) == pytest.approx(frame["ClaimNb"].sum())
            assert np.nansum(
                np.asarray(expected.y) * np.asarray(volume.y)
            ) == pytest.approx(eglm.rate_model.predict(frame).sum())


def test_validate_train_test_column_rejects_invalid_values():
    from easy_glm.core.split import validate_train_test_column

    df = pl.DataFrame({"split": [1, 2, 0]})
    with pytest.raises(ValueError, match="only 1"):
        validate_train_test_column(df, "split")


def test_public_random_train_test_split_is_reproducible(synthetic_insurance_data):
    from easy_glm import add_train_test_split

    data = synthetic_insurance_data.drop("traintest")
    first = add_train_test_split(data, train_fraction=0.7, seed=12)
    second = add_train_test_split(data, train_fraction=0.7, seed=12)
    assert first["traintest"].to_list() == second["traintest"].to_list()
    assert set(first["traintest"].unique()) == {0, 1}
    with pytest.raises(ValueError, match="overwrite"):
        add_train_test_split(first)


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


def test_save_load_round_trip_with_an_interaction(synthetic_insurance_data, tmp_path):
    """A two-stage fit (mains frozen, interaction cells on top) must survive
    save/load: stage 1's estimator alone cannot score the composed mains+cells
    spec, so both are written and the pair is rebuilt."""
    from easy_glm import TwoStageFit, fit_two_stage, rate_tables

    df = synthetic_insurance_data
    spec = DesignSpec.from_data(
        df,
        ["VehAge", "Region", "DrivAge"],
        min_level_share=0.02,
        weight_col="Exposure",
        interactions=[("DrivAge", "Region")],
        min_cell_exposure=0.005,
    )
    fit = fit_two_stage(
        df,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=0.001,
    )
    assert isinstance(fit, TwoStageFit)
    eglm = EasyGLM(fit, to_rate_model(fit, exposure_col="Exposure"), rate_tables(fit))

    model_dir = tmp_path / "two_stage"
    eglm.save(model_dir)
    assert (model_dir / "glm_model_stage2.joblib").exists()

    loaded = EasyGLM.load(model_dir)
    assert isinstance(loaded.glm, TwoStageFit)
    assert loaded.glm.alpha_stage2 == fit.alpha_stage2
    assert loaded.predictors == eglm.predictors
    np.testing.assert_array_equal(loaded.glm.coef, fit.coef)
    np.testing.assert_array_equal(
        loaded.predict(df.head(50)).to_numpy(), eglm.predict(df.head(50)).to_numpy()
    )
