"""Coefficient trajectories read the original fitted paths, never refit models."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import polars as pl
import pytest
from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV

from easy_glm import DesignSpec, fit_glm, fit_two_stage
from easy_glm.core.fit import GLMFit, TwoStageFit
from easy_glm.workflow.diagnostics import COEFFICIENT_PATH_SCHEMA, coefficient_path


@pytest.fixture(scope="module")
def training_data() -> pl.DataFrame:
    rng = np.random.default_rng(81)
    age = rng.uniform(18, 80, 600)
    region = rng.choice(["A", "B=zone", "C>=zone"], len(age))
    exposure = rng.uniform(0.4, 1, len(age))
    rate = np.exp(
        -0.3
        + 0.3 * (age > 40)
        + 0.4 * (region == "B=zone")
        + 0.5 * ((age < 40) & (region == "C>=zone"))
    )
    return pl.DataFrame(
        {
            "age>=name": age,
            "region=label": region,
            "exposure": exposure,
            "claims": rng.poisson(rate * exposure).astype(float),
        }
    )


def specification(frame: pl.DataFrame, *, interaction: bool = False) -> DesignSpec:
    return DesignSpec.from_data(
        frame,
        ["age>=name", "region=label"],
        knots={"age>=name": [40.0, 60.0]},
        interactions=[("age>=name", "region=label")] if interaction else None,
        min_cell_exposure=0.0,
    )


FIT_OPTIONS = {
    "family": "poisson",
    "weight_col": "exposure",
    "divide_target_by_weight": True,
}


@pytest.fixture(scope="module")
def cv_fit(training_data) -> GLMFit:
    return fit_glm(
        training_data,
        specification(training_data),
        "claims",
        cv=2,
        n_alphas=5,
        l1_ratio=[0.5, 1.0],
        **FIT_OPTIONS,
    )


@pytest.fixture(scope="module")
def two_stage_cv_fit(training_data) -> TwoStageFit:
    fitted = fit_two_stage(
        training_data,
        specification(training_data, interaction=True),
        "claims",
        cv=2,
        n_alphas=4,
        **FIT_OPTIONS,
    )
    assert isinstance(fitted, TwoStageFit)
    return fitted


def assert_cv_stage_matches_stored_folds(fit: GLMFit, rows: pl.DataFrame) -> None:
    model = fit.model
    stored = np.asarray(model.coef_path_)
    alphas = np.atleast_2d(model.alphas_)
    ratios = np.atleast_1d(model.l1_ratio)
    assert rows.height == np.prod(stored.shape[1:])
    assert rows["source"].unique().to_list() == ["cv_fold_mean"]
    assert rows["folds"].unique().to_list() == [stored.shape[0]]
    for ratio_index, ratio in enumerate(ratios):
        for alpha_index, alpha in enumerate(alphas[ratio_index]):
            point = rows.filter(
                (pl.col("l1_ratio") == ratio) & (pl.col("alpha") == alpha)
            ).sort("feature_index")
            np.testing.assert_array_equal(
                point["coefficient"].to_numpy(),
                stored[:, ratio_index, alpha_index, :].mean(axis=0),
            )
            np.testing.assert_array_equal(
                point["coefficient_std"].to_numpy(),
                stored[:, ratio_index, alpha_index, :].std(axis=0),
            )
    selected = rows.filter(pl.col("selected"))
    assert selected.height == fit.spec.n_features
    assert selected["alpha"].unique().to_list() == [float(model.alpha_)]
    assert selected["l1_ratio"].unique().to_list() == [float(model.l1_ratio_)]


def test_single_stage_cv_reads_fold_paths_and_preserves_original_fit(
    cv_fit, monkeypatch
):
    original = cv_fit.model.coef_path_.copy()
    original_coef = cv_fit.coef.copy()

    def unexpected_refit(*args, **kwargs):
        raise AssertionError("Reading a stored coefficient path must never fit.")

    monkeypatch.setattr(GeneralizedLinearRegressor, "fit", unexpected_refit)
    monkeypatch.setattr(GeneralizedLinearRegressorCV, "fit", unexpected_refit)
    rows = coefficient_path(cv_fit)
    assert rows["stage"].unique().to_list() == [1]
    assert_cv_stage_matches_stored_folds(cv_fit, rows)
    np.testing.assert_array_equal(cv_fit.model.coef_path_, original)
    np.testing.assert_array_equal(cv_fit.coef, original_coef)


def test_feature_identity_comes_from_metadata_even_with_delimiters_in_names(cv_fit):
    rows = coefficient_path(cv_fit)
    for index, feature in enumerate(cv_fit.spec.features):
        metadata = rows.filter(pl.col("feature_index") == index).row(0, named=True)
        assert metadata["feature"] == feature.name
        assert metadata["variable"] == feature.variable
        assert metadata["kind"] == feature.kind
        assert metadata["knot"] == feature.knot
        assert metadata["level"] == feature.level
    assert "Intercept" not in rows["feature"].to_list()
    assert rows["feature_index"].max() == cv_fit.spec.n_features - 1


def test_two_stage_cv_keeps_main_and_cell_paths_separate(two_stage_cv_fit):
    rows = coefficient_path(two_stage_cv_fit)
    assert rows["stage"].unique().sort().to_list() == [1, 2]
    assert_cv_stage_matches_stored_folds(
        two_stage_cv_fit.stage1, rows.filter(pl.col("stage") == 1)
    )
    stage2 = rows.filter(pl.col("stage") == 2)
    assert_cv_stage_matches_stored_folds(two_stage_cv_fit.stage2, stage2)
    assert stage2["kind"].unique().to_list() == ["cell"]
    for index, feature in enumerate(two_stage_cv_fit.stage2.spec.features):
        metadata = stage2.filter(pl.col("feature_index") == index).row(0, named=True)
        assert (metadata["cell_a"], metadata["cell_b"]) == feature.cell
        assert metadata["variable"] == feature.variable
    selected = stage2.filter(pl.col("selected")).sort("feature_index")
    # Fold-trained trajectories differ from the final-offset full-training refit.
    assert not np.allclose(
        selected["coefficient"].to_numpy(), two_stage_cv_fit.stage2.coef
    )


def test_fixed_alpha_is_one_real_point_without_fabricating_a_path(training_data):
    fitted = fit_glm(
        training_data, specification(training_data), "claims", alpha=0.02, **FIT_OPTIONS
    )
    rows = coefficient_path(fitted).sort("feature_index")
    assert rows.height == fitted.spec.n_features
    assert rows["alpha"].unique().to_list() == [0.02]
    assert rows["source"].unique().to_list() == ["fixed_fit"]
    assert rows["folds"].unique().to_list() == [0]
    assert rows["selected"].all()
    assert rows["coefficient_std"].null_count() == rows.height
    np.testing.assert_array_equal(rows["coefficient"].to_numpy(), fitted.coef)


def test_explicit_fixed_stage_two_preserves_its_single_point(training_data):
    fitted = fit_two_stage(
        training_data,
        specification(training_data, interaction=True),
        "claims",
        cv=2,
        n_alphas=4,
        stage2_alpha=0.03,
        **FIT_OPTIONS,
    )
    assert isinstance(fitted, TwoStageFit)
    rows = coefficient_path(fitted)
    assert rows.filter(pl.col("stage") == 1)["source"].unique().to_list() == [
        "cv_fold_mean"
    ]
    stage2 = rows.filter(pl.col("stage") == 2).sort("feature_index")
    assert stage2.height == fitted.stage2.spec.n_features
    assert stage2["alpha"].unique().to_list() == [0.03]
    assert stage2["source"].unique().to_list() == ["fixed_fit"]
    np.testing.assert_array_equal(stage2["coefficient"].to_numpy(), fitted.stage2.coef)


def test_tiny_neighbouring_alphas_do_not_all_receive_the_selected_marker(cv_fit):
    fitted = deepcopy(cv_fit)
    shape = fitted.model.alphas_.shape
    grid = np.geomspace(1e-9, 1e-13, shape[1])
    fitted.model.alphas_ = np.broadcast_to(grid, shape).copy()
    fitted.model.alpha_ = grid[2]
    fitted.model.l1_ratio_ = float(np.atleast_1d(fitted.model.l1_ratio)[0])
    selected = coefficient_path(fitted).filter(pl.col("selected"))
    assert selected.height == fitted.spec.n_features
    assert selected["alpha"].unique().to_list() == [grid[2]]


def test_mismatched_path_cannot_silently_attach_wrong_feature_metadata(cv_fit):
    fitted = deepcopy(cv_fit)
    fitted.model.coef_path_ = fitted.model.coef_path_[..., :-1]
    with pytest.raises(ValueError, match="design features"):
        coefficient_path(fitted)


def test_empty_feature_spec_returns_consistent_schema(cv_fit):
    fitted = deepcopy(cv_fit)
    fitted.spec = DesignSpec({})
    fitted.model.coef_path_ = fitted.model.coef_path_[..., :0]
    rows = coefficient_path(fitted)
    assert rows.is_empty()
    assert rows.schema == COEFFICIENT_PATH_SCHEMA
