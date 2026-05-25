import numpy as np
import polars as pl
import pytest

from easy_glm.benchmarking.data_generators import (
    generate_all_datasets,
    generate_binomial_dataset,
    generate_gamma_dataset,
    generate_gaussian_dataset,
    generate_poisson_dataset,
)
from easy_glm.benchmarking.metrics import (
    binomial_deviance,
    compute_all_metrics,
    gamma_deviance,
    gaussian_deviance,
    mae,
    poisson_deviance,
    rmse,
)
from easy_glm.core.model import fit_lasso_glm


class TestDataGenerators:
    N_ROWS = 200
    SEED = 123

    @pytest.mark.parametrize(
        "gen_fn,family",
        [
            (generate_poisson_dataset, "poisson"),
            (generate_gamma_dataset, "gamma"),
            (generate_gaussian_dataset, "gaussian"),
            (generate_binomial_dataset, "binomial"),
        ],
    )
    def test_shape_and_columns(self, gen_fn, family):
        df = gen_fn(n_rows=self.N_ROWS, seed=self.SEED)
        assert isinstance(df, pl.DataFrame)
        assert df.height == self.N_ROWS
        expected_cols = {
            *(f"num_{i}" for i in range(30)),
            *(f"cat_{i}" for i in range(20)),
            "Exposure",
            "Response",
            "traintest",
        }
        assert set(df.columns) == expected_cols

    def test_all_numeric_columns_are_numeric(self):
        df = generate_poisson_dataset(n_rows=self.N_ROWS, seed=self.SEED)
        for i in range(30):
            col = f"num_{i}"
            assert df[col].dtype.is_numeric(), f"{col} should be numeric"
        assert df["Exposure"].dtype.is_numeric()
        assert df["Response"].dtype.is_numeric()
        assert df["traintest"].dtype.is_integer()

    def test_all_categorical_columns_are_string(self):
        df = generate_poisson_dataset(n_rows=self.N_ROWS, seed=self.SEED)
        for i in range(20):
            col = f"cat_{i}"
            assert df[col].dtype == pl.String, f"{col} should be string"

    def test_poisson_response_nonnegative_integer(self):
        df = generate_poisson_dataset(n_rows=self.N_ROWS, seed=self.SEED)
        response = df["Response"].to_numpy()
        assert np.all(response >= 0)
        assert np.all(np.mod(response, 1) == 0)

    def test_gamma_response_positive(self):
        df = generate_gamma_dataset(n_rows=self.N_ROWS, seed=self.SEED)
        assert df["Response"].min() > 0

    def test_binomial_response_zero_one(self):
        df = generate_binomial_dataset(n_rows=self.N_ROWS, seed=self.SEED)
        response = df["Response"].to_numpy()
        assert np.all((response == 0) | (response == 1))

    def test_exposure_positive(self):
        for gen_fn in [
            generate_poisson_dataset,
            generate_gamma_dataset,
            generate_gaussian_dataset,
            generate_binomial_dataset,
        ]:
            df = gen_fn(n_rows=self.N_ROWS, seed=self.SEED)
            assert df["Exposure"].min() > 0

    def test_traintest_split(self):
        df = generate_poisson_dataset(n_rows=1000, seed=self.SEED)
        counts = df["traintest"].value_counts()
        train_count = counts.filter(pl.col("traintest") == 1)["count"].item()
        assert 650 <= train_count <= 850

    def test_deterministic_with_seed(self):
        df1 = generate_poisson_dataset(n_rows=self.N_ROWS, seed=42)
        df2 = generate_poisson_dataset(n_rows=self.N_ROWS, seed=42)
        assert df1.equals(df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_poisson_dataset(n_rows=self.N_ROWS, seed=1)
        df2 = generate_poisson_dataset(n_rows=self.N_ROWS, seed=2)
        assert not df1.equals(df2)

    def test_no_nulls(self):
        for gen_fn in [
            generate_poisson_dataset,
            generate_gamma_dataset,
            generate_gaussian_dataset,
            generate_binomial_dataset,
        ]:
            df = gen_fn(n_rows=self.N_ROWS, seed=self.SEED)
            nans = df.null_count()
            total = sum(nans.row(0))
            assert total == 0

    def test_no_infinities(self):
        for gen_fn in [
            generate_poisson_dataset,
            generate_gamma_dataset,
            generate_gaussian_dataset,
            generate_binomial_dataset,
        ]:
            df = gen_fn(n_rows=self.N_ROWS, seed=self.SEED)
            for col in df.columns:
                if df[col].dtype.is_numeric():
                    arr = df[col].to_numpy()
                    assert np.all(np.isfinite(arr)), f"Infs in {col}"

    def test_generate_all_datasets(self):
        datasets = generate_all_datasets(seed=self.SEED, n_rows=100)
        assert set(datasets.keys()) == {"poisson", "gamma", "gaussian", "binomial"}
        for df in datasets.values():
            assert isinstance(df, pl.DataFrame)
            assert df.height == 100


class TestMetrics:
    def test_poisson_deviance_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        d = poisson_deviance(y, y)
        assert abs(d) < 1e-10

    def test_gamma_deviance_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        d = gamma_deviance(y, y)
        assert abs(d) < 1e-10

    def test_gaussian_deviance_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        d = gaussian_deviance(y, y)
        assert abs(d) < 1e-10

    def test_binomial_deviance_perfect(self):
        y = np.array([0.0, 1.0, 1.0])
        y_pred = np.array([1e-10, 1 - 1e-10, 1 - 1e-10])
        d = binomial_deviance(y, y_pred)
        assert d < 1e-8

    def test_binomial_deviance_worst(self):
        y = np.array([0.0, 1.0, 0.0])
        y_pred = np.array([1 - 1e-10, 1e-10, 1 - 1e-10])
        d = binomial_deviance(y, y_pred)
        assert d > 10

    def test_rmse_zero_for_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert abs(rmse(y, y)) < 1e-10

    def test_rmse_nonzero_for_imperfect(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([1.0, 1.0, 2.0])
        assert rmse(y_true, y_pred) > 0.5

    def test_mae(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([1.0, 1.0, 2.0])
        assert abs(mae(y_true, y_pred) - 1.0 / 3.0) < 1e-10

    def test_compute_all_metrics_returns_correct_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        result = compute_all_metrics(y, y, "poisson")
        assert set(result.keys()) == {"deviance", "rmse", "mae"}


class TestFitLassoGlmNewFamilies:
    def test_gaussian_family_fits(self):
        df = generate_gaussian_dataset(n_rows=500, seed=42)
        numeric_cols = [
            c for c in df.columns if c.startswith("num_") and c in df.columns
        ]
        subset = df.select(numeric_cols + ["Response", "Exposure", "traintest"])
        model = fit_lasso_glm(
            dataframe=subset,
            target="Response",
            model_type="Gaussian",
            train_test_col="traintest",
            use_cv=False,
        )
        assert model.coef_ is not None

    def test_binomial_family_fits(self):
        df = generate_binomial_dataset(n_rows=500, seed=42)
        numeric_cols = [
            c for c in df.columns if c.startswith("num_") and c in df.columns
        ]
        subset = df.select(numeric_cols + ["Response", "Exposure", "traintest"])
        model = fit_lasso_glm(
            dataframe=subset,
            target="Response",
            model_type="Binomial",
            train_test_col="traintest",
            use_cv=False,
        )
        assert model.coef_ is not None

    def test_gaussian_with_cv_fits(self):
        df = generate_gaussian_dataset(n_rows=500, seed=42)
        numeric_cols = [
            c for c in df.columns if c.startswith("num_") and c in df.columns
        ]
        subset = df.select(numeric_cols + ["Response", "Exposure", "traintest"])
        model = fit_lasso_glm(
            dataframe=subset,
            target="Response",
            model_type="Gaussian",
            train_test_col="traintest",
            use_cv=True,
            cv_params={"alphas": [0.01, 0.1, 1.0]},
        )
        assert model.coef_ is not None

    def test_invalid_gaussian_target_rejected(self):
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [np.nan, 1.0],
                "traintest": [1, 1],
            }
        )
        with pytest.raises(ValueError, match="Gaussian"):
            fit_lasso_glm(df, "y", "traintest", "Gaussian", use_cv=False)

    def test_invalid_binomial_target_rejected(self):
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [-0.5, 0.8],
                "traintest": [1, 1],
            }
        )
        with pytest.raises(ValueError, match="Binomial"):
            fit_lasso_glm(df, "y", "traintest", "Binomial", use_cv=False)

    def test_invalid_family_rejected(self):
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "traintest": [1, 1],
            }
        )
        with pytest.raises(ValueError, match="must be"):
            fit_lasso_glm(df, "y", "traintest", "InvalidFamily", use_cv=False)


class TestBenchmarkRunner:
    def test_run_benchmarks_with_small_dataset(self):
        from easy_glm.benchmarking.benchmark import run_benchmarks
        from easy_glm.benchmarking.data_generators import generate_all_datasets

        datasets = generate_all_datasets(seed=42, n_rows=300)
        result = run_benchmarks(datasets=datasets, n_rows=300)
        assert isinstance(result, pl.DataFrame)
        assert result.height >= 4
        expected_cols = {
            "Dataset",
            "Method",
            "Deviance",
            "RMSE",
            "MAE",
            "FitTime_s",
            "PredTime_s",
            "NParams",
        }
        assert set(result.columns) == expected_cols
        methods = result["Method"].unique().to_list()
        assert "statsmodels" in methods
        assert "catboost" in methods

    def test_benchmark_metrics_are_positive(self):
        from easy_glm.benchmarking.benchmark import run_benchmarks
        from easy_glm.benchmarking.data_generators import generate_all_datasets

        datasets = generate_all_datasets(seed=42, n_rows=300)
        result = run_benchmarks(datasets=datasets, n_rows=300)
        successful = result.filter(pl.col("Deviance").is_not_null())
        assert successful.height > 0
        assert successful["Deviance"].min() >= 0
        assert successful["RMSE"].min() >= 0
        assert successful["MAE"].min() >= 0
