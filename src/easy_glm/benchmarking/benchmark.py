from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from easy_glm.benchmarking.metrics import compute_all_metrics
from easy_glm.core.model import fit_lasso_glm, predict_with_model

_FAMILIES: dict[str, str] = {
    "poisson": "Poisson",
    "gamma": "Gamma",
    "gaussian": "Gaussian",
    "binomial": "Binomial",
}

_CATBOOST_OBJECTIVES: dict[str, str] = {
    "poisson": "Poisson",
    "gamma": "RMSE",
    "gaussian": "RMSE",
    "binomial": "Logloss",
}


def _prep_train_test(
    df: pl.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    pdf = df.to_pandas()
    predictor_cols = [
        c for c in pdf.columns if c not in {"Response", "Exposure", "traintest"}
    ]
    numeric_cols = [c for c in predictor_cols if c.startswith("num_")]
    cat_cols = [c for c in predictor_cols if c.startswith("cat_")]

    x_features = pdf[numeric_cols].copy()
    for c in cat_cols:
        dummies = pd.get_dummies(pdf[c], prefix=c, drop_first=True).astype(np.float64)
        x_features = pd.concat([x_features, dummies], axis=1)

    y = pdf["Response"].to_numpy()
    w = pdf["Exposure"].to_numpy()
    is_train = pdf["traintest"].to_numpy() == 1

    x_train = x_features.iloc[is_train].reset_index(drop=True)
    x_test = x_features.iloc[~is_train].reset_index(drop=True)
    y_train = y[is_train]
    y_test = y[~is_train]
    w_train = w[is_train]
    w_test = w[~is_train]

    return x_train, y_train, w_train, x_test, y_test, w_test


def _fit_easy_glm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    family: str,
    use_cv: bool,
) -> tuple[Any, float, float, np.ndarray, int]:
    fam_label = _FAMILIES[family]
    df = pl.DataFrame(x_train)
    df = df.with_columns(
        pl.Series("Response", y_train),
        pl.Series("traintest", np.ones(len(y_train), dtype=np.int8)),
    )
    if w_train is not None:
        df = df.with_columns(pl.Series("Exposure", w_train))

    t0 = time.perf_counter()
    model = fit_lasso_glm(
        dataframe=df,
        target="Response",
        model_type=fam_label,
        weight_col="Exposure" if w_train is not None else None,
        train_test_col="traintest",
        use_cv=use_cv,
    )
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = predict_with_model(model, x_train)
    pred_time = time.perf_counter() - t0

    n_params = len(model.coef_)
    return model, fit_time, pred_time, y_pred, n_params


def _fit_statsmodels(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    family: str,
) -> tuple[Any, float, float, np.ndarray, int]:
    import statsmodels.api as sm

    sm_families = {
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(sm.families.links.Log()),
        "gaussian": sm.families.Gaussian(),
        "binomial": sm.families.Binomial(),
    }

    t0 = time.perf_counter()
    x_design = sm.add_constant(x_train.to_numpy(dtype=np.float64))
    weights = w_train if w_train is not None else np.ones_like(y_train)
    glm_model = sm.GLM(
        y_train, x_design, family=sm_families[family], freq_weights=weights
    )
    result = glm_model.fit()
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    x_design_pred = sm.add_constant(x_train.to_numpy(dtype=np.float64))
    y_pred = result.predict(x_design_pred)
    pred_time = time.perf_counter() - t0

    n_params = len(result.params)
    return result, fit_time, pred_time, y_pred, n_params


def _fit_catboost(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    family: str,
) -> tuple[Any, float, float, np.ndarray, int]:
    from catboost import CatBoostRegressor

    objective = _CATBOOST_OBJECTIVES[family]
    if family == "gamma":
        y_train_clipped = np.clip(y_train, 1e-6, None)
    else:
        y_train_clipped = y_train.copy()

    t0 = time.perf_counter()
    model = CatBoostRegressor(
        loss_function=objective,
        iterations=500,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        x_train,
        y_train_clipped,
        sample_weight=w_train if w_train is not None else None,
    )
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(x_train)
    pred_time = time.perf_counter() - t0

    n_params = model.tree_count_
    return model, fit_time, pred_time, y_pred, n_params


def _run_dataset_benchmarks(
    dataset_name: str,
    df: pl.DataFrame,
) -> list[dict[str, Any]]:
    family_key = dataset_name.lower()
    fam_label = _FAMILIES[family_key]
    results: list[dict[str, Any]] = []

    x_train, y_train, w_train, x_test, y_test, w_test = _prep_train_test(df)

    print(
        f"\n  Benchmarking {family_key} ({x_train.shape[1]} features, "
        f"{x_train.shape[0]} train / {x_test.shape[0]} test rows)..."
    )

    rows = [
        (
            "easy_glm (no CV)",
            lambda x=x_train, y=y_train, w=w_train, fl=fam_label: _fit_easy_glm(
                x, y, w, fl, use_cv=False
            ),
        ),
        (
            "easy_glm (CV)",
            lambda x=x_train, y=y_train, w=w_train, fl=fam_label: _fit_easy_glm(
                x, y, w, fl, use_cv=True
            ),
        ),
        (
            "statsmodels",
            lambda x=x_train, y=y_train, w=w_train, fk=family_key: _fit_statsmodels(
                x, y, w, fk
            ),
        ),
        (
            "catboost",
            lambda x=x_train, y=y_train, w=w_train, fk=family_key: _fit_catboost(
                x, y, w, fk
            ),
        ),
    ]

    for method_name, fit_fn in rows:
        print(f"    Fitting {method_name}...", end=" ", flush=True)
        try:
            _model, fit_t, pred_t, y_train_pred, n_params = fit_fn()
        except Exception as exc:
            print(f"FAILED: {exc}")
            results.append(
                {
                    "Dataset": dataset_name,
                    "Method": method_name,
                    "Deviance": None,
                    "RMSE": None,
                    "MAE": None,
                    "FitTime_s": None,
                    "PredTime_s": None,
                    "NParams": None,
                }
            )
            continue

        t0 = time.perf_counter()
        if method_name.startswith("easy_glm"):
            y_test_pred = predict_with_model(_model, pl.DataFrame(x_test))
        elif method_name == "statsmodels":
            import statsmodels.api as sm

            x_stest = sm.add_constant(x_test.to_numpy(dtype=np.float64))
            y_test_pred = _model.predict(x_stest)
        else:
            y_test_pred = _model.predict(x_test)

        test_pred_time = time.perf_counter() - t0

        if family_key == "gamma":
            y_test_pred = np.clip(y_test_pred, 1e-6, None)
        if family_key == "binomial":
            y_test_pred = np.clip(y_test_pred, 1e-15, 1 - 1e-15)

        metrics = compute_all_metrics(y_test, y_test_pred, family_key)
        print("done")
        results.append(
            {
                "Dataset": dataset_name,
                "Method": method_name,
                "Deviance": round(metrics["deviance"], 4),
                "RMSE": round(metrics["rmse"], 4),
                "MAE": round(metrics["mae"], 4),
                "FitTime_s": round(fit_t, 3),
                "PredTime_s": round(pred_t + test_pred_time, 4),
                "NParams": n_params,
            }
        )

    return results


def run_benchmarks(
    datasets: dict[str, pl.DataFrame] | None = None,
    seed: int = 42,
    n_rows: int = 200_000,
) -> pl.DataFrame:
    from easy_glm.benchmarking.data_generators import generate_all_datasets

    if datasets is None:
        print(f"Generating synthetic datasets ({n_rows:,} rows, seed={seed})...")
        datasets = generate_all_datasets(seed=seed, n_rows=n_rows)
        print("Done generating data.\n")

    all_results: list[dict[str, Any]] = []
    for name, df in datasets.items():
        all_results.extend(_run_dataset_benchmarks(name, df))

    results_df = pl.DataFrame(all_results)

    col_widths = {
        "Dataset": 10,
        "Method": 18,
        "Deviance": 12,
        "RMSE": 12,
        "MAE": 12,
        "FitTime_s": 10,
        "PredTime_s": 10,
        "NParams": 10,
    }

    header = "  ".join(f"{k:<{v}}" for k, v in col_widths.items())
    sep = "  ".join("-" * v for v in col_widths.values())
    bar = "=" * (len(header) + 2)

    print(f"\n{bar}")
    print(f"  Benchmark Results — {n_rows:,} rows, {seed=}")
    print(bar)
    print(header)
    print(sep)

    for row in all_results:
        line = "  ".join(
            str(row.get(k, "") if row.get(k) is not None else "FAIL")[: v - 1].ljust(v)
            for k, v in col_widths.items()
        )
        print(line)

    print(bar)
    print()

    return results_df


if __name__ == "__main__":
    run_benchmarks()
