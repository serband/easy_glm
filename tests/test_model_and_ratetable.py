import numpy as np
import polars as pl

from easy_glm import (
    fit_lasso_glm,
    generate_blueprint,
    predict_with_model,
    prepare_data,
    ratetable,
)
from easy_glm.core.data import load_external_dataframe


def _sample_dataset(n: int = 400, missing_frac: float = 0.1) -> pl.DataFrame:
    df = load_external_dataframe()
    if df.height > n:
        df = df.head(n)
    rng = np.random.default_rng(42)
    mask = (rng.random(df.height) < 0.7).astype(np.int8)
    df = df.with_columns(pl.Series("traintest", mask))

    # Introduce missing values
    if missing_frac > 0:
        for col_name, col_type in df.schema.items():
            if col_name in ["VehAge", "Region"]:  # Target specific columns for NA
                na_mask = rng.random(df.height) < missing_frac
                df = df.with_columns(
                    pl.when(na_mask)
                    .then(None)
                    .otherwise(df[col_name])
                    .alias(col_name)
                )
    return df


def test_fit_lasso_glm_and_predict():
    df = _sample_dataset()
    predictors = ["VehAge", "Region", "VehGas"]
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


def test_ratetable_basic():
    df = _sample_dataset()
    predictors = ["VehAge", "Region", "VehGas"]
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
