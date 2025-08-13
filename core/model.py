from typing import Optional
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import dask.dataframe as dd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor
import polars as pl

def fit_lasso_glm(
    dataframe: pl.DataFrame,
    target: str,
    train_test_col: str,
    model_type: str,
    weight_col: Optional[str] = None,
    DivideTargetByWeight: bool = False,
) -> GeneralizedLinearRegressor:

    # --- Polars -> pandas (no 'copy' kw) ---
    df = dataframe.to_pandas()

    # --- Basic validation ---
    if df.shape[0] == 0:
        raise ValueError("The input DataFrame is empty.")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names.")

    required = [target, train_test_col] + ([weight_col] if weight_col else [])
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'.")

    fam = model_type.lower()
    if fam not in {"poisson", "gamma"}:
        raise ValueError("model_type must be 'Poisson' or 'Gamma'.")

    # --- Target/weight checks ---
    def invalid_target_exists(pdf: pd.DataFrame, col: str, allow_zero: bool) -> bool:
        s = pdf[col]
        bad = ((s < 0) if allow_zero else (s <= 0)) | np.isinf(s) | s.isna()
        return bool(bad.any())

    if fam == "poisson":
        if invalid_target_exists(df, target, allow_zero=True):
            raise ValueError("Invalid Poisson target values (<0, inf, or NaN).")
    else:
        if invalid_target_exists(df, target, allow_zero=False):
            raise ValueError("Invalid Gamma target values (<=0, inf, or NaN).")

    if weight_col:
        w = df[weight_col]
        if bool(((w <= 0) | np.isinf(w) | w.isna()).any()):
            raise ValueError("Weight column has invalid values (<=0, inf, NaN).")

    # --- Optional target adjustment (unchanged logic) ---
    if (not DivideTargetByWeight) and (weight_col is not None):
        df[target] = df[target] / df[weight_col]

    # --- Train subset ---
    train_df = df[df[train_test_col] == 1]
    if train_df.empty:
        raise ValueError("Training subset is empty (no rows with traintest==1).")

    # --- Pick text-like features and categorize via dask-ml ---
    exclude = {target, train_test_col} | ({weight_col} if weight_col else set())
    def is_text_like(series: pd.Series) -> bool:
        dt = series.dtype
        return (
            ptypes.is_object_dtype(dt)
            or ptypes.is_string_dtype(dt)   # catches pandas StringDtype / string[pyarrow]
        )

    text_cols = [c for c in train_df.columns if c not in exclude and is_text_like(train_df[c])]
    if text_cols:
        ddf = dd.from_pandas(train_df, npartitions=1)
        ddf = Categorizer(columns=text_cols).fit_transform(ddf)  # set dtype='category'
        train_df = ddf.compute()

    # --- Fit glum ---
    features = [c for c in train_df.columns if c not in exclude]
    X = train_df[features]
    y = train_df[target].to_numpy().ravel()
    sw = train_df[weight_col].to_numpy().ravel() if weight_col else None

    model = GeneralizedLinearRegressor(
        family=fam,
        l1_ratio=1,
        fit_intercept=True,
        alpha_search=True,
        scale_predictors=True,
    )
    model.fit(X, y, sample_weight=sw)
    return model
