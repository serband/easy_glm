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
    """Fit a L1-regularized GLM (Poisson/Gamma) using glum."""
    df = dataframe.to_pandas()
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
    if (not DivideTargetByWeight) and (weight_col is not None):
        df[target] = df[target] / df[weight_col]
    train_df = df[df[train_test_col] == 1]
    if train_df.empty:
        raise ValueError("Training subset is empty (no rows with traintest==1).")
    exclude = {target, train_test_col} | ({weight_col} if weight_col else set())
    def is_text_like(series: pd.Series) -> bool:
        dt = series.dtype
        return ptypes.is_object_dtype(dt) or ptypes.is_string_dtype(dt)
    text_cols = [c for c in train_df.columns if c not in exclude and is_text_like(train_df[c])]
    if text_cols:
        ddf = dd.from_pandas(train_df, npartitions=1)
        ddf = Categorizer(columns=text_cols).fit_transform(ddf)
        train_df = ddf.compute()
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


def _encode_like_training(df: pd.DataFrame, model: GeneralizedLinearRegressor) -> pd.DataFrame:
    """Best-effort encoding of object/string columns to mimic fit pipeline.

    This mirrors the minimal categorizer logic used in fit_lasso_glm so that
    downstream prediction on raw (possibly string-typed) data won't error.
    """
    obj_cols = [c for c in df.columns if ptypes.is_object_dtype(df[c].dtype) or ptypes.is_string_dtype(df[c].dtype)]
    if obj_cols:
        ddf = dd.from_pandas(df, npartitions=1)
        ddf = Categorizer(columns=obj_cols).fit_transform(ddf)
        df = ddf.compute()
    return df


def predict_with_model(
    model: GeneralizedLinearRegressor,
    new_data: pl.DataFrame | pd.DataFrame,
    return_polars: bool = False,
) -> np.ndarray | pl.Series:
    """Generate predictions on new data.

    Args:
        model: Fitted GeneralizedLinearRegressor from glum.
        new_data: Polars or pandas DataFrame containing the required feature columns.
        return_polars: If True, return a Polars Series; else a NumPy array.

    Returns:
        Predictions as numpy array or Polars Series.
    """
    if isinstance(new_data, pl.DataFrame):
        pdf = new_data.to_pandas()
    else:
        pdf = new_data.copy()
    pdf = _encode_like_training(pdf, model)
    preds = model.predict(pdf)
    if return_polars:
        return pl.Series(name="prediction", values=preds)
    return preds
