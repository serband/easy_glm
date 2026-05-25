import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import polars as pl
from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV


def fit_lasso_glm(
    dataframe: pl.DataFrame,
    target: str,
    train_test_col: str,
    model_type: str,
    weight_col: str | None = None,
    divide_target_by_weight: bool = False,
    use_cv: bool = True,
    cv_params: dict | None = None,
) -> GeneralizedLinearRegressor | GeneralizedLinearRegressorCV:
    """Fit a L1-regularized GLM (Poisson/Gamma/Gaussian/Binomial) using glum.

    Args:
        dataframe: Input Polars DataFrame.
        target: Name of the target column.
        train_test_col: Column with 1=train, 0=test.
        model_type: 'Poisson', 'Gamma', 'Gaussian', or 'Binomial'.
        weight_col: Optional sample weight column.
        divide_target_by_weight: If True, divide target by weight before fitting.
        use_cv: If True (default), use GeneralizedLinearRegressorCV for
            cross-validated alpha/l1_ratio selection. If False, use
            GeneralizedLinearRegressor with alpha_search.
        cv_params: Optional dict of keyword arguments forwarded to
            GeneralizedLinearRegressorCV. Applied on top of defaults:
            alphas=None, l1_ratio=[0, 0.5, 1.0], max_iter=150,
            fit_intercept=True, scale_predictors=True.
    """
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
    if fam not in {"poisson", "gamma", "gaussian", "binomial"}:
        raise ValueError(
            "model_type must be 'Poisson', 'Gamma', 'Gaussian', or 'Binomial'."
        )

    def _any_bad(s: pd.Series) -> bool:
        return bool((np.isinf(s) | s.isna()).any())

    if fam == "poisson":
        s = df[target]
        if bool(((s < 0) | np.isinf(s) | s.isna()).any()):
            raise ValueError("Invalid Poisson target values (<0, inf, or NaN).")
    elif fam == "gamma":
        s = df[target]
        if bool(((s <= 0) | np.isinf(s) | s.isna()).any()):
            raise ValueError("Invalid Gamma target values (<=0, inf, or NaN).")
    elif fam == "gaussian":
        if _any_bad(df[target]):
            raise ValueError("Invalid Gaussian target values (inf or NaN).")
    elif fam == "binomial":
        s = df[target]
        if bool(((s < 0) | (s > 1) | np.isinf(s) | s.isna()).any()):
            raise ValueError(
                "Invalid Binomial target values (outside [0, 1], inf, or NaN)."
            )
    if weight_col:
        w = df[weight_col]
        if bool(((w <= 0) | np.isinf(w) | w.isna()).any()):
            raise ValueError("Weight column has invalid values (<=0, inf, NaN).")
    if divide_target_by_weight and (weight_col is not None):
        df[target] = df[target] / df[weight_col]
    train_df = df[df[train_test_col] == 1]
    if train_df.empty:
        raise ValueError("Training subset is empty (no rows with traintest==1).")
    exclude = {target, train_test_col} | ({weight_col} if weight_col else set())

    def is_text_like(series: pd.Series) -> bool:
        dt = series.dtype
        return ptypes.is_object_dtype(dt) or ptypes.is_string_dtype(dt)

    text_cols = [
        c for c in train_df.columns if c not in exclude and is_text_like(train_df[c])
    ]
    for col in text_cols:
        train_df[col] = train_df[col].astype("category")

    features = [c for c in train_df.columns if c not in exclude]
    x_data = train_df[features]
    y = train_df[target].to_numpy().ravel()
    sw = train_df[weight_col].to_numpy().ravel() if weight_col else None

    if use_cv:
        cv_defaults: dict = {
            "family": fam,
            "alphas": None,
            "l1_ratio": [0, 0.5, 1.0],
            "max_iter": 150,
            "fit_intercept": True,
            "scale_predictors": True,
        }
        if cv_params:
            cv_defaults.update(cv_params)
        model = GeneralizedLinearRegressorCV(**cv_defaults)
    else:
        model = GeneralizedLinearRegressor(
            family=fam,
            l1_ratio=1,
            fit_intercept=True,
            alpha_search=True,
            scale_predictors=True,
        )
    model.fit(x_data, y, sample_weight=sw)
    return model


def _encode_like_training(
    df: pd.DataFrame, model: GeneralizedLinearRegressor | GeneralizedLinearRegressorCV
) -> pd.DataFrame:
    """Best-effort encoding of object/string columns to mimic fit pipeline.

    Converts object/string columns to integer category codes so that
    glum (which requires numeric input) can predict without error.
    """
    obj_cols = [
        c
        for c in df.columns
        if ptypes.is_object_dtype(df[c].dtype) or ptypes.is_string_dtype(df[c].dtype)
    ]
    for col in obj_cols:
        df[col] = df[col].astype("category")
    return df


def predict_with_model(
    model: GeneralizedLinearRegressor | GeneralizedLinearRegressorCV,
    new_data: pl.DataFrame | pd.DataFrame,
    return_polars: bool = False,
) -> np.ndarray | pl.Series:
    """Generate predictions on new data.

    Args:
        model: Fitted GeneralizedLinearRegressor or GeneralizedLinearRegressorCV
            from glum.
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
