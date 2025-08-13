from typing import Optional, Sequence, Callable, Any
import polars as pl
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import dask.dataframe as dd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor
import functools

def typechecked_ratetable(func: Callable) -> Callable:
    """
    Decorator to check types of ratetable arguments at runtime for user feedback.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "model" in kwargs and not isinstance(kwargs["model"], GeneralizedLinearRegressor):
            raise TypeError("model must be a GeneralizedLinearRegressor")
        if "dataset" in kwargs and not isinstance(kwargs["dataset"], pl.DataFrame):
            raise TypeError("dataset must be a polars.DataFrame")
        if "col_name" in kwargs and not isinstance(kwargs["col_name"], str):
            raise TypeError("col_name must be a string")
        if "levels" in kwargs and not isinstance(kwargs["levels"], (list, tuple, np.ndarray)):
            raise TypeError("levels must be a sequence (list, tuple, or np.ndarray)")
        return func(*args, **kwargs)
    return wrapper

@typechecked_ratetable
def ratetable(
    *,
    model: GeneralizedLinearRegressor,
    dataset: pl.DataFrame,
    col_name: str,
    levels: Sequence[Any],
    prepare: Optional[Callable[[pl.DataFrame], pl.DataFrame]] = None,
    random_seed: Optional[int] = None,
    include_raw: bool = True,
) -> pl.DataFrame:
    """
    Build a rate table (ICE) by varying a single column over specified levels.

    This function generates a table showing the effect of changing one variable (e.g., 'VehAge')
    on the model's prediction, holding all other features constant. It is commonly used for
    insurance pricing, interpretability, and relativity analysis.

    Parameters
    ----------
    model : GeneralizedLinearRegressor
        A fitted GLM model (from glum) trained on pandas DataFrame.
    dataset : pl.DataFrame
        The source Polars DataFrame containing all features.
    col_name : str
        The name of the column to vary (e.g., 'VehAge').
    levels : Sequence
        The values to use for the column (e.g., breakpoints or levels).
    prepare : Callable, optional
        A function to preprocess the duplicated rows before prediction (e.g., easy_glm.prepare_data).
        Should accept and return a Polars DataFrame.
    random_seed : int, optional
        Seed for reproducible random row selection.
    include_raw : bool, default True
        If True, include the raw model prediction in the output table.

    Returns
    -------
    pl.DataFrame
        A tidy Polars DataFrame with columns:
        - col_name (the variable varied)
        - relativity (prediction / min(prediction), so min is always 1)
        - prediction (raw model output, if include_raw is True)

    Example
    -------
    >>> tbl = ratetable(
    ...     model=model,
    ...     dataset=df,               # your Polars df
    ...     col_name="VehAge",
    ...     levels=d['VehAge'],
    ...     prepare=lambda df: easy_glm.prepare_data(
    ...         df=df,
    ...         modelling_variables=predictor_variables,
    ...         formats=d,
    ...         table_name="line_prepped",
    ...     ),
    ...     random_seed=42,
    ... )
    ... print(tbl)
    shape: (17, 3)
    ┌────────┬────────────┬────────────┐
    │ VehAge ┆ relativity ┆ prediction │
    │ ---    ┆ ---        ┆ ---        │
    │ f64    ┆ f64        ┆ f64        │
    ╞════════╪════════════╪════════════╡
    │ 0.0    ┆ 14.501172  ┆ 0.044589   │
    │ 1.0    ┆ 19.245184  ┆ 0.059176   │
    │ ...    ┆ ...        ┆ ...        │
    │ 100.0  ┆ 1.0        ┆ 0.003075   │

    Notes
    -----
    - The function duplicates a random row, varies the target column, and predicts for each value.
    - Relativity is normalized so the median prediction is 1.
    - Use the 'prepare' argument to apply any necessary preprocessing (e.g., encoding, feature engineering).
    - Output is sorted by the varied column.
    - Works for both numeric and categorical columns.
    """

    # 1) Pick one random row
    random_row = dataset.sample(n=1, shuffle=True, seed=random_seed)

    # 2) Duplicate it N times
    N = len(levels)
    duplicated = pl.concat([random_row] * N, how="vertical")

    # 3) Replace the target column with the provided levels (row-wise)
    #    IMPORTANT: use pl.Series to avoid making a list column
    duplicated = duplicated.with_columns(pl.Series(col_name, list(levels)))

    # 4) Optional pre-processing hook (your own function)
    if prepare is not None:
        duplicated = prepare(duplicated)

    # 5) Convert to pandas and categorize string/object columns
    pdf = duplicated.to_pandas()

    # pick text-like columns except obvious non-features if needed
    # (keep simple: categorize all object/string columns)
    obj_cols = [c for c in pdf.columns if ptypes.is_object_dtype(pdf[c].dtype) or ptypes.is_string_dtype(pdf[c].dtype)]
    if obj_cols:
        ddf = dd.from_pandas(pdf, npartitions=1)
        ddf = Categorizer(columns=obj_cols).fit_transform(ddf)
        pdf = ddf.compute()

    # 6) Predict
    preds = model.predict(pdf)
    preds = np.asarray(preds, dtype=float)

    # 7) Normalize to median value == 1
    base = np.median(preds)
    relativity = preds / base if base != 0 else np.nan

    # 8) Return a tidy Polars table
    out = {
        col_name: list(levels),
        "relativity": relativity.tolist(),
    }
    if include_raw:
        out["prediction"] = preds.tolist()

    return pl.DataFrame(out).sort(col_name)