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
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "model" in kwargs and not isinstance(kwargs["model"], GeneralizedLinearRegressor):
            raise TypeError("model must be a GeneralizedLinearRegressor")
        if "dataset" in kwargs and not isinstance(kwargs["dataset"], pl.DataFrame):
            raise TypeError("dataset must be a polars.DataFrame")
        if "col_name" in kwargs and not isinstance(kwargs["col_name"], str):
            raise TypeError("col_name must be a string")
        if "levels" in kwargs and not isinstance(kwargs["levels"], (list, tuple, np.ndarray)):
            raise TypeError("levels must be a sequence")
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
    random_row = dataset.sample(n=1, shuffle=True, seed=random_seed)
    duplicated = pl.concat([random_row] * len(levels), how="vertical").with_columns(pl.Series(col_name, list(levels)))
    if prepare is not None:
        duplicated = prepare(duplicated)
    pdf = duplicated.to_pandas()
    obj_cols = [c for c in pdf.columns if ptypes.is_object_dtype(pdf[c].dtype) or ptypes.is_string_dtype(pdf[c].dtype)]
    if obj_cols:
        ddf = dd.from_pandas(pdf, npartitions=1)
        ddf = Categorizer(columns=obj_cols).fit_transform(ddf)
        pdf = ddf.compute()
    preds = np.asarray(model.predict(pdf), dtype=float)
    base = np.median(preds)
    relativity = preds / base if base != 0 else np.nan
    out = {col_name: list(levels), "relativity": relativity.tolist()}
    if include_raw:
        out["prediction"] = preds.tolist()
    return pl.DataFrame(out).sort(col_name)
