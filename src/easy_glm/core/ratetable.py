import functools
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas.api.types as ptypes
import polars as pl
from glum import (
    GeneralizedLinearRegressor,
    GeneralizedLinearRegressorCV,
)


def typechecked_ratetable(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "model" in kwargs and not isinstance(
            kwargs["model"], GeneralizedLinearRegressor | GeneralizedLinearRegressorCV
        ):
            raise TypeError(
                "model must be a GeneralizedLinearRegressor or "
                "GeneralizedLinearRegressorCV"
            )
        if "dataset" in kwargs and not isinstance(kwargs["dataset"], pl.DataFrame):
            raise TypeError("dataset must be a polars.DataFrame")
        if "col_name" in kwargs and not isinstance(kwargs["col_name"], str):
            raise TypeError("col_name must be a string")
        if "levels" in kwargs and not isinstance(
            kwargs["levels"], list | tuple | np.ndarray
        ):
            raise TypeError("levels must be a sequence")
        return func(*args, **kwargs)

    return wrapper


@typechecked_ratetable
def ratetable(
    *,
    model: GeneralizedLinearRegressor | GeneralizedLinearRegressorCV,
    dataset: pl.DataFrame,
    col_name: str,
    levels: Sequence[Any],
    prepare: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    random_seed: int | None = None,
    include_raw: bool = True,
) -> pl.DataFrame:
    """Build a rate table for a single predictor.

    Picks one representative row from *dataset*, duplicates it once per
    factor level, replaces the target column with each level, computes
    predictions through *model*, and derives relativities relative to the
    median prediction.

    Because a GLM with log link is multiplicative
    (``prediction = exp(intercept + β₁x₁ + … + βₙxₙ)``), the relativity
    **ratios** between two levels of the same variable are independent of
    the other feature values — the single-row approach gives the correct
    shape.  The median baseline only affects the absolute scale of the
    relativities, not their relative ordering.

    For models with interactions or non-linear terms, consider using a
    full-dataset marginal-effects approach instead.

    .. deprecated:: 0.3
        Use :func:`easy_glm.rate_tables`, which reads exact relativities off
        the coefficients.
    """
    warnings.warn(
        "ratetable is deprecated since easy_glm 0.3 and will be removed in 0.4; "
        "use rate_tables instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    random_row = dataset.sample(n=1, shuffle=True, seed=random_seed)
    duplicated = pl.concat([random_row] * len(levels), how="vertical").with_columns(
        pl.Series(col_name, list(levels))
    )
    if prepare is not None:
        duplicated = prepare(duplicated)
    pdf = duplicated.to_pandas()
    obj_cols = [
        c
        for c in pdf.columns
        if ptypes.is_object_dtype(pdf[c].dtype) or ptypes.is_string_dtype(pdf[c].dtype)
    ]
    for col in obj_cols:
        pdf[col] = pdf[col].astype("category")
    preds = np.asarray(model.predict(pdf), dtype=float)
    base = np.median(preds)
    relativity = preds / base if base != 0 else np.full_like(preds, np.nan)
    out: dict[str, Any] = {col_name: list(levels), "relativity": relativity.tolist()}
    if include_raw:
        out["prediction"] = preds.tolist()
    return pl.DataFrame(out).sort(col_name)
