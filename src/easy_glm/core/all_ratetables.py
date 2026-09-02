import warnings
from typing import Any

import polars as pl
from glum import (
    GeneralizedLinearRegressor,
    GeneralizedLinearRegressorCV,
)

from .prepare import prepare_data
from .ratetable import ratetable


def _make_prepare_fn(
    predictor_variables: list[str],
    blueprint: dict[str, Any],
):
    """Return a callable suitable as ``ratetable(..., prepare=...)``."""

    def _prepare(df: pl.DataFrame) -> pl.DataFrame:
        return prepare_data(
            df=df,
            modelling_variables=predictor_variables,
            formats=blueprint,
            table_name="line_prepped",
        )

    return _prepare


def generate_all_ratetables(
    model: GeneralizedLinearRegressor | GeneralizedLinearRegressorCV,
    dataset: pl.DataFrame,
    predictor_variables: list[str],
    blueprint: dict[str, Any],
    random_seed: int = 42,
) -> dict[str, pl.DataFrame]:
    """Generate rate tables for each predictor in predictor_variables.

    .. deprecated:: 0.3
        Use :func:`easy_glm.rate_tables`.
    """
    warnings.warn(
        "generate_all_ratetables is deprecated since easy_glm 0.3 and will be removed in 0.4; "
        "use rate_tables instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    prepare_fn = _make_prepare_fn(predictor_variables, blueprint)
    all_ratetables: dict[str, pl.DataFrame] = {}
    for var in predictor_variables:
        levels = blueprint.get(var)
        if levels is None:
            print(f"Warning: No blueprint found for variable '{var}'. Skipping.")
            continue
        tbl = ratetable(
            model=model,
            dataset=dataset,
            col_name=var,
            levels=levels,
            prepare=prepare_fn,
            random_seed=random_seed,
        )
        all_ratetables[var] = tbl
    return all_ratetables
