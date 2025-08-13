from typing import Dict, Any, List
import polars as pl
from glum import GeneralizedLinearRegressor
from .prepare import prepare_data
from .ratetable import ratetable

def generate_all_ratetables(
    model: GeneralizedLinearRegressor,
    dataset: pl.DataFrame,
    predictor_variables: List[str],
    blueprint: Dict[str, Any],
    random_seed: int = 42,
) -> Dict[str, pl.DataFrame]:
    """Generate rate tables for each predictor in predictor_variables."""
    all_ratetables: Dict[str, pl.DataFrame] = {}
    for var in predictor_variables:
        levels = blueprint.get(var)
        if levels is None:
            print(f"Warning: No blueprint found for variable '{var}'. Skipping.")
            continue
        prepare_fn = lambda d: prepare_data(  # noqa: E731
            df=d,
            modelling_variables=predictor_variables,
            formats=blueprint,
            table_name="line_prepped",
        )
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
