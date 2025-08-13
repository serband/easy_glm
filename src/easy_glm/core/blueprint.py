from typing import Dict, Any
import numpy as np
import polars as pl
from .transforms import lump_rare_levels_pl

def generate_blueprint(dataframe: pl.DataFrame, threshold: float = 0.0025) -> Dict[str, Any]:
    """Generate preprocessing blueprint for each column.

    Numeric columns -> quantile breakpoints (5%..100% step 5%)
    Categorical columns -> retained levels after lumping rare ones to 'Other'.
    """
    blueprint: Dict[str, Any] = {}
    for column in dataframe.columns:
        try:
            col_data = dataframe[column]
            dtype = col_data.dtype
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                quantiles = np.arange(0.05, 1.05, 0.05).tolist()
                breaks = [col_data.quantile(q) for q in quantiles]
                unique_breaks = sorted(set(breaks))
                blueprint[column] = unique_breaks
            else:
                lumped_levels = lump_rare_levels_pl(dataframe[column], threshold=threshold)
                levels = np.unique(lumped_levels).tolist()
                if 'Other' in levels:
                    levels.remove('Other')
                blueprint[column] = levels
        except Exception as e:  # pragma: no cover - defensive
            print(f"Error processing column '{column}': {e}")
            blueprint[column] = f"Error: Unable to process this column. Error message: {e}"
    return blueprint
