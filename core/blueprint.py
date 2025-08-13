from typing import Dict, Any
import numpy as np
import polars as pl
from .transforms import lump_rare_levels_pl

def generate_blueprint(dataframe: pl.DataFrame, threshold: float = 0.0025) -> Dict[str, Any]:
    """
    Generate a blueprint dictionary describing how to preprocess each column for modeling.

    Analyzes each column in the input Polars DataFrame:
    - Numeric columns: computes quantile breakpoints (5th to 100th percentile, step 5%) and returns sorted unique break values.
    - Categorical columns: lumps rare levels (below threshold) into 'Other', returns list of levels (excluding 'Other').
    - Columns that cannot be processed get an error message.

    Args:
        dataframe (pl.DataFrame): Input data.
        threshold (float): Minimum proportion for a categorical level to not be lumped (default 0.0025).

    Returns:
        Dict[str, Any]: Mapping of column names to blueprint (breakpoints or levels).

    Example:
        >>> blueprint = generate_blueprint(df)
        >>> print(blueprint['VehAge'])  # Numeric breakpoints
        >>> print(blueprint['VehBrand'])  # Categorical levels
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
        except Exception as e:
            print(f"Error processing column '{column}': {str(e)}")
            blueprint[column] = f"Error: Unable to process this column. Error message: {str(e)}"
    return blueprint
