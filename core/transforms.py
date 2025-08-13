from typing import List
import numpy as np
import polars as pl

def o_matrix(col_name: str, brks) -> List[str]:
    """
    Generate SQL CASE statements for one-hot style ordinal splits at given breakpoints.

    Args:
        col_name (str): Name of the column to split.
        brks (list or np.ndarray): Numeric breakpoints for splitting.

    Returns:
        List[str]: SQL CASE statements for each breakpoint.

    Raises:
        TypeError: If col_name is not a string or brks is not a list/array of numbers.
        ValueError: If col_name is empty, brks is empty, or contains non-numeric values.

    Example:
        >>> o_matrix('age', [20, 30])
        ["CASE WHEN age IS NULL THEN CASE WHEN AVG(age) OVER () < 20 THEN 1 ELSE 0 END ELSE CASE WHEN age < 20 THEN 1 ELSE 0 END END AS 'age20'", ...]
    """
    if not isinstance(col_name, str):
        raise TypeError(f"col_name must be a string, got {type(col_name)}")
    if not col_name.strip():
        raise ValueError("col_name cannot be empty or whitespace only")
    if isinstance(brks, np.ndarray):
        brks = brks.tolist()
    if not isinstance(brks, list):
        raise TypeError(f"brks must be a list or numpy array, got {type(brks)}")
    if len(brks) == 0:
        raise ValueError("brks cannot be empty")
    for i, val in enumerate(brks):
        if not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"All values in brks must be numeric. Value at index {i} is {type(val)}: {val}")
        if np.isnan(val) or np.isinf(val):
            raise ValueError(f"Break values cannot be NaN or infinite. Value at index {i}: {val}")
    clean_col_name = col_name.strip()
    sql_statements = []
    for val in brks:
        sql_statement = (
            f"CASE WHEN {clean_col_name} IS NULL "
            f"THEN CASE WHEN AVG({clean_col_name}) OVER () < {val} THEN 1 ELSE 0 END "
            f"ELSE CASE WHEN {clean_col_name} < {val} THEN 1 ELSE 0 END END "
            f"AS '{clean_col_name}{val}'"
        )
        sql_statements.append(sql_statement)
    return sql_statements

def lump_fun(col_name: str, levels: List, other_category: str = 'Other') -> str:
    """
    Create a SQL CASE statement to group categorical levels, lumping rare/unseen levels into 'Other'.

    Args:
        col_name (str): Name of the categorical column.
        levels (List): Levels to keep as-is; others are lumped.
        other_category (str): Name for the catch-all category (default 'Other').

    Returns:
        str: SQL CASE statement for lumping levels.

    Raises:
        TypeError: If col_name/other_category is not a string or levels is not a list/array.
        ValueError: If col_name/other_category is empty, or levels is empty.

    Example:
        >>> lump_fun('brand', ['Toyota', 'Honda'])
        "CASE WHEN CAST(brand AS VARCHAR) IN ('Toyota', 'Honda') THEN CAST(brand AS VARCHAR) ELSE 'Other' END AS brand_lumped"
    """
    if not isinstance(col_name, str):
        raise TypeError(f"col_name must be a string, got {type(col_name)}")
    if not col_name.strip():
        raise ValueError("col_name cannot be empty or whitespace only")
    if isinstance(levels, np.ndarray):
        levels = levels.tolist()
    if not isinstance(levels, list):
        raise TypeError(f"levels must be a list or numpy array, got {type(levels)}")
    if len(levels) == 0:
        raise ValueError("levels cannot be empty - must specify at least one level to keep")
    if not isinstance(other_category, str):
        raise TypeError(f"other_category must be a string, got {type(other_category)}")
    if not other_category.strip():
        raise ValueError("other_category cannot be empty or whitespace only")
    clean_col_name = col_name.strip()
    clean_other_category = other_category.strip()
    clean_levels = []
    for i, level in enumerate(levels):
        if level is None:
            raise ValueError(f"Level at index {i} cannot be None/null")
        level_str = str(level).replace("'", "''")
        clean_levels.append(level_str)
    seen = set()
    unique_levels = []
    for level in clean_levels:
        if level not in seen:
            seen.add(level)
            unique_levels.append(level)
    levels_str = ", ".join(f"'{level}'" for level in unique_levels)
    sql_statement = (
        f"CASE WHEN CAST({clean_col_name} AS VARCHAR) IN ({levels_str}) "
        f"THEN CAST({clean_col_name} AS VARCHAR) "
        f"ELSE '{clean_other_category}' END AS {clean_col_name}_lumped"
    )
    return sql_statement

def lump_rare_levels_pl(column_series: pl.Series, total_count: int = None, threshold: float = 0.001, fill_value: str = 'Unknown') -> pl.Series:
    """
    Replace rare levels in a Polars categorical/Utf8 series with 'Other'.

    Args:
        column_series (pl.Series): Input categorical column.
        total_count (int, optional): Total number of rows (default: inferred).
        threshold (float): Minimum proportion for a level to not be lumped (default: 0.001).
        fill_value (str): Value to replace nulls (default: 'Unknown').

    Returns:
        pl.Series: Series with rare levels replaced by 'Other'.

    Example:
        >>> lump_rare_levels_pl(df['region'], threshold=0.01)
        # Returns a series with rare regions replaced by 'Other'
    """
    if total_count is None:
        total_count = column_series.len()
    level_counts = column_series.to_frame().group_by(column_series.name).agg(pl.len().alias('counts'))
    rare_levels = level_counts.filter(pl.col('counts') / total_count < threshold)[column_series.name].to_list()
    lump_expression = pl.when(pl.col(column_series.name).is_in(rare_levels)) \
                        .then(pl.lit('Other')) \
                        .otherwise(pl.col(column_series.name))
    return column_series.to_frame().with_columns(lump_expression.alias(column_series.name))[column_series.name]
