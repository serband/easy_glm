from typing import List
import numpy as np
import polars as pl

def o_matrix(col_name: str, brks) -> List[str]:
    if not isinstance(col_name, str):
        raise TypeError("col_name must be a string")
    if not col_name.strip():
        raise ValueError("col_name cannot be empty")
    if isinstance(brks, np.ndarray):
        brks = brks.tolist()
    if not isinstance(brks, list) or len(brks) == 0:
        raise ValueError("brks must be a non-empty list")
    sql_statements = []
    for val in brks:
        sql_statements.append(
            f"CASE WHEN {col_name} IS NULL THEN CASE WHEN AVG({col_name}) OVER () < {val} THEN 1 ELSE 0 END ELSE CASE WHEN {col_name} < {val} THEN 1 ELSE 0 END END AS '{col_name}{val}'"
        )
    return sql_statements

def lump_fun(col_name: str, levels: List, other_category: str = 'Other') -> str:
    if not isinstance(col_name, str) or not col_name.strip():
        raise ValueError("col_name must be non-empty string")
    if isinstance(levels, np.ndarray):
        levels = levels.tolist()
    if not isinstance(levels, list) or not levels:
        raise ValueError("levels must be a non-empty list")
    cleaned = []
    for l in levels:
        if l is None:
            raise ValueError("None level not allowed")
        cleaned.append(str(l).replace("'", "''"))
    unique_levels = list(dict.fromkeys(cleaned))
    levels_str = ", ".join(f"'{lvl}'" for lvl in unique_levels)
    return (
        f"CASE WHEN CAST({col_name} AS VARCHAR) IN ({levels_str}) THEN CAST({col_name} AS VARCHAR) ELSE '{other_category}' END AS {col_name}_lumped"
    )

def lump_rare_levels_pl(column_series: pl.Series, total_count: int = None, threshold: float = 0.001, fill_value: str = 'Unknown') -> pl.Series:
    if total_count is None:
        total_count = column_series.len()
    level_counts = column_series.to_frame().group_by(column_series.name).agg(pl.len().alias('counts'))
    rare_levels = level_counts.filter(pl.col('counts') / total_count < threshold)[column_series.name].to_list()
    expr = pl.when(pl.col(column_series.name).is_in(rare_levels)).then(pl.lit('Other')).otherwise(pl.col(column_series.name))
    return column_series.to_frame().with_columns(expr.alias(column_series.name))[column_series.name]
