import numpy as np
import polars as pl


def quote_identifier(identifier: str) -> str:
    if not isinstance(identifier, str):
        raise TypeError("identifier must be a string")
    if not identifier.strip():
        raise ValueError("identifier cannot be empty")
    escaped_identifier = identifier.replace('"', '""')
    return f'"{escaped_identifier}"'


def o_matrix(col_name: str, brks) -> list[str]:
    """
    Note:
        Null values in the column are imputed with the column's average value.
    """
    if not isinstance(col_name, str):
        raise TypeError("col_name must be a string")
    if not col_name.strip():
        raise ValueError("col_name cannot be empty")
    if isinstance(brks, np.ndarray):
        brks = brks.tolist()
    if not isinstance(brks, list) or len(brks) == 0:
        raise ValueError("brks must be a non-empty list")
    sql_statements = []
    quoted_col_name = quote_identifier(col_name)
    for val in brks:
        alias = quote_identifier(f"{col_name}{val}")
        sql_statements.append(
            f"CASE WHEN {quoted_col_name} IS NULL THEN "
            f"CASE WHEN AVG({quoted_col_name}) OVER () < {val} THEN 1 ELSE 0 END "
            f"ELSE CASE WHEN {quoted_col_name} < {val} THEN 1 ELSE 0 END END "
            f"AS {alias}"
        )
    return sql_statements


def lump_fun(col_name: str, levels: list, other_category: str = "Other") -> str:
    """
    Note:
        Null values in the column will be lumped into the `other_category`.
    """
    if not isinstance(col_name, str) or not col_name.strip():
        raise ValueError("col_name must be non-empty string")
    if isinstance(levels, np.ndarray):
        levels = levels.tolist()
    if not isinstance(levels, list) or not levels:
        raise ValueError("levels must be a non-empty list")
    cleaned = []
    for level in levels:
        if level is None:
            raise ValueError("None level not allowed")
        cleaned.append(str(level).replace("'", "''"))
    unique_levels = list(dict.fromkeys(cleaned))
    levels_str = ", ".join(f"'{lvl}'" for lvl in unique_levels)
    quoted_col_name = quote_identifier(col_name)
    alias = quote_identifier(f"{col_name}_lumped")
    escaped_other_category = other_category.replace("'", "''")
    return (
        f"CASE WHEN CAST({quoted_col_name} AS VARCHAR) IN ({levels_str}) "
        f"THEN CAST({quoted_col_name} AS VARCHAR) "
        f"ELSE '{escaped_other_category}' END AS {alias}"
    )


def lump_rare_levels_pl(
    column_series: pl.Series,
    total_count: int | None = None,
    threshold: float = 0.001,
    fill_value: str = "Unknown",
) -> pl.Series:
    if total_count is None:
        total_count = column_series.len()
    level_counts = (
        column_series.to_frame()
        .group_by(column_series.name)
        .agg(pl.len().alias("counts"))
    )
    rare_levels = level_counts.filter(pl.col("counts") / total_count < threshold)[
        column_series.name
    ].to_list()
    expr = (
        pl.when(pl.col(column_series.name).is_in(rare_levels))
        .then(pl.lit("Other"))
        .otherwise(pl.col(column_series.name))
    )
    return column_series.to_frame().with_columns(expr.alias(column_series.name))[
        column_series.name
    ]
