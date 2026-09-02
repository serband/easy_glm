import numpy as np
import polars as pl


def quote_identifier(identifier: str) -> str:
    if not isinstance(identifier, str):
        raise TypeError("identifier must be a string")
    if not identifier.strip():
        raise ValueError("identifier cannot be empty")
    escaped_identifier = identifier.replace('"', '""')
    return f'"{escaped_identifier}"'


def o_matrix(
    col_name: str, brks, *, null_indicator: bool = False
) -> list[str]:
    """Generate o-matrix SQL expressions for a numeric column.

    Each breakpoint produces one binary column (1 if value < breakpoint,
    else 0).  The result is a set of monotonically decreasing step functions
    suitable as GLM features.

    Null handling:
        Null values are assigned 0 for all breakpoint columns (i.e. they
        are treated as if they were *above* every breakpoint).  When
        ``null_indicator=True``, an additional ``{col}_is_null`` column
        is appended so the model can learn a separate null effect.

    Parameters
    ----------
    col_name : str
        Column name.
    brks : list of float
        Breakpoints (sorted ascending).
    null_indicator : bool
        If True, append a ``{col}_is_null`` binary column.
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
            f"CASE WHEN {quoted_col_name} IS NULL THEN 0 "
            f"ELSE CASE WHEN {quoted_col_name} < {val} THEN 1 ELSE 0 END END "
            f"AS {alias}"
        )
    if null_indicator:
        null_alias = quote_identifier(f"{col_name}_is_null")
        sql_statements.append(
            f"CASE WHEN {quoted_col_name} IS NULL THEN 1 ELSE 0 END "
            f"AS {null_alias}"
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


def one_hot_fun(col_name: str, levels: list) -> list[str]:
    """Generate one-hot SQL expressions for a categorical column.

    Produces one 0/1 column per kept level (except the first, which serves
    as the reference) plus an ``{col}_Other`` catch-all column for unseen
    or NULL values.

    Parameters
    ----------
    col_name : str
        Column name.
    levels : list of str
        Kept levels (post-lumping). Must be non-empty. The first level is
        the reference and is dropped.

    Returns
    -------
    list[str]
        SQL expressions, one per non-reference level + one ``{col}_Other``.
    """
    if not isinstance(col_name, str) or not col_name.strip():
        raise ValueError("col_name must be a non-empty string")
    if isinstance(levels, np.ndarray):
        levels = levels.tolist()
    if not isinstance(levels, list) or not levels:
        raise ValueError("levels must be a non-empty list")

    cleaned = []
    for level in levels:
        cleaned.append(str(level).replace("'", "''"))

    quoted = quote_identifier(col_name)
    sql_statements: list[str] = []

    # Non-reference levels (drop first as reference)
    ref_level = cleaned[0]
    all_levels_str = ", ".join(f"'{lvl}'" for lvl in cleaned)

    for level in cleaned[1:]:
        alias = quote_identifier(f"{col_name}_{level}")
        sql_statements.append(
            f"CASE WHEN CAST({quoted} AS VARCHAR) = '{level}' THEN 1 ELSE 0 END "
            f"AS {alias}"
        )

    # Other catch-all: anything not in known levels, including NULL
    other_alias = quote_identifier(f"{col_name}_Other")
    sql_statements.append(
        f"CASE WHEN CAST({quoted} AS VARCHAR) NOT IN ({all_levels_str}) "
        f"OR {quoted} IS NULL THEN 1 ELSE 0 END AS {other_alias}"
    )

    return sql_statements


def lump_rare_levels_pl(
    column_series: pl.Series,
    total_count: int | None = None,
    threshold: float = 0.001,
    fill_value: str = "Other",
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
        .then(pl.lit(fill_value))
        .otherwise(pl.col(column_series.name))
    )
    return column_series.to_frame().with_columns(expr.alias(column_series.name))[
        column_series.name
    ]
