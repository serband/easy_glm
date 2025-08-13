
import duckdb
import polars as pl

from .transforms import lump_fun, o_matrix


def prepare_data(
    modelling_variables: list[str],
    additional_columns: list[str] | None = None,
    traintest_column: str | None = None,
    df: pl.DataFrame | None = None,
    table_name: str = "dataset",
    formats: dict | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pl.DataFrame:
    """Apply blueprint-driven SQL style transformations via DuckDB."""
    if formats is None:
        formats = {}
    if con is None:
        if df is not None:
            con = duckdb.connect(":memory:")
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        else:
            con = duckdb.connect()
    else:
        if not isinstance(con, duckdb.DuckDBPyConnection):  # pragma: no cover
            print("Warning: The provided connection is not a duckdb connection. Proceeding anyway.")
    tables = con.execute("SHOW TABLES").fetchall()
    if table_name not in [t[0] for t in tables]:
        raise ValueError(
            f"The specified table '{table_name}' does not exist in the database. "
            f"Available tables are: {', '.join([t[0] for t in tables])}"
        )
    expressions: list[str] = []
    if additional_columns is None:
        additional_columns = []
    if traintest_column and traintest_column not in additional_columns:
        additional_columns.append(traintest_column)
    for var in modelling_variables:
        if var not in con.execute(f"PRAGMA table_info({table_name})").df()["name"].tolist():
            print(f"Warning: Column '{var}' not found in the table. Skipping.")
            continue
        if var in formats:
            dict_values = formats[var]
            if all(isinstance(x, int | float) for x in dict_values):
                expressions.extend(o_matrix(var, dict_values))
            else:
                expressions.append(lump_fun(var, dict_values))
        else:
            expressions.append(f"'{var}'")
    for col in additional_columns:
        if col in con.execute(f"PRAGMA table_info({table_name})").df()["name"].tolist():
            expressions.append(col)
        else:
            print(f"Warning: Additional column '{col}' not found in the table. Skipping.")
    query = f"SELECT {', '.join(expressions)} FROM {table_name}"
    result_df = con.execute(query).df()
    if df is not None and con is not None:
        con.close()
    return pl.DataFrame(result_df)
