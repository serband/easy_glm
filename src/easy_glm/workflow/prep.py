"""Data steps of a :class:`~easy_glm.workflow.project.Project`:
load → rename → recode → type override → derive → filter → split.

Every function is pure (frame in, frame out) so the exporter can emit the
same operations as plain polars code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .project import (
    DataConfig,
    DataSource,
    Project,
    Split,
    premium_offset_column,
)

NUMERIC_DTYPES_FOR_SPLIT = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)

_SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "len": len,
    "range": range,
    "list": list,
    "True": True,
    "False": False,
    "None": None,
}


def eval_expr(expr: str) -> pl.Expr:
    """Evaluate a polars expression string in a namespace limited to ``pl``,
    ``np`` and a few builtins. The result must be a ``pl.Expr``."""
    try:
        value = eval(expr, {"__builtins__": _SAFE_BUILTINS}, {"pl": pl, "np": np})
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        raise ValueError(f"Cannot evaluate expression {expr!r}: {exc}") from exc
    if not isinstance(value, pl.Expr):
        raise ValueError(
            f"Expression {expr!r} is not a polars expression (got {type(value).__name__})"
        )
    return value


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_source(
    source: DataSource, *, sample_rows: int | None = None, seed: int = 42
) -> pl.DataFrame:
    """Read the configured file into polars. ``sample_rows`` takes a seeded
    random sample (exploration only; never used for fitting)."""
    path = Path(source.path)
    if not path.exists():
        raise FileNotFoundError(path)
    opts = dict(source.options)
    kind = source.type.lower()
    if kind == "parquet":
        df = pl.read_parquet(path, **opts)
    elif kind == "csv":
        df = pl.read_csv(path, **opts)
    elif kind in ("ipc", "arrow", "feather"):
        df = pl.read_ipc(path, **opts)
    elif kind in ("xlsx", "excel"):
        df = pl.read_excel(path, **opts)
    elif kind == "sas7bdat":
        import pandas as pd

        opts.setdefault("encoding", "latin-1")
        df = pl.from_pandas(pd.read_sas(path, **opts))
    else:
        raise ValueError(f"Unsupported source type {source.type!r}")
    if sample_rows is not None and sample_rows < df.height:
        df = df.sample(n=sample_rows, seed=seed)
    return df


def infer_source_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return {
        "parquet": "parquet",
        "pq": "parquet",
        "csv": "csv",
        "txt": "csv",
        "sas7bdat": "sas7bdat",
        "xlsx": "xlsx",
        "xls": "xlsx",
        "arrow": "ipc",
        "feather": "ipc",
        "ipc": "ipc",
    }.get(suffix, "parquet")


# --------------------------------------------------------------------------
# Variable steps
# --------------------------------------------------------------------------
def recode_expr(column: str, mapping: dict[str, str], default: str | None) -> pl.Expr:
    """``pl.Expr`` that maps levels of ``column`` (compared as strings)."""
    col = pl.col(column).cast(pl.Utf8)
    if not mapping:
        return col.alias(column)
    default_expr = col if default is None else pl.lit(default)
    return col.replace_strict(
        mapping, default=default_expr, return_dtype=pl.Utf8
    ).alias(column)


def premium_offset_expr(premium: str) -> pl.Expr:
    """``log(premium)`` aliased to :func:`premium_offset_column` — the offset of
    a rate-change model, written exactly like this in the exported script."""
    return pl.col(premium).cast(pl.Float64).log().alias(premium_offset_column(premium))


def add_premium_offset(df: pl.DataFrame, data: DataConfig) -> pl.DataFrame:
    """Add ``log(<current premium>)`` when a column has the ``current_premium``
    role, so a model can offset on it and fit the *change* from today's premium.

    Runs **after** the row filters, so a filter such as ``pl.col('Premium') > 0``
    does its job first. A premium that is not a positive, finite number has no
    logarithm, so the whole frame is refused with a message naming how many rows
    are wrong — a silent null there would become a NaN offset and a fit that
    fails deep inside the solver.
    """
    premium = next((c for c, r in data.roles.items() if r == "current_premium"), None)
    if premium is None or premium not in df.columns:
        return df
    value = df[premium].cast(pl.Float64, strict=False)
    bad = int((~(value.is_finite() & (value > 0))).fill_null(True).sum())
    if bad:
        raise ValueError(
            f"Current premium column {premium!r} has {bad:,} row(s) that are not a "
            "positive number (zero, negative, missing or infinite), so "
            f"log({premium}) — the rate-change offset — cannot be computed. Add a "
            f"row filter such as pl.col({premium!r}) > 0 on the Variables page, or "
            "give the column another role."
        )
    return df.with_columns(premium_offset_expr(premium))


def apply_variables(df: pl.DataFrame, data: DataConfig) -> pl.DataFrame:
    """Renames, recodes, type overrides, derived columns, filters and — when a
    column has the ``current_premium`` role — the derived ``log(premium)``
    offset column, in that order."""
    renames = {k: v for k, v in data.renames.items() if k in df.columns and k != v}
    if renames:
        df = df.rename(renames)

    recodes = [
        recode_expr(col, rc.mapping, rc.default)
        for col, rc in data.recodes.items()
        if col in df.columns
    ]
    if recodes:
        df = df.with_columns(recodes)

    casts = []
    for col, kind in data.types.items():
        if col not in df.columns:
            continue
        if kind == "categorical":
            casts.append(pl.col(col).cast(pl.Utf8))
        elif kind == "numeric":
            casts.append(pl.col(col).cast(pl.Float64, strict=False))
        else:
            raise ValueError(f"types[{col!r}] must be 'categorical' or 'numeric'")
    if casts:
        df = df.with_columns(casts)

    for d in data.derived:  # sequential: later expressions may use earlier ones
        df = df.with_columns(eval_expr(d.expr).alias(d.name))

    for f in data.filters:
        df = df.filter(eval_expr(f))

    return add_premium_offset(df, data)


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------
def add_split_column(df: pl.DataFrame, split: Split) -> pl.DataFrame:
    """Ensure the split column exists with 1 = train, 0 = holdout."""
    if split.mode == "column":
        if split.column not in df.columns:
            raise KeyError(f"Split column {split.column!r} not found")
        dtype = df.schema[split.column]
        if dtype in NUMERIC_DTYPES_FOR_SPLIT:
            try:
                value = float(split.train_value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Split column {split.column!r} is numeric but the value meaning "
                    f"TRAIN is {split.train_value!r}; enter a number"
                ) from None
            flag = (pl.col(split.column).cast(pl.Float64) == pl.lit(value)).cast(
                pl.Int64
            )
        else:
            # text / categorical / boolean indicators: compare as text
            flag = (
                pl.col(split.column).cast(pl.Utf8) == pl.lit(str(split.train_value))
            ).cast(pl.Int64)
        out = df.with_columns(flag.alias(split.column))
        if out[split.column].sum() == 0:
            raise ValueError(
                f"No row of {split.column!r} equals the TRAIN value "
                f"{split.train_value!r}; check the value on the Split page"
            )
        return out
    if split.mode == "random":
        name = str(split.column).strip()
        if not name:
            raise ValueError("The random split column needs a name")
        if name in df.columns:
            raise ValueError(
                f"The random split column {name!r} would overwrite an existing data "
                "column; choose another name on the Split page"
            )
        is_train = np.random.default_rng(split.seed).random(df.height) < split.fraction
        return df.with_columns(pl.Series(name, is_train.astype(np.int64)))
    raise ValueError(f"Unknown split mode {split.mode!r}")


def train_holdout(df: pl.DataFrame, split: Split) -> tuple[pl.DataFrame, pl.DataFrame]:
    if split.column not in df.columns:
        df = add_split_column(df, split)
    return df.filter(pl.col(split.column) == 1), df.filter(pl.col(split.column) == 0)


# --------------------------------------------------------------------------
# One-shot
# --------------------------------------------------------------------------
def prepare(project: Project, df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Load (if ``df`` is None) and apply every data step, returning the full
    prepared frame with the split column (1 = train, 0 = holdout)."""
    if df is None:
        df = load_source(project.data.source)
    df = apply_variables(df, project.data)
    return add_split_column(df, project.data.split)


def column_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Schema overview used by the Variables page."""
    rows: list[dict[str, Any]] = []
    n = max(df.height, 1)
    for name, dtype in df.schema.items():
        s = df[name]
        rows.append(
            {
                "column": name,
                "dtype": str(dtype),
                "null_share": s.null_count() / n,
                "n_unique": s.n_unique(),
                "example": (
                    str(s.drop_nulls().head(1).to_list()[0])
                    if s.null_count() < df.height
                    else ""
                ),
            }
        )
    return pl.DataFrame(rows)
