"""Pinned, row-preserving catalogue validation and enrichment."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import polars as pl

_UNAMBIGUOUS_NUMERIC_KEY = re.compile(
    r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"
)


def restore_schema(frame: pl.DataFrame, schema: dict[str, str] | None) -> pl.DataFrame:
    """Restore JSON-carried physical column types, especially typed keys."""
    if not schema:
        return frame
    conversions = []
    for name, type_name in schema.items():
        if name not in frame.columns:
            continue
        dtype = getattr(pl, type_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported catalogue dtype {type_name!r} for {name!r}")
        conversions.append(pl.col(name).cast(dtype, strict=True))
    return frame.with_columns(conversions)


def load_catalogue(settings: dict[str, Any]) -> pl.DataFrame:
    """Load an inline master or a hash-pinned CSV/Parquet source."""
    if "catalogue" in settings:
        return restore_schema(
            pl.DataFrame(settings["catalogue"]), settings.get("catalogue_schema")
        )
    path = Path(settings["source_path"])
    expected = settings.get("source_sha256")
    if not expected:
        raise ValueError("A catalogue source path requires source_sha256")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Catalogue source {path} changed since its recipe was saved")
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path, infer_schema_length=0)
    raise ValueError("Catalogue source must be CSV or Parquet")


def validate_catalogue(
    table: pl.DataFrame, key: str, attributes: tuple[str, ...]
) -> None:
    required = (key, *attributes)
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"Catalogue missing required columns {missing}")
    values = table[key]
    if values.null_count() or values.cast(pl.String).str.strip_chars().eq("").any():
        raise ValueError(f"Catalogue key {key!r} contains blank/null values")
    if values.n_unique() != table.height:
        duplicates = (
            table.group_by(key).len().filter(pl.col("len") > 1).head(3)[key].to_list()
        )
        raise ValueError(f"Catalogue key {key!r} has duplicates: {duplicates}")


def align_catalogue_key(
    frame: pl.DataFrame, table: pl.DataFrame, key: str
) -> pl.DataFrame:
    """Safely align a CSV catalogue key to a numeric experience key.

    CSV is read as strings so identifiers such as ``001`` survive upload.  A
    numeric experience key can nevertheless join a CSV whose values are plain
    numeric literals.  Only those unambiguous literals are converted.  Values
    with leading zeroes stay strings and retain the existing type-mismatch
    error, rather than silently turning a meaningful ``001`` into ``1``.
    """
    if key not in frame.columns or key not in table.columns:
        return table
    experience_dtype = frame.schema[key]
    catalogue_dtype = table.schema[key]
    if experience_dtype == catalogue_dtype or not experience_dtype.is_numeric():
        return table
    if catalogue_dtype not in (pl.String, pl.Categorical, pl.Enum):
        return table
    values = table[key].cast(pl.String)
    unambiguous = values.str.contains(_UNAMBIGUOUS_NUMERIC_KEY.pattern).fill_null(False)
    if not bool(unambiguous.all()):
        return table
    try:
        return table.with_columns(pl.col(key).cast(experience_dtype, strict=True))
    except pl.exceptions.InvalidOperationError:
        return table


def join_catalogue(
    frame: pl.DataFrame,
    table: pl.DataFrame,
    key: str,
    attributes: tuple[str, ...],
    *,
    require_coverage: bool = True,
) -> pl.DataFrame:
    """Join selected master fields without changing row count or conflicting with history."""
    if key not in frame.columns:
        raise ValueError(f"Experience missing catalogue key {key!r}")
    table = align_catalogue_key(frame, table, key)
    validate_catalogue(table, key, attributes)
    if frame.schema[key] != table.schema[key]:
        raise ValueError(
            f"Catalogue key {key!r} has a different type in experience and master"
        )
    source = table.select(
        (key, *attributes, *(["multiplier"] if "multiplier" in table.columns else []))
    )
    colliding = [name for name in attributes if name in frame.columns]
    if "multiplier" in frame.columns:
        raise ValueError("Experience already has a multiplier column")
    joined = (
        frame.with_row_index("__catalogue_row_index")
        .join(source, on=key, how="left", validate="m:1", suffix="__master")
        .sort("__catalogue_row_index")
    )
    if joined.height != frame.height:
        raise ValueError("Catalogue join changed the number of experience rows")
    if require_coverage:
        unknown = joined.filter(
            pl.col(key).is_not_null() & pl.col("__catalogue_row_index").is_not_null()
        )
        # A right-side presence marker is needed even when attributes may all be null.
        covered = frame.select(key).join(
            table.select(key).with_columns(pl.lit(True).alias("__covered")),
            on=key,
            how="left",
        )
        if covered["__covered"].null_count():
            examples = (
                covered.filter(pl.col("__covered").is_null())[key].head(3).to_list()
            )
            raise ValueError(
                f"Catalogue {key!r} has no entry for {covered['__covered'].null_count()} experience rows; examples {examples}"
            )
        del unknown
    for name in colliding:
        historical, master = joined[name], joined[f"{name}__master"]
        disagreement = ~(
            (historical == master).fill_null(False)
            | (historical.is_null() & master.is_null())
        )
        if disagreement.any():
            raise ValueError(
                f"Historical {name!r} conflicts with catalogue for key {key!r}"
            )
        joined = joined.drop(f"{name}__master")
    return joined.drop("__catalogue_row_index")
