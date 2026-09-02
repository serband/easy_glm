"""Regenerate ``tests/fixtures/french_motor_50k.parquet`` (the golden-test fixture).

Recipe (exact; the golden numbers depend on it):

    df = load_external_dataframe()            # cached freMTPL2freq, 677,991 rows
    df = df.select(COLUMNS).sort("IDpol")
    df = df.sample(n=50_000, seed=20260902).sort("IDpol")
    IDpol and the categorical columns cast to text

Run from the repository root::

    .venv/bin/python tests/fixtures/make_french_motor_50k.py [--check]

``--check`` compares the regenerated frame with the checked-in file instead of
writing (used by ``tests/test_golden.py`` when the cache is present).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

COLUMNS = [
    "IDpol",
    "ClaimNb",
    "Exposure",
    "DrivAge",
    "VehAge",
    "BonusMalus",
    "Density",
    "VehPower",
    "Region",
    "VehBrand",
    "VehGas",
    "Area",
]
N_ROWS = 50_000
SEED = 20260902
FIXTURE = Path(__file__).with_name("french_motor_50k.parquet")


def regenerate(source: pl.DataFrame) -> pl.DataFrame:
    sub = source.select(COLUMNS).sort("IDpol").sample(n=N_ROWS, seed=SEED).sort("IDpol")
    return sub.with_columns(
        pl.col("IDpol").cast(pl.Utf8),
        pl.col("Region", "VehBrand", "VehGas", "Area").cast(pl.Utf8),
    )


def main() -> None:
    from easy_glm import load_external_dataframe

    frame = regenerate(load_external_dataframe())
    if "--check" in sys.argv:
        current = pl.read_parquet(FIXTURE)
        ok = frame.equals(current)
        print("fixture matches recipe:", ok)
        sys.exit(0 if ok else 1)
    FIXTURE.parent.mkdir(exist_ok=True)
    frame.write_parquet(FIXTURE, compression="zstd", compression_level=19)
    print(f"wrote {FIXTURE} ({frame.height} rows, {FIXTURE.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
