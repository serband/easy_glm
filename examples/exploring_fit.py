"""Review a saved model: factor A/E, a table change and snapshots.

This is deliberately a post-fit example. First create ``my_model.easyglm``
with ``python examples/basic_usage.py``, then run:

    python examples/exploring_fit.py my_model.easyglm

The script uses the checked-in French motor sample only to calculate review
statistics. It does not fit a replacement model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from easy_glm.engine import RateModel
from easy_glm.workflow import ae_by_variable

DATA = Path(__file__).resolve().parents[1] / "tests/fixtures/french_motor_50k.parquet"


def review(model_path: Path) -> None:
    """Load ``model_path`` and print an actuarial review of its holdout fit."""
    rate_model = RateModel.from_json(model_path)
    df = pl.read_parquet(DATA)
    rng = np.random.default_rng(42)
    df = df.with_columns(
        pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64)
    )
    holdout = df.filter(pl.col("traintest") == 0)
    actual = holdout["ClaimNb"].to_numpy()
    expected_before = rate_model.predict(holdout)
    exposure = holdout["Exposure"].to_numpy()

    print(f"Reviewing {model_path}: {len(rate_model.variables)} rating variables")
    print(f"Overall holdout A/E: {actual.sum() / expected_before.sum():.3f}\n")
    print("A/E range by factor:")
    for variable in rate_model.variables:
        table = ae_by_variable(holdout, variable, actual, expected_before, exposure)
        print(
            f"  {variable:15s} {table['ae'].min():.2f}–{table['ae'].max():.2f} "
            f"across {table.height} bands or levels"
        )

    # Capture the imported model before any commercial change.
    baseline_version = rate_model.create_snapshot("Imported model")
    variable = "DrivAge"
    row = rate_model.variables[variable].table[
        len(rate_model.variables[variable].table) // 2
    ]
    before = row.relativity
    rate_model.update_relativity(variable, row.from_, row.to_, before * 1.05)
    expected_after = rate_model.predict(holdout)
    expected_change = expected_after.sum() / expected_before.sum() - 1
    rate_model.base_rate *= expected_before.sum() / expected_after.sum()
    expected_rebalanced = rate_model.predict(holdout)
    changed_version = rate_model.create_snapshot(
        "Illustrative 5% DrivAge change, rebalanced"
    )
    if row.from_ is None:
        band = f"< {row.to_}"
    elif row.to_ is None:
        band = f"≥ {row.from_}"
    else:
        band = f"[{row.from_}, {row.to_})"

    print(
        f"\nIllustrative change: {variable} {band} {before:.3f} → "
        f"{before * 1.05:.3f}"
    )
    print(
        f"Expected claims changed by {expected_change:+.2%}. "
        "The base rate was then rebalanced; total expected claims now agree: "
        f"{np.isclose(expected_rebalanced.sum(), expected_before.sum(), rtol=1e-9)}."
    )
    print(
        f"Snapshots available: {baseline_version} (before), {changed_version} (after)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=Path("my_model.easyglm"),
        help="saved .easyglm model (default: my_model.easyglm)",
    )
    args = parser.parse_args()
    if not args.model.exists():
        print(
            f"No saved model at {args.model}. Run 'python examples/basic_usage.py' "
            "first, then pass the .easyglm file to this review script."
        )
        return
    review(args.model)


if __name__ == "__main__":
    main()
