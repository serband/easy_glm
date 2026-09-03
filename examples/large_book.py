"""Fit a large book to show the compact design-matrix path.

Generates a synthetic motor book of ``--rows`` rows (default 300,000 — above
``easy_glm.core.design.SPARSE_ROW_THRESHOLD`` (200,000), where ``DesignSpec``
switches on its own from one float64 column per band to one int32 bin index
per row) and fits a Poisson frequency model on it, timing the fit and
reporting the design's memory footprint against the formula
``DesignSpec.expected_design_bytes`` predicts.

Run as a script:
    python examples/large_book.py                # 300,000 rows, compact path
    python examples/large_book.py --rows 100000   # below the threshold, dense
    python examples/large_book.py --rows 300000 --sparse false   # force dense
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl

from easy_glm import DesignSpec, fit_glm


def synthetic_book(n: int, seed: int = 0) -> pl.DataFrame:
    """A motor book with the shape a real one has: skewed numerics, a
    handful of categorical levels, no interactions — big enough to show the
    compact path, small enough to fit in well under a minute."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n).astype(float)
    bonus = rng.integers(50, 230, n).astype(float)
    density = rng.lognormal(5.0, 1.5, n)
    region = rng.choice([f"Region{i:02d}" for i in range(15)], n)
    exposure = rng.uniform(0.1, 1.0, n)
    mu = np.exp(
        -3.0 + 0.012 * (60 - age) + 0.006 * (bonus - 100) + 0.05 * np.log1p(density)
    )
    claims = rng.poisson(mu * exposure)
    return pl.DataFrame(
        {
            "DrivAge": age,
            "BonusMalus": bonus,
            "Density": density,
            "Region": region,
            "Exposure": exposure,
            "ClaimNb": claims.astype(float),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=300_000)
    parser.add_argument(
        "--sparse",
        type=str,
        default=None,
        help="'true' / 'false' to force the compact or dense design; default "
        "chooses by row count",
    )
    args = parser.parse_args()
    sparse = {"true": True, "false": False, None: None}[
        args.sparse.lower() if args.sparse else None
    ]

    df = synthetic_book(args.rows)
    predictors = ["DrivAge", "BonusMalus", "Density", "Region"]
    spec = DesignSpec.from_data(df, predictors, weight_col="Exposure")
    used_sparse = args.rows >= 200_000 if sparse is None else sparse
    print(
        f"{args.rows:,} rows x {spec.n_features} design columns "
        f"({'compact' if used_sparse else 'dense'} path)"
    )
    print(f"expected design bytes (compact): {spec.expected_design_bytes(args.rows):,}")

    t0 = time.perf_counter()
    fit = fit_glm(
        df,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=0.001,
        sparse=sparse,
    )
    dt = time.perf_counter() - t0
    print(
        f"fitted in {dt:.1f}s, {int((fit.coef != 0).sum())} of {len(fit.coef)} terms kept"
    )

    # Scoring never builds a design matrix, however the book was fitted.
    preds = fit.predict(df.head(1000))
    print(
        f"scored 1,000 rows without a design matrix, mean prediction {preds.mean():.4g}"
    )


if __name__ == "__main__":
    main()
