"""Quickstart: fit a model, inspect relativities, score, check A/E.

Copy-paste into a notebook or run as a script:
    python examples/basic_usage.py
"""

from pathlib import Path

import numpy as np
import polars as pl

import easy_glm

# ---------------------------------------------------------------------------
# 1. Load data & create a train / holdout split
#
# tests/fixtures/french_motor_50k.parquet is a 50,000-row sample of the
# public French motor third-party liability dataset (freMTPL2freq), checked
# into the repo so this runs offline. Use easy_glm.load_external_dataframe()
# for the full ~678k-row dataset (downloads the CASdatasets .rda on first use).
# ---------------------------------------------------------------------------

DATA = Path(__file__).resolve().parents[1] / "tests/fixtures/french_motor_50k.parquet"
df = pl.read_parquet(DATA)

# Add a train-test split column (1 = train, 0 = holdout)
rng = np.random.default_rng(42)
df = df.with_columns(pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64))

print(f"Loaded: {len(df):,} rows × {len(df.columns)} cols")
# → Loaded: 50,000 rows × 13 cols (12 in the fixture + the traintest split)

# ---------------------------------------------------------------------------
# 2. Pick predictors & fit (one call)
# ---------------------------------------------------------------------------

PREDICTORS = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

eglm = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=PREDICTORS,
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    cv=5,  # or alpha=0.001 for a quick fit
)

print(eglm)
# → EasyGLM(model_type='Poisson', target='ClaimNb',
#           predictors=['VehAge', 'Region', 'VehGas', 'DrivAge', 'BonusMalus', 'Density'],
#           alpha=0.00169, base_rate=0.0397)

# ---------------------------------------------------------------------------
# 3. Inspect relativities — one table per predictor
# ---------------------------------------------------------------------------

for name, table in eglm.relativities.items():
    print(f"\n--- {name} ---")
    print(table.head(5))
# → --- VehAge ---
#   ┌──────┬──────┬────────────┬────────┬────────────┬─────────┐
#   │ from ┆ to   ┆ label      ┆ coef   ┆ relativity ┆ is_base │
#   ╞══════╪══════╪════════════╪════════╪════════════╪═════════╡
#   │ null ┆ 1.0  ┆ < 1.0      ┆ 0.21   ┆ 1.23       ┆ false   │
#   │ 1.0  ┆ 2.0  ┆ [1.0, 2.0) ┆ 0.0    ┆ 1.0        ┆ true    │
#   ...
#   The last row ("Other / Unknown") is the null bin.

# Which knots / levels did the lasso keep?
print(eglm.coef_table(drop_zero=True))

# ---------------------------------------------------------------------------
# 4. Optional: quick matplotlib charts
# ---------------------------------------------------------------------------

# easy_glm.plot_all_ratetables(eglm.relativities)

# ---------------------------------------------------------------------------
# 5. Score holdout & check overall A/E
# ---------------------------------------------------------------------------

holdout = df.filter(pl.col("traintest") == 0)
preds = eglm.rate_model.predict(holdout)
ae = holdout["ClaimNb"].sum() / preds.sum()
print(f"\nHoldout A/E: {ae:.4f}")
# → Holdout A/E: ~1.00

# ---------------------------------------------------------------------------
# 6. Per-variable A/E (on holdout) — bins by each variable's own rate-table
#    bands and reports the spread across bins; a well-calibrated factor keeps
#    every bin's A/E close to 1 (summing across all bins would just give the
#    overall A/E back, since the bins partition the same rows).
# ---------------------------------------------------------------------------

from easy_glm.workflow import ae_by_variable  # noqa: E402

actual_total = holdout["ClaimNb"].to_numpy()
weight = holdout["Exposure"].to_numpy()
for var in PREDICTORS:
    table = ae_by_variable(holdout, var, actual_total, preds, weight)
    print(
        f"  {var:15s}  A/E ranges {table['ae'].min():.3f}-{table['ae'].max():.3f}  "
        f"({table.height} bins)"
    )
# → VehAge          A/E ranges 0.66-1.23  (16 bins)
#   Region          A/E ranges 0.00-2.27  (22 bins)
#   ...

# ---------------------------------------------------------------------------
# 7. Save for later scoring / editing
# ---------------------------------------------------------------------------

eglm.save("my_model")
eglm.rate_model.to_json("my_model.easyglm")
print("\nSaved my_model/ and my_model.easyglm")
