"""Quickstart: fit a model, inspect relativities, score, check A/E.

Copy-paste into a notebook or run as a script:
    python examples/basic_usage.py
"""

import numpy as np
import polars as pl

import easy_glm

# ---------------------------------------------------------------------------
# 1. Load data & create a train / holdout split
# ---------------------------------------------------------------------------

df = easy_glm.load_external_dataframe()

# Add a train-test split column (1 = train, 0 = holdout)
rng = np.random.default_rng(42)
df = df.with_columns(pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64))

print(f"Loaded: {len(df):,} rows × {len(df.columns)} cols")
# → Loaded: 678,013 rows × 12 cols

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
#           alpha=0.001234, base_rate=0.0712)

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
# 6. Per-variable A/E (on holdout)
# ---------------------------------------------------------------------------

for var in PREDICTORS:
    result = eglm.rate_model.compute_ae_for_variable(holdout, var)
    for subset_name, buckets in result["subsets"].items():
        if not buckets:
            continue
        actual = sum(b["actual"] for b in buckets)
        expected = sum(b["expected"] for b in buckets)
        if expected == 0:
            continue
        ae = actual / expected
        print(f"  {var:15s}  {subset_name:5s}  A/E = {ae:.4f}  ({len(buckets)} bins)")
# → VehAge          A/E = 0.98  (20 bins)
#   Region          A/E = 1.01  (11 bins)
#   ...

# ---------------------------------------------------------------------------
# 7. Save for later scoring / editing
# ---------------------------------------------------------------------------

eglm.save("my_model")
eglm.rate_model.to_json("my_model.easyglm")
print("\nSaved my_model/ and my_model.easyglm")
