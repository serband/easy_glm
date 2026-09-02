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
df = df.with_columns(
    pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64)
)

print(f"Loaded: {len(df):,} rows × {len(df.columns)} cols")
# → Loaded: 678,013 rows × 12 cols

# ---------------------------------------------------------------------------
# 2. Pick predictors & fit (one call)
# ---------------------------------------------------------------------------

PREDICTORS = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]
BASE_RATE = 0.05

eglm = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=PREDICTORS,
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    use_cv=True,
    base_rate=BASE_RATE,
)

print(eglm)
# → EasyGLM(model_type='Poisson', target='ClaimNb',
#           predictors=['VehAge', 'Region', 'VehGas', 'DrivAge', 'BonusMalus', 'Density'],
#           base_rate=0.05)

# ---------------------------------------------------------------------------
# 3. Inspect relativities — one table per predictor
# ---------------------------------------------------------------------------

for name, table in eglm.relativities.items():
    print(f"\n--- {name} ---")
    print(table.head(5))
# → --- VehAge ---
#   shape: (5, 3)
#   ┌────────┬────────────┬────────────┐
#   │ VehAge ┆ relativity ┆ prediction │
#   │ ---    ┆ ---        ┆ ---        │
#   │ i64    ┆ f64        ┆ f64        │
#   ╞════════╪════════════╪════════════╡
#   │ 0      ┆ 0.78       ┆ 0.041      │
#   │ 1      ┆ 0.85       ┆ 0.045      │
#   │ 2      ┆ 0.92       ┆ 0.049      │
#   │ 3      ┆ 0.98       ┆ 0.052      │
#   │ 4      ┆ 1.05       ┆ 0.056      │
#   └────────┴────────────┴────────────┘

# ---------------------------------------------------------------------------
# 4. Optional: quick matplotlib charts
# ---------------------------------------------------------------------------

# easy_glm.plot_all_ratetables(eglm.relativities, eglm.blueprint)

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
