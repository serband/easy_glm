"""Step-by-step pipeline: blueprint → prepare → fit → rate tables → RateModel.

Use this when you need control between stages (custom blueprint, reuse
prepared data for several fits, inspect intermediate tables).  Most users
should start with ``basic_usage.py`` (``EasyGLM.fit``).

Run as a script:
    python examples/advanced_pipeline.py
"""

import numpy as np
import polars as pl

import easy_glm
from easy_glm.engine import RateModel

# ---------------------------------------------------------------------------
# 0. Load data & split
# ---------------------------------------------------------------------------

df = easy_glm.load_external_dataframe()
rng = np.random.default_rng(42)
df = df.with_columns(
    pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64)
)

PREDICTORS = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]
BASE_RATE = 0.1

# ---------------------------------------------------------------------------
# 1. Blueprint — quantile breaks (numeric) & lump rare levels (categorical)
#    Generate on TRAIN rows only to avoid leakage.
# ---------------------------------------------------------------------------

train_df = df.filter(pl.col("traintest") == 1)
blueprint = easy_glm.generate_blueprint(train_df)

print("Blueprint for VehAge (first 5 breakpoints):", blueprint["VehAge"][:5])
print("Blueprint for Region (retained levels):", blueprint["Region"])
# → Blueprint for VehAge: [0.0, 1.0, 2.0, 3.0, 4.0]
#   Blueprint for Region: ['Aquitaine', 'Basse-Normandie', ...]

# ---------------------------------------------------------------------------
# 2. Prepare — DuckDB transforms blueprint → model-ready feature columns
# ---------------------------------------------------------------------------

prepped = easy_glm.prepare_data(
    df=df,
    modelling_variables=PREDICTORS,
    additional_columns=["Exposure", "ClaimNb", "traintest"],
    formats=blueprint,
    traintest_column="traintest",
    table_name="cars",
)
print(f"\nPrepared: {len(prepped.columns)} columns × {len(prepped):,} rows")
# → Prepared: ~130 columns × 678,013 rows

# ---------------------------------------------------------------------------
# 3. Fit LASSO GLM on prepared data
# ---------------------------------------------------------------------------

model = easy_glm.fit_lasso_glm(
    dataframe=prepped,
    target="ClaimNb",
    model_type="Poisson",
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    use_cv=True,
)

print(f"Intercept: {model.intercept_:.4f}")
print(f"Non-zero coefficients: {(model.coef_ != 0).sum()} / {len(model.coef_)}")
if hasattr(model, "alpha_"):
    print(f"Best alpha: {model.alpha_:.6f}")
# → Intercept: -3.1234
#   Non-zero coefficients: 45 / 120
#   Best alpha: 0.001234

# ---------------------------------------------------------------------------
# 4. Single rate table (one variable, manual control)
# ---------------------------------------------------------------------------

vehage_tbl = easy_glm.ratetable(
    model=model,
    dataset=df,
    col_name="VehAge",
    levels=blueprint["VehAge"],
    prepare=lambda d: easy_glm.prepare_data(
        df=d,
        modelling_variables=PREDICTORS,
        formats=blueprint,
        table_name="line_prepped",
    ),
    random_seed=42,
)
print(f"\nSingle rate table — VehAge ({len(vehage_tbl)} bins):")
print(vehage_tbl.head())

# ---------------------------------------------------------------------------
# 5. All rate tables
# ---------------------------------------------------------------------------

all_tables = easy_glm.generate_all_ratetables(
    model=model,
    dataset=df,
    predictor_variables=PREDICTORS,
    blueprint=blueprint,
    random_seed=42,
)
print(f"\nGenerated {len(all_tables)} rate tables: {list(all_tables.keys())}")

# ---------------------------------------------------------------------------
# 6. Build RateModel (portable scorer)
# ---------------------------------------------------------------------------

rm = RateModel.from_rate_tables(
    all_tables=all_tables,
    blueprint=blueprint,
    base_rate=BASE_RATE,
    model_type="poisson",
    target="ClaimNb",
    weight_col="Exposure",
    exposure_col="Exposure",
    train_test_col="traintest",
)

# ---------------------------------------------------------------------------
# 7. Score & check A/E — using library features, not manual aggregation
# ---------------------------------------------------------------------------

holdout = df.filter(pl.col("traintest") == 0)
holdout_preds = rm.predict(holdout)
print(f"\nOverall holdout A/E: {holdout['ClaimNb'].sum() / holdout_preds.sum():.4f}")
# → Overall holdout A/E: ~1.00

print("\nPer-variable A/E on holdout (using compute_ae_for_variable):")
for var in PREDICTORS:
    ae_result = rm.compute_ae_for_variable(holdout, var)
    # Get bucket-level results from the 'all' subset (or 'train' + 'test')
    subsets = ae_result["subsets"]
    for split_name in ["train", "test"]:
        if split_name not in subsets:
            continue
        buckets = subsets[split_name]
        actual = sum(b["actual"] for b in buckets)
        expected = sum(b["expected"] for b in buckets)
        ae = actual / expected if expected > 0 else float("nan")
        print(f"  {var:15s}  {split_name:5s}  A/E = {ae:.4f}  ({len(buckets)} bins)")
# → VehAge          train  A/E = 0.98  (20 bins)
#   VehAge          test   A/E = 1.01  (20 bins)
#   Region          train  A/E = 0.99  (11 bins)
#   ...

# ---------------------------------------------------------------------------
# 8. Optional: matplotlib A/E grid via plot_all_ratetables
# ---------------------------------------------------------------------------

# easy_glm.plot_all_ratetables(all_tables, blueprint)

# ---------------------------------------------------------------------------
# 9. Save & reload
# ---------------------------------------------------------------------------

rm.to_json("french_motor.easyglm")
print("\nExported → french_motor.easyglm")

loaded = RateModel.from_json("french_motor.easyglm")
sample = df.select(PREDICTORS).head(3)
print(f"Reloaded predictions: {loaded.predict(sample)}")
