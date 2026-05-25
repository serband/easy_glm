"""Test script: build a model end-to-end, save it, and launch the editor."""

import numpy as np
import polars as pl

from easy_glm import (
    fit_lasso_glm,
    generate_all_ratetables,
    generate_blueprint,
    prepare_data,
)
from easy_glm.engine import RateModel

# 1. Create synthetic insurance data
rng = np.random.default_rng(42)
n = 1000

df = pl.DataFrame(
    {
        "VehAge": rng.integers(0, 16, size=n),
        "DrivAge": rng.integers(18, 80, size=n),
        "Region": rng.choice(["North", "South", "Urban"], size=n, p=[0.35, 0.35, 0.30]),
        "Exposure": rng.uniform(0.25, 1.0, size=n),
    }
)

region_effect = np.select(
    [df["Region"] == "South", df["Region"] == "Urban"],
    [0.25, 0.65],
    default=0.0,
)
log_freq = (
    -2.4
    + (df["VehAge"].to_numpy() * 0.055)
    + ((80 - df["DrivAge"].to_numpy()) * 0.01)
    + region_effect
)
df = df.with_columns(
    ClaimNb=pl.Series(rng.poisson(df["Exposure"].to_numpy() * np.exp(log_freq))),
    traintest=pl.Series((rng.random(n) < 0.75).astype(np.int8)),
)

predictors = ["VehAge", "DrivAge", "Region"]
print("1. Generating blueprint...")
bp = generate_blueprint(df)
print(f"   Blueprint: {['DRIVAGE_BP', bp['DrivAge'], 'REGION_LEVELS', bp['Region']]}")
print(f"   VehAge breakpoints: {bp['VehAge']}")
print(f"   Region levels: {bp['Region']}")

print("2. Preparing data...")
prepped = prepare_data(
    df=df,
    modelling_variables=predictors,
    additional_columns=["Exposure", "ClaimNb", "traintest"],
    formats=bp,
    traintest_column="traintest",
    table_name="cars",
)
print(f"   Prepared shape: {prepped.shape}")

print("3. Fitting LASSO GLM (Poisson)...")
model = fit_lasso_glm(
    dataframe=prepped,
    target="ClaimNb",
    model_type="Poisson",
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
)
print(f"   Coefficients: {len(model.coef_)} features, intercept={model.intercept_:.4f}")

print("4. Generating rate tables...")
all_tables = generate_all_ratetables(
    model=model,
    dataset=df,
    predictor_variables=predictors,
    blueprint=bp,
    random_seed=42,
)
for var, tbl in all_tables.items():
    print(f"   {var}: {tbl.height} levels")

print("5. Creating RateModel...")
rm = RateModel.from_rate_tables(
    all_tables,
    bp,
    base_rate=0.05,
    model_type="poisson",
    target="ClaimNb",
    weight_col="Exposure",
    train_test_col="traintest",
)
print(f"   Variables: {list(rm.variables.keys())}")
print(f"   Metadata: model_type={rm.metadata.model_type}, target={rm.metadata.target}")
print(f"   Snapshots: {rm.current_version}")

print("6. Saving to model.easyglm...")
rm.to_json("model.easyglm")
print("   Saved!")

# 7. Test scoring
print("\n7. Testing predictions...")
sample = df.head(3).select(predictors)
preds = rm.predict(sample)
print(f"   Input:\n{sample}")
print(f"   Predictions: {preds}")

# 8. Test column mapping
print("\n8. Testing column mapping...")
aliased = sample.rename({"DrivAge": "driver_age"})
preds2 = rm.predict(aliased, column_map={"driver_age": "DrivAge"})
print(f"   Mapped predictions: {preds2}")
assert np.allclose(preds, preds2), "Column mapping failed!"
print("   Column mapping works!")

# 9. Test reload
print("\n9. Testing JSON roundtrip...")
rm2 = RateModel.from_json("model.easyglm")
preds3 = rm2.predict(sample)
assert np.allclose(preds, preds3), "Roundtrip failed!"
print("   Roundtrip works!")

# 10. Test editing + snapshot
print("\n10. Testing edit + snapshot...")
drivage_table = rm.variables["DrivAge"].table
edit_row = drivage_table[2]
print(
    f"   Editing row: from={edit_row.from_}, to={edit_row.to_}, old_rel={edit_row.relativity:.3f}"
)
rm.update_relativity("DrivAge", from_=edit_row.from_, to_=edit_row.to_, new_value=2.0)
rm.create_snapshot("Doubled a bin")
print(f"   v2 prediction: {rm.predict(sample)}")
rm.switch_to(1)
print(f"   v1 prediction: {rm.predict(sample)}")
rm.switch_to(2)

# 11. Test metrics computation
print("\n11. Testing actual vs expected...")
from easy_glm.ui.metrics import compute_actual_expected

result = compute_actual_expected(rm, df, "DrivAge", formula="sum_weighted")
for subset_name, buckets in result["subsets"].items():
    n = len(buckets)
    print(
        f"   {subset_name}: {n} buckets, "
        f"actual range [{buckets[0]['actual']:.3f}, {buckets[-1]['actual']:.3f}]"
    )

print("\n✓ All tests passed!")
print("\nTo launch the UI editor, run:")
print("  from easy_glm.engine import RateModel")
print("  from easy_glm.ui import launch_editor")
print("  rm = RateModel.from_json('model.easyglm')")
print("  rm.launch_editor(data=df)")
