"""Fit a Poisson model for annual claim frequency."""

import easy_glm

# Download the public French motor data once; later runs use the local cache.
df = easy_glm.load_external_dataframe().sample(n=50_000, seed=42)

# Use 70% of the data to fit the model. The remaining 30% are kept unseen so
# that the fitted model can be checked on separate test data.
df = easy_glm.add_train_test_split(df, train_fraction=0.7, seed=42)

model = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=["DrivAge", "Region", "BonusMalus", "Density"],
    weight_col="Exposure",
    train_test_col="traintest",
    # ClaimNb divided by Exposure is the annual claim frequency.
    divide_target_by_weight=True,
    # Five-fold cross-validation chooses the penalty strength.
    cv=5,
)

print("Fitted relativities")
print(f"Base claim frequency: {model.base_rate:.5f} claims per policy-year")
print("\nBonusMalus")
print(model.relativities["BonusMalus"].select("label", "relativity", "exposure"))
print("\nRegion")
print(model.relativities["Region"].select("label", "relativity", "exposure"))

# These plots show the factor shapes fitted by the model.
easy_glm.plot_all_ratetables(model.relativities)

print("\nActual and model-expected claim frequency")
# These charts check actual and model-expected claim frequency by fitted band
# or level, separately for the training and test data. Exposure is shown below.
model.plot_actual_vs_expected(df)
