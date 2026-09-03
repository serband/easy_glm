"""Choose the bands for one rating factor.

Run ``basic_usage.py`` first. That model lets EasyGLM choose the bands. This
example sets the BonusMalus bands explicitly and leaves everything else the
same.
"""

import easy_glm

df = easy_glm.load_external_dataframe().sample(n=50_000, seed=42)
df = easy_glm.add_train_test_split(df, train_fraction=0.7, seed=42)

# By default, EasyGLM uses the training data to propose up to 20 bands for each
# numeric factor. It starts with bands containing roughly equal numbers of
# training rows. During fitting, it can decide that no change is needed at a
# boundary, so neighbouring bands may end with the same relativity.

# Here we choose a small set of round BonusMalus boundaries to make the rate
# table easier to review. Each value is a point at which its fitted relativity
# is allowed to change.
bonus_malus_boundaries = [55, 60, 70, 80, 90, 100]

model = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=["DrivAge", "Region", "BonusMalus", "Density"],
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    knots={"BonusMalus": bonus_malus_boundaries},
    cv=5,
)

print("BonusMalus rate table")
print(model.relativities["BonusMalus"].select("label", "relativity", "exposure"))

easy_glm.plot_all_ratetables({"BonusMalus": model.relativities["BonusMalus"]})

# export_rate_tables.py uses the complete saved fit to create an Excel workbook.
model.save("french_motor_model")

# score_new_data.py uses the smaller file containing only the fitted rate tables.
model.rate_model.to_json("french_motor.easyglm")

print("\nSaved french_motor_model for the Excel export example.")
print("Saved french_motor.easyglm for the new-data scoring example.")
