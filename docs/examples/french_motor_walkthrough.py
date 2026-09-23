# Building a pricing model
# Run one cell at a time. Explanations: examples/pricing_walkthrough.md

# %% 1 — Modelling step
from easy_glm import load_external_dataframe

data = (
    load_external_dataframe()
    .sort("IDpol")
    .sample(n=50_000, seed=20260902)
    .sort("IDpol")
)

# %% 2 — Modelling step
from easy_glm import PricingSession

work = PricingSession(
    data,
    family="poisson",
    claims="ClaimNb",
    exposure="Exposure",
    id="IDpol",
    ignored=["VehGas"],
    train_fraction=0.70,
    seed=42,
)
work.summary()

# %% 3 — Modelling step
# work = PricingSession(
#     data,
#     family="poisson",
#     claims="ClaimNb",
#     exposure="Exposure",
#     id="IDpol",
#     ignored=["VehGas"],
#     split="traintest",  # Existing column: 1 = training, 0 = holdout.
# )

# %% 4 — Modelling step
work.bands(default=8)

# %% 5 — Modelling step
work.bands("VehAge", number=6).show()

# %% 6 — Modelling step
work.bands("DrivAge", cuts=[25, 35, 45, 55, 65, 75]).show()

# %% 7 — Modelling step
work.bands("VehAge", cuts=[1, 3, 6, 10, 15]).show()
work.bands("BonusMalus", cuts=[50, 60, 75, 100, 125]).show()
work.bands("Density", cuts=[50, 200, 1_000, 5_000, 10_000]).show()

# %% 8 — Modelling step
work.categories("Region").show()

# %% 9 — Modelling step
work.save_settings("motor_settings.json")

# To reopen these settings with the same policies:
# work = PricingSession.from_settings(data, "motor_settings.json")

# %% 10 — Modelling step
basic = work.fit_glm("Age model", factors=["DrivAge", "VehAge"])
basic.summary()

# %% 11 — Modelling step
basic.relativities("DrivAge").show()
basic.ae("DrivAge").show()
basic.ae("VehAge").show()

# %% 12 — Modelling step
# work.bands("DrivAge", cuts=[25, 35, 45, 55, 65, 70, 75])
# different_bands = basic.refit("Different driver-age bands")
# different_bands.compare(basic, cv=True)

# %% 13 — Modelling step
basic.find_missing_factors().show()

# %% 14 — Modelling step
basic.ae("BonusMalus").show()
basic.ae("Density").show()

# %% 15 — Modelling step
main = basic.refit("Four-factor model", add=["BonusMalus", "Density"])
main.compare(basic, cv=True)
main.relativities("BonusMalus").show()
main.ae("BonusMalus").show()
main.ae("Density").show()
main.find_missing_factors().show()

# %% 16 — Modelling step
# without_density = main.refit("Without density", remove=["Density"])
# without_density.compare(main, cv=True)

# %% 17 — Modelling step
main.find_interactions().show()
main.ae("DrivAge", "BonusMalus").show()

# %% 18 — Modelling step
first = main.fit_interaction(
    "DrivAge",
    "BonusMalus",
    name="First interaction",
)
first.summary()
first.relativities("DrivAge", "BonusMalus").show()
first.ae("DrivAge", "BonusMalus").show()
first.compare(main)

# %% 19 — Modelling step
first.find_missing_factors().show()
first.find_interactions().show()
first.ae("VehAge", "Region").show()

# %% 20 — Modelling step
second = first.fit_interaction(
    "VehAge",
    "Region",
    name="Second interaction",
)
second.summary()
second.relativities("VehAge", "Region").show()
second.ae("VehAge", "Region").show()
second.compare(first)

# %% 21 — Modelling step
second.relativities("DrivAge").show()
second.ae("DrivAge").show()

# %% 22 — Modelling step
rates = second.edit_rates(name="Reviewed rates")
rates.set_relativity("DrivAge", lower=25, upper=35, value=0.95)
rates.preview().show()

# %% 23 — Modelling step
adjusted = rates.apply(refit_later_interactions=False)
adjusted.ae("DrivAge").show()
adjusted.compare(second)

# %% 24 — Modelling step
refit_rates = second.edit_rates(name="Reviewed rates with refitted interactions")
refit_rates.set_relativity("DrivAge", lower=25, upper=35, value=0.95)
refitted = refit_rates.apply(refit_later_interactions=True)
refitted.ae("DrivAge").show()
refitted.compare(adjusted)

# %% 25 — Modelling step
# rates.rebalance()
# rates.preview().show()
# balanced = rates.apply(refit_later_interactions=False)

# %% 26 — Modelling step
cell_review = second.edit_rates(name="One interaction cell changed")
cell_review.set_pair_relativity(
    "DrivAge",
    "BonusMalus",
    lower_a=25,
    upper_a=35,
    lower_b=50,
    upper_b=60,
    value=1.05,
)
cell_review.preview().show()
cell_candidate = cell_review.apply(refit_later_interactions=False)
cell_candidate.relativities("DrivAge", "BonusMalus").show()

# %% 27 — Modelling step
refitted.validate_holdout(compare_with=[basic, main, first, second, adjusted])
refitted.ae("DrivAge", subset="holdout").show()
refitted.ae("VehAge", "Region", subset="holdout").show()

# %% 28 — Modelling step
accepted = refitted

# %% 29 — Modelling step
accepted.to_excel("motor_pricing_tables.xlsx")
accepted.save("motor_pricing_model.easyglm")

# %% 30 — Modelling step
from easy_glm import PricingModel  # Reopen a saved fitted model.

reopened = PricingModel.load("motor_pricing_model.easyglm", data=data)
reopened.relativities("DrivAge").show()
reopened.ae("DrivAge").show()

# %% 31 — Modelling step
predicted_frequency = reopened.predict(data)
expected_claims = reopened.predict(data, expected=True)
