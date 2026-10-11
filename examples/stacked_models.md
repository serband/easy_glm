# Stacking models and building rating files

This example builds the sort of rating structure used when vehicle, occupation
and postcode are maintained as separate files. We start with a small frequency
GLM, add two interaction tables, then build a vehicle file, an occupation file
and a postcode file. Run a block, look at its tables or A/E, then decide whether
you want to carry that result into the next block.

> Stacked models are unreleased work; see [the current status](../docs/HANDOVER.md).

Each model after the first is a multiplier on the score built so far. For example,
the occupation model is fitted after the GLM, the two interaction tables and the
vehicle file; it learns what those have left unexplained. `ae` shows actual divided
by expected claims. It uses training rows unless you ask for holdout.

## Load experience and the complete catalogues

The package provides synthetic motor experience and complete vehicle, occupation
and postcode catalogues. Some catalogue codes are deliberately absent from the
experience, so we can check that the finished files still contain them.

```python
import polars as pl

from easy_glm import PricingModel, PricingSession, load_motor_catalogue_example

data, vehicles, occupations, postcodes = load_motor_catalogue_example(
    rows=6_000,
    seed=42,
)
data.head()
```

The identifiers are strings. `"0061"`, for example, is a valid vehicle code even
when it has no claims history in this sample.

```python
vehicles.head()
occupations.head()
postcodes.head()
```

If you also want to run this in the workbench, save the experience and the three
catalogues first. Parquet preserves string keys such as `"0061"`. The keys are
`VehicleCode`, `OccupationCode` and `Postcode`; use those names when you choose
the catalogue model.

```python
data.write_parquet("motor_experience.parquet")
vehicles.write_parquet("vehicle_catalogue.parquet")
occupations.write_parquet("occupation_catalogue.parquet")
postcodes.write_parquet("postcode_catalogue.parquet")
```

## Set the response, split and GLM bands

`PricingSession` is the working model session. It records the claim count,
exposure, policy identifier and split once, so you do not repeat them for every
fit. Here `traintest=1` means training and `0` means holdout. Searches, bands and
fits use training rows.

```python
work = PricingSession(
    data,
    family="poisson",
    claims="Claims",
    exposure="Exposure",
    id="PolicyID",
    split="traintest",
)
work.bands(default=8)
work.bands("DriverAge", cuts=[25, 35, 45, 55, 65, 75]).show()
work.bands("BonusMalus", cuts=[60, 75, 90, 100, 125]).show()
```

The age and bonus-malus cuts are used by the GLM and interaction tables. CatBoost
uses its numeric inputs as values rather than these bands. A catalogue model
returns one rate for each code in its catalogue.

## Fit and inspect the main effects

Start with three understandable main effects. `cv=3` chooses the GLM penalty on
three training folds; `n_alphas=8` keeps this example quick.

```python
main = work.fit_glm(
    "Main effects",
    factors=["DriverAge", "BonusMalus", "Region"],
    cv=3,
    n_alphas=8,
)
main.summary()
main.relativities("DriverAge").show()
main.ae("DriverAge").show()
```

Now inspect what the GLM has missed. The searches rank possible additions; they
do not change the model. The A/E views let you see the pattern behind a result
before you decide whether it is worth fitting.

```python
main.find_missing_factors().show()
main.ae("VehicleCode").show()
main.find_interactions().show()
main.ae("DriverAge", "Region").show()
```

## Add two pair tables, one after the other

A pair table is a two-factor CatBoost correction exported as a table you can
inspect. The first table is fitted on top of the GLM. The second is fitted on top
of the GLM and the first table. In the Python API, using a result as the parent of
the next fit is how you take it forward; there is no separate Keep button.

```python
first_pair = main.fit_interaction(
    "DriverAge",
    "Region",
    name="Age and region",
    trials=2,
    prefix_trials=1,
)
first_pair.relativities("DriverAge", "Region").show()
first_pair.ae("DriverAge").show()

second_pair = first_pair.fit_interaction(
    "BonusMalus",
    "Region",
    name="Age, region and bonus-malus",
    trials=2,
    prefix_trials=1,
)
second_pair.ae("BonusMalus", "Region").show()
```

The second pair adds to the first; it does not replace it or refit it. Before using
both as the base for the vehicle file, compare the GLM-only and current A/E.

These calls show the separate tables and the combined score. `main` still gives
the original GLM if you want to come back and compare it.

```python
second_pair.component("Main effects").show()
second_pair.component("Age and region").show()
second_pair.component("Age, region and bonus-malus").show()
main.ae("DriverAge").show()
second_pair.ae("DriverAge").show()
main.compare(second_pair)
```

## Fit the complete vehicle catalogue

The vehicle model learns from make, model, power and body type in the experience.
It then scores every `VehicleCode` in `vehicles`, including codes with no experience.
That is how it produces a complete vehicle file. `tune=True` asks Optuna to choose
settings on training folds; two trials are enough only to keep this example quick.

```python
vehicle = second_pair.fit_catalogue(
    "Vehicle",
    catalogue=vehicles,
    key="VehicleCode",
    factors=["Make", "Model", "Power", "BodyType"],
    tune=True,
    tuning_budget="trials",
    trials=2,
    cv=3,
    seed=42,
    thread_count=2,
)
vehicle.component("Vehicle").show()
vehicle.component("Vehicle").coverage(data)
vehicle.ae("Power").show()
```

The vehicle rates are extra multipliers after the GLM and both pair tables. They
are not complete policy prices on their own.

## Preview and accept a vehicle amendment

Suppose the vehicle rates are too wide for implementation. Start a review, cap
and round the vehicle table, then use the preview to see what this does to the
complete score. `apply` returns the reviewed result and leaves `vehicle` unchanged
for comparison.

```python
review = vehicle.edit_rates(name="Reviewed vehicle")
review.cap_component("Vehicle", lower=0.70, upper=1.50)
review.round_component("Vehicle", decimals=4)
review.preview().show()

accepted_vehicle = review.apply(refit_later_components=False)
accepted_vehicle.compare(vehicle)
accepted_vehicle.component("Vehicle").show()
```

There is nothing later to refit here. If you amend an earlier model in a longer
chain, `False` keeps later files as previously fitted; `True` rebuilds them from
the amended score. Those choices give different files.

## Add occupation and postcode files

Next build occupation from the reviewed vehicle result. It is fitted after the
GLM and both interaction tables too. `tune=False` means use the settings shown for
one training fit rather than run Optuna.

```python
occupation = accepted_vehicle.fit_catalogue(
    "Occupation",
    catalogue=occupations,
    key="OccupationCode",
    factors=["OccupationGroup", "PhysicalWork", "DrivingForWork"],
    tune=False,
    iterations=12,
    depth=3,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    seed=42,
    thread_count=2,
)
occupation.component("Occupation").show()
occupation.ae("OccupationGroup").show()
```

Then build postcode on top of everything through occupation.

```python
complete = occupation.fit_catalogue(
    "Postcode",
    catalogue=postcodes,
    key="Postcode",
    factors=["AreaType", "Density", "DistanceToCentre"],
    tune=False,
    iterations=12,
    depth=3,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    seed=42,
    thread_count=2,
)
complete.component("Postcode").show()
complete.ae("AreaType").show()
complete.ae("DriverAge", "Region").show()
```

## Inspect the cumulative score, then request holdout

The A/E below uses the full score: GLM, both pair tables, vehicle, occupation and
postcode. `cv=True` refits the recipes in training folds to compare candidates;
use it to choose what should go forward, not as the final independent check.

```python
complete.summary()
complete.ae("DriverAge").show()
occupation.ae("DriverAge").show()
complete.compare(occupation, cv=True)
```

When you have a candidate worth testing, request the untouched holdout rows.

```python
complete.compare(occupation, subset="holdout")
complete.ae("OccupationGroup", subset="holdout").show()
```

## Export, reopen and score an unseen catalogue code

The three lookup exports contain every supplied code. Saving the model keeps the
fitted GLM, interaction tables and catalogue files together. Loading it later
does not fit anything again.

```python
complete.export_lookup("Vehicle", "vehicle_factors.csv")
complete.export_lookup("Occupation", "occupation_factors.csv")
complete.export_lookup("Postcode", "postcode_factors.parquet")
complete.to_excel("motor_rating_tables.xlsx")
complete.save("motor_model.easyglm")

saved = PricingModel.load("motor_model.easyglm")
saved.predict(data.head())
saved.predict(data.head(), expected=True)
```

`"0061"` is in the vehicle catalogue but absent from the training data. It still
scores because the vehicle file was built for the complete catalogue.

```python
vehicle_rates = saved.component("Vehicle").show().table
vehicle_rates.filter(pl.col("VehicleCode") == "0061")

unseen_vehicle = data.head(1).with_columns(pl.lit("0061").alias("VehicleCode"))
saved.predict(unseen_vehicle)
```

The first prediction is a yearly unit rate. `expected=True` multiplies it by the
row's exposure to give expected claims. A full score needs all three catalogue keys.
