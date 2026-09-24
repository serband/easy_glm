# Stacking models and building rating files

This example fits a GLM, adds two interaction tables, then builds vehicle,
occupation and postcode files. Each new fit uses the complete model before it
as its offset. We review the vehicle file before fitting occupation, inspect
actual versus expected, and export the accepted files.

The order is your choice. The final section starts with CatBoost and adds a
GLM. Every named model remains available for comparison.

## Load the data and catalogues

`load_motor_catalogue_example` supplies invented motor data and three complete
catalogues. It needs no download. The catalogue codes include entries with no
training experience, so we can check that the exported files cover the full
lists. These are example attributes and effects, not recommended rating factors.

```python
from easy_glm import PricingModel, PricingSession, load_motor_catalogue_example

data, vehicles, occupations, postcodes = load_motor_catalogue_example()
data.head()
```

The experience data contains claims, exposure and rating keys. Attributes such
as vehicle power and occupation group are in the catalogues:

```python
vehicles.head()
occupations.head()
postcodes.head()
```

Each catalogue has one row per code. Keep codes such as `"0001"` as strings;
the leading zeros are part of the identifier.

## Set up the modelling session

`PricingSession` keeps the target, weights, split and bin settings together.
Here `Claims` is a count and `Exposure` is years. We fit Poisson frequency.
The existing `traintest` column uses **1 for training and 0 for holdout**.

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

These cuts control the GLM and the interaction tables. CatBoost uses the raw
attributes; a catalogue file then stores one score for each catalogue key.

## Fit the main effects

```python
main = work.fit_glm(
    "Main", factors=["DriverAge", "BonusMalus", "Region"],
    cv=3, n_alphas=8,
)
main.summary()
main.relativities("DriverAge").show()
main.ae("DriverAge").show()
```

The example uses three training folds and a shorter penalty search to keep the
run small. `main` is a fitted model; later fits will not change its tables.

## Add interactions in order

Inspect what the model misses before choosing an interaction:

```python
main.find_missing_factors().show()
main.find_interactions().show()
main.ae("DriverAge", "Region").show()
```

For this worked sequence, fit driver age with region, then bonus-malus with
region. The second fit offsets the first interaction's accepted table as well
as the GLM. Two tuning trials keep this example short.

```python
first_pair = main.fit_interaction(
    "DriverAge", "Region", name="First interaction",
    trials=2, prefix_trials=1,
)
first_pair.relativities("DriverAge", "Region").show()

second_pair = first_pair.fit_interaction(
    "BonusMalus", "Region", name="Two interactions",
    trials=2, prefix_trials=1,
)
second_pair.ae("BonusMalus", "Region").show()
```

## Build the vehicle file

`fit_catalogue` fits a component using the listed catalogue attributes and
scores every row of the full catalogue. The resulting model uses that lookup
file for scoring and for any later offset.

The vehicle component uses make, model, power and body type. Each is determined
by `VehicleCode`. Driver age stays outside this component because a
vehicle-only file cannot assign different scores to the same code by driver age.

```python
vehicle = second_pair.fit_catalogue(
    "Vehicle",
    catalogue=vehicles,
    key="VehicleCode",
    factors=["Make", "Model", "Power", "BodyType"],
    trials=2, cv=3,
)
vehicle.component("Vehicle").show()
vehicle.component("Vehicle").coverage(data)
vehicle.ae("Power").show()
vehicle.ae("DriverAge").show()
```

The component table shows the vehicle multipliers and their training support.
A code with no training rows still gets a score from its attributes. Review
those entries as well as the codes with experience. `coverage(data)` reports
missing keys and any use of an explicitly chosen fallback. The original main
effects and both interactions remain separately inspectable:

```python
vehicle.component("Main").show()
vehicle.component("First interaction").show()
vehicle.component("Two interactions").show()
```

Missing or duplicate codes and conflicting attributes are rejected. An unknown
scoring key raises an error by default; it does not silently receive 1.0.

## Review the vehicle rates before continuing

Try a cap and four-decimal rounding. The limits below demonstrate the controls;
choose your own limits after reviewing a real file.

```python
review = vehicle.edit_rates(name="Reviewed vehicle")
review.cap_component("Vehicle", lower=0.70, upper=1.50)
review.round_component("Vehicle", decimals=4)
review.preview().show()
```

The preview shows the proposed rate changes and the effect on the complete
model's expected claims. Apply creates a new named model; `vehicle` stays
unchanged:

```python
accepted_vehicle = review.apply(refit_later_components=False)
accepted_vehicle.ae("Power").show()
accepted_vehicle.compare(vehicle)
```

There are no components after vehicle yet. When amending an earlier component
in a longer model, `refit_later_components=False` keeps later rates fixed.
Use `True` to fit them again against the amended model. The two choices can
produce different pricing results.

## Build the occupation file

Fit occupation against the reviewed vehicle lookup, both interaction tables
and the GLM:

```python
occupation = accepted_vehicle.fit_catalogue(
    "Occupation",
    catalogue=occupations,
    key="OccupationCode",
    factors=["OccupationGroup", "PhysicalWork", "DrivingForWork"],
    trials=2, cv=3,
)
occupation.component("Occupation").show()
occupation.ae("OccupationGroup").show()
occupation.ae("DriverAge").show()
```

The occupation file contains an additional multiplier. It does not contain
the vehicle factor or the full prediction again.

## Build the postcode file

The same call works for postcode. These example codes and their attributes
are fictional.

```python
complete = occupation.fit_catalogue(
    "Postcode",
    catalogue=postcodes,
    key="Postcode",
    factors=["AreaType", "Density", "DistanceToCentre"],
    trials=2, cv=3,
)
complete.component("Postcode").show()
complete.ae("AreaType").show()
complete.ae("Region").show()
complete.ae("DriverAge", "Region").show()
```

Check the complete model across earlier rating factors too. Adding a new
component can change their A/E even though their original rates are fixed.

## Compare and request holdout validation

```python
complete.summary()
complete.compare(occupation, cv=True)
```

CV rebuilds the fitted components inside training folds. These are tuning
results for the chosen sequence. Request holdout only when ready:

```python
complete.compare(occupation, subset="holdout")
complete.ae("OccupationGroup", subset="holdout").show()
```

## Export all three files and the complete model

```python
complete.export_lookup("Vehicle", "vehicle_factors.csv")
complete.export_lookup("Occupation", "occupation_factors.csv")
complete.export_lookup("Postcode", "postcode_factors.parquet")
complete.to_excel("motor_rating_tables.xlsx")
complete.save("motor_model.easyglm")
```

Each file covers its full supplied catalogue, including unused codes. The
vehicle export contains the accepted capped and rounded values. Later
components were fitted using those same values.

The complete model combines all its components. Reopen it and score rows
without refitting:

```python
saved = PricingModel.load("motor_model.easyglm")
saved.predict(data.head())
saved.predict(data.head(), expected=True)
```

The first prediction is annual frequency; `expected=True` returns expected
claims for each row's exposure. Keep the keys needed by all three files in
the data supplied for full-model scoring.

## Start with CatBoost and add a GLM

This is a separate candidate from the same session. CatBoost fits first; the
GLM then fits the additional age and region effects against its predictions.

```python
catboost_base = work.fit_catboost(
    "CatBoost base", factors=["DriverAge", "BonusMalus"], trials=2, cv=3,
)
glm_correction = catboost_base.fit_glm(
    "GLM after CatBoost", factors=["DriverAge", "Region"], cv=3, n_alphas=8,
)
glm_correction.ae("DriverAge").show()
glm_correction.summary()
```

Driver age occurs in both components here. The GLM age table is the correction
after CatBoost, not the total age effect across the model. Components remain
separate so their contributions can be inspected and changed deliberately.

GLMs, CatBoost models, interaction tables and catalogue components can be
repeated or interleaved. The next fit always offsets the complete accepted
prediction. Changing their fitting order requires new fits and can change
how effects are allocated.
