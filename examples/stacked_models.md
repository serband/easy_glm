# Stacking models and building rating files

This is an interactive pricing workflow. Run the blocks in order in one Python
session and inspect each returned table or chart before making the next modelling
decision. A fitted checkpoint is immutable: a later fit or a rate review creates
a new named model.

> **Scope.** The mixed-stack API in this page is current local work and is not in
> the published v0.472 release. The API workflow hub is [workflows.md](workflows.md).
> The data, catalogues and effects below are synthetic teaching material, not a
> rating recommendation.

A **component** is one fitted multiplier in the ordered score. A later component
uses the complete prediction from all earlier accepted components as its offset.
An **A/E** view compares actual with expected claims; it is shown on training rows
unless you explicitly request holdout data.

## Load experience and the complete catalogues

The packaged example has motor experience plus a complete vehicle, occupation and
postcode catalogue. The training data intentionally omits some catalogue codes,
so a later export can prove that the file still covers them.

```python
import polars as pl

from easy_glm import PricingModel, PricingSession, load_motor_catalogue_example

data, vehicles, occupations, postcodes = load_motor_catalogue_example(
    rows=6_000,
    seed=42,
)
data.head()
```

The identifier columns are strings. For example, vehicle code `"0061"` is a
valid catalogue key even when it has no training rows.

```python
vehicles.head()
occupations.head()
postcodes.head()
```

To reproduce the same inputs in the workbench, write the source data and
catalogues before fitting. Parquet preserves the string catalogue keys, including
leading zeroes in `VehicleCode`. The catalogue keys are `VehicleCode`,
`OccupationCode` and `Postcode`; keep those names when selecting each recipe.

```python
data.write_parquet("motor_experience.parquet")
vehicles.write_parquet("vehicle_catalogue.parquet")
occupations.write_parquet("occupation_catalogue.parquet")
postcodes.write_parquet("postcode_catalogue.parquet")
```

## Set the response, split and GLM bands

`PricingSession` records the response convention once. Here the response is claim
count, exposure supplies policy-years, and the supplied split uses `1` for training
and `0` for holdout. Searches, bands and fitting use training rows only.

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

Those cuts are the deployed GLM and pair-table bands. Direct CatBoost components
use raw values; a catalogue component stores one multiplier per catalogue key.

## Fit and inspect the main effects

Start with a deliberately small main-effects GLM. `cv=3` chooses its penalty on
three training folds; `n_alphas=8` keeps the example quick.

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

Search the main model's residuals before choosing an addition. Searches only rank
candidates; they never add a factor or a table by themselves.

```python
main.find_missing_factors().show()
main.ae("VehicleCode").show()
main.find_interactions().show()
main.ae("DriverAge", "Region").show()
```

## Add two pair tables, one after the other

A pair table is a separately fitted two-factor CatBoost correction converted to
an inspectable table. In this Python API, `first_pair` and `second_pair` are named
candidate checkpoints for review; there is no GUI-style **Keep** call. Continuing
from a checkpoint with the next `fit_...` call is the explicit decision to proceed
with it. `first_pair` offsets the main GLM. `second_pair` offsets the main GLM
**and** the first table. The names below identify cumulative checkpoints, not just
the last multiplier.

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

The second pair is a separate candidate table. It does not replace or jointly refit
the first one. Choosing it as the parent of the vehicle fit is the explicit decision
to continue from that checkpoint.

Inspect the accepted components and the cumulative score before adding the vehicle
file. Earlier named checkpoints remain unchanged.

```python
second_pair.component("Main effects").show()
second_pair.component("Age and region").show()
second_pair.component("Age, region and bonus-malus").show()
main.ae("DriverAge").show()
second_pair.ae("DriverAge").show()
main.compare(second_pair)
```

## Fit the complete vehicle catalogue

A catalogue component learns from the listed attributes, then scores every key in
the supplied catalogue. It therefore makes a complete vehicle rating file rather
than a table limited to experienced vehicles. This example makes the tuning choice
explicit: Optuna uses two candidate trials and three training folds.

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

The vehicle component has the whole main-and-pair scorer as its incoming offset.
The lookup itself is only the additional vehicle multiplier.

## Preview and accept a vehicle amendment

A rate review is a draft. Cap and round the lookup, inspect the effect on the
**complete** score, then publish a new named checkpoint. `vehicle` remains
available unchanged for comparison.

```python
review = vehicle.edit_rates(name="Reviewed vehicle")
review.cap_component("Vehicle", lower=0.70, upper=1.50)
review.round_component("Vehicle", decimals=4)
review.preview().show()

accepted_vehicle = review.apply(refit_later_components=False)
accepted_vehicle.compare(vehicle)
accepted_vehicle.component("Vehicle").show()
```

There are no later components at this point. If an earlier component already had
a suffix, `refit_later_components=False` would preserve that suffix as historical
rates; `True` would refit the affected suffix against the amended score. Choose
one deliberately, because the two policies can give different results.

## Add occupation and postcode files

The occupation component is fitted from the reviewed, capped vehicle checkpoint,
including both pair tables. It uses manual CatBoost settings: `tune=False` means
one full training fit with the stated parameters, without Optuna or
cross-validation.

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

Postcode is one more incremental catalogue. It offsets the complete accepted
score through occupation.

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

A/E below is for the complete score: main effects, both pair tables, vehicle,
occupation and postcode. The comparison with `cv=True` rebuilds the chosen
recipes inside training folds. It is tuning evidence, not independent validation.

```python
complete.summary()
complete.ae("DriverAge").show()
occupation.ae("DriverAge").show()
complete.compare(occupation, cv=True)
```

Holdout is deliberately separate. Ask for it only after the modelling decision
is ready to evaluate.

```python
complete.compare(occupation, subset="holdout")
complete.ae("OccupationGroup", subset="holdout").show()
```

## Export, reopen and score an unseen catalogue code

The three lookup exports contain the full supplied catalogues. Saving preserves
the complete accepted scorer. Reopening does not refit it.

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

Vehicle code `"0061"` is in the supplied catalogue but absent from the training
experience in this synthetic example. Full-model scoring still works because the
vehicle lookup was built from the complete catalogue.

```python
vehicle_rates = saved.component("Vehicle").show().table
vehicle_rates.filter(pl.col("VehicleCode") == "0061")

unseen_vehicle = data.head(1).with_columns(pl.lit("0061").alias("VehicleCode"))
saved.predict(unseen_vehicle)
```

The score above is an annual unit prediction; `expected=True` multiplies it by
that row's exposure. Keep all three lookup keys when scoring the full model.
