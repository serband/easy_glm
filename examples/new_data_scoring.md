# Score new data and export rating files

This worked example uses invented motor data to demonstrate scoring other
datasets. It builds main effects, vehicle, occupation and postcode models from
one experience dataset, then produces:

- a renewal extract with the complete fitted chain;
- each full rating file with its own accepted model;
- a new vehicle master with the saved CatBoost model, as a candidate file.

There are two different vehicle-file jobs here. An accepted vehicle file returns
the rates you have reviewed and signed off. A refreshed vehicle master is scored
with the saved CatBoost fit to create a candidate file for review; it does not
alter the accepted file.

Run the blocks in one notebook or Python session. The data is synthetic. For the
reasoning behind the modelling steps, see [building rating files](stacked_models.md).

## Build a small fitted chain

The loader returns experience plus complete vehicle, occupation and postcode
catalogues. `PricingSession` holds the response, exposure and split. Each
`fit_catalogue` call learns from the experience, after the preceding models, and
then produces a rate for every code in its catalogue.

```python
import polars as pl  # Select columns and construct a small new vehicle file.

from easy_glm import (
    PricingModel,  # Reopen the saved fitted model later.
    PricingSession,  # Define the response, split and modelling session.
    load_motor_catalogue_example,  # Experience and full example rating files.
)

data, vehicles, occupations, postcodes = load_motor_catalogue_example(
    rows=6_000,
    seed=42,
)
work = PricingSession(
    data,
    family="poisson",
    claims="Claims",
    exposure="Exposure",
    id="PolicyID",
    split="traintest",  # 1 = training; 0 = holdout.
)
work.bands("DriverAge", cuts=[25, 35, 45, 55, 65, 75])
main = work.fit_glm(
    "Main effects",
    factors=["DriverAge", "BonusMalus", "Region"],
    alpha=0.001,
)
```

The manual tree settings make this page quick to run. When you are choosing a
model, use [Optuna tuning](model_variations.md#standalone-catboost-with-a-trial-budget)
instead. Both approaches fit on training rows, not holdout.

```python
vehicle = main.fit_catalogue(
    "Vehicle",
    catalogue=vehicles,
    key="VehicleCode",
    factors=["Make", "Power", "BodyType"],
    tune=False,
    iterations=12,
    depth=3,
    seed=42,
    thread_count=2,
)
vehicle.component("Vehicle").show()
```

Cap the vehicle rates, inspect the preview, then carry the reviewed vehicle file
into the occupation and postcode fits.

```python
review = vehicle.edit_rates(name="Reviewed vehicle")
review.cap_component("Vehicle", lower=0.85, upper=1.15)
review.preview().show()
accepted_vehicle = review.apply(refit_later_components=False)

occupation = accepted_vehicle.fit_catalogue(
    "Occupation",
    catalogue=occupations,
    key="OccupationCode",
    factors=["OccupationGroup", "PhysicalWork", "DrivingForWork"],
    tune=False,
    iterations=12,
    depth=3,
    seed=42,
    thread_count=2,
)
complete = occupation.fit_catalogue(
    "Postcode",
    catalogue=postcodes,
    key="Postcode",
    factors=["AreaType", "Density", "DistanceToCentre"],
    tune=False,
    iterations=12,
    depth=3,
    seed=42,
    thread_count=2,
)
```

## Score a renewal extract with the complete chain

`score` keeps the rows you provide and adds score columns. It uses the fitted
models and reviewed rates; it does not refit anything or create a new split.

This renewal extract has the rating factors and catalogue keys needed by the full
model. It does not need claims or a training/holdout flag.

```python
renewals = data.head(20).drop("Claims", "traintest")
renewal_rates = complete.score(renewals, column_prefix="renewal_")
renewal_rates.preview()
renewal_rates.export("scoring_output/renewal_rates.parquet")
```

`renewal_prediction` is a claim frequency per year. Pass the exposure column when
you want expected claim counts; that produces `renewal_expected`.

```python
renewal_counts = complete.score(
    renewals,
    exposure_col="Exposure",
    column_prefix="renewal_",
)
renewal_counts.preview()
renewal_counts.export("scoring_output/renewal_expected.csv")
```

Each export writes the requested data file and a nearby `.metadata.json` file
describing the score. Keep them together. CSV and Parquet keep row order; Parquet
also retains column types.

## Score an individual rating file

For a vehicle rating file, you want the vehicle multiplier only, including the
reviewed cap. It does not need driver age, occupation or postcode, so the input
can contain only `VehicleCode`. `scope="model"` asks for that one model's rate.

```python
vehicle_rates = complete.score(
    vehicles.select("VehicleCode"),
    scope="model",
    model="Vehicle",
    column_prefix="vehicle_",
)
vehicle_rates.preview()
vehicle_rates.export("scoring_output/vehicle_rates.csv")

occupation_rates = complete.score(
    occupations.select("OccupationCode"),
    scope="model",
    model="Occupation",
    column_prefix="occupation_",
)
occupation_rates.export("scoring_output/occupation_rates.parquet")

postcode_rates = complete.score(
    postcodes.select("Postcode"),
    scope="model",
    model="Postcode",
    column_prefix="postcode_",
)
postcode_rates.export("scoring_output/postcode_rates.parquet")
```

The added columns are `vehicle_multiplier`, `occupation_multiplier` and
`postcode_multiplier`. Keep codes such as `"0061"` as strings so their leading
zeroes survive. Accepted-file scoring stops on an unknown code unless the saved
file defines a fallback.

## Inspect the combined score through an earlier model

Sometimes you want to see the combined rate after Vehicle but before Occupation
and Postcode. That needs the GLM factors as well as `VehicleCode`.
`scope="through"` asks for this point in the chain.

```python
through_vehicle = complete.score(
    renewals,
    scope="through",
    model="Vehicle",
    column_prefix="through_vehicle_",
)
through_vehicle.preview()
through_vehicle.export("scoring_output/through_vehicle.parquet")
```

## Score a refreshed master as a candidate file

A refreshed vehicle master needs a new candidate file, not the reviewed rates
from the old lookup. `scope="catalogue"` calls the saved CatBoost vehicle model.
Give it `VehicleCode` and the fields it learnt from; claims and GLM fields are
not required.

```python
new_vehicles = pl.DataFrame({
    "VehicleCode": ["NEW-001", "NEW-002"],
    "Make": ["Make1", "Make1"],
    "Power": [80.0, 140.0],
    "BodyType": ["Hatchback", "SUV"],
})
vehicle_candidate = complete.score(
    new_vehicles,
    scope="catalogue",
    model="Vehicle",
    column_prefix="vehicle_",
)
vehicle_candidate.preview()
vehicle_candidate.export("scoring_output/new_vehicle_candidate.parquet")
```

`vehicle_candidate_multiplier` is the CatBoost vehicle correction before the
manual cap. Review it as a new candidate; creating it does not replace the
accepted file or change later models. Older saved files can still return their
accepted rates, but must be refitted before they can score a refreshed master.

## Reopen the fit without the training data

Save the complete fitted model when you need to score later without loading the
training data.

```python
complete.save("scoring_output/approved_model.easyglm")
approved = PricingModel.load("scoring_output/approved_model.easyglm")
approved.score(
    renewals,
    exposure_col="Exposure",
    column_prefix="approved_",
).export("scoring_output/approved_renewals.parquet")
```

## Use the workbench

1. Fit the model chain from the experience dataset.
2. In **Project & data → Data**, add a named CSV or Parquet file under
   **Additional scoring datasets**. It is separate from the modelling data.
3. Open **Score on new data** and choose the dataset first, then the saved model.
4. For renewals choose **Overall prediction**. For a vehicle file choose
   **One rating model** then **Vehicle**; this returns only the vehicle multiplier.
5. Read the required columns shown on screen. Select exposure if you need expected
   claims or amounts. **Advanced cumulative checkpoints** gives a combined score
   part-way through the chain. A refreshed-master candidate sits under its rating
   model.
6. Click **Check inputs**, fix any missing columns, types or keys, then preview
   the first 20 rows.
7. Download CSV or Parquet with its score-details JSON.

Uploaded files and Python frames belong to the running session. Saved local-file
references are checked when you reopen a project. Replacing one scoring dataset
does not change the others or the fitted model.

If you open the workbench from Python, pass the additional frames by name:

```python
# from easy_glm.app import launch
# launch(
#     data=data,
#     scoring_data={
#         "Renewals": renewals,
#         "Full vehicle file": vehicles,
#         "Full occupation file": occupations,
#         "Full postcode file": postcodes,
#     },
# )
```

The workbench keeps these frames available for scoring. It does not fit them.
