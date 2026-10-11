# Score new data and export rating files

Use a fitted model on another dataset without refitting it. This example builds
main effects, vehicle, occupation and postcode models from one experience
dataset, then scores:

- a renewal extract with the complete fitted chain;
- each full rating file with its own accepted model;
- a new vehicle master with the saved CatBoost model, as a candidate file.

The accepted rating files can contain manual rate amendments. Scoring a new
master with CatBoost produces new candidate rates; it does not replace those
amendments.

> These new scoring controls are unreleased branch work. They are not in PyPI
> v0.472. See [the current status](../docs/HANDOVER.md).

Run the blocks in order in one notebook or session. The data below is synthetic.
For the fitting and review decisions in more detail, see
[building rating files](stacked_models.md).

## Build a small fitted chain

The loader returns experience and three complete catalogues. `PricingSession`
holds the response, exposure and training split. Each `fit_catalogue` call learns
from the experience, using the complete preceding model as its offset, then
stores a rate for every catalogue key.

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

Manual settings keep this scoring demonstration quick. For a modelling exercise,
use [Optuna tuning](model_variations.md#standalone-catboost-with-a-trial-budget).
Neither method fits on holdout rows.

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

Preview a vehicle cap, then apply it as a new named model. Later fits use the
amended vehicle rates.

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

`score` returns the input rows and columns with one added score column. It uses
the saved fits and accepted rates. It does not fit, create a training split or
apply the row filters used to select the training experience.

This extract has the required rating factors and catalogue keys. It has no
claims or training split.

```python
renewals = data.head(20).drop("Claims", "traintest")
renewal_rates = complete.score(renewals, column_prefix="renewal_")
renewal_rates.preview()
renewal_rates.export("scoring_output/renewal_rates.parquet")
```

`renewal_prediction` is claim frequency. To produce expected claim counts, name
the exposure column. This adds `renewal_expected` instead.

```python
renewal_counts = complete.score(
    renewals,
    exposure_col="Exposure",
    column_prefix="renewal_",
)
renewal_counts.preview()
renewal_counts.export("scoring_output/renewal_expected.csv")
```

Every Python export writes the data file and a neighbouring `.metadata.json`
file recording what was scored. Keep both. CSV and Parquet retain the source
row order; Parquet also retains the source column types.

## Score an individual rating file

`scope="model"` returns the selected model's own contribution. For Vehicle,
that is its accepted multiplier, including the cap above. It does not include
main effects, occupation, postcode or exposure. A vehicle file therefore needs
only its vehicle key; it does not need driver age or other catalogue keys.

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
`postcode_multiplier`. Codes such as `"0061"` remain strings, including their
leading zeroes. An unknown key blocks accepted-file scoring unless that saved
lookup has an explicit fallback.

For a GLM, its own contribution includes its base rate but excludes an external
offset. In a workbench plan,
selecting a named GLM also includes its accepted interaction tables. Older flat
recipes expose their separately named components; no ownership is inferred.

## Inspect the combined score through an earlier model

`scope="through"` includes all accepted components up to the selected model.
Here it returns main effects and the reviewed vehicle file, before occupation
and postcode. Unlike the vehicle multiplier alone, it needs driver age,
bonus-malus, region and the vehicle key.

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

Accepted lookup scoring returns the stored rates by key. To score new vehicle
attributes or genuinely new codes, use `scope="catalogue"`. This uses the saved
native CatBoost fit for Vehicle. Supply its key and the attributes it learned
from; claims and the preceding GLM factors are not needed.

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

`vehicle_candidate_multiplier` contains the native fitted correction. It does
not carry the manual vehicle cap. Review this file separately; scoring it does
not replace the accepted file or change later models. A saved lookup from before
native fits were retained can still score its accepted keys, but cannot produce
this candidate file; refit that catalogue model to enable it.

## Reopen the fit without the training data

Save the complete fitted bundle. Reopening it needs no training dataset to score
new data, including a new native catalogue candidate.

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

1. Build and fit the models from the modelling dataset as usual.
2. In **Project & data → Data**, use **Additional scoring datasets** to add a
   named CSV or Parquet file. It does not replace the modelling data or reset fits.
3. Open **Score on new data** and choose the dataset.
4. For renewals, choose **Overall prediction** and the pricing model whose listed
   models you want to use. For a vehicle file, choose **One rating model** and
   **Vehicle**. That returns the vehicle multiplier without the other models.
5. Read the summary of the dataset, models and output. The screen shows the
   required columns before you score. Use exposure when you want expected claims
   or amounts. Earlier combined scores are under **Advanced cumulative
   checkpoints**; a refreshed-master candidate is a separate option under its
   rating model.
6. Click **Check inputs**. Resolve missing factors, wrong types or uncovered keys.
7. Preview the first 20 rows, then download CSV or Parquet and its score-details
   JSON. The preview repeats which dataset and model produced the scores.

Uploaded files and Python frames belong to this running session. Local-file
references can be saved with the project; reopening reports missing or changed
files. Replacing a scoring dataset changes that dataset only.

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

The launcher snapshots those frames for the session. It does not send them into
the modelling fit merely because they are available for scoring.
