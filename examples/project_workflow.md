# Save, edit and run a model recipe

Use this when you want to rerun a model setup, hand it to someone else, or use
the setup downloaded from the workbench. A `Project` saves where the data lives,
which columns play each role, the split, variable settings and model definitions.
The fitted rates are saved separately in an `.easyglm` file.

`Project.from_dict(settings)` reads that setup from a Python dictionary. It is
the same structure used by project JSON and the workflow exported from the
workbench. `validate`, used below, checks that the settings work with your data.

This page shows how to build and change a recipe in Python. For a model-building
session with searches and charts, use [Building a pricing model](pricing_walkthrough.md).

Run the blocks in order. They write the example data, recipe and outputs into the
current directory.

## Load the example data

The package supplies a small invented motor dataset and complete vehicle,
occupation and postcode catalogues, so there is no separate download. The
[rating-file walkthrough](stacked_models.md) explains the catalogue workflow.

```python
from easy_glm import load_motor_catalogue_example

data, vehicles, occupations, postcodes = load_motor_catalogue_example()
data.head()
```

We will fit a direct vehicle CatBoost model later. It needs vehicle attributes on
every row it fits and scores, so join them now. Each `VehicleCode` must match no
more than one catalogue row.

```python
data = data.join(vehicles, on="VehicleCode", how="left", validate="m:1")
data.write_parquet("experience.parquet")
```

The [catalogue API](stacked_models.md#fit-the-complete-vehicle-catalogue) is the
alternative when you want a complete lookup scored by code. The direct CatBoost
model used here instead needs `Make`, `Power` and `BodyType` on scoring data.

## Write the setup once

There are three parts to the recipe below:

| Part | What we are specifying |
| --- | --- |
| `data` | File location, shared column roles and the existing train/holdout flags. |
| `design` | Default numeric binning and any variable-specific treatment. |
| `models` | Named fitting recipes. `predictors` selects the main effects of this GLM. |

`Claims` is a count and `Exposure` is policy-years. Dividing claims by exposure
and fitting with exposure weights gives a frequency model. Its basic prediction
is a yearly frequency; expected claim counts include exposure.

The existing `traintest` column has **1 for training and 0 for holdout**. Giving
both values explicitly prevents other labels from becoming holdout by accident.

We make `DriverAge`, `BonusMalus` and `Region` available to models, then select
only age and bonus-malus in the first fit. `OccupationCode` and `Postcode` are
explicitly ignored. Any column left out of this list is simply not selected.

Numeric predictors use eight automatic bands unless you override them. Here age
uses the cuts we specify, bonus-malus uses six automatic bands, and `Region` is
categorical. Automatic boundaries use training rows only. Explicit cuts win over
a variable's band count, which wins over the default. `[35, 45)` includes 35 and
excludes 45.

The GLM penalty reduces unstable fitted effects. Here three training folds choose
from eight candidates; holdout rows are not used.

```python
recipe = {
    "name": "Motor frequency",
    "data": {
        "source": {"type": "parquet", "path": "experience.parquet"},
        "roles": {
            "PolicyID": "id",
            "Claims": "target",
            "Exposure": "weight",
            "traintest": "split",
            "DriverAge": "predictor",
            "BonusMalus": "predictor",
            "Region": "predictor",
            "OccupationCode": "ignore",
            "Postcode": "ignore",
        },
        "split": {
            "mode": "column", "column": "traintest",
            "train_value": 1, "holdout_value": 0,
        },
    },
    "design": {
        "defaults": {"n_bins": 8},
        "variables": {
            "DriverAge": {"kind": "step", "knots": [25, 35, 45, 55, 65, 75]},
            "BonusMalus": {"kind": "step", "n_bins": 6},
            "Region": {"kind": "categorical"},
        },
    },
    "models": {
        "Age and bonus": {
            "family": "poisson", "link": "log",
            "target": "Claims", "weight": "Exposure",
            "divide_target_by_weight": True,
            "predictors": ["DriverAge", "BonusMalus"],
            "penalty": {"cv": 3, "n_alphas": 8},
        },
    },
    "champion": "Age and bonus",
}
```

You do not need to fill every possible field. EasyGLM uses defaults for omitted
settings and writes the resulting settings when you save the recipe.

## Read, check and fit the recipe

The next calls do separate jobs:

- `Project.from_dict` reads the settings.
- `prepare` reads the file and applies the project preparation. It has not fitted
  a model.
- `run_model` fits the named model using training rows.

`validate` returns a list of setup problems. An empty list means the checks found
nothing to stop the fit.

```python
from easy_glm.workflow import Project, prepare, run_model

project = Project.from_dict(recipe)
prepared = prepare(project)
problems = project.validate(columns=prepared.columns)
if problems:
    raise ValueError(problems)
```

Fix any listed problem before fitting. Then check that the split is as intended:

```python
prepared["traintest"].value_counts().sort("traintest")
```

This example contains 4,500 training rows and 1,500 holdout rows. Now fit:

```python
main_run = run_model(project, prepared, "Age and bonus")
main_run.metrics["train"]
main_run.tables["DriverAge"]
```

`main_run` contains the fitted tables and training metrics. These figures are
in-sample. Rerunning `run_model` fits again; it does not retrieve a saved result.

## Save the setup and reopen it

`to_json` writes the recipe. `from_json` reads it again.

```python
project.to_json("motor_project.json")
reopened_project = Project.from_json("motor_project.json")
reopened_project.models["Age and bonus"].predictors
```

When you already have settings as a Python dictionary, JSON string or output from
another tool, use `from_dict` directly:

```python
import json
from pathlib import Path

settings = json.loads(Path("motor_project.json").read_text())
same_project = Project.from_dict(settings)
same_project.models["Age and bonus"].predictors
```

Opening a recipe never recovers fitted coefficients. Save the fitted model as
well if you need to score with exactly those rates.

## Try another main effect

To test region, copy the recipe, name the candidate and add `Region` to its
predictors. The copy is a separate set of settings; it does not alter or refit
the first result.

```python
revised = project.copy()
revised.models["With region"] = revised.models.pop("Age and bonus")
revised.models["With region"].predictors = ["DriverAge", "BonusMalus", "Region"]
revised.champion = "With region"

region_run = run_model(revised, prepared, "With region")
region_run.metrics["train"]
main_run.metrics["train"]
```

You now have two results to compare. For residual searches, A/E charts,
cross-validated comparisons and rate review, use the
[interactive workflow](pricing_walkthrough.md).

## Describe a GLM followed by CatBoost

This recipe says: fit Main effects first, then fit Vehicle after it. Each entry
names the factors and settings for that part of the model.

Here the GLM fits age, bonus-malus and region. Vehicle CatBoost then fits make,
power and body type after the GLM score. EasyGLM calculates that offset for you;
there is no offset column to create.

This replay uses a fixed 100-tree fit. The workbench defaults to Optuna tuning. The
[model variations](model_variations.md#standalone-catboost-with-a-trial-budget) show the
API's explicit trial and time budgets.

```python
stack_recipe = project.to_dict()
stack_recipe["models"] = {
    "GLM then vehicle": {
        "family": "poisson", "link": "log",
        "target": "Claims", "weight": "Exposure",
        "divide_target_by_weight": True,
        "components": [
            {
                "component_id": "main", "name": "Main effects", "kind": "glm",
                "factors": ["DriverAge", "BonusMalus", "Region"],
                "settings": {"cv": 3, "n_alphas": 8},
            },
            {
                "component_id": "vehicle", "name": "Vehicle", "kind": "catboost",
                "factors": ["Make", "Power", "BodyType"],
                "settings": {
                    "tune": False, "iterations": 100, "depth": 4,
                    "learning_rate": 0.05, "seed": 42, "thread_count": 2,
                },
            },
        ],
    },
}
stack_recipe["champion"] = "GLM then vehicle"

stack_project = Project.from_dict(stack_recipe)
problems = stack_project.validate(columns=prepared.columns)
if problems:
    raise ValueError(problems)
stack_run = run_model(stack_project, prepared, "GLM then vehicle")
stack_run.predict(prepared.head())
```

Predictions include the GLM and vehicle CatBoost. They are unit frequencies.
CatBoost does not refit the GLM. A direct CatBoost model remains a fitted model;
it is not converted into an exact GLM rate table.

Use `stack_run.predict(data)` for the complete prediction. An individual
component table or estimator shows that component's own contribution.

## Export the accepted setup and the fitted scorer

Choose the export based on what you need next:

| Export | What it keeps | Does it refit? |
| --- | --- | --- |
| Project JSON | Data path, preparation and model settings | Opening it does not; fitting it does. |
| Fitted `.easyglm` bundle | The current complete fitted scorer | No. |
| Python scorer | Code containing the saved scorer | No. |
| Python workflow | Code rebuilding the accepted fitting recipe | Yes. |
| Excel component workbook | Tables and component information for review | No. |

`write_stack_bundle` saves the complete fitted result. `export_stack_excel`
writes the review workbook. With the shorter interactive API, use
`model.save(...)` and `model.to_excel(...)`.

```python
from easy_glm.workflow.export import write_stack_bundle
from easy_glm.pricing.excel import export_stack_excel

stack_project.to_json("stack_project.json")
write_stack_bundle(stack_project, stack_run, "accepted_model.easyglm")
export_stack_excel(stack_run, "accepted_tables.xlsx")
```

Reopen the bundle when you need to score. `expected=True` turns the unit
frequency into expected claims using each row's exposure.

```python
from easy_glm import PricingModel

saved = PricingModel.load("accepted_model.easyglm")
saved.predict(data.head())
saved.predict(data.head(), expected=True)
```

## Generate the two Python exports

`to_script` writes a fitting workflow from the recipe. `to_scoring_script` writes
a scorer that uses the fitted rates. Use the scorer when you need the same fitted
output; use the workflow when you deliberately want to rebuild the model.

```python
from easy_glm.workflow import to_script, to_scoring_script

Path("rebuild_model.py").write_text(
    to_script(stack_project, "GLM then vehicle", run=stack_run,
              output_prefix="rebuilt_model"),
    encoding="utf-8",
)
Path("score_model.py").write_text(
    to_scoring_script(stack_run), encoding="utf-8",
)
```

Check `rebuild_model.py` before running it. Its `SOURCE_PATH` points to the
experience file, so change it if the file moves. It keeps the saved split, fits
components in order, and writes `rebuilt_model.easyglm` and
`rebuilt_model_rate_tables.xlsx`. It includes the amendments you applied, not
searches or candidates you rejected.

Run the workflow in a new Python process:

```python
import subprocess
import sys

subprocess.run([sys.executable, "rebuild_model.py"], check=True)
rebuilt = PricingModel.load("rebuilt_model.easyglm")
rebuilt.predict(data.head())
```

Refitting can give different values, particularly with tuning or when later
corrections were kept after an earlier rate amendment. Use the saved bundle or
Python scorer when you need the accepted rates without refitting.

Load the Python scorer and call `predict`. A mixed stack returns a unit value by
default; add `exposure_col` for expected counts:

```python
import runpy

scorer = runpy.run_path("score_model.py")
scorer["predict"](data.head())
scorer["predict"](data.head(), exposure_col="Exposure")
```

The scorer expects the same prepared attributes and key types as the fitted data.
Keep your normal data preparation before this step.

## Continue between GUI and API

- Open `motor_project.json` with **Saved project** in the workbench to load the
  recipe. It does not carry fitted models with it.
- After fitting in the workbench, use **Export → Rebuild this model → Download
  Python workflow** when you want the order, settings and amendments as code.
- Use **Score new data → Download Python scorer** or the fitted bundle when you
  want to apply the saved model without fitting again.
- Use `PricingSession` for interactive fitting and review. See the
  [full rating-file example](stacked_models.md).
