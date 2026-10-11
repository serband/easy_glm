# Save, edit and run a model recipe

A `Project` holds the instructions for a modelling job: where to read the data,
which columns are the response and weight, how to split the rows, how to treat
variables, and which models to fit. It does not hold the fitted model.

`Project.from_dict(settings)` turns a Python dictionary of those instructions
into a `Project`. This is the same format used by project JSON and by the
workbench's generated workflow. It does not load data or start fitting.

Use this page when you want to save a recipe, change it deliberately, or
understand the workflow downloaded from the workbench. For interactive fitting
and charts, start with [Building a pricing model](pricing_walkthrough.md).
For the complete set of workflows, see [Examples and workflows](workflows.md).

The mixed-model sections describe the current implementation, which is not
included in the published v0.472 package. The documentation and release status
are listed in [the handover](../docs/HANDOVER.md).

Run the Python blocks in order in one session. The example writes its data,
project and exports into the current directory.

## Load the example data

The package supplies a small invented motor dataset and complete vehicle,
occupation and postcode catalogues. We use it here so the example needs no
download. The [rating-file walkthrough](stacked_models.md) explains the
catalogues in detail.

```python
from easy_glm import load_motor_catalogue_example

data, vehicles, occupations, postcodes = load_motor_catalogue_example()
data.head()
```

We will fit a vehicle CatBoost model later. A direct CatBoost model needs its
attributes on the data it fits and scores, so join the vehicle attributes now.
Each vehicle code must match at most one row of the catalogue.

```python
data = data.join(vehicles, on="VehicleCode", how="left", validate="m:1")
data.write_parquet("experience.parquet")
```

The [catalogue API](stacked_models.md#fit-the-complete-vehicle-catalogue) is another option:
it retains the complete catalogue and scores by code. A direct CatBoost model,
as used on this page, requires `Make`, `Power` and `BodyType` when scoring.

## Write the setup once

There are three parts to the recipe below:

| Part | What we are specifying |
| --- | --- |
| `data` | File location, shared column roles and the existing train/holdout flags. |
| `design` | Default numeric binning and any variable-specific treatment. |
| `models` | Named fitting recipes. `predictors` selects the main effects of this GLM. |

Here `Claims` is a count and `Exposure` is years. Dividing the count by exposure
and fitting with exposure weights gives a frequency model. Predictions are
unit frequencies; expected claim counts include exposure.

The existing `traintest` column has **1 for training and 0 for holdout**. Giving
both values explicitly prevents other labels from becoming holdout by accident.

We make `DriverAge`, `BonusMalus` and `Region` available as predictors but select
only the first two in the first model. We ignore `OccupationCode` and `Postcode`
in this project. The other columns remain unassigned; their presence does not
add them to a model.

Numeric variables get eight automatic bins unless we specify otherwise.
`DriverAge` has explicit cuts and `BonusMalus` has six automatic bins. Automatic
boundaries come from training rows only. Explicit cuts take precedence over a
variable's bin count; that count takes precedence over the default. The age
band `[35, 45)` includes 35 and excludes 45. `Region` is categorical.

The penalty shrinks fitted GLM coefficients. Here three training folds choose
it from eight candidates. The holdout is not used to choose the penalty.

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

There is no need to fill every field. EasyGLM supplies omitted defaults when
reading the recipe and writes them out when you save it.

## Read, check and fit the recipe

The three calls below do separate jobs:

- `Project.from_dict` reads the settings. It can migrate older supported project
  formats; a newer unsupported format is rejected. Unknown fields can raise
  warnings, so check spelling rather than assuming a setting was applied.
- `prepare` reads the file, applies the project's preparation and creates the
  prepared training/holdout flag. It has not fitted anything yet.
- `run_model` fits the named entry in `project.models`, using training rows.

`validate` returns a list of setup problems. An empty list means no problems
were found by those checks.

```python
from easy_glm.workflow import Project, prepare, run_model

project = Project.from_dict(recipe)
prepared = prepare(project)
problems = project.validate(columns=prepared.columns)
if problems:
    raise ValueError(problems)
```

If that returns a problem, correct the recipe before fitting. Check the split:

```python
prepared["traintest"].value_counts().sort("traintest")
```

This example contains 4,500 training rows and 1,500 holdout rows. Now fit:

```python
main_run = run_model(project, prepared, "Age and bonus")
main_run.metrics["train"]
main_run.tables["DriverAge"]
```

`main_run` is the result, with its fitted tables and the setup used for the fit.
The metrics above are in-sample training metrics. We have not asked to display
holdout results. `run_model` is a fitting call; rerunning it fits again.

## Save the setup and reopen it

`to_json` writes the recipe. `from_json` is the convenient way to read that file;
it calls `from_dict` internally.

```python
project.to_json("motor_project.json")
reopened_project = Project.from_json("motor_project.json")
reopened_project.models["Age and bonus"].predictors
```

When the settings come from an LLM, another application or an existing JSON
string, use `from_dict` directly:

```python
import json
from pathlib import Path

settings = json.loads(Path("motor_project.json").read_text())
same_project = Project.from_dict(settings)
same_project.models["Age and bonus"].predictors
```

Neither opening call recovers fitted coefficients. Keep the fitted scorer as
well if you need to use those exact rates later.

## Try another main effect

Make a separate project copy, give the candidate a new name, then add `Region`
to that candidate's main effects. `copy` makes independent settings; it does not
copy or refit the fitted result.

```python
revised = project.copy()
revised.models["With region"] = revised.models.pop("Age and bonus")
revised.models["With region"].predictors = ["DriverAge", "BonusMalus", "Region"]
revised.champion = "With region"

region_run = run_model(revised, prepared, "With region")
region_run.metrics["train"]
main_run.metrics["train"]
```

Both results remain available. For residual searches, plots, cross-validated
comparisons and rate review through short calls, use the
[interactive workflow](pricing_walkthrough.md). You do not need to write chart
or offset helpers around this recipe.

## Describe a GLM followed by CatBoost

`components` defines the fitting order **inside one complete named result**.
It is the fitting representation used by stacked workflow exports. Each
component has its own `factors` and fitting `settings`. Its `component_id`
identifies it in the saved recipe; its `name` is the label you inspect.

Here the GLM fits age, bonus-malus and region. The vehicle CatBoost then fits
make, power and body type with the complete GLM prediction as its offset.
The package calculates and passes that offset. We do not create an offset
column ourselves.

We use a manual 100-tree fit for this small replay example. The workbench
defaults to Optuna tuning. The
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

The predictions include both components. They are unit frequencies. CatBoost
does not refit the first GLM. Direct CatBoost is retained as a fitted model;
unlike a two-variable interaction table, it is not exported as an exact GLM
rate table.

Use `stack_run.predict(data)` for the complete prediction. A component's fitted
estimator or rate table contains only that component's contribution.

In a workbench project you may also see `model_plan`. That describes the
editable screen setup: named models, GLM-owned interaction tables and references
to saved fits. `run_model` fits an entry in `models`; it does not execute
`model_plan` by itself. Use the GUI to fit an unfinished plan, or download a
Python workflow from the fitted version. There is no need to hand-write the
workbench's plan IDs and fit references.

Per-model variable JSON in the GUI is a smaller selection format. It is not a
complete `Project` dictionary. Shared roles JSON is smaller again: column names
mapped to roles. Use the editor labelled for the thing you want to change.

## Export the accepted setup and the fitted scorer

These exports do different jobs:

| Export | What it keeps | Does it refit? |
| --- | --- | --- |
| Project JSON | Data path, preparation and model settings | Opening it does not; fitting it does. |
| Fitted `.easyglm` bundle | The current complete fitted scorer | No. |
| Python scorer | Code containing the saved scorer | No. |
| Python workflow | Code rebuilding the accepted fitting recipe | Yes. |
| Excel component workbook | Tables and component information for review | No. |

`write_stack_bundle` saves the complete result returned by the project workflow.
`export_stack_excel` writes its component workbook. In the shorter interactive
API the same actions are `model.save(...)` and `model.to_excel(...)`.

```python
from easy_glm.workflow.export import write_stack_bundle
from easy_glm.pricing.excel import export_stack_excel

stack_project.to_json("stack_project.json")
write_stack_bundle(stack_project, stack_run, "accepted_model.easyglm")
export_stack_excel(stack_run, "accepted_tables.xlsx")
```

Reopen the bundle for scoring. `expected=True` returns claim counts for each
row's exposure instead of a unit frequency.

```python
from easy_glm import PricingModel

saved = PricingModel.load("accepted_model.easyglm")
saved.predict(data.head())
saved.predict(data.head(), expected=True)
```

## Generate the two Python exports

`to_script` generates a **fitting workflow**. Pass the fitted run so it can use
the accepted recipe and its saved setup. `to_scoring_script` generates a
**frozen scorer** instead.

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

Inspect `rebuild_model.py`. Its `SOURCE_PATH` points to the experience file;
change it if that file moves. The script reads it, keeps the saved split, fits
the components in order and writes `rebuilt_model.easyglm` and
`rebuilt_model_rate_tables.xlsx`. Accepted amendments are replayed; residual
searches and discarded candidates are not. An adopted frozen starting model
remains frozen.

Run the workflow in a new Python process:

```python
import subprocess
import sys

subprocess.run([sys.executable, "rebuild_model.py"], check=True)
rebuilt = PricingModel.load("rebuilt_model.easyglm")
rebuilt.predict(data.head())
```

Refitting can change fitted values, especially with tuning or when earlier
rates were amended while later fitted corrections were kept fixed. Use the
saved bundle or Python scorer when you need the accepted rates without refitting.

Load the Python scorer and call its `predict`. For a mixed stack this returns
unit values by default; supply `exposure_col` for expected counts:

```python
import runpy

scorer = runpy.run_path("score_model.py")
scorer["predict"](data.head())
scorer["predict"](data.head(), exposure_col="Exposure")
```

These rows have the same attributes and key types as the training data. The
export expects prepared scoring columns; it is not a general replacement for
your data preparation.

## Continue between GUI and API

- Open `motor_project.json` using **Saved project** in the workbench to use its
  setup. Retained fit references do not carry the fitted artifacts with them.
- After fitting in the workbench, use **Export → Rebuild this model → Download
  Python workflow** for the accepted order, setup and amendments.
- Use **Score new data → Download Python scorer** or the fitted bundle to
  apply the saved complete model without fitting again.
- Use `PricingSession` when you want to inspect, fit and choose each next change
  interactively. See [the full rating-file example](stacked_models.md).
