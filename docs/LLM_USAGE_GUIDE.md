# EasyGLM guide for an LLM helping a pricing actuary

Use this guide to help an actuary make the next modelling decision. Start with
what they are trying to learn or change. Give the shortest public call that
moves that work forward. Do not turn a modelling session into a framework
exercise.

## Where to start

Install the package as `easy-glm` and import it as `easy_glm`. Use the examples
below as complete, public workflows. When a call is not familiar, inspect the
installed signature before using it in a live pricing model.

Choose the example that matches the job:

- [Workflow map](../examples/workflows.md)
- [Interactive pricing walkthrough](../examples/pricing_walkthrough.md)
- [Stacked model workflow](../examples/stacked_models.md)
- [Model variations](../examples/model_variations.md)
- [Frequency, pure premium, severity and binary models](../examples/model_families.md)
- [Project recipe workflow](../examples/project_workflow.md)
- [Browser workbench walkthrough](../examples/workbench_walkthrough.md)

## Rules for the assisting LLM

- Establish the target, denominator or weight, offset, split and prediction
  unit before fitting.
- Work on training data for fit, screening and tuning. Show holdout only when
  the actuary asks to validate.
- Keep a fitted model unchanged. A new choice of bands, factors or tables makes
  a new fit or revision; it does not rewrite history.
- State whether a result is a candidate, a fitted model, an accepted table, or
  a validation result. Those are different things.
- Keep shared data setup separate from the factors used in each model.
- Do not invent Python methods from a workbench label. Inspect the installed
  signature when a detail matters.
- Do not silently accept a variable, interaction or table edit because a
  search suggested it.
- Use the complete accepted score for A/E and residual work. A single table is
  only one multiplier in that score.

## Installation and a quick check

```bash
python -m pip install easy-glm
```

Check which package the notebook or script will use:

```python
import importlib.metadata
import sys
import easy_glm

print(sys.executable)
print(importlib.metadata.version("easy_glm"))
print(easy_glm.__file__)
```

The standard package includes GLM fitting, CatBoost, Optuna, Excel export and
plotting. There is no separate interaction extra.

## The objects that must not be confused

| Thing | What it is | What it does not do |
|---|---|---|
| `Project` | A modelling recipe: data source, roles, split, design choices and named model recipes. | It does not load data, fit a model, or restore a fitted scorer by itself. |
| `Project.from_dict(...)` | Parses project JSON into that recipe. | It is not a fit, a reload of a saved model, or a validation of the source data. |
| `project.models[name]` | An explicit materialised model recipe that `run_model` can fit. | It is not the browser's editable named-model plan. |
| Browser model plan | An editable GUI plan for named models and their order. | It is not a documented public Python executor. Do not import desktop internals into a customer script. |
| `ModelRun` | The result of fitting one explicit recipe, with its scorer and fit evidence. | It is not a promise that later edits are included. |
| Frozen scorer / `.easyglm` / Python scorer export | A saved scoring artefact for a specific accepted fit. | It does not re-run fitting or a variable search. |
| Python workflow export | A script that prepares data and refits accepted recipes. | It may not reproduce every tuned or solver result bit-for-bit. |

A `Project` is the recipe. A fitted run or frozen scorer is the result. Keep
both when the model needs to be reproducible.

## Data, target, weight and split

Start by writing the unit of the target down. These are common patterns:

| Model | Target | Weight | Divide target by weight? | Family/link |
|---|---|---|---|---|
| Claim frequency | Claim count | Exposure | Yes | Poisson/log |
| Severity | Total incurred on claims | Claim count | Yes | Gamma/log |
| Pure premium | Total loss | Exposure | Yes | Tweedie/log |
| Propensity | 0/1 event | optional fitting weight | Usually no | Binomial/logit |

Do not divide a target that is already a rate. Do not multiply exposure into a
prediction twice. For frequency, a unit rate of 0.12 on 0.5 exposure is 0.06
expected claims.

The normal split convention is `1 = training`, `0 = holdout`:

```python
from easy_glm import add_train_test_split

data = add_train_test_split(
    data,
    train_fraction=0.70,
    seed=42,
    column="traintest",
)
```

Keep an approved existing split if there is one. For a non-numeric source flag,
make the two meanings explicit in the `Project` split settings instead of
assuming the package can infer them.

An offset is already on the link scale. With a log link, a baseline premium
amount normally becomes `log(baseline_premium)`, not the raw premium column.
Exposure weighting is not an exposure offset.

## Worked example: French motor frequency

`PricingSession` is the public interactive API. It retains the shared data,
roles, split and future-fit settings; each `PricingModel` returned by a fit is
a named checkpoint. This is a good route for an analyst working in a notebook.
The cells below use the columns in the
[French motor walkthrough](../examples/pricing_walkthrough.md). Load its data
first, then run these cells in order in the same notebook.

```python
import polars as pl
from easy_glm.pricing import PricingSession

# data is a Polars DataFrame with these source columns.
work = PricingSession(
    data,
    family="poisson",
    claims="ClaimNb",
    exposure="Exposure",
    id="IDpol",
    split="traintest",       # 1 = training, 0 = holdout
    ignored=["PolicyStart"],
)

# Numeric predictors use automatic training-data bands unless overridden.
work.bands(default=8)
work.bands("DrivAge", cuts=[25, 35, 45, 55, 65, 75])

main = work.fit_glm(
    "Main effects",
    factors=["DrivAge", "VehAge"],
)
print(main.summary())
print(main.ae("DrivAge"))
```

Fit a new checkpoint after deciding which missing factor is worth adding. Do
not overwrite `main` and then describe it as the two-factor model.

```python
print(main.find_missing_factors(["BonusMalus", "Density", "Area"]))

with_rating_factors = main.refit(
    "Main effects plus rating factors",
    add=["BonusMalus", "Area"],
)
print(with_rating_factors.ae("Area"))
```

A programmatic pair fit produces a new named checkpoint. It offsets the full
accepted parent scorer, including earlier accepted pair tables.

```python
with_pair = with_rating_factors.fit_interaction(
    "DrivAge",
    "Area",
    name="Age by area correction",
    trials=8,
    prefix_trials=4,
    min_weight_share=0.001,
)
print(with_pair.ae("DrivAge", "Area"))
```

Use `fit_catboost` when the next correction is a model rather than a pair
table. Use `fit_catalogue` to learn from attributes in a separate keyed dataset
and produce a lookup score for every key. Both append a named component after
the complete current score:

```python
correction = with_pair.fit_catboost(
    "CatBoost correction",
    factors=["VehBrand", "VehPower"],
    tune=True,
    tuning_budget="trials",
    trials=12,
    cv=5,
)

print(correction.validate_holdout(compare_with=with_pair))
```

The `trials` or time budget controls CatBoost tuning on training folds. It is
not a timeout on GLM fitting, preparation, scoring, the final refit, or
holdout validation. Set `tune=False` with explicit CatBoost parameters only
when the actuary deliberately wants a fixed, single training fit.

## Project JSON and explicit workflow recipes

Use `Project.from_dict` when project JSON is the source of a reproducible
**recipe**. Parsing it creates a `Project`
object. Preparation loads and transforms data. Validation checks columns and
settings. `run_model` fits an explicit entry from `project.models`.
The following cell assumes that `frequency_project.json` already exists.

```python
import json
from pathlib import Path

from easy_glm.workflow import Project, prepare, run_model

payload = json.loads(Path("frequency_project.json").read_text())
project = Project.from_dict(payload)       # Parse only: no source read and no fit.
prepared = prepare(project)                # Load source, apply roles/filters/split.
problems = project.validate(columns=prepared.columns)
if problems:
    raise ValueError("\n".join(problems))

run = run_model(project, prepared, "Frequency")
print(run.metrics["train"])
```

That code requires `Frequency` to be an explicit materialised model recipe in
`project.models`. It is not the public way to run an editable browser
`model_plan`. The workbench owns the current named-model plan and its saved fit
artifacts; do not copy imports from `easy_glm.desktop.*` into a customer script.

Use the Project recipe to set roles once and define each model's own factor
design. Shared roles say what a source column means. They do not automatically
put every predictor into every model.

- An ID, target, weight/exposure, offset and split are shared setup.
- Each named model selects its own factors.
- Numeric bands and categorical handling belong to that model's design.
- A column left unassigned is absent from main effects. In the workbench it can
  still be deliberately considered for a GLM pair correction where eligible.

Automatic numeric cuts are learned from **training rows**. A model-wide default
is used unless that model gives the factor an override. Explicit cuts override
the automatic count. They are not competing settings:

```python
# Illustrative project configuration; put this in the named model's design.
# Default: eight training-data bands for numeric factors in this model.
# FactorA: six automatic training-data bands for this model.
# FactorB: explicit cuts; the default does not change these cuts.
```

Do not use a global `for` loop to mutate every variable just because it makes a
demo shorter. In a pricing model, the actuary should be able to see which
factor was given which treatment.

### Factor design choices worth stating explicitly

`step` is the normal binned factor: each band has a separate relativity.
`linear` is piecewise linear: the curve is continuous and its slope can change
at named knots. `continuous` is the same idea with one slope and no interior
knots. Use linear or continuous only when a smooth, within-band response is
the modelling choice; do not substitute it merely to reduce a table's size.

A numeric linear or continuous factor can be clamped outside a stated range
and constrained to be increasing or decreasing. A monotone constraint applies
to that factor's own curve. It does not make a categorical factor or a pair
table monotone. Categorical `max_levels` retains the most common **training
weight**, not simply the most rows, then sends the rest to `Other`.

### Optional AGLM curves

In the French motor example above, these calls give driver age an AGLM curve
rather than step bands.
`n_breaks` asks for a training-data grid; the default is 100 points including
the first and last training values. This does not change the model's normal
`n_bins` setting for step bands.

```python
# Keep the fitted model unchanged. This sets the curve for the next refit.
work.numeric(
    "DrivAge",
    kind="aglm",
    breaks=[18, 25, 35, 50, 65, 80],
    extrapolation="flat",
).show()
```

`flat` holds the curve at each end of that grid. Use `extend` when the fitted
outer slope should continue outside the grid. `kind="aglm_l"` is the other
AGLM curve option. Neither AGLM option supports a monotone constraint.

The same choice in a project recipe is a `VariableDesign`. The list is the
complete grid, including both endpoints:

```python
from easy_glm.workflow import VariableDesign

project.design.variables["DrivAge"] = VariableDesign(
    kind="aglm",
    breaks=[18, 25, 35, 50, 65, 80],
    extrapolation="flat",
)
```

See [Numeric curves](../examples/numeric_curves.md) for a complete comparison
with step bands, review and export. The project still follows the same
`Project.from_dict` → `prepare` → `run_model` recipe workflow above.

## Screening and variable importance

Use a one-way screen or residual-factor search to decide what to investigate.
It is evidence, not a model change. Check its failures as well as its ranking:
an invalid target, weight, offset or sparse factor can make a candidate fail
without showing that the factor has no pricing value.

Permutation importance is different. It scores the fitted model again after
shuffling one source column at a time and reports the training-loss increase.
It does not refit the model and it is not a proof of causality. If an
importance sample is used, record its seed, row/weight coverage and whether it
was selected before looking at the results.

## Browser workbench: the short route

The workbench follows this modelling order:

1. **Project & data:** load the data and choose a single model or a staged
   model plan.
2. **Shared columns:** assign target, exposure/weight, ID, split and columns to
   ignore. Apply this shared setup before modelling.
3. **Models:** name the ordered models and choose GLM, CatBoost or Catalogue
   (a lookup built from a separate keyed dataset).
   Later models use the complete accepted score before them as their offset.
4. **Variables:** choose factors and their treatment for the selected model.
   A model's choices do not change another model's factor list.
5. **Fit:** fit the named next model. Review its training A/E and other
   diagnostics before continuing.
6. **GLM pair corrections:** search or add a pair candidate, fit it on the
   frozen current score, review it, then explicitly keep it. Keeping a pair
   creates a revised GLM checkpoint; it does not silently edit the old fit.
7. **Diagnostics and export:** choose the complete model or a named component
   scope deliberately. Export only an accepted, current fit.

A pair table belongs to its GLM. It is not a top-level CatBoost model in the
actuary's model list. Pair 2 uses main effects plus pair 1. The next named
model uses the complete accepted GLM including its pair tables.

The browser UI is the authoritative route for the current named model plan.
For a public Python API, use `PricingSession` or an explicit `Project.models`
recipe instead.

## Diagnostics and validation

A/E, residual searches and lift plots need a clear scoring scope.

- **Main effects only** means the GLM before its accepted pair tables.
- **After a pair** means the cumulative GLM score through that table.
- **Full model** means every accepted model and pair table through the selected
  endpoint.

A pair multiplier on its own is not the prediction to use for A/E. Its parent
score is part of the prediction. Label training diagnostics as in-sample.
Tuning is a training-fold exercise. Holdout is a separate request:

```python
print(correction.validate_holdout(compare_with=with_pair))
```

Use A/E to check calibration and Gini/lift to check ranking. Neither replaces
the other. Before calling a poor group result a modelling issue, inspect its
exposure and claim count.

## Catalogue models

A catalogue model learns from training rows using attributes joined by key
from a separate dataset. It then scores every key in that dataset and stores
the results as a lookup. The model chain keeps this as one named component.

The score from a catalogue component is usually **incremental**: it multiplies the
complete score entering the component. Use the full stack scorer for an overall
prediction or A/E. Do not mistake one component's multiplier for the full
score.

A first component can instead be a complete scorer when it is deliberately the
starting model. State which one it is. Pin the auxiliary dataset and key used
for a fitted model. If the live source file changes later, that is a new model
input, not a reason to rewrite an old fitted checkpoint.

## Changes after a model is fitted

Changing bands, factors, a rate table or an earlier pair changes the model
history.

- A main-effect or pair-table amendment is a revision of that model's
  checkpoint.
- Earlier edits can make later pairs and later named models need refitting.
- Refit the affected suffix when the later corrections should learn from the
  amended score.
- Keep a later correction fixed only when that is the deliberate pricing
  decision, and label it as fixed relative to the amended upstream score.
- Do not report a full model as current while its suffix says it needs refit.

Preview a table edit before applying it. Save both the accepted result and the
reason for the adjustment.

For the French motor GLM above, address the row you want to change explicitly:

```python
review = main.edit_rates("Driver-age review")
review.set_relativity("DrivAge", lower=25, upper=35, value=1.05)
review.rebalance()             # Restore the fitted training total after the edit.
review.preview().show()
revised_main = review.apply(refit_later_interactions=True)
```

`rebalance()` changes the base level to restore the training expected total
after an edit. It does not prove the amended shape is appropriate. Smooth a
table only after inspecting support and before/after A/E; preserve the preview
and the reason for the change. Smoothing, capping and rounding are pricing
decisions, not automatic repair tools.

## Score additional datasets

Use a frozen `PricingModel` to score new data. No target or training split is
needed. `score` retains the input rows and columns, applies the saved column
transformations, and skips training filters and split creation.

```python
result = approved.score(renewals, exposure_col="Exposure", column_prefix="renewal_")
result.preview()
result.export("renewal_scores.parquet")
```

Omit `exposure_col` for unit predictions. `scope="through", model="Component"`
scores cumulatively through that named component. `scope="model",
model="Component"` returns the accepted component contribution only. A named
GLM in a logical workbench plan includes its accepted owned interaction tables;
legacy flat components stay separate. `scope="catalogue", model="Component"`
scores a refreshed keyed dataset with its retained native fit as a candidate,
excluding accepted lookup amendments. Older lookup-only bundles cannot produce
native candidates.

In the GUI, add files under **Project & data → Data → Additional scoring
datasets**, then open **Score on new data**. Choose the dataset and the saved
model to score. Choose **Overall prediction** for the complete saved model, or
**One rating model** for a named model's contribution. The summary
names the dataset, model and output; required columns are shown before checking
values and keys. Earlier cumulative scores are under **Advanced cumulative
checkpoints**. Native refreshed-dataset candidates are separate from the
selected accepted lookup. Preview and download the scored file with its details
JSON. `launch(data=experience, scoring_data={"Additional data": scoring_data})`
registers a Python frame separately from the modelling source. Local-file
references can persist in the project; uploaded files and Python frames are
session-only.

The [complete scoring example](../examples/new_data_scoring.md) shows the
end-to-end workflow on additional datasets.

## Export: workflow versus scorer

Choose the export based on what must be reproduced.

| Need | Export | Meaning |
|---|---|---|
| Score exactly like a specific approved fit | Frozen scorer / `.easyglm` / Python scorer export | Uses saved tables and accepted components. It does not fit. |
| Re-run the accepted modelling recipe on data | Python workflow export | Prepares data and refits the accepted recipe. Solver and tuning results may vary. |
| Review implementation tables | Excel export | Writes the tables and named components for review or hand-off. |

A workflow export needs an accessible source path or data-loading block. Treat
that path as something to edit for another machine. A frozen scorer is the
right artefact for reproducing the exact saved rate tables.

## Command line and scale

For an explicit project recipe, the released command-line route is:

```bash
easy-glm validate project.json
easy-glm run project.json --model Frequency --out outputs
easy-glm export project.json --model Frequency --out outputs --script
```

`validate` checks the recipe and data preparation; it does not fit a model.
`run` and workflow `export` fit again. Use an accepted frozen scorer where
exact saved predictions matter.

Start with a readable model. More numeric bands and a larger pair grid create
more parameters and less support per cell. A wide variable screen is many
candidate fits, not one fast global operation. The package scores fitted rate
tables without rebuilding a design matrix, but fitting and tuning still need a
realistic memory and time budget.

## Common errors to stop before they spread

| Symptom | Check first |
|---|---|
| A prediction total looks wrong | Is the target already a rate? Was it divided by weight? Was exposure multiplied exactly once? |
| An expected factor is absent | Is it shared as a predictor but omitted from this model's factor list? Was an unapplied draft used? |
| A pair is unavailable | Are the family and link multiplicative? Is there an accepted GLM checkpoint to own it? Are both columns eligible? |
| A tuning job runs too long | Reduce CatBoost trial count or its tuning-time budget; do not describe it as a GLM timeout. |
| A later model is stale | An upstream factor, table, or pair changed. Refit the suffix or make the fixed-correction decision explicit. |
| A generated Python script differs from a saved scorer | One refits a workflow; the other scores frozen tables. Compare like with like. |
| A JSON file loaded successfully but fitting fails | `Project.from_dict` parsed a recipe. Run `prepare`, validate prepared columns, then fit an explicit model. |
| A feature search changed the model | A search should only produce evidence. Check whether somebody explicitly applied its recommendation. |

## Continue a modelling session with another LLM

Provide this alongside the project JSON and fitted artefacts:

```text
Objective: frequency / severity / pure premium / probability
Data: source name, row grain, snapshot date, key
Target and unit: ...
Weight or exposure and target-division choice: ...
Offset: ...
Split: source column and which values are train and holdout
Current accepted chain: named models in order, with accepted pair tables
Current checkpoint: ...
Unfinished work: candidate, amendment, or suffix needing refit
Diagnostics reviewed: training only / holdout requested on ...
Exports available: project recipe / frozen scorer / workflow script / Excel
Next decision: ...
```

Do not ask the next LLM to infer the current scorer from a project recipe. Give
it the saved fit or frozen scorer as well when exact scoring matters.

## Reference checks

Before using a version-specific call, check the installed signature and the
relevant example. For the workbench flow, see [Model flow](MODEL_FLOW.md).
