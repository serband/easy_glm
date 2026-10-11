# Build a model plan

For worked steps, start with the [Workbench walkthrough](../examples/workbench_walkthrough.md).
The [workflow map](../examples/workflows.md) links the GUI, interactive API,
rating-file examples and [Project recipe guide](../examples/project_workflow.md).

Set up the model in **Project & data**, then fit it on **Model**.

Use the setup navigation in **Project & data** to work on one decision at a time.
Open data and choose **Single model** or **Build in stages**. Assign the target,
weight/exposure, ID, train/test split and any ignored columns. These assignments
apply to the whole project. If the data has a split column, choose its training
and holdout values explicitly.

For a single model, choose its predictors, then define its method, family/link
and variable treatment. For models in stages, define the shared response
convention, then set **Model name** and **Model type** in each numbered card.
Types are GLM, CatBoost or catalogue lookup. Each card shows
the complete preceding score it will use as its offset. Continue to **Variables**
to choose factors separately for each model. The models run in the order shown.
You can add another model later; you do not need
to decide the entire sequence before fitting.

Every model uses the same starting data, response, weight/exposure and
training/holdout split. A downstream model automatically uses the complete
preceding score as its offset, including earlier accepted tables and files.
Rate-table models use log or logit links. Models in stages use a log link and
the currently supported Poisson, Tweedie or Gaussian family.

## Choose variables

Open the **Variables** setup step and choose the model you want to configure.
Its position, name, method and incoming offset identify the model being edited. Its Table
and JSON views define only that model's variables. Search the alphabetical list,
select factors and set their treatment. Targets, weights, IDs, split columns,
offsets and ignored columns cannot become factors.

GLMs use numeric bands. Set a default bin count, a
count for a particular variable, or explicit cut points. Automatic cuts use
training data. Each model keeps its own definitions. CatBoost uses raw numeric
or categorical values. A catalogue model also needs its complete lookup file
and join key.

Save variables for this model before moving to the next model. A single model
starts with factor selection; its method and variable treatment follow under
**Model details**. The JSON editor changes only the named model, while **Shared
column roles JSON** changes project assignments.

For example, this JSON selects three variables and gives driver age fixed cuts:

```json
{
  "defaults": {"n_bins": 8},
  "variables": {
    "DrivAge": {"kind": "step", "knots": [25, 35, 45, 55, 65]},
    "VehAge": {"kind": "step", "n_bins": 6},
    "Region": {"kind": "categorical"}
  }
}
```

Only columns under `variables` are selected. This does not change shared column
assignments or another model's variables. A column can be selected in several
stages. Shared target, weight and split settings are inherited downstream; they
cannot be changed separately in a later stage.

## Fit in order

Continue to **Model**. The list shows each model's variables, fit status and
preceding offset. Set the next model's fitting options, then click **Fit Model
1 — Main effects**, for example. That action saves the options and starts
fitting. **Save fitting settings** keeps the options without running a fit.
Unsaved options are labelled and leaving the page asks before discarding them.
Fit the first model before the second, then the second before the third. A blank
future model does not prevent fitting the configured first model. A model with
missing variables takes you back to its Variables editor.

New CatBoost models and catalogues default to **Optuna tuning** on training
folds. Choose a trial count or tuning time. **Manual settings** fits the entered
iterations, depth, learning rate and regularisation once, without tuning.

The tuning-time budget counts this model's candidate CatBoost fits across folds.
Data preparation, earlier models, scoring and the final full-training fit are
outside it. A fit already running can finish before tuning stops. Holdout rows
are not used to choose settings. Results record completed trials and the chosen
settings; **No change selected** means the preceding score performed at least
as well as the completed candidates.

## Add interactions inside a GLM

A planned GLM can contain its main effects and several accepted interaction
tables. You do not need to declare those tables in Project & data. Fit the main
effects first, then investigate residuals and add tables inside that named GLM.
The next named model uses the GLM and all of its accepted tables as its offset.

On **Model**, the review sits directly beneath its fitted GLM and names that
model. Use **Manage interactions** on another GLM to open its review there.
Its **Interactions** section lists accepted tables in order. Use **Find
interactions** to investigate the current accepted training score, or **Add
interaction** to choose a pair directly. Either column
may be absent from the main effects; choosing it here does not add a main effect
or change its shared role.

Fit a candidate, inspect its scoring table and comparison, then choose **Keep
interaction** or **Discard candidate**. A candidate is separate from the accepted
score. Fitting or discarding one does not change downstream models. Its saved
source must still match before it can be kept.

The second table offsets the GLM and the first accepted table, including any
accepted amendments. Earlier main effects and tables remain fixed. The deployed
tables supply the offset; the earlier raw CatBoost learner does not.

Interactions are optional. Continue to the next named model as soon as the GLM
is fitted. If you return later and change its accepted score, the revised working
plan marks affected later tables and models as needing a new fit. The old complete
fitted chain remains available.

## Add, revise or start again

These actions are in **Project & data**:

- **Add model** in **Models** appends a numbered card with GLM selected. Set its
  name and type, then open **Variables** and select that model. Choose **Build
  in stages** when a later model will use the current model's score as its offset.
- **Create revised plan** lets you change a fitted model. That model and the
  models after it need fitting again. An unchanged earlier prefix can be reused.
- **Change model structure** returns to the Single model / Build in stages
  choice. Choose a route and confirm to replace the working plan, or choose
  **Keep current plan** to return without changes. Saved fitted versions are kept.

After a fit, **Revise shared assignments** starts a copy of the plan before
unlocking its target, weight and split assignments. Applying those changes
requires fitting again from the first stage. The original recipes are kept;
their old fits do not match the revised shared assignments.

Unfitted models can be edited, moved or removed directly. Fitted versions stay
unchanged when you revise the working plan. Saved model names and types are
shown consistently in setup, Variables and Model. Failed structure saves remain
visible for correction or retry.

Predictor screening and one-way feature selection remain optional tools on
**Explore**; they are separate from required setup.

## Inspect and export

On **Diagnostics**, choose the fitted version and **Fit to inspect**. The final
fit includes every model in that version. **After Vehicle model**, for example,
includes the main GLM, its accepted interactions and vehicle model, leaving out
the later occupation model. Within a GLM, inspect the cumulative score through its
main effects or an accepted interaction. These checkpoints include any incoming
earlier models.
These are cumulative predictions for the original response, not isolated
correction multipliers.

Training results appear first. Choose Holdout explicitly to evaluate it. Grouped
A/E uses source variables and catalogue attributes available at the selected
stage. Inspecting an earlier stage does not refit it.

**Compare**, **Rate tables** and **Export** use the selected saved fitted version.
Table changes require preview and explicit application. Exports can contain the
complete scorer, Excel component tables or complete catalogue lookup files.

On **Export**, **Download Python workflow** rebuilds the selected model chain
from its source data. It includes accepted interaction tables, amendments and
the offset order. Change `SOURCE_PATH` in the script if the data file has moved.
The script keeps the saved training split and writes a new fitted scorer and
Excel workbook. Refitting can change the fitted values. **Download Python
scorer** uses the saved fits to score new rows without refitting.
Complete-score export uses the selected accepted fitted version, not an unfinished
planned suffix. Incremental table and file exports identify their required
preceding score.

An older project keeps its saved models. Adopt a saved version into the current
plan explicitly when it has a supported log-link scorer, or start a new plan.
Adoption keeps the complete accepted
scorer as a frozen first stage. Add models after it; start a new plan to replace
that starting scorer. Project JSON contains setup and fitted
references; it does not contain the fitted model artifacts. Export the fitted
scorer to keep a deployable model.

Older flat plans retain their interaction stages and scoring without automatic
regrouping. New plans use the named-model structure above.
