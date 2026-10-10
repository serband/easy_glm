# Build a model plan

Set up the model in **Project & data**, then fit it on **Model**.

Open data and choose **Single model** or **Build in stages**. Assign the target,
weight/exposure, ID, train/test split and any ignored columns. These assignments
apply to the whole project. If the data has a split column, choose its training
and holdout values explicitly.

For a single model, choose its predictors, then define its method, family/link
and variable treatment. For models in stages, define the shared response
convention, then give each stage a name and method: GLM, CatBoost, interaction
table or catalogue lookup. Choose variables separately for each stage. The
models run in the order shown. You can add another model later; you do not need
to decide the entire sequence before fitting.

Every model uses the same starting data, response, weight/exposure and
training/holdout split. A downstream model automatically uses the complete
preceding score as its offset, including earlier accepted tables and files.
Rate-table models use log or logit links. Models in stages use a log link and
the currently supported Poisson, Tweedie or Gaussian family.

## Choose variables

Stay in **Project & data** and choose the model you want to configure. Its Table
and JSON views define only that model's variables. Search the alphabetical list,
select factors and set their treatment. Targets, weights, IDs, split columns,
offsets and ignored columns cannot become factors.

GLMs and interaction tables use numeric bands. Set a default bin count, a
count for a particular variable, or explicit cut points. Automatic cuts use
training data. Each model keeps its own definitions. CatBoost uses raw numeric
or categorical values. A catalogue model also needs its complete lookup file
and join key.

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

Open **Model**. The list shows each model's variables, fit status and preceding
offset. Set the next model's fitting options, then click its **Fit** button.
That action saves the options and starts fitting. Fit the first model before
the second, then the second before the third.

New CatBoost models and catalogues default to **Optuna tuning** on training
folds. Choose a trial count or tuning time. **Manual settings** fits the entered
iterations, depth, learning rate and regularisation once, without tuning.

The tuning-time budget counts this model's candidate CatBoost fits across folds.
Data preparation, earlier models, scoring and the final full-training fit are
outside it. A fit already running can finish before tuning stops. Holdout rows
are not used to choose settings. Results record completed trials and the chosen
settings; **No change selected** means the preceding score performed at least
as well as the completed candidates.

## Add, revise or start again

These actions are in **Project & data**:

- **Add stage** appends a model. Give it a name and method, then choose its
  variables. Choose **Build in stages** when a later model will use the current
  model's score as its offset.
- **Create revised plan** lets you change a fitted model. That model and the
  models after it need fitting again. An unchanged earlier prefix can be reused.
- **Change model structure** returns to the Single model / Build in stages
  choice. Choose a route and confirm to replace the working plan, or choose
  **Keep current plan** to return without changes. Saved fitted versions are kept.

After a fit, **Revise shared assignments** starts a copy of the plan before
unlocking its target, weight and split assignments. Applying those changes
requires fitting again from the first stage. The original recipes are kept;
their old fits do not match the revised shared assignments.

Unfitted stages can be edited, moved or removed directly. Fitted versions stay
unchanged when you revise the working plan.

## Inspect and export

On **Diagnostics**, choose the fitted version and **Fit to inspect**. The final
fit includes every model in that version. **After Vehicle model**, for example,
includes the main GLM and vehicle model, leaving out the later occupation model.
These are cumulative predictions for the original response, not isolated
correction multipliers.

Training results appear first. Choose Holdout explicitly to evaluate it. Grouped
A/E uses source variables and catalogue attributes available at the selected
stage. Inspecting an earlier stage does not refit it.

**Compare**, **Rate tables** and **Export** use the selected saved fitted version.
Table changes require preview and explicit application. Exports can contain the
complete scorer, Excel component tables or complete catalogue lookup files.

An older project keeps its saved models. Adopt a saved version into the current
plan explicitly when it has a supported log-link scorer, or start a new plan.
Adoption keeps the complete accepted
scorer as a frozen first stage. Add models after it; start a new plan to replace
that starting scorer. Project JSON contains setup and fitted
references; it does not contain the fitted model artifacts. Export the fitted
scorer to keep a deployable model.
