# Build a model plan

Use **Project & data** to choose **Single model** or **Build in stages**.
Give each model a name and choose its method: GLM, CatBoost, interaction table
or catalogue lookup. The models run in the order shown. You can add another
model later; you do not need to decide the entire sequence before fitting.

Every model uses the same starting data, response, weight/exposure and
training/holdout split. A downstream model automatically uses the complete
preceding score as its offset, including earlier accepted tables and files.
Rate-table models use log or logit links. Models in stages use a log link and
the currently supported Poisson, Tweedie or Gaussian family.

## Choose variables

Open **Variables** and choose the model you want to configure. Its Table and
JSON views define only that model's variables. Search the alphabetical list,
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

Only columns under `variables` are selected. This does not change project roles
or another model's variables.

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
  variables. Adding to a single model turns the plan into models in stages.
- **Create revised plan** lets you change a fitted model. That model and the
  models after it need fitting again. An unchanged earlier prefix can be reused.
- **Start a new plan** clears the current setup. Saved fitted versions remain
  available separately; starting again does not delete them.

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
