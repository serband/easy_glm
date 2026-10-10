# Build models with different variables

Open **Model → Build a model in stages**. Choose a model from the list to work
on it. Its variables, settings and fitting controls appear together. **Show model
map** opens the connections between models when you want to see the whole flow.

All models use the project's starting data and train/holdout split. Each addition
has its own variable selection. Predictor and unassigned columns are available;
targets, weights, IDs, split columns, offsets and ignored columns are excluded.

## Fit a model

1. Choose **Start a new model chain** and give the first model a name.
2. Choose GLM or CatBoost, select its variables and set the fitting options.
3. Click **Fit model**. This saves those settings and starts fitting.

**Save draft** saves settings without fitting. Selecting a saved draft opens its
editor. If you leave an unsaved draft, choose **Keep editing**, **Save draft** or
**Discard changes**.

Each draft has a **Variables for this model** section. Use **Table** to search
columns, select them, and change how this model uses them. The list is alphabetical.
Use **JSON** to select and define many variables at once, then choose **Apply
variable JSON**. Fix invalid entries before saving or fitting.

GLMs and interaction tables have their own numeric bands: a default count,
a count for one variable, or explicit cuts. Automatic cuts use training data.
Project settings provide the initial defaults; saving the model stores its own
copy. Later changes to project defaults do not change that saved definition.
Direct CatBoost models use raw numeric or categorical values, without GLM bands.

For example, this JSON selects three variables for this model and gives driver
age fixed cuts:

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

Only columns listed under `variables` are selected. This JSON does not change
project roles or the variable selections in other models. A project-wide ignored
column remains unavailable in every model.

## Add a model after another model

Select a fitted model and click **Add model after this**. Choose the new addition's
variables and settings, then click **Fit model**. Its offset is the selected
model's complete fitted score, including earlier interaction tables, catalogue
files and applied amendments.

The target, weight/exposure, family, log link and training/holdout split are
inherited from the first model and shown read-only. The model offset is automatic;
there is no column to choose or prediction array to supply. Any original external
offset is already included in the complete preceding score.

For example, fit a main GLM, add a vehicle CatBoost model after it, then add an
occupation model after the vehicle model. The occupation model uses both earlier
parts as its offset. Their variables remain separate and their fitted values stay
fixed.

## Revise a fitted model

Fitted settings are read-only. Choose **Revise model** to copy the last addition
into a new named draft. A child revision uses the same exact fitted parent; a
standalone revision starts on its own. Change its variables or settings and fit
the new version. The original model and its existing children stay unchanged.

To change an earlier part of the sequence, select that earlier named model and
revise it there. Inherited components cannot be edited inside a child.

## Inspect and export

A fitted model opens on **Results**. **Settings** shows its saved recipe.
**Inspect actual versus expected**, **Compare**, **Inspect and amend tables** and
**Export** use that selected complete model. Training results are labelled
in-sample; holdout evaluation appears only when you request it. Table changes
still require a preview and explicit application as a new named model.

On **Diagnostics**, choose **Saved chain (final model)**, then **Fit to inspect**:

- **Final fit — after Occupation model** uses every model in that saved chain.
- **After Vehicle model** uses the GLM and vehicle model, leaving out the later
  occupation model.
- **After Main GLM** uses the GLM alone.

The page lists the models included in the selected fit. A downstream model's
A/E includes its earlier models as offsets. Choose **Group by** to inspect A/E
by a variable; the choices include unmodelled source variables and catalogue
attributes available at that point. This inspection uses the saved fits and
does not refit them. Training rows appear first; choose Holdout to evaluate it.

An unavailable original parent does not stop you inspecting or extending a child
whose own complete fit is available. It does prevent revising that child's last
addition against the missing original fit, even if a newer fit has the same
parent model name.

**Needs a new fit** means an unfitted saved setup refers to an earlier starting
fit that is no longer available. **Cannot score** means the selected model's
own fitted files are unavailable. In either case, inspect the saved recipe and
choose **Fit saved recipe as new model** to fit every saved component from the
beginning under a new name. The original saved model stays unchanged.

Project JSON stores model settings and connections; it does not contain fitted
model artifacts. Use the fitted scorer export to preserve a deployable model.
The existing **Single model** workflow remains available.
