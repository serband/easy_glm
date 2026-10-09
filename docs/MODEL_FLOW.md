# Build models with different variables

Open **Model → Build a model in stages**. Choose a model from the list to work
on it. Its variables, settings and fitting controls appear together. **Show model
map** opens the connections between models when you want to see the whole flow.

All models use the project's starting data and train/holdout split. Each addition
has its own variable selection. Predictor and unassigned columns are available;
targets, weights, IDs, split columns, offsets and ignored columns are excluded.

## Fit a model

1. Choose **New standalone model** and give it a name.
2. Choose GLM or CatBoost, select its variables and set the fitting options.
3. Click **Fit model**. This saves those settings and starts fitting.

**Save draft** saves settings without fitting. Selecting a saved draft opens its
editor. If you leave an unsaved draft, choose **Keep editing**, **Save draft** or
**Discard changes**.

Search the available variables to find a column. The selected variables stay
visible and alphabetical while you search. GLMs and interaction tables use the
shared numeric bands on **Variables**. Direct CatBoost models use raw columns.

## Add a model after another model

Select a fitted model and click **Add model after this**. Choose the new addition's
variables and settings, then click **Fit model**. Its offset is the selected
model's complete fitted score, including earlier interaction tables, catalogue
files and applied amendments.

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
