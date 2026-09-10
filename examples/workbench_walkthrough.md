# Workbench walkthrough

## 1. Open your data

Run `easy-glm-workbench`. On **Project & data**, enter a CSV, Parquet or Excel
file path, or choose **Saved project** to reopen exported project JSON. Paths
refer to files on the computer running EasyGLM.

You can also supply a pandas or Polars dataframe directly:

```python
import easy_glm

df = easy_glm.load_swedish_motorcycle_data()
df = df.filter(df["Exposure"] > 0)
easy_glm.launch_workbench(data=df)
```

The launcher takes a Parquet snapshot of the dataframe. Changes in the browser
do not change your Python variable or source file. Keep the terminal open.

## 2. Assign roles and split the data

On **Variables**, use the table or JSON editor to assign the target, weight and
predictors. For the Swedish example, use `ClaimAmount` as target, `Exposure`
as weight, and `OwnerAge`, `Gender`, `Area`, `RiskClass`, `VehAge` and `BonusClass`
as predictors. Ignore `ClaimNb`: known claim counts would leak information.

Preview and apply your variable settings. On **Model**, configure a reproducible
training/holdout split, such as 70% training with seed 42. If you already have a
split column, specify which value means training.

## 3. Define and fit a model

Create a model, select its family, target and weight, and choose its predictors.
For annual incurred cost, use Tweedie with power 1.5 and **Divide target by weight**.
For claim frequency, use Poisson with claim count and exposure instead.

**Factor design** controls the main effects and optional interactions. Interactions
fit in a second stage, leaving the fitted main effects fixed. Review the settings,
apply them, then select **Fit model**. Fitting runs in the background.

![Model design](../docs/images/workbench-model-design.png)

## 4. Check diagnostics

Inspect training and holdout actual-versus-expected curves. Exposure bars share
the chart on a secondary axis. Review lift, Gini and the regularisation path.
Permutation importance ranks the performance loss from shuffling each training
predictor. Residual-factor searches help identify omitted variables or interactions.
Check suggested changes on holdout data before keeping them.

![Diagnostics](../docs/images/workbench-diagnostics.png)

## 5. Compare alternatives

Fit a second model and open **Compare**. Choose the two models to compare their
metrics, double lift and rate-table differences. Designate a champion explicitly.

## 6. Adjust rate tables

On **Rate tables**, the original fit stays visible. Choose an adjustment from
the dropdown, change its options and inspect the preview. **Apply adjustment**
saves it. Moving average uses the last N points; isotonic smoothing enforces an
increasing or decreasing shape. Caps/floors and manual row edits are also available.

Check actual versus expected underneath. Undo restores the prior applied state;
named snapshots keep adjustment versions. Rebalancing changes the base rate to
restore the original fitted training total.

## 7. Export your work

**Export** offers Excel rate tables, a `.easyglm` JSON scorer, project JSON,
a Python reproduction script and an HTML report. Exports use applied settings;
unsaved previews are excluded.

Export the project before stopping the server. It preserves data locations,
roles, split, model definitions, adjustments and snapshots. Reopen it and refit
to continue; fitted runs and session Undo are not included. Use the scorer to
keep the fitted rates. A Python reproduction script refits from the source data.

The previous Streamlit interface remains available with
`easy-glm-workbench --legacy-streamlit`, including its additional preparation
editors and fitted-session persistence.
