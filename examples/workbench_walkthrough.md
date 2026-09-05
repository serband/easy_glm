# Workbench walkthrough

This walkthrough uses the public Swedish motorcycle portfolio to fit annual
incurred claim cost with a Tweedie GLM. The same screens are used for a Poisson,
Gamma, Gaussian or binomial model.

The workbench is a browser interface over the same EasyGLM workflow available
from Python. It records the choices you make and can export them as a Python
script; it does not fit or change a model until you ask it to.

## 1. Start with the data

The shortest reproducible route is to load the data in Python, clean anything
that must be dealt with before modelling, then pass the frame to the workbench:

```python
import polars as pl

import easy_glm

df = easy_glm.load_swedish_motorcycle_data()
df = df.filter(pl.col("Exposure") > 0)

easy_glm.launch_workbench(data=df)
```

A pandas dataframe is also accepted. EasyGLM converts the supplied frame to
Polars and writes a temporary Parquet snapshot because the browser workbench
runs in a separate process; it is not a live reference to the Python variable.

Alternatively, start `easy-glm-workbench` and choose a local data file on
**Project & data**. A path points to a file on the machine running the
workbench. Uploading from the browser first stores a local copy of that file.
After trying a built-in sample, use **Start over and choose another sample** on
the same page. The workbench warns before discarding an unsaved setup, then
returns to the two sample choices.

## 2. Assign the column roles

On **Variables**, use:

- `ClaimAmount` as the target;
- `Exposure` as the weight;
- `OwnerAge`, `Gender`, `Area`, `RiskClass`, `VehAge` and `BonusClass` as
  predictors; and
- `ClaimNb` as ignored for this model, because a known claim count would leak
  information into the incurred-cost prediction.

Review the detected data types and any renames or recodes before continuing.
Ignored columns are also excluded from the residual-factor search.

## 3. Create the split

On **Split**, choose a random 70/30 training and holdout split with seed 42.
The training rows are used to fit and select the model. Holdout rows are kept
out of those decisions and are used to check how the finished model behaves on
unseen data.

If the data already contains a split column, select that column and explicitly
identify the value which means “training”.

## 4. Define the model

On **Model**, create a model named `BurnCost` and choose:

- family: `tweedie`;
- target: `ClaimAmount`;
- weight: `Exposure`;
- **Divide target by weight**: selected; and
- Tweedie power: `1.5`.

This fits annual incurred claim cost. The power is fixed for this example;
automatic Tweedie-power selection is future work.

![Tweedie model definition in the workbench](../docs/images/workbench-model-design.png)

The factor-design section shows how every numeric predictor is banded. Leaving
the number of bins at zero uses the default quantile design. Entering another
number recalculates the suggested knots. Applying a comma-separated knot list
switches that factor to an explicit custom design.

Interactions are optional. The preview shows the exposure in each candidate
cell, but an interaction is not part of the model until **Add interaction** has
been selected. Sparse cells below the chosen exposure threshold receive no
separate adjustment.

## 5. Fit it

Keep cross-validation selected, use five folds, and select **Fit model**.
EasyGLM shuffles those folds reproducibly using the project’s split seed. If the
model contains interactions, their second-stage validation uses out-of-fold
main-model predictions rather than fitted values from the same rows.

The model page then shows the selected penalty, retained coefficients,
training and holdout metrics, and the regularisation path.

## 6. Check the diagnostics

On **Diagnostics**:

1. Compare training and holdout A/E by each fitted factor. A pattern which is
   present only in training is a warning sign; a pattern in both may indicate a
   design that is too coarse.
2. Review lift and Gini as ranking diagnostics, not calibration measures.
3. Use residual-factor search to find omitted structure. The search uses
   training rows only and excludes fields marked ignored; validate anything it
   suggests on holdout data.
4. Fit a second model before using the challenger overlay or double-lift chart.
   With no incumbent model, double lift uses the null model as its benchmark.

![Training and holdout diagnostics in the workbench](../docs/images/workbench-diagnostics.png)

## 7. Compare and adjust

Create a second model when you want to test a different design, penalty or
interaction. **Compare** shows the performance and rate-table differences
between the selected model and its challenger.

On **Rate tables**, inspect the fitted relativities and exposure behind every
band. Smoothing, caps and manual edits create an adjusted prediction without
refitting the original model, so the actual, fitted and adjusted lines remain
separate. Check both training and holdout views after an adjustment.

## 8. Export or resume later

Use **Export** to download:

- the complete workflow as Python;
- a portable EasyGLM scorer;
- Excel rate tables; or
- a self-contained HTML model report.

Saving the project preserves data locations, roles, preparation, split, model
definitions and rate-table adjustments. Fitted runs are stored beside it and
restored only while the data and setup still match.

For the equivalent Python fit, see [basic_usage.py](basic_usage.py). To create
a ready-to-open workbench project in code, see
[easy_glm_demo.py](easy_glm_demo.py).
