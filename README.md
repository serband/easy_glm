# EasyGLM

Fit insurance pricing GLMs, check model performance and export rating tables
from a local browser workbench.

## Install and open

Requires Python 3.10–3.14. Run in a terminal:

```bash
pip install --upgrade easy_glm
easy-glm-workbench
```

Your browser opens automatically. Keep the terminal open while you work.
Restart the workbench after upgrading.

## Build a model

1. **Load data.** Open your file or try the French motor claim-frequency or
   Swedish motorcycle claim-cost example. Supports CSV, Parquet, Excel,
   Arrow/Feather and SAS files.
2. **Choose variables.** Set the target, exposure/weight and predictors. Check
   for missing data, possible leakage and redundant predictors before fitting.
3. **Explore.** See observed rates and exposure by variable.
4. **Fit.** Choose a model family, training/holdout split, factor shapes and
   interactions, then click **Fit model**.
5. **Review.** Check actual versus expected, lift, Gini and variable importance.
   Compare alternative models.
6. **Adjust.** Smooth, cap/floor or edit relativities, then apply your changes.
   The original fit stays available for comparison.

Supports Poisson, Gamma, Tweedie, Gaussian, binomial and inverse Gaussian GLMs,
with regularisation and two-stage interactions.

![Model design in EasyGLM](docs/images/workbench-model-design.png)

## Export and keep your work

| Export | What you get |
| --- | --- |
| Excel | Applied rating tables |
| Scorer (`.easyglm`) | Applied rates for scoring new data |
| Project JSON | Model setup, applied adjustments and named snapshots |
| HTML report | Data summaries, importance, coefficient paths, rating factors and diagnostics |
| Python script | Reproduce the model from its source data |

Before closing, save the **project JSON** and **scorer**. Reopening a project
requires its source data and a refit; fitted runs and session Undo are not saved
in project JSON.

## Already working in Python?

Open a pandas or Polars dataframe in the workbench:

```python skip-test
import easy_glm

easy_glm.launch_workbench(data=df)
```

[Workbench walkthrough](examples/workbench_walkthrough.md) ·
[Python examples](examples/README.md) · [Changelog](CHANGELOG.md)

Experimental software: validate results before using them for pricing.
[MIT licence](LICENSE).
