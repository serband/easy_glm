# EasyGLM

Build insurance pricing GLMs and rating files, check A/E and export rating
tables. EasyGLM works in a local browser workbench or from Python.

## Install and open

Requires Python 3.10–3.14. Run in a terminal:

```bash
python -m pip install --upgrade easy-glm
easy-glm-workbench
```

Your browser opens automatically. Keep the terminal open while you work.
Restart the workbench after upgrading.

Install as `easy-glm`; import in Python as `easy_glm`. The standard installation
includes the workbench, GLM fitting, CatBoost interactions and Optuna tuning.

## Build a pricing model

Open data, then choose the route in **Project & data**:

- **Single model** for one GLM, CatBoost or catalogue model.
- **Build in stages** for a sequence of models built from the same data.

Next, set the columns shared by the work: target, exposure or weight, ID, and
the train/holdout split. Mark columns that must never be used as **Ignore**.
For a staged model, name each model and choose its type before selecting any
factors. Every model has its own factor selection. EasyGLM carries the complete
prediction from earlier models into the next fit as an offset.

A typical motor sequence is:

```text
GLM main effects + reviewed interaction tables
    → Vehicle model or catalogue
    → Occupation model or catalogue
    → Postcode model or catalogue
```

Fit each model on **Model** before moving to the next one. A GLM owns its
interaction tables: fit the GLM, inspect A/E and missing-factor results, then
choose any interactions worth keeping. In **Diagnostics**, inspect the cumulative
score through the model or interaction you select, or the completed model chain.

Fitting, searches and CatBoost tuning use training rows. Keep holdout rows for
the validation check once you have a candidate worth testing.

GLMs support Poisson, Gamma, Tweedie, Gaussian, binomial and inverse Gaussian
families. CatBoost and staged models use a log link and currently support
Poisson, Tweedie and Gaussian.

## Export and keep your work

| Export | What you get |
| --- | --- |
| Project JSON | The setup: source-data location, shared columns, model definitions and factor choices. It does not contain data or fitted rates. |
| Fitted scorer | A single-model `.easyglm` file, a stacked-model JSON bundle or a Python scorer. Each scores without fitting again. |
| Python workflow | A script that fits the accepted workflow again from source data. Fresh fits can differ slightly. |
| Excel | A single-model rating-table workbook or a stacked-model component workbook. |
| Catalogue lookup file | A full CSV lookup for each fitted catalogue model. |
| HTML report | A single-model report with summaries, rating factors and diagnostics. |

For a staged model, use **Score on new data** to choose a saved fitted version,
the dataset and whether you need one rating model or the cumulative score.
Download CSV or Parquet results after checking the required columns.

Save both the project JSON and a fitted scorer. The project keeps the setup;
keep the scorer to reuse the saved rates without refitting.

## Already working in Python?

Open a pandas or Polars dataframe in the workbench:

```python skip-test
import easy_glm

easy_glm.launch_workbench(data=df)
```

For interactive Python modelling, use
[PricingSession](examples/pricing_walkthrough.md). For a reproducible settings
file, use the [Project workflow](examples/project_workflow.md). For model
chains, catalogue files and scoring new data, use the
[worked workflows](examples/workflows.md).

[Workbench walkthrough](examples/workbench_walkthrough.md) ·
[Python examples](examples/README.md) · [Changelog](CHANGELOG.md)

## Why I built this

This started with my wish to port R's [aglm](https://CRAN.R-project.org/package=aglm)
to Python and give it a GUI. It's built primarily for my own pricing work and will
probably have plenty of bugs, because… vibecode yo. Check the results before
using them.

[MIT licence](LICENSE).
