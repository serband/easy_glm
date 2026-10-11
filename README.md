# EasyGLM

EasyGLM is a python package that makes fitting regularised generalised linear models (GLMs) for use in general insurance pricing a cinch. It produces factor relativity tables to inspect, amend and use in a way that naturally fit into standard pricing workflows. It can be used both programmatically within a python session or in the local browser using the GUI workbench.

## Install

Requires Python 3.10–3.14. The installation includes the local workbench and
CatBoost.

```bash
python -m pip install --upgrade easy-glm
```

Install as `easy-glm`; import as `easy_glm`.

## Use the workbench

```bash
easy-glm-workbench
```

The workbench opens in your browser and runs locally. Keep the terminal open
while you work.

In **Project & data**, load your data, choose a single model or a sequence of
models, and assign the target and any exposure, weight, ID or train/test columns.
Choose the model family and predictors. For GLM numeric factors, use automatic
bands, set your own cuts or choose a curve, then fit the model.

Inspect the relativities and A/E, search for missing factors or interactions,
and compare candidate models. You can amend rates, preview the effect of a
change before applying it, and export the tables to Excel.

The [workbench walkthrough](examples/workbench_walkthrough.md) shows this
process on screen.

## Use it from Python

`PricingSession` keeps the data, target, exposure, split and band settings
together while you build models. Each fit returns a named model you can inspect,
compare or build on.

This example fits driver age and vehicle age to the public French motor
claim-frequency data. EasyGLM downloads the data and caches it for later use.

```python
from easy_glm import PricingSession, load_external_dataframe

data = load_external_dataframe().sample(n=10_000, seed=42)

study = PricingSession(
    data,
    family="poisson",
    claims="ClaimNb",
    exposure="Exposure",
    id="IDpol",
)
model = study.fit_glm("Age model", factors=["DrivAge", "VehAge"])

model.summary()
model.relativities("DrivAge").show()
model.ae("DrivAge").show()
```

The session creates a 70/30 train/holdout split, keeping rows with the same ID
together. Fitting and searches use training data; the A/E above is also on
training data. Earlier named fits remain available when you change bands or
fit another model.

Export the tables and save the fitted model:

```python
model.to_excel("age_model.xlsx")
model.save("age_model.easyglm")
```

To open the same data in the workbench:

```python skip-test
import easy_glm

easy_glm.launch_workbench(data=data)
```

The [pricing walkthrough](examples/pricing_walkthrough.md) continues with
custom bands, missing-factor searches, interactions, rate amendments and
holdout validation.

## Add interactions or further models

Add and review interaction tables after fitting the main effects. Each new
interaction uses the GLM and earlier tables as its offset, keeping their rates
fixed.

You can also fit CatBoost on its own or combine GLMs and CatBoost in a sequence.
Each model has its own predictors and uses the complete prediction from all
earlier models as its offset. Sequential models support Poisson, Tweedie and
Gaussian with a log link.

See the [model workflow examples](examples/model_variations.md) for different
orders and tuning settings. The [model family examples](examples/model_families.md)
cover severity, pure premium and claim occurrence.

## Save and score

A saved fitted model reuses the rates without refitting. In the workbench you
can also export the project settings and a Python script that repeats the fits.

In **Score on new data**, select the new data and the saved model or sequence
to apply. Check the required columns, score the data, and download the results
as CSV or Parquet. The [scoring example](examples/new_data_scoring.md) shows
this in Python and the workbench.

[All examples and workflows](examples/workflows.md) · [Changelog](CHANGELOG.md)

## Why I built this

This started with my wish to port R's [aglm](https://CRAN.R-project.org/package=aglm)
to Python and give it a GUI. It's built primarily for my own pricing work and will
be buggy, because… vibecode yo. Check the results before
using them.

[MIT licence](LICENSE).
