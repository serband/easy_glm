# Numeric curves

Fit driver age as bands, then as a continuous curve. We will compare the rates,
choose the ages where the curve can change direction, add a driver-age/vehicle-age
interaction, and amend and export the rates.

## Set up

`PricingSession` keeps the data and modelling settings together. Here we use
8,000 rows of French motor data, with claim counts in `ClaimNb` and exposure in
`Exposure`. We reserve 30% for holdout and use the rest to build the model.

Run the blocks in order in the same notebook or session.

```python
from easy_glm import PricingSession, load_external_dataframe

data = load_external_dataframe().sample(n=8_000, seed=42)
work = PricingSession(
    data,
    family="poisson",
    claims="ClaimNb",
    exposure="Exposure",
    id="IDpol",
    train_fraction=0.70,
    seed=42,
)
```

## Fit driver-age bands

Start with a two-factor frequency model: driver age and vehicle age. The
driver-age cuts below give ages 25–34 the same relativity, ages 35–49 another,
and so on. Vehicle age gets six bands chosen from the training data.

```python
work.bands("DrivAge", cuts=[25, 35, 50, 65])
work.bands("VehAge", number=6)
banded = work.fit_glm("Driver-age bands", factors=["DrivAge", "VehAge"], alpha=0.001)
banded.relativities("DrivAge").show()
banded.ae("DrivAge").show()
```

Look at the driver-age rates alongside A/E. A gap between actual and expected
frequency may suggest that a band is too wide or that another factor is missing.

`alpha` controls how strongly the fit pulls effects towards zero. We use
`alpha=0.001` here; omit it to let training cross-validation choose the penalty.

## Try a smooth driver-age curve

A curve lets the driver-age rate vary within a band. With `kind="aglm"`, the
log-relativity follows a continuous curve that can change slope at chosen age
points. The fitted rates therefore have no jumps at those points.

Ask for six age points with `n_breaks=6`. EasyGLM spaces them by quantiles of the
training ages, including the youngest and oldest. Repeated ages can give fewer
points. The default is 100 points if you leave out `n_breaks`.

```python
work.numeric("DrivAge", kind="aglm", n_breaks=6, extrapolation="flat").show()
automatic = banded.refit("Automatic driver-age curve")
automatic.relativities("DrivAge").show()
automatic.ae("DrivAge").show()
automatic.compare(banded)
```

Compare the age curve and A/E with the banded fit. These charts and comparisons
use training data. Look at where the curve closes a gap between actual and
expected, and whether the fitted age pattern is suitable for your rates.

`extrapolation="flat"` keeps the rate constant outside the first and last age
points. `"extend"` continues the fitted trend at each end. This determines, for
example, the rate for an age older than any age used to fit the curve.

## Set an explicit grid

To choose the ages yourself, pass `breaks`. Below, the curve can change slope
at 25, 35 and 50, between endpoints 18 and 80. Five points give four intervals.
These explicit ages replace the automatic points.

```python
work.numeric(
    "DrivAge",
    kind="aglm",
    breaks=[18, 25, 35, 50, 80],
    extrapolation="extend",
).show()
fixed = automatic.refit("Fixed driver-age curve")
fixed.relativities("DrivAge").show()
fixed.ae("DrivAge").show()
fixed.compare(automatic)
```

Look for an age pattern you can use in the rating structure, as well as a good
fit. `fixed` keeps these ages and its fitted rates; the earlier models remain
available for comparison.

## Add an interaction to the fixed curve

Suppose the remaining driver-age pattern varies with vehicle age. Fit the pair
on top of `fixed`. CatBoost uses that model's predictions as its offset, then
EasyGLM converts the correction into a two-factor rating table. The GLM rates
stay fixed.

The table uses our driver-age intervals and vehicle-age bands. A small grid
makes the cells easier to review. In the workbench you can choose coarser bands
for the interaction table while keeping the GLM curve.

```python
interaction = fixed.fit_interaction(
    "DrivAge", "VehAge", name="Driver and vehicle age", trials=1, prefix_trials=1
)
interaction.relativities("DrivAge", "VehAge").show()
interaction.ae("DrivAge", "VehAge").show()
interaction.compare(fixed)
```

Inspect the interaction rates and A/E by cell. Check the exposure behind a
large correction before deciding to keep it. `interaction.compare(fixed)`
shows how the combined model compares with the GLM alone.

## Amend a curve and inspect A/E

For a curve, an interval's table relativity is the rate at its starting age.
The edit below sets the rate at age 25 to 0.95. It does not make every age from
25 to 34 equal to 0.95: the adjoining sections change to keep the curve
continuous.

Preview the proposed rates and their effect on A/E and expected claims:

```python
review = interaction.edit_rates("Reviewed driver-age curve")
review.set_relativity("DrivAge", lower=25, upper=35, value=0.95)
review.preview().show()
```

If you want to keep the edit, apply it to a new model. Here we keep the fitted
interaction table as it is:

```python
amended = review.apply(refit_later_interactions=False)
amended.ae("DrivAge").show()
amended.compare(interaction)
```

Compare `amended` with `interaction` to see the effect of the rate change.
The original fit is still available. Once you have chosen the age treatment,
request `amended.validate_holdout(compare_with=[banded, automatic, fixed, interaction])`
to check the candidates on the reserved rows.

## Export and reopen

Export the reviewed curves and interaction table to Excel. Save the fitted
model as well so you can reopen and score it without fitting again:

```python
amended.to_excel("numeric_curves.xlsx")
amended.save("numeric_curves.easyglm")

from easy_glm import PricingModel  # Reopen fitted rates without refitting.

reopened = PricingModel.load("numeric_curves.easyglm", data=data)
reopened.ae("DrivAge").show()
```

## Workbench and stored recipes

In the workbench's GLM variable settings, choose **AGLM smooth curve**, set the
age points and tail behaviour, then apply and refit. For saved settings in
Python, use the [Project guide](project_workflow.md) and
[curve settings reference](../docs/MODELLING_CONVENTIONS.md#optional-aglm-numeric-curves).

## R comparison

For the fitting methods and checks against R `aglm`, see the
[method reference](../docs/MODELLING_CONVENTIONS.md#optional-aglm-numeric-curves).
The defaults can give different fitted results from R.

Missing ages get a separate effect by default. If you disable that setting,
these curve methods require complete age values. They do not support infinite
values or monotone constraints.
