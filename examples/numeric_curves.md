# Numeric curves

This example compares driver-age bands with an AGLM smooth curve on French motor
data, changes the curve's grid, fits an interaction, amends a relativity and
exports the result. The new curve choices are part of the
[current unreleased implementation](../docs/HANDOVER.md).

## Set up

`PricingSession` holds the data, response, split and variable settings. Each fit
gets a name and keeps its own settings. All the following blocks run in the same
notebook or session.

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

Numeric variables use step bands by default: each band has its own relativity.
We supply the driver-age cuts and use six automatic vehicle-age bands.

```python
work.bands("DrivAge", cuts=[25, 35, 50, 65])
work.bands("VehAge", number=6)
banded = work.fit_glm("Driver-age bands", factors=["DrivAge", "VehAge"], alpha=0.001)
banded.relativities("DrivAge").show()
banded.ae("DrivAge").show()
```

This example fixes the penalty at `alpha=0.001` to make the comparison quick.
Leave out `alpha` to select it using training cross-validation.

## Try a smooth driver-age curve

`work.numeric` selects an AGLM curve for a numeric variable. With `kind="aglm"`,
the fit combines a straight-line effect with a smooth adjustment between grid
points. Unlike step bands, the relativity is continuous at the grid points.
Regularisation acts on those fitted columns; it is different from EasyGLM's
existing band-slope `linear` method.

Automatic grid points use interpolated quantiles of the **training** values.
`n_breaks=6` requests six points, including both endpoints; ties may reduce that
number. Omitting it requests 100 points. It does not use the step-band default.

```python
work.numeric("DrivAge", kind="aglm", n_breaks=6, extrapolation="flat").show()
automatic = banded.refit("Automatic driver-age curve")
automatic.relativities("DrivAge").show()
automatic.ae("DrivAge").show()
automatic.compare(banded)
```

`flat` holds the curve level beyond its first and last points. The other option,
`extend`, continues its fitted outer slopes and is the AGLM default. Choose the
tail behaviour deliberately, especially when scoring ages outside the data range.
`banded` is still available and unchanged.

## Set an explicit grid

Here we replace the automatic grid with `[18, 25, 35, 50, 80]`. The first and last
values are endpoints, so this is four intervals. Explicit points take precedence
over the requested automatic count.

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

These comparisons show in-sample training results. They are not independent
validation. Once you have chosen candidates, request
`fixed.validate_holdout(compare_with=[banded, automatic])` to inspect the reserved
holdout. If those results inform more fitting choices, it is no longer untouched.

The advanced `kind="aglm_l"` option uses a different bend basis, including the
raw numeric value. To try it as a separate candidate:

```python
work.numeric(
    "DrivAge", kind="aglm_l", breaks=[18, 25, 35, 50, 80], extrapolation="flat"
)
l_curve = fixed.refit("Alternative driver-age curve")
l_curve.ae("DrivAge").show()
l_curve.compare(fixed)
```

## Add an interaction to the fixed curve

The interaction starts from `fixed`, including its saved driver-age grid and
complete fitted score. It does not use the later settings used for `l_curve`.
CatBoost fits against that score as an offset, and its correction becomes a
two-factor rating table.

The parent grids also define the interaction-table intervals. We deliberately
used a small driver-age grid so the table can be inspected. Two 100-point grids
would require over 10,000 cells and are refused before fitting. In the workbench,
the interaction panel previews the cell count and lets you choose coarser table
bands without changing the fitted GLM curve.

```python
interaction = fixed.fit_interaction(
    "DrivAge", "VehAge", name="Driver and vehicle age", trials=1, prefix_trials=1
)
interaction.relativities("DrivAge", "VehAge").show()
interaction.ae("DrivAge", "VehAge").show()
interaction.compare(fixed)
```

## Amend a curve and inspect A/E

An AGLM table's relativity is at the start of each finite interval. Changing that
node changes the adjacent slopes to keep the curve continuous. With `extend`,
the outer slopes stay fixed while their endpoint levels move.

We try 0.95 at age 25. This is an example edit, not a recommended rate.

```python
review = interaction.edit_rates("Reviewed driver-age curve")
review.set_relativity("DrivAge", lower=25, upper=35, value=0.95)
review.preview().show()
```

Apply it to a new model, keeping the interaction table fixed:

```python
amended = review.apply(refit_later_interactions=False)
amended.ae("DrivAge").show()
amended.compare(interaction)
```

## Export and reopen

The workbook includes the current curves, tail settings and interaction table.
The saved model scores the same curve, including values beyond the grid.

```python
amended.to_excel("numeric_curves.xlsx")
amended.save("numeric_curves.easyglm")

from easy_glm import PricingModel  # Reopen fitted rates without refitting.

reopened = PricingModel.load("numeric_curves.easyglm", data=data)
reopened.ae("DrivAge").show()
```

## Workbench and stored recipes

Under **Project & data**, select the modelling path, assign the shared target,
weight and split, then select the GLM's variables. In **How this model uses it**,
choose **AGLM smooth curve**. Choose automatic grid points or enter the full grid,
then choose Extend or Flat. Apply the variables and fit on **Model**. Changing a
grid requires a new fit; saved fitted versions retain their original curve.

In a `Project` recipe, the equivalent variable setting is:

```python
from easy_glm.workflow import VariableDesign

driver_age = VariableDesign(
    kind="aglm", breaks=[18, 25, 35, 50, 80], extrapolation="extend"
)
```

Put that definition under `project.design.variables["DrivAge"]` before fitting,
or use the workbench's model-specific Variables JSON. Existing step `knots` and
`n_bins` are separate controls; do not combine them with an AGLM curve.

## R comparison

The basis and weighted penalised objective are checked against R `aglm` 0.4.1
and `glmnet` 5.1 using explicit convergence settings. Default fitted outputs can
differ: glmnet may stop earlier on the redundant AGLM columns. EasyGLM uses glum
for fitting; it does not reproduce glmnet's default stopping point.

Missing values use an additional null indicator by default. With that indicator
disabled, AGLM methods reject missing values. Infinite values and monotone
constraints on these methods are rejected.
