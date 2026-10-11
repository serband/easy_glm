# Building a pricing model

We will build a claim-frequency model on French motor data: fit driver age and
vehicle age, look for missing factors, add two interaction tables, amend rates,
check the model on holdout and export it to Excel.

The code follows the order you would use while modelling. Each new fit gets a
name, so you can return to an earlier model and try a different change. Run the
blocks in order in the same notebook or Python session.

For more worked examples and the browser workflow, see
[Examples and workflows](workflows.md).

## Load the data

Install EasyGLM:

```bash
python -m pip install easy-glm
```

`load_external_dataframe` downloads the French motor data and caches it for
later use. Take 50,000 rows for this walkthrough:

```python
from easy_glm import load_external_dataframe

data = (
    load_external_dataframe()
    .sort("IDpol")
    .sample(n=50_000, seed=20260902)
    .sort("IDpol")
)
```

## Define the project

A `PricingSession` holds the data and settings shared by the models you build:
the target, exposure, family, split and variable bands. We will call it `work`.
Creating the session sets up the project; it does not fit a model.

Use a Poisson model for claim frequency. `ClaimNb` is the claim count and
`Exposure` is in policy-years. The fitted prediction is frequency; the A/E
checks use expected claim counts.

`ignored` excludes a column from all modelling and searches. Here we exclude
`VehGas`, the fuel type. Leave the list empty if you have no exclusions.

```python
from easy_glm import PricingSession

work = PricingSession(
    data,
    family="poisson",
    claims="ClaimNb",
    exposure="Exposure",
    id="IDpol",
    ignored=["VehGas"],
    train_fraction=0.70,
    seed=42,
)
work.summary()
```

The split puts 70% of IDs into training and reserves the rest for holdout.
Rows sharing an ID stay together, even if you reorder the data. Every model
from this session uses the same split.

If you already have a split column, pass its name instead. `PricingSession`
expects **1 for training** and **0 for holdout**, with no missing or other
values. Convert labels such as `"Train"` and `"Test"` to those values first.
This alternative keeps the existing assignments:

```python
# work = PricingSession(
#     data,
#     family="poisson",
#     claims="ClaimNb",
#     exposure="Exposure",
#     id="IDpol",
#     ignored=["VehGas"],
#     split="traintest",  # Existing column: 1 = training, 0 = holdout.
# )
```

Start with `DrivAge` and `VehAge`. Leave `BonusMalus`, `Density`, `Area`,
`Region`, `VehPower` and `VehBrand` out of the first GLM, but available for
searches and later fits. For example, we will use `Region` only in an
interaction. Unlike these unselected factors, the ignored `VehGas` cannot be
used later. Claims, exposure, ID and split are automatically excluded from
factor searches.

## Set how numeric variables are binned

EasyGLM groups numeric values into bands and fits a relativity for each band.
For example, a driver-age band from 25 to below 35 gives ages 25–34 the same
driver-age relativity.

You can ask EasyGLM to choose the bands or set the boundaries yourself. Start
with eight automatic bands as the default for numeric variables:

```python
work.bands(default=8)
```

Automatic boundaries use training data only. Repeated values can give fewer
distinct bands than requested.

Give vehicle age its own count of six bands:

```python
work.bands("VehAge", number=6).show()
```

`VehAge` now uses six bands; other numeric variables still use eight. `.show()`
displays the boundaries and training exposure in each band.

For driver age, choose the boundaries yourself:

```python
work.bands("DrivAge", cuts=[25, 35, 45, 55, 65, 75]).show()
```

Age 35 goes into the band from 35 to below 45. These cuts replace automatic
binning for driver age, so changing the default count will not move them.

Set explicit cuts for vehicle age, bonus-malus and density as well:

```python
work.bands("VehAge", cuts=[1, 3, 6, 10, 15]).show()
work.bands("BonusMalus", cuts=[50, 60, 75, 100, 125]).show()
work.bands("Density", cuts=[50, 200, 1_000, 5_000, 10_000]).show()
```

The vehicle-age cuts replace the six-band setting we tried above. The order
of precedence is **explicit cuts → variable's own count → default count**.
Diagnostics and interaction tables use these bands too.

Region codes identify categories. Tell EasyGLM to treat them that way:

```python
work.categories("Region").show()
```

Save the setup if you want to return to these column choices, split and bands:

```python
work.save_settings("motor_settings.json")

# To reopen these settings with the same policies:
# work = PricingSession.from_settings(data, "motor_settings.json")
```

## Fit the first GLM

`factors` names the main effects to fit together. Start with driver age and
vehicle age:

```python
basic = work.fit_glm("Age model", factors=["DrivAge", "VehAge"])
basic.summary()
```

EasyGLM uses five training folds to choose the regularisation penalty. The
summary lists the fitted factors and any dropped because they have no usable
variation.

`basic` is the fitted Age model. Changing the session settings later leaves
this fit available with its original rates.

Inspect the driver-age rates and A/E for both factors:

```python
basic.relativities("DrivAge").show()
basic.ae("DrivAge").show()
basic.ae("VehAge").show()
```

The charts show actual and predicted frequency, with exposure bars behind
them. The table also shows claim counts and A/E. An A/E above one means the
model underpredicts claims in that band.

### Try revised bands

Look at where actual and expected differ, and how much exposure supports the
gap. To try another set of driver-age bands, change the settings and refit:

```python
# work.bands("DrivAge", cuts=[25, 35, 45, 55, 65, 70, 75])
# different_bands = basic.refit("Different driver-age bands")
# different_bands.compare(basic, cv=True)
```

`cv=True` compares the models on held-back training folds. Each fold learns
its own automatic bands and fit. Without `cv=True`, the comparison uses the
rows fitted, so it is an in-sample check. We will leave holdout until the model
choices are settled.

## Find missing main effects

Ask which unused factors might explain what the Age model misses:

```python
basic.find_missing_factors().show()
```

The search ranks factors to investigate. It does not add them to the model.
Look at the Age model's A/E by bonus-malus and density:

```python
basic.ae("BonusMalus").show()
basic.ae("Density").show()
```

You can group A/E by a factor even when it is absent from the GLM. Here we will
test adding both bonus-malus and density:

```python
main = basic.refit("Four-factor model", add=["BonusMalus", "Density"])
main.compare(basic, cv=True)
main.relativities("BonusMalus").show()
main.ae("BonusMalus").show()
main.ae("Density").show()
main.find_missing_factors().show()
```

`refit` estimates all four main effects together. Compare the models, then
inspect the new rates and A/E. Keep the extra factors if they improve the model
and the fitted effects make sense for pricing.

To check whether density earns its place, fit a version without it:

```python
# without_density = main.refit("Without density", remove=["Density"])
# without_density.compare(main, cv=True)
```

## Fit the first interaction

An interaction lets the effect of one factor vary with another. For example,
the driver-age pattern may differ by bonus-malus.

Search for pairs, then inspect A/E for driver age and bonus-malus:

```python
main.find_interactions().show()
main.ae("DrivAge", "BonusMalus").show()
```

Passing two factors gives an A/E heatmap. Look for a pattern across cells,
alongside their exposure and claims. Fit this pair on top of the four-factor
GLM:

```python
first = main.fit_interaction(
    "DrivAge", "BonusMalus", name="First interaction",
)
first.summary()
first.relativities("DrivAge", "BonusMalus").show()
first.ae("DrivAge", "BonusMalus").show()
first.compare(main)
```

CatBoost uses the GLM prediction as its offset and fits a correction from the
two raw factors. Training cross-validation chooses its settings. EasyGLM then
turns the correction into a rating table using our bands. That table is included
in `first`; the GLM rates stay fixed.

Review the table and A/E, then compare `first` with `main`. The summary shows
the table's validation result and how much fit was lost when converting CatBoost
to a table. If no improvement was selected, the correction leaves the score
unchanged.

## Fit the next interaction

Search again using `first`, so the search accounts for the interaction already
added:

```python
first.find_missing_factors().show()
first.find_interactions().show()
first.ae("VehAge", "Region").show()
```

Region is absent from the main GLM. Use it with vehicle age in a second
interaction, without making it a main effect:

```python
second = first.fit_interaction(
    "VehAge", "Region", name="Second interaction",
)
second.summary()
second.relativities("VehAge", "Region").show()
second.ae("VehAge", "Region").show()
second.compare(first)
```

| Model | Prediction used for A/E and scoring |
| --- | --- |
| `main` | GLM |
| `first` | GLM × first interaction table |
| `second` | GLM × first table × second table |

The second fit uses the complete prediction from `first` as its offset,
including the first rating table. It leaves the GLM and first table fixed.
Compare the result with `first` before choosing whether to keep it.

## Amend a relativity

Inspect the driver-age rates and A/E in `second`:

```python
second.relativities("DrivAge").show()
second.ae("DrivAge").show()
```

Try setting the relativity for ages 25–34 to 0.95. `edit_rates` opens a rate
review, where you can preview a change before applying it:

```python
rates = second.edit_rates(name="Reviewed rates")
rates.set_relativity("DrivAge", lower=25, upper=35, value=0.95)
rates.preview().show()
```

The preview shows the old and proposed rates, A/E and total expected claims.
Use it to assess the size of the change.

Apply the edit while keeping both interaction tables fixed:

```python
adjusted = rates.apply(refit_later_interactions=False)
adjusted.ae("DrivAge").show()
adjusted.compare(second)
```

Or keep the driver-age edit and refit the interactions against the changed
GLM rates:

```python
refit_rates = second.edit_rates(name="Reviewed rates with refitted interactions")
refit_rates.set_relativity("DrivAge", lower=25, upper=35, value=0.95)
refitted = refit_rates.apply(refit_later_interactions=True)
refitted.ae("DrivAge").show()
refitted.compare(adjusted)
```

`refitted` keeps the 0.95 edit. It refits the first interaction, then the second
using the revised score. `adjusted`, `refitted` and the original `second` remain
available to compare.

If you want the rate amendment to leave total training expected claims at their
previous level, use `rebalance()` before applying. Preview that change too:

```python
# rates.rebalance()
# rates.preview().show()
# balanced = rates.apply(refit_later_interactions=False)
```

You can also edit one interaction cell. Set the correction to 1.05 for drivers
aged 25–34 with bonus-malus from 50 to below 60:

```python
cell_review = second.edit_rates(name="One interaction cell changed")
cell_review.set_pair_relativity(
    "DrivAge", "BonusMalus",
    lower_a=25, upper_a=35,
    lower_b=50, upper_b=60,
    value=1.05,
)
cell_review.preview().show()
cell_candidate = cell_review.apply(refit_later_interactions=False)
cell_candidate.relativities("DrivAge", "BonusMalus").show()
```

This keeps the second interaction fixed. Set `refit_later_interactions=True`
to refit it against the changed first table.

## Compare on holdout

Once you have chosen the factors, bands and amendments, compare the candidates
on the reserved holdout:

```python
refitted.validate_holdout(compare_with=[basic, main, first, second, adjusted])
refitted.ae("DrivAge", subset="holdout").show()
refitted.ae("VehAge", "Region", subset="holdout").show()
```

For this Poisson fit, lower deviance means better fit. Check overall A/E and
the factor and pair plots as well. If you use holdout to make more modelling
changes, those rows have become part of development.

Choose the version to export. Below we choose `refitted`; change this to `main`,
`first`, `second` or `adjusted` if you prefer one of those:

```python
accepted = refitted
```

## Export and reopen

Write the reviewed tables to Excel and save the fitted model:

```python
accepted.to_excel("motor_pricing_tables.xlsx")
accepted.save("motor_pricing_model.easyglm")
```

The workbook contains the base rate, main-effect rates and interaction tables,
including the edits. The saved model contains the fitted rates and settings;
it does not contain the data.

Reopen it with the data to inspect rates and A/E without fitting again:

```python
from easy_glm import PricingModel  # Reopen a saved fitted model.

reopened = PricingModel.load("motor_pricing_model.easyglm", data=data)
reopened.relativities("DrivAge").show()
reopened.ae("DrivAge").show()
```

Use the reopened model to calculate frequency or expected claim counts:

```python
predicted_frequency = reopened.predict(data)
expected_claims = reopened.predict(data, expected=True)
```

`predicted_frequency` is claims per policy-year. `expected_claims` uses each
row's exposure to give the expected claim count.
