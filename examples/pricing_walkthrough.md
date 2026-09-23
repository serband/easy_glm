# Building a pricing model

> Review copy for the new interactive workflow. Release is paused.

Run these blocks in order in the same notebook or Python session. Inspect each result before choosing the next change. All the code to run is on this page.

## Load the policies

The complete package is installed with:

```bash
python -m pip install easy-glm
```

`PricingSession` keeps the data, column choices and bands together. `load_external_dataframe` loads the French motor policies and caches the download. We take 50,000 policies for this example.

```python
from easy_glm import PricingSession, load_external_dataframe

data = (
    load_external_dataframe()
    .sort("IDpol")
    .sample(n=50_000, seed=20260902)
    .sort("IDpol")
)
```

For your own policies, replace that loading block with your usual import:

```python
# import polars as pl
# data = pl.read_parquet("motor_policies.parquet")
```

## Set the response and split

`ClaimNb` contains claim counts and `Exposure` contains policy-years. The package fits frequency with exposure weights and calculates expected claim counts for A/E checks.

`work` is our modelling session. Creating it does not fit a model.

```python
work = PricingSession(
    data,
    family="poisson",
    claims="ClaimNb",
    exposure="Exposure",
    id="IDpol",
    train_fraction=0.70,
    seed=42,
)
work.summary()
```

This splits by policy ID: 70% go into training and the rest into holdout. Rows sharing an ID stay together, and reordering the policies does not change their split. Every model from the session uses it. If your data already has a split column, use that instead:

```python
# work = PricingSession(
#     data,
#     family="poisson",
#     claims="ClaimNb",
#     exposure="Exposure",
#     id="IDpol",
#     split="traintest",  # Existing column: 1 = training, 0 = holdout.
# )
```

Claims, exposure, ID and split are excluded from factor searches. Other columns remain available to investigate; they enter the main GLM only when we name them in a fit.

We will start with `DrivAge` and `VehAge`. We leave `BonusMalus`, `Density`, `Area`, `Region`, `VehPower`, `VehBrand` and `VehGas` out of the first GLM, then investigate them.

## Choose the bands

Set eight bands as the default for numeric variables:

```python
work.bands(default=8)
```

Automatic boundaries use training values. Repeated values may produce fewer distinct bands. To give vehicle age its own count:

```python
work.bands("VehAge", number=6).show()
```

That overrides the eight-band default for `VehAge`. The preview shows the boundaries and training exposure in each band.

Use explicit driver-age cuts:

```python
work.bands("DrivAge", cuts=[25, 35, 45, 55, 65, 75]).show()
```

These replace automatic binning for `DrivAge`. The band starting at 35 includes age 35 and ends just before 45. Changing the default count will not move these boundaries.

For the remaining fits, also use fixed cuts for vehicle age, bonus-malus and density. Each line changes one variable's settings:

```python
work.bands("VehAge", cuts=[1, 3, 6, 10, 15]).show()
work.bands("BonusMalus", cuts=[50, 60, 75, 100, 125]).show()
work.bands("Density", cuts=[50, 200, 1_000, 5_000, 10_000]).show()
```

The precedence is **explicit cuts, then the variable's own count, then the default count**. The same settings supply bands for diagnostics and interaction tables.

Region codes represent categories, not a numeric scale:

```python
work.categories("Region").show()
```

Save the column choices, split settings and bands as JSON:

```python
work.save_settings("motor_settings.json")

# To reopen these settings with the same policies:
# work = PricingSession.from_settings(data, "motor_settings.json")
```

## Fit the first GLM

`factors` lists this model's main effects:

```python
basic = work.fit_glm("Age model", factors=["DrivAge", "VehAge"])
basic.summary()
```

By default, five training folds select the regularisation penalty. The summary identifies the fitted factors and any dropped because they have no usable variation.

`basic` keeps its own fitted tables and settings. Later changes to `work` do not alter it.

Inspect the driver-age relativities, then actual versus expected for both factors:

```python
basic.relativities("DrivAge").show()
basic.ae("DrivAge").show()
basic.ae("VehAge").show()
```

A/E charts show actual and predicted frequency with exposure behind them. The table includes claim counts and A/E. Above one means actual claims exceed expected claims.

Inspect persistent gaps and bands with little exposure. To try different boundaries, change the settings and fit a separately named candidate:

```python
# work.bands("DrivAge", cuts=[25, 35, 45, 55, 65, 70, 75])
# different_bands = basic.refit("Different driver-age bands")
# different_bands.compare(basic, cv=True)
```

`cv=True` compares predictions on held-back training folds, rebuilding learned bands and fits within each fold. Without it, `compare` reports current in-sample training results.

## Find missing main effects

Search the Age model's training residuals:

```python
basic.find_missing_factors().show()
```

The search ranks variables outside the main GLM. Nothing is added automatically. Inspect candidates using the current model's A/E:

```python
basic.ae("BonusMalus").show()
basic.ae("Density").show()
```

Neither variable needs to be in the GLM to inspect its A/E. For this example, test adding both:

```python
main = basic.refit("Four-factor model", add=["BonusMalus", "Density"])
main.compare(basic, cv=True)
main.relativities("BonusMalus").show()
main.ae("BonusMalus").show()
main.ae("Density").show()
main.find_missing_factors().show()
```

`refit` estimates all four main effects together and leaves `basic` unchanged. Use the comparison and fitted shapes to decide which model to retain. These choices demonstrate the actions; inspect your search results before making the same additions.

To test removing a factor:

```python
# without_density = main.refit("Without density", remove=["Density"])
# without_density.compare(main, cv=True)
```

## Fit the first interaction

Search the four-factor model's residuals and inspect driver age by bonus-malus:

```python
main.find_interactions().show()
main.ae("DrivAge", "BonusMalus").show()
```

Two variables produce an A/E heatmap. Check the exposure and claims supporting each cell.

```python
first = main.fit_interaction(
    "DrivAge", "BonusMalus", name="First interaction",
)
first.summary()
first.relativities("DrivAge", "BonusMalus").show()
first.ae("DrivAge", "BonusMalus").show()
first.compare(main)
```

CatBoost receives the two raw columns, with the fixed GLM as its offset. Training cross-validation selects its settings. The correction is then converted into a table on our selected bands. **That table** is what we score, inspect and export.

The summary reports validation of the table and the loss from converting CatBoost to a table. A search that finds no improvement produces a neutral correction. The pair can also capture remaining one-way effects.

The matrix displays at most three decimal places and marks unsupported cells. Its scoring values retain full precision.

## Fit the next interaction

Search again from `first`, which includes the first interaction table:

```python
first.find_missing_factors().show()
first.find_interactions().show()
first.ae("VehAge", "Region").show()
```

Region is absent from the main GLM. We can deliberately use it only in an interaction:

```python
second = first.fit_interaction(
    "VehAge", "Region", name="Second interaction",
)
second.summary()
second.relativities("VehAge", "Region").show()
second.ae("VehAge", "Region").show()
second.compare(first)
```

| Model | Scoring prediction |
| --- | --- |
| `main` | GLM |
| `first` | GLM × first interaction table |
| `second` | GLM × first table × second table |

The second CatBoost fit offsets the prediction from `first`, using its **table**, not its raw CatBoost predictions. Adding the second pair leaves the main tables and first pair fixed.

Keep all three candidates available. You can export whichever you accept.

## Amend a relativity

Inspect driver age in the complete model:

```python
second.relativities("DrivAge").show()
second.ae("DrivAge").show()
```

Open a rate review to try an edit. The value below demonstrates the operation; it is not a recommended driver-age rate.

```python
rates = second.edit_rates(name="Reviewed rates")
rates.set_relativity("DrivAge", lower=25, upper=35, value=0.95)
rates.preview().show()
```

The preview shows the old and proposed rates, A/E and total expected claims. `second` has not changed.

To keep the later interaction tables fixed as a pricing adjustment:

```python
adjusted = rates.apply(refit_later_interactions=False)
adjusted.ae("DrivAge").show()
adjusted.compare(second)
```

Alternatively, refit them against the amended main table:

```python
refit_rates = second.edit_rates(name="Reviewed rates with refitted interactions")
refit_rates.set_relativity("DrivAge", lower=25, upper=35, value=0.95)
refitted = refit_rates.apply(refit_later_interactions=True)
refitted.ae("DrivAge").show()
refitted.compare(adjusted)
```

The 0.95 edit stays in place. Fixed numeric cuts allow the same edited band to be identified in each validation fold. Both versions remain separate from `second`.

Rebalancing is a separate choice. In the original rate review, `rebalance()` would restore the training total from before the edit. Preview it before applying:

```python
# rates.rebalance()
# rates.preview().show()
# balanced = rates.apply(refit_later_interactions=False)
```

To amend an interaction cell, specify both bands. This example changes the first pair for drivers aged 25–34 with bonus-malus from 50 to below 60:

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

Here we deliberately keep the second pair fixed. Use `True` to refit that later pair against the edited first table. The full first table, including the edited cell, stays fixed.

## Compare on holdout

All searches and A/E views above used training rows. Once the specifications and edits are settled, compare the candidates on the reserved holdout:

```python
refitted.validate_holdout(compare_with=[basic, main, first, second, adjusted])
refitted.ae("DrivAge", subset="holdout").show()
refitted.ae("VehAge", "Region", subset="holdout").show()
```

For this Poisson model, compare deviance and A/E, then calibration by factor and pair. If these holdout results prompt further tuning, the holdout is now part of development.

Choose what to export explicitly. Change the next line to `main`, `first`, `second` or `adjusted` if that is the model you accept:

```python
accepted = refitted
```

## Export and reopen

```python
accepted.to_excel("motor_pricing_tables.xlsx")
accepted.save("motor_pricing_model.easyglm")
```

The workbook contains the current base rate, main effects and ordered interaction tables, including accepted edits, support and validation information. The saved model contains scoring tables and settings, but no policy data.

Reopen without fitting:

```python
from easy_glm import PricingModel  # Reopen a saved fitted model.

reopened = PricingModel.load("motor_pricing_model.easyglm", data=data)
reopened.relativities("DrivAge").show()
reopened.ae("DrivAge").show()
```

Score policies:

```python
predicted_frequency = reopened.predict(data)
expected_claims = reopened.predict(data, expected=True)
```

The first returns claims per policy-year; the second uses policy exposure to return expected claim counts.
