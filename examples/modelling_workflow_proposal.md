# Building and reviewing a pricing model

**Agreed workflow.** This page records the design. Use [the pricing walkthrough](pricing_walkthrough.md) for the implemented calls and complete examples. Release remains paused.

We are modelling motor claim frequency. Start with driver age and vehicle age, investigate what they miss, then add rating factors and interactions. The additions below illustrate decisions an actuary might make; they are not claims about which factors a search has selected.

## 1. Set up the data once

Tell EasyGLM which columns contain claims, exposure, the policy identifier and the existing split. In this example, `traintest` contains 1 for training and 0 for holdout.

```python
work = PricingSession(
    data,
    family="poisson",
    claims="ClaimNb",
    exposure="Exposure",
    id="IDpol",
    split="traintest",
)
```

The package fits claim frequency with exposure weights. It handles conversion between frequency and expected claim counts. Claims, exposure, ID and split are excluded from factor searches automatically. Other columns remain available to investigate without entering the GLM automatically.

Choose bands directly:

```python
work.bands(default=8)
work.bands("DrivAge", cuts=[25, 35, 45, 55, 65, 75])
work.bands("VehAge", number=6)
```

Automatic boundaries use training data. Supplied cuts take precedence. A band preview should show the actual boundaries and exposure, so you can change them before fitting.

## 2. Fit a small main-effects GLM

```python
basic = work.fit_glm("Age model", factors=["DrivAge", "VehAge"])
```

Use five-fold cross-validation to choose the regularisation by default. The result should immediately show the selected factors, fitting warnings, training A/E and Poisson deviance.

Now look at the fitted shape and the model's errors:

```python
basic.relativities("DrivAge")
basic.ae("DrivAge")
basic.ae("VehAge")
```

`relativities` shows the fitted rates and exposure by band. `ae` shows actual and predicted frequency as separate lines, exposure behind them, and the A/E ratio in the accompanying table. It uses the model's fitted bands automatically. You do not calculate predictions, construct arrays or write plotting code.

Check the shape, thinly populated bands and systematic under- or overprediction. If the bands need changing, change that factor's settings and fit a separately named alternative. Keep `basic` for comparison.

## 3. Find missing main effects

```python
basic.find_missing_factors()
basic.ae("BonusMalus")
basic.ae("Density")
```

The search ranks remaining patterns in the training residuals. Each result should show a residual plot, exposure support and a clear route to inspect the variable. Looking at A/E for an omitted factor must work: expected claims still come from the current full model, grouped by that candidate's bands.

Suppose the evidence supports adding bonus-malus and density. Make that decision explicitly:

```python
main = basic.refit("Four-factor model", add=["BonusMalus", "Density"])
main.compare(basic)
main.ae("BonusMalus")
main.find_missing_factors()
```

This refits the main GLM and estimates all its main effects together. Compare cross-validated deviance, calibration and the fitted shapes. Keep the simpler model if the extra factors do not help. A high residual-search score alone is not a reason to add a factor.

## 4. Fit the first CatBoost interaction

Search using the revised main model:

```python
main.find_interactions()
main.ae("DrivAge", "BonusMalus")
```

The two-variable A/E view should be a heatmap with exposure and claims available for each cell. Suppose we decide to try this pair:

```python
first = main.fit_interaction("DrivAge", "BonusMalus", name="First interaction")
first.relativities("DrivAge", "BonusMalus")
first.compare(main)
```

The package does the fitting work:

- Keep the fitted GLM fixed as the baseline.
- Fit CatBoost on the two raw columns, with the log of the GLM's predicted frequency as the offset for this Poisson model.
- Tune using training cross-validation.
- Convert the correction into a two-way table on the selected bands.
- Use that table for all scoring and diagnostics from this point onwards.

The output should show whether the **table** improves validation loss, how much accuracy was lost converting CatBoost to a table, and which cells have little or no support. Display relativities to at most three decimal places; retain full precision for scoring.

This pair can capture remaining one-way effects as well as an interaction. If it does not improve the model, keep `main`.

## 5. Fit the next interaction on top of everything already accepted

Search again from `first`:

```python
first.find_missing_factors()
first.find_interactions()
first.ae("VehAge", "Region")
```

These residuals must include the GLM and the first interaction table. Region can be considered here even though it is absent from the main GLM.

Suppose we choose vehicle age by region:

```python
second = first.fit_interaction("VehAge", "Region", name="Second interaction")
second.compare(first)
second.ae("VehAge", "Region")
```

The scoring sequence is:

```text
Main model:          GLM
First interaction:   GLM × table 1
Second interaction:  GLM × table 1 × table 2
```

The second CatBoost fit uses the log of the frequency from **GLM × table 1** as its offset, never the first CatBoost model's raw predictions. After conversion, table 2 becomes the next scoring component. Earlier main and interaction tables remain fixed. Further pairs follow the same pattern, in the order selected.

## 6. Inspect and amend relativities

```python
second.relativities("DrivAge")
second.relativities("VehAge", "Region")
```

Show the current relativity, exposure, claims and A/E together. To try a change, create a separate candidate. For example, the following proposes 0.95 for drivers aged 25 to under 35; the value is illustrative:

```python
rates = second.edit_rates(name="Reviewed rates")
rates.set_relativity("DrivAge", lower=25, upper=35, value=0.95)
rates.preview()
```

The preview should show original versus proposed relativities, before/after A/E, and the change in total expected claims. Nothing changes in `second`.

An edit to a main effect or an earlier interaction affects the baseline used by later interactions. The package must make the choice explicit: keep the later tables fixed as a pricing adjustment, or refit those later interactions against the amended baseline. It must preserve the deliberate edit in either case. In this example, choose to refit the affected interactions:

```python
adjusted = rates.apply(refit_later_interactions=True)
adjusted.compare(second)
adjusted.ae("DrivAge")
```

Rebalancing the base rate is a separate choice. Offer to restore a chosen training total, showing the effect before applying it. Do not silently rebalance, rerun fits or erase manual changes.

## 7. Test the finished candidates

Until now, fitting, searches and model choices have used training data and cross-validation. Once the specifications and table changes are settled, compare the shortlisted models on the same reserved holdout:

```python
adjusted.validate_holdout(compare_with=[main, first, second])
adjusted.ae("DrivAge", subset="holdout")
adjusted.ae("VehAge", "Region", subset="holdout")
```

Show Poisson deviance, overall A/E, actual and expected claims, and exposure. Look at calibration by the important factors and pairs, not just the total. Gini can be a separate ranking measure rather than the headline assessment of fit.

Keep a simpler candidate if the new stage is unsupported or unstable. If holdout results drive further changes, treat that holdout as development evidence; it is no longer a fresh final test.

Cross-validation must also respect the sequence: learn automatic bands and fit upstream models inside the training portion of each fold. Later stages use the earlier **tables** built within that fold.

## 8. Export the model you actually intend to use

```python
adjusted.to_excel("motor_pricing_tables.xlsx")
adjusted.save("motor_pricing_model.easyglm")
```

The workbook should contain the current base rate, main-effect tables, ordered interaction tables, band boundaries, support, manual amendments and validation summary. State the exposure and missing/unseen-value rules. The export must reproduce the same predictions as `adjusted`, including every accepted table edit.

The saved model should reopen with its scoring tables, settings and change history. No refitting should be needed to reproduce its predictions.
