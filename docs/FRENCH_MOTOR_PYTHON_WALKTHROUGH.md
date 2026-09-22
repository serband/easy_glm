# Building models in Python

Open the [Python example](examples/french_motor_walkthrough.py) in VS Code or Jupyter. Run one cell at a time. The examples below use the objects created in that file; the [code reference](FRENCH_MOTOR_PYTHON_REFERENCE.md) contains the full blocks and additional examples.

## Load data and set the split

Run Cells 1–2. The example uses the bundled French motor data. For your own data, change `DATA_PATH`, adapt the column names and roles, and remove the sample-specific 50,000-row assertion.

```python
raw_without_split = pl.read_parquet(DATA_PATH).sort("IDpol")
raw = add_train_test_split(
    raw_without_split,
    train_fraction=0.70,
    seed=42,
    column="traintest",
)
train = raw.filter(pl.col("traintest") == 1)
locked_holdout = raw.filter(pl.col("traintest") == 0)
```

Keep the same split for every model you compare. Use training data for fitting, variable searches and interaction searches. Leave the holdout until you have chosen the model specification.

The sample gives 34,887 training rows and 15,113 holdout rows. Its exposure includes values above one; this example keeps them as supplied.

## Set variable roles and bands

Run Cell 3. Roles determine which columns are available for modelling. The predictor list passed to a fit determines which ones that particular model uses.

```python
settings_project.data.roles = {
    "IDpol": "id",
    "ClaimNb": "target",
    "Exposure": "weight",
    "traintest": "split",
    "DrivAge": "predictor",
    "VehAge": "predictor",
    "BonusMalus": "predictor",
    "Density": "predictor",
}
```

Leave a column out of this mapping to keep it unassigned. It can still be searched or used in an interaction. An unassigned column is not automatically added to the main GLM.

For numeric variables, choose a default bin count, a count for an individual variable, or explicit cuts:

```python
settings_project.design.defaults.n_bins = 8

# This variable gets its own automatic bin count.
settings_project.design.variables["VehPower"] = VariableDesign(
    kind="step", n_bins=6,
)

# This variable uses these exact cut points.
settings_project.design.variables["DrivAge"] = VariableDesign(
    kind="step", knots=[25, 35, 45, 55, 65, 75],
)
```

A value on a cut goes into the band to its right. For example, age 35 belongs to `[35, 45)`. Automatic cuts use training data. Repeated values can produce fewer bins than requested.

Explicit cuts override the bin count. Changing the default from 8 to 10 will not change `DrivAge` while its explicit cuts remain in place.

The four numeric main effects in the example use:

```python
CUSTOM_KNOTS = {
    "DrivAge": [25, 35, 45, 55, 65, 75],
    "VehAge": [1, 3, 6, 10, 15],
    "BonusMalus": [50, 60, 75, 100, 125],
    "Density": [50, 200, 1_000, 5_000, 10_000],
}
```

To save or edit these choices as JSON:

```python
settings_json = json.dumps(settings_project.to_dict(), indent=2)
settings_roundtrip = Project.from_dict(json.loads(settings_json))
```

The relevant sections are `data.roles`, `data.split`, `design.defaults` and `design.variables`. This is the full Python project format. The smaller Variables JSON shown in the workbench uses a different structure.

Cell 3 also prints the resulting cuts. On this sample, requesting six bins for `VehPower` gives four distinct cuts: 5, 6, 7 and 8.

## Fit the first GLM

Run Cell 4. Start with `DrivAge` and `VehAge`:

```python
skinny = EasyGLM.fit(
    train,
    target="ClaimNb",
    model_type="Poisson",
    predictors=["DrivAge", "VehAge"],
    weight_col="Exposure",
    divide_target_by_weight=True,
    cv=5,
    n_alphas=8,
    **easyglm_design_kwargs(settings_roundtrip, ["DrivAge", "VehAge"]),
    base="modal",
)
```

`divide_target_by_weight=True` fits `ClaimNb / Exposure`, weighted by `Exposure`. Do not divide the target yourself as well. `cv=5` uses five folds to select the penalty from eight values.

`easyglm_design_kwargs` is a helper in the example file. It passes the bin settings into `EasyGLM.fit`. For designs it cannot represent, such as different automatic bin counts for individual variables, it directs you to `Project` and `run_model`; it does not discard the setting.

Predictions from `EasyGLM.predict` are rates:

```python
predicted_rate = skinny.predict(train).to_numpy()
expected_claims = predicted_rate * train["Exposure"].to_numpy()
```

`RateModel.predict` can apply exposure itself. Use `exposure_col=None` when you want rates from that method.

## Inspect A/E and the fitted tables

Run Cell 5. It produces training A/E plots with exposure by band.

![Driver-age A/E and exposure](examples/french_motor_outputs/skinny_train_ae_DrivAge.png)

Here the total training A/E is 0.99999. The age bands still differ: the oldest band is around 0.89 and the 45–55 band around 1.08. Check these against exposure and the fitted relativities before changing the bands or adding more flexibility.

For the package's interactive A/E plots:

```python
plots = skinny.plot_actual_vs_expected(
    train.drop("traintest"), show=False,
)
plots["DrivAge"]["All"].show()
```

Dropping the split column makes this a training-only plot labelled `All`. Passing the full dataset here would also expose the holdout results.

Keep solver warnings with the model output. Some runs of this example emitted glum line-search warnings; investigate those fits before relying on their numerical optimum.

## Search for missing main effects

Run Cell 6. `totals` converts the model's rate predictions into actual and expected claim totals for the residual search:

```python
actual, expected, exposure = totals(
    train, skinny.glm, skinny.predict(train).to_numpy(),
)

factor_search = residual_factor_search(
    train,
    ["BonusMalus", "Density", "Area", "Region", "VehPower", "VehBrand", "VehGas"],
    actual,
    expected,
    exposure,
    n_bins=8,
    dispersion=skinny_dispersion,
)
print(factor_search)
```

Cell 6 calculates `skinny_dispersion` from the training residuals. The first results are:

| Variable | Residual signal |
|---|---:|
| BonusMalus | 45.04 |
| Area | 13.85 |
| Density | 11.30 |
| Region | 6.09 |

Inspect A/E for the candidates you want to pursue. The score ranks residual patterns; it is not a p-value. In particular, check whether `Area`, `Density` and `Region` are identifying overlapping effects before adding all three.

### Run the optional one-way screen

Use the shadow-screen example in the [code reference](FRENCH_MOTOR_PYTHON_REFERENCE.md) when you want to test variables individually against noise:

```python
from easy_glm.workflow.feature_selection import select_variables

screen = select_variables(
    settings_roundtrip,
    raw,
    family="poisson",
    divide_target_by_weight=True,
    importance_sample_pct=30.0,
    include_unassigned=True,
    n_alphas=8,
    repeats=5,
    seed=42,
)
```

This compares each variable with four shuffled copies and a random variable. It fits on all training rows and measures importance on a 30% sample, falling back to all training rows if the sample has insufficient support. It does not use the holdout.

Use this for one-way screening. Use the residual search above to assess what a fitted model still misses. A variable that fails the one-way screen can still matter in an interaction.

## Change the main model and refit

Run Cell 7. The example adds `BonusMalus` and `Density`:

```python
reviewed_additions = ["BonusMalus", "Density"]
REVIEWED_PREDICTORS = ["DrivAge", "VehAge", *reviewed_additions]

reviewed_main = EasyGLM.fit(
    train,
    target="ClaimNb",
    model_type="Poisson",
    predictors=REVIEWED_PREDICTORS,
    weight_col="Exposure",
    divide_target_by_weight=True,
    cv=5,
    n_alphas=8,
    **easyglm_design_kwargs(settings_roundtrip, REVIEWED_PREDICTORS),
    base="modal",
)
```

Keep `skinny` so you can compare it with `reviewed_main`. The training results are:

| Model | Mean Poisson deviance | A/E |
|---|---:|---:|
| DrivAge + VehAge | 0.482658 | 1.0000 |
| Add BonusMalus + Density | 0.464219 | 0.9998 |

Cell 7 reruns the searches and plots BonusMalus A/E. Area's residual score is now below zero; VehGas and Region still show residual signal. Use the new results when deciding what to investigate next.

To change bands, edit the settings and refit under a new name. To test automatic bands, remove that variable's explicit cuts first. The code reference includes an executable comparison of eight versus ten default bins.

If you change a main effect after fitting interactions, refit the downstream interactions against the changed main model.

## Search for interactions

Cell 7 also reruns the pair search using the refitted main model:

```python
pair_search = residual_pair_search(
    train,
    SEARCH_VARIABLES,
    reviewed_actual,
    reviewed_expected,
    reviewed_exposure,
    knots=DIAGNOSTIC_KNOTS,
    n_bins=8,
    top=12,
    dispersion=reviewed_dispersion,
)
```

`SEARCH_VARIABLES` includes both selected factors and unassigned candidates. The search adjusts for residual one-way margins before ranking pairs.

On this fit, `DrivAge × BonusMalus` is the leading pair. Inspect its cell A/E and exposure, then specify the pair you want to fit. The example names it explicitly; it does not automatically add the top result.

## Fit the first interaction

Run Cell 8 first. It puts the main model's settings into a `Project` so the interaction fitter can rebuild that model inside each validation fold. It checks that the main predictions still match `reviewed_main`.

Then run Cell 9. The main calls are:

```python
pair1_config = copy.deepcopy(main_config)
pair1_config.pair_method = "sequential_catboost"
pair1_config.pair_stages = [
    PairStageConfig(
        stage_id="driver_bonus",
        a="DrivAge",
        b="BonusMalus",
        search=PairSearchConfig(trials=2, prefix_trials=2),
    )
]
project.models["Pair 1"] = pair1_config

pair1_run = run_model(
    project, training_only, "Pair 1",
    main_effects_cache=main_effects_cache,
    pair_stages_cache=pair_stages_cache,
)
```

This uses a small Optuna search. Increase the search budget if needed; two trials are the settings used for this example.

CatBoost fits the two raw variables with the main prediction held fixed as an offset. Its fitted correction is then approximated by a two-way rate table on your chosen bands. That approximation can lose some accuracy. The resulting table, rather than the CatBoost predictions, is used for scoring and subsequent interactions.

The correction can include remaining one-way effects. It is not constrained to be a pure interaction.

![DrivAge × BonusMalus relativities and exposure](examples/french_motor_outputs/pair1_relativity_heatmap.png)

Each cell shows its relativity, with training exposure underneath. Check the thin cells and the size of the correction as well as the CV result:

| CV loss without the new table | CV loss with the new table |
|---:|---:|
| 0.467049 | 0.465230 |

## Search again using the model with the interaction

Run Cell 10. Use `pair1_run.predict` to include the main model and the saved interaction table:

```python
actual, expected, exposure = totals(
    training_only,
    pair1_run.config,
    pair1_run.predict(training_only),
)
```

Pass these totals to the residual searches. Do not use the original GLM's predictions after adding interaction tables. Exclude pairs already in the model from the list of candidates for another interaction.

Cell 10 prints both one-way and pair results, plus A/E by cell.

## Fit the next interaction

Run Cell 11. Copy the first interaction model and append the next pair:

```python
pair2_config = copy.deepcopy(pair1_config)
pair2_config.pair_stages.append(
    PairStageConfig(
        stage_id="vehicle_density",
        a="VehAge",
        b="Density",
        search=PairSearchConfig(trials=2, prefix_trials=2),
    )
)
project.models["Pair 2"] = pair2_config

pair2_run = run_model(
    project, training_only, "Pair 2",
    main_effects_cache=main_effects_cache,
    pair_stages_cache=pair_stages_cache,
)
```

For this Poisson model the offsets are:

```text
First interaction:   log(main rate)
Second interaction:  log(main rate × first interaction's table factor)
```

The main tables and first interaction table stay fixed when the second is added. Cell 11 checks that they have not changed.

For one policy, the calculation is:

```text
Main rate                  0.117405332
First table factor       × 0.914269531
Rate after first table   = 0.107340118
Second table factor      × 1.002732780
Final rate               = 0.107633455
```

The offset for the second fit is `log(0.107340118)`. Exposure is applied afterwards to turn the final rate into an expected claim count.

Do not create an offset vector from the full training fit and pass it into pair CV yourself. `run_model` rebuilds the preceding models within the validation folds.

### Use a variable only in an interaction

The [code reference](FRENCH_MOTOR_PYTHON_REFERENCE.md) includes a `DrivAge × Area` fit with `Area` left unassigned. Name `Area` as a pair parent and leave it out of the main predictor list. The result has an Area axis in the pair table but no Area main-effect table.

Target, weight, exposure, offset, ID, time and split fields cannot be pair parents.

### Change or remove an earlier stage

Copy the model configuration before changing it. Change the main predictor list, cuts or ordered pair list, then call `run_model` on the revised model. Downstream stages must be rebuilt against the new preceding model; do not carry their old fitted tables across unchanged.

If the new interaction does not improve CV loss, the fitter can retain a neutral correction. Thin cells alone are inconclusive; they do not establish that the interaction has no effect.

## Select the model and check holdout

At the end of Cell 11, choose the model to take forward:

```python
accepted_run = pair2_run   # or pair1_run or mains_run
accepted_model_name = accepted_run.name
accepted_config = accepted_run.config
```

Then run Cell 12. The measured results for the example are:

| Model | Train mean deviance | Holdout mean deviance | Holdout A/E | Holdout Gini |
|---|---:|---:|---:|---:|
| Two main effects | 0.482658 | 0.489994 | 0.9926 | 9.00% |
| Four main effects | 0.464219 | 0.470614 | 0.9988 | 30.09% |
| Add DrivAge × BonusMalus | 0.461834 | 0.470020 | 1.0034 | 30.23% |
| Add VehAge × Density | 0.460535 | 0.471336 | 1.0038 | 29.10% |

The second interaction has worse holdout deviance and Gini than the first. That is a reason to question the addition. There is no uncertainty estimate for the difference in this example.

You can reject an addition and retain the existing model. Do not repeatedly alter the specification against the same holdout and then describe it as independent validation.

## Save the fit and export the tables

To resume work on the main GLM, save its settings and fitted object:

```python
settings_roundtrip.to_json(OUTPUT / "reviewed_main_project.json")
reviewed_main.save(OUTPUT / "reviewed_main_fit")
resumed_main = EasyGLM.load(OUTPUT / "reviewed_main_fit")
```

Load these fit files only from a trusted source: they include joblib objects. The save-and-resume example in the code reference checks that predictions are unchanged after loading.

For the selected model, run Cells 14–15. They save:

| File | Use |
|---|---|
| Project JSON | Roles, bands, split and model specifications |
| RateModel JSON | The fitted main and interaction tables for scoring |
| Excel workbook | Reviewing the rate tables |
| Scores CSV | Policy-level rates and expected claims |
| Python scorer | Applying the saved tables |
| Results JSON | Model comparisons and output paths |

The export uses `accepted_run`. The script reloads the saved model and runs the generated scorer to check their predictions against it. CatBoost is not needed to score the saved tables.

## Files and setup

- [Python example](examples/french_motor_walkthrough.py): run the numbered cells individually.
- [Code reference](FRENCH_MOTOR_PYTHON_REFERENCE.md): full code, automatic-bin comparisons, one-way screening, save/resume and pair-only variables.
- [Recorded results](examples/french_motor_walkthrough_results.md): numbers, timings and verification notes.

These examples use the development version on `codex/french-motor-python-walkthrough`, based on source revision `22bbd7d` or later. From that checkout:

```bash
python -m pip install -e ".[pairs]"
```

When using an AI assistant, give it these files and your current model specification. Ask for the next change you want to make, inspect the output, then decide whether to keep it.
