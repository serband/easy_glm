# Building a pricing model

Run the Python blocks on this page in order, in the same notebook or Python session. All imports, setup and plotting code are included. Stop after each fit to inspect the results before making the next change.

## Install and import

Install the complete release into the Python environment that will run the lesson:

```bash
python -m pip install easy-glm==0.472
```

Download [the 50,000-row French motor sample](https://raw.githubusercontent.com/serband/easy_glm/v0.472/tests/fixtures/french_motor_50k.parquet) and save it as `french_motor_50k.parquet` beside your notebook or script.

```python
import copy
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import TwoSlopeNorm

from easy_glm import EasyGLM, RateModel, add_train_test_split
from easy_glm.engine.models import level_label
from easy_glm.workflow import (
    DataSource, Penalty, Project, Split, VariableDesign,
    ae_by_pair, ae_by_variable, build_design, gini, model_metrics,
    null_model_predict, pearson_dispersion, prepare,
    residual_factor_search, residual_pair_search, run_model,
    to_scoring_script, totals, unit_values,
)
from easy_glm.workflow.project import PairSearchConfig, PairStageConfig

DATA_PATH = Path("french_motor_50k.parquet").resolve()
OUTPUT = Path(os.environ.get(
    "EASY_GLM_LESSON_OUTPUT", "french_motor_lesson_output"
)).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
```

## Load data and set the split

Read the sample and make a 70/30 split. For your own data, change `DATA_PATH` and the column names used below.

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

# Save the split with the data so reopening the project uses the same rows.
SPLIT_DATA_PATH = OUTPUT / "french_motor_50k_fixed_split.parquet"
raw.write_parquet(SPLIT_DATA_PATH)
print({"train": train.height, "holdout": locked_holdout.height})
```

Keep the same split for every model you compare. Use training data for fitting, variable searches and interaction searches. Leave the holdout until you have chosen the model specification.

The sample gives 34,887 training rows and 15,113 holdout rows. Its exposure includes values above one; this example keeps them as supplied.

## Set variable roles and bands

Roles determine which columns are available for modelling. The predictor list passed to a fit determines which ones that particular model uses.

```python
settings_project = Project(name="French motor frequency")
settings_project.data.source = DataSource(type="parquet", path=str(SPLIT_DATA_PATH))
settings_project.data.split = Split(
    mode="column", column="traintest", train_value=1, holdout_value=0,
)
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
settings_project.design.defaults.min_level_share = 0.0025

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
for variable, cuts in CUSTOM_KNOTS.items():
    settings_project.design.variables[variable] = VariableDesign(
        kind="step", knots=cuts,
    )
settings_project.design.variables["Area"] = VariableDesign(
    kind="categorical", max_levels=6,
)
```

To save or edit these choices as JSON:

```python
settings_json = json.dumps(settings_project.to_dict(), indent=2)
settings_roundtrip = Project.from_dict(json.loads(settings_json))
errors = settings_roundtrip.validate(columns=train.columns)
if errors:
    raise ValueError(errors)
(OUTPUT / "variables_project.json").write_text(settings_json, encoding="utf-8")
print(settings_json)

# Use these same cuts when grouping the diagnostics.
DIAGNOSTIC_KNOTS = {
    name: list(design.knots)
    for name, design in settings_roundtrip.design.variables.items()
    if isinstance(design.knots, list)
}
```

The relevant sections are `data.roles`, `data.split`, `design.defaults` and `design.variables`. This is the full Python project format. The smaller Variables JSON shown in the workbench uses a different structure.

To see the automatic cuts, build the design on training data. This copy removes the explicit vehicle-age cuts so you can compare the default count, the vehicle-power override and the fixed driver-age cuts:

```python
binning_demo = Project.from_dict(settings_roundtrip.to_dict())
del binning_demo.design.variables["VehAge"]
binning_spec = build_design(
    binning_demo, train, ["VehAge", "VehPower", "DrivAge"], weight_col="Exposure",
)
print({name: list(binning_spec[name].knots)
       for name in ["VehAge", "VehPower", "DrivAge"]})
```

On this sample, requesting six bins for `VehPower` gives four distinct cuts: 5, 6, 7 and 8.

## Fit the first GLM

Start with `DrivAge` and `VehAge`. Define this function to pass the saved bin settings into `EasyGLM.fit`:

```python
def easyglm_design_kwargs(project: Project, predictors: list[str]) -> dict[str, object]:
    """Pass explicit cuts and shared bin defaults to EasyGLM.fit."""
    defaults = project.design.defaults
    knots: dict[str, list[float]] = {}
    categorical: list[str] = []
    for variable in predictors:
        design = project.design.variables.get(variable, VariableDesign())
        unsupported = (
            design.kind not in (None, "step", "categorical")
            or design.clamp is not None
            or design.monotone is not None
            or design.penalty_weight != 1.0
            or design.levels is not None
        )
        if unsupported:
            raise ValueError(
                f"Use Project/run_model for the design settings on {variable}."
            )
        if design.kind == "categorical":
            categorical.append(variable)
        if isinstance(design.knots, list):
            knots[variable] = [float(value) for value in design.knots]
        elif design.n_bins not in (None, defaults.n_bins):
            raise ValueError(
                f"{variable} uses per-variable automatic n_bins={design.n_bins}. "
                "Use Project/run_model to preserve per-variable automatic bin "
                "settings."
            )
    return {
        "n_bins": defaults.n_bins,
        "min_level_share": defaults.min_level_share,
        "null_indicator": defaults.null_indicator,
        "knots": knots,
        "categorical": categorical,
    }
```

Then fit:

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

`divide_target_by_weight=True` fits `ClaimNb / Exposure`, weighted by `Exposure`. `cv=5` uses five folds to select the penalty from eight values.

`easyglm_design_kwargs` above handles explicit cuts and a shared automatic bin count. For different automatic counts per variable, use `Project` and `run_model`, shown in the interaction section below.

Predictions are claims per unit of exposure. Multiply by exposure to get expected claims:

```python
predicted_rate = skinny.predict(train).to_numpy()
expected_claims = predicted_rate * train["Exposure"].to_numpy()
```

`RateModel.predict` can apply exposure itself. Use `exposure_col=None` when you want rates from that method.

## Inspect A/E and the fitted tables

Calculate actual and expected claim counts, then plot A/E with exposure by band. Define the plotting function once:

```python
def plot_ae_support(
    frame: pl.DataFrame,
    variable: str,
    actual: np.ndarray,
    expected: np.ndarray,
    exposure: np.ndarray,
    *,
    title: str,
) -> pl.DataFrame:
    """Plot training A/E with the exposure supporting every fixed band."""
    table = ae_by_variable(
        frame,
        variable,
        actual,
        expected,
        exposure,
        knots=DIAGNOSTIC_KNOTS.get(variable),
    )
    labels = table["label"].to_list()
    positions = np.arange(len(labels))
    figure, ae_axis = plt.subplots(figsize=(9, 4))
    exposure_axis = ae_axis.twinx()
    exposure_axis.bar(
        positions,
        table["exposure"].to_numpy(),
        color="#d8e6f3",
        label="Exposure",
    )
    ae_axis.set_zorder(exposure_axis.get_zorder() + 1)
    ae_axis.patch.set_visible(False)
    ae_axis.plot(
        positions,
        table["ae"].to_numpy(),
        color="#b23a48",
        marker="o",
        label="A/E",
    )
    ae_axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ae_axis.set_xticks(positions, labels, rotation=35, ha="right")
    ae_axis.set_ylabel("Actual / expected")
    exposure_axis.set_ylabel("Exposure")
    ae_axis.set_title(title)
    figure.tight_layout()
    return table
```

```python
skinny_actual, skinny_expected, skinny_exposure = totals(
    train, skinny.glm, skinny.predict(train).to_numpy(),
)
print({"train_ae": float(skinny_actual.sum() / skinny_expected.sum())})
for variable in ["DrivAge", "VehAge"]:
    ae_table = plot_ae_support(
        train, variable, skinny_actual, skinny_expected, skinny_exposure,
        title=f"Two-factor model: {variable}",
    )
    print(ae_table)
    plt.savefig(OUTPUT / f"skinny_train_ae_{variable}.png", dpi=130)
    plt.show()

print(skinny.rate_model.to_dict()["variables"]["DrivAge"])
```

![Driver-age A/E and exposure](../docs/examples/french_motor_outputs/skinny_train_ae_DrivAge.png)

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

`totals` converts the model's rate predictions into actual and expected claim totals for the residual search:

```python
actual, expected, exposure = totals(
    train, skinny.glm, skinny.predict(train).to_numpy(),
)

skinny_dispersion = pearson_dispersion(
    actual, expected, n_params=len(skinny.glm.coef) + 1,
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

`pearson_dispersion` estimates the dispersion from this fit. The first results are:

| Variable | Residual signal |
|---|---:|
| BonusMalus | 45.04 |
| Area | 13.85 |
| Density | 11.30 |
| Region | 6.09 |

Inspect A/E for the candidates you want to pursue. The score ranks residual patterns; it is not a p-value. In particular, check whether `Area`, `Density` and `Region` are identifying overlapping effects before adding all three.

### Run the optional one-way screen

Run this optional block to test variables individually against noise. Later blocks do not depend on it:

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
print(pl.DataFrame(screen["rows"]).sort("margin", descending=True))
print({key: screen[key] for key in [
    "training_rows", "importance_rows", "fallback_reasons",
]})
```

This compares each variable with four shuffled copies and a random variable. It fits on all training rows and measures importance on a 30% sample, falling back to all training rows if the sample has insufficient support. It does not use the holdout.

Use this for one-way screening. Use the residual search above to assess what a fitted model still misses. A variable that fails the one-way screen can still matter in an interaction.

## Change the main model and refit

Add `BonusMalus` and `Density`, then refit:

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

Recalculate the residuals from this model and search again:

```python
reviewed_actual, reviewed_expected, reviewed_exposure = totals(
    train, reviewed_main.glm, reviewed_main.predict(train).to_numpy(),
)
reviewed_dispersion = pearson_dispersion(
    reviewed_actual, reviewed_expected, n_params=len(reviewed_main.glm.coef) + 1,
)
remaining_variables = ["Area", "Region", "VehPower", "VehBrand", "VehGas"]
reviewed_factor_search = residual_factor_search(
    train, remaining_variables,
    reviewed_actual, reviewed_expected, reviewed_exposure,
    n_bins=8, dispersion=reviewed_dispersion,
)
print(reviewed_factor_search)
print(plot_ae_support(
    train, "BonusMalus", reviewed_actual, reviewed_expected, reviewed_exposure,
    title="Four-factor model: BonusMalus",
))
plt.show()
```

Area's residual score is now below zero; VehGas and Region still show residual signal.

### Change the bands

To test automatic bands, remove that variable's explicit cuts first. This optional comparison fits vehicle age with eight and ten requested bins. Driver age keeps its explicit cuts, and `reviewed_main` stays unchanged:

```python
default_8 = Project.from_dict(settings_roundtrip.to_dict())
del default_8.design.variables["VehAge"]
edited_json = default_8.to_dict()
edited_json["design"]["defaults"]["n_bins"] = 10
default_10 = Project.from_dict(edited_json)

bin_challengers = {}
for label, candidate_project in {
    "default_8": default_8,
    "default_10": default_10,
}.items():
    bin_challengers[label] = EasyGLM.fit(
        train,
        target="ClaimNb",
        model_type="Poisson",
        predictors=["DrivAge", "VehAge"],
        weight_col="Exposure",
        train_test_col="traintest",
        divide_target_by_weight=True,
        cv=5,
        n_alphas=8,
        base="modal",
        **easyglm_design_kwargs(candidate_project, ["DrivAge", "VehAge"]),
    )

print({
    name: list(model.spec["VehAge"].knots)
    for name, model in bin_challengers.items()
})
```

If you change a main effect after fitting interactions, refit the downstream interactions against the changed main model.

## Search for interactions

Use the refitted main model for the pair search:

```python
SEARCH_VARIABLES = ["DrivAge", "VehAge", "BonusMalus", "Density", "Area", "Region"]
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
print(pair_search)

# Inspect the pair before fitting it.
print(ae_by_pair(
    train, "DrivAge", "BonusMalus",
    reviewed_actual, reviewed_expected, reviewed_exposure,
    knots_a=DIAGNOSTIC_KNOTS["DrivAge"],
    knots_b=DIAGNOSTIC_KNOTS["BonusMalus"],
).filter(pl.col("exposure") > 0))
```

`SEARCH_VARIABLES` includes both selected factors and unassigned candidates. The search adjusts for residual one-way margins before ranking pairs.

On this fit, `DrivAge × BonusMalus` is the leading pair. Inspect its cell A/E and exposure, then specify the pair you want to fit. The example names it explicitly; it does not automatically add the top result.

## Fit the first interaction

Put the main model's settings into a `Project` so the interaction fitter can rebuild that model inside each validation fold. This fits the same main model and checks that its predictions match:

```python
project = Project.from_dict(settings_roundtrip.to_dict())
main_config = project.new_model(
    "Reviewed main",
    family="poisson",
    divide_target_by_weight=True,
    predictors=REVIEWED_PREDICTORS,
)
main_config.penalty = Penalty(cv=5, n_alphas=8, l1_ratio=1.0)
main_config.base = "modal"
prepared = prepare(project, raw)
training_only = prepared.filter(pl.col(project.data.split.column) == 1)
assert training_only.height == train.height

main_effects_cache = {}
pair_stages_cache = {}
mains_run = run_model(
    project,
    training_only,
    "Reviewed main",
    main_effects_cache=main_effects_cache,
    pair_stages_cache=pair_stages_cache,
)
assert mains_run.spec.to_dict() == reviewed_main.spec.to_dict()
np.testing.assert_allclose(
    mains_run.predict(training_only),
    reviewed_main.predict(training_only).to_numpy(),
    rtol=1e-8,
    atol=1e-12,
)
```

Copy those settings and add the first pair:

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
print(pair1_run.pair_stages[0])
```

This uses a small Optuna search. Increase the search budget if needed; two trials are the settings used for this example.

CatBoost fits the two raw variables with the main prediction held fixed as an offset. Its fitted correction is then approximated by a two-way rate table on your chosen bands. That approximation can lose some accuracy. The resulting table, rather than the CatBoost predictions, is used for scoring and subsequent interactions.

The correction can include remaining one-way effects. It is not constrained to be a pure interaction.

Define the heatmap function, then plot the fitted table:

```python
def plot_pair_heatmap(pair_table, *, title: str):
    """Relativity plus fitting exposure, with the table's real row labels."""
    matrix = pair_table.cell_matrix
    support = np.asarray([cell.fitting_weight for cell in pair_table.cells]).reshape(
        matrix.shape
    )
    labels = [
        [level_label(row, axis.other_label) for row in axis.table]
        for axis in pair_table.axes
    ]
    figure, axis = plt.subplots(figsize=(11, 7))
    maximum_delta = max(float(np.max(np.abs(matrix - 1.0))), 1e-6)
    normalization = TwoSlopeNorm(
        vmin=1.0 - maximum_delta, vcenter=1.0, vmax=1.0 + maximum_delta
    )
    image = axis.imshow(matrix, cmap="RdBu_r", norm=normalization, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.3f}\n({support[row, column]:.0f})",
                ha="center",
                va="center",
                fontsize=6,
            )
    axis.set_xticks(np.arange(matrix.shape[1]), labels[1], rotation=40, ha="right")
    axis.set_yticks(np.arange(matrix.shape[0]), labels[0])
    axis.set_xlabel(pair_table.parents[1])
    axis.set_ylabel(pair_table.parents[0])
    axis.set_title(title + "\ncell text: relativity (training exposure)")
    figure.colorbar(image, ax=axis, label="Relativity")
    figure.tight_layout()
    return figure
```

```python
figure = plot_pair_heatmap(
    pair1_run.rate_model.pair_tables[0], title="DrivAge × BonusMalus",
)
figure.savefig(OUTPUT / "pair1_relativity_heatmap.png", dpi=130)
plt.show()
```

![DrivAge × BonusMalus relativities and exposure](../docs/examples/french_motor_outputs/pair1_relativity_heatmap.png)

Each cell shows its relativity, with training exposure underneath. Check the thin cells and the size of the correction as well as the CV result:

| CV loss without the new table | CV loss with the new table |
|---:|---:|
| 0.467049 | 0.465230 |

## Search again using the model with the interaction

Use `pair1_run.predict` to include the main model and the saved interaction table:

```python
actual, expected, exposure = totals(
    training_only,
    pair1_run.config,
    pair1_run.predict(training_only),
)
```

Pass these totals to the residual searches. Exclude pairs already in the model, whichever way round their parents are listed:

```python
pair1_dispersion = pearson_dispersion(
    actual, expected, n_params=len(pair1_run.fit.coef) + 1,
)
fitted_pairs = {frozenset(stage.parents) for stage in pair1_run.pair_stages}
remaining_pairs = [
    (a, b) for index, a in enumerate(SEARCH_VARIABLES)
    for b in SEARCH_VARIABLES[index + 1:]
    if frozenset((a, b)) not in fitted_pairs
]
print(residual_factor_search(
    training_only, remaining_variables, actual, expected, exposure,
    n_bins=8, dispersion=pair1_dispersion,
))
print(residual_pair_search(
    training_only, SEARCH_VARIABLES, actual, expected, exposure,
    knots=DIAGNOSTIC_KNOTS, pairs=remaining_pairs,
    n_bins=8, top=12, dispersion=pair1_dispersion,
))
print(ae_by_pair(
    training_only, "VehAge", "Density", actual, expected, exposure,
    knots_a=DIAGNOSTIC_KNOTS["VehAge"],
    knots_b=DIAGNOSTIC_KNOTS["Density"],
).filter(pl.col("exposure") > 0))
```

These calls use predictions from the main model plus the first interaction table. `pair1_run.fit.predict` would omit the interaction table.

## Fit the next interaction

Copy the first interaction model and append the next pair:

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

The main tables and first interaction table stay fixed when the second is added. Check them and print each policy's calculation:

```python
assert (pair2_run.rate_model.to_dict()["variables"]
        == mains_run.rate_model.to_dict()["variables"])
assert (pair2_run.rate_model.to_dict()["pair_tables"][0]
        == pair1_run.rate_model.to_dict()["pair_tables"][0])
assert pair2_run.rate_model.base_rate == mains_run.rate_model.base_rate

main_rate = mains_run.predict(training_only)
first_pair_rate = pair1_run.predict(training_only)
final_rate = pair2_run.predict(training_only)
print(training_only.select("IDpol").head(8).with_columns(
    pl.Series("main_rate", main_rate[:8]),
    pl.Series("first_table_factor", (first_pair_rate / main_rate)[:8]),
    pl.Series("second_offset", np.log(first_pair_rate[:8])),
    pl.Series("second_table_factor", (final_rate / first_pair_rate)[:8]),
    pl.Series("final_rate", final_rate[:8]),
))
print(pair2_run.pair_stages[-1])
actual2, expected2, exposure2 = totals(
    training_only, pair2_run.config, final_rate,
)
print(ae_by_pair(
    training_only, "VehAge", "Density", actual2, expected2, exposure2,
    knots_a=DIAGNOSTIC_KNOTS["VehAge"],
    knots_b=DIAGNOSTIC_KNOTS["Density"],
).filter(pl.col("exposure") > 0))
plot_pair_heatmap(pair2_run.rate_model.pair_tables[-1], title="VehAge × Density")
plt.show()
```

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

This optional alternative fits `DrivAge × Area` directly on top of the main model. `Area` remains unassigned and stays out of the main predictor list. It creates a separate model; later blocks still use `pair2_run`:

```python
assert "Area" not in main_config.predictors
assert project.data.roles.get("Area") in (None, "unassigned")
pair_only_config = copy.deepcopy(main_config)
pair_only_config.pair_method = "sequential_catboost"
pair_only_config.pair_stages = [
    PairStageConfig(
        stage_id="driver_area_optional",
        a="DrivAge",
        b="Area",
        search=PairSearchConfig(trials=1, prefix_trials=1),
    )
]
project.models["Optional pair-only parent"] = pair_only_config
pair_only_run = run_model(
    project,
    training_only,
    "Optional pair-only parent",
    main_effects_cache=main_effects_cache,
    pair_stages_cache=pair_stages_cache,
)
assert (
    pair_only_run.rate_model.to_dict()["variables"]
    == mains_run.rate_model.to_dict()["variables"]
)
assert pair_only_run.pair_stages[0].parents == ("DrivAge", "Area")
assert "Area" not in pair_only_run.rate_model.variables
print(pair_only_run.pair_stages[0])
```

Target, weight, exposure, offset, ID, time and split fields cannot be pair parents.

### Change or remove an earlier stage

Copy the model configuration before changing it. Change the main predictor list, cuts or ordered pair list, then call `run_model` on the revised model. Downstream stages must be rebuilt against the new preceding model; do not carry their old fitted tables across unchanged.

If the new interaction does not improve CV loss, the fitter can retain a neutral correction. Thin cells alone are inconclusive; they do not establish that the interaction has no effect.

## Select the model and check holdout

Choose the model to take forward:

```python
accepted_run = pair2_run   # or pair1_run or mains_run
accepted_model_name = accepted_run.name
accepted_config = accepted_run.config
```

Calculate deviance, A/E and Gini for the same rows in each model. This function takes unit rates and converts them to counts for A/E and Gini:

```python
def checkpoint_metrics(
    frame: pl.DataFrame,
    config,
    fit,
    prediction_rate: np.ndarray,
) -> dict[str, float]:
    """Calculate A/E, Gini and mean deviance on the supplied rows."""
    actual, expected, exposure = totals(frame, config, prediction_rate)
    observed_rate, fitting_weight = unit_values(frame, config)
    deviance = float(
        fit.model.family_instance.deviance(
            observed_rate,
            prediction_rate,
            sample_weight=fitting_weight,
        )
    )
    return {
        "ae": float(actual.sum() / expected.sum()),
        "gini": float(gini(actual, expected, exposure)),
        "mean_deviance": deviance / float(fitting_weight.sum()),
    }
```

```python
models_to_compare = {
    "Two main effects": (skinny.glm, skinny.glm,
                         lambda frame: skinny.predict(frame).to_numpy()),
    "Four main effects": (mains_run.config, mains_run.fit, mains_run.predict),
    "First interaction": (pair1_run.config, pair1_run.fit, pair1_run.predict),
    "Second interaction": (pair2_run.config, pair2_run.fit, pair2_run.predict),
}
comparison_rows = []
for name, (config, fit, predict) in models_to_compare.items():
    for subset, frame in {"train": training_only, "holdout": locked_holdout}.items():
        comparison_rows.append({
            "model": name, "subset": subset,
            **checkpoint_metrics(frame, config, fit, predict(frame)),
        })
comparison = pl.DataFrame(comparison_rows)
print(comparison)

frames = {"train": training_only, "holdout": locked_holdout}
final_metrics = model_metrics(
    accepted_run.fit,
    {name: accepted_run.predict(frame) for name, frame in frames.items()},
    frames,
    accepted_config,
    {name: null_model_predict(project, accepted_config, training_only, frame)
     for name, frame in frames.items()},
)
print(final_metrics)
```

Measured results for the example:

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
project.to_json(OUTPUT / "reviewed_main_project.json")
reviewed_main.save(OUTPUT / "reviewed_main_fit")
resumed_main = EasyGLM.load(OUTPUT / "reviewed_main_fit")
```

Check the loaded fit against the original:

```python
np.testing.assert_allclose(
    resumed_main.predict(train).to_numpy(),
    reviewed_main.predict(train).to_numpy(), rtol=1e-12,
)
```

Load fit files only from a trusted source: they include joblib objects.

For the selected model, export:

| File | Use |
|---|---|
| Project JSON | Roles, bands, split and model specifications |
| RateModel JSON | The fitted main and interaction tables for scoring |
| Excel workbook | Reviewing the rate tables |
| Scores CSV | Policy-level rates and expected claims |
| Python scorer | Applying the saved tables |
| Results JSON | Model comparisons and output paths |

```python
artifact_paths = {
    "project": OUTPUT / "french_motor_project.json",
    "model": OUTPUT / "french_motor_accepted.easyglm",
    "excel": OUTPUT / "french_motor_accepted_tables.xlsx",
    "scores": OUTPUT / "french_motor_holdout_scores.csv",
    "scorer": OUTPUT / "french_motor_frozen_scorer.py",
}
project.champion = accepted_model_name
project.to_json(artifact_paths["project"])
accepted_run.rate_model.to_json(artifact_paths["model"])
accepted_run.rate_model.to_excel(artifact_paths["excel"])

holdout_rate = accepted_run.predict(locked_holdout)
holdout_scores = locked_holdout.select("IDpol", "Exposure").with_columns(
    pl.Series("prediction_rate", holdout_rate),
    pl.Series("expected_claims", holdout_rate * locked_holdout["Exposure"].to_numpy()),
)
holdout_scores.write_csv(artifact_paths["scores"])
loaded_scores = pl.read_csv(artifact_paths["scores"])

restored = RateModel.from_json(artifact_paths["model"])
np.testing.assert_allclose(
    restored.predict(locked_holdout, exposure_col=None), holdout_rate, rtol=1e-12
)
np.testing.assert_allclose(
    loaded_scores["prediction_rate"].to_numpy(), holdout_rate, rtol=1e-12
)
scorer_source = to_scoring_script(accepted_run, output_prefix="frozen")
artifact_paths["scorer"].write_text(scorer_source, encoding="utf-8")
scorer_namespace = {"__name__": "french_motor_frozen_scorer"}
exec(
    compile(scorer_source, str(artifact_paths["scorer"]), "exec"),
    scorer_namespace,
)
np.testing.assert_allclose(
    scorer_namespace["predict"](locked_holdout, exposure_col=None),
    holdout_rate,
    rtol=1e-12,
)
print({name: str(path) for name, path in artifact_paths.items()})

replayed_project = Project.from_json(artifact_paths["project"])
replayed_prepared = prepare(replayed_project)
np.testing.assert_array_equal(
    replayed_prepared.filter(pl.col("traintest") == 1)["IDpol"].to_numpy(),
    training_only["IDpol"].to_numpy(),
)
np.testing.assert_array_equal(
    replayed_prepared.filter(pl.col("traintest") == 0)["IDpol"].to_numpy(),
    locked_holdout["IDpol"].to_numpy(),
)
```

```python
results = {
    "accepted_model": accepted_model_name,
    "comparisons": comparison.to_dicts(),
    "final_metrics": final_metrics,
    "artifacts": {name: str(path) for name, path in artifact_paths.items()},
}
(OUTPUT / "lesson_results.json").write_text(
    json.dumps(results, indent=2, default=str), encoding="utf-8",
)
```

The export uses `accepted_run`. These checks reload the saved tables and run the generated scorer against the same policies. CatBoost is not needed to score the saved tables.

The [recorded results](../docs/examples/french_motor_walkthrough_results.md) include timings and verification notes.
