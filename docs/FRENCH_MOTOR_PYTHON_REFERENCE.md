# French motor Python reference: complete code and additional examples

Start with the [practical lesson](../examples/pricing_walkthrough.md) for the
explanations and results. This reference preserves the complete code blocks,
additional examples and technical checks for looking things up as you work.

This is a worked Python modelling session for a 50,000-row French motor sample.
It is written primarily as context for an LLM helping an actuary
interactively. The companion script is
[`examples/french_motor_walkthrough.py`](examples/french_motor_walkthrough.py).
Run one numbered cell at a time, inspect what it prints or plots, discuss the
result, and then decide whether to continue. It is deliberately not a one-click
model-selection pipeline.

The example models claim frequency:

- `ClaimNb` is the claim-count total for a policy row;
- `Exposure` is the observation weight;
- the fitted response is `ClaimNb / Exposure`;
- a unit prediction is expected claims per unit exposure;
- multiplying that prediction by exposure gives expected claim count.

The fixture has 50,000 rows, 1,971 claims and 26,273.658314 units of exposure.
Its maximum exposure is 2.01. That value is part of the source data and is not
capped in this example.

Install the complete release into the Python environment that will run the lesson:

```bash
python -m pip install easy-glm==0.472
```

Download [the 50,000-row French motor sample](https://raw.githubusercontent.com/serband/easy_glm/v0.472/tests/fixtures/french_motor_50k.parquet)
and save it as `french_motor_50k.parquet` beside the notebook or script. Print
the package version, module path and signatures used below before trusting this
guide. A useful LLM instruction is: “Use my installed EasyGLM 0.472 API and
signatures; do not invent methods or arguments.”

The runnable cells are the source of truth. Recorded output from a completed
run belongs in
[`examples/french_motor_walkthrough_results.md`](examples/french_motor_walkthrough_results.md),
so this guide does not turn a result from an older specification into an
apparently current fact.

## The working rule: run, inspect, discuss, choose

Every chapter has four parts:

1. **Run** the matching companion-script cell.
2. **Inspect** the named object, table, plot or assertion.
3. **Discuss** what the evidence says and what it does not say.
4. **Choose** an explicit next model specification. Never turn a ranking into an
   automatic inclusion rule.

The holdout frame is created at the beginning but remains untouched until the
main effects, bins and ordered pair stages are locked. Searches use training data
only. Pair-stage validation rebuilds its upstream model within the relevant
folds; the example never supplies hand-made offsets from full-training
predictions to CV.

## 1. Check the environment and load the helper functions

The first cell establishes paths, imports, reproducibility settings and small
display helpers. Run it before every later cell in a fresh process.

```python
# %% 1 — Setup, feature check, and small reusable display helpers
from __future__ import annotations

import copy
import inspect
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import TwoSlopeNorm

import easy_glm
from easy_glm import EasyGLM, RateModel, add_train_test_split
from easy_glm.engine.models import level_label
from easy_glm.workflow import (
    DataSource,
    Penalty,
    Project,
    Split,
    VariableDesign,
    ae_by_pair,
    ae_by_variable,
    build_design,
    gini,
    model_metrics,
    null_model_predict,
    pearson_dispersion,
    permutation_importance,
    prepare,
    residual_factor_search,
    residual_pair_search,
    run_model,
    to_scoring_script,
    totals,
    unit_values,
)
from easy_glm.workflow.project import PairSearchConfig, PairStageConfig

required = {
    "run_model pair cache": "pair_stages_cache"
    in inspect.signature(run_model).parameters,
    "sampled importance API": "importance_sample_pct"
    in inspect.signature(permutation_importance).parameters,
    "automatic pair search": hasattr(PairSearchConfig(), "prefix_trials"),
}
missing = [name for name, present in required.items() if not present]
if missing:
    raise RuntimeError(
        "This lesson uses APIs included in easy-glm 0.472. Install the standard "
        "package with `python -m pip install easy-glm==0.472`. "
        f"Missing: {', '.join(missing)}. Installed version: {easy_glm.__version__}."
    )

DATA_PATH = Path(
    os.environ.get(
        "EASY_GLM_FRENCH_MOTOR_DATA",
        str(Path.cwd() / "french_motor_50k.parquet"),
    )
)
OUTPUT = Path(
    os.environ.get(
        "EASY_GLM_LESSON_OUTPUT",
        str(Path.cwd() / "french_motor_lesson_output"),
    )
)
OUTPUT.mkdir(parents=True, exist_ok=True)
if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"French motor data not found at {DATA_PATH}. Download "
        "https://raw.githubusercontent.com/serband/easy_glm/v0.472/tests/fixtures/"
        "french_motor_50k.parquet and save it as french_motor_50k.parquet, or "
        "set EASY_GLM_FRENCH_MOTOR_DATA to its local path."
    )

# Fixed, reviewed cuts are known before any fold is made. This prevents a
# validation fold from influencing its own bands.
CUSTOM_KNOTS = {
    "DrivAge": [25.0, 35.0, 45.0, 55.0, 65.0, 75.0],
    "VehAge": [1.0, 3.0, 6.0, 10.0, 15.0],
    "BonusMalus": [50.0, 60.0, 75.0, 100.0, 125.0],
    "Density": [50.0, 200.0, 1_000.0, 5_000.0, 10_000.0],
}
SKINNY_PREDICTORS = ["DrivAge", "VehAge"]
REVIEWED_PREDICTORS = ["DrivAge", "VehAge", "BonusMalus", "Density"]
SEARCH_VARIABLES = [
    "DrivAge",
    "VehAge",
    "BonusMalus",
    "Density",
    "Area",
    "Region",
]


def fitted_totals(model: EasyGLM, frame: pl.DataFrame):
    """Observed counts, expected counts, and exposure for one frame."""
    prediction_rate = model.predict(frame).to_numpy()
    return totals(frame, model.glm, prediction_rate)


def easyglm_design_kwargs(project: Project, predictors: list[str]) -> dict[str, object]:
    """Translate this lesson's numeric step settings to the short core API."""
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
                f"{variable} has settings outside this lesson's small "
                "EasyGLM bridge; use Project/run_model directly."
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


def checkpoint_metrics(
    frame: pl.DataFrame,
    config,
    fit,
    prediction_rate: np.ndarray,
) -> dict[str, float]:
    """Comparable count-model metrics from one checkpoint on one locked frame."""
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


print(f"Python executable: {sys.executable}")
print(f"easy_glm module: {easy_glm.__file__}")
print(f"EasyGLM.fit{inspect.signature(EasyGLM.fit)}")
print(f"run_model{inspect.signature(run_model)}")
print(f"easy_glm {easy_glm.__version__}; output folder: {OUTPUT}")
```

Inspect the printed Python executable, source module path, package metadata and
source version. If an LLM proposes an argument that is absent from the printed
signature, stop and reconcile the environment rather than guessing.

**Review pause.** This tutorial needs the ordinary package plus the optional
`pairs` dependencies for CatBoost teacher fitting. Frozen pair tables do not
need CatBoost to score, but this training session does.

**Example LLM prompt**

> Read the environment output. Confirm which EasyGLM source is actually running,
> whether the pair-training dependencies are available, and whether the public
> signatures match the next cell. Do not fit anything yet.
> Stop after this review; do not run or choose the next cell for me.

## 2. Load the fixture, state the units and make one fixed split

The second cell reads the exact local Parquet fixture, checks its totals, sorts
by `IDpol`, and draws the split once with NumPy seed 42. Sorting first makes the
seed meaningful even if Parquet row-group order changes. The resulting training
frame has 34,887 rows; `locked_holdout` has 15,113 rows.

```python
# %% 2 — Load all 50,000 policies, define units, and lock the split
raw_without_split = pl.read_parquet(DATA_PATH).sort("IDpol")
raw = add_train_test_split(
    raw_without_split,
    train_fraction=0.70,
    seed=42,
    column="traintest",
)
SPLIT_DATA_PATH = OUTPUT / "french_motor_50k_fixed_split.parquet"
raw.write_parquet(SPLIT_DATA_PATH)
train = raw.filter(pl.col("traintest") == 1)
locked_holdout = raw.filter(pl.col("traintest") == 0)

assert raw.height == 50_000
assert raw["ClaimNb"].sum() == 1_971
assert np.isclose(raw["Exposure"].sum(), 26_273.658314)
assert np.isclose(raw["Exposure"].max(), 2.01)
assert train.height + locked_holdout.height == raw.height
assert raw["Exposure"].min() > 0
split_counts = {"train": train.height, "holdout": locked_holdout.height}
print(split_counts)
print(
    "Target units: ClaimNb is a count; ClaimNb / Exposure is the unit claim "
    "frequency fitted by the Poisson GLM. Predictions below are unit rates "
    "until multiplied by Exposure."
)
```

Inspect the schema, null counts, target range, exposure range, claim total and
exposure total. A frequency model needs nonnegative counts and strictly positive
finite exposures. Do not silently drop or cap a row to make validation pass.

`raw` contains both partitions and the integer `traintest` flag. `train` and
`locked_holdout` are views of that fixed decision. Do not redraw this split for
challengers.

**Review pause.** Decide whether a random policy-row split is suitable for the
real business problem. A time, customer or grouped split must be built at source;
changing that choice later invalidates model comparisons.

**Example LLM prompt**

> Verify the count/exposure units and the train/holdout counts. Explain why
> exposure is a weight and why `divide_target_by_weight=True` is required. Flag
> any domain issue without editing the data.
> Stop after this review; do not run or choose the next cell for me.

## 3. Write the complete canonical configuration, including bins

The workbench’s compact “Variables JSON” and `Project.to_dict()` are different
schemas. This Python walkthrough uses the canonical `Project` schema. Do not feed
a workbench object with top-level `assignments` or `binning` into
`Project.from_dict()`.

The third cell creates `settings_project`, serialises it to `settings_json`, and
round-trips it through `Project.from_dict()`. Inspect these four sections in the
printed JSON:

- `data.roles`: target, weight, split and assigned predictors;
- `data.split`: the already-created `traintest` flag and its 1/0 meanings;
- `design.defaults`: default number of numeric bins, rare-level threshold and
  null-indicator choice;
- `design.variables`: per-variable overrides, including literal custom cuts.

```python
# %% 3 — Canonical Variables settings JSON, including fixed custom cuts
settings_project = Project(name="French motor interactive lesson")
settings_project.data.source = DataSource(type="parquet", path=str(SPLIT_DATA_PATH))
settings_project.data.roles = {
    "IDpol": "id",
    "ClaimNb": "target",
    "Exposure": "weight",
    "traintest": "split",
    **dict.fromkeys(REVIEWED_PREDICTORS, "predictor"),
}
settings_project.data.split = Split(
    mode="column",
    column="traintest",
    train_value=1,
    holdout_value=0,
)
settings_project.design.defaults.n_bins = 8
settings_project.design.defaults.min_level_share = 0.0025
for variable, knots in CUSTOM_KNOTS.items():
    settings_project.design.variables[variable] = VariableDesign(
        kind="step", knots=knots, n_bins=8
    )
# If VehPower is later promoted, it would use six training-derived quantile
# bins. The main variables above instead use pinned business cuts; explicit
# knots take precedence over n_bins.
settings_project.design.variables["VehPower"] = VariableDesign(
    kind="step", knots="quantile", n_bins=6
)
settings_project.design.variables["Area"] = VariableDesign(
    kind="categorical", max_levels=6
)

# Unassigned candidates such as Area and Region are represented by absence
# from data.roles. They remain available for training-only residual search.
assert "Area" not in settings_project.data.roles
assert "Region" not in settings_project.data.roles
settings_json = json.dumps(settings_project.to_dict(), indent=2, sort_keys=True)
settings_roundtrip = Project.from_dict(json.loads(settings_json))
assert settings_roundtrip.to_dict() == settings_project.to_dict()
assert not settings_roundtrip.validate(columns=train.columns)
DIAGNOSTIC_KNOTS = {
    variable: [float(value) for value in design.knots]
    for variable, design in settings_roundtrip.design.variables.items()
    if isinstance(design.knots, list)
}
(OUTPUT / "variables_project.json").write_text(settings_json, encoding="utf-8")
print(settings_json)

# Three numeric binning cases, inspected on training rows only:
# inherited default (VehAge), per-variable automatic count (VehPower), and
# literal custom cuts (DrivAge). The final reviewed model pins VehAge too; this
# temporary copy exists only to make the difference concrete.
binning_demo = Project.from_dict(settings_roundtrip.to_dict())
del binning_demo.design.variables["VehAge"]
binning_spec = build_design(
    binning_demo,
    train,
    ["VehAge", "VehPower", "DrivAge"],
    weight_col="Exposure",
)
binning_examples = {
    "inherited_default_VehAge": list(binning_spec["VehAge"].knots),
    "per_variable_6_bins_VehPower": list(binning_spec["VehPower"].knots),
    "literal_custom_DrivAge": list(binning_spec["DrivAge"].knots),
}
print(binning_examples)

# A safe JSON edit round-trip: modify the actual canonical shape, reconstruct,
# validate against real columns, and keep the original settings unchanged.
edited_settings = json.loads(settings_json)
edited_settings["design"]["defaults"]["n_bins"] = 10
edited_project = Project.from_dict(edited_settings)
assert edited_project.design.defaults.n_bins == 10
assert not edited_project.validate(columns=train.columns)
```

Role absence means **unassigned**. An unassigned source column is still eligible
for a reviewed pair stage under the current pair-parent rules; it does not become
a GLM main effect unless it is explicitly added to `cfg.predictors` and assigned
the predictor role. This distinction lets an actuary use a variable only in an
interaction without inventing a dummy main effect.

Cuts in the canonical JSON are literal and auditable. They survive project
serialisation, refitting, pair-stage fold-local main fits and training-script
exports. A cut learned from the complete dataset and pasted back into this JSON
would leak holdout information. Use reviewed business cuts or training-derived
cuts only.

This abridged excerpt is copied from the printed canonical shape:

```json
{
  "data": {
    "roles": {
      "IDpol": "id", "ClaimNb": "target", "Exposure": "weight",
      "traintest": "split", "DrivAge": "predictor", "VehAge": "predictor",
      "BonusMalus": "predictor", "Density": "predictor"
    },
    "split": {
      "mode": "column", "column": "traintest",
      "train_value": 1, "holdout_value": 0
    }
  },
  "design": {
    "defaults": {"n_bins": 8, "min_level_share": 0.0025},
    "variables": {
      "DrivAge": {"kind": "step", "knots": [25, 35, 45, 55, 65, 75]},
      "VehPower": {"kind": "step", "knots": "quantile", "n_bins": 6},
      "Area": {"kind": "categorical", "max_levels": 6}
    }
  }
}
```

The actual printout also carries defaults for omitted fields. Preserve those on
round-trip instead of rebuilding a smaller dictionary by hand.

**Worked side branch: refit inherited bins.** A change to the default does not
affect a variable with literal cuts. This opt-in branch removes only `VehAge`'s
override, then fits two training-only challengers so the default change really
reaches the model. `DrivAge` keeps its custom cuts in both.

```python
default_8 = Project.from_dict(settings_roundtrip.to_dict())
del default_8.design.variables["VehAge"]
default_10 = Project.from_dict(default_8.to_dict())
default_10.design.defaults.n_bins = 10

bin_challengers = {}
for label, candidate_project in {
    "default_8": default_8,
    "default_10": default_10,
}.items():
    bin_challengers[label] = EasyGLM.fit(
        train,
        target="ClaimNb",
        model_type="Poisson",
        predictors=SKINNY_PREDICTORS,
        weight_col="Exposure",
        train_test_col="traintest",
        divide_target_by_weight=True,
        cv=5,
        n_alphas=8,
        base="modal",
        **easyglm_design_kwargs(candidate_project, SKINNY_PREDICTORS),
    )

print({
    name: list(model.spec["VehAge"].knots)
    for name, model in bin_challengers.items()
})
```

The six-bin `VehPower` example is also real, but it is an automatic
per-variable override. Use `build_design` or the `Project` path for it so CV can
learn those cuts on the correct training partition. Do not mutate the champion’s
cuts in place after seeing holdout performance.

**Example LLM prompt**

> Read `settings_json`. List every assigned role, default bin rule and explicit
> cut. Tell me which source columns remain unassigned. Check that no holdout value
> was used to choose a cut.
> Stop after this review; do not run or choose the next cell for me.

## 4. Fit a deliberately skinny core `EasyGLM`

The first model uses only two main effects. This is intentional: it makes the
next residual search easy to understand. `EasyGLM.fit` learns the design from
training rows, chooses the lasso penalty by five-fold CV and converts the fit to
portable rate tables.

```python
# %% 4 — Fit a deliberately skinny two-factor core model
skinny = EasyGLM.fit(
    train,
    target="ClaimNb",
    model_type="Poisson",
    predictors=SKINNY_PREDICTORS,
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    cv=5,
    n_alphas=8,
    **easyglm_design_kwargs(settings_roundtrip, SKINNY_PREDICTORS),
    base="modal",
)
assert skinny.predict(train).len() == train.height
np.testing.assert_allclose(
    skinny.predict(train).to_numpy(),
    skinny.rate_model.predict(train, exposure_col=None),
    rtol=1e-12,
)
print(skinny)
```

Inspect `skinny.glm.alpha`, `skinny.spec`, the two relativity tables and their
exposure support. The tables are the model: each prediction is the base rate
times the selected row from every main table. `skinny.glm.predict(train)` and
`skinny.rate_model.predict(train, exposure_col=None)` should agree to floating
point tolerance.

For a quick training-only chart, the public convenience method can be used as:

```python
skinny_plots = skinny.plot_actual_vs_expected(
    train.drop("traintest"),
    show=False,
)
skinny_plots["DrivAge"]["All"].show()
```

Dropping the split flag makes this an explicitly training-only frame and gives
the plot the `All` key. Do not pass `raw` here before the holdout lock is opened.

**Review pause.** Check whether the base level, extreme bands and sparse bands
are intelligible. A fit completing successfully does not make its binning an
actuarial decision.

Keep solver warnings in the review record. Some verified replays emitted glum
line-search convergence warnings. If one appears, ask the LLM to identify the
affected fit and investigate convergence before accepting its numerical result.
Matching exported predictions proves scoring consistency, not convergence.

**Example LLM prompt**

> Explain the fitted base rate and each main relativity table in plain insurance
> language. Identify thin bands or abrupt changes. Do not add a predictor yet.
> Stop after this review; do not run or choose the next cell for me.

## 5. Aggregate training A/E is necessary, not sufficient

The fifth cell scores training rows and computes totals on consistent scales.
Read the printed aggregate training A/E rather than copying a number from this
page. A level close to one says the total is balanced; it does not say risk
differences are explained.

```python
# %% 5 — Inspect training A/E and exposure support; do not open holdout outcomes
skinny_train_actual, skinny_train_expected, skinny_train_exposure = fitted_totals(
    skinny, train
)
skinny_train_ae = float(skinny_train_actual.sum() / skinny_train_expected.sum())
skinny_train_gini = gini(
    skinny_train_actual, skinny_train_expected, skinny_train_exposure
)
skinny_ae_tables = {}
for variable in SKINNY_PREDICTORS:
    skinny_ae_tables[variable] = plot_ae_support(
        train,
        variable,
        skinny_train_actual,
        skinny_train_expected,
        skinny_train_exposure,
        title=f"Skinny model training A/E and support — {variable}",
    )
    plt.savefig(OUTPUT / f"skinny_train_ae_{variable}.png", dpi=130)
    plt.show()
print({"train_ae": skinny_train_ae, "train_gini": skinny_train_gini})
skinny_checkpoint_metrics = checkpoint_metrics(
    train, skinny.glm, skinny.glm, skinny.predict(train).to_numpy()
)
print(skinny_checkpoint_metrics)
# REVIEW PAUSE: ask your LLM to explain weakly supported bands and whether
# a visible A/E pattern is signal, noise, or a reason to revisit the cuts.
```

Inspect aggregate A/E alongside Gini/deviance metrics and one-way A/E tables.
Calibration and discrimination answer different questions:

- A/E near 1 means aggregate expected claims match aggregate actual claims;
- Gini measures ordering, subject to its domain requirements;
- one-way A/E shows systematic under- or over-prediction within a factor;
- deviance measures predictive loss, not business acceptability.

**Worked side branch: poor A/E but good Gini.** Multiplying every prediction by
a constant leaves ordering, and therefore Gini, unchanged while moving A/E.
Conversely, a model can be balanced overall while missing every important rating
factor. Never use one headline metric as a complete model review.

```python
deliberately_miscalibrated = skinny_train_expected * 1.25
miscalibrated_ae = float(
    skinny_train_actual.sum() / deliberately_miscalibrated.sum()
)
miscalibrated_gini = gini(
    skinny_train_actual,
    deliberately_miscalibrated,
    skinny_train_exposure,
)
assert np.isclose(miscalibrated_gini, skinny_train_gini)
print({
    "original_ae": skinny_train_ae,
    "scaled_ae": miscalibrated_ae,
    "original_gini": skinny_train_gini,
    "scaled_gini": miscalibrated_gini,
})
```

**Example LLM prompt**

> Compare aggregate A/E, Gini and the displayed one-way A/E. Give separate
> conclusions about level, ordering and local misfit.
> Stop after this review; do not run or choose the next cell for me.

## 6. Search training residuals; distinguish search from Boruta screening

The sixth cell asks which omitted variables still structure the skinny model’s
training residuals. Read the table produced by the current run. Its values are
search scores, not effect sizes, p-values or automatic acceptance decisions.

```python
# %% 6 — Search training residuals for missing factors and pair structure
missing_variables = [
    "BonusMalus",
    "Density",
    "Area",
    "Region",
    "VehPower",
    "VehBrand",
    "VehGas",
]
skinny_dispersion = pearson_dispersion(
    skinny_train_actual,
    skinny_train_expected,
    n_params=len(skinny.glm.coef) + 1,
)
skinny_factor_search = residual_factor_search(
    train,
    missing_variables,
    skinny_train_actual,
    skinny_train_expected,
    skinny_train_exposure,
    n_bins=8,
    dispersion=skinny_dispersion,
)
skinny_pair_search = residual_pair_search(
    train,
    SEARCH_VARIABLES,
    skinny_train_actual,
    skinny_train_expected,
    skinny_train_exposure,
    knots=DIAGNOSTIC_KNOTS,
    n_bins=8,
    top=12,
    dispersion=skinny_dispersion,
)
print(skinny_factor_search)
print(skinny_pair_search)
# REVIEW PAUSE: these are prompts for investigation, not an automatic
# predictor selector. Discuss causality, stability, support, and leakage.
```

The residual-factor search evaluates what is missing **after the current complete
scorer**. One-way shadow/Boruta-style feature selection asks a different
question: whether a candidate fitted on its own beats shuffled and random
controls. It is optional, slower and useful earlier in exploration. Do not call
the two rankings interchangeable.

Here is an executable opt-in screen. It fits a separate one-way model for each
eligible predictor or unassigned source column. Four shuffled copies and one
uniform-noise variable form its controls. The fixed seed and 30% importance
sample make the scoring repeatable; the underlying candidate fits still use all
training rows.

```python
RUN_OPTIONAL_SHADOW_SCREEN = False
if RUN_OPTIONAL_SHADOW_SCREEN:
    from easy_glm.workflow.feature_selection import select_variables

    shadow_screen = select_variables(
        settings_roundtrip,
        raw,
        family="poisson",
        divide_target_by_weight=True,
        n_alphas=8,
        repeats=5,
        seed=42,
        importance_sample_pct=30.0,
        include_unassigned=True,
    )
    shadow_rows = pl.DataFrame(shadow_screen["rows"]).sort(
        "margin", descending=True, nulls_last=True
    )
    print({
        "method": shadow_screen["method"],
        "training_rows": shadow_screen["training_rows"],
        "importance_rows": shadow_screen["importance_rows"],
        "fallback_reasons": shadow_screen["fallback_reasons"],
    })
    print(shadow_rows)
```

The pair search first removes remaining one-way margin misfit and then looks for
two-way residual structure. A high score is an invitation to inspect cells,
support, business causality and stability.

**Worked side branch: correlated factors.** If `Area`, `Density` and `Region`
share information, their ranks can move after one enters the model. Fit a
reviewed addition, rerun the search, and discuss whether a second factor adds a
distinct explanation. Do not automatically take the top four.

**Example LLM prompt**

> Interpret the residual-factor and pair-search tables. Separate duplicated
> geographic information from plausible distinct effects. Recommend a small
> reviewed candidate set, with reasons, but do not mutate the model.
> Stop after this review; do not run or choose the next cell for me.

## 7. Make literal main-effect decisions and refit the core GLM

The seventh cell uses the literal `REVIEWED_PREDICTORS` list and literal cuts.
That line is the decision boundary. The code does not derive the list from a
ranking. `reviewed_main` is a fresh core `EasyGLM` using the locked specification.

```python
# %% 7 — Apply literal reviewed additions, then refit the main GLM
# Reviewed choice for this fixture: add BonusMalus and Density. Area is
# interesting too, but its relationship with Density deserves a separate
# business discussion. We do not take `head(2)` from the search table.
reviewed_additions = ["BonusMalus", "Density"]
assert REVIEWED_PREDICTORS == SKINNY_PREDICTORS + reviewed_additions
reviewed_main = EasyGLM.fit(
    train,
    target="ClaimNb",
    model_type="Poisson",
    predictors=REVIEWED_PREDICTORS,
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    cv=5,
    n_alphas=8,
    **easyglm_design_kwargs(settings_roundtrip, REVIEWED_PREDICTORS),
    base="modal",
)
reviewed_actual, reviewed_expected, reviewed_exposure = fitted_totals(
    reviewed_main, train
)
reviewed_dispersion = pearson_dispersion(
    reviewed_actual,
    reviewed_expected,
    n_params=len(reviewed_main.glm.coef) + 1,
)
reviewed_factor_search = residual_factor_search(
    train,
    ["Area", "Region", "VehPower", "VehBrand", "VehGas"],
    reviewed_actual,
    reviewed_expected,
    reviewed_exposure,
    n_bins=8,
    dispersion=reviewed_dispersion,
)
reviewed_pair_search = residual_pair_search(
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
reviewed_checkpoint_metrics = checkpoint_metrics(
    train,
    reviewed_main.glm,
    reviewed_main.glm,
    reviewed_main.predict(train).to_numpy(),
)
reviewed_ae_table = plot_ae_support(
    train,
    "BonusMalus",
    reviewed_actual,
    reviewed_expected,
    reviewed_exposure,
    title="Reviewed main training A/E and support — BonusMalus",
)
plt.savefig(OUTPUT / "reviewed_main_train_ae_BonusMalus.png", dpi=130)
plt.show()
print(
    {
        "reviewed_predictors": REVIEWED_PREDICTORS,
        "train_ae": float(reviewed_actual.sum() / reviewed_expected.sum()),
        "train_gini": gini(reviewed_actual, reviewed_expected, reviewed_exposure),
    }
)
print(reviewed_factor_search)
print(reviewed_pair_search)
print(reviewed_checkpoint_metrics)
```

Inspect the new tables, support and training diagnostics. Then rerun the residual
search. A candidate’s earlier score can shrink because the reviewed model now
explains the same structure elsewhere.

**Worked side branch: save and continue.** At this checkpoint save
`settings_json`, the literal reviewed list, cuts, seed and printed diagnostics.
That is enough to reconstruct the decision later. Saving only fitted coefficients
without their design meaning is not enough.

```python
checkpoint_path = OUTPUT / "reviewed_main_project.json"
fit_checkpoint = OUTPUT / "reviewed_main_fit"
settings_roundtrip.to_json(checkpoint_path)
reviewed_main.save(fit_checkpoint)

continued_project = Project.from_json(checkpoint_path)
resumed_main = EasyGLM.load(fit_checkpoint)
assert continued_project.to_dict() == settings_roundtrip.to_dict()
np.testing.assert_allclose(
    resumed_main.predict(train).to_numpy(),
    reviewed_main.predict(train).to_numpy(),
    rtol=1e-12,
)
resumed_actual, resumed_expected, resumed_exposure = fitted_totals(
    resumed_main, train
)
np.testing.assert_allclose(resumed_expected, reviewed_expected, rtol=1e-12)
print({
    "continued_from": str(checkpoint_path),
    "fit_checkpoint": str(fit_checkpoint),
    "reviewed_predictors": REVIEWED_PREDICTORS,
    "cuts": CUSTOM_KNOTS,
    "split_seed": continued_project.data.split.seed,
    "resumed_training_ae": float(resumed_actual.sum() / resumed_expected.sum()),
})
```

`EasyGLM.save` includes joblib estimator files. Load that checkpoint only when it
came from a trusted source. It is useful for continuing this reviewed main-model
session. Chapter 14 separately exports a frozen `RateModel` JSON for staged
production scoring; that JSON does not execute a pickled estimator.

**Review pause.** Accept the main specification only after checking direction,
shape, sparse bands, correlation and stability. Do not inspect holdout yet.

**Example LLM prompt**

> Compare the skinny and reviewed-main training diagnostics. Explain which
> residual signals fell and which remain. Challenge any selected factor whose
> effect looks unstable or redundant.
> Stop after this review; do not run or choose the next cell for me.

## 8. Bridge the reviewed core fit into the advanced workflow

Sequential pair stages need a `Project` because nested validation must rebuild
the upstream main model inside each fold. The eighth cell creates the minimal
advanced configuration, calls `run_model`, and asserts that its main-only design
and predictions match the reviewed core fit.

```python
# %% 8 — Bridge the reviewed core fit into canonical Project/run_model settings
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
print(
    "The Project bridge refits the same reviewed design. Its predictions "
    "match the core fit; the Project is Python settings, not a GUI dependency."
)
```

This parity assertion is central. If the bridge’s predictors, cuts, family,
target, weight, split, penalty path or seed differ, pair CV would validate a
different upstream model from the one just reviewed. Fix the configuration;
never suppress the assertion.

**Example LLM prompt**

> Compare `reviewed_main` and `mains_run`: predictor order, feature names,
> coefficients, selected alpha and training predictions. Explain any mismatch
> before allowing an interaction.
> Stop after this review; do not run or choose the next cell for me.

## 9. Fit the first reviewed pair table

The ninth cell adds one explicit `PairStageConfig` and runs a deliberately
small two-trial automatic search. The parent pair itself is a literal reviewed
choice from the earlier diagnostic; the code does not take the first row of a
ranking. Increase the search budget only as a separate, recorded modelling
choice.

```python
# %% 9 — Add the first reviewed pair stage with a two-trial automatic search
# Literal reviewed pair: the post-refit search highlighted DrivAge × BonusMalus.
first_pair = ("DrivAge", "BonusMalus")
pair1_config = copy.deepcopy(main_config)
pair1_config.pair_method = "sequential_catboost"
pair1_config.pair_stages = [
    PairStageConfig(
        stage_id="driver_bonus",
        a=first_pair[0],
        b=first_pair[1],
        search=PairSearchConfig(trials=2, prefix_trials=2),
    )
]
project.models["Pair 1"] = pair1_config
pair1_run = run_model(
    project,
    training_only,
    "Pair 1",
    main_effects_cache=main_effects_cache,
    pair_stages_cache=pair_stages_cache,
)
assert len(pair1_run.rate_model.pair_tables) == 1
assert pair1_run.pair_stages[0].parents == first_pair
print(pair1_run.pair_stages[0])
# A neutral table is a legitimate CV result. Never promise improvement.
```

For each outer fold, the workflow fits the main GLM on that fold’s training
partition, trains a CatBoost teacher on the pair’s raw parents with the main
scorer as its link-scale baseline, and distils the teacher into a loss-selected
table. The deployed object is that table, not raw CatBoost predictions.

Inspect the pair axes, cell exposures, cell relativities, teacher loss,
table-approximation loss and prefix CV evidence. A parent may be unassigned and
used only by the pair stage; it does not silently enter the main GLM.

**Worked side branch: neutral interaction.** If the table is near 1 throughout,
has weak CV improvement, or concentrates changes in thin cells, the valid
decision is to reject it. “The algorithm produced a table” is not evidence that
the interaction belongs in the model.

**Example LLM prompt**

> Review pair 1 as an actuary. Separate teacher fit from deployed-table evidence.
> Identify cells with weak support and say whether the table earns its complexity.
> Stop after this review; do not run or choose the next cell for me.

## 10. Search again from the complete pair-1 scorer

The tenth cell recomputes residuals using `pair1_run.predict(training_only)`. This is the
main tables multiplied by the first deployed pair table. It is not the main-only
GLM prediction and not the CatBoost teacher prediction.

```python
# %% 10 — Re-search from the complete pair-1 deployed scorer
pair1_prediction = pair1_run.predict(training_only)
pair1_actual, pair1_expected, pair1_exposure = totals(
    training_only, pair1_run.config, pair1_prediction
)
pair1_dispersion = pearson_dispersion(
    pair1_actual,
    pair1_expected,
    n_params=len(pair1_run.fit.coef) + 1,
)
remaining_pairs = [
    (a, b)
    for index, a in enumerate(SEARCH_VARIABLES)
    for b in SEARCH_VARIABLES[index + 1 :]
    if (a, b) != first_pair
]
pair1_residual_search = residual_pair_search(
    training_only,
    SEARCH_VARIABLES,
    pair1_actual,
    pair1_expected,
    pair1_exposure,
    knots=DIAGNOSTIC_KNOTS,
    pairs=remaining_pairs,
    n_bins=8,
    top=12,
    dispersion=pair1_dispersion,
)
pair1_factor_search = residual_factor_search(
    training_only,
    ["Area", "Region", "VehPower", "VehBrand", "VehGas"],
    pair1_actual,
    pair1_expected,
    pair1_exposure,
    n_bins=8,
    dispersion=pair1_dispersion,
)
pair1_checkpoint_metrics = checkpoint_metrics(
    training_only, pair1_run.config, pair1_run.fit, pair1_prediction
)
pair1_cell_ae = ae_by_pair(
    training_only,
    *first_pair,
    pair1_actual,
    pair1_expected,
    pair1_exposure,
    knots_a=DIAGNOSTIC_KNOTS[first_pair[0]],
    knots_b=DIAGNOSTIC_KNOTS[first_pair[1]],
)
pair1_heatmap = plot_pair_heatmap(
    pair1_run.rate_model.pair_tables[0],
    title="Pair 1 deployed relativities",
)
pair1_heatmap.savefig(OUTPUT / "pair1_relativity_heatmap.png", dpi=130)
plt.show()
print(pair1_factor_search)
print(pair1_residual_search)
print(pair1_checkpoint_metrics)
print(pair1_cell_ae.filter(pl.col("exposure") > 0))
# REVIEW PAUSE: this search uses the complete returned RateModel, including
# pair 1. Ask whether a second table is supported and operationally useful.
```

This sequencing prevents pair 2 from taking credit for structure already
captured by pair 1. The next pair’s fold-local offset is built from the fold’s
main tables plus its fold’s pair-1 **table**. No full-training prediction vector
is handed into CV as an offset.

**Review pause.** Reranking after each accepted stage is mandatory. A pair that
looked useful before pair 1 may now be neutral, and a different residual pattern
may become visible.

**Example LLM prompt**

> Compare the pair-search results before and after pair 1. Explain which signal
> pair 1 removed. Propose at most one next pair for review; do not append all
> positive scores.
> Stop after this review; do not run or choose the next cell for me.

## 11. Add pair 2 and prove the prefix stayed frozen

The eleventh cell fits the ordered two-stage configuration with a shared cache.
It asserts that the main rate tables and pair-1 table in `pair2_run` equal the
already-reviewed prefix in `pair1_run`.

```python
# %% 11 — Append pair 2 canonically and prove the deployed prefix stayed frozen
# Literal teaching choice after reviewing cell 10. It is not selected by
# indexing the search result, and neutral CV remains an acceptable outcome.
second_pair = ("VehAge", "Density")
first_table_before = copy.deepcopy(pair1_run.rate_model.to_dict()["pair_tables"][0])
pair2_config = copy.deepcopy(pair1_config)
pair2_config.pair_stages.append(
    PairStageConfig(
        stage_id="vehicle_density",
        a=second_pair[0],
        b=second_pair[1],
        search=PairSearchConfig(trials=2, prefix_trials=2),
    )
)
project.models["Pair 2"] = pair2_config
pair2_run = run_model(
    project,
    training_only,
    "Pair 2",
    main_effects_cache=main_effects_cache,
    pair_stages_cache=pair_stages_cache,
)
assert len(pair2_run.rate_model.pair_tables) == 2
assert pair2_run.rate_model.to_dict()["pair_tables"][0] == first_table_before
assert (
    pair2_run.rate_model.to_dict()["variables"]
    == mains_run.rate_model.to_dict()["variables"]
)
assert pair2_run.rate_model.base_rate == mains_run.rate_model.base_rate
pair2_prefix = pair2_run.rate_model.clone()
pair2_prefix.pair_tables = pair2_prefix.pair_tables[:1]
np.testing.assert_allclose(
    pair2_prefix.predict(training_only, exposure_col=None),
    pair1_run.predict(training_only),
    rtol=1e-12,
)
main_unit_prediction = mains_run.predict(training_only)
pair1_unit_prediction = pair1_run.predict(training_only)
final_unit_prediction = pair2_run.predict(training_only)
pair1_factor = pair1_unit_prediction / main_unit_prediction
pair2_factor = final_unit_prediction / pair1_unit_prediction
np.testing.assert_allclose(
    final_unit_prediction,
    main_unit_prediction * pair1_factor * pair2_factor,
    rtol=1e-12,
)
offset_audit = (
    training_only.select("IDpol")
    .head(8)
    .with_columns(
        pl.Series("main_unit_prediction", main_unit_prediction[:8]),
        pl.Series("pair1_factor", pair1_factor[:8]),
        pl.Series("prefix_unit_prediction", pair1_unit_prediction[:8]),
        pl.Series("log_prefix_offset", np.log(pair1_unit_prediction[:8])),
        pl.Series("pair2_factor", pair2_factor[:8]),
        pl.Series("final_unit_prediction", final_unit_prediction[:8]),
    )
)
print(offset_audit)
print(
    "The stage-2 offset is log(prefix unit rate), not a residual target and not "
    "an expected claim count. Pair CV constructs fold-local offsets internally."
)
print([artifact.status for artifact in pair2_run.pair_stages])

pair2_actual, pair2_expected, pair2_exposure = totals(
    training_only, pair2_run.config, final_unit_prediction
)
pair2_checkpoint_metrics = checkpoint_metrics(
    training_only, pair2_run.config, pair2_run.fit, final_unit_prediction
)
pair2_cell_ae = ae_by_pair(
    training_only,
    *second_pair,
    pair2_actual,
    pair2_expected,
    pair2_exposure,
    knots_a=DIAGNOSTIC_KNOTS[second_pair[0]],
    knots_b=DIAGNOSTIC_KNOTS[second_pair[1]],
)
print(pair2_checkpoint_metrics)
print(pair2_cell_ae.filter(pl.col("exposure") > 0))

# This is an explicit human decision after reviewing all training evidence.
# Change this literal to mains_run or pair1_run if that is the accepted model.
accepted_run = pair2_run
accepted_model_name = accepted_run.name
accepted_config = accepted_run.config
```

The construction is:

```text
pair 1 baseline = main rate tables
pair 2 baseline = main rate tables × deployed pair-1 table
final scorer     = main rate tables × pair-1 table × pair-2 table
```

The teacher is temporary. The final scorer is a frozen sequence of ordinary
lookup tables. The prefix assertions protect the audit story: appending pair 2
must not refit or replace the accepted full-training pair-1 table.

`accepted_run = pair2_run` is a literal training-stage choice for this teaching
run. It is made before any holdout outcome is read. If the training review rejects
pair 2, change that line to `pair1_run`; if it rejects both pairs, use
`mains_run`. Later cells always score and export `accepted_run`, so the choice is
visible rather than inferred from the last object fitted.

**Worked side branch: an unassigned pair-only parent.** Keep the column absent
from `cfg.predictors`, leave its role unassigned, and name it only as a stage
parent. Confirm the main spec is unchanged and the pair axis is present. If the
column is target, weight, exposure, offset, premium, ID, split or time, it is not
eligible.

```python
RUN_OPTIONAL_PAIR_ONLY_PARENT = False
if RUN_OPTIONAL_PAIR_ONLY_PARENT:
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

**Example LLM prompt**

> Verify the frozen-prefix assertions and explain the offset seen by each pair
> teacher. Confirm that no CatBoost object is required by the final scorer.
> Stop after this review; do not run or choose the next cell for me.

## 12. Lock choices, then open holdout once

Only now does the twelfth cell score `locked_holdout`. It reports family-aware
metrics and A/E for training and holdout, using the explicitly chosen complete
scorer. It also reports all four checkpoints so deterioration cannot be hidden
behind the selected name.

```python
# %% 12 — Unlock holdout once: report metrics from the final deployed tables
checkpoint_runs = {
    "skinny": (skinny.glm, skinny.glm),
    "reviewed_main": (mains_run.config, mains_run.fit),
    "pair1": (pair1_run.config, pair1_run.fit),
    "pair2": (pair2_run.config, pair2_run.fit),
}
holdout_checkpoint_metrics = {
    "skinny": checkpoint_metrics(
        locked_holdout,
        skinny.glm,
        skinny.glm,
        skinny.predict(locked_holdout).to_numpy(),
    ),
    **{
        name: checkpoint_metrics(
            locked_holdout,
            config,
            fit,
            run.predict(locked_holdout),
        )
        for name, run, (config, fit) in (
            ("reviewed_main", mains_run, checkpoint_runs["reviewed_main"]),
            ("pair1", pair1_run, checkpoint_runs["pair1"]),
            ("pair2", pair2_run, checkpoint_runs["pair2"]),
        )
    },
}
final_predictions = {
    "train": accepted_run.predict(training_only),
    "holdout": accepted_run.predict(locked_holdout),
}
final_frames = {"train": training_only, "holdout": locked_holdout}
final_null_predictions = {
    name: null_model_predict(project, accepted_config, training_only, frame)
    for name, frame in final_frames.items()
}
final_metrics = model_metrics(
    accepted_run.fit,
    final_predictions,
    final_frames,
    accepted_config,
    final_null_predictions,
)
print(final_metrics)
print(holdout_checkpoint_metrics)
print(
    "Holdout was excluded from fitting, residual search, pair choice, and CV. "
    "A worse or neutral holdout result is evidence, not a workflow failure."
)
```

Interpret changes as validation evidence, not a new tuning signal. If holdout is
poor, record the failure and design a future challenger using training-only
decisions. Repeatedly changing bins or interactions against the same holdout
turns it into training data.

The recorded teaching run found that pair 2 made holdout deviance and Gini worse
than pair 1 despite a small internal CV improvement. The measured values are in
the [results companion](examples/french_motor_walkthrough_results.md). Keep pair
2 here as a negative learning example and a reproducible predeclared export;
do not describe the export as deployment approval and do not switch back to pair
1 after seeing this holdout.

Check at least:

- row count, exposure and claim support in each partition;
- A/E and mean deviance for a Poisson frequency model;
- Gini only where its validity conditions hold;
- training-to-holdout deterioration;
- thin pair cells or unseen levels that fall to table fallbacks.

**Example LLM prompt**

> Treat the specification as locked. Compare training and holdout metrics and
> state whether validation supports deployment, further independent validation,
> or rejection. Do not suggest tuning on this holdout.
> Stop after this review; do not run or choose the next cell for me.

## 13. Inspect a pair table as a heatmap, with support beside it

The thirteenth cell pivots training pair relativities to three decimals and
keeps the exposure/support table beside them. Rounding is for display only; the
scorer retains full precision.

```python
# %% 13 — Inspect a pair relativity heatmap, labelled to at most three decimals
accepted_pair_heatmap = None
if accepted_run.rate_model.pair_tables:
    accepted_pair_heatmap = plot_pair_heatmap(
        accepted_run.rate_model.pair_tables[0],
        title="Accepted model pair 1 (training-defined fixed cuts)",
    )
    accepted_pair_heatmap.savefig(
        OUTPUT / "accepted_pair1_relativity_heatmap.png", dpi=130
    )
    plt.show()
else:
    print("The accepted main-effects model has no pair heatmap.")
```

Read across both axes. A striking relativity in a low-support cell may be noise;
a smooth-looking cell grid can still be redundant with the main margins. Review
the multiplicative combined price effect, not a pair table in isolation.

**Example LLM prompt**

> Describe the strongest and weakest cells, always quoting their support. Check
> whether the pattern is plausible after accounting for both parent main effects.
> Stop after this review; do not run or choose the next cell for me.

## 14. Freeze, export and independently verify scoring parity

The fourteenth cell writes the accepted teaching `RateModel` JSON, Excel rate
tables, readable CSV scores and a standalone scoring script. It reloads or
executes the frozen artifacts and asserts prediction parity. “Accepted” means
the explicit pre-holdout lesson choice; it is not a production approval.

```python
# %% 14 — Freeze JSON, Excel, CSV scores, and a scoring script; verify parity
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

JSON is the portable frozen model. Excel/CSV are review artefacts. The scoring
script is code that applies the frozen tables. A project/training script is a
different artefact: it is a recipe that refits and therefore needs a real data
source path and the training dependencies.

Keep these together:

- source-data identity and split recipe;
- canonical project JSON and literal decisions;
- fitted frozen model and table exports;
- package/source version and environment;
- validation metrics and review notes.

Use `exposure_col=None` to compare rates. Multiply rates by exposure to compare
expected claim counts.

**Example LLM prompt**

> Audit `artifact_paths` and the parity checks. Tell me which file is sufficient
> for frozen scoring, which files are for human review, and what is required to
> retrain the model.
> Stop after this review; do not run or choose the next cell for me.

## 15. Recap the model and write the handover

The last cell prints the chosen main effects, ordered pair stages, source/split
identity, metrics and artefact paths. Copy that output into the modelling record,
then add the reasons for every inclusion and rejection.

```python
# %% 15 — End the lesson with review questions, not an automatic winner
lesson_results = {
    "split_counts": split_counts,
    "skinny_train_ae": skinny_train_ae,
    "skinny_train_gini": skinny_train_gini,
    "reviewed_predictors": REVIEWED_PREDICTORS,
    "accepted_model": accepted_model_name,
    "pair_stages": [artifact.stage_id for artifact in accepted_run.pair_stages],
    "training_checkpoints": {
        "skinny": skinny_checkpoint_metrics,
        "reviewed_main": reviewed_checkpoint_metrics,
        "pair1": pair1_checkpoint_metrics,
        "pair2": pair2_checkpoint_metrics,
    },
    "holdout_checkpoints": holdout_checkpoint_metrics,
    "final_metrics": final_metrics,
    "artifacts": {name: str(path) for name, path in artifact_paths.items()},
}
(OUTPUT / "lesson_results.json").write_text(
    json.dumps(lesson_results, indent=2, default=str), encoding="utf-8"
)
print(json.dumps(lesson_results, indent=2, default=str))
print(
    "Discuss with your LLM: Which conclusions are stable across train and "
    "holdout? Which cuts need business review? Did either pair table win CV, "
    "or was neutral preferred? What monitoring would you require before use?"
)
```

A useful handover separates facts from judgement:

- **Facts:** data totals, split, APIs, exact bins, fitted alpha, metrics, table
  support, parity tolerances and paths.
- **Judgements:** why variables and pairs were accepted, why alternatives were
  rejected, whether shapes are credible, and what validation remains.
- **Open risks:** correlated geography, sparse cells, distribution shift,
  repeated holdout use, operational fallbacks and data-quality dependencies.

**Final LLM prompt**

> Produce a concise actuarial model handover from the printed recap and our
> review notes. Preserve the exact target/weight/split, main list, cuts and stage
> order. Separate measured evidence from our decisions. State that final scoring
> uses frozen main and pair tables, not CatBoost teachers. List unresolved risks
> and the next independent validation step.
> Stop after this review; do not run or choose the next cell for me.

## Common mistakes this walkthrough is designed to prevent

- Treating aggregate A/E near 1 as proof that the model is adequate.
- Looking at holdout while still choosing variables, bins or pairs.
- Calling a residual search an automatic selector.
- Confusing optional one-way shadow screening with current-model residual search.
- Promoting an unassigned pair-only parent into the main GLM without a decision.
- Learning cuts on all 50,000 rows.
- Building pair-CV offsets from full-training predictions.
- Feeding pair 2 the pair-1 CatBoost teacher instead of the deployed pair-1 table.
- Scoring only the main GLM after pair tables have been accepted.
- Multiplying exposure twice.
- Treating rounded display tables as the exact scorer.
- Saving coefficients without their design, base choices and table meanings.
