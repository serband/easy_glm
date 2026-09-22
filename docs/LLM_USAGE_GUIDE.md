# EasyGLM: a practical usage guide for an LLM assistant

**API examples verified with v0.472.**

Give this whole file to the LLM helping you. It is a usage reference, not a request to modify the package. Install as `easy-glm`; import in Python as `easy_glm`. Examples use Polars DataFrames.

Earlier design discussions and some older source docstrings do not describe every detail of the current desktop workbench. In particular, distinguish legacy GLM interactions from the newer ordered CatBoost interaction tables.

A runnable companion, [`examples/llm_workflow.py`](examples/llm_workflow.py), creates synthetic data and exercises fitting, screening, adjustments, exports and optional interaction training. It is useful when checking an unfamiliar work environment before using company data.

## Contents

1. [Instructions to the assisting LLM](#1-instructions-to-the-assisting-llm)
2. [What the package does](#2-what-the-package-does)
3. [Installation and environment checks](#3-installation-and-environment-checks)
4. [Choose the right interface](#4-choose-the-right-interface)
5. [Data, targets, weights and offsets](#5-data-targets-weights-and-offsets)
6. [Training and holdout data](#6-training-and-holdout-data)
7. [A first Python model](#7-a-first-python-model)
8. [The project workflow](#8-the-project-workflow)
9. [Variables, roles and model selections](#9-variables-roles-and-model-selections)
10. [Numeric binning and factor design](#10-numeric-binning-and-factor-design)
11. [Variables JSON](#11-variables-json)
12. [One-way feature selection](#12-one-way-feature-selection)
13. [Fitting the main effects](#13-fitting-the-main-effects)
14. [Interactions fitted in order](#14-interactions-fitted-in-order)
15. [Legacy GLM interactions](#15-legacy-glm-interactions)
16. [Diagnostics and missing-factor searches](#16-diagnostics-and-missing-factor-searches)
17. [Champion and challenger comparison](#17-champion-and-challenger-comparison)
18. [Rate-table adjustments and rebalancing](#18-rate-table-adjustments-and-rebalancing)
19. [Scoring new data](#19-scoring-new-data)
20. [Saving, exporting and reproducing a model](#20-saving-exporting-and-reproducing-a-model)
21. [Using the desktop workbench](#21-using-the-desktop-workbench)
22. [Command-line usage](#22-command-line-usage)
23. [Performance and troubleshooting](#23-performance-and-troubleshooting)
24. [API reference and common mistakes](#24-api-reference-and-common-mistakes)
25. [Useful prompts and handover template](#25-useful-prompts-and-handover-template)
26. [Verification and source references](#26-verification-and-source-references)

## 1. Instructions to the assisting LLM

The following is a suitable starting instruction for a work assistant:

> Help me use EasyGLM for actuarial modelling. Treat the attached guide as the reference for version 0.472. Check my installed version before relying on version-specific features. Use the actual package API; do not invent methods or arguments. Explain choices in plain language and give runnable Python using my column names. Establish what the target, weight, exposure and offset mean before fitting. Preserve my training/holdout split and binning decisions. Use training data for fitting and searches, and reserve holdout data for validation. Keep the main GLM and ordered interaction tables distinct. Use the complete deployed table model when scoring or calculating residual diagnostics. Explain any change to my assumptions or final predictor list. Show the smallest useful next step and verify that it worked.

When responding:

- Start with the user's objective: frequency, severity, pure premium, probability, or a change to an existing premium.
- Ask for the schema, relevant settings and the exact error where needed. A small representative or synthetic example is usually enough to debug an API problem.
- Distinguish an explanation, a proposed setting, an executed fit and a validated result. Do not imply code was run when it was only written.
- Do not replace existing bin cuts, split flags, offsets or manual adjustments just to make an example run.
- Do not silently drop failed screening candidates or automatically accept every suggested interaction.
- Separate sample code from the user's business decisions. For example, a demonstration seed, Tweedie power or bin count is not an actuarial recommendation.
- Prefer the `Project` workflow for a complete workbench-equivalent process. Prefer `EasyGLM.fit` for a short ordinary GLM example. Use lower-level functions only when their additional control is needed.
- If documentation and the installed version disagree, inspect signatures and the installed source before suggesting a workaround.

## 2. What the package does

EasyGLM fits regularised generalised linear models and turns them into interpretable rate tables. It also offers a local browser workbench, training-only variable screening, diagnostics, table adjustments and exports.

A typical workflow is:

**Load data → define variables and bins → define the split → optionally screen variables → fit main effects → optionally fit interactions in order → review diagnostics and holdout results → adjust tables → export.**

For a log-link model, the deployed unit prediction has this form:

```text
base rate
× main-effect relativities
× interaction-table multipliers
× exp(external log offset, if supplied)
```

Exposure multiplication is a separate scoring choice. A frequency prediction of `0.12` and an exposure of `0.5` produce `0.06` expected claims, not `0.12` expected claims.

The main objects are:

| Object | Meaning |
|---|---|
| `Project` | Data preparation, roles, split, shared design settings, model configurations and reviewed adjustments |
| `ModelConfig` | The target/family/penalty/predictors/interactions of one named model |
| `DesignSpec` | The precise encoding of predictors into GLM features |
| `GLMFit` | A fitted GLM and its design; with sequential CatBoost stages this represents the main GLM, not the whole deployed model |
| `ModelRun` | A workflow result, including the fit, current scorer, metrics and pair-stage evidence |
| `RateModel` | The portable table scorer, including applied adjustments and ordered pair tables |
| `EasyGLM` | A convenient wrapper around design, GLM fitting and table conversion |

A bin is a numeric interval. A relativity is a multiplier relative to a selected base level. An offset is a fixed contribution to the linear predictor whose coefficient is not estimated. A penalty shrinks fitted coefficients to reduce complexity. Cross-validation (CV) compares fits using subdivisions of the training data.

## 3. Installation and environment checks

Install into the Python environment that will actually run the notebook, script or workbench:

```bash
python -m pip install easy-glm==0.472
```

Version 0.472 declares Python `>=3.10,<3.15`. Dependency and wheel availability still depend on the operating system and interpreter. CI runs Python 3.10–3.13 on Linux; this is not a claim that every Windows environment has been tested.

The standard installation includes the workbench, GLM fitting, CatBoost, Optuna, Excel export and plotting. No separate interaction installation is needed. Saved tables score without calling CatBoost or Optuna.

Check the active environment:

```python
import sys
import importlib.metadata
import easy_glm

print(sys.executable)
print(importlib.metadata.version("easy_glm"))
print(easy_glm.__file__)
```

In a notebook, installing into a different Python is a common source of confusion. Use that kernel's interpreter, and restart the kernel after an upgrade if it already imported the package.

To check an API rather than guess:

```python
import inspect
from easy_glm import EasyGLM
from easy_glm.workflow import run_model
from easy_glm.workflow.feature_selection import select_variables

print(inspect.signature(EasyGLM.fit))
print(inspect.signature(run_model))
print(inspect.signature(select_variables))
```

## 4. Choose the right interface

| Need | Interface |
|---|---|
| Interactive modelling and table review | Current Svelte browser workbench |
| One ordinary GLM with minimal code | `EasyGLM.fit(...)` |
| Reproducible variables, bins, screening, multiple models and ordered interactions | `Project`, `prepare`, `run_model` |
| Custom design encoders or a non-table-compatible link | `DesignSpec`, `fit_glm` |
| Score an approved frozen model | `RateModel` or a frozen scoring script |
| Refit a saved project from a terminal | `easy-glm run project.json` |

The current desktop interface and the legacy Streamlit interface are different applications. Do not give instructions for Streamlit widgets when the user is looking at the current workbench.

## 5. Data, targets, weights and offsets

### Establish the units first

Before fitting, write down:

1. What does one row represent?
2. Is the target an amount/count or an already-divided rate/average?
3. What does the weight measure?
4. Should prediction return a rate, an amount, or a probability?
5. Is an offset already logged, or is it an ordinary multiplier?

Common configurations are:

| Objective | Target | Weight | Divide target by weight? | Typical family/link |
|---|---|---|---|---|
| Claim frequency | Claim count | Exposure | Yes | Poisson/log |
| Frequency already calculated | Claims / exposure | Exposure | No | Poisson/log |
| Severity from aggregated losses | Total loss on claims | Claim count | Yes | Gamma/log, using suitable positive-loss rows |
| Severity already averaged | Average claim amount | Claim count | No | Gamma/log |
| Pure premium from losses | Total loss | Exposure | Yes | Tweedie/log |
| Pure premium already calculated | Loss / exposure | Exposure | No | Tweedie/log |
| Binary event | 0/1 event | Optional fitting weight | Usually no | Binomial/logit |

These are modelling patterns, not automatic settings. Confirm the data meaning, particularly when an existing `weight` column is not ordinary exposure.

Weights used by the GLM must be finite and strictly positive. Targets must satisfy the selected family: Gamma is strictly positive; Poisson and the supported Tweedie range allow zero and positive values; binomial responses are in `[0,1]`. Investigate invalid rows before changing or removing them.

**Do not divide an already-divided target again. Do not multiply a total prediction by exposure twice.**

### Prediction and diagnostic scales

- `EasyGLM.predict(frame)` returns a Polars Series on the fitted response scale; if the target was divided by weight, this is per unit weight.
- `GLMFit.predict(frame)` returns a NumPy array on that same response scale.
- `ModelRun.predict(prepared)` uses the current complete table scorer with exposure multiplication disabled. Use it for workflow diagnostics.
- `RateModel.predict(frame)` normally uses its stored exposure setting.
- `RateModel.predict(frame, exposure_col=None)` explicitly disables exposure multiplication; it still includes the model's offset.

The workflow's `totals(frame, config, unit_prediction)` returns `(actual, expected, weight)` on consistent diagnostic scales:

| Configuration | Diagnostic actual | Diagnostic expected |
|---|---|---|
| No weight | target | unit prediction |
| Weight and divide=True | target | unit prediction × weight |
| Weight and divide=False | target × weight | unit prediction × weight |

That last row matters: with fitting weights and an already-averaged response, diagnostics aggregate weighted amounts. Use `totals` rather than reconstructing this logic inconsistently.

### Offsets and existing premiums

`offset_col` identifies a column already on the link scale. With a log link, an existing baseline amount `B` generally supplies `log(B)`, not `B`.

In a `Project`, assigning a positive premium column the `current_premium` role derives its logarithm during preparation. New models use that derived offset unless an explicit offset role is set. Filter invalid premium rows deliberately; zero or negative premiums have no finite logarithm.

Do not add an exposure offset merely because a frequency model has exposure weights. The target-division/weight convention and an exposure-offset formulation are different specifications; an LLM should establish which is intended.

### Links and families

EasyGLM defaults to **log for every family except binomial**, which defaults to logit. This includes Gaussian; do not assume its default is identity.

Multiplicative rate-table conversion supports log and logit. With logit, tables contain **odds** relativities and the base is base odds; scoring converts odds into probabilities. The scorer refuses exposure multiplication for logit models.

A lower-level identity-link fit can be used through `fit_glm` and `fit.predict`, but cannot be converted to the ordinary multiplicative `RateModel`. Consequently, do not promise an identity-link model will complete `EasyGLM.fit` or the table-based workflow just because the underlying GLM accepts that link.

Tweedie power must be strictly between 1 and 2; the default is 1.5. It is a distribution choice, not the lasso penalty or an exposure exponent. The interaction tuner does not choose it automatically.

## 6. Training and holdout data

The core convention is **1 = training, 0 = holdout**.

Create a reproducible random split:

```python
from easy_glm import add_train_test_split

raw = add_train_test_split(raw, train_fraction=0.7, seed=42, column="traintest")
```

This helper refuses to overwrite an existing split column. Keep an existing approved split rather than drawing a new one when comparing models.

For a project with textual source flags:

```python
from easy_glm.workflow import Split

project.data.split = Split(
    mode="column",
    column="Partition",
    train_value="TRAIN",
    holdout_value="TEST",
)
```

`prepare` converts the selected flags to the workflow convention. Providing both values makes the intended partition explicit; inspect row counts and unmatched/missing source flags. Do not repeatedly run preparation on already-prepared data, particularly after renaming columns or normalising textual flags.

For a project-generated split:

```python
project.data.split = Split(mode="random", column="traintest", fraction=0.7, seed=42)
```

Automatic cuts and categorical levels are learned using training data. Ordinary GLM CV chooses penalties within that training design; do not describe it as re-learning every main-effect cut independently in each CV fold. Sequential pair validation has additional fold-local upstream fitting.

The random split helper is not a grouped or chronological split generator. If policies, customers or time periods must stay together, create the appropriate source split explicitly. Also do not claim the built-in penalty CV or pair CV becomes grouped/time-based merely because the outer holdout is chronological.

## 7. A first Python model

This example expects a Parquet file with `Claims`, `Exposure`, `DriverAge`, `VehicleAge`, `Region` and a valid `traintest` column:

```python
import numpy as np
import polars as pl
from easy_glm import EasyGLM

raw = pl.read_parquet("portfolio.parquet")
model = EasyGLM.fit(
    raw,
    target="Claims",
    model_type="Poisson",
    predictors=["DriverAge", "VehicleAge", "Region"],
    weight_col="Exposure",
    divide_target_by_weight=True,
    train_test_col="traintest",
    cv=5,
    n_alphas=20,
    l1_ratio=1.0,
    n_bins=20,
    knots={"DriverAge": [25, 35, 45, 55, 65]},
)

frequency = model.predict(raw)
expected_claims = model.rate_model.predict(raw)
np.testing.assert_allclose(
    expected_claims,
    frequency.to_numpy() * raw["Exposure"].to_numpy(),
)
print(model.coef_table())
print(model.relativities["DriverAge"])
model.rate_model.to_json("frequency.easyglm")
model.rate_model.to_excel("frequency_rate_tables.xlsx")
```

`EasyGLM.fit` filters training rows itself. `fit_glm`, the lower-level building block, does not: pass it training rows only.

Use `model_type=` with `EasyGLM.fit`, and `family=` with `fit_glm` or `ModelConfig`. They are different signatures. `n_alphas` is accepted by `EasyGLM.fit` through forwarded fitting arguments. Project-specific options such as `pair_stages` should not be sent through those forwarded GLM arguments.

For the complete synthetic example:

```bash
python llm_workflow.py --out guide-output
python llm_workflow.py --out guide-output-pairs --with-pairs
```

The second command also runs interaction training. Both use the standard installation. The example uses a smaller tuning budget than the workbench defaults.

## 8. The project workflow

Use this interface to mirror workbench behaviour and retain settings explicitly:

```python
import polars as pl
from easy_glm.workflow import (
    Project, DataSource, Split, VariableDesign, Penalty,
    prepare, run_model, train_holdout, totals,
)

raw = pl.read_parquet("portfolio.parquet")
project = Project(name="Motor frequency")
project.data.source = DataSource(type="parquet", path="portfolio.parquet")
project.data.roles = {
    "Claims": "target",
    "Exposure": "weight",
    "DriverAge": "predictor",
    "VehicleAge": "predictor",
    "Region": "predictor",
    "traintest": "split",
}
project.data.split = Split(
    mode="column", column="traintest", train_value=1, holdout_value=0,
)
project.design.defaults.n_bins = 20
project.design.variables["DriverAge"] = VariableDesign(knots=[25, 35, 45, 55, 65])
project.design.variables["VehicleAge"] = VariableDesign(n_bins=8)
project.design.variables["Region"] = VariableDesign(kind="categorical")

cfg = project.new_model("Frequency", family="poisson", divide_target_by_weight=True)
cfg.penalty = Penalty(cv=5, n_alphas=20, l1_ratio=1.0)
prepared = prepare(project, raw)
problems = project.validate("Frequency", columns=prepared.columns)
if problems:
    raise ValueError("\n".join(problems))
run = run_model(project, prepared, "Frequency")
train, holdout = train_holdout(prepared, project.data.split)
actual, expected, weight = totals(holdout, run.config, run.predict(holdout))
print(run.metrics)
print("Holdout A/E:", actual.sum() / expected.sum())
project.to_json("project.json")
```

Passing `raw` to `prepare(project, raw)` avoids loading the source again. `prepare(project)` instead reads `project.data.source`. It returns the full prepared data, not a sample. `run_model` fits only its training partition.

Supported source loaders include Parquet, CSV, IPC/Feather, Excel and SAS7BDAT. `DataSource.options` passes loader-specific options. Explicitly set the type for unusual file extensions. Use a path the receiving machine can read; a project JSON does not contain the entire source dataset.

Preparation applies renames, categorical recodes, type overrides, derived expressions, filters, premium-offset derivation and split handling. Expressions use Polars syntax, for example:

```python
from easy_glm.workflow import Derived, Recode

project.data.recodes["Region"] = Recode(mapping={"N": "North", "S": "South"})
project.data.derived.append(
    Derived(name="VehicleAgeSquared", expr="pl.col('VehicleAge') ** 2")
)
project.data.filters.append("pl.col('Exposure') > 0")
```

These settings must be applied before fitting a new prepared frame. Do not interpret a typed expression as SQL, and do not assume derived-only columns will appear in the one-way screen's raw-source candidate list.

## 9. Variables, roles and model selections

The Variables page determines which source columns are eligible and how they are prepared. A model has a separate list of selected main effects.

| Role | Purpose |
|---|---|
| `target` | Response to model |
| `weight` | Fitting weight and diagnostic aggregation weight |
| `exposure` | Separate scoring exposure, when intentionally configured |
| `offset` | Existing link-scale offset |
| `current_premium` | Positive premium from which a log offset is derived |
| `predictor` | Eligible for modelling |
| `id` | Identifier, excluded from automatic candidate screening |
| `time` | Time grouping for diagnostics, excluded from automatic screening |
| `split` | Source training/holdout indicator |
| `unassigned` | No modelling role yet; optionally screened |
| `ignore` | Deliberately excluded |

Setting six columns to `predictor` does not necessarily mean an existing model contains six main effects. Inspect both:

```python
print(project.predictors)
print(project.models["Frequency"].predictors)
```

`project.new_model(...)` initially copies the current predictor-role list. In an existing model, select the desired eligible variables in Factor design or deliberately update `cfg.predictors`. Do not indiscriminately overwrite every model's selection: a reduced challenger may intentionally contain fewer variables.

If roles were changed in the browser but not applied, the model page may still reflect the saved state. Also check whether a predictor search/filter is hiding rows. After fitting, inspect `run.dropped_predictors`; constant or all-null training columns can be excluded as unusable.

Use `project.rename_column(old, new)` and `project.apply_role_change(column, role)` for changes to an existing project with model references. Direct dictionary assignments are appropriate for building a new example, but may leave existing references inconsistent if used as ad-hoc edits.

## 10. Numeric binning and factor design

### Shared defaults and overrides

In Variables, set the project default number of bins. For a particular numeric variable choose the default, its own automatic bin count, or explicit cut points.

Python equivalents:

```python
from easy_glm.workflow import VariableDesign

project.design.defaults.n_bins = 20
project.design.variables["DriverAge"] = VariableDesign(n_bins=10)
project.design.variables["VehicleAge"] = VariableDesign(knots=[0, 1, 2, 3, 4, 5])
```

When editing an existing `VariableDesign`, update its relevant attributes rather than replacing the whole object and accidentally losing a clamp, monotonicity constraint or penalty weight. For example, switching from explicit cuts to an automatic count requires `vd.knots = "quantile"` as well as `vd.n_bins = 10`.

Automatic numeric cuts are training-row quantiles. They are not guaranteed equal-width or equal-exposure intervals. Tied values can produce fewer bins than requested. The desktop numeric-bin count control accepts 2–200.

Explicit cuts must be finite, strictly increasing and without duplicates. For cuts `25, 35`, the ordinary intervals are:

```text
x < 25
25 <= x < 35
x >= 35
```

A value exactly on a cut enters the interval to its right. Missing values have their own treatment; do not confuse an explicit missing row with either open-ended numeric tail. Custom cuts define boundaries, not a list of categories.

The preview displays training-row counts using the proposed model bins. It updates automatically and fits the chart to the available width. It is a preview of the draft; applying changed settings is a separate step. Applied shared-bin changes can require refitting affected models and interaction stages.

### Design kinds

| Kind | Behaviour |
|---|---|
| `step` | One constant relativity per numeric interval; the ordinary numeric default |
| `linear` | A continuous piecewise-linear effect on the link scale, with a slope per band |
| `continuous` | A single slope over a clamped numeric range; no interior knots |
| `categorical` | A separate effect for retained levels, plus an Other bucket |

For example:

```python
project.design.variables["DriverAge"] = VariableDesign(
    kind="linear", knots=[25, 35, 45, 55, 65], clamp=[18, 85],
    monotone="decreasing", penalty_weight=1.0,
)
```

With a log link, a piecewise-linear effect on the link scale becomes an exponential curve on the response scale. It is not a staircase, and it is not a straight line in the plotted relativity. Lasso can set band slopes to zero, producing flat sections.

`clamp` controls the linear/continuous range; the effect is flat beyond it. For `continuous`, additional bin cuts do not add slopes to the main effect. Pair-table axes still use numeric bin settings separately.

Monotonicity applies to the numeric main-effect curve, not to the complete prediction after interactions. Categorical factors and interaction cells do not accept these main-effect monotonicity constraints.

`penalty_weight` multiplies that variable's L1 penalty. Zero removes its L1 penalty, but does not eliminate the ridge penalty when `l1_ratio < 1`.

Categorical levels are retained according to training frequency or fitting-weight share and applicable settings. The default minimum level share is 0.0025. Explicit `levels` can define them; the first is the reference level. Other collects pooled, unseen and null values. A real level named Other may require a distinct catch-all label.

## 11. Variables JSON

The desktop **Variables JSON** editor is a compact setup view. It is not the complete `Project.to_dict()` schema.

For a source with `Claims`, `Exposure`, `DriverAge`, `VehicleAge`, `Region`, `PolicyId` and `traintest`, a representative desktop setup is:

```json
{
  "renames": {},
  "assignments": {
    "target": "Claims",
    "weight": "Exposure",
    "exposure": null,
    "offset": null,
    "current_premium": null,
    "time": null,
    "split": "traintest"
  },
  "roles": {
    "predictor": ["DriverAge", "VehicleAge", "Region"],
    "id": ["PolicyId"],
    "unassigned": [],
    "ignore": []
  },
  "types": {"categorical": ["Region"]},
  "binning": {
    "default_bins": 20,
    "overrides": {
      "DriverAge": {"method": "quantile", "bins": 10},
      "VehicleAge": {"method": "cuts", "cuts": [0, 1, 2, 3, 4, 5]}
    }
  }
}
```

**Use the current editor's JSON as the template**, retaining any additional current fields such as split settings. Names in this compact editor refer to raw source columns, including when there is a rename. The canonical project design keys refer to the resulting prepared names.

A bin override may use:

- `{"method": "quantile", "bins": 10}`;
- `{"method": "cuts", "cuts": [25, 35, 45]}`;
- `{"method": "integer", "fallback_bins": 20}` for the supported integer-cut representation. Integer cuts fall back to automatic cuts if the range is unsuitable.

For canonical project Python, use `project.design.defaults.n_bins` and `project.design.variables[name]`. Do not add a top-level `binning` key to a full project JSON and expect `Project.from_dict` to apply it. The desktop's binning adapter supplies that extra compact view; the older workflow-only `variable_setup_json` function by itself does not emit it.

## 12. One-way feature selection

### What it answers

The question is: **does this variable, considered on its own, show more useful signal than its shuffled copies and random noise?**

It screens one candidate at a time. It is Boruta-style, but is not the classical random-forest Boruta algorithm, nor a formal significance test.

Candidates include predictor-role columns and, when enabled, unassigned raw-source columns. Target, weight, exposure, offset, current premium, split, ID and time columns are protected. Ignored columns are not candidates. Generated derived-only columns are not screened as raw-source candidates.

A count of three candidates means three eligible variables in this setup. It does **not** necessarily mean three unused variables: selected predictors are also included.

### What runs

For each candidate:

1. Keep training rows only.
2. Build its design using the configured bins/type/settings.
3. Make four separately shuffled copies and an independent uniform random numeric control.
4. Fit a GLM containing the candidate and its five controls; five-fold CV chooses the penalty.
5. Calculate repeated permutation importance on the training rows.
6. Compare the real variable with the strongest control, with a zero floor.

The shuffled copies use the candidate's design; the random control uses the default numeric bins. The importance is the increase in mean deviance after shuffling. It is not a regression coefficient, percentage premium contribution, p-value, or the importance in the final multivariable model.

CV selects the penalty. The final importance measurements use the full training fit; they are **not out-of-fold importance estimates**. Holdout data does not participate in either operation.

### Python usage

Call this on the **raw**, not already-prepared, data:

```python
from easy_glm.workflow.feature_selection import select_variables

selection_options = {
    "family": "poisson",
    "divide_target_by_weight": True,
    "include_unassigned": True,
    "n_alphas": 20,
    "repeats": 5,
    "seed": 42,
}
screening_project = project.copy()
screening_project.models = {}
screening_project.champion = None
screening_project.exploration = {}
selection = select_variables(screening_project, raw, **selection_options)
ranked = pl.DataFrame(selection["rows"]).sort(
    "importance", descending=True, nulls_last=True,
)
print(ranked.select("variable", "status", "importance", "threshold", "margin"))
```

For Tweedie, set `family="tweedie"` and `tweedie_power=1.5` or the chosen supported power. The target, weight and offset come from the screening project's roles, not an arbitrary selected model's overrides. Align those roles with the modelling problem before starting.

The function does not mutate your predictor roles. Review results, then make deliberate changes. Useful statuses are:

| Status | Interpretation |
|---|---|
| `signal` | Real importance exceeds the control threshold beyond numerical tolerance |
| `no_signal` | The screen did not demonstrate a stronger one-way effect |
| `skipped` | Unusable for the screen, for example constant or all-null training values |
| `failed` | A fit or calculation failed; inspect the reason rather than treating it as no signal |

A no-signal variable may matter through an interaction. Correlated predictors, sparse data, coarse bins, penalty choices and random variation affect interpretation. Conversely, a one-way signal may be redundant once other predictors are present. Use the screen to guide review, not as proof that a variable can never matter.

The desktop shows ranked importance against the control benchmark and provides search, outcome filtering and table ordering. Selecting or ignoring rows stages role changes; review and Apply commits the changes to the project. A cancelled or obsolete job must not replace the saved successful screening recipe.

### Preserve the screen in a generated Python workflow

The desktop stores this recipe automatically after a successful search. When building a project in Python, explicitly retain the original screening setup **before later role changes**:

```python
import copy

project.exploration["feature_selection"] = {
    "version": 1,
    "project": screening_project.to_dict(),
    "options": copy.deepcopy(selection_options),
    "result": copy.deepcopy(selection),
}
# Now make reviewed changes, for example:
# project.apply_role_change("Noise", "ignore")
```

This is a version-specific recipe schema, not a separate public fit method. Calling `select_variables` alone does not automatically save the recipe into `project`.

The generated workflow includes `RUN_FEATURE_SELECTION = True`, an editable switch, and writes a deliberate screening report to its output directory. It reruns the original candidate list and settings; it does not silently replace the final reviewed model predictors with a new automatic selection.

As of 0.472, the desktop screening worker exchanges data, progress and results through memory pipes. It no longer needs the temporary job files that caused access-denied errors on some machines. The plain Python `select_variables` function works in memory. Deliberately saving a report is a separate file operation.

## 13. Fitting the main effects

Use either an explicit alpha or CV to choose alpha:

```python
cfg.penalty = Penalty(cv=5, n_alphas=20, l1_ratio=1.0)
# Alternative for a deliberately fixed penalty:
# cfg.penalty = Penalty(alpha=0.01, cv=None, l1_ratio=1.0)
```

Alpha is penalty strength, not a probability threshold. Larger values usually shrink more terms. `l1_ratio=1` is lasso; between zero and one mixes lasso and ridge. Main GLM fitting permits ridge, but the one-way screening interface requires `l1_ratio > 0`.

`n_alphas` controls the regularisation path resolution. It is not the bin count, the number of CV folds, or the number of CatBoost trials.

In the workflow, design settings are shared across models while predictor membership and fitting settings belong to each model. To create a reduced challenger, copy the configuration and change its main-effect list deliberately; use the same prepared rows and split when evaluating it.

For lower-level control:

```python
from easy_glm import DesignSpec, fit_glm, to_rate_model

spec = DesignSpec.from_data(
    train, ["DriverAge", "Region"], n_bins=10,
    knots={"DriverAge": [25, 35, 45, 55, 65]},
    weight_col="Exposure",
)
fit = fit_glm(
    train, spec, "Claims", family="poisson",
    weight_col="Exposure", divide_target_by_weight=True,
    cv=5, n_alphas=20,
)
scorer = to_rate_model(fit, exposure_col="Exposure")
```

Here `train` must already be filtered. `fit_glm` requires an explicit `alpha` or `cv`; `EasyGLM.fit` supplies CV=5 if neither is given. Never manually score through a newly rebuilt design matrix when the supplied `fit.predict` or `RateModel.predict` can do it correctly.

## 14. Interactions fitted in order

### The modelling idea

Each stage fits a flexible correction using exactly two raw predictor columns. The baseline is the deployed prediction from the main effects plus all earlier interaction tables.

```text
Main GLM
  → fit DriverAge × Region against the main GLM baseline
  → turn that CatBoost correction into a bin-aligned table
  → fit VehicleAge × Region against main GLM + the first table
  → turn the second correction into another table
```

The next interaction uses the **table approximation** of earlier stages, not their original CatBoost predictions. The main-effect tables stay frozen. Pair corrections are allowed to learn remaining one-way effects; they are not constrained to be mathematically pure interactions or to have zero margins.

Main effects and pair features are evaluated separately. The pair learner's raw-column features do not replace the main encoders or accidentally refit them. A pair parent must have the predictor role, but need not be selected as a main effect in this model.

### Python configuration

Use imports from `workflow.project` for the pair configuration classes; they are not all re-exported by `easy_glm.workflow`:

```python
from easy_glm.workflow.project import PairStageConfig, PairSearchConfig

cfg.pair_method = "sequential_catboost"
cfg.interactions = []
cfg.pair_stages = [
    PairStageConfig(
        stage_id="driver_region",
        a="DriverAge", b="Region",
        min_weight_share=0.001,
        seed=42,
        search=PairSearchConfig(trials=8, prefix_trials=4),
    ),
    PairStageConfig(
        stage_id="vehicle_region",
        a="VehicleAge", b="Region",
        min_weight_share=0.001,
        seed=42,
        search=PairSearchConfig(trials=8, prefix_trials=4),
    ),
]
run = run_model(project, prepared, "Frequency", progress=print)
```

Use unique stable stage IDs; `main` is reserved. Reversed duplicates such as A×B and B×A are the same pair and cannot both be added.

**Important Python default:** constructing `PairStageConfig` without `search=PairSearchConfig(...)` uses its fixed candidate list. The desktop's automatic-tuning workflow supplies a search configuration. Do not assume every Python-created stage automatically invokes Optuna.

### Tuning and runtime

Optuna searches shallow CPU CatBoost settings. In 0.472 its automatic space is depth 2–5, iterations 40–160 in steps of 20, learning rate 0.03–0.15, and L2 leaf regularisation 0.1–20. Search defaults are eight trials and four prefix trials. The accepted ranges are 1–16 trials and 1–8 prefix trials, with prefix trials no greater than trials.

Five outer folds assess the current stage. Earlier prefixes use fold-local training and inner selection where required, rather than giving a validation row predictions from a model trained on that row. The bounded search also considers leaving the correction neutral. Selection evaluates the resulting table's validation loss, not just the original CatBoost teacher's loss.

Two input factors do not guarantee an instant fit: later stages require upstream fits inside validation. `prefix_trials` governs that earlier-stage search budget. It is not another user-facing interaction or a second final model to score.

Current preflight limits include eight stages, 10,000 cells per pair, an estimated peak-memory limit of 3 GiB, an upper workload bound of 5,000 teacher fits and a 900-second fitting budget. These are guards, not promises about peak process memory or wall-clock duration. Rows are not silently sampled to meet them. Reduce unnecessary stages, trial budgets or grid size deliberately if a guard refuses the fit.

### Supported families and Tweedie

Ordered CatBoost pair stages currently support **Poisson/log and Tweedie/log only**. Gamma, Gaussian, binomial and identity-link pair stages are not implemented in this released path. They may still be valid main-GLM problems; do not silently substitute a different family just to enable CatBoost.

Tweedie pair fitting uses the configured power. It may rescale large targets internally for numerical stability, restoring the correct prediction scale. The baseline is included exactly once.

### How the CatBoost fit becomes a table

The table uses the configured numeric cuts and categorical levels, including fallback/missing rows. The released conversion groups **training observations into these cells** and chooses a constant correction per cell to minimise the corresponding loss against the teacher's fitted means. Do not describe it as a generic 3D ICE sweep or an arithmetic average of relativities: that is not this implementation.

For a cell, let `b` be the upstream unit prediction, `t` the complete teacher unit prediction, `w` the fitting weight and `p` the Tweedie power. The multiplier is:

```text
sum(w × t × b^(1-p)) / sum(w × b^(2-p))
```

For Poisson (`p=1`) this reduces to `sum(w × t) / sum(w × b)`. Empty or insufficiently supported cells remain neutral at 1.0, with a reason recorded. The default minimum share 0.001 means 0.1% of fitting weight, not 0.1% of observed losses.

This intentionally loses some flexibility compared with raw CatBoost predictions. The deployed table is the model used for subsequent stages, diagnostics, scoring and exports.

### Reviewing and changing stages

Use the interaction heatmap/matrix to view the correction. Values are displayed to at most three decimal places for readability; the saved scorer retains underlying precision. A correction of 1.10 means a 10% multiplier relative to the upstream prediction, not a 10% absolute claim probability.

Read the supported-cell count, validation loss before/after and table approximation loss together. “No improvement” can be a valid result: the neutral candidate was preferred. Do not force a non-neutral table to make a stage look successful.

Changing stage order changes the modelling problem. Changing upstream settings, bins, main effects or applied tables may invalidate downstream stages. The workbench can reuse a compatible prefix and fit remaining stages; do not suppress its stale/refit indicators. A saved export made before the refit is not evidence that the changed model is current.

For diagnostics use `run.predict(...)` or `run.rate_model`, not `run.fit.predict(...)`. With sequential stages, `run.fit` represents the main GLM. Likewise, `run.alpha_stage2` and legacy `cells_kept` do not summarise the CatBoost stages; inspect `run.pair_stages` and `run.rate_model.pair_tables`.

## 15. Legacy GLM interactions

Existing legacy models can use `Interaction(a=..., b=...)` in `cfg.interactions`, with `pair_method="legacy_glm"`.

The main GLM is fitted first and frozen. A second GLM fits the selected binned interaction cells with the main prediction as offset and no extra intercept. Multiple legacy interactions belong to that second fit; this is different from fitting each CatBoost pair in sequence.

Do not mix `cfg.interactions` and `cfg.pair_stages` in the same model. Do not replace an existing legacy model's method silently. Legacy interaction parents need their main-effect design; sequential pair stages have separate axes and can use a predictor that is absent from the selected main effects.

A cell multiplier is a correction, but not proof of a pure interaction with no one-way component or overall level shift. Main tables and base rate staying frozen does not make the complete fitted total invariant.

## 16. Diagnostics and missing-factor searches

### Begin with the complete model

```python
from easy_glm.workflow import totals, ae_by_variable

actual, expected, weights = totals(holdout, run.config, run.predict(holdout))
region_ae = ae_by_variable(holdout, "Region", actual, expected, weights)
print(region_ae)
```

This includes ordered interaction tables and current applied adjustments. An offset must be present and correctly prepared. Do not use the main GLM alone for a supposedly complete-model diagnostic.

Overall A/E is `sum(actual) / sum(expected)`, not the average of row-level ratios. A/E above 1 indicates underprediction; below 1 indicates overprediction. A zero expected total makes the ratio undefined and should be investigated.

Useful views include:

- **A/E by variable:** check calibration across bands or levels and inspect exposure as well as ratios.
- **Time stability:** inspect calibration across the time column; it does not itself change the train/holdout split.
- **A/E by pair:** look for systematic residual structure across two variables.
- **Lift and Gini:** assess ordering/discrimination. The displayed normalised Gini is exposure-weighted and is not ROC AUC.
- **Variable importance:** review the deviance increase after shuffling. Establish whether the view is for the original fit or the current adjusted scorer.
- **Regularisation path and coefficients:** inspect shrinkage and retained terms. These are GLM diagnostics, not a CatBoost coefficient decomposition.

Coefficient magnitudes depend on encoding, scaling and penalty structure. Do not rank raw coefficients as if they were comparable variable importances. Permutation-importance error bars show repeat variability, not automatically a confidence interval or a hypothesis test.

### Missing variables and interactions

These searches are different from the pre-fit one-way screen. They ask what structure remains after the **current complete model**.

The desktop residual searches use training rows and `run.predict(train)`. Existing unordered pairs in either legacy interactions or ordered pair stages are excluded from the missing-interaction candidates. A fitted pair can still have residual misfit; that does not mean it should be re-added as a duplicate stage.

The factor search ranks excess residual variation by candidate. The pair search first removes marginal main-effect misfit inside its candidate grid so a bad one-way shape is not automatically called an interaction. Its screening grid can be coarser than the model's final rate-table grid.

Low-level Python helpers accept arrays rather than a model. **The caller is responsible for passing full-model predictions, using training data and filtering the candidate list.** They do not discover your applied pair stages automatically:

```python
from itertools import combinations
from easy_glm.workflow import residual_pair_search, pearson_dispersion

actual, expected, weights = totals(train, run.config, run.predict(train))
existing = {frozenset((item.a, item.b)) for item in run.config.interactions}
existing.update(frozenset((item.a, item.b)) for item in run.config.pair_stages)
pairs = [
    pair for pair in combinations(run.config.predictors, 2)
    if frozenset(pair) not in existing
]
phi = (
    1.0 if run.config.family == "poisson"
    else pearson_dispersion(actual, expected, len(run.fit.coef))
)
if pairs:
    suggestions = residual_pair_search(
        train, run.config.predictors, actual, expected, weights,
        pairs=pairs, dispersion=phi,
    )
    print(suggestions)
```

For non-Poisson outcomes, dispersion scaling matters. Treat these scores as prioritisation aids rather than calibrated significance probabilities. Inspect exposure/support, fit a proposed change, and compare holdout results before accepting it.

## 17. Champion and challenger comparison

The Compare page places the champion on the left and challenger on the right. Compare two fitted models on the same rows, response units and applicable offsets. Confirm which applied table adjustments are included.

Useful comparisons include overall and segment A/E, deviance, Gini, factor inclusion, base rate and changed relativities. A relativity difference alone is not the total premium change: a base-rate change also affects the result.

Double lift groups rows in increasing order of champion/challenger predicted ratio using equal-exposure bands. Each line shows **actual divided by that model's predicted total** within the band. A line near 1 is calibrated there.

Bin numbers are ranks/groups, not values of the prediction ratio. “Band 10” does not mean the champion predicts ten times the challenger. Read any displayed ratio range or percentile/rank label accordingly. The exposure bars use a separate axis.

A challenger with improved training fit but worse holdout calibration may be overfitting. Small differences should be assessed against exposure, stability and business use, not only a single headline metric.

## 18. Rate-table adjustments and rebalancing

Use the rate-table editor to inspect the fitted effect and preview changes before applying them. Tools can smooth, cap/floor or round supported factor tables. Review the changed expected total as well as the curve.

Two ways to edit from Python serve different purposes:

1. For a standalone scorer, clone it and call its supported update methods.
2. For a `Project` workflow, store `Adjustment` objects and rebuild so the reviewed decisions are retained in project/export state.

Example for an ordinary step factor, continuing from the project example:

```python
from easy_glm.workflow import Adjustment, rebuild_rate_model, rebalance_override

cfg = project.models[run.name]
row = next(r for r in run.tables["DriverAge"].to_dicts() if r["from"] is not None)
cfg.adjustments.append(Adjustment(
    variable="DriverAge",
    from_=row["from"],
    to_=row["to"],
    relativity=float(row["relativity"]) * 1.05,
))
rebuild_rate_model(project, run, prepared)
# Optional: restore the original fitted training expected total after review.
cfg.base_rate_override = rebalance_override(project, run, prepared)
rebuild_rate_model(project, run, prepared)
```

Here `relativity=` is an **absolute replacement value**, not a change of `+0.05`. Derive the exact interval keys from the fitted table; do not invent rounded boundaries.

This simple example is for a model without downstream pair dependencies. Main-table or earlier pair edits can make later fitted interactions stale. Rebuilding tables alone does not retrain those downstream stages. Follow the workbench's refit flow or a deliberate project refit/replay before describing the whole model as current.

`rebalance_override` restores the original fitted **training expected total**. `solve_base_rate(run, frame, target_ratio)` solves for an actual/expected ratio on the supplied rows. These are not the same target. For a log-link premium model, a target loss ratio of 0.65 means expected premium should be actual loss divided by 0.65. The solver returns a value; assign it to `cfg.base_rate_override` and rebuild to apply it.

Smoothing can preserve a weighted mean of log relativities without preserving total expected claims or premium. Capping and rounding can also change the total. Never assume a shape adjustment is financially neutral.

Use snapshots for named reviewed adjustment states. The workbench records adjustments and the base-rate override; it does not preserve an independent new fitted model for each snapshot. For ordered pair models, compatibility with the fitted prefix matters when restoring an old state.

For standalone scoring edits, `RateModel.clone()` creates an independent copy. Avoid modifying internal table lists directly because scoring caches must also be updated. Pair-stage edits have stable stage/cell identities; do not address them through the legacy `A×B` table-edit convention by guessing a name.

## 19. Scoring new data

```python
from easy_glm import RateModel

scorer = RateModel.from_json("frequency.easyglm")
unit_prediction = scorer.predict(new_prepared_data, exposure_col=None)
row_prediction = scorer.predict(new_prepared_data)  # Uses saved exposure setting.
```

A table scorer does not replay all `Project` preparation steps. Supply the required prepared names, recoded categories, derived predictors and offsets. It does not need targets or holdout flags merely to score, unless a required predictor or preparation step depends on them.

Do not call full training preparation on a production scoring frame blindly: it may try to generate a split or filter target-dependent rows. Build and verify the applicable preparation steps explicitly. A column rename map is not a substitute for recoding or derivation.

For `column_map`, mapping direction is **incoming data name → model name**, for example `{"driver_age": "DriverAge"}`.

Unknown categorical levels generally use Other; check warnings about large unmatched shares, particularly integer-versus-float-versus-string category codes. An unseen `4.0` string is not necessarily the trained `"4"` level. Numeric tails follow the open-ended bins or clamped curve. Missing numeric behaviour depends on the fitted null handling.

A missing offset or expected exposure column can produce a warning and predictions that omit that factor. Treat that as a scoring-contract failure in a production workflow, not as harmless console noise. Check required columns and prediction units explicitly.

To verify an export:

```python
import numpy as np

np.testing.assert_allclose(
    scorer.predict(holdout, exposure_col=None),
    run.predict(holdout),
    rtol=1e-10, atol=1e-12,
)
```

Use the scorer exported from that same run. Check representative missing values, category fallbacks and exact cut boundaries as well as normal rows. Compare predictions with tolerances rather than requiring repeated numerical fits or timestamped bundles to have identical bytes.

## 20. Saving, exporting and reproducing a model

Choose the artifact according to its purpose:

| Artifact | Preserves | Does not automatically do |
|---|---|---|
| Project JSON | Preparation, roles, bins, models, stages, adjustments and stored screening recipe | Embed source data or guarantee a fitted desktop session is restored |
| `.easyglm` JSON | Current scoring tables, metadata, base rate and ordered pair tables | Refit or prepare arbitrary raw input |
| Training Python script | The modelling workflow and its configuration | Guarantee identical new fits on changed data |
| Frozen scoring Python | Exact saved table scorer | Train, tune CatBoost, or repeat full source preparation |
| Excel rate tables | Reviewable current rates and pair-table outputs | Replace the verified Python scoring contract merely by copying rounded cells |
| HTML report | Model diagnostics, tables and supporting summaries in one file | Act as a live model or refit itself |
| `EasyGLM.save(...)` bundle | The wrapper's fitted design/estimators/scorer/tables | Serve as a project JSON or interchangeable `.easyglm` JSON |

A workflow export example:

```python
from pathlib import Path
from easy_glm.workflow import to_script, to_scoring_script, to_report_html

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
project.to_json(out / "project.json")
run.rate_model.to_json(out / "model.easyglm")
run.rate_model.to_excel(out / "rate_tables.xlsx")
(out / "training_workflow.py").write_text(
    to_script(project, run.name, run=run, output_prefix="refitted"), encoding="utf-8",
)
(out / "frozen_scoring.py").write_text(
    to_scoring_script(run, output_prefix="frozen"), encoding="utf-8",
)
(out / "report.html").write_text(
    to_report_html(project, {run.name: run}, prepared, champion=run.name),
    encoding="utf-8",
)
```

The training script needs a usable source path. Save/upload-only or in-memory data to a suitable file and set the source before expecting a portable replay. Paths copied from another machine must be adapted.

For ordinary GLM exports, supplying `run=` records the learned design and resolved penalty; without a run the script derives the design and runs configured CV. For sequential CatBoost pair models, the training script deliberately replays the main GLM and ordered pair fitting/tuning even when `run=` is provided. Use frozen scoring or `.easyglm` for exact saved deployed tables.

If a feature-selection recipe was saved, the training script can repeat it, using the original screening candidates and bin settings. Final reviewed predictors remain explicit. The recipe does not authorize the script to drop a final predictor automatically.

`EasyGLM.to_excel(...)` describes the fitted GLM tables. Use the current `rate_model.to_excel(...)` when applied adjustments or pair tables are the desired output.

Frozen scoring does not require CatBoost/Optuna, but it still uses EasyGLM and its normal dependencies. “Standalone script” here does not mean a dependency-free implementation of every scoring rule.

## 21. Using the desktop workbench

Launch from a terminal:

```bash
python -m easy_glm.app
python -m easy_glm.app project.json --port 8501
python -m easy_glm.app project.json --port 8501 --headless
```

Or from Python:

```python
from easy_glm import launch_workbench

process = launch_workbench("project.json", port=8501)
# Later, when finished:
# process.terminate()
# process.wait()
```

`launch_workbench(data=raw, port=8501)` can launch from a pandas or Polars frame, but currently persists input for the separate workbench process. It is not a promise of disk-free launch. `headless=True` suppresses opening the browser; it does not turn the GUI launcher into a modelling pipeline.

The current workbench binds to loopback (`localhost`/`127.0.0.1`). It is a local application, not an automatically published multi-user web service.

### Page-by-page flow

1. **Project & data:** load or configure the source and inspect row/column counts. Keep a record of data version and units.
2. **Variables:** assign roles and types. Set numeric-bin defaults/overrides. Define the training split. Preview and Apply changes. Optionally check predictors and run one-way selection; review its staged role changes before applying them.
3. **Explore:** inspect distributions, missingness and relationships. An exploration sample does not mean model fitting or feature selection will automatically sample the data.
4. **Model:** choose/create the named model, target, family, weight, offset and target-division setting. Select main effects in Factor design. Add interactions in the intended order. Save settings, then fit or fit remaining stages.
5. **Diagnostics:** inspect the selected fitted model and holdout calibration. Residual searches use training data; inspect suggestions and add a reviewed interaction back on Model.
6. **Compare:** choose champion and challenger side by side. Compare like-for-like rows and applied tables.
7. **Rate tables:** inspect plots and interaction matrices. Preview/apply adjustments, inspect their total effect, rebalance when appropriate and manage snapshots.
8. **Export:** retain project JSON, approved scoring tables, the desired Python export and any Excel/report artifacts.

Applied edits in the Svelte workbench belong to the running process. **Export project JSON to keep them.** Opening an existing project loads a copy; do not assume it autosaves back over the original file. A browser refresh during a live process is different from restarting the server. Legacy Streamlit persistence/autosave instructions do not establish what the current workbench saves.

Draft, applied configuration and fitted results are separate states. A successful preview does not apply an edit. Saving settings does not fit them. A fitted result may become stale after an upstream change. Resolve that state before exporting something described as the final model.

## 22. Command-line usage

```bash
easy-glm validate project.json
easy-glm run project.json --model Frequency --out outputs
easy-glm export project.json --model Frequency --out outputs --script
easy-glm export project.json --model Frequency --out outputs --report --excel
easy-glm workbench project.json
```

Module form, useful if the console script is not on PATH:

```bash
python -m easy_glm.cli validate project.json
python -m easy_glm.cli run project.json --model Frequency --out outputs
```

`validate` checks preparation and project/model consistency; it is not a successful fit. Artifact-producing `run`/`export` commands fit afresh, including `export --script`. Do not expect them to reuse the desktop's in-memory fitted result.

Specify `--model` when the project contains several models. A `.easyglm` file is a scorer, not a project, and is refused where project JSON is required. Outputs are named using a safe project/model prefix under `--out`.

Exit codes are 0 for success, 1 for an actionable project/data/fit failure, and 2 for usage errors. Use `--help` on the installed version for the exact command options.

## 23. Performance and troubleshooting

### Performance

Use modest model complexity to establish a working baseline before expanding the analysis. Fine bins multiplied across a pair create many cells, often with little support. Inspect actual bin counts rather than assuming the requested count was attained.

Screening 428 candidates means 428 candidate GLMs, each with five controls, a five-fold penalty path and repeated importance scoring. It is not a single quick global fit. Unassigned columns can enlarge that workload substantially.

The GLM uses float64. Large designs can use a compact representation; scoring uses table lookups rather than building a full design matrix. Do not suggest converting the model design to float32 as a casual memory optimisation.

The screen has design guards, including 1,024 feature columns and an estimated compact design size of 512 MiB per candidate. Pair stages have additional workload/grid/memory guards. These are distinct from the workbench's exploration sample setting.

Repeated comparable fits may reuse compatible main/pair prefixes in the workbench. A change to data, bins, order, fitting settings or upstream tables can invalidate that reuse. Never force reuse merely because two models have similar display names.

### Troubleshooting table

| Symptom | Check and response |
|---|---|
| Upgrade appears not to work | Inspect `sys.executable`, installed version and `easy_glm.__file__`; restart the kernel/workbench process after upgrading |
| Feature selection writes `easyglm_selection_.../progress.tmp` | That is the old file transport; verify the running process really uses 0.472 or later, not only that a different environment was upgraded |
| Other access-denied errors remain | Identify the exact operation and path. Launch logs, in-memory-frame handoff, other background workers, caches and deliberate exports may still use files; 0.472 changed feature selection specifically |
| Permission error on a report or export | Choose a writable, approved output folder. Explicit exports are intended file writes |
| Workbench port is occupied | Stop the intended old process or choose another `--port`; do not terminate unrelated processes blindly |
| Predictors are missing on Model | Check applied versus draft roles, the selected model's main-effect list, search filters and dropped constant/all-null columns |
| Candidate count seems too low | Inspect predictor plus optional unassigned roles and protected fields; derived-only columns are not raw-source candidates |
| Candidate count seems too high | Unassigned source columns are included when that option is enabled; assign ID/time/ignore roles deliberately |
| All or many screen rows failed | Read their reasons; check target/weight/offset validity, convergence and design size. Do not label failures no-signal |
| Unexpected claim totals | Check whether the target was already a rate, whether division is enabled and whether exposure was multiplied twice |
| Predictions suddenly differ after export | Compare the same prepared rows, current adjustments, offsets, exposure convention and full pair-table scorer; distinguish training replay from frozen scoring |
| Many categoricals fall into Other | Compare source type and labels, especially numeric codes rendered as strings/floats |
| Too few bins | Tied values, low variation and actual training range can reduce automatic cuts; inspect the preview |
| Unsupported pair family/link | CatBoost stages currently require Poisson/log or Tweedie/log; do not silently change the actuarial target specification |
| Missing CatBoost/Optuna | Reinstall `easy-glm` in the active environment; both are standard dependencies |
| Pair stage says No improvement | The neutral correction may have won on table CV loss; inspect evidence rather than force an effect |
| Pair stages need refitting after an edit | Earlier tables are downstream offsets; update downstream fits before accepting/exporting the changed pipeline |
| Gini and A/E tell different stories | Gini assesses ranking; A/E assesses level/calibration. One does not replace the other |
| Repeated fits differ slightly | Compare predictions with numerical tolerance; refitting/tuning and stored timestamps need not give identical bytes |

For filesystem problems, ask for the exact error and version first. Do not prescribe moving the entire modelling workflow to a new temp directory when the failing operation can be identified more precisely.

## 24. API reference and common mistakes

### Import map

| Operation | Import |
|---|---|
| Simple fit / split / scorer | `from easy_glm import EasyGLM, add_train_test_split, RateModel` |
| Workbench | `from easy_glm import launch_workbench` |
| Preparation and model workflow | `from easy_glm.workflow import Project, prepare, run_model, train_holdout` |
| Ordinary configuration classes | `from easy_glm.workflow import DataSource, Split, VariableDesign, Penalty, Adjustment` |
| CatBoost stage configuration | `from easy_glm.workflow.project import PairStageConfig, PairSearchConfig` |
| One-way screen | `from easy_glm.workflow.feature_selection import select_variables` |
| Diagnostics | `from easy_glm.workflow import totals, ae_by_variable, ae_by_pair, residual_factor_search, residual_pair_search` |
| Exports | `from easy_glm.workflow import to_script, to_scoring_script, to_report_html` |
| Design and direct fitting | `from easy_glm import DesignSpec, StepEncoder, LinearEncoder, CategoricalEncoder, fit_glm, fit_two_stage, to_rate_model` |

For direct constructors with many parameters, inspect the installed signature instead of deriving arguments from a GUI label. Workflow functions are a better Python integration point than private desktop HTTP endpoints or underscore-prefixed helpers.

### Mistakes an LLM must avoid

- Calling the package's screen classical Boruta or claiming it proves a variable has no effect.
- Saying CV importance was measured out of fold when it was measured on the final training fit.
- Passing prepared data into a function that prepares raw data again.
- Treating predictor eligibility and model main-effect inclusion as identical.
- Writing compact Variables JSON into the full project schema.
- Omitting `search=PairSearchConfig(...)` while claiming a Python-created pair uses Optuna.
- Fitting every interaction independently against the same main-only offset.
- Feeding teacher predictions to later stages when deployed tables are required.
- Calling `run.fit.predict` a full prediction for a sequential pair model.
- Claiming each pair is constrained to have no one-way effect.
- Claiming pair distillation is a generic ICE-grid averaging procedure.
- Describing Gaussian's package default as identity.
- Treating logit relativities as probability multipliers.
- Interpreting double-lift band numbers as actual prediction ratios.
- Rebalancing against the holdout merely to improve the validation display.
- Applying a direct table edit without updating scorer caches or project adjustment records.
- Assuming a rebuilt scorer has refitted stale downstream interactions.
- Exporting only rounded Excel cells and assuming exact scoring parity.
- Saying a training script is a frozen scorer or that `.easyglm` is a project JSON.
- Promising the whole workbench is disk-free because feature selection now is.
- Claiming company data, assumptions or model results were reviewed when only a synthetic example was tested.

## 25. Useful prompts and handover template

### Start a modelling task

> Use the attached EasyGLM guide. I have [row meaning] with target [column and units], weight [column and meaning], and offset [column and whether logged]. My objective is [frequency/severity/pure premium/rate change]. First check the units and split, then give me a minimal Project-based workflow. Preserve my existing cuts and explain each non-obvious setting.

### Review a screen

> Explain these one-way selection results in terms of importance, control threshold and margin. Distinguish no-signal, skipped and failed. Do not automatically drop variables. Tell me which issues need a better one-way design and which may still justify interaction testing.

### Review interactions

> Confirm that each interaction is fitted against the main effects plus all previous deployed interaction tables. Show me the stage order, upstream baseline, selected table, support and CV change. Use the final table model for diagnostics, and identify any stage that needs refitting.

### Debug scoring differences

> Compare these two prediction paths on the same rows. Check preparation, target division, exposure multiplication, offset scale, manual adjustments and ordered pair tables before changing model settings. Show an assertion that should pass after the issue is resolved.

### Reproduce a workbench model in Python

> Use my exported Project JSON and the source schema. Preserve its roles, split, numeric bins, screening recipe, final predictors, model order and adjustments. Explain which export refits and which preserves the frozen scorer. Do not overwrite the existing artifacts.

### Compact model handover

Keep this next to the project and scoring artifact:

```text
EasyGLM version:
Python executable/version:
Source file and data version:
One row represents:
Target and units:
Weight and meaning:
Divide target by weight:
Offset column and scale:
Scoring exposure setting:
Training/holdout definition and counts:
Family, link and Tweedie power if applicable:
Default bins and per-variable overrides:
Selected main effects:
Screening recipe and reviewed decisions:
Ordered interaction stages:
Applied table adjustments and base-rate override:
Fit freshness / stages requiring refit:
Training and holdout metrics:
Scoring input preparation:
Approved scorer path:
Training replay path:
Project JSON path:
Export parity check performed:
```

## 26. Verification and source references

This guide was checked against release `v0.472`. The companion uses synthetic data; it is not a validation of any work dataset or business assumption.

The companion checks:

- a direct GLM's unit predictions versus exposure-scaled table predictions;
- the project preparation and model fit;
- training-only screening and preservation of its original recipe;
- a reviewed table adjustment and restoration of the fitted training total;
- optional Optuna/CatBoost pair training;
- `.easyglm` and frozen-script scoring parity;
- creation of the training script, Excel tables and HTML report.

During preparation of this guide, both generated training scripts (ordinary GLM and CatBoost pair workflow) were also executed. Their exported predictions matched the original examples within numerical tolerance, and both replays retained all four original screening candidates, including the variable later marked Ignore. All 23 Python snippets were syntax-checked, and the main fitting, screening, diagnostics, adjustment and export examples were executed.

Use these version-pinned source references when verifying a detail. Source signatures and the relevant implementation take precedence over historical planning documents:

- [Installation metadata](https://github.com/serband/easy_glm/blob/v0.472/pyproject.toml)
- [EasyGLM public wrapper](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/core/easyglm.py)
- [Project configuration and validation](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/workflow/project.py)
- [Preparation and splitting](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/workflow/prep.py)
- [Desktop bin settings](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/desktop/binning.py)
- [One-way screening](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/workflow/feature_selection.py)
- [Model runs and current scoring](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/workflow/run.py)
- [Ordered pair fitting and tuning](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/workflow/pair_stages.py)
- [Loss-based table conversion](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/workflow/pair_distillation.py)
- [Diagnostic helpers](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/workflow/diagnostics.py)
- [Desktop full-model residual searches](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/desktop/review_worker.py)
- [Training and scoring exports](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/workflow/export.py)
- [Portable scorer](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/engine/rate_model.py)
- [Desktop launcher and session behaviour](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/desktop/__init__.py)
- [Command-line implementation](https://github.com/serband/easy_glm/blob/v0.472/src/easy_glm/cli.py)

When using a later version, recheck these contracts before extending the guide with new claims.
