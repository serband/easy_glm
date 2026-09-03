# easy_glm

**LASSO GLMs → rate tables → calibrate in a browser → score portfolios.**

Python toolkit for insurance pricing: fit a regularised GLM on your data, read
exact per-variable relativities and a base rate straight off the
coefficients, adjust them in an interactive editor (with A/E charts), and
ship a portable `.easyglm` model for scoring. Built on
[glum](https://glum.readthedocs.io/en/latest/); inspired by R
[aglm](https://github.com/kkondo1981/aglm).

There are no references to any private pricing model here — everything below
is a variable going into an ordinary GLM. Every numeric variable defaults to
a **step (banded) factor**; you choose a smooth curve or a monotone shape
explicitly when you want one.

**Every code block on this page is tested.** `tests/test_readme.py` extracts
every ```python block below, in order, and runs it in one process; if a block
here does not run, the test suite is red. See
[`docs/checks/readme-gate.md`](docs/checks/readme-gate.md) for the last
result.

This page walks the whole workflow in order — install, load data, design a
model, fit it, read and adjust the tables, score exactly, export to Excel and
to a Python script, check the diagnostics, price a rate change and a lapse
model, fit a book of millions of rows, and open the same model in the
browser workbench or from the command line. Every block is complete and
copy-pasteable in the order shown.

## Contents

1. [Install](#1-install)
2. [Load the data](#2-load-the-data)
3. [The quick way: `EasyGLM.fit`](#3-the-quick-way-easyglmfit)
4. [Taking control: design the model yourself](#4-taking-control-design-the-model-yourself)
5. [Rate tables](#5-rate-tables)
6. [Interactions: mains frozen, cells on top](#6-interactions-mains-frozen-cells-on-top)
7. [Adjustments and rebalancing](#7-adjustments-and-rebalancing)
8. [Exact scoring, `.easyglm`, and the one-line invariant](#8-exact-scoring-easyglm-and-the-one-line-invariant)
9. [Excel export](#9-excel-export)
10. [Books of millions of rows](#10-books-of-millions-of-rows)
11. [A project file, and the exported Python script](#11-a-project-file-and-the-exported-python-script)
12. [Diagnostics: lift, Gini, A/E, double lift](#12-diagnostics-lift-gini-ae-double-lift)
13. [Rate change: pricing the move from today's premium](#13-rate-change-pricing-the-move-from-todays-premium)
14. [A lapse model (binomial)](#14-a-lapse-model-binomial)
15. [The workbench](#15-the-workbench)
16. [The command line](#16-the-command-line)
17. [The project file and the script, round-tripped](#17-the-project-file-and-the-script-round-tripped)

---

## 1. Install

```bash
pip install easy_glm
# the browser workbench and relativity editor need Streamlit + Plotly:
pip install "easy_glm[ui]"
```

Python 3.10–3.13. `[viz]` adds matplotlib/seaborn/plotnine charts;
`[benchmark]` adds the statsmodels/CatBoost comparison in
`easy_glm.benchmarking`; `[dev]` is everything needed to run this repository's
own test suite.

## 2. Load the data

Every example on this page runs on
[`tests/fixtures/french_motor_50k.parquet`](tests/fixtures/french_motor_50k.parquet):
a 50,000-row sample of the public French motor third-party liability dataset
(`freMTPL2freq`, distributed by the [CASdatasets](http://cas.uqam.ca/)
project). It is checked into the repository, so everything below runs
offline — no download, no network access. `easy_glm.load_external_dataframe()`
downloads the full ~678,000-row dataset the same way if you want more rows to
play with; `tests/fixtures/make_french_motor_50k.py` is the exact recipe that
produced the sample.

```python
import numpy as np
import polars as pl

DATA = "tests/fixtures/french_motor_50k.parquet"
df = pl.read_parquet(DATA)
print(df.shape)
# → (50000, 12)

# A train/holdout split column: 1 = train (fitting), 0 = holdout (validation).
rng = np.random.default_rng(42)
df = df.with_columns(
    pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64)
)
```

## 3. The quick way: `EasyGLM.fit`

Most of the time you only need `EasyGLM.fit`: it builds a design (step knots
for numeric factors, one-hot with an `Other` bucket for categoricals), fits
an L1-penalised (lasso) GLM with [glum](https://glum.readthedocs.io/en/latest/),
and reads exact rate tables and a calibrated base rate straight off the
coefficients.

```python
import easy_glm

PREDICTORS = ["DrivAge", "Region", "BonusMalus", "Density"]

eglm = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=PREDICTORS,
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,   # frequency = ClaimNb / Exposure
    alpha=0.001,                    # a fixed penalty — see the note below
    monotone={"BonusMalus": "increasing"},   # relativity never decreases with BonusMalus
)
print(eglm)
# → EasyGLM(model_type='Poisson', target='ClaimNb',
#           predictors=['DrivAge', 'Region', 'BonusMalus', 'Density'],
#           alpha=0.001, base_rate=0.0418651)

holdout = df.filter(pl.col("traintest") == 0)
ae = holdout["ClaimNb"].sum() / eglm.rate_model.predict(holdout).sum()
print(f"Holdout A/E: {ae:.4f}")
# → Holdout A/E: 0.9943
```

**Fixed alpha vs. cross-validation.** `alpha=0.001` above is a fixed penalty
strength, chosen so this page fits in well under a second. Pass `cv=5`
instead (the default when neither is given) to pick the penalty by 5-fold
cross-validation over a 20-point path — the more careful choice for a model
you are about to rely on, at the cost of roughly 10–20 seconds on this 50k-row
fixture rather than a fraction of a second. Nothing else about the call
changes.

Families are `"Poisson"`, `"Gamma"`, `"Tweedie"`, `"Gaussian"` (all a log
link) and `"Binomial"` (a logit link — [§14](#14-a-lapse-model-binomial)).

## 4. Taking control: design the model yourself

`EasyGLM.fit` is three calls you can make yourself, which is what lets you
mix step, categorical, linear and monotone terms in one design and inspect
every stage:

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `DesignSpec.from_data(train, predictors)` | Quantile knots per numeric factor, frequency-ordered levels per categorical — **train rows only** |
| 2 | `fit_glm(train, spec, target, ...)` | L1/elastic-net glum fit; `alpha=` or `cv=`; `monotone=` |
| 3 | `rate_tables(fit)` / `to_rate_model(fit)` | Exact relativities and base rate, read off the coefficients |

```python
from easy_glm import DesignSpec, fit_glm, rate_tables, to_rate_model

train_df = df.filter(pl.col("traintest") == 1)

spec = DesignSpec.from_data(
    train_df,
    PREDICTORS,
    weight_col="Exposure",   # level frequencies and quantile knots weighted by exposure
    linear=["Density"],      # a piecewise-linear (smooth) curve instead of steps
)
print(spec)
# → DesignSpec(DrivAge: step(19 knots), Region: categorical(21 levels),
#              BonusMalus: step(8 knots), Density: linear(19 knots, clamp 0–27000))
```

* **`DrivAge` and `BonusMalus` default to step (banded) factors** — one
  relativity per band, changing only at a knot. This is the default for
  every numeric variable; you opt into anything else.
* **`Region` is categorical** because it is not numeric: one relativity per
  level, with an `Other` bucket for rare or unseen values.
* **`Density` is `linear`** (piecewise-linear, one slope per band on the log
  scale) via the `linear=[...]` argument — a smooth curve instead of steps.
  `kind="continuous"` (a `DesignSpec.from_data(..., linear=[...], knots={...:
  []})`-style single band, no interior knots) is the same encoder with no
  knots at all: one straight line on the raw clamped value, sharing the same
  rate table, editor, Excel sheet and exported script. The lasso penalises
  each band's **slope**, not its bend, so a stretch the data does not argue
  about comes back exactly flat.
* **`monotone={"BonusMalus": "increasing"}`** (next step) is a sign
  constraint on the step increments (or, for a linear/continuous term, on the
  band slopes): the relativity is never allowed to decrease as `BonusMalus`
  rises. Categorical and interaction terms cannot be constrained — a
  constraint binds a factor's own curve, never an adjustment on top of it.

```python
fit = fit_glm(
    train_df, spec, target="ClaimNb", family="poisson",
    weight_col="Exposure", divide_target_by_weight=True,
    alpha=0.001,                              # fixed, for speed — cv=5 also works here
    monotone={"BonusMalus": "increasing"},
)
print(fit)
# → GLMFit(family='poisson', link='log', target='ClaimNb', alpha=0.001,
#          features=71, non_zero=38, variables=['DrivAge', 'Region', 'BonusMalus', 'Density'])
```

## 5. Rate tables

```python
tables = rate_tables(fit)
print(tables["BonusMalus"])
# → shape: (10, 7) — from / to / label / coef / relativity / exposure / is_base
#   relativity climbs from 1.00 (< 53, the most-exposed band) to 4.30 (≥ 95),
#   never decreasing — the monotone constraint above holding exactly.

print(fit.coef_table(drop_zero=True).height, "non-zero terms")
# → 39 non-zero terms
```

Relativity 1.00 sits on the most-exposed band of each factor by default
(`base="modal"`; pass `base="reference"` for the lowest band or reference
level instead). The last row of every table is `Other / Unknown` — the bucket
for nulls and, for a categorical, unseen levels. Optional matplotlib/seaborn
charts: `easy_glm.plot_all_ratetables(tables)` (needs `pip install
"easy_glm[viz]"`).

## 6. Interactions: mains frozen, cells on top

Adding `A × B` **never moves a main-effect table or the base rate**. A model
with an interaction is fitted in two stages: stage 1 is the main-effect
model — bit for bit the fit the same model gives with no interaction at all —
and stage 2 fits the interaction's cells on top of it, offset by stage 1's
own linear predictor and with no intercept of its own. Every cell is
therefore a pure adjustment: 1.00 means "no adjustment", nothing else.

```python
from easy_glm import fit_two_stage, base_rate

spec_int = DesignSpec.from_data(
    train_df, PREDICTORS, weight_col="Exposure", linear=["Density"],
    interactions=[("DrivAge", "BonusMalus")],
)
two_stage_fit = fit_two_stage(
    train_df, spec_int, "ClaimNb", family="poisson",
    weight_col="Exposure", divide_target_by_weight=True, alpha=0.001,
    monotone={"BonusMalus": "increasing"},
)

t_mains = rate_tables(fit)             # the model without the interaction
t_int = rate_tables(two_stage_fit)     # the same model plus DrivAge × BonusMalus
same = np.allclose(
    t_mains["DrivAge"]["relativity"].to_numpy(),
    t_int["DrivAge"]["relativity"].to_numpy(),
    rtol=1e-10,
)
print("main tables identical with the interaction added:", same)
# → True
print("base rate:", round(base_rate(fit), 5), round(base_rate(two_stage_fit), 5))
# → base rate: 0.04341 0.04341   (the same number; they differ only by solver noise)
```

`t_int["DrivAge×BonusMalus"]` is the cell table: `from_a`/`to_a` (DrivAge's
band), `from_b`/`to_b` (BonusMalus's band), `kept` (did the cell clear the
minimum-exposure floor, default 0.5% of the interaction's training
exposure?), `relativity`. A cell below the floor reads 1.00 with its exposure
alongside, rather than an unstable number from too little data.

## 7. Adjustments and rebalancing

The rate tables the lasso hands you are a starting point, not a final answer.
`easy_glm.engine.tooling` computes what a smoothing, a cap/floor or a
rounding would do to one variable's table — before anything is applied — and
every tool returns one relativity per table row, in table order, ready to
write back with `RateModel.update_relativity`:

```python
from easy_glm.engine import tooling

rm = to_rate_model(fit, exposure_col="Exposure", train_test_col="traintest")
before_total = rm.predict(train_df).sum()

# Cap BonusMalus at 3.00× — the top band above was 4.30×
cfg = rm.variables["BonusMalus"]
result = tooling.cap_floor(cfg, "BonusMalus", cap=3.0)
print(result.note)
for row, value in zip(cfg.table, result.values):
    rm.update_relativity("BonusMalus", row.from_, row.to_, value)

# Smooth DrivAge: a 3-band exposure-weighted moving average in log space
cfg = rm.variables["DrivAge"]
result = tooling.smooth_moving_average(cfg, "DrivAge", window=3)
print(result.note)
for row, value in zip(cfg.table, result.values):
    rm.update_relativity("DrivAge", row.from_, row.to_, value)

after_total = rm.predict(train_df).sum()
print(f"Total expected claims moved by {after_total / before_total - 1:+.2%}")
# → Total expected claims moved by -4.04%

# Put the level back exactly, without touching a single relativity
rm.base_rate *= before_total / after_total
print("Rebalanced:", np.isclose(rm.predict(train_df).sum(), before_total, rtol=1e-9))
# → Rebalanced: True
```

**Say this plainly, because it surprises people**: capping and smoothing move
the total expected claims on the book, even though smoothing preserves the
exposure-weighted mean of the *log* relativities (a shape rule, not a money
rule — a premium is a product of relativities, and a book is the *sum* of
those products). Rebalancing the base rate afterwards is one multiplication
and one undo-able step; it is the off-balance correction a rate review
always needs, and it moves no relativity.

## 8. Exact scoring, `.easyglm`, and the one-line invariant

The rate tables are not an approximation of the GLM — **before any manual
adjustment**, `RateModel.predict` reproduces `GLMFit.predict` to
floating-point precision, nulls and unseen levels included. This is the
promise the whole product rests on, and it is one line to check on any model
you build (a freshly-built `RateModel`, not the `rm` from §7, which now
carries the cap and the smoothing on top of the fit on purpose):

```python
from easy_glm.engine import RateModel

fresh_rm = to_rate_model(fit, exposure_col="Exposure", train_test_col="traintest")
assert np.allclose(
    fresh_rm.predict(holdout, exposure_col=None), fit.predict(holdout), rtol=1e-10
)
print("Invariant holds: RateModel reproduces the GLM to 1e-10.")
```

Once you *have* adjusted a model, the invariant that matters is different —
not equality with the raw fit, but that a `.easyglm` file round-trips through
JSON with the adjustments intact. `.easyglm` is the portable scoring file:
plain JSON, no glum, no Python environment required to read it back.

```python
import tempfile, os

path = os.path.join(tempfile.mkdtemp(), "french_motor.easyglm")
rm.to_json(path)   # rm is the adjusted model from §7

reloaded = RateModel.from_json(path)
print(np.allclose(reloaded.predict(holdout), rm.predict(holdout)))
# → True
print(np.round(reloaded.predict(holdout.head(3)), 5))
# → [0.00172 0.00687 0.08667]
```

`rm.predict(data)` multiplies by the exposure column recorded in the model
(here `Exposure`); pass `exposure_col=None` for a per-unit rate, as the
invariant check above does — `fit.predict` is always per-unit.

## 9. Excel export

```python
xlsx_path = os.path.join(tempfile.mkdtemp(), "rate_tables.xlsx")
out = rm.to_excel(xlsx_path)
print("wrote", out)
```

`RateModel.to_excel` writes the tables **as the scorer uses them** — manual
adjustments included — one sheet per variable plus a `Summary` sheet (base
rate, how to read the numbers, every `x_base` point) and a `Coefficients`
sheet. `EasyGLM.to_excel` writes the *fitted* (pre-adjustment) tables instead,
labelled as such; use whichever answers the question you are asking.

## 10. Books of millions of rows

Nothing above changes for a bigger book: the design matrix stores **one
integer per row per rating factor** rather than a column of noughts and ones
per band, so memory grows with the number of *factors*, not the number of
bands. This "compact" form switches on by itself at 200,000 rows
(`easy_glm.core.design.SPARSE_ROW_THRESHOLD`); `fit_glm(..., sparse=True /
False)` forces either. Coefficients, the non-zero set and predictions agree
between the two forms to 1e-10 (`tests/test_scale.py`,
[`docs/checks/g-scale.md`](docs/checks/g-scale.md)), and scoring never builds
a design matrix at all, whichever way the model was fitted.

```python
def synthetic_book(n, seed=0):
    """A motor book with the shape a real one has, for demonstration only."""
    r = np.random.default_rng(seed)
    age = r.integers(18, 90, n).astype(float)
    bonus = r.integers(50, 230, n).astype(float)
    density = r.lognormal(5.0, 1.5, n)
    region = r.choice([f"Region{i:02d}" for i in range(15)], n)
    exposure = r.uniform(0.1, 1.0, n)
    mu = np.exp(-3.0 + 0.012 * (60 - age) + 0.006 * (bonus - 100) + 0.05 * np.log1p(density))
    claims = r.poisson(mu * exposure)
    return pl.DataFrame({
        "DrivAge": age, "BonusMalus": bonus, "Density": density,
        "Region": region, "Exposure": exposure, "ClaimNb": claims.astype(float),
    })

import time

big_df = synthetic_book(300_000)
big_predictors = ["DrivAge", "BonusMalus", "Density", "Region"]
big_spec = DesignSpec.from_data(big_df, big_predictors, weight_col="Exposure")
print(f"{len(big_df):,} rows, {big_spec.n_features} design columns")
# → 300,000 rows, 75 design columns

t0 = time.perf_counter()
big_fit = fit_glm(
    big_df, big_spec, "ClaimNb", family="poisson", weight_col="Exposure",
    divide_target_by_weight=True, alpha=0.001,
)
print(f"fitted in {time.perf_counter() - t0:.1f}s")
# → fitted in well under a second on this synthetic book
```

Measured on a 24 GB laptop with a 227-column real rating structure
(`scripts/bench_scale.py`, full numbers in
[`docs/checks/g-scale.md`](docs/checks/g-scale.md)):

| rows | fit | peak memory |
|---:|---:|---:|
| 200,000 | 1 s | 0.37 GB |
| 1,000,000 | 4 s | 0.86 GB |
| 5,000,000 | 21 s | 2.59 GB |

5,000,000 rows with the same rating structure could not be fitted on that
machine at all before this: the dense design matrix alone would have been
about 8 GB. `examples/large_book.py --rows 1000000` runs the same benchmark
from the command line, and `pytest -m slow` is the CI-grade version of the
5,000,000-row memory budget (it needs about 3 GB free and is not run by the
default test suite).

## 11. A project file, and the exported Python script

Everything so far is plain Python. `easy_glm.workflow.Project` is the same
model wrapped in one JSON file — data location, column roles, the design and
one or more model configurations — which is what lets the browser workbench,
the command line and a reproducible exported script all agree on exactly one
model. Building one from Python is the same information as the design above,
just declared once:

```python
import subprocess
import sys
from pathlib import Path

from easy_glm.workflow import Project, VariableDesign, prepare, run_model, to_script

project = Project(name="french_motor_demo")
project.data.source.path = DATA
project.data.roles = {
    "ClaimNb": "target",
    "Exposure": "weight",
    "DrivAge": "predictor",
    "Region": "predictor",
    "BonusMalus": "predictor",
    "Density": "predictor",
}
project.data.split.mode = "random"
project.data.split.fraction = 0.7
project.data.split.seed = 42
project.design.variables["Density"] = VariableDesign(kind="linear")
project.design.variables["BonusMalus"] = VariableDesign(monotone="increasing")

project.new_model(
    "freq", family="poisson", divide_target_by_weight=True, predictors=PREDICTORS
)
project.models["freq"].penalty.alpha = 0.001
project.models["freq"].penalty.cv = None

print("problems:", project.validate("freq"))
# → problems: []

df_prepared = prepare(project)   # loads DATA, applies renames/recodes/filters/split
run = run_model(project, df_prepared, "freq")
print(run.summary())
# → {'name': 'freq', 'family': 'poisson', 'alpha': 0.001, ...,
#    'holdout_ae': 0.9948, 'holdout_gini': 0.3285, ...}

project_path = Path("french_motor_demo.easyglm-project.json")
project.to_json(project_path)
```

`to_script` writes the whole model as a **self-contained, runnable Python
file** — every knot, every level and the resolved alpha spelled out, so it
neither depends on the project file nor re-runs cross-validation when
executed:

```python
script_text = to_script(project, "freq", run=run, output_prefix="freq")
Path("freq_export.py").write_text(script_text)

result = subprocess.run([sys.executable, "freq_export.py"], capture_output=True, text=True)
print("exit code:", result.returncode)
# → exit code: 0
```

Running `freq_export.py` writes `freq.easyglm` (among other artefacts) in the
current directory, fitted from the data the project points at — the same
model `run` already holds, to floating-point precision (checked in
[§17](#17-the-project-file-and-the-script-round-tripped)).

## 12. Diagnostics: lift, Gini, A/E, double lift

`easy_glm.workflow` has the diagnostics as plain functions on
actual/expected/weight arrays, so they work whether or not you use a
`Project` — pass `run.predict(df)` (or `rm.predict(df, exposure_col=None)`)
for the expected side.

```python
from easy_glm.workflow import Interaction, ae_by_variable, double_lift, gini, lift_table, totals

# A challenger model with an interaction, for the double-lift comparison below
project.new_model(
    "freq_interaction", family="poisson", divide_target_by_weight=True,
    predictors=PREDICTORS, interactions=[Interaction("DrivAge", "BonusMalus")],
)
project.models["freq_interaction"].penalty.alpha = 0.001
project.models["freq_interaction"].penalty.cv = None
challenger_run = run_model(project, df_prepared, "freq_interaction")
project.to_json(project_path)   # the project file now has both models

holdout_p = df_prepared.filter(pl.col(project.data.split.column) == 0)
cfg = project.models["freq"]
a_total, e_champ, w = totals(holdout_p, cfg, run.predict(holdout_p))
_, e_chal, _ = totals(holdout_p, cfg, challenger_run.predict(holdout_p))

print("Gini champion:  ", gini(a_total, e_champ, w))
print("Gini challenger:", gini(a_total, e_chal, w))
# → Gini champion:   0.3285
#   Gini challenger: 0.3289  (the interaction buys a little lift, as expected)

print(lift_table(a_total, e_champ, w, n_bins=5).select("bin", "ae", "exposure"))
# → 5 equal-exposure bins ordered by predicted rate, each with its own A/E

print(double_lift(a_total, e_champ, e_chal, w, n_bins=5).select("bin", "ae_a", "ae_b"))
# → bins ordered by (champion / challenger) predicted rate: where the two
#   models disagree most, "ae_a" and "ae_b" show which one is closer to 1

print(ae_by_variable(holdout_p, "Region", a_total, e_champ, w).head(5))
# → one row per level of Region: exposure, actual, expected, "ae"
```

**A rough edge worth knowing about**: summing `actual` and `expected` across
*every* bin of `ae_by_variable`'s own output always reproduces the model's
overall A/E, whatever variable you grouped by — the bins partition the same
rows. What is informative is the *spread* of `ae` across bins (a
well-calibrated factor keeps every bin close to 1), not a total across all of
them; `examples/advanced_pipeline.py` shows the fix (`table["ae"].min()` /
`.max()`) if you build this by hand.

## 13. Rate change: pricing the move from today's premium

A rate review usually prices the **change** from the premium you charge
today, not a price from scratch. Give the column holding today's premium the
role `current_premium` and `easy_glm` derives `log(<premium>)` and pre-fills
it as the offset of every new model in the project: every relativity becomes
a *multiplier on that premium* (1.00 = "moves with the base risk and no
more"), and the base rate becomes the overall change.

```python
from easy_glm.workflow import solve_base_rate

# A stand-in "current premium" column — in practice this is already in your book.
df_premium = pl.read_parquet(DATA).with_columns(
    (pl.col("Exposure") * 180.0).alias("CurrentPremium")
)
premium_path = Path("french_motor_with_premium.parquet")
df_premium.write_parquet(premium_path)

project_rc = Project(name="rate_change_demo")
project_rc.data.source.path = str(premium_path)
project_rc.data.roles = {
    "ClaimNb": "target", "Exposure": "weight", "CurrentPremium": "current_premium",
    "DrivAge": "predictor", "Region": "predictor",
    "BonusMalus": "predictor", "Density": "predictor",
}
project_rc.data.split.mode = "random"
project_rc.data.split.fraction = 0.7
project_rc.data.split.seed = 42
project_rc.new_model(
    "change", family="poisson", divide_target_by_weight=True, predictors=PREDICTORS
)
project_rc.models["change"].penalty.alpha = 0.001
project_rc.models["change"].penalty.cv = None

df_rc = prepare(project_rc)
run_rc = run_model(project_rc, df_rc, "change")
print(run_rc.rate_model.relativity_label)
# → multiplier on current premium
print(run_rc.rate_model.base_rate)
# → the overall change for the base risk, e.g. 0.000272 (a tiny number here
#   because the stand-in premium above is unrelated to the true cost)
```

The Model page's **target loss ratio** box is `solve_base_rate` in closed
form: it sets the base rate so total actual ÷ total expected on the rows
given equals the ratio you ask for — for a rate-change model that ratio *is*
the loss ratio the book would be written at — without moving a single
relativity.

```python
train_rc = df_rc.filter(pl.col(project_rc.data.split.column) == 1)
print(solve_base_rate(run_rc, train_rc, target_ratio=0.65))
# → the base rate that puts this book at a 65% loss ratio
```

For an ordinary (non-rate-change) model, `target_ratio=1.00` is the familiar
rebalance to overall A/E = 1. Binomial models are refused here — a
probability is not proportional to a base rate.

## 14. A lapse model (binomial)

`log` and `logit` are both multiplicative links, so a lapse (or conversion)
model compiles to the same kind of rate table as a frequency model — read as
**odds relativities** — and the scorer converts back to a probability.
Because a probability is not an amount, such a model refuses to be
multiplied by an exposure column.

```python
rng2 = np.random.default_rng(7)
df_lapse = pl.read_parquet(DATA)
# A stand-in lapse flag — in practice this comes from your policy admin
# system. Here, older and higher-bonus-malus drivers lapse more often.
logit_p = 1.5 - 0.02 * df_lapse["DrivAge"].to_numpy() + 0.01 * df_lapse["BonusMalus"].to_numpy()
p = 1 / (1 + np.exp(-logit_p))
df_lapse = df_lapse.with_columns(pl.Series("Lapsed", (rng2.random(len(df_lapse)) < p).astype(float)))
df_lapse = df_lapse.with_columns(pl.Series("traintest", rng2.random(len(df_lapse)) < 0.7, dtype=pl.Int64))
train_lapse = df_lapse.filter(pl.col("traintest") == 1)
holdout_lapse = df_lapse.filter(pl.col("traintest") == 0)

lapse_spec = DesignSpec.from_data(train_lapse, ["DrivAge", "BonusMalus"])
lapse_fit = fit_glm(train_lapse, lapse_spec, "Lapsed", family="binomial", alpha=0.0005)
lapse_rm = to_rate_model(lapse_fit)
print(lapse_rm.relativity_label)
# → odds relativity

prob = lapse_rm.predict(holdout_lapse)   # a probability in (0, 1), no exposure
print(prob[:5])
# → [0.8696 0.6112 0.7094 0.8175 0.7363]
```

`examples/lapse_model.py` also shows the exposure refusal:
`lapse_rm.predict(holdout_lapse, exposure_col="Exposure")` raises a
`ValueError` naming the reason, rather than silently returning a number that
is not a count.

## 15. The workbench

The same `Project` opens in an Emblem-style browser workbench: ten pages
(project & data, variables, explore, split, design, model, diagnostics,
compare, rate tables, export) over the same engine used on this page —
nothing the GUI does that plain Python above could not also do.

```bash
pip install "easy_glm[ui]"
easy-glm-workbench french_motor_demo.easyglm-project.json
# or, from a source checkout: python -m easy_glm.app french_motor_demo.easyglm-project.json
```

![the linear-term editor on the Design page](docs/checks/img/w2_design_linear.png)

![A/E by pair of variables on the Diagnostics page](docs/checks/img/w2_diagnostics_pair.png)

The stand-alone relativity editor (baseline vs. working copy, A/E charts,
save/download) opens directly from a fitted model — this call opens a
browser tab and returns immediately, so it is marked `skip-test` below rather
than run by the test suite:

```python skip-test
eglm.rate_model.launch_editor(data=df, port=8501)
# Opens http://localhost:8501; Ctrl-C the Python process when you're done.
```

## 16. The command line

Everything the workbench does to a project, without a browser — a scheduled
refit, a build server, or a colleague without Python:

```bash
easy-glm run project.json --out artefacts/     # fit; write scorer, Excel, script, report
easy-glm export project.json --script          # or --report / --excel, combinable
easy-glm validate project.json                 # exit 1 and list the problems
easy-glm workbench project.json                # open it in the browser
```

Every artefact command **fits afresh** from the data the project points at —
persisted workbench runs are never reused — which is what makes the exported
script and report self-contained. Problems are messages with a non-zero exit
code, never a traceback, so a scheduled job can tell success from failure.
Here it is invoked as a module, the way this repository's own tests do
(`easy-glm ...` is the equivalent after `pip install easy_glm`):

```python
cli_result = subprocess.run(
    [sys.executable, "-m", "easy_glm.cli", "validate", str(project_path)],
    capture_output=True, text=True,
)
print(cli_result.stdout)
print("exit:", cli_result.returncode)
# → french_motor_demo.easyglm-project.json: valid · models: freq, freq_interaction
#   exit: 0
```

## 17. The project file and the script, round-tripped

The project file is the single source of truth: it round-trips through JSON
exactly, and the script exported from it in [§11](#11-a-project-file-and-the-exported-python-script)
reproduces the workbench's own fit to floating-point precision — not
approximately, to about 1e-16.

```python
from easy_glm.engine import RateModel

reloaded_project = Project.from_json(project_path)
print(list(reloaded_project.models))
# → ['freq', 'freq_interaction']

original_preds = run.rate_model.predict(df_prepared.head(50), exposure_col=None)
script_rm = RateModel.from_json("freq.easyglm")   # written by freq_export.py in §11
script_preds = script_rm.predict(df_prepared.head(50), exposure_col=None)
print("workbench run and exported script agree to 5 decimals:",
      np.max(np.abs(original_preds - script_preds)) < 1e-5)
# → workbench run and exported script agree to 5 decimals: True
```

Open `french_motor_demo.easyglm-project.json` in the workbench
([§15](#15-the-workbench)) and you get back exactly the model built in Python
above — the same models, the same design, the same fitted numbers.

---

## Architecture

```
Raw data → DesignSpec → fit_glm (glum) → rate_tables / to_rate_model → RateModel (.easyglm)
                                   ↑
                         EasyGLM.fit() runs all of this

Project (JSON) → workflow.run_model → ModelRun ─┬→ workflow.to_script (a runnable .py)
                                                 ├→ workflow.to_report_html (one HTML file)
                                                 └→ easy_glm.app (the browser workbench)
```

| Component | Role |
|-----------|------|
| `DesignSpec` | Feature definitions (step knots, linear bands, levels); builds the design matrix; JSON |
| `fit_glm` / `fit_two_stage` | Penalised glum fit; `GLMFit` / `TwoStageFit` (coefficients, predictions) |
| `rate_tables` / `to_rate_model` | Exact relativities and base rate from the coefficients |
| `RateModel` | Production scoring, A/E, adjustments, JSON roundtrip, the editor |
| `easy_glm.engine.tooling` | Smooth / cap-floor / round one variable's table |
| `easy_glm.workflow` | `Project` spec, prep steps, diagnostics, `run_model`, `to_script`, `to_report_html` |
| `easy_glm.app` | The Streamlit workbench over the workflow engine |
| `easy_glm.cli` | The `easy-glm` command line over the same workflow engine |
| `EasyGLM` | One-call fit, save/load of the whole pipeline |

Package layout, module map and conventions: [`AGENTS.md`](AGENTS.md). Design
notes for the workbench: [`docs/WORKBENCH_PLAN.md`](docs/WORKBENCH_PLAN.md).
Plain-language write-ups of every 0.4 piece, each regenerated from the tests
that back it: [`docs/checks/`](docs/checks/).

## Development

```bash
black . && ruff check . && mypy src/easy_glm/core src/easy_glm/workflow --ignore-missing-imports && pytest -q
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md) for the
full set of commands (single-test invocations, the scale benchmark, the
Playwright persona tests).

## License

MIT — see [`LICENSE`](LICENSE).
