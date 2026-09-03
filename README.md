# easy_glm

**LASSO GLMs → rate tables → calibrate in a browser → score portfolios.**

Python toolkit for insurance pricing: fit a regularised GLM on your data, export
per-variable relativities, tweak them in an interactive editor (with A/E charts),
and ship a portable `.easyglm` model for scoring. Built on [glum](https://glum.readthedocs.io/en/latest/); inspired by R [aglm](https://github.com/kkondo1981/aglm).

```bash
pip install git+https://github.com/serband/easy_glm.git
# optional UI: pip install "easy_glm[ui]"
```

---

## 1. Fit a model (one shot)

Most of the time you only need **`EasyGLM.fit`**. It builds a *design spec*
(step knots for numeric factors, one-hot with an `Other` bucket for
categoricals), fits an L1-penalised GLM with glum, and reads **exact** rate
tables and a calibrated base rate straight off the coefficients.

Add a **`traintest`** column to your data: **1 = train** (fitting), **0 = holdout**
(validation). Pass the full dataframe; only `traintest == 1` rows are used to
build the spec and fit the GLM.

```python
import easy_glm
import polars as pl
import numpy as np

df = easy_glm.load_external_dataframe()
df = df.with_columns(
    pl.Series("traintest", np.random.default_rng(42).random(len(df)) < 0.7, dtype=pl.Int64)
)

predictors = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

eglm = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=predictors,
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,   # frequency = ClaimNb / Exposure
    cv=5,                           # or alpha=0.001 for a quick fit
    monotone={"BonusMalus": "increasing"},   # optional sign constraints
)
print(eglm)
```

Families: `"Poisson"`, `"Gamma"`, `"Tweedie"`, `"Gaussian"` (all log link),
`"Binomial"` (logit; no multiplicative tables).

### View relativities

Per-variable tables are on **`eglm.relativities`** — a dict of Polars frames with
`from` / `to` (bin edges or level), `label`, `coef`, `relativity`, `exposure`
(the training exposure in that band) and `is_base`.
Relativity 1.0 sits on the most exposed bin of each variable; the null / `Other`
row is last. `eglm.coef_table(drop_zero=True)` lists the knots and levels the
lasso kept.

```python
print(eglm.relativities["DrivAge"])
print(eglm.coef_table(drop_zero=True))

# Optional: matplotlib charts (pip install "easy_glm[viz]")
easy_glm.plot_all_ratetables(eglm.relativities)
```

### Score

`eglm.rate_model` is a portable lookup-table scorer. It **reproduces the GLM
exactly** (to floating-point precision), nulls and unseen levels included, and
its base rate is calibrated automatically (pass `base_rate=` to override).

```python
test = df.filter(pl.col("traintest") == 0)
preds = eglm.rate_model.predict(test)          # multiplied by Exposure
freq = eglm.predict(test)                      # GLM, per unit exposure
print(f"Test A/E: {test['ClaimNb'].sum() / preds.sum():.4f}")

eglm.save("my_model")                          # spec + glum model + tables
eglm.rate_model.to_json("model.easyglm")       # scorer only
eglm.to_excel("rate_tables.xlsx")              # fitted tables + coefficients
eglm.rate_model.to_excel("rate_tables_adjusted.xlsx")   # tables as scored, incl. editor changes
```

Any `RateModel` — including one edited in the browser and downloaded as
`.easyglm` — exports the same way: `RateModel.from_json("revised.easyglm").to_excel("revised.xlsx")`.

**Performance and size.** A fixed-`alpha` fit of the full French motor set
(~680k rows, 9 predictors) takes a couple of seconds and `cv=5` over a 20-point
alpha path around 20 seconds. Books much bigger than that are fine: the design
matrix stores **one integer per row per rating factor** rather than a column of
noughts and ones per band, so memory grows with the number of *factors*, not
the number of bands. Measured on a 24 GB laptop with a 227-column design
(`scripts/bench_scale.py`):

| rows | fit | peak memory |
|---:|---:|---:|
| 200,000 | 1 s | 0.4 GB |
| 1,000,000 | 4 s | 0.8 GB |
| 5,000,000 | 21 s | 2.6 GB |

The compact form switches itself on at 200,000 rows (`fit_glm(..., sparse=)`
forces either); it is float64 throughout and gives the same coefficients,
the same non-zero set and predictions agreeing to 1e-10 with the dense matrix
(`tests/test_scale.py`, `docs/checks/g-scale.md`). Scoring never builds a
design matrix at all — `predict` adds up the rate-table lookups in row
chunks — so a fitted model scores a book of any size in one pass.

---

## 2. Building blocks (when you need control)

`EasyGLM.fit` is three calls you can make yourself:

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `DesignSpec.from_data(train_df, predictors)` | Quantile knots per numeric, frequency-ordered levels per categorical — **train only**. JSON round-trip; edit by hand. |
| 2 | `fit_glm(train_df, spec, target, ...)` | glum L1/elastic-net fit on `spec.build(train_df)`; `alpha=` or `cv=`; `monotone=` (step increments or piecewise-linear band slopes) |
| 3 | `rate_tables(fit)` / `to_rate_model(fit)` | Exact relativities + base rate from the coefficients |

```python
from easy_glm import DesignSpec, fit_glm, rate_tables, to_rate_model

train_df = df.filter(pl.col("traintest") == 1)

spec = DesignSpec.from_data(
    train_df, predictors,
    n_bins=20, min_level_share=0.0025,
    knots={"VehAge": list(range(1, 21))},     # hand-picked knots
    weight_col="Exposure",
)

fit = fit_glm(
    train_df, spec, target="ClaimNb", family="poisson",
    weight_col="Exposure", divide_target_by_weight=True,
    alpha=0.001, monotone={"DrivAge": "decreasing"},
)
print(fit.coef_table(drop_zero=True))

tables = rate_tables(fit)
rm = to_rate_model(fit, exposure_col="Exposure", train_test_col="traintest")
rm.to_json("model.easyglm")
```

Full script: [`examples/advanced_pipeline.py`](examples/advanced_pipeline.py).

> **Upgrading from 0.2 / 0.3?** The blueprint / DuckDB pipeline (`generate_blueprint`,
> `prepare_data`, `fit_lasso_glm`, `ratetable`, `generate_all_ratetables`) was removed in
> 0.4; use the building blocks above. `RateModel.from_rate_tables(tables, base_rate)` now
> takes the table format produced by `rate_tables` / the Excel export (no blueprint).
> Models saved by 0.2 must be refitted; 0.3 `.easyglm` files open unchanged.

---

## 3. Adjust relativities, save, and score

After fitting, open the **relativity editor** to review A/E by band, nudge factors,
and export a revised model. The fitted baseline is never overwritten — edits live
in a working copy until you save or download.

```python
# From a fitted EasyGLM or any RateModel
eglm.rate_model.launch_editor(data=df)   # opens a browser tab; Python keeps running
# or:  rm.launch_editor(data=df)
```

**In the UI**

1. Pick a variable → relativity curve + A/E chart + editable table.
2. Change *Revised* relativities; charts update (toggle auto-recompute off on large data).
3. **Download** → saves `your_name.easyglm`, or save named copies in-session.

**Score with the saved model** (no refit — pure lookup tables):

```python
from easy_glm.engine import RateModel

rm = RateModel.from_json("my_revision.easyglm")

holdout = df.filter(pl.col("traintest") == 0)
premiums_or_freq = rm.predict(holdout)

# Per-variable calibration check
ae = rm.compute_ae_for_variable(new_business, "DrivAge")
```

Install UI dependencies if needed: `pip install "easy_glm[ui]"` (Streamlit + Plotly).

---

## 4. The Workbench — the whole workflow in the browser

```bash
pip install "easy_glm[ui]"
easy-glm-workbench                      # or: python -m easy_glm.app my.easyglm-project.json
```

An Emblem-style GUI over the same engine. Ten pages, one project file:

| Page | What you do |
|------|-------------|
| Project & data | open/save a project, point at parquet / csv / sas7bdat / xlsx, optional sample |
| Variables | roles (target, weight, exposure, offset, **current premium**, split, id, predictor, ignore), renames, type overrides, **level recodes**, derived columns (polars expressions), row filters |
| Explore | exposure & observed rate by band; **leakage report** (single-factor deviance explained, target proxies, identifier-like columns, post-outcome names) with one-click ignore / acknowledge |
| Split | indicator column or seeded random split; train/holdout balance |
| Design | per-predictor kind (**step** by default · **linear** piecewise curve · **continuous** one straight line · **categorical**), knots (quantile / integer / custom), clamp range, null column, level share, monotone direction, **penalty weight** (0 = unpenalised); exposure + rate preview per bin |
| Model | family (incl. **Tweedie power** and binomial), target/weight/offset, penalty (fixed alpha or CV), predictors, **target loss ratio**; fit; coefficients kept; regularisation path |
| Diagnostics | A/E by any variable (in or out of the model, champion vs challenger), lift & Gini, double lift vs a challenger or a premium column, residual factor search |
| Compare | two fitted models side by side: metrics on train and holdout, A/E with both expected lines, lift, double lift, **which relativities differ**, make champion |
| Rate tables | relativities with A/E (challenger overlaid) and training exposure per band, inline edits saved as adjustments (no refit), **tools** (smooth in log space — moving average or isotonic — cap/floor, round), **undo/redo**, named **snapshots** with a diff between any two, Excel / `.easyglm` download |
| Export | the whole workflow as a **runnable Python script** (explicit knots, levels, resolved alpha, adjustments), a **self-contained HTML report**, project JSON, artefacts |

Everything the GUI does edits a `Project` spec (`easy_glm.workflow`) that is autosaved
as JSON; the exported script reproduces the GUI model exactly (this is tested).
Design notes: [`docs/WORKBENCH_PLAN.md`](docs/WORKBENCH_PLAN.md).

**Comparing two models, and writing one up.** Pick a challenger once in the sidebar
("Compare with") and every page uses it: Diagnostics and Rate tables draw its expected
line next to the champion's, and the **Compare** page puts the two models' metrics
side by side with the double lift and a table of *which relativities actually differ*
— one row per band whose relativity moved by more than a tolerance you set (1 % by
default), on the log scale, with interactions compared cell by cell and the base rates
against each other; two identical models give an empty table
(`workflow.relativity_diff`). When you are done, **Download HTML report** on the Export
page writes the whole model up as **one self-contained file** — summary and split, a
block per rating factor with its relativities and its actual-vs-expected on train and
holdout, interaction heatmaps, lift and Gini, the comparison section, every coefficient
and the reproducing Python script — a few hundred kB (350–400 kB for the French
motor set), nothing fetched from the internet when it is opened, so it can be
emailed or attached to a filing (`workflow.to_report_html`).

---

## 5. Rate change: fit the move from today's premium

The standard rate review does not price from scratch — it prices the **change**
from the premium you charge today. Put `log(current premium)` in the offset and
every relativity becomes a *multiplier on that premium*: 1.00 means "this band
moves with the base risk and no more", 1.20 means "20 % more than that".

```python
from easy_glm import DesignSpec, fit_glm, to_rate_model

# the premium your book charges today (here: a stand-in built from exposure)
df = df.with_columns((pl.col("Exposure") * 180.0).alias("CurrentPremium"))
df = df.with_columns(pl.col("CurrentPremium").log().alias("log_CurrentPremium"))
train_df = df.filter(pl.col("traintest") == 1)

spec = DesignSpec.from_data(train_df, predictors, weight_col="Exposure")
fit = fit_glm(
    train_df, spec, "ClaimNb", family="poisson",
    offset_col="log_CurrentPremium",           # the offset carries the premium
    alpha=0.001,
)
rm = to_rate_model(fit, offset_is_premium=True)   # label the tables accordingly
print(rm.relativity_label)   # "multiplier on current premium"
print(rm.base_rate)          # the change for the base risk
```

In the **workbench** this is one setting: give the premium column the role
**current premium** on the Variables page. `log_<premium>` is derived for you,
every new model offsets on it, and the Rate tables, Export and Excel pages say
what the numbers mean. The Model page then has a **Target loss ratio** box:
type the loss ratio you want the book written at and the base rate that gets
there is solved in closed form — the relativities do not move
(`workflow.solve_base_rate`). Worked through end to end, with the numbers, in
[`docs/checks/e-f-extras-cli.md`](docs/checks/e-f-extras-cli.md).

Two more knobs that belong to the same page:

* **Penalty weight** per factor (Design page, or
  `DesignSpec.from_data(..., penalty_weight={"Region": 0.0})`): 1 is normal,
  2 shrinks a factor twice as hard, **0 leaves it unpenalised** so every level
  survives the lasso.
* **Tweedie power** (Model page, or `fit_glm(..., family="tweedie",
  tweedie_power=1.7)`) and **binomial** models, whose tables are *odds*
  relativities and whose scorer returns probabilities.

---

## 6. Command line

Everything the workbench does to a project, without a browser:

```bash
easy-glm run project.json --out artefacts/   # fit; write scorer, Excel, script, report
easy-glm export project.json --script        # or --report / --excel (combinable)
easy-glm validate project.json               # exit 1 and list the problems
easy-glm workbench project.json              # open it in the browser
```

`run` prints the fit summary (rows, alpha, base rate, A/E, Gini, deviance
explained) and writes four files: the `.easyglm` scorer, the Excel rate tables,
a runnable Python script and the self-contained HTML report. Every command fits
afresh from the data the project points at, so the script it writes has every
knot, level and the resolved alpha in it. Problems are messages with a non-zero
exit code, never a traceback, so a scheduled job can tell success from failure.

---

## Install (development)

```bash
uv venv && uv pip install -e ".[dev]"        # dev already includes streamlit + plotly
```

Extras: `ui` (workbench + editor), `viz` (matplotlib/seaborn/plotnine charts),
`benchmark` (statsmodels + catboost), `dev` (tests, lint, ui).

Python **3.10–3.13**. Optional extras: `[ui]`, `[dev]`, `[viz]`.

---

## Architecture

```
Raw data → DesignSpec → fit_glm (glum) → rate_tables / to_rate_model → RateModel (.easyglm)
                                   ↑
                         EasyGLM.fit() runs all of this
```

| Component | Role |
|-----------|------|
| `DesignSpec` | Feature definitions (step knots, levels); builds the design matrix; JSON |
| `fit_glm` / `GLMFit` | Penalised glum fit, coefficient table, predictions |
| `rate_tables` / `to_rate_model` | Exact relativities and base rate from coefficients |
| `to_excel` / `write_rate_tables_xlsx` | Rate tables as an `.xlsx` workbook (one sheet per variable) |
| `easy_glm.workflow` | `Project` spec, prep steps, leakage report, diagnostics, `run_model`, `to_script` |
| `easy_glm.app` | Streamlit workbench over the workflow engine |
| `EasyGLM` | One-call fit, save/load full pipeline |
| `RateModel` | Production scoring, A/E, JSON roundtrip, editor |

Package layout, benchmarks, and module map: see [`AGENTS.md`](AGENTS.md).

---

## Development

```bash
black . && ruff check . && pytest -q
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Roadmap

- [x] Monotone constraints (`monotone={"DrivAge": "decreasing"}`)
- [x] Configurable knots / levels per variable (`DesignSpec`)
- [ ] Two-way interactions (`A × B` tables)
- [x] Piecewise-linear terms (`kind="linear"` / `"continuous"`: one penalised slope per band, so the curve is flat unless the data insists; monotone constraints supported)
- [ ] CLI (`python -m easy_glm build ...`)
- [ ] Drag-to-edit relativities (GAMChanger-style)
- [ ] Multi-model A/E comparison in the editor

---

## License

MIT — see [`LICENSE`](LICENSE).
