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
`from` / `to` (bin edges or level), `label`, `coef`, `relativity` and `is_base`.
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

**Performance.** On the bundled French motor set (~680k rows, 6 predictors) a
fixed-`alpha` fit takes about a second and `cv=5` over a 20-point alpha path
around 10–20 seconds; peak memory is ~1 GB (the design matrix is dense float64).

---

## 2. Building blocks (when you need control)

`EasyGLM.fit` is three calls you can make yourself:

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `DesignSpec.from_data(train_df, predictors)` | Quantile knots per numeric, frequency-ordered levels per categorical — **train only**. JSON round-trip; edit by hand. |
| 2 | `fit_glm(train_df, spec, target, ...)` | glum L1/elastic-net fit on `spec.build(train_df)`; `alpha=` or `cv=`; `monotone=` |
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

An Emblem-style GUI over the same engine. Nine pages, one project file:

| Page | What you do |
|------|-------------|
| Project & data | open/save a project, point at parquet / csv / sas7bdat / xlsx, optional sample |
| Variables | roles (target, weight, exposure, offset, split, id, predictor, ignore), renames, type overrides, **level recodes**, derived columns (polars expressions), row filters |
| Explore | exposure & observed rate by band; **leakage report** (single-factor deviance explained, target proxies, identifier-like columns, post-outcome names) with one-click ignore / acknowledge |
| Split | indicator column or seeded random split; train/holdout balance |
| Design | per-predictor knots (quantile / integer / custom), null column, level share, monotone direction; exposure + rate preview per bin |
| Model | family, target/weight/offset, penalty (fixed alpha or CV), predictors; fit; coefficients kept; regularisation path |
| Diagnostics | A/E by any variable (in or out of the model, champion vs challenger), lift & Gini, double lift vs a challenger or a premium column, residual factor search |
| Rate tables | relativities with A/E, inline edits saved as adjustments (no refit), Excel / `.easyglm` download |
| Export | the whole workflow as a **runnable Python script** (explicit knots, levels, resolved alpha, adjustments), project JSON, artefacts |

Everything the GUI does edits a `Project` spec (`easy_glm.workflow`) that is autosaved
as JSON; the exported script reproduces the GUI model exactly (this is tested).
Design notes: [`docs/WORKBENCH_PLAN.md`](docs/WORKBENCH_PLAN.md).

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
- [ ] Piecewise-linear (L-dummy) terms
- [ ] CLI (`python -m easy_glm build ...`)
- [ ] Drag-to-edit relativities (GAMChanger-style)
- [ ] Multi-model A/E comparison in the editor

---

## License

MIT — see [`LICENSE`](LICENSE).
