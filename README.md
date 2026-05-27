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

Most of the time you only need **`EasyGLM.fit`**. It bins numeric factors, lumps
sparse categoricals, runs cross-validated LASSO, and builds lookup-table relativities.

Add a **`traintest`** column to your data: **1 = train** (fitting), **0 = holdout**
(validation). Pass the full dataframe; only `traintest == 1` rows are used to
build bins and fit the GLM.

```python
import easy_glm
import polars as pl
import numpy as np

df = easy_glm.load_external_dataframe()
df = df.with_columns(
    pl.when(pl.lit(np.random.rand(df.height) < 0.7))
    .then(1)
    .otherwise(0)
    .alias("traintest")
)

predictors = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

eglm = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=predictors,
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    use_cv=True,
    base_rate=0.05,
)

test = df.filter(pl.col("traintest") == 0)
preds = eglm.rate_model.predict(test)
print(f"Test A/E: {test['ClaimNb'].sum() / preds.sum():.4f}")

# Ship the model
eglm.rate_model.to_json("model.easyglm")
```

Families: `"Poisson"`, `"Gamma"`, `"Gaussian"`, `"Binomial"`.

If your split column is not named `traintest`, pass `train_test_col="your_column"`.

**Performance.** On large portfolios (e.g. the bundled French motor set, ~680k rows),
default `use_cv=True` can take **several minutes** — glum searches many alphas ×
`l1_ratio` values. For interactive work, use a smaller sample, turn CV off, or
narrow the search:

```python
# faster iteration (~seconds on full French motor data)
eglm = easy_glm.EasyGLM.fit(..., use_cv=False)

# still CV, but much smaller grid
eglm = easy_glm.EasyGLM.fit(
    ...,
    use_cv=True,
    cv_params={"n_alphas": 5, "l1_ratio": [1.0]},
)
```

---

## 2. Fit step-by-step (when you need control)

Use the segmented pipeline if you want a **custom blueprint**, several fits on the
same prepared data, or to inspect tables between stages.

`EasyGLM.fit` runs these steps internally — `fit_lasso_glm` is **not** a separate
product; it only fits on **already prepared** columns (step 3).

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `generate_blueprint(train_df)` | Quantile breaks (numeric), lump rare levels (categorical) — **train only** |
| 2 | `prepare_data(...)` | DuckDB transforms → model-ready features |
| 3 | `fit_lasso_glm(prepped, ..., train_test_col=...)` | CV LASSO on rows where split **== 1** |
| 4 | `generate_all_ratetables(...)` | One relativity table per predictor |
| 5 | `RateModel.from_rate_tables(...)` | Portable scorer + editor input |

```python
from easy_glm.engine import RateModel

train_df = df.filter(pl.col("traintest") == 1)

blueprint = easy_glm.generate_blueprint(train_df)

prepped = easy_glm.prepare_data(
    df=df,
    modelling_variables=predictors,
    additional_columns=["Exposure", "ClaimNb", "traintest"],
    formats=blueprint,
    traintest_column="traintest",
    table_name="cars",
)

model = easy_glm.fit_lasso_glm(
    dataframe=prepped,
    target="ClaimNb",
    model_type="Poisson",
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    use_cv=True,
)

all_tables = easy_glm.generate_all_ratetables(
    model=model,
    dataset=df,
    predictor_variables=predictors,
    blueprint=blueprint,
)

rm = RateModel.from_rate_tables(
    all_tables=all_tables,
    blueprint=blueprint,
    base_rate=0.05,
    model_type="poisson",
    target="ClaimNb",
    weight_col="Exposure",
    exposure_col="Exposure",
    train_test_col="traintest",
)
rm.to_json("model.easyglm")
```

Shortcut for steps 4–5: `RateModel.from_glm_model(model, dataset=df, blueprint=blueprint, ...)`.

Full script: [`examples/advanced_pipeline.py`](examples/advanced_pipeline.py).

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

## Install (development)

```bash
python setup_dev.py          # editable install + venv
# or: uv pip install git+https://github.com/serband/easy_glm.git
```

Python **3.10–3.13**. Optional extras: `[ui]`, `[dev]`, `[viz]`.

---

## Architecture

```
Raw data → blueprint → prepare_data → fit_lasso_glm → rate tables → RateModel (.easyglm)
                                              ↑
                                    EasyGLM.fit() runs all of this
```

| Component | Role |
|-----------|------|
| `EasyGLM` | One-call fit, save/load full pipeline |
| `RateModel` | Production scoring, A/E, JSON roundtrip, editor |
| `fit_lasso_glm` | Low-level glum fit on prepared features only |

Package layout, benchmarks, and module map: see [`AGENTS.md`](AGENTS.md).

---

## Development

```bash
black . && ruff check . && pytest -q
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Roadmap

- [ ] Automated monotonic binning / isotonic smoothing
- [ ] CLI (`python -m easy_glm build ...`)
- [ ] Configurable blueprint strategies
- [ ] Drag-to-edit relativities (GAMChanger-style)
- [ ] Multi-model A/E comparison in the editor

---

## License

MIT — see [`LICENSE`](LICENSE).
