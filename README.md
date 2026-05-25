# easy_glm

Python package to automate building insurance ratetables using (fused) LASSO regularised GLMs. Internally it leverages [glum](https://glum.readthedocs.io/en/latest/) for fitting, providing a higher-level interface tailored to insurance pricing workflows (blueprints, preprocessing, model fitting, rate table extraction & plotting). Inspired by the R package [aglm](https://github.com/kkondo1981/aglm). Packaged with a modern `src/` layout.

## Project Status

| Feature | Status |
|---|---|
| Blueprint generation (auto-detect numeric/categorical, quantile breaks, rare-level lumping) | ✅ |
| Data preparation (DuckDB-powered SQL transforms, null handling, o-matrix binarization) | ✅ |
| GLM fitting (LASSO, CV and non-CV, Poisson/Gamma/Gaussian/Binomial) | ✅ |
| Rate table extraction (per-variable relativities from fitted model) | ✅ |
| RateModel engine (.easyglm JSON export, scoring, versioning, snapshots) | ✅ |
| EasyGLM one-shot pipeline (fit → predict → serialize) | ✅ |
| Streamlit Relativity Editor (baseline vs. working copy, A/E, save named models) | ✅ |
| Benchmarking suite (easy_glm vs statsmodels vs CatBoost) | ✅ |
| CI (3.10–3.13, lint, format, test, coverage) | ✅ |

### Roadmap (v0.2 → v1.0)

- [ ] Automated monotonic binning / isotonic smoothing of rate tables
- [ ] CLI entry point (`python -m easy_glm build ...`)
- [ ] Configurable blueprint strategies (equal-frequency vs fixed breaks)
- [ ] GAMChanger-style interactive relativity editing (drag points on curve)
- [ ] `mypy` type-checking in CI
- [ ] Multi-model comparison in the editor (side-by-side A/E for multiple saved models)

---

## Installation & Setup

This project uses `uv` for fast dependency management and `venv` for virtual environments.

### Prerequisites

1. **Python 3.10–3.13** — CI tests these versions.
2. **uv** — Fast Python package installer and resolver

```bash
# On Unix/Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installing from Git (Single Command)

```bash
uv pip install git+https://github.com/serband/easy_glm.git
```

### Quick Setup

```bash
python setup_dev.py
```

---

## Usage

### Quick Start (One-Shot Pipeline)

The `EasyGLM` class bundles the entire workflow — blueprint
generation, data preparation, CV model fitting, and rate table
extraction — into a single call.

```python
import easy_glm
import polars as pl
import numpy as np

df = easy_glm.load_external_dataframe()
df = df.with_columns(
    pl.when(pl.lit(np.random.rand(df.height) < 0.7))
    .then(1).otherwise(0).alias("traintest")
)

predictors = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

eglm = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=predictors,
    weight_col="Exposure",
    divide_target_by_weight=True,
    use_cv=True,
    base_rate=0.05,
)

print(f"CV alpha: {eglm.model.alpha_:.6f}")
print(f"Intercept: {eglm.model.intercept_:.4f}")

# Predict on raw data
preds = eglm.predict(df.head(10))

# Export as .easyglm (portable JSON model)
eglm.rate_model.to_json("model.easyglm")

# Serialize the entire pipeline (including blueprint + fitted GLM)
eglm.save("my_model")
reloaded = easy_glm.EasyGLM.load("my_model")
```

### Step-by-Step Workflow

If you need fine-grained control over each stage:

#### 1. Generate a Blueprint

```python
blueprint = easy_glm.generate_blueprint(df)
# {'VehAge': [0.0, 2.0, 4.0, ...], 'Region': ['North', 'South', 'Urban', ...], ...}
```

The blueprint auto-detects numeric vs categorical columns.
Numeric columns get quantile breakpoints (5% steps). Categorical
columns get retained levels after lumping rare ones into 'Other'.

#### 2. Prepare Data

```python
prepped = easy_glm.prepare_data(
    df=df,
    modelling_variables=predictors,
    additional_columns=["Exposure", "ClaimNb", "traintest"],
    formats=blueprint,
    traintest_column="traintest",
    table_name="cars",
)
# Numeric vars → binarised via o-matrix expansion
# Categorical vars → lumped via CASE WHEN logic
```

#### 3. Fit the LASSO GLM

```python
model = easy_glm.fit_lasso_glm(
    dataframe=prepped,
    target="ClaimNb",
    model_type="Poisson",
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    use_cv=True,              # CV selects optimal alpha / l1_ratio
)
```

Available families: `"Poisson"`, `"Gamma"`, `"Gaussian"`, `"Binomial"`.

#### 4. Extract Rate Tables

```python
all_tables = easy_glm.generate_all_ratetables(
    model=model,
    dataset=df,
    predictor_variables=predictors,
    blueprint=blueprint,
)
# Returns a dict[str, pl.DataFrame] — one table per variable
print(all_tables["DrivAge"].head())
```

#### 5. Build a RateModel for Scoring

```python
from easy_glm.engine import RateModel

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
```

#### 6. Score & Validate

```python
# Score new data
test = df.filter(pl.col("traintest") == 0)
preds = rm.predict(test)
print(f"Test A/E: {test['ClaimNb'].sum() / preds.sum():.4f}")

# Score without exposure multiplication
raw_preds = rm.predict(test, exposure_col=None)

# Use a differently-named exposure column
renamed = test.rename({"Exposure": "Exp"})
preds = rm.predict(renamed, exposure_col="Exp")

# Compute actual vs expected for a single variable
ae = rm.compute_ae_for_variable(test, "DrivAge")
for bucket in ae["subsets"]["train"]:
    print(f"{bucket['level']}: actual={bucket['actual']:.3f}, "
          f"expected={bucket['expected']:.3f}")
```

#### 7. Serialize the Rate Model

```python
rm.to_json("french_motor.easyglm")

# Reload — predictions are identical
loaded = RateModel.from_json("french_motor.easyglm")
np.testing.assert_array_equal(rm.predict(test), loaded.predict(test))
```

The `.easyglm` format is human-readable JSON containing From/To/Relativity
lookup tables, model metadata, and a full history of versioned snapshots.

### Relativity Editor

Launch the interactive editor to visually refine the model's relativities.
The original model is never modified — all edits go into a working copy.

```python
rm.launch_editor(data=df)
```

This opens a new browser tab (non-blocking — your Python session continues).

**Editor layout:**

| Section | Description |
|---|---|
| **Sidebar** | Variable overview (non-constant only), column mapping, A/E formula, save/reset controls, saved models list |
| **Relativity chart** | Overlaid original (gray dashed) and revised (blue solid) relativities on the same axes |
| **A/E chart** | Overlaid faded-original vs solid-revised actual vs expected, with exposure bars behind |
| **Editable table** | *Original* column (read-only) alongside *Revised* column (editable) — change a value and both charts update |
| **Distribution** | Expandable histogram of the selected variable |

**Workflow:**

1. Select a variable from the sidebar to see its relativities
2. Edit the *Revised* values in the table — the charts update reactively
3. Toggle **Auto-recompute** off for large datasets, then click **Recompute A/E** manually
4. Type a name and click **Save Working Copy** to store the revision in-memory
5. Click **Download** to export the revision as a `.easyglm` file
6. Click **Reset Working Copy** to discard all edits and start over

**Using a saved revision:**

```python
# Saved models are standard RateModel instances
revised = RateModel.from_json("my_revision_v1.easyglm")
new_preds = revised.predict(test)
```

### Complete End-to-End Script

```python
import easy_glm
import polars as pl
import numpy as np
from easy_glm.engine import RateModel

# 1. Load and split data
df = easy_glm.load_external_dataframe()
df = df.with_columns(
    pl.when(pl.lit(np.random.rand(df.height)) < 0.7)
    .then(1).otherwise(0).alias("traintest")
)

predictors = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

# 2. Build model with CV
eglm = easy_glm.EasyGLM.fit(
    data=df, target="ClaimNb", model_type="Poisson",
    predictors=predictors, weight_col="Exposure",
    divide_target_by_weight=True, use_cv=True, base_rate=0.05,
)

# 3. Export and score
eglm.rate_model.to_json("model.easyglm")
rm = RateModel.from_json("model.easyglm")

test = df.filter(pl.col("traintest") == 0)
preds = rm.predict(test)
ratio = test["ClaimNb"].sum() / preds.sum()
print(f"Overall test A/E: {ratio:.4f}")

# 4. Per-variable calibration
for var in rm.non_constant_variables:
    ae = rm.compute_ae_for_variable(test, var)
    ratios = []
    for bucket in ae["subsets"].get("test", ae["subsets"]["all"]):
        if bucket["expected"] > 0:
            ratios.append(bucket["actual"] / bucket["expected"])
    if ratios:
        print(f"  {var}: A/E range [{min(ratios):.3f}, {max(ratios):.3f}]")

# 5. Launch the editor to refine
rm.launch_editor(data=df)
```

---

## Architecture

```
easy_glm/
├── src/easy_glm/
│   ├── __init__.py          # Public API (EasyGLM, prepare_data, fit_lasso_glm, ...)
│   ├── core/
│   │   ├── blueprint.py     # generate_blueprint (quantile breaks, level lumping)
│   │   ├── prepare.py       # prepare_data (DuckDB SQL transforms)
│   │   ├── model.py         # fit_lasso_glm, predict_with_model
│   │   ├── ratetable.py     # ratetable (per-variable relativity extraction)
│   │   ├── all_ratetables.py # generate_all_ratetables
│   │   ├── transforms.py    # o_matrix, lump_fun, lump_rare_levels_pl
│   │   ├── data.py          # load_external_dataframe (with caching)
│   │   ├── plots.py         # plot_all_ratetables (matplotlib/seaborn)
│   │   └── easyglm.py       # EasyGLM pipeline class (fit/predict/save/load)
│   ├── engine/
│   │   ├── rate_model.py    # RateModel (predict, clone, update_relativity,
│   │   │                      snapshots, JSON roundtrip, compute_ae_for_variable)
│   │   ├── _scoring.py      # score_numeric (np.searchsorted), score_categorical
│   │   └── models.py        # Dataclasses: FromToRow, VariableConfig, Snapshot, ...
│   ├── ui/
│   │   ├── __init__.py      # launch_editor (non-blocking Streamlit subprocess)
│   │   ├── app.py           # Streamlit relativity editor
│   │   ├── charts.py        # Plotly charts (histogram, relativity, A/E)
│   │   └── metrics.py       # compute_actual_expected, formula helpers
│   └── benchmarking/
│       ├── benchmark.py     # run_benchmarks (easy_glm vs statsmodels vs CatBoost)
│       ├── data_generators.py # Synthetic data for Poisson/Gamma/Gaussian/Binomial
│       └── metrics.py       # Deviance, RMSE, MAE per family
├── tests/                   # 121 tests (pytest)
├── examples/                # basic_usage.py, benchmark_demo.py, easy_glm_demo.py
└── pyproject.toml
```

---

## Dependencies

| Category | Packages |
|---|---|
| Core | `polars`, `numpy`, `pyarrow`, `glum`, `pandas`, `scikit-learn`, `duckdb` |
| Plotting | `matplotlib`, `seaborn` |
| Data | `rdata` (French Motor dataset) |
| GLM internals | `numba`, `llvmlite` (required by glum) |
| UI (optional) | `streamlit`, `plotly` |
| Viz (optional) | `plotnine` |
| Dev | `pytest`, `pytest-cov`, `black`, `ruff`, `jupyter`, `build`, `twine` |

> **Note:** `dask-ml` was removed in v0.2 in favor of pandas native `astype("category")`,
> eliminating ~100MB of transitive dependencies (dask, distributed, etc.).

---

## Development

### Code Quality

```bash
black .           # Format
ruff check .      # Lint
pytest -q         # Run tests (121 tests)
```

### Test Performance Tuning

CI sets `EASY_GLM_MAX_ROWS=500` to limit dataset size for quicker tests. Mimic locally:
```bash
export EASY_GLM_MAX_ROWS=500
pytest -q
```

---

## Contributing

See `CONTRIBUTING.md`. Quick checklist:
```bash
ruff check .
black .
pytest -q
```

## License

MIT — see `LICENSE`.
