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

## Usage Example

For a complete runnable script, see `examples/basic_usage.py`.

### 1. Load Data & Build a CV-Fitted Model

easy_glm uses cross-validation by default to find the optimal
LASSO regularisation strength. The `EasyGLM` class bundles
blueprint generation, data preparation, model fitting, and rate
table extraction into a single call.

```python
import easy_glm
import polars as pl
import numpy as np

# Load the French Motor dataset (cached after first download)
df = easy_glm.load_external_dataframe()

# Train/test split
df = df.with_columns(
    pl.when(pl.lit(np.random.rand(df.height) < 0.7))
    .then(1).otherwise(0).alias("traintest")
)

predictors = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

# One-shot pipeline with CV: blueprint → prep → fit → rate tables
eglm = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=predictors,
    weight_col="Exposure",
    divide_target_by_weight=True,
    use_cv=True,            # cross-validate regularisation strength
    base_rate=0.05,
)

print(f"CV selected alpha: {eglm.model.alpha_:.6f}")
```

### 2. Export as .easyglm & Score New Data

```python
# Export the portable model
eglm.rate_model.to_json("french_motor.easyglm")

# Reload and score
from easy_glm.engine import RateModel

rm = RateModel.from_json("french_motor.easyglm")
test = df.filter(pl.col("traintest") == 0)
preds = rm.predict(test)

# Overall calibration
print(f"Test A/E: {test['ClaimNb'].sum() / preds.sum():.4f}")
```

### 3. Refine Relativities with the Editor

Launch the editor to visually adjust the model's relativities.
The original model is never modified — all edits go into a working
copy, which you can save as a named revision.

```python
rm.launch_editor(data=df)
```

**What you'll see:**

- **Overlaid relativity chart** — original (gray dashed) and revised
  (blue solid) on the same axes, so you can see exactly what changed.
- **Overlaid A/E chart** — faded original vs. solid revised,
  with exposure bars behind. Auto-recomputes on every edit.
- **Editable table** — shows *Original* relativity (read-only) alongside
  *Revised* (editable). Change a value, and both charts update.
- **Sidebar** — lists only variables with non-constant relativities.
  Click any variable to jump to it.

**Saving your work:**

Type a name (e.g. `my_revision_v1`) and click **Save Working Copy**.
The revision is stored in-memory and appears in the Saved Models list.
Click **Download** to export it as a `.easyglm` file.

**Resetting:**

Click **Reset Working Copy** to discard all edits and start fresh from
the original model.

### 4. Using a Saved Revision

```python
# The revision is a standard RateModel — score with it directly
revised = RateModel.from_json("my_revision_v1.easyglm")
revised_preds = revised.predict(test)
print(f"Revised test A/E: {test['ClaimNb'].sum() / revised_preds.sum():.4f}")
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
