# easy_glm

Python package to automate building insurance ratetables using (fused) LASSO regularised GLMs. Internally it leverages [glum](https://glum.readthedocs.io/en/latest/) for fitting, providing a higher-level interface tailored to insurance pricing workflows (blueprints, preprocessing, model fitting, rate table extraction & plotting). Inspired by the R package [aglm](https://github.com/kkondo1981/aglm). Packaged with a modern `src/` layout.

## Installation & Setup

This project uses `uv` for fast dependency management and `venv` for virtual environments to ensure reproducibility.

### Prerequisites

1. **Python 3.10–3.13** - CI tests these versions.
2. **uv** - Fast Python package installer and resolver

Install `uv`:
```bash
# On Unix/Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installing from Git (Single Command)

You can install the package directly from Git using a single command:

```bash
uv pip install git+https://github.com/serband/easy_glm.git
```

This is the fastest way to get started with easy_glm without cloning the repository.

### Quick Setup

Choose one of the following methods to set up your development environment:

#### Option 1: Cross-platform Python script (Recommended)
```bash
python setup_dev.py
```

#### Option 2: Direct installation from Git
```bash
uv pip install git+https://github.com/serband/easy_glm.git
```

#### Option 3: Platform-specific scripts

**On Windows (PowerShell):**
```powershell
.\setup_dev.ps1
```

**On Unix/Linux/macOS:**
```bash
chmod +x setup_dev.sh
./setup_dev.sh
```

### Manual Setup

If you prefer to set up manually:

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment:**
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On Unix/Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   uv pip install -r requirements-dev.txt
   uv pip install -e .
   ```

## Usage Example

Here's a complete example of how to use `easy_glm` to build and visualize insurance rate tables.

For a minimal runnable script, see `examples/basic_usage.py`.

### 1. Import Libraries and Load Data

First, import the necessary libraries and load the sample dataset. The package includes a function to load a sample French motor insurance dataset.

```python
import easy_glm 
import polars as pl
import numpy as np

# Load the sample dataset
df = easy_glm.load_external_dataframe()

# Create a train-test split for validation
df = df.with_columns(
    pl.when(pl.lit(np.random.rand(df.height) < 0.7))
    .then(1)
    .otherwise(0)
    .alias("traintest")
)
```

### 2. Generate a Preprocessing Blueprint

The `generate_blueprint` function analyzes the dataframe and creates a "blueprint" that defines how each variable should be preprocessed for modeling.
- **Numeric columns**: It computes quantile breakpoints.
- **Categorical columns**: It identifies the levels to keep, lumping rare ones into an 'Other' category.

```python
# Generate the blueprint for the dataset 
blueprint = easy_glm.generate_blueprint(df)
```

### 3. Prepare Data for Modeling

Using the blueprint, the `prepare_data` function transforms the raw data into a feature matrix suitable for the GLM. It applies the transformations defined in the blueprint (binning for numerics, lumping for categoricals).

```python
# Define predictor variables
predictor_variables = ['VehAge', 'Region', 'VehGas', 'DrivAge', 'BonusMalus', 'Density']

# Prepare the dataset for modelling
prepped_data = easy_glm.prepare_data(
    df=df, 
    modelling_variables=predictor_variables, 
    additional_columns=['Exposure', 'ClaimNb', 'traintest'], 
    formats=blueprint, 
    traintest_column='traintest', 
    table_name='cars'
)
```

### 4. Fit the LASSO GLM

Fit a LASSO-regularized Generalized Linear Model (GLM) using the prepared data. The `fit_lasso_glm` function uses cross-validation to find the optimal regularization strength.

```python
# Fit the model
model = easy_glm.fit_lasso_glm(
    dataframe=prepped_data, 
    target="ClaimNb", 
    model_type="Poisson", 
    weight_col="Exposure", 
    train_test_col="traintest",
    divide_target_by_weight=True
)
```

### 5. Predict on New Data (Optional)

If you have already prepared data (i.e. ran `prepare_data` with the same blueprint & predictors) you can obtain predictions using the helper:

```python
# Assume `prepped_data` as above and `model` fitted
new_rows_prepped = prepped_data.head(10).select(pl.all().exclude(["ClaimNb", "Exposure", "traintest"]))
preds = easy_glm.predict_with_model(model, new_rows_prepped)
```

If you start from raw rows, run `prepare_data` first with the same `formats` (blueprint) and predictor list.

**Alternatively, use the `EasyGLM` pipeline** (see [section below](#8-easyglm-pipeline-one-shot-approach)) — it handles blueprint prepping internally, so you can predict directly from raw data:

```python
preds = eglm.predict(raw_data)
```

### 6. Generate All Rate Tables

With a fitted model, you can now generate the rate tables for all predictor variables. The `generate_all_ratetables` function loops through each variable and calculates its relativity.

```python
# Generate rate tables for all predictor variables
all_tables = easy_glm.generate_all_ratetables(
    model=model,
    dataset=df,
    predictor_variables=predictor_variables,
    blueprint=blueprint
)

# You can access the rate table for a specific variable like this:
print(all_tables['VehAge'])
```

### 7. Plot the Rate Tables

Finally, visualize the relativities using the `plot_all_ratetables` function. This will generate a plot for each variable, making it easy to interpret the model's results.
- **Numeric variables** are shown as line plots.
- **Categorical variables** are shown as bar plots.

```python
# Plot all rate tables
easy_glm.plot_all_ratetables(all_tables, blueprint)
```

This will produce a series of plots, one for each variable.

### 8. Export as .easyglm & Score on New Data

The `RateModel` (in `easy_glm.engine`) converts rate tables into a portable JSON format
(``.easyglm``) with From/To/Relativity lookup tables, metadata, and versioning for audit trails.
This is the format you use to deploy the model and score new data.

**Create and export:**

```python
from easy_glm.engine import RateModel

rm = RateModel.from_rate_tables(
    all_tables=all_tables,
    blueprint=blueprint,
    base_rate=0.1,
    model_type="poisson",
    target="ClaimNb",
    weight_col="Exposure",
    exposure_col="Exposure",
)
rm.to_json("model.easyglm")
```

Or use the convenience wrapper:

```python
from easy_glm.engine import create_rate_model

rm = create_rate_model(all_tables, blueprint, base_rate=0.1,
                       model_type="poisson", exposure_col="Exposure",
                       save_to="model.easyglm")
```

**Score new data:**

```python
loaded = RateModel.from_json("model.easyglm")
preds = loaded.predict(new_data)                     # uses stored exposure
preds = loaded.predict(new_data, exposure_col=None)   # skip exposure
preds = loaded.predict(new_data, exposure_col="Exp2") # override exposure column
```

**Calibration (Actual vs Expected):**

See `examples/basic_usage.py` for the full end-to-end workflow including scoring on the test
set, bucketing with `qcut`, and per-variable actual-vs-expected matplotlib charts.

### 9. EasyGLM Pipeline (One-Shot Approach)

The `EasyGLM` class bundles blueprint generation, data preparation, model fitting (with cross-validation by default), and rate table extraction into a single pipeline. It also handles serialization so you can save and reload trained models.

**Fit + predict + save in one shot:**

```python
import easy_glm

# Load data and create train/test split (same as Section 1)
raw = easy_glm.load_external_dataframe()
raw = raw.with_columns(
    pl.when(pl.lit(np.random.rand(raw.height) < 0.7))
    .then(1).otherwise(0).alias("traintest")
)

predictors = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

# One call: blueprint → prep → fit (CV by default) → rate tables
eglm = easy_glm.EasyGLM.fit(
    data=raw,
    target="ClaimNb",
    model_type="Poisson",
    predictors=predictors,
    weight_col="Exposure",
    divide_target_by_weight=True,
    base_rate=0.05,
)

# Predict on raw data — blueprint prepping is handled internally
preds = eglm.predict(raw.head(10))

# Access rate tables directly
for name, tbl in eglm.relativities.items():
    print(f"{name}:\n{tbl.head(3)}")

# Serialize the entire pipeline
eglm.save("my_model")

# Reload later
reloaded = easy_glm.EasyGLM.load("my_model")
```

#### CV vs non-CV

By default, `EasyGLM.fit()` uses `GeneralizedLinearRegressorCV` for automatic alpha and l1_ratio selection via cross-validation. To disable CV and use the simpler `alpha_search` approach, pass `use_cv=False`:

```python
eglm = easy_glm.EasyGLM.fit(
    data=raw, target="ClaimNb", model_type="Poisson",
    predictors=predictors, weight_col="Exposure",
    divide_target_by_weight=True, use_cv=False,
)
```

You can also customise CV parameters:

```python
eglm = easy_glm.EasyGLM.fit(
    ...,
    cv_params={"l1_ratio": [1.0], "n_alphas": 10, "max_iter": 200},
)
```

## Development

### Activating the Environment

After initial setup, activate your environment:

```bash
# On Windows
venv\Scripts\activate

# On Unix/Linux/macOS
source venv/bin/activate
```

### Code Quality

The project includes code quality tools:

```bash
# Format code
black .

# Lint code
ruff check .

# Run tests
pytest
```


## Project Structure

```
easy_glm/
├── src/easy_glm/        # Library code (packaged)
│   ├── core/            # Blueprint, transforms, model, ratetable, EasyGLM
│   ├── engine/          # RateModel with versioning & editor
│   └── ui/              # Streamlit editor for relativities
├── tests/               # Pytest test suite
├── examples/            # Usage examples
├── pyproject.toml       # Packaging configuration
├── requirements*.txt    # Dependency constraint files
├── setup_dev.*          # Dev environment helpers
└── README.md
```

## Dependencies

### Core Dependencies
- **duckdb**: Fast analytical database for data processing (v1.3+)
- **polars**: Fast dataframes library (v1.17+)
- **numpy**: Numerical computing
- **pyarrow**: Columnar data format
- **glum**: GLM implementation (v3.0+)
- **pandas**: Data manipulation and analysis
- **matplotlib**: Plotting library
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning utilities
- **dask-ml**: Text column encoding for model fitting
- **numba** / **llvmlite**: JIT compilation (required by glum)
- **rdata**: R .rda data file parsing (dataset loading)

### Optional Dependencies
- **ui**: `streamlit`, `plotly` — relativity editor web UI
- **viz**: `plotnine` — ggplot-style plotting

### Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatter
- **ruff**: Fast Python linter
- **jupyter**: Notebook environment

## Additional Usage Ideas

Roadmap ideas:
* ~~Export all ratetables to .easyglm portable model~~ ✓ (via `RateModel.to_json()`)
* ~~Inverse transform scoring for new raw data (auto-prepare + predict)~~ ✓ (via `EasyGLM.predict`)
* Automated monotonic binning / isotonic smoothing option
* CLI entry point (`python -m easy_glm build ...`)
* Optional caching of downloaded demo dataset
* Configurable blueprint strategies (equal-frequency vs fixed breaks)
* GAMChanger-style UI for interactive relativity editing

### Test Performance Tuning

CI sets `EASY_GLM_MAX_ROWS=500` to limit dataset size for quicker tests. You can mimic locally:
```bash
export EASY_GLM_MAX_ROWS=500
pytest -q
```

## Contributing

See `CONTRIBUTING.md` for the full guide. Quick checklist:
```bash
ruff check .
black .
pytest
```

## License

MIT – see `LICENSE`.
