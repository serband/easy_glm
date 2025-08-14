# easy_glm

Python package to automate building insurance ratetables using (fused) LASSO regularised GLMs. Internally it leverages [GLUM](https://glum.readthedocs.io/en/latest/) for fitting, providing a higher-level interface tailored to insurance pricing workflows (blueprints, preprocessing, model fitting, rate table extraction & plotting). Inspired by the R package [aglm](https://github.com/kkondo1981/aglm). Packaged with a modern `src/` layout.

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
    additional_columns=['Exposure', 'ClaimNb'], 
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
│   └── core/            # Core implementation modules
├── tests/               # Pytest test suite
├── examples/            # Usage examples
├── test.py              # Lightweight smoke script
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

### Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatter
- **ruff**: Fast Python linter
- **jupyter**: Notebook environment

## Additional Usage Ideas

Roadmap ideas:
* Export all ratetables to CSV / Parquet bundle
* Inverse transform scoring for new raw data (auto-prepare + predict)
* Automated monotonic binning / isotonic smoothing option
* CLI entry point (`python -m easy_glm build ...`)
* Optional caching of downloaded demo dataset
* Configurable blueprint strategies (equal-frequency vs fixed breaks)

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
