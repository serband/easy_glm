"""Minimal usage example for easy_glm after installation.
Run with: python -m examples.basic_usage (from project root when editable-installed)
"""
import easy_glm
import polars as pl
import numpy as np

# Load demo dataset
raw = easy_glm.load_external_dataframe()

# Create train/test flag (70% train)
raw = raw.with_columns(
    (pl.Series(np.random.rand(raw.height)) < 0.7).cast(pl.Int8).alias("traintest")
)

predictors = ["VehAge","Region","VehGas","DrivAge","BonusMalus","Density"]

# Build blueprint (bin numeric & gather categorical levels)
blueprint = easy_glm.generate_blueprint(raw)

# Prepare data
prepped = easy_glm.prepare_data(
    df=raw,
    modelling_variables=predictors,
    additional_columns=["Exposure","ClaimNb","traintest"],
    formats=blueprint,
    traintest_column="traintest",
    table_name="cars"
)

# Fit L1 GLM (Poisson frequency model)
model = easy_glm.fit_lasso_glm(
    dataframe=prepped,
    target="ClaimNb",
    model_type="Poisson",
    weight_col="Exposure",
    train_test_col="traintest",
    DivideTargetByWeight=True,
)

# One rate table
vehage_tbl = easy_glm.ratetable(
    model=model,
    dataset=raw,
    col_name="VehAge",
    levels=blueprint["VehAge"],
    prepare=lambda d: easy_glm.prepare_data(
        df=d,
        modelling_variables=predictors,
        formats=blueprint,
        table_name="line_prepped",
    ),
    random_seed=42,
)
print(vehage_tbl.head())

# All rate tables
all_tables = easy_glm.generate_all_ratetables(
    model=model,
    dataset=raw,
    predictor_variables=predictors,
    blueprint=blueprint,
)
print(f"Generated {len(all_tables)} rate tables: {list(all_tables)}")
