"""Lightweight developer smoke script (not a formal test).
Run manually: python test.py
"""


import polars as pl
import numpy as np
import easy_glm

raw = easy_glm.load_external_dataframe()
raw = raw.with_columns(((pl.Series(np.random.rand(raw.height)) < 0.7).cast(pl.Int8)).alias("traintest"))

predictors = ["VehAge","Region","VehGas","DrivAge","BonusMalus","Density"]
blueprint = easy_glm.generate_blueprint(raw)
prepped = easy_glm.prepare_data(
    df=raw,
    modelling_variables=predictors,
    additional_columns=["Exposure","ClaimNb","traintest"],
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
    DivideTargetByWeight=True,
)
veh_tbl = easy_glm.ratetable(
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
print("VehAge table sample:\n", veh_tbl.head())
all_tables = easy_glm.generate_all_ratetables(
    model=model,
    dataset=raw,
    predictor_variables=predictors,
    blueprint=blueprint,
)
print("Generated tables:", list(all_tables))

if __name__ == "__main__":
    print("Smoke run complete.")

