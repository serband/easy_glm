"""Minimal EasyGLM usage (recommended entry point).

Run from project root:
    PYTHONPATH=src python examples/basic_usage.py
"""

import numpy as np
import polars as pl

import easy_glm

raw = easy_glm.load_external_dataframe()
raw = raw.with_columns(
    (pl.Series(np.random.rand(raw.height)) < 0.7).cast(pl.Int8).alias("traintest")
)

predictors = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

eglm = easy_glm.EasyGLM.fit(
    data=raw,
    target="ClaimNb",
    model_type="Poisson",
    predictors=predictors,
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    use_cv=True,
    base_rate=0.1,
)

print(eglm.summary())
print(eglm.predict(raw.head(3)))

eglm.rate_model.to_json("french_motor.easyglm")
print("Wrote french_motor.easyglm")

test = raw.filter(pl.col("traintest") == 0)
preds = eglm.rate_model.predict(test)
print(f"Test A/E: {test['ClaimNb'].sum() / preds.sum():.4f}")
