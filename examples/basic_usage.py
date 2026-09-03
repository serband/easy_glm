"""First frequency model: fit, interpret, validate and save a scorer.

Run this first in the example curriculum:

    python examples/basic_usage.py

It writes ``my_model.easyglm``. Pass that file to ``exploring_fit.py`` for
post-fit review or ``scoring_editor.py`` to score new business.
"""

from pathlib import Path

import numpy as np
import polars as pl

import easy_glm

# One policy period per row. ClaimNb is the claim count; Exposure is time on
# risk. Dividing the target by exposure therefore fits claim frequency.
DATA = Path(__file__).resolve().parents[1] / "tests/fixtures/french_motor_50k.parquet"
df = pl.read_parquet(DATA)
rng = np.random.default_rng(42)
df = df.with_columns(pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64))

PREDICTORS = ["DrivAge", "Region", "BonusMalus", "Density"]
model = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=PREDICTORS,
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    alpha=0.001,  # use cv=5 when choosing a production penalty
)

print(model)
print("\nBonus-malus table:")
print(model.relativities["BonusMalus"].select("label", "relativity", "exposure"))

holdout = df.filter(pl.col("traintest") == 0)
expected = model.rate_model.predict(holdout)
ae = holdout["ClaimNb"].sum() / expected.sum()
print(f"\nHoldout A/E: {ae:.3f}")
print("A/E near 1 is an overall check. Review each factor before changing a price.")

model.rate_model.to_json("my_model.easyglm")
print("\nWrote my_model.easyglm")
