"""Score new business with the ``.easyglm`` file from ``advanced_pipeline.py``.

The file contains the fitted rate tables. Loading it does not refit the GLM.
"""

import polars as pl

import easy_glm

model = easy_glm.RateModel.from_json("french_motor.easyglm")

new_business = pl.DataFrame(
    {
        "DrivAge": [35, 52, 22],
        "Region": ["Ile-de-France", "Bretagne", "Nord-Pas-de-Calais"],
        "BonusMalus": [50, 68, 90],
        "Density": [2000, 500, 8000],
        "Exposure": [1.0, 0.5, 0.75],
    }
)

# Without exposure, the model returns annual claim frequency.
claim_frequency = model.predict(new_business, exposure_col=None)

# With exposure, the model returns expected claims for each record.
expected_claims = model.predict(new_business)

scored_business = new_business.with_columns(
    pl.Series("PredictedClaimFrequency", claim_frequency),
    pl.Series("ExpectedClaims", expected_claims),
)

print(scored_business)
