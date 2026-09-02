"""Step-by-step pipeline: DesignSpec → fit_glm → rate tables → RateModel.

Use this when you need control between stages (hand-tuned knots, several
fits on one design, monotone constraints, inspecting coefficients). Most
users should start with ``basic_usage.py`` (``EasyGLM.fit``).

Run as a script:
    python examples/advanced_pipeline.py
"""

import numpy as np
import polars as pl

import easy_glm
from easy_glm.engine import RateModel

# ---------------------------------------------------------------------------
# 0. Load data & split
# ---------------------------------------------------------------------------

df = easy_glm.load_external_dataframe()
rng = np.random.default_rng(42)
df = df.with_columns(pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64))
train_df = df.filter(pl.col("traintest") == 1)
holdout = df.filter(pl.col("traintest") == 0)

PREDICTORS = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

# ---------------------------------------------------------------------------
# 1. Design spec — how each predictor becomes features. Built on TRAIN rows
#    only. Numeric -> step knots (1{x >= k}), categorical -> one-hot + Other.
# ---------------------------------------------------------------------------

spec = easy_glm.DesignSpec.from_data(
    train_df,
    PREDICTORS,
    n_bins=20,  # quantile knots per numeric variable
    min_level_share=0.0025,  # rarer levels are lumped into "Other"
    knots={"VehAge": list(range(1, 21))},  # or hand-pick knots per variable
    weight_col="Exposure",  # level frequencies weighted by exposure
)
print(spec)
print("VehAge knots:", spec["VehAge"].knots[:5], "...")
print(
    "Region levels:",
    spec["Region"].levels[:4],
    "... reference =",
    spec["Region"].reference,
)
print("Design matrix:", spec.n_features, "columns")
spec.to_json("french_motor_spec.json")  # JSON round-trip; edit by hand if you like

# ---------------------------------------------------------------------------
# 2. Fit — L1-penalised GLM via glum on spec.build(train). Pass alpha for a
#    quick fit or cv=k to pick it by cross-validation. Optional monotone
#    constraints on numeric variables (sign bounds on the step increments).
# ---------------------------------------------------------------------------

fit = easy_glm.fit_glm(
    train_df,
    spec,
    target="ClaimNb",
    family="poisson",
    weight_col="Exposure",
    divide_target_by_weight=True,  # frequency = ClaimNb / Exposure
    alpha=0.001,  # or: cv=5, n_alphas=20
    monotone={"BonusMalus": "increasing"},
)
print(fit)
print(fit.coef_table(drop_zero=True))  # only the knots/levels the lasso kept

# ---------------------------------------------------------------------------
# 3. Rate tables — exact, straight off the coefficients. Relativity 1.0 sits
#    on the most exposed bin (base="modal"); base_rate is the prediction for
#    that base risk.
# ---------------------------------------------------------------------------

tables = easy_glm.rate_tables(fit)
print("\nVehAge table:")
print(tables["VehAge"])
print("base rate:", easy_glm.base_rate(fit))

# ---------------------------------------------------------------------------
# 4. RateModel — portable lookup-table scorer; reproduces the GLM exactly.
# ---------------------------------------------------------------------------

rm = easy_glm.to_rate_model(fit, exposure_col="Exposure", train_test_col="traintest")
glm_pred = fit.predict(holdout)  # per unit exposure
rm_pred = rm.predict(holdout, exposure_col=None)
print("\nmax |RateModel / GLM - 1| on holdout:", np.abs(rm_pred / glm_pred - 1).max())
# → ~1e-16

# ---------------------------------------------------------------------------
# 5. A/E on holdout, per variable
# ---------------------------------------------------------------------------

holdout_freq = holdout.with_columns(
    (pl.col("ClaimNb") / pl.col("Exposure")).alias("ClaimNb")
)
overall = holdout["ClaimNb"].sum() / rm.predict(holdout).sum()
print(f"\nOverall holdout A/E: {overall:.4f}")
for var in PREDICTORS:
    buckets = rm.compute_ae_for_variable(holdout_freq, var)["subsets"]["all"]
    actual = sum(b["actual"] * b["exposure"] for b in buckets)
    expected = sum(b["expected"] * b["exposure"] for b in buckets)
    print(f"  {var:12s} A/E = {actual / expected:.4f}  ({len(buckets)} rows)")

# ---------------------------------------------------------------------------
# 6. Optional charts, save & reload
# ---------------------------------------------------------------------------

# easy_glm.plot_all_ratetables(tables)

rm.to_json("french_motor.easyglm")
loaded = RateModel.from_json("french_motor.easyglm")
print("\nReloaded predictions:", loaded.predict(holdout.head(3)))
