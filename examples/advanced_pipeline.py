"""Step-by-step pipeline: DesignSpec → fit_glm → rate tables → RateModel.

Use this when you need control between stages (hand-tuned knots, several
fits on one design, monotone constraints, inspecting coefficients). Most
users should start with ``basic_usage.py`` (``EasyGLM.fit``).

Run as a script:
    python examples/advanced_pipeline.py
"""

from pathlib import Path

import numpy as np
import polars as pl

import easy_glm
from easy_glm.engine import RateModel

# ---------------------------------------------------------------------------
# 0. Load data & split
#
# The 50,000-row checked-in fixture (a sample of the public French motor
# third-party liability dataset) keeps this offline; swap in
# easy_glm.load_external_dataframe() for the full ~678k-row dataset.
# ---------------------------------------------------------------------------

DATA = Path(__file__).resolve().parents[1] / "tests/fixtures/french_motor_50k.parquet"
df = pl.read_parquet(DATA)
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
# 5. A/E on holdout, per variable — easy_glm.workflow.ae_by_variable works on
#    plain actual/expected/weight arrays, so it needs no RateModel at all;
#    it groups by the same bands the rate table uses.
# ---------------------------------------------------------------------------

from easy_glm.workflow import ae_by_variable  # noqa: E402

actual_total = holdout["ClaimNb"].to_numpy()
expected_total = rm.predict(holdout)  # already at total (per-policy) scale
weight = holdout["Exposure"].to_numpy()

overall = actual_total.sum() / expected_total.sum()
print(f"\nOverall holdout A/E: {overall:.4f}")
for var in PREDICTORS:
    # Summing every bin's actual/expected always reproduces the overall A/E
    # (they partition the same rows); what is informative is the *spread*
    # across bins — a well-calibrated factor keeps every bin's A/E near 1.
    table = ae_by_variable(holdout, var, actual_total, expected_total, weight)
    lo, hi = table["ae"].min(), table["ae"].max()
    print(f"  {var:12s} A/E ranges {lo:.3f}-{hi:.3f} across {table.height} bins")

# ---------------------------------------------------------------------------
# 6. Optional charts, save & reload
# ---------------------------------------------------------------------------

# easy_glm.plot_all_ratetables(tables)

rm.to_json("french_motor.easyglm")
loaded = RateModel.from_json("french_motor.easyglm")
print("\nReloaded predictions:", loaded.predict(holdout.head(3)))
