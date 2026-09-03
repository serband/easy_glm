"""End-to-end demo: fit, inspect, save, and optionally launch the editor.

This is a longer interactive walkthrough.  For a shorter first read see
``basic_usage.py``.

Run as a script:
    python examples/easy_glm_demo.py
"""

from pathlib import Path

import numpy as np
import polars as pl

import easy_glm
from easy_glm.engine import RateModel

# ---------------------------------------------------------------------------
# Config — edit these to taste
# ---------------------------------------------------------------------------

SAMPLE_SIZE = 10_000  # rows to use (0 = full dataset)
BASE_RATE = 0.05
USE_CV = True
LAUNCH_EDITOR = False  # set True to open the relativity editor in a browser tab
EDITOR_PORT = 8501

PREDICTORS = [
    "VehAge",
    "Region",
    "VehGas",
    "DrivAge",
    "BonusMalus",
    "Density",
]

# ---------------------------------------------------------------------------
# 1. Load & sample
# ---------------------------------------------------------------------------

print("Loading the checked-in French motor sample...")
DATA = Path(__file__).resolve().parents[1] / "tests/fixtures/french_motor_50k.parquet"
df = pl.read_parquet(DATA)

if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
    df = df.sample(n=SAMPLE_SIZE, seed=42)

rng = np.random.default_rng(42)
df = df.with_columns(pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64))
print(f"  {len(df):,} rows × {len(df.columns)} cols")

# ---------------------------------------------------------------------------
# 2. Fit
# ---------------------------------------------------------------------------

print("\nFitting EasyGLM...")
eglm = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=PREDICTORS,
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    cv=5 if USE_CV else None,
    alpha=None if USE_CV else 0.001,
    base_rate=BASE_RATE,
)

print(eglm.summary())
# → {'model_type': 'Poisson', 'target': 'ClaimNb', 'weight_col': 'Exposure', ...}
print(f"  Intercept: {eglm.model.intercept_:.4f}")
print(
    f"  Non-constant variables: {list(eglm.rate_model.non_constant_variables.keys())}"
)

# ---------------------------------------------------------------------------
# 3. Sample predictions
# ---------------------------------------------------------------------------

sample = df.select(PREDICTORS).head(5)
preds = eglm.rate_model.predict(sample)
print(f"\nSample predictions: {preds.round(6)}")
# → [0.0342 0.0517 0.0289 0.0412 0.0365]

# ---------------------------------------------------------------------------
# 4. Holdout A/E
# ---------------------------------------------------------------------------

holdout = df.filter(pl.col("traintest") == 0)
holdout_preds = eglm.rate_model.predict(holdout)
ae_overall = holdout["ClaimNb"].sum() / holdout_preds.sum()
print(f"\nHoldout A/E (overall): {ae_overall:.4f}")

# ---------------------------------------------------------------------------
# 5. Per-variable A/E drill-down (spread across each variable's own bands)
# ---------------------------------------------------------------------------

from easy_glm.workflow import ae_by_variable  # noqa: E402

print("\nPer-variable A/E on holdout:")
holdout_actual = holdout["ClaimNb"].to_numpy()
holdout_weight = holdout["Exposure"].to_numpy()
for var in PREDICTORS:
    table = ae_by_variable(holdout, var, holdout_actual, holdout_preds, holdout_weight)
    print(
        f"  {var:15s}  A/E ranges {table['ae'].min():.3f}-{table['ae'].max():.3f}  "
        f"({table.height} bins)"
    )

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------

eglm.save("demo_model")
eglm.rate_model.to_json("demo_model.easyglm")
print("\nSaved → demo_model/  and  demo_model.easyglm")

# ---------------------------------------------------------------------------
# 7. Reload and score
# ---------------------------------------------------------------------------

loaded = RateModel.from_json("demo_model.easyglm")
print(
    f"Reloaded — {len(loaded.variables)} variables, "
    f"base_rate={loaded.base_rate}, "
    f"{len(loaded.snapshots)} snapshot(s)"
)

new_business = df.select(PREDICTORS).head(3)
print(f"Reloaded predictions: {loaded.predict(new_business).round(6)}")

# ---------------------------------------------------------------------------
# 8. Launch the relativity editor (browser)
# ---------------------------------------------------------------------------

if LAUNCH_EDITOR:
    print(f"\nLaunching editor at http://localhost:{EDITOR_PORT} ...")
    print("(Close the browser tab and Ctrl-C this script when done.)")
    eglm.launch_editor(data=df, port=EDITOR_PORT)
