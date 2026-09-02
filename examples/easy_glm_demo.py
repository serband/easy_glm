"""End-to-end demo: fit, inspect, save, and optionally launch the editor.

This is a longer interactive walkthrough.  For a shorter first read see
``basic_usage.py``.

Run as a script:
    python examples/easy_glm_demo.py
"""

import numpy as np
import polars as pl

import easy_glm
from easy_glm.engine import RateModel

# ---------------------------------------------------------------------------
# Config — edit these to taste
# ---------------------------------------------------------------------------

SAMPLE_SIZE = 10_000          # rows to use (0 = full dataset)
BASE_RATE = 0.05
USE_CV = True
LAUNCH_EDITOR = True
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

print("Loading French motor insurance dataset...")
df = easy_glm.load_external_dataframe()

if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
    df = df.sample(n=SAMPLE_SIZE, seed=42)

rng = np.random.default_rng(42)
df = df.with_columns(
    pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64)
)
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
    use_cv=USE_CV,
    base_rate=BASE_RATE,
)

print(eglm.summary())
# → {'model_type': 'Poisson', 'target': 'ClaimNb', 'weight_col': 'Exposure', ...}
print(f"  Intercept: {eglm.model.intercept_:.4f}")
print(f"  Non-constant variables: {list(eglm.rate_model.non_constant_variables.keys())}")

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
# 5. Per-variable A/E drill-down
# ---------------------------------------------------------------------------

print("\nPer-variable A/E on holdout:")
for var in PREDICTORS:
    result = eglm.rate_model.compute_ae_for_variable(holdout, var)
    subsets = result["subsets"]
    for split_name in ("train", "test"):
        if split_name not in subsets:
            continue
        buckets = subsets[split_name]
        actual = sum(b["actual"] for b in buckets)
        expected = sum(b["expected"] for b in buckets)
        ae = actual / expected if expected > 0 else float("nan")
        print(f"  {var:15s}  {split_name:5s}  A/E = {ae:.4f}  ({len(buckets)} bins)")

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
print(f"Reloaded — {len(loaded.variables)} variables, "
      f"base_rate={loaded.base_rate}, "
      f"{len(loaded.snapshots)} snapshot(s)")

new_business = df.select(PREDICTORS).head(3)
print(f"Reloaded predictions: {loaded.predict(new_business).round(6)}")

# ---------------------------------------------------------------------------
# 8. Launch the relativity editor (browser)
# ---------------------------------------------------------------------------

if LAUNCH_EDITOR:
    print(f"\nLaunching editor at http://localhost:{EDITOR_PORT} ...")
    print("(Close the browser tab and Ctrl-C this script when done.)")
    eglm.launch_editor(data=df, port=EDITOR_PORT)
