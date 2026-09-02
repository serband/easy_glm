"""Explore a fitted model: relativity plots, A/E charts, train-vs-test.

Assumes you already have a fitted ``EasyGLM`` object.  If you don't, run
``basic_usage.py`` first, or fit one inline (see the commented block at the
top of this file).

Run as a script:
    python examples/exploring_fit.py
"""

import numpy as np
import polars as pl

import easy_glm

# ---------------------------------------------------------------------------
# Either load a previously saved model or fit one now
# ---------------------------------------------------------------------------

# --- Option A: reload from disk (if you ran basic_usage.py first) ---
# eglm = easy_glm.EasyGLM.load("my_model")
# df = easy_glm.load_external_dataframe()
# rng = np.random.default_rng(42)
# df = df.with_columns(
#     pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64)
# )

# --- Option B: fit inline (standalone run) ---
df = easy_glm.load_external_dataframe()
rng = np.random.default_rng(42)
df = df.with_columns(
    pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64)
)

PREDICTORS = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

eglm = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=PREDICTORS,
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    use_cv=True,
    base_rate=0.05,
)

# ---------------------------------------------------------------------------
# 1. Relativity tables — one per predictor
# ---------------------------------------------------------------------------

print("=== Relativities ===\n")
for name, table in eglm.relativities.items():
    print(f"{name}  ({len(table)} bins)")
    print(table.head(3), "\n")

# ---------------------------------------------------------------------------
# 2. Non-constant variables (signal-bearing)
# ---------------------------------------------------------------------------

non_const = eglm.rate_model.non_constant_variables
print("=== Variables with signal ===")
for name in sorted(non_const):
    rels = [r.relativity for r in eglm.rate_model.variables[name].table]
    print(f"  {name:15s}  range: [{min(rels):.3f}, {max(rels):.3f}]")
# → Variables where all bins have the same relativity are excluded.

# ---------------------------------------------------------------------------
# 3. Matplotlib relativity charts (numeric → line, categorical → bar)
# ---------------------------------------------------------------------------

# easy_glm.plot_all_ratetables(eglm.relativities, eglm.blueprint)

# ---------------------------------------------------------------------------
# 4. A/E on holdout — per variable, per bin
# ---------------------------------------------------------------------------

holdout = df.filter(pl.col("traintest") == 0)

print("\n=== A/E on holdout (per variable) ===\n")
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

        # Show first and last bucket for each variable
        first = buckets[0]
        last = buckets[-1]
        print(
            f"  {var:15s}  {split_name:5s}  "
            f"A/E = {ae:.4f}  "
            f"first bin: A={first['actual']:.3f} / E={first['expected']:.3f}  "
            f"last bin:  A={last['actual']:.3f} / E={last['expected']:.3f}"
        )
    print()

# ---------------------------------------------------------------------------
# 5. Tweak a relativity, recompute A/E, compare
# ---------------------------------------------------------------------------

print("=== Tweak & compare ===\n")

# Save current state
rm = eglm.rate_model
original_preds = rm.predict(holdout)
original_ae = holdout["ClaimNb"].sum() / original_preds.sum()
print(f"Before tweak — overall A/E: {original_ae:.4f}")

# Nudge one relativity
var = "DrivAge"
config = rm.variables[var]
row = config.table[len(config.table) // 2]  # middle bin
print(f"Tweaking {var} bin [{row.from_}, {row.to_}): {row.relativity:.4f} → {row.relativity * 1.2:.4f}")
rm.update_relativity(var, row.from_, row.to_, row.relativity * 1.2)

# Re-score
tweaked_preds = rm.predict(holdout)
tweaked_ae = holdout["ClaimNb"].sum() / tweaked_preds.sum()
print(f"After  tweak — overall A/E: {tweaked_ae:.4f}")

# Drill into the tweaked variable
tweaked_ae_result = rm.compute_ae_for_variable(holdout, var)
for split_name, buckets in tweaked_ae_result["subsets"].items():
    actual = sum(b["actual"] for b in buckets)
    expected = sum(b["expected"] for b in buckets)
    ae = actual / expected if expected > 0 else float("nan")
    print(f"  {var:15s}  {split_name:5s}  A/E = {ae:.4f}")

# ---------------------------------------------------------------------------
# 6. Reset to original (discard the tweak)
# ---------------------------------------------------------------------------

rm.create_snapshot("tweaked")
rm.switch_to(1)  # back to version 1 (original fit)
print(f"\nReset to version 1 — A/E: "
      f"{holdout['ClaimNb'].sum() / rm.predict(holdout).sum():.4f}")
