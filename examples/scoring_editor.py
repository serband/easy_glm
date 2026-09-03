""".easyglm roundtrip: save, reload, score, and launch the relativity editor.

This example focuses on what happens *after* fitting — the portable
``.easyglm`` file is the artifact you ship for production scoring or
hand off to an actuary to review in the editor.

Run as a script:
    python examples/scoring_editor.py
"""

from pathlib import Path

import numpy as np
import polars as pl

import easy_glm
from easy_glm.engine import RateModel

# ---------------------------------------------------------------------------
# 0. Fit a quick model (skip if you already have a .easyglm file)
# ---------------------------------------------------------------------------

DATA = Path(__file__).resolve().parents[1] / "tests/fixtures/french_motor_50k.parquet"
df = pl.read_parquet(DATA)
rng = np.random.default_rng(42)
df = df.with_columns(pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64))

PREDICTORS = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

eglm = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=PREDICTORS,
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    cv=5,
    base_rate=0.05,
)

# ---------------------------------------------------------------------------
# 1. Export as .easyglm (portable JSON)
# ---------------------------------------------------------------------------

eglm.rate_model.to_json("portfolio_v1.easyglm")
print("Exported → portfolio_v1.easyglm")

# ---------------------------------------------------------------------------
# 2. Reload on a different machine / process (no refit needed)
# ---------------------------------------------------------------------------

rm = RateModel.from_json("portfolio_v1.easyglm")
print(f"Loaded: {len(rm.variables)} variables, base_rate={rm.base_rate}")
print(f"Snapshots: {len(rm.snapshots)}")
for s in rm.list_snapshots():
    print(f"  v{s['version']}: {s['description']} ({s['timestamp'][:19]})")

# ---------------------------------------------------------------------------
# 3. Score new business (pure lookup — no glum, no DuckDB)
# ---------------------------------------------------------------------------

new_business = pl.DataFrame(
    {
        "VehAge": [3, 8, 0],
        "Region": ["Ile-de-France", "Bretagne", "Nord-Pas-de-Calais"],
        "VehGas": ["Regular", "Diesel", "Regular"],
        "DrivAge": [35, 52, 22],
        "BonusMalus": [50, 68, 90],
        "Density": [2000, 500, 8000],
    }
)

premiums = rm.predict(new_business)
for i, p in enumerate(premiums):
    print(f"  Risk {i + 1}: {p:.6f}")
# → Risk 1: 0.053832
#   Risk 2: 0.188926
#   ...

# ---------------------------------------------------------------------------
# 4. Column mapping — when dataset column names differ from model variables
# ---------------------------------------------------------------------------

mismatched_data = pl.DataFrame(
    {
        "vehicle_age": [3, 8],
        "region_code": ["Ile-de-France", "Bretagne"],
        "fuel": ["Regular", "Diesel"],
        "driver_age": [35, 52],
        "bonus_malus": [50, 68],
        "pop_density": [2000, 500],
    }
)

# keys are the *dataset* column names, values the model variables they map to
rm.column_mapping = {
    "vehicle_age": "VehAge",
    "region_code": "Region",
    "fuel": "VehGas",
    "driver_age": "DrivAge",
    "bonus_malus": "BonusMalus",
    "pop_density": "Density",
}

mapped_preds = rm.predict(mismatched_data)
print(f"\nWith column mapping: {mapped_preds.round(6)}")

# ---------------------------------------------------------------------------
# 5. Create a named snapshot & save a revision
# ---------------------------------------------------------------------------

rm.create_snapshot("Initial import (auto-mapped)")
rm.to_json("portfolio_v1.easyglm")  # overwrite with snapshots included
print(f"\nSnapshots after save: {len(rm.snapshots)}")

# ---------------------------------------------------------------------------
# 6. Launch the relativity editor (browser)
# ---------------------------------------------------------------------------

# rm.launch_editor(data=df)
# print("Editor launched at http://localhost:8501")
