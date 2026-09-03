"""Fit the change from the premium you charge today (0.4 pieces E1/E4).

Shows the standard rate-review setup: give the column holding today's premium
the role ``current_premium`` in a :class:`~easy_glm.workflow.Project`, and
``log(<premium>)`` is derived and pre-filled as every new model's offset. The
resulting rate tables are *multipliers on the current premium* — the base
rate is the overall change, each relativity a differential change — and
:func:`~easy_glm.workflow.solve_base_rate` sets the base rate to hit a target
loss ratio without moving a single relativity.

Run as a script:
    python examples/rate_change.py
"""

from pathlib import Path

import polars as pl

from easy_glm.workflow import (
    Project,
    VariableDesign,
    prepare,
    run_model,
    solve_base_rate,
)

DATA = Path(__file__).resolve().parents[1] / "tests/fixtures/french_motor_50k.parquet"

# ---------------------------------------------------------------------------
# 1. A stand-in "current premium" column (in practice this is already in
#    your book) — here, a plausible-looking premium unrelated to the true cost.
# ---------------------------------------------------------------------------

df = pl.read_parquet(DATA).with_columns(
    (pl.col("Exposure") * 180.0).alias("CurrentPremium")
)
premium_path = Path("french_motor_with_premium.parquet")
df.write_parquet(premium_path)

# ---------------------------------------------------------------------------
# 2. Build the project: CurrentPremium gets role "current_premium"
# ---------------------------------------------------------------------------

project = Project(name="rate_change_demo")
project.data.source.path = str(premium_path)
project.data.roles = {
    "ClaimNb": "target",
    "Exposure": "weight",
    "CurrentPremium": "current_premium",
    "DrivAge": "predictor",
    "Region": "predictor",
    "BonusMalus": "predictor",
    "Density": "predictor",
}
project.data.split.mode = "random"
project.data.split.fraction = 0.7
project.data.split.seed = 42
project.design.variables["Density"] = VariableDesign(kind="linear")

project.new_model(
    "change",
    family="poisson",
    divide_target_by_weight=True,
    predictors=["DrivAge", "Region", "BonusMalus", "Density"],
)
project.models["change"].penalty.alpha = 0.001
project.models["change"].penalty.cv = None

print("offset column (derived automatically):", project.offset_column)

# ---------------------------------------------------------------------------
# 3. Fit and read the labels
# ---------------------------------------------------------------------------

df_prepared = prepare(project)
run = run_model(project, df_prepared, "change")
rm = run.rate_model
print("table label:", rm.relativity_label)  # "multiplier on current premium"
print(f"base rate (the overall change from today's premium): {rm.base_rate:.6f}")

# ---------------------------------------------------------------------------
# 4. Solve for a target loss ratio — relativities never move
# ---------------------------------------------------------------------------

train = df_prepared.filter(pl.col(project.data.split.column) == 1)
for target_ratio in (0.60, 0.65, 0.70):
    new_base = solve_base_rate(run, train, target_ratio=target_ratio)
    print(f"  target loss ratio {target_ratio:.0%}  ->  base rate {new_base:.6f}")
