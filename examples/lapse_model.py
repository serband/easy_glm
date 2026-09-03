"""A lapse (binomial) model: odds relativities and a probability scorer.

``log`` and ``logit`` are both multiplicative links, so a binomial fit compiles
to the same kind of rate table as a frequency model — read here as **odds
relativities** — and the scorer converts back to a probability. Because a
probability is not an amount, such a model refuses to be multiplied by an
exposure column.

Run as a script:
    python examples/lapse_model.py
"""

from pathlib import Path

import numpy as np
import polars as pl

from easy_glm import DesignSpec, fit_glm, rate_tables, to_rate_model

DATA = Path(__file__).resolve().parents[1] / "tests/fixtures/french_motor_50k.parquet"

# ---------------------------------------------------------------------------
# 1. A stand-in lapse flag (in practice this comes from your policy admin
#    system): here, older and higher-bonus-malus drivers lapse more often.
# ---------------------------------------------------------------------------

df = pl.read_parquet(DATA)
rng = np.random.default_rng(7)
logit_p = 1.5 - 0.02 * df["DrivAge"].to_numpy() + 0.01 * df["BonusMalus"].to_numpy()
p = 1.0 / (1.0 + np.exp(-logit_p))
df = df.with_columns(pl.Series("Lapsed", (rng.random(len(df)) < p).astype(float)))
df = df.with_columns(pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64))
train_df = df.filter(pl.col("traintest") == 1)
holdout = df.filter(pl.col("traintest") == 0)

# ---------------------------------------------------------------------------
# 2. Fit — family="binomial" is the only thing that changes vs. a frequency
#    model; no weight column (a lapse is a 0/1 outcome per policy, not a count
#    over exposure).
# ---------------------------------------------------------------------------

spec = DesignSpec.from_data(train_df, ["DrivAge", "BonusMalus"])
fit = fit_glm(train_df, spec, "Lapsed", family="binomial", alpha=0.0005)
print(fit)

tables = rate_tables(fit)
print(tables["DrivAge"].head(4))

rm = to_rate_model(fit)
print("relativity label:", rm.relativity_label)  # "odds relativity"

prob = rm.predict(holdout)  # a probability in (0, 1), no exposure multiplier
print("predicted lapse probability, first 5 rows:", prob[:5].round(4))
print(
    f"holdout observed lapse rate: {holdout['Lapsed'].mean():.4f}, "
    f"predicted mean: {prob.mean():.4f}"
)

try:
    rm.predict(holdout, exposure_col="Exposure")
except ValueError as exc:
    print("refused, as expected:", exc)
