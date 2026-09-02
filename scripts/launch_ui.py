"""Launch the easy_glm UI. First run 'python test_ui_demo.py' to generate model.easyglm."""

import numpy as np
import polars as pl

from easy_glm.engine import RateModel
from easy_glm.ui import launch_editor

# Load the model
rm = RateModel.from_json("model.easyglm")

# Create some data to pass to the UI
rng = np.random.default_rng(42)
n = 500
df = pl.DataFrame(
    {
        "VehAge": rng.integers(0, 16, size=n),
        "DrivAge": rng.integers(18, 80, size=n),
        "Region": rng.choice(["North", "South", "Urban"], size=n),
        "Exposure": rng.uniform(0.25, 1.0, size=n),
    }
)
region_effect = np.select(
    [df["Region"] == "South", df["Region"] == "Urban"],
    [0.25, 0.65],
    default=0.0,
)
log_freq = (
    -2.4
    + (df["VehAge"].to_numpy() * 0.055)
    + ((80 - df["DrivAge"].to_numpy()) * 0.01)
    + region_effect
)
df = df.with_columns(
    ClaimNb=pl.Series(rng.poisson(df["Exposure"].to_numpy() * np.exp(log_freq))),
    traintest=pl.Series((rng.random(n) < 0.75).astype(np.int8)),
)

launch_editor(rm, data=df, port=8501)
