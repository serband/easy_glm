"""Text-label split fixture for browser acceptance and local review."""

import argparse

import numpy as np
import polars as pl
import uvicorn

from easy_glm.desktop.server import create_app
from easy_glm.workflow import Project, Split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8811)
    port = parser.parse_args().port
    rng = np.random.default_rng(42)
    n = 12000
    age = rng.integers(18, 85, n)
    exposure = rng.uniform(0.2, 1, n)
    raw = pl.DataFrame(
        {
            "Claims": rng.poisson(exposure * (0.08 + (age < 25) * 0.08)),
            "Exposure": exposure,
            "DriverAge": age,
            "Region": rng.choice(["North", "South", "East", "West"], n),
            "train_test": np.where(np.arange(n) % 4 == 0, "test", "TRAIN"),
        }
    )
    project = Project(name="Split mapping review")
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "DriverAge": "predictor",
        "Region": "predictor",
        "train_test": "split",
    }
    project.data.split = Split(column="train_test")
    project.new_model("Frequency", family="poisson")
    uvicorn.run(
        create_app(project, raw, port=port),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
