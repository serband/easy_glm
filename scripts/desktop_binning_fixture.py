"""Numeric boundary fixture for browser acceptance and local review."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import uvicorn

from easy_glm.desktop.server import create_app
from easy_glm.workflow import Project, Split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8822)
    args = parser.parse_args()
    rng = np.random.default_rng(73)
    size = 2400
    age = rng.integers(18, 85, size)
    exposure = rng.uniform(0.2, 1, size)
    raw = pl.DataFrame(
        {
            "Claims": rng.poisson(exposure * (0.2 + (age < 25) * 0.15)),
            "Exposure": exposure,
            "DriverAge": age,
            "VehicleAge": ([-1, 0, 1, 2, 3, 4, 5, 6, None] * 267)[:size],
            "Mileage": rng.uniform(100, 25000, size),
            "Region": rng.choice(["North", "South", "East", "West"], size),
            "train_test": np.where(np.arange(size) % 4 == 0, "test", "TRAIN"),
        }
    )
    project = Project(name="Numeric binning review")
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "DriverAge": "predictor",
        "VehicleAge": "predictor",
        "Mileage": "predictor",
        "Region": "predictor",
        "train_test": "split",
    }
    project.data.split = Split(
        column="train_test", train_value="TRAIN", holdout_value="test"
    )
    model = project.new_model("Frequency", family="poisson")
    model.divide_target_by_weight = True
    model.penalty.alpha = 0.01
    model.penalty.cv = None
    with TemporaryDirectory(prefix="easyglm-binning-") as directory:
        path = Path(directory) / "portfolio.parquet"
        raw.write_parquet(path)
        project.data.source.path = str(path)
        uvicorn.run(
            create_app(project, raw, port=args.port),
            host="127.0.0.1",
            port=args.port,
            log_level="warning",
        )


if __name__ == "__main__":
    main()
