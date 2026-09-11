"""Synthetic calendar-time portfolio for testing and reviewing time diagnostics."""

import argparse

import numpy as np
import polars as pl
import uvicorn

from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import Project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8813)
    parser.add_argument("--mock-examples", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(519)
    n = 12000
    year = rng.integers(2020, 2025, n)
    age = rng.integers(18, 80, n)
    weight = rng.uniform(0.3, 1, n)
    rate = (0.09 + 0.08 * (age < 30)) * (1 + 0.08 * (year - 2020))
    rate *= 1 + 0.12 * (year - 2020) * (age < 30)
    raw = pl.DataFrame(
        {
            "Year": year,
            "DriverAge": age,
            "Region": rng.choice(["North", "South", "East"], n),
            "Exposure": weight,
            "Claims": rng.poisson(weight * rate),
        }
    )
    if args.mock_examples:
        import easy_glm

        sample = pl.DataFrame(
            {
                "ClaimNb": [0, 1, 0, 2, 1, 0],
                "Exposure": [1.0] * 6,
                "ClaimAmount": [0.0, 10.0, 0.0, 20.0, 10.0, 0.0],
                "IDpol": list(range(6)),
                **{
                    c: list(range(6))
                    for c in [
                        "DrivAge",
                        "Region",
                        "BonusMalus",
                        "Density",
                        "OwnerAge",
                        "Gender",
                        "Area",
                        "RiskClass",
                        "VehAge",
                        "BonusClass",
                    ]
                },
            }
        )
        easy_glm.load_external_dataframe = lambda: sample
        easy_glm.load_swedish_motorcycle_data = lambda: sample
    project = Project(name="Time stability · synthetic example")
    project.data.roles = {
        "Year": "time",
        "DriverAge": "predictor",
        "Region": "predictor",
        "Exposure": "weight",
        "Claims": "target",
    }
    project.data.split.mode = "random"
    config = project.new_model("Frequency", family="poisson")
    config.penalty.alpha = 0.001
    config.penalty.cv = None
    uvicorn.run(
        create_app(project, raw, port=args.port),
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
