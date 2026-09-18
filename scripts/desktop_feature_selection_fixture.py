"""Small planted-signal portfolio for desktop feature-selection browser checks."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import uvicorn

from easy_glm.desktop.server import create_app
from easy_glm.workflow import Project, Split
from easy_glm.workflow.project import VariableDesign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8825)
    args = parser.parse_args()

    rng = np.random.default_rng(174)
    size = 240
    risk = rng.integers(0, 2, size)
    exposure = rng.uniform(0.7, 1.3, size)
    claims = rng.poisson(exposure * np.where(risk == 1, 2.5, 0.08))
    split = np.where(np.arange(size) % 4 == 0, "test", "TRAIN")
    claims[split == "test"] = 100  # holdout outcomes must not affect the screen
    raw = pl.DataFrame(
        {
            "Claims": claims,
            "Exposure": exposure,
            "Risk": risk,
            "Noise": rng.normal(size=size),
            "Prospect": rng.normal(size=size),
            "Flat": np.ones(size),
            "train_test": split,
        }
    )
    project = Project(name="One-way feature selection review")
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "Risk": "predictor",
        "Noise": "predictor",
        "train_test": "split",
    }
    project.data.split = Split(
        column="train_test", train_value="TRAIN", holdout_value="test"
    )
    project.design.defaults.n_bins = 6
    project.design.variables["Risk"] = VariableDesign(kind="step", knots=[0.5])

    with TemporaryDirectory(prefix="easyglm-feature-selection-") as directory:
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
