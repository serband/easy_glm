"""Small real workbench server used by the wide model-variables browser test."""

from __future__ import annotations

import numpy as np
import polars as pl
import uvicorn

from easy_glm.desktop.server import create_app
from easy_glm.workflow import Project


def main() -> None:
    rows = 360
    rng = np.random.default_rng(4)
    raw = pl.DataFrame(
        {
            "PolicyID": np.arange(rows),
            "Claims": rng.poisson(0.12, rows),
            "Exposure": rng.uniform(0.2, 1.0, rows),
            "DriverAge": rng.integers(18, 90, rows),
            **{
                f"OptionalFactor{number:03d}": rng.integers(0, 9, rows)
                for number in range(500)
            },
        }
    )
    project = Project(name="Wide variable test")
    project.data.split.mode = "random"
    project.data.roles = {
        "PolicyID": "id",
        "Claims": "target",
        "Exposure": "weight",
        "DriverAge": "predictor",
    }
    uvicorn.run(
        create_app(project, raw, port=8771),
        host="127.0.0.1",
        port=8771,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
