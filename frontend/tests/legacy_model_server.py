"""Dedicated, fresh legacy-project server for saved-model browser regressions."""

from __future__ import annotations

import numpy as np
import polars as pl
import uvicorn

from easy_glm.desktop.server import create_app
from easy_glm.workflow import Project


def main() -> None:
    rows = 480
    rng = np.random.default_rng(31)
    raw = pl.DataFrame(
        {
            "PolicyID": np.arange(rows),
            "Claims": rng.poisson(0.16, rows),
            "Exposure": rng.uniform(0.2, 1.0, rows),
            "DriverAge": rng.integers(18, 90, rows),
            "VehicleAge": rng.integers(0, 20, rows),
            "Region": rng.choice(["A", "B", "C"], rows),
        }
    )
    project = Project(name="Legacy saved-model regression")
    project.data.split.mode = "random"
    project.data.roles = {
        "PolicyID": "id",
        "Claims": "target",
        "Exposure": "weight",
        "DriverAge": "predictor",
        "VehicleAge": "predictor",
        "Region": "predictor",
    }
    uvicorn.run(
        create_app(project, raw, port=8787),
        host="127.0.0.1",
        port=8787,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
