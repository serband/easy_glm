"""File-backed screening fixture, isolated from every user session."""

import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import uvicorn

from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import Project

port = int(os.environ.get("EASYGLM_SCREENING_PORT", "8787"))

with tempfile.TemporaryDirectory(prefix="easyglm_screening_browser_") as folder:
    rng = np.random.default_rng(42)
    age = rng.integers(18, 90, 2000)
    claims = rng.poisson(0.5, 2000)
    raw = pl.DataFrame(
        {
            "Claims": claims,
            "Exposure": rng.uniform(0.1, 1, 2000),
            "DriverAge": age,
            "VehicleAge": age * 2,
            "Region": [
                None if i % 5 else "North" if i % 2 else "South" for i in range(2000)
            ],
            "AnnualMileage": claims * 1000,
        }
    )
    path = Path(folder) / "source.parquet"
    raw.write_parquet(path)
    project = Project(name="Screening test portfolio")
    project.data.source.type = "parquet"
    project.data.source.path = str(path)
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "DriverAge": "predictor",
        "VehicleAge": "predictor",
        "Region": "predictor",
        "AnnualMileage": "predictor",
    }
    project.data.split.mode = "random"
    uvicorn.run(
        create_app(project, raw, port=port),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
