"""File-backed source for the browser export workflow."""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import uvicorn

from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import Project

with tempfile.TemporaryDirectory(prefix="easyglm_export_browser_") as folder:
    rng = np.random.default_rng(42)
    raw = pl.DataFrame(
        {
            "Claims": rng.poisson(0.15, 4000),
            "Exposure": rng.uniform(0.1, 1, 4000),
            "DriverAge": rng.integers(18, 90, 4000),
            "Region": rng.choice(["North", "South", "East", "West"], 4000),
        }
    )
    path = Path(folder) / "source.parquet"
    raw.write_parquet(path)
    project = Project(name="Export test · portfolio")
    project.data.source.type = "parquet"
    project.data.source.path = str(path)
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "DriverAge": "predictor",
        "Region": "predictor",
    }
    uvicorn.run(
        create_app(project, raw, port=8770),
        host="127.0.0.1",
        port=8770,
        log_level="warning",
    )
