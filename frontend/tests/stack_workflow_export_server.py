"""Small persisted-data server for stacked Python export browser checks."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import uvicorn

from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import Project

port = int(os.environ.get("EASYGLM_STACK_WORKFLOW_EXPORT_PORT", "8790"))

with tempfile.TemporaryDirectory(prefix="easyglm_stack_workflow_export_") as folder:
    rng = np.random.default_rng(20261011)
    rows = 360
    driver_age = rng.choice([25, 55], size=rows)
    annual_mileage = rng.choice(["low", "high"], size=rows)
    vehicle_age = rng.choice([1, 12], size=rows)
    region = rng.choice(["North", "South"], size=rows)
    rate = np.where(driver_age == 55, 1.25, 0.82)
    rate *= np.where((annual_mileage == "high") & (vehicle_age == 12), 1.8, 0.8)
    rate *= np.where(region == "South", 1.1, 0.95)
    raw = pl.DataFrame(
        {
            "Claims": rng.poisson(rate * 0.28),
            "Exposure": rng.uniform(0.5, 1.5, size=rows),
            "DriverAge": driver_age,
            "AnnualMileage": annual_mileage,
            "VehicleAge": vehicle_age,
            "Region": region,
        }
    )
    source = Path(folder) / "stack-workflow-export.parquet"
    raw.write_parquet(source)
    project = Project(name="Stack workflow export browser fixture")
    project.data.source.type = "parquet"
    project.data.source.path = str(source)
    project.data.split.mode = "random"
    project.data.roles = {
        "Claims": "target",
        "Exposure": "weight",
        "DriverAge": "predictor",
        "AnnualMileage": "predictor",
        "VehicleAge": "predictor",
        "Region": "predictor",
    }
    uvicorn.run(
        create_app(project, raw, port=port),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
