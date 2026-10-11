"""Deterministic interaction-signal fixture for browser lifecycle tests."""

import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import uvicorn

from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import Project

port = int(os.environ.get("EASYGLM_CORE_INTERACTIONS_PORT", "8790"))

with tempfile.TemporaryDirectory(prefix="easyglm_core_interactions_browser_") as folder:
    rng = np.random.default_rng(20261010)
    rows = 2400
    driver_age = rng.choice([25, 55], size=rows)
    region = rng.choice(["North", "South"], size=rows)
    mileage = rng.choice(["low", "high"], size=rows)
    vehicle_age = rng.choice([1, 12], size=rows)
    # The main-effects GLM uses DriverAge and Region. The intentionally omitted
    # mileage x vehicle-age multiplier leaves a supported residual pair for search.
    rate = np.where(driver_age == 55, 1.30, 0.82) * np.where(
        region == "South", 1.18, 0.92
    )
    rate *= np.where((mileage == "high") & (vehicle_age == 12), 2.6, 0.72)
    raw = pl.DataFrame(
        {
            "PolicyID": np.arange(rows),
            "Claims": rng.poisson(rate * 0.35),
            "Exposure": np.ones(rows),
            "DriverAge": driver_age,
            "VehicleAge": vehicle_age,
            "Region": region,
            "AnnualMileage": mileage,
            "InternalCode": rng.integers(0, 20, rows),
        }
    )
    path = Path(folder) / "core-interactions.parquet"
    raw.write_parquet(path)
    project = Project(name="Core interactions browser fixture")
    project.data.source.type = "parquet"
    project.data.source.path = str(path)
    project.data.split.mode = "random"
    project.data.roles = {
        "PolicyID": "id",
        "Claims": "target",
        "Exposure": "weight",
        "DriverAge": "predictor",
        "VehicleAge": "predictor",
        "Region": "predictor",
        "AnnualMileage": "predictor",
        "InternalCode": "ignore",
    }
    uvicorn.run(
        create_app(project, raw, port=port),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
