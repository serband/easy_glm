"""Run the local experiment: python -m easy_glm.desktop [--project PATH]."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    import numpy as np
    import polars as pl
    import uvicorn

    from easy_glm.workflow.prep import load_source
    from easy_glm.workflow.project import Project

    from .server import create_app

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--launch-id", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.project:
        project = Project.from_json(args.project)
        path = Path(project.data.source.path)
        if not path.is_absolute():
            project.data.source.path = str(
                (args.project.resolve().parent / path).resolve()
            )
        raw = load_source(project.data.source)
    else:
        rng = np.random.default_rng(42)
        raw = pl.DataFrame(
            {
                "PolicyID": np.arange(12000),
                "Claims": rng.poisson(0.12, 12000),
                "Exposure": rng.uniform(0.1, 1, 12000),
                "DriverAge": rng.integers(18, 90, 12000),
                "VehicleAge": rng.integers(0, 25, 12000),
                "Region": rng.choice(["North", "South", "East", "West"], 12000),
                "AnnualMileage": rng.gamma(4, 3000, 12000),
                "InternalCode": rng.integers(0, 50, 12000),
            }
        )
        project = Project(name="Motor portfolio · synthetic sample")
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
    print(f"EasyGLM experiment: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(
        create_app(project, raw, port=args.port, launch_id=args.launch_id),
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
