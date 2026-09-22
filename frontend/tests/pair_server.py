"""Small real desktop server for sequential pair browser acceptance tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import uvicorn

from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import Project


def main() -> None:
    rng = np.random.default_rng(103)
    size = 220
    x = rng.normal(size=size)
    a = rng.choice(["low", "mid", "high"], size=size)
    b = rng.choice(["red", "blue", "green"], size=size)
    c = rng.choice(["urban", "rural", "coastal"], size=size)
    d = rng.normal(size=size)
    mean = np.exp(
        0.1
        + 0.2 * x
        + 0.55 * ((a == "high") & (b == "red"))
        - 0.4 * ((b == "green") & (c == "rural"))
    )
    raw = pl.DataFrame(
        {
            "x": x,
            "A": a,
            "B": b,
            "C": c,
            "D": d,
            "y": rng.poisson(mean).astype(float),
            "split": [1] * 180 + [0] * 40,
        }
    )
    project = Project(name="Pair acceptance sample")
    project.data.roles = {
        "x": "predictor",
        "A": "predictor",
        "B": "predictor",
        "C": "predictor",
        "y": "target",
        "split": "split",
    }
    project.data.split.column = "split"
    uvicorn.run(
        create_app(project, raw, port=8831),
        host="127.0.0.1",
        port=8831,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
