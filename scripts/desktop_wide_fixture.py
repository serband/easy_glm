"""Disposable 2,000-column server for the browser scale test (no disk writes)."""

import polars as pl
import uvicorn

from easy_glm.desktop.server import create_app
from easy_glm.workflow.project import Project

if __name__ == "__main__":
    raw = pl.DataFrame({f"variable_{i:04d}": list(range(2000)) for i in range(2000)})
    uvicorn.run(
        create_app(Project(name="Wide synthetic benchmark"), raw, port=8771),
        host="127.0.0.1",
        port=8771,
        log_level="warning",
    )
