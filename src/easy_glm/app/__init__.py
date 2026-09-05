"""easy_glm Workbench — a browser GUI for the whole modelling workflow.

Launch::

    python -m easy_glm.app                 # empty project
    python -m easy_glm.app my.easyglm-project.json

or from Python::

    from easy_glm.app import launch
    launch("my.easyglm-project.json")

The GUI edits a :class:`easy_glm.workflow.Project`; everything it does can be
exported as a Python script from the Export page. It is included in the normal
``pip install easy_glm`` installation.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _temporary_project_for(data: Any) -> Path:
    """Persist an in-memory Polars or pandas frame for the workbench process."""
    import polars as pl

    if isinstance(data, pl.DataFrame):
        frame = data
    else:
        try:
            frame = pl.from_pandas(data)
        except Exception as exc:  # noqa: BLE001 - replace conversion internals
            raise TypeError(
                "data must be a Polars DataFrame or pandas DataFrame"
            ) from exc

    from easy_glm.workflow import Project

    folder = Path(tempfile.mkdtemp(prefix="easy_glm_workbench_"))
    data_path = folder / "data.parquet"
    project_path = folder / "in_memory.easyglm-project.json"
    frame.write_parquet(data_path)
    project = Project(name="in-memory data")
    project.data.source.type = "parquet"
    project.data.source.path = str(data_path)
    project.data.split.mode = "random"
    split_name = "traintest"
    suffix = 2
    while split_name in frame.columns:
        split_name = f"traintest_{suffix}"
        suffix += 1
    project.data.split.column = split_name
    project.to_json(project_path)
    return project_path


def launch(
    project_path: str | Path | None = None,
    *,
    data: Any | None = None,
    port: int = 8501,
    block: bool = False,
    headless: bool = False,
) -> subprocess.Popen:
    """Start the workbench in a separate Streamlit process.

    Pass either a saved workbench ``project_path`` or an in-memory Polars or
    pandas ``data`` frame. With a frame, the workbench opens with the data
    loaded and the user assigns its modelling roles in the browser.
    """
    if project_path is not None and data is not None:
        raise ValueError("pass project_path or data, not both")
    if data is not None:
        project_path = _temporary_project_for(data)

    main = Path(__file__).with_name("main.py")
    args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(main),
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
        "--server.showEmailPrompt",
        "false",
        "--server.maxUploadSize",
        "2048",
    ]
    if headless:
        args += ["--server.headless", "true"]
    args.append("--")
    if project_path is not None:
        args.append(f"--project={project_path}")
    proc = subprocess.Popen(args)
    if block:
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
    return proc


__all__ = ["launch"]
