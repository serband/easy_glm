"""easy_glm Workbench — a browser GUI for the whole modelling workflow.

Launch::

    python -m easy_glm.app                 # empty project
    python -m easy_glm.app my.easyglm-project.json

or from Python::

    from easy_glm.app import launch
    launch("my.easyglm-project.json")

The GUI edits a :class:`easy_glm.workflow.Project`; everything it does can be
exported as a Python script (Export page). Requires ``pip install "easy_glm[ui]"``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch(
    project_path: str | Path | None = None,
    *,
    port: int = 8501,
    block: bool = False,
    headless: bool = False,
) -> subprocess.Popen:
    """Start the workbench in a separate Streamlit process."""
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
        proc.wait()
    return proc


__all__ = ["launch"]
