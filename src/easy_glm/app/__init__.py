"""easy_glm Workbench — a browser GUI for the whole modelling workflow.

Launch::

    python -m easy_glm.app                 # empty project
    python -m easy_glm.app my.easyglm-project.json

or from Python::

    from easy_glm.app import launch
    launch("my.easyglm-project.json")

The GUI edits a :class:`easy_glm.workflow.Project`; everything it does can be
exported as a Python script from the Export page. It is included in the normal
``pip install easy-glm`` installation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _as_frame(data: Any):
    """Convert the two supported in-memory launch inputs without mutating them."""
    import polars as pl

    if isinstance(data, pl.DataFrame):
        return data
    try:
        return pl.from_pandas(data)
    except Exception as exc:  # noqa: BLE001 - replace conversion internals
        raise TypeError("data must be a Polars DataFrame or pandas DataFrame") from exc


def _temporary_project_and_scoring(
    data: Any, scoring_data: dict[str, Any] | None = None
) -> tuple[Path, Path | None]:
    """Persist an in-memory Polars or pandas frame for the workbench process."""
    frame = _as_frame(data)

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
    if not scoring_data:
        return project_path, None
    if not isinstance(scoring_data, dict) or not all(
        isinstance(name, str) and name.strip() for name in scoring_data
    ):
        raise TypeError(
            "scoring_data must map non-empty dataset names to pandas or Polars frames"
        )
    sources: dict[str, str] = {}
    for index, (name, scoring_frame) in enumerate(scoring_data.items(), start=1):
        path = folder / f"scoring_{index}.parquet"
        _as_frame(scoring_frame).write_parquet(path)
        sources[name] = str(path)
    manifest = folder / "scoring_data.json"
    manifest.write_text(json.dumps(sources), encoding="utf-8")
    return project_path, manifest


def _temporary_project_for(data: Any) -> Path:
    """Persist an in-memory frame, retaining the historical Path return value."""
    project_path, _manifest = _temporary_project_and_scoring(data)
    return project_path


def _url_for(host: str | None, port: int) -> str:
    shown_host = host or "localhost"
    if shown_host == "0.0.0.0":
        shown_host = "localhost"
    return f"http://{shown_host}:{port}"


def _streamlit_args(
    project_path: str | Path | None,
    *,
    port: int,
    headless: bool,
    host: str | None,
) -> list[str]:
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
        # Keep imported classes/modules stable for the lifetime of a workbench.
        # Hot reload evicts modules while another session may be importing them.
        "--server.fileWatcherType",
        "none",
    ]
    if host:
        args += ["--server.address", host]
    if headless:
        args += ["--server.headless", "true"]
    args.append("--")
    if project_path is not None:
        args.append(f"--project={project_path}")
    return args


def _launcher_args(
    project_path: str | Path | None,
    *,
    port: int,
    headless: bool,
    host: str | None,
) -> list[str]:
    args = [sys.executable, "-m", "easy_glm.app", "--legacy-streamlit"]
    if project_path is not None:
        args.append(str(project_path))
    args += ["--port", str(port)]
    if host:
        args += ["--host", host]
    if headless:
        args.append("--headless")
    return args


def _launcher_env() -> dict[str, str]:
    """Child env that avoids parent-kernel path injection."""
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"

    python_dir = str(Path(sys.executable).resolve().parent)
    path_parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    norm_python_dir = os.path.normcase(os.path.normpath(python_dir))
    deduped = [
        p
        for p in path_parts
        if os.path.normcase(os.path.normpath(p)) != norm_python_dir
    ]
    env["PATH"] = os.pathsep.join([python_dir, *deduped])
    return env


def launch_streamlit(
    project_path: str | Path | None = None,
    *,
    data: Any | None = None,
    port: int = 8501,
    block: bool = False,
    headless: bool = False,
    host: str | None = "localhost",
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

    url = _url_for(host, port)
    print(f"easy_glm workbench: opening at {url}")
    if block:
        proc = subprocess.Popen(
            _streamlit_args(
                project_path=project_path, port=port, headless=headless, host=host
            ),
            env=_launcher_env(),
        )
    else:
        fd, log_path = tempfile.mkstemp(prefix="easy_glm_workbench_", suffix=".log")
        with os.fdopen(fd, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                _launcher_args(
                    project_path=project_path, port=port, headless=headless, host=host
                ),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=_launcher_env(),
            )
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            print(
                "easy_glm workbench: process started; "
                f"if startup fails, see {log_path}",
            )
            return proc
        details = Path(log_path).read_text(encoding="utf-8", errors="replace").strip()
        if not details:
            details = "(no child-process output)"
        raise RuntimeError(
            "easy_glm workbench failed to start. "
            f"Command: {' '.join(_launcher_args(project_path=project_path, port=port, headless=headless, host=host))}\n"
            f"Expected URL: {url}\n"
            f"Startup output:\n{details}"
        )
    if block:
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
    return proc


def launch(
    project_path: str | Path | None = None,
    *,
    data: Any | None = None,
    port: int = 8501,
    block: bool = False,
    headless: bool = False,
    host: str | None = "localhost",
    legacy_streamlit: bool = False,
) -> subprocess.Popen:
    """Open the Svelte workbench and return its server process.

    Accept a project JSON or a pandas/Polars frame. With neither, open an empty
    workspace. ``headless`` suppresses the browser; ``block`` waits until exit.
    The workbench binds to loopback only. Applied edits stay in this session;
    download the project JSON to keep them. The legacy Streamlit interface is
    available with ``legacy_streamlit=True``.
    """
    if legacy_streamlit:
        return launch_streamlit(
            project_path,
            data=data,
            port=port,
            block=block,
            headless=headless,
            host=host,
        )
    from easy_glm.desktop import launch as launch_desktop

    process = launch_desktop(
        project_path, data=data, port=port, open_browser=not headless, host=host
    )
    if block:
        try:
            # A bounded wait keeps Python responsive to Ctrl+C across runtimes.
            while True:
                try:
                    process.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return process


__all__ = ["launch", "launch_streamlit"]
