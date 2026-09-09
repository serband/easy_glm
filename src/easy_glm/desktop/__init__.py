"""Experimental local Svelte workbench. Separate from the Streamlit workbench."""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any


def launch(
    project_path: str | Path | None = None,
    *,
    data: Any | None = None,
    port: int = 0,
    open_browser: bool = True,
) -> subprocess.Popen:
    """Launch the experiment with a project copy or a pandas/Polars frame.

    Return the server process; call ``process.terminate()`` to stop it. Applied
    edits live only for this process. Export project JSON to retain them.
    """
    from easy_glm.app import _launcher_env, _temporary_project_for

    if data is not None and project_path is not None:
        raise ValueError("Pass project_path or data, not both.")
    if data is not None:
        project_path = _temporary_project_for(data)
    if not port:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535, or 0 for automatic")
    launch_id = secrets.token_urlsafe(16)
    args = [
        sys.executable,
        "-m",
        "easy_glm.desktop",
        "--port",
        str(port),
        f"--launch-id={launch_id}",
    ]
    if project_path is not None:
        args += ["--project", str(Path(project_path).resolve())]
    fd, log_path = tempfile.mkstemp(prefix="easyglm_svelte_", suffix=".log")
    with os.fdopen(fd, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            args, env=_launcher_env(), stdout=log, stderr=subprocess.STDOUT
        )
    url = f"http://127.0.0.1:{port}"
    # A real readiness check, with proxy bypass for corporate environments.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and process.poll() is None:
        try:
            with opener.open(url + "/health", timeout=0.3) as response:
                if (
                    response.status == 200
                    and json.load(response).get("launch_id") == launch_id
                ):
                    print(f"EasyGLM experiment: {url}\nServer log: {log_path}")
                    if open_browser:
                        webbrowser.open(url)
                    return process
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.05)
    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError(
        f"Workbench did not become ready. See {log_path}\n" + Path(log_path).read_text()
    )
