"""The installed public workbench launches Svelte, including in-memory frames."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from easy_glm import launch_workbench
from easy_glm.app import _launcher_env


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get(url, token=None):
    headers = {"X-EasyGLM-Token": token} if token else {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(
        urllib.request.Request(url, headers=headers), timeout=1
    ) as response:
        return response.read()


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@pytest.mark.parametrize(
    "frame", [pl.DataFrame({"claims": [0, 1, 2]}), pd.DataFrame({"claims": [0, 1, 2]})]
)
def test_public_in_memory_launch_is_ready_svelte_and_owns_the_process(
    frame, monkeypatch
):
    opened = []
    monkeypatch.setattr("easy_glm.desktop.webbrowser.open", opened.append)
    port = free_port()
    process = launch_workbench(data=frame, port=port, headless=True)
    url = f"http://127.0.0.1:{port}"
    try:
        assert process.poll() is None
        assert "easy_glm.desktop" in process.args
        assert b'<div id="app">' in get(url)
        session = json.loads(get(url + "/api/session"))
        snapshot = json.loads(get(url + "/api/variables", session["token"]))
        assert snapshot["row_count"] == 3
        assert snapshot["columns"] == [{"name": "claims", "dtype": "Int64"}]
        assert opened == []
    finally:
        stop(process)
    with pytest.raises(urllib.error.URLError):
        get(url + "/health")


def test_empty_default_opens_browser_only_after_ready(monkeypatch):
    port = free_port()
    visited = []
    monkeypatch.setattr(
        "easy_glm.desktop.webbrowser.open",
        lambda url: visited.append(json.loads(get(url + "/health"))),
    )
    process = launch_workbench(port=port)
    try:
        assert len(visited) == 1 and visited[0]["status"] == "ok"
        url = f"http://127.0.0.1:{port}"
        token = json.loads(get(url + "/api/session"))["token"]
        snapshot = json.loads(get(url + "/api/variables", token))
        assert snapshot["row_count"] == 0 and snapshot["columns"] == []
        assert snapshot["models"] == []
    finally:
        stop(process)


def test_public_launcher_failure_and_host_errors_are_actionable(tmp_path):
    with pytest.raises(ValueError, match="workbench is local"):
        launch_workbench(host="0.0.0.0", headless=True)
    with pytest.raises(RuntimeError, match="File not found"):
        launch_workbench(tmp_path / "missing.json", port=free_port(), headless=True)


def test_public_blocking_mode_terminates_child_on_interrupt(monkeypatch):
    class Process:
        terminated = False
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt

        def terminate(self):
            self.terminated = True

    process = Process()
    monkeypatch.setattr("easy_glm.desktop.launch", lambda *args, **kwargs: process)
    assert launch_workbench(block=True, headless=True) is process
    assert process.terminated and process.calls == 2


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, "-m", "easy_glm.app"],
        [sys.executable, "-m", "easy_glm.cli", "workbench"],
        [str(Path(sys.executable).parent / "easy-glm-workbench")],
    ],
)
def test_public_command_starts_headless_svelte_and_exits_with_interrupt(
    command, tmp_path
):
    import signal

    port = free_port()
    log = tmp_path / "launcher.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [*command, "--port", str(port), "--headless"],
            env=_launcher_env(),
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if json.loads(get(url + "/health"))["status"] == "ok":
                    break
            except urllib.error.URLError:
                if process.poll() is not None:
                    pytest.fail(log.read_text())
                time.sleep(0.05)
        else:
            pytest.fail(log.read_text())
        assert b'<div id="app">' in get(url)
        process.send_signal(signal.SIGINT)
        process.wait(timeout=10)
        with pytest.raises(urllib.error.URLError):
            get(url + "/health")
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
