"""End-to-end persona runs against a real workbench server.

Skipped unless ``EASY_GLM_E2E=1`` and Playwright is importable. The server is
started from the interpreter named by ``EASY_GLM_SERVER_PYTHON`` (default: the
current one) so the tests themselves only need polars, numpy and playwright —
they never import ``easy_glm``; project files are written as plain JSON.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl
import pytest

playwright = pytest.importorskip("playwright.sync_api")
if not os.environ.get("EASY_GLM_E2E"):
    pytest.skip(
        "set EASY_GLM_E2E=1 to run the persona e2e tests", allow_module_level=True
    )

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "french_motor_50k.parquet"
SERVER_PYTHON = os.environ.get("EASY_GLM_SERVER_PYTHON", sys.executable)
MAIN = ROOT / "src" / "easy_glm" / "app" / "main.py"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_data(folder: Path) -> Path:
    """French-motor fixture plus a synthetic current premium (for the offset step)."""
    df = pl.read_parquet(FIXTURE)
    rng = np.random.default_rng(3)
    base = 0.07 * (1 + 0.01 * (df["BonusMalus"].cast(pl.Float64) - 100))
    noise = rng.uniform(0.85, 1.15, df.height)
    df = df.with_columns(
        (base * noise * 250.0).alias("current_premium"),
    )
    path = folder / "policies.parquet"
    df.write_parquet(path)
    return path


def project_dict(name: str, data: Path, *, offset: bool, cv: bool) -> dict:
    roles = {
        "IDpol": "id",
        "ClaimNb": "target",
        "Exposure": "weight",
        "DrivAge": "predictor",
        "VehAge": "predictor",
        "BonusMalus": "predictor",
        "Density": "predictor",
        "VehPower": "predictor",
        "VehGas": "predictor",
        "Region": "predictor",
        "VehBrand": "predictor",
        "Area": "predictor",
        "current_premium": "ignore",
    }
    derived = []
    if offset:
        derived.append(
            {"name": "log_current_premium", "expr": "pl.col('current_premium').log()"}
        )
        roles["log_current_premium"] = "offset"
    predictors = [
        "DrivAge",
        "VehAge",
        "BonusMalus",
        "Density",
        "VehPower",
        "VehGas",
        "Region",
    ]
    return {
        "name": name,
        "version": 2,
        "data": {
            "source": {"type": "parquet", "path": str(data), "options": {}},
            "roles": roles,
            "recodes": {"Area": {"mapping": {"E": "D", "F": "D"}, "default": None}},
            "derived": derived,
            "filters": ["pl.col('Exposure') > 0.02"],
            "split": {
                "mode": "random",
                "column": "traintest",
                "fraction": 0.7,
                "seed": 7,
            },
        },
        "design": {"variables": {}},
        "models": {
            "freq_v1": {
                "family": "poisson",
                "target": "ClaimNb",
                "weight": "Exposure",
                "divide_target_by_weight": True,
                "predictors": predictors,
                "penalty": (
                    {"alpha": None, "cv": 2, "n_alphas": 5, "l1_ratio": 1.0}
                    if cv
                    else {"alpha": 0.001, "cv": None}
                ),
            }
        },
        "champion": "freq_v1",
    }


class Server:
    def __init__(self, project_path: Path, port: int):
        self.port = port
        self.url = f"http://localhost:{port}"
        self.proc = subprocess.Popen(
            [
                SERVER_PYTHON,
                "-m",
                "streamlit",
                "run",
                str(MAIN),
                "--server.port",
                str(port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
                "--",
                f"--project={project_path}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{self.url}/_stcore/health", timeout=2
                ) as r:
                    if r.status == 200:
                        return
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
            if self.proc.poll() is not None:
                break
        out = self.proc.stdout.read() if self.proc.stdout else ""
        raise RuntimeError(f"workbench did not start on {port}:\n{out[-2000:]}")

    def stop(self) -> str:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        return self.proc.stdout.read() if self.proc.stdout else ""


@pytest.fixture(scope="session")
def e2e_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("e2e")


@pytest.fixture(scope="session")
def data_path(e2e_dir) -> Path:
    return write_data(e2e_dir)


def _project_file(
    folder: Path, name: str, data: Path, *, offset: bool, cv: bool
) -> Path:
    path = folder / f"{name}.easyglm-project.json"
    path.write_text(
        json.dumps(project_dict(name, data, offset=offset, cv=cv), indent=2)
    )
    return path


@pytest.fixture(scope="module")
def actuary_server(e2e_dir, data_path):
    path = _project_file(e2e_dir, "actuary", data_path, offset=False, cv=False)
    srv = Server(path, free_port())
    yield srv, path
    log = srv.stop()
    assert "Traceback" not in log, log[-3000:]


@pytest.fixture(scope="module")
def scientist_server(e2e_dir, data_path):
    path = _project_file(e2e_dir, "scientist", data_path, offset=False, cv=True)
    srv = Server(path, free_port())
    yield srv, path
    log = srv.stop()
    assert "Traceback" not in log, log[-3000:]


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()
