from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

from easy_glm.engine import RateModel


def launch_editor(
    rm: RateModel,
    data: pl.DataFrame | None = None,
    test_data: pl.DataFrame | None = None,
    port: int = 8501,
    **kwargs: Any,
) -> None:
    tmpdir = tempfile.mkdtemp(prefix="easy_glm_ui_")
    model_path = os.path.join(tmpdir, "model.easyglm")
    rm.to_json(model_path)

    args: list[str] = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).parent / "app.py"),
        "--",
        f"--model-path={model_path}",
        f"--server.port={str(port)}",
    ]

    if data is not None:
        data_path = os.path.join(tmpdir, "data.parquet")
        data.write_parquet(data_path)
        args.append(f"--data-path={data_path}")

    if test_data is not None:
        test_path = os.path.join(tmpdir, "test_data.parquet")
        test_data.write_parquet(test_path)
        args.append(f"--test-path={test_path}")

    for key, value in kwargs.items():
        if value is not None:
            args.append(f"--{key}={value}")

    subprocess.Popen(args)


def _parse_args() -> dict[str, str]:
    opts: dict[str, str] = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            if "=" in arg:
                key, val = arg.split("=", 1)
                opts[key[2:].replace("-", "_")] = val
    return opts
