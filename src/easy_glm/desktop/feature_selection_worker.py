"""Isolated feature selection worker. Its folder is allocated by the server."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

from easy_glm.desktop.fit_worker import write_json
from easy_glm.workflow.project import Project


def main() -> None:
    folder = Path(sys.argv[1])

    def progress(value: dict[str, Any]) -> None:
        write_json(folder / "progress.json", value)

    try:
        from easy_glm.workflow.feature_selection import select_variables

        result = select_variables(
            Project.from_json(folder / "project.json"),
            pl.read_parquet(folder / "raw.parquet"),
            **json.loads((folder / "options.json").read_text(encoding="utf-8")),
            progress=progress,
        )
    except Exception as exc:  # Worker boundary: expose only an actionable error.
        result = {"error": str(exc)}
    write_json(folder / "result.json", result)


if __name__ == "__main__":
    main()
