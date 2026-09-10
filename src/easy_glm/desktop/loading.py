"""Load local workbench inputs without mutating the active session."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import Field

from easy_glm.desktop.modeling import Revision
from easy_glm.workflow.prep import infer_source_type, load_source, prepare
from easy_glm.workflow.project import Project


class OpenProject(Revision):
    kind: Literal["data", "project"]
    path: str = Field(min_length=1)


def load_project_input(kind: str, path: str | Path) -> tuple[Project, pl.DataFrame]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"File not found: {source}")
    if kind == "project":
        if source.suffix.lower() == ".easyglm":
            raise ValueError("Choose a project JSON file; .easyglm is a scoring model.")
        project = Project.from_json(source)
        if not project.data.source.path:
            return project, pl.DataFrame()
        data_path = Path(project.data.source.path).expanduser()
        if not data_path.is_absolute():
            data_path = source.parent / data_path
        project.data.source.path = str(data_path.resolve())
    else:
        if source.suffix.lower() not in {
            ".parquet",
            ".pq",
            ".csv",
            ".txt",
            ".xlsx",
            ".xls",
            ".ipc",
            ".arrow",
            ".feather",
            ".sas7bdat",
        }:
            raise ValueError("Choose a CSV, Parquet, Excel, Arrow or SAS data file.")
        project = Project(name=source.stem)
        project.data.source.path = str(source)
        project.data.source.type = infer_source_type(source)
        project.data.split.mode = "random"
    raw = load_source(project.data.source)
    if kind == "data":
        split = "traintest"
        suffix = 2
        while split in raw.columns:
            split = f"traintest_{suffix}"
            suffix += 1
        project.data.split.column = split
    # Catch invalid filters, recodes and derived expressions before replacement.
    # Missing modelling roles are valid during project setup.
    prepare(project, raw)
    return project, raw
