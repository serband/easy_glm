"""Load local workbench inputs without mutating the active session."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from pydantic import Field, model_validator

import easy_glm
from easy_glm.desktop.modeling import Revision
from easy_glm.workflow.prep import infer_source_type, load_source, prepare
from easy_glm.workflow.project import Project
from easy_glm.workflow.starters import starter_project

SourceType = Literal["auto", "csv", "parquet", "excel", "xlsx", "ipc", "sas7bdat"]
Example = Literal["french_motor", "swedish_motorcycle"]
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


class OpenProject(Revision):
    kind: Literal["data", "project", "example"]
    path: str | None = Field(default=None, min_length=1)
    example: Example | None = None
    source_type: SourceType = "auto"

    @model_validator(mode="after")
    def required_input(self) -> OpenProject:
        if self.kind == "example":
            if self.example is None:
                raise ValueError("Choose an example dataset.")
        elif not self.path or not self.path.strip():
            raise ValueError("Enter the path to a data file or saved project.")
        return self


def new_input_folder() -> Path:
    """Retain successful sources for exported projects, like the Python launcher."""
    return Path(tempfile.mkdtemp(prefix="easy_glm_workbench_"))


def upload_basename(filename: str) -> str:
    """Use only a safe basename, including for names sent by Windows browsers."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = re.sub(r'[\x00-\x1f\x7f<>:"|?*]', "_", name)
    if name in ("", ".", ".."):
        raise ValueError("Choose a file with a valid filename.")
    name = name.rstrip(" .")
    if not name:
        raise ValueError("Choose a file with a valid filename.")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
    }
    if name.split(".", 1)[0].upper() in reserved:
        name = "_" + name
    if len(name.encode("utf-8")) > 240:
        raise ValueError("The filename is too long; rename the file and try again.")
    return name


def load_example_input(example: str, folder: Path) -> tuple[Project, pl.DataFrame]:
    """Load a cached public example and persist its source without fitting it."""
    if example == "french_motor":
        frame = easy_glm.load_external_dataframe()
        frame = frame.sample(n=min(50_000, frame.height), seed=42)
        filename = "french_motor_sample.parquet"
    elif example == "swedish_motorcycle":
        frame = easy_glm.load_swedish_motorcycle_data()
        filename = "swedish_motorcycle_sample.parquet"
    else:
        raise ValueError("Choose the French motor or Swedish motorcycle example.")
    path = folder / filename
    project = starter_project(example, str(path))
    # Examples supply data and roles, never an automatically created model.
    project.models.clear()
    project.champion = None
    time_column = "SyntheticYear"
    while time_column in frame.columns:
        time_column += "_demo"
    years = np.resize(np.arange(2020, 2025, dtype=np.int64), frame.height)
    np.random.default_rng(42).shuffle(years)
    frame = frame.with_columns(pl.Series(time_column, years))
    project.data.roles[time_column] = "time"
    project.exploration["example"] = {"synthetic_time_column": time_column}
    prepare(project, frame)
    frame.write_parquet(path)
    return project, frame


def load_project_input(
    kind: str,
    path: str | Path,
    source_type: SourceType = "auto",
    *,
    uploaded: bool = False,
) -> tuple[Project, pl.DataFrame]:
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
            if uploaded:
                raise ValueError(
                    "This project uses a relative data path. Open the project by its "
                    "local file path so EasyGLM can find the data beside it."
                )
            data_path = source.parent / data_path
        project.data.source.path = str(data_path.resolve())
    elif kind == "data":
        if source_type == "auto" and source.suffix.lower() not in {
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
        project.data.source.type = (
            infer_source_type(source) if source_type == "auto" else source_type
        )
        project.data.split.mode = "random"
    else:
        raise ValueError("Choose a data file or a saved project.")
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
