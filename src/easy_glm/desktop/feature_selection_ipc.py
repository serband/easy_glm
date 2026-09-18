"""In-memory messages for the feature-selection subprocess."""

from __future__ import annotations

import json
from typing import Any, BinaryIO

import polars as pl

from easy_glm.desktop.fit_worker import json_safe
from easy_glm.workflow.project import Project

MESSAGE_LIMIT = 8_000_000


def read_message(stream: BinaryIO) -> dict[str, Any] | None:
    line = stream.readline(MESSAGE_LIMIT + 1)
    if not line:
        return None
    if len(line) > MESSAGE_LIMIT or not line.endswith(b"\n"):
        raise ValueError(
            "Feature selection message exceeded its size limit or was incomplete."
        )
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Invalid feature selection message.")
    return value


def write_message(stream: BinaryIO, value: dict[str, Any]) -> None:
    message = (json.dumps(json_safe(value), allow_nan=False) + "\n").encode("utf-8")
    if len(message) > MESSAGE_LIMIT:
        raise ValueError("Feature selection message exceeded its size limit.")
    stream.write(message)
    stream.flush()


def write_request(
    stream: BinaryIO, project: Project, raw: pl.DataFrame, options: dict[str, Any]
) -> None:
    write_message(stream, {"project": project.to_dict(), "options": options})
    raw.write_ipc_stream(stream)
    stream.flush()


def read_request(stream: BinaryIO) -> tuple[Project, pl.DataFrame, dict[str, Any]]:
    request = read_message(stream)
    if request is None:
        raise ValueError("Feature selection did not receive its input.")
    return (
        Project.from_dict(request["project"]),
        pl.read_ipc_stream(stream),
        request["options"],
    )
