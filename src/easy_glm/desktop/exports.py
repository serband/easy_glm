"""Read-only attachments from an applicable fitted model and applied tables."""

from __future__ import annotations

import math
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import polars as pl

from easy_glm.desktop.diagnostic_views import compatible
from easy_glm.desktop.modeling import Revision
from easy_glm.workflow.export import to_script
from easy_glm.workflow.prep import prepare
from easy_glm.workflow.project import Project, safe_filename
from easy_glm.workflow.report import to_report_html
from easy_glm.workflow.run import ModelRun, rebuild_rate_model

ExportFormat = Literal["xlsx", "easyglm", "python", "html"]


class ExportRequest(Revision):
    format: ExportFormat
    challenger: str | None = None


@dataclass(frozen=True)
class ExportAttachment:
    content: bytes
    filename: str
    media_type: str

    @property
    def disposition(self) -> str:
        ascii_name = self.filename.encode("ascii", "replace").decode().replace("?", "_")
        return f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(self.filename, safe="")}'


def _report_importance(packet: dict | None, training_rows: int) -> pl.DataFrame | None:
    """A damaged diagnostic cache is a miss, never a broken report download."""
    if (
        packet is None
        or packet.get("basis") != "original"
        or packet.get("subset") != "train"
        or packet.get("repeats") != 5
        or packet.get("seed") != 42
        or packet.get("training_rows") != training_rows
        or not isinstance(packet.get("rows"), list)
    ):
        return None
    rows = packet["rows"]
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("variable"), str):
            return None
        for key in ("importance", "std"):
            value = row.get(key)
            if not isinstance(value, int | float) or not math.isfinite(value):
                return None
        if row["std"] < 0:
            return None
    return pl.DataFrame(rows)


def export_attachment(
    project: Project,
    raw: pl.DataFrame,
    sources: dict[str, Path],
    name: str,
    format: ExportFormat,
    challenger: str | None = None,
) -> ExportAttachment:
    """Render a captured project/fit snapshot; never fit or change source artifacts.

    ``sources`` contains application-owned fit folders, never user-supplied paths.
    Unpickled runs are private copies, rebuilt with the captured applied adjustments.
    The caller releases the project lock before preparation and rendering.
    """
    frame = prepare(project, raw)
    runs: dict[str, ModelRun] = {}
    for model_name, source in sources.items():
        with (source / "fit.pkl").open("rb") as handle:
            run = pickle.load(handle)
        if not isinstance(run, ModelRun) or run.name != model_name:
            raise ValueError("The fitted artifact does not match the selected model.")
        runs[model_name] = rebuild_rate_model(project, run, frame)
    run = runs[name]
    if challenger is not None:
        compatible(run, runs[challenger])
    prefix = f"{safe_filename(project.name, 'project')}_{safe_filename(name)}"
    if format == "python":
        return ExportAttachment(
            to_script(project, name, run=run, output_prefix=prefix).encode("utf-8"),
            prefix + ".py",
            "text/x-python",
        )
    if format == "html":
        from easy_glm.desktop.importance_cache import read_packet

        packet = read_packet(sources[name])
        importance = _report_importance(packet, run.train_rows)
        return ExportAttachment(
            to_report_html(
                project,
                runs,
                frame,
                champion=name,
                challenger=challenger,
                importance=importance,
            ).encode("utf-8"),
            prefix + "_report.html",
            "text/html",
        )
    with tempfile.TemporaryDirectory(prefix="easyglm_export_") as folder:
        if format == "xlsx":
            filename = prefix + "_rate_tables.xlsx"
            path = Path(folder) / filename
            run.rate_model.to_excel(path)
            media_type = (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        elif format == "easyglm":
            filename = prefix + ".easyglm"
            path = Path(folder) / filename
            run.rate_model.to_json(path)
            media_type = "application/json"
        else:
            raise ValueError("Choose a supported export format.")
        return ExportAttachment(path.read_bytes(), filename, media_type)
