"""Loopback API for the experimental Svelte Variables workbench.

The project is an in-memory copy. No endpoint writes source files or fits a model.
"""

from __future__ import annotations

import json
import secrets
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from easy_glm.workflow.explore import univariate
from easy_glm.workflow.prep import apply_variables
from easy_glm.workflow.project import SINGLE_ROLES, Project
from easy_glm.workflow.variables import (
    BULK_ROLE_GROUPS,
    apply_roles_grid,
    parse_variable_setup_json,
    variable_setup_changes,
    variable_setup_json,
)


class Edit(BaseModel):
    """A complete variable draft based on a specific server revision."""

    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=0)
    setup: dict[str, Any]


def create_app(
    project: Project, raw: pl.DataFrame, *, port: int, launch_id: str = ""
) -> FastAPI:
    """Serve one local, in-memory project and bundled assets."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    current = deepcopy(project)
    revision = 0
    lock = threading.RLock()
    token = secrets.token_urlsafe(32)
    host = f"127.0.0.1:{port}"
    origin = f"http://{host}"
    static = Path(__file__).with_name("static")

    @app.middleware("http")
    async def local_only(request: Request, call_next: Any) -> Any:
        if request.headers.get("host") != host:
            return JSONResponse({"detail": "Use the loopback launch URL."}, 403)
        supplied_origin = request.headers.get("origin")
        if supplied_origin and supplied_origin != origin:
            return JSONResponse({"detail": "Cross-origin access is refused."}, 403)
        top_level_navigation = (
            request.method == "GET"
            and request.url.path == "/"
            and request.headers.get("sec-fetch-mode") == "navigate"
            and request.headers.get("sec-fetch-dest") == "document"
        )
        if (
            request.headers.get("sec-fetch-site") == "cross-site"
            and not top_level_navigation
        ):
            return JSONResponse({"detail": "Cross-site access is refused."}, 403)
        if request.url.path.startswith("/api/") and request.url.path != "/api/session":
            if not secrets.compare_digest(
                request.headers.get("x-easyglm-token", ""), token
            ):
                return JSONResponse({"detail": "Reopen the local workbench tab."}, 403)
        if request.method == "POST":
            if (
                request.headers.get("content-type", "").split(";")[0]
                != "application/json"
            ):
                return JSONResponse({"detail": "JSON is required."}, 415)
            # Bound the actual stream, not just the optional Content-Length header.
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 8_000_000:
                    return JSONResponse({"detail": "Variable draft exceeds 8 MB."}, 413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'none'"
        )
        return response

    def snapshot() -> dict[str, Any]:
        return {
            "name": current.name,
            "revision": revision,
            "row_count": raw.height,
            "columns": [
                {"name": name, "dtype": str(dtype)}
                for name, dtype in raw.schema.items()
            ],
            "setup": json.loads(variable_setup_json(current, raw.columns)),
            "single_roles": list(SINGLE_ROLES),
            "group_roles": list(BULK_ROLE_GROUPS),
            "models": list(current.models),
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "launch_id": launch_id}

    @app.get("/api/session")
    def session() -> dict[str, str]:
        return {"token": token}

    @app.get("/api/variables")
    def variables() -> dict[str, Any]:
        with lock:
            return snapshot()

    def candidate(
        edit: Edit,
    ) -> tuple[Project, list[dict[str, str]], list[tuple[str, str]]]:
        if edit.revision != revision:
            raise HTTPException(
                409,
                "Another tab applied changes. Reload saved variables before applying.",
            )
        rows, errors = parse_variable_setup_json(
            current, raw.columns, json.dumps(edit.setup)
        )
        if errors:
            raise HTTPException(422, errors)
        result = deepcopy(current)
        _, notices = apply_roles_grid(result, raw.columns, rows)
        return result, variable_setup_changes(current, raw.columns, rows), notices

    @app.post("/api/variables/preview")
    def preview(edit: Edit) -> dict[str, Any]:
        with lock:
            _, changes, notices = candidate(edit)
            return {"changes": changes, "notices": notices}

    @app.post("/api/variables/apply")
    def apply(edit: Edit) -> dict[str, Any]:
        nonlocal current, revision
        with lock:
            result, changes, notices = candidate(edit)
            if changes:
                current = result
                revision += 1
            return {**snapshot(), "notices": notices}

    @app.get("/api/project")
    def export_project() -> dict[str, Any]:
        with lock:
            return current.to_dict()

    @app.get("/api/plot")
    def plot(column: str) -> dict[str, Any]:
        # FastAPI executes this synchronous route in its worker thread pool.
        # Copy briefly under lock; aggregation does not block edits or health.
        with lock:
            saved, plot_revision = deepcopy(current), revision
        if column not in raw.columns:
            raise HTTPException(404, "Unknown source column.")
        try:
            frame = raw.head(50_000)
            prepared = apply_variables(frame, saved.data)
            final = saved.data.renames.get(column, column)
            result = univariate(prepared, final, n_bins=16, max_levels=25)
            table = result["table"].select("label", "exposure").to_dicts()
            return {
                "column": final,
                "revision": plot_revision,
                "rows": prepared.height,
                "sampled": raw.height > 50_000,
                "table": table,
            }
        except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError) as exc:
            raise HTTPException(422, f"Cannot preview this column: {exc}") from exc

    app.mount(
        "/assets",
        StaticFiles(directory=static / "assets", check_dir=False),
        name="assets",
    )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static / "index.html")

    return app
