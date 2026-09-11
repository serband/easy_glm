"""Loopback API for the experimental Svelte Variables workbench.

The project is an in-memory copy. Fits run in isolated child processes.
No endpoint overwrites source files or existing Streamlit fit caches.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import threading
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import polars as pl
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from easy_glm.desktop.exploration import ExplorationCache
from easy_glm.desktop.exports import ExportRequest, export_attachment
from easy_glm.desktop.jobs import FitJobs, model_key
from easy_glm.desktop.loading import (
    MAX_UPLOAD_BYTES,
    OpenProject,
    SourceType,
    load_example_input,
    load_project_input,
    new_input_folder,
    upload_basename,
)
from easy_glm.desktop.modeling import (
    ModelEdit,
    ReducedModelEdit,
    Revision,
    SplitEdit,
    edit_model,
    setup_info,
)
from easy_glm.desktop.reviews import ReviewEdit, ReviewJobs
from easy_glm.desktop.screening import ScreeningJobs, ScreeningRequest
from easy_glm.desktop.splits import (
    apply_split_setup,
    split_counts,
    split_setup,
    split_values,
)
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
    session_id: str
    revision: int = Field(ge=0)
    setup: dict[str, Any]


def create_app(
    project: Project,
    raw: pl.DataFrame,
    *,
    port: int,
    launch_id: str = "",
    restore_folder: Path | None = None,
) -> FastAPI:
    """Serve one local, in-memory project and bundled assets."""
    jobs = FitJobs()
    if restore_folder is not None:
        jobs.restore(project, raw, restore_folder)
    reviews = ReviewJobs()
    exploration = ExplorationCache()
    screenings = ScreeningJobs()
    undo_steps = {}
    redo_steps = {}
    if restore_folder is not None and (restore_folder / "history.json").exists():
        history = json.loads((restore_folder / "history.json").read_text())
        for name, stacks in history.items():
            if name not in project.models:
                raise ValueError("Restored edit history refers to an unknown model.")
            for key, destination in (("undo", undo_steps), ("redo", redo_steps)):
                entries = stacks.get(key, [])
                if len(entries) > 50:
                    raise ValueError("Restored edit history exceeds its session limit.")
                destination[name] = []
                for entry in entries:
                    saved = project.to_dict()
                    saved["models"][name]["adjustments"] = entry["adjustments"]
                    saved["models"][name]["base_rate_override"] = entry[
                        "base_rate_override"
                    ]
                    cfg = Project.from_dict(saved).models[name]
                    destination[name].append((cfg.adjustments, cfg.base_rate_override))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        screenings.close()
        reviews.close()
        jobs.close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    current = deepcopy(project)
    revision = 0
    lock = threading.RLock()

    def complete_fit(job: dict[str, Any], result: dict[str, Any]) -> None:
        nonlocal revision
        with lock, jobs.lock:
            name = job["name"]
            if jobs.jobs.get(name) is not job or job["cancel"]:
                return
            if job["key"] != model_key(current, name):
                job.update(
                    status="stale", message="Settings changed. Fit this model again."
                )
                return
            cfg = current.models[name]
            cleared = bool(cfg.adjustments) or cfg.base_rate_override is not None
            cfg.adjustments = []
            cfg.base_rate_override = None
            undo_steps.pop(name, None)
            redo_steps.pop(name, None)
            if cleared:
                revision += 1
                screenings.invalidate((session_id, project_id, revision))
            job.update(
                key=model_key(current, name),
                status="complete",
                message=(
                    "Unchanged main effects reused. Diagnostics and rate tables are ready."
                    if result.get("reuse", {}).get("main_effects")
                    else "Diagnostics and rate tables are ready."
                ),
                result=result,
            )

    jobs.on_complete = complete_fit
    token = secrets.token_urlsafe(32)
    session_id = secrets.token_urlsafe(16)
    project_id = hashlib.sha256(
        json.dumps(
            {
                "name": project.name,
                "source": project.to_dict()["data"]["source"],
                "schema": [(name, str(dtype)) for name, dtype in raw.schema.items()],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
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
                return JSONResponse(
                    {
                        "detail": "The local session changed. Reconnect to keep your draft.",
                        "code": "session_expired",
                    },
                    401,
                    headers={"Cache-Control": "no-store"},
                )
        if request.method == "POST":
            upload = request.url.path == "/api/project/upload"
            required_type = "application/octet-stream" if upload else "application/json"
            if request.headers.get("content-type", "").split(";")[0] != required_type:
                return JSONResponse({"detail": f"{required_type} is required."}, 415)
            if not upload:
                # Uploads are bounded and streamed to disk by their own endpoint.
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > 8_000_000:
                        return JSONResponse(
                            {"detail": "Variable draft exceeds 8 MB."}, 413
                        )
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
            "session_id": session_id,
            "project_id": project_id,
            "revision": revision,
            "row_count": raw.height,
            "columns": [
                {"name": name, "dtype": str(dtype)}
                for name, dtype in raw.schema.items()
            ],
            "setup": {
                **json.loads(variable_setup_json(current, raw.columns)),
                "split": split_setup(current),
            },
            "single_roles": list(SINGLE_ROLES),
            "group_roles": list(BULK_ROLE_GROUPS),
            "models": list(current.models),
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "launch_id": launch_id}

    @app.get("/api/session")
    def session() -> dict[str, str]:
        return {"token": token, "session_id": session_id}

    @app.get("/api/variables")
    def variables() -> dict[str, Any]:
        with lock:
            return snapshot()

    def candidate(
        edit: Edit | ScreeningRequest,
        *,
        validate_split: bool = True,
    ) -> tuple[Project, list[dict[str, str]], list[tuple[str, str]]]:
        if edit.session_id != session_id:
            raise HTTPException(
                409,
                "The server restarted. Reconnect and preview your draft again before applying.",
            )
        if edit.revision != revision:
            raise HTTPException(
                409,
                "Another tab applied changes. Reload saved variables before applying.",
            )
        setup = dict(edit.setup)
        requested_split = setup.pop("split", None)
        rows, errors = parse_variable_setup_json(
            current, raw.columns, json.dumps(setup)
        )
        if errors:
            raise HTTPException(422, errors)
        result = deepcopy(current)
        _, notices = apply_roles_grid(result, raw.columns, rows)
        try:
            if requested_split is not None:
                if not isinstance(requested_split, dict):
                    raise ValueError("Split settings must be a JSON object.")
                apply_split_setup(result, requested_split)
            time_column = result.column_with_role("time")
            if time_column:
                from easy_glm.workflow.time_diagnostics import time_values

                time_values(apply_variables(raw, result.data)[time_column])
            changed_split = result.data.split != current.data.split
            if validate_split and (changed_split or result.column_with_role("split")):
                counts = split_counts(result, raw)
                notices.append(
                    (
                        "info",
                        f'{counts["train"]:,} training · {counts["holdout"]:,} holdout rows',
                    )
                )
        except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError) as exc:
            raise HTTPException(422, str(exc)) from exc
        changes = variable_setup_changes(current, raw.columns, rows)
        changed_split = result.data.split != current.data.split
        if changed_split:
            split = result.data.split
            changes.append(
                {
                    "raw column": (
                        requested_split.get("column", split.column)
                        if requested_split
                        else split.column
                    ),
                    "name": split.column,
                    "role": "split",
                    "type": (
                        f"Training: {split.train_value!r}; holdout: {split.holdout_value!r}"
                        if split.mode == "column"
                        else f"Random: {split.fraction:.0%} training; seed {split.seed}"
                    ),
                }
            )
        return result, changes, notices

    @app.post("/api/variables/split-values")
    def variable_split_values(edit: Edit) -> dict[str, Any]:
        with lock:
            saved, _, _ = candidate(edit, validate_split=False)
            frame = raw.clone()
        try:
            return split_values(saved, frame)
        except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/variables/screen", status_code=202)
    def screen_start(edit: ScreeningRequest) -> dict[str, Any]:
        with lock:
            saved, _, _ = candidate(edit)
            generation = session_id, project_id, revision
            try:
                return screenings.start(
                    saved,
                    raw.clone(),
                    generation,
                    edit.setup,
                    edit.options.model_dump(),
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.get("/api/screenings/{key}")
    def screen_status(key: str) -> dict[str, Any]:
        with lock:
            try:
                return screenings.status(key, (session_id, project_id, revision))
            except KeyError as exc:
                raise HTTPException(404, str(exc.args[0])) from exc

    @app.post("/api/screenings/{key}/cancel")
    def screen_cancel(key: str, edit: Revision) -> dict[str, Any]:
        with lock:
            check_revision(edit)
            try:
                screenings.cancel(key)
                return screenings.status(key, (session_id, project_id, revision))
            except KeyError as exc:
                raise HTTPException(404, str(exc.args[0])) from exc

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
                screenings.invalidate((session_id, project_id, revision))
                jobs.invalidate(current)
            return {**snapshot(), "notices": notices}

    def check_revision(edit: Revision) -> None:
        if edit.session_id != session_id or edit.revision != revision:
            raise HTTPException(
                409,
                "Applied settings changed. Reconnect and review the current settings before saving or fitting.",
            )

    def replace_project(
        edit: Revision, loaded: Project, loaded_raw: pl.DataFrame
    ) -> tuple[dict[str, Any], FitJobs, ReviewJobs]:
        nonlocal current, raw, revision, session_id, project_id, token, jobs, reviews, exploration
        with lock:
            check_revision(edit)
            previous_jobs, previous_reviews = jobs, reviews
            jobs, reviews = FitJobs(), ReviewJobs()
            exploration = ExplorationCache()
            jobs.on_complete = complete_fit
            current, raw = loaded, loaded_raw
            revision = 0
            session_id = secrets.token_urlsafe(16)
            project_id = secrets.token_hex(32)
            token = secrets.token_urlsafe(32)
            screenings.invalidate((session_id, project_id, revision))
            undo_steps.clear()
            redo_steps.clear()
            result = snapshot()
        return result, previous_jobs, previous_reviews

    @app.post("/api/project/open")
    def open_project(edit: OpenProject) -> dict[str, Any]:
        with lock:
            check_revision(edit)
        folder = None
        committed = False
        try:
            if edit.kind == "example":
                folder = new_input_folder()
                loaded, loaded_raw = load_example_input(str(edit.example), folder)
            else:
                loaded, loaded_raw = load_project_input(
                    edit.kind, str(edit.path), edit.source_type
                )
            result, previous_jobs, previous_reviews = replace_project(
                edit, loaded, loaded_raw
            )
            committed = True
        except HTTPException:
            raise
        except Exception as exc:  # Invalid inputs leave the current session intact.
            label = "example" if edit.kind == "example" else "file"
            raise HTTPException(422, f"Could not open this {label}: {exc}") from exc
        finally:
            if folder is not None and not committed:
                shutil.rmtree(folder, ignore_errors=True)
        # A finishing worker may acquire the project lock; never join it inside.
        previous_reviews.close()
        previous_jobs.close()
        return result

    @app.post("/api/project/upload")
    async def upload_project(
        request: Request,
        kind: Literal["data", "project"],
        filename: str = Query(min_length=1, max_length=255),
        session_id: str = Query(min_length=1),
        revision: int = Query(ge=0),
        source_type: SourceType = "auto",
    ) -> dict[str, Any]:
        edit = Revision(session_id=session_id, revision=revision)
        with lock:
            check_revision(edit)
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "The upload exceeds 512 MiB.")
        folder = None
        committed = False
        try:
            name = upload_basename(filename)
            folder = new_input_folder()
            source = folder / name
            received = 0
            with source.open("xb") as destination:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "The upload exceeds 512 MiB.")
                    await run_in_threadpool(destination.write, chunk)
            if received == 0:
                raise ValueError("Choose a non-empty file.")
            loaded, loaded_raw = await run_in_threadpool(
                load_project_input, kind, source, source_type, uploaded=True
            )
            result, previous_jobs, previous_reviews = await run_in_threadpool(
                replace_project, edit, loaded, loaded_raw
            )
            committed = True
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, f"Could not open this upload: {exc}") from exc
        finally:
            if folder is not None and not committed:
                await run_in_threadpool(shutil.rmtree, folder, ignore_errors=True)
        await run_in_threadpool(previous_reviews.close)
        await run_in_threadpool(previous_jobs.close)
        return result

    @app.get("/api/workbench")
    def workbench() -> dict[str, Any]:
        with lock:
            saved, saved_raw = deepcopy(current), raw.clone()
            saved_revision, saved_session = revision, session_id
            saved_jobs = jobs.status(saved)
        return {
            **setup_info(saved, saved_raw),
            "revision": saved_revision,
            "session_id": saved_session,
            "jobs": saved_jobs,
            "champion": saved.champion,
        }

    @app.post("/api/models/save")
    def save_model(edit: ModelEdit) -> dict[str, Any]:
        nonlocal current, revision
        with lock:
            check_revision(edit)
            try:
                candidate_project = edit_model(current, edit)
            except (ValueError, TypeError, KeyError) as exc:
                raise HTTPException(422, str(exc)) from exc
            if candidate_project.to_dict() != current.to_dict():
                current = candidate_project
                revision += 1
                screenings.invalidate((session_id, project_id, revision))
                jobs.invalidate(current)
            return snapshot()

    @app.post("/api/models/{name}/reduce", status_code=202)
    def reduce_model(name: str, edit: ReducedModelEdit) -> dict[str, Any]:
        nonlocal current, revision
        from easy_glm.workflow.reduction import reduced_challenger

        with lock:
            check_revision(edit)
            try:
                source = jobs.artifact(current, name)
                if source.name != edit.fit_id:
                    raise ValueError(
                        "The original fit changed. Refresh variable importance first."
                    )
                candidate_project = reduced_challenger(
                    current, name, edit.name, edit.predictors
                )
                # Queue before publishing the new definition: a busy fit leaves no orphan model.
                job = jobs.start(deepcopy(candidate_project), raw.clone(), edit.name)
                current = candidate_project
                revision += 1
                screenings.invalidate((session_id, project_id, revision))
                return {
                    "snapshot": snapshot(),
                    "name": edit.name,
                    "source": name,
                    "job": job,
                }
            except (ValueError, KeyError) as exc:
                raise HTTPException(409, str(exc)) from exc

    @app.post("/api/split")
    def save_split(edit: SplitEdit) -> dict[str, Any]:
        nonlocal current, revision
        with lock:
            check_revision(edit)
            candidate_project = deepcopy(current)
            for key, value in edit.model_dump(
                exclude={"session_id", "revision"}
            ).items():
                setattr(candidate_project.data.split, key, value)
            for column, role in list(candidate_project.data.roles.items()):
                if role == "split" and (edit.mode == "random" or column != edit.column):
                    candidate_project.data.roles.pop(column)
            if edit.mode == "column":
                candidate_project.apply_role_change(edit.column, "split")
        # Preparation is outside the project lock. The revision is rechecked on commit.
        try:
            split_counts(candidate_project, raw)
        except (ValueError, TypeError, pl.exceptions.PolarsError) as exc:
            raise HTTPException(422, str(exc)) from exc
        info = setup_info(candidate_project, raw)
        if info["problems"]:
            raise HTTPException(422, "; ".join(info["problems"]))
        with lock:
            check_revision(edit)
            if candidate_project.to_dict() != current.to_dict():
                current = candidate_project
                revision += 1
                screenings.invalidate((session_id, project_id, revision))
                jobs.invalidate(current)
            return snapshot()

    @app.post("/api/models/{name}/fit", status_code=202)
    def start_fit(name: str, edit: Revision) -> dict[str, Any]:
        with lock:
            check_revision(edit)
            if name not in current.models:
                raise HTTPException(404, "Save a model definition before fitting.")
            problems = current.validate(name)
            if problems:
                raise HTTPException(422, "; ".join(problems))
            try:
                return jobs.start(deepcopy(current), raw.clone(), name)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc

    @app.post("/api/models/{name}/cancel")
    def cancel_fit(name: str, edit: Revision) -> dict[str, Any]:
        with lock:
            check_revision(edit)
            jobs.cancel(name)
            return jobs.status(current)

    @app.get("/api/jobs")
    def job_status() -> dict[str, Any]:
        with lock:
            return jobs.status(current)

    @app.get("/api/results/{name}")
    def results(name: str) -> dict[str, Any]:
        with lock:
            try:
                result = jobs.result(current, name)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            return {
                **{k: v for k, v in result.items() if k != "tables"},
                "table_index": [
                    {"name": key, "rows": len(table["rows"])}
                    for key, table in result["tables"].items()
                ],
            }

    @app.get("/api/results/{name}/table")
    def result_table(
        name: str, variable: str, offset: int = 0, limit: int = 200
    ) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 500:
            raise HTTPException(
                422, "Use a nonnegative offset and a page of 1–500 rows."
            )
        with lock:
            try:
                result = jobs.result(current, name)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            if variable not in result["tables"]:
                raise HTTPException(404, "Unknown fitted variable.")
            table = result["tables"][variable]
            return {
                "columns": table["columns"],
                "rows": table["rows"][offset : offset + limit],
                "total": len(table["rows"]),
                "offset": offset,
            }

    @app.post("/api/review/{name}", status_code=202)
    def review_start(name: str, edit: ReviewEdit) -> dict[str, Any]:
        nonlocal revision
        from easy_glm.workflow.project import TableSnapshot

        with lock:
            check_revision(edit)
            try:
                source = jobs.artifact(current, name)
                request = edit.model_dump(exclude={"session_id", "revision"})
                saved = deepcopy(current)
                cfg = saved.models[name]
                if edit.action == "champion":
                    current.champion = name
                    revision += 1
                    screenings.invalidate((session_id, project_id, revision))
                    return {"snapshot": snapshot()}
                from easy_glm.desktop.importance_cache import is_importance

                if is_importance(request):
                    request["challenger"] = None
                if edit.challenger and not is_importance(request):
                    if edit.challenger == name:
                        raise ValueError("Choose a different challenger.")
                    other_source = jobs.artifact(current, edit.challenger)
                    request["_challenger_source"] = str(other_source)
                    request["challenger_fit_id"] = other_source.name
                if edit.action in ("include_factor", "include_factors", "include_pair"):
                    from easy_glm.workflow.project import Interaction

                    if edit.action in ("include_factor", "include_factors"):
                        eligible = jobs.result(current, name)["review_variables"]
                        variables = (
                            edit.variables
                            if edit.action == "include_factors"
                            else [edit.variable]
                        )
                        if not variables or len(set(variables)) != len(variables):
                            raise ValueError("Select distinct missing factors.")
                        for variable in variables:
                            if (
                                variable not in eligible
                                or saved.data.roles.get(variable)
                                not in (None, "unassigned", "predictor")
                                or variable == saved.data.split.column
                                or variable in cfg.predictors
                            ):
                                raise ValueError(
                                    "Choose eligible missing factors not already included."
                                )
                            saved.apply_role_change(variable, "predictor")
                            cfg.predictors.append(variable)
                    else:
                        if (
                            edit.a == edit.b
                            or edit.a not in cfg.predictors
                            or edit.b not in cfg.predictors
                        ):
                            raise ValueError(
                                "Choose two distinct predictors in this model."
                            )
                        if any(
                            {i.a, i.b} == {edit.a, edit.b} for i in cfg.interactions
                        ):
                            raise ValueError("This interaction is already included.")
                        cfg.interactions.append(Interaction(edit.a, edit.b))
                    problems = saved.validate(name)
                    if problems:
                        raise ValueError("; ".join(problems))
                    current.data = saved.data
                    current.models = saved.models
                    revision += 1
                    screenings.invalidate((session_id, project_id, revision))
                    jobs.invalidate(current)
                    return {"snapshot": snapshot()}
                if edit.action == "delete_snapshot":
                    if not edit.options.get("confirmed"):
                        raise ValueError(
                            "Confirm deleting this named snapshot. Undo cannot restore it."
                        )
                    if not any(s.name == edit.snapshot for s in cfg.snapshots):
                        raise ValueError("Choose an existing snapshot.")
                    cfg.snapshots = [
                        s for s in cfg.snapshots if s.name != edit.snapshot
                    ]
                    current.models[name].snapshots = cfg.snapshots
                    revision += 1
                    screenings.invalidate((session_id, project_id, revision))
                    jobs.edited(current, name, jobs.jobs[name]["result"])
                    return {"snapshot": snapshot()}
                if edit.action == "snapshot":
                    label = edit.snapshot.strip()
                    if (
                        not label
                        or label in ("__fitted__", "__current__")
                        or any(s.name == label for s in cfg.snapshots)
                    ):
                        raise ValueError("Give this snapshot a new, non-empty name.")
                    cfg.snapshots.append(
                        TableSnapshot(
                            label,
                            adjustments=deepcopy(cfg.adjustments),
                            base_rate_override=cfg.base_rate_override,
                        )
                    )
                    current.models[name].snapshots = cfg.snapshots
                    revision += 1
                    screenings.invalidate((session_id, project_id, revision))
                    jobs.edited(current, name, jobs.jobs[name]["result"])
                    return {"snapshot": snapshot()}
                if edit.action in (
                    "undo",
                    "redo",
                    "restore_snapshot",
                    "reset",
                    "reset_variable",
                ):
                    if edit.action in ("undo", "redo"):
                        stack = (
                            undo_steps if edit.action == "undo" else redo_steps
                        ).get(name, [])
                        if not stack:
                            raise ValueError("No table edit to " + edit.action + ".")
                        restore = stack[-1]
                        cfg.adjustments = deepcopy(restore[0])
                        cfg.base_rate_override = restore[1]
                    elif edit.action == "restore_snapshot":
                        matched = next(
                            (s for s in cfg.snapshots if s.name == edit.snapshot), None
                        )
                        if matched is None:
                            raise ValueError("Choose an existing snapshot.")
                        cfg.adjustments = deepcopy(matched.adjustments)
                        cfg.base_rate_override = matched.base_rate_override
                    elif edit.action == "reset_variable":
                        if edit.variable not in jobs.result(current, name)["tables"]:
                            raise ValueError("Choose a fitted table to reset.")
                        cfg.adjustments = [
                            a for a in cfg.adjustments if a.variable != edit.variable
                        ]
                    else:
                        cfg.adjustments = []
                        cfg.base_rate_override = None
                    request.update(action="restore", restore_project=saved.to_dict())
                request["original_action"] = edit.action
                request["model"] = name
                request["fit_id"] = source.name
                return reviews.start(deepcopy(current), source, request, revision)
            except (ValueError, KeyError) as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.get("/api/reviews/{key}")
    def review_status(key: str) -> dict[str, Any]:
        with lock:
            try:
                task = reviews.get(key)
            except ValueError as exc:
                raise HTTPException(404, str(exc)) from exc
            stale = (
                task["revision"] != revision
                or jobs.jobs.get(task["request"]["model"], {}).get("id")
                != task["request"]["fit_id"]
            )
            from easy_glm.desktop.importance_cache import is_importance

            if is_importance(task["request"]):
                try:
                    stale = (
                        jobs.artifact(current, task["request"]["model"]).name
                        != task["request"]["fit_id"]
                    )
                except ValueError:
                    stale = True
            if task["request"].get("challenger"):
                stale = stale or jobs.jobs.get(task["request"]["challenger"], {}).get(
                    "id"
                ) != task["request"].get("challenger_fit_id")
            data = task.get("data", {})
            return {
                "id": key,
                "status": "stale" if stale else task["status"],
                "data": {
                    k: v for k, v in data.items() if k not in ("project", "result")
                },
                "can_apply": not stale
                and "project" in data
                and data.get("changed", True)
                and task["status"] == "complete",
            }

    @app.post("/api/reviews/{key}/cancel")
    def review_cancel(key: str, edit: Revision) -> dict[str, Any]:
        with lock:
            check_revision(edit)
            try:
                reviews.cancel(key)
            except ValueError as exc:
                raise HTTPException(404, str(exc)) from exc
            return {"cancelled": True}

    @app.post("/api/reviews/{key}/apply")
    def review_apply(key: str, edit: Revision) -> dict[str, Any]:
        nonlocal current, revision
        with lock:
            check_revision(edit)
            try:
                task = reviews.get(key)
                if (
                    task["revision"] != revision
                    or jobs.jobs.get(task["request"]["model"], {}).get("id")
                    != task["request"]["fit_id"]
                    or (
                        task["request"].get("challenger")
                        and jobs.jobs.get(task["request"]["challenger"], {}).get("id")
                        != task["request"].get("challenger_fit_id")
                    )
                    or task["status"] != "complete"
                    or "project" not in task.get("data", {})
                    or not task.get("data", {}).get("changed", True)
                ):
                    raise ValueError(
                        "Preview is no longer applicable. Preview the edit again."
                    )
                name = task["request"]["model"]
                jobs.result(current, name)
                cfg = current.models[name]
                candidate_project = Project.from_dict(task["data"]["project"])
                candidate_config = candidate_project.models[name]
                candidate_result = task["data"]["result"]
                before = (deepcopy(cfg.adjustments), cfg.base_rate_override)
                action = task["request"]["original_action"]
                if action == "undo":
                    undo_steps[name].pop()
                    redo_steps.setdefault(name, []).append(before)
                else:
                    undo_steps.setdefault(name, []).append(before)
                    undo_steps[name] = undo_steps[name][-50:]
                    if action == "redo":
                        redo_steps[name].pop()
                    else:
                        redo_steps[name] = []
                cfg.adjustments = candidate_config.adjustments
                cfg.base_rate_override = candidate_config.base_rate_override
                revision += 1
                screenings.invalidate((session_id, project_id, revision))
                jobs.edited(current, name, candidate_result)
                return snapshot()
            except (ValueError, KeyError) as exc:
                raise HTTPException(409, str(exc)) from exc

    @app.get("/api/review-info/{name}")
    def review_info(name: str) -> dict[str, Any]:
        with lock:
            if name not in current.models:
                raise HTTPException(404, "Unknown model")
            try:
                columns = jobs.result(current, name)["review_variables"]
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            return {
                "variables": columns,
                "predictors": current.models[name].predictors,
                "interactions": [i.__dict__ for i in current.models[name].interactions],
                "model_names": list(current.models),
                "variable_info": jobs.result(current, name)
                .get("diagnostic_info", {})
                .get("variables", []),
                "snapshots": [s.name for s in current.models[name].snapshots],
                "adjustments": [a.__dict__ for a in current.models[name].adjustments],
                "base_rate_override": current.models[name].base_rate_override,
                "link": jobs.result(current, name)["link"],
                "undo": bool(undo_steps.get(name)),
                "redo": bool(redo_steps.get(name)),
            }

    @app.post("/api/exports/{name}")
    def export_model(name: str, edit: ExportRequest) -> Response:
        with lock:
            check_revision(edit)
            if name not in current.models:
                raise HTTPException(404, "Choose an existing model.")
            if edit.challenger is not None:
                if edit.format != "html":
                    raise HTTPException(
                        422, "A comparison is available in the HTML report only."
                    )
                if edit.challenger == name:
                    raise HTTPException(422, "Choose a different comparison model.")
                if edit.challenger not in current.models:
                    raise HTTPException(404, "Choose an existing comparison model.")
            names = [name] + ([edit.challenger] if edit.challenger else [])
            try:
                sources = {model: jobs.artifact(current, model) for model in names}
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            saved, saved_raw = deepcopy(current), raw.clone()
        try:
            attachment = export_attachment(
                saved, saved_raw, sources, name, edit.format, edit.challenger
            )
        except Exception as exc:  # Export errors are returned to the UI.
            raise HTTPException(422, f"Could not export this model: {exc}") from exc
        return Response(
            attachment.content,
            media_type=attachment.media_type,
            headers={
                "Content-Disposition": attachment.disposition,
                "Cache-Control": "no-store",
            },
        )

    @app.get("/api/project")
    def export_project() -> dict[str, Any]:
        with lock:
            return current.to_dict()

    @app.get("/api/explore")
    def explore(
        column: str | None = None,
        model: str | None = None,
        n_bins: int = Query(default=20, ge=5, le=50),
    ) -> dict[str, Any]:
        with lock:
            saved, saved_raw = deepcopy(current), raw.clone()
            generation = session_id, project_id, revision
            cache = exploration
        try:
            result = cache.view(
                saved, saved_raw, generation, column=column, model=model, n_bins=n_bins
            )
        except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError) as exc:
            raise HTTPException(422, f"Cannot explore these data: {exc}") from exc
        with lock:
            if generation != (session_id, project_id, revision):
                raise HTTPException(
                    409, "Applied settings changed. Refresh this chart."
                )
        return result

    @app.get("/api/plot")
    def plot(column: str) -> dict[str, Any]:
        # FastAPI executes this synchronous route in its worker thread pool.
        # Copy briefly under lock; aggregation does not block edits or health.
        with lock:
            saved, plot_revision = deepcopy(current), revision
            plot_raw = raw.clone()
        if column not in plot_raw.columns:
            raise HTTPException(404, "Unknown source column.")
        try:
            frame = plot_raw.head(50_000)
            prepared = apply_variables(frame, saved.data)
            final = saved.data.renames.get(column, column)
            result = univariate(prepared, final, n_bins=16, max_levels=25)
            table = result["table"].select("label", "exposure").to_dicts()
            return {
                "column": final,
                "revision": plot_revision,
                "rows": prepared.height,
                "sampled": plot_raw.height > 50_000,
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
