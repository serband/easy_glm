"""Loopback API for the experimental Svelte Variables workbench.

The project is an in-memory copy. Fits run in isolated child processes.
No endpoint overwrites source files or existing Streamlit fit caches.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import secrets
import shutil
import threading
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from easy_glm.desktop.binning import (
    apply_binning_setup,
    binning_changes,
    binning_columns,
    binning_preview,
    binning_setup,
    validate_changed_linear_binning,
)
from easy_glm.desktop.exploration import ExplorationCache
from easy_glm.desktop.exports import ExportRequest, export_attachment
from easy_glm.desktop.feature_selection import (
    FeatureSelectionJobs,
    FeatureSelectionRequest,
)
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
from easy_glm.desktop.model_plan import (
    ModelPlanEdit,
    PlanFitRef,
    materialise_plan_prefix,
    plan_model_name,
    validate_model_plan,
)
from easy_glm.desktop.modeling import (
    FitRequest,
    ModelEdit,
    ReducedModelEdit,
    Revision,
    SplitEdit,
    edit_model,
    pair_candidate_groups,
    setup_info,
    stack_factor_stages,
    validate_stack_factor_scope,
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
from easy_glm.workflow.model_plan import ModelPlan, new_plan_id
from easy_glm.workflow.prep import apply_variables
from easy_glm.workflow.project import (
    SINGLE_ROLES,
    ModelConfig,
    ModelLineage,
    Project,
    pair_stage_fingerprint,
)
from easy_glm.workflow.stack_types import ComponentRecipe
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


class BinningPreviewEdit(Edit):
    column: str


class StackAmendEdit(Revision):
    operation: Literal["cap", "round"]
    lower: float | None = None
    upper: float | None = None
    decimals: int | None = Field(default=None, ge=0, le=12)
    refit_later: bool = False
    new_name: str | None = None
    fit_id: str | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    feature_selections = FeatureSelectionJobs()
    undo_steps = {}
    redo_steps = {}
    catalogue_sources: dict[str, str] = {}
    # Schema collection avoids loading a master file just to restore the Variables
    # page after a project is reopened.  The pinned source/hash remains part of the
    # stage recipe; this cache is only presentation metadata.
    catalogue_schema_cache: dict[tuple[str, str], list[dict[str, str]]] = {}

    def catalogue_descriptor(stage: ComponentRecipe) -> dict[str, Any] | None:
        """Return saved catalogue columns for the stage editor and later stages."""
        source_path = stage.settings.get("source_path")
        source_hash = stage.settings.get("source_sha256")
        if not isinstance(source_path, str) or not isinstance(source_hash, str):
            return None
        source = Path(source_path).resolve()
        canonical = str(source)
        if catalogue_sources.get(canonical) != source_hash or not source.is_file():
            return None
        cache_key = (canonical, source_hash)
        columns = catalogue_schema_cache.get(cache_key)
        if columns is None:
            try:
                if source.suffix.lower() == ".parquet":
                    schema = pl.scan_parquet(source).collect_schema()
                elif source.suffix.lower() == ".csv":
                    schema = pl.scan_csv(source, infer_schema_length=0).collect_schema()
                else:
                    return None
            except (OSError, pl.exceptions.PolarsError):
                return None
            columns = [
                {"name": name, "dtype": str(dtype)} for name, dtype in schema.items()
            ]
            catalogue_schema_cache[cache_key] = columns
        # ``rows`` is deliberately unknown after restart: counting a CSV would
        # scan the full master file.  The editor only needs the durable schema
        # to choose its key and attributes without asking for another upload.
        return {
            "filename": source.name,
            "rows": None,
            "key_columns": columns,
            "attribute_columns": columns,
            "columns": columns,
        }

    def register_catalogue_sources(saved_project: Project) -> None:
        """Restore only pinned catalogue files whose saved digest still matches."""
        components = [
            component
            for config in saved_project.models.values()
            for component in getattr(config, "components", ())
        ]
        if saved_project.model_plan is not None:
            components.extend(saved_project.model_plan.stages)
        for component in components:
            if component.kind != "catalogue":
                continue
            source_path = component.settings.get("source_path")
            source_hash = component.settings.get("source_sha256")
            if not isinstance(source_path, str) or not isinstance(source_hash, str):
                continue
            source = Path(source_path).resolve()
            try:
                if source.is_file() and _file_sha256(source) == source_hash:
                    catalogue_sources[str(source)] = source_hash
            except OSError:
                continue

    register_catalogue_sources(project)

    def validate_plan_catalogues(
        stages: list[ComponentRecipe], *, require_source: bool
    ) -> None:
        """Allow empty planned stages; require unchanged pinned uploads for fitting."""
        for stage in stages:
            if stage.kind != "catalogue" or stage.settings.get("catalogue") is not None:
                continue
            source_path = stage.settings.get("source_path")
            source_hash = stage.settings.get("source_sha256")
            if source_path is None and source_hash is None and not require_source:
                continue
            if not isinstance(source_path, str) or not isinstance(source_hash, str):
                raise ValueError(
                    f"{stage.name}: upload a catalogue in Project & data before fitting."
                )
            canonical = str(Path(source_path).resolve())
            if catalogue_sources.get(canonical) != source_hash:
                raise ValueError(
                    "Catalogue source is not a pinned upload in this session."
                )
            if _file_sha256(Path(canonical)) != source_hash:
                raise ValueError("Catalogue contents changed since upload.")

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
        feature_selections.close()
        reviews.close()
        jobs.close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    current = deepcopy(project)
    revision = 0
    lock = threading.RLock()
    # These references are server-issued when a stage is queued.  They are
    # intentionally never reconstructed by comparing old saved recipes: a new
    # chain with identical settings must start cleanly.
    plan_attempts: dict[tuple[str, str], PlanFitRef] = {}
    plan_refs: list[PlanFitRef] = [
        PlanFitRef(
            current.model_plan.plan_id,
            item["stage_id"],
            item["model_name"],
            item["fit_id"],
        )
        for item in (current.model_plan.fit_refs if current.model_plan else [])
    ]

    def persist_plan_refs() -> None:
        if current.model_plan is not None:
            current.model_plan.fit_refs = [
                {
                    "stage_id": item.stage_id,
                    "model_name": item.model_name,
                    "fit_id": item.fit_id,
                }
                for item in plan_refs
                if item.plan_id == current.model_plan.plan_id
            ]

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
            cleared = False
            if not cfg.pair_stages:
                cleared = bool(cfg.adjustments) or cfg.base_rate_override is not None
                cfg.adjustments = []
                cfg.base_rate_override = None
                undo_steps.pop(name, None)
                redo_steps.pop(name, None)
            else:
                # Undo states captured against older fitted axes/provenance
                # cannot safely cross an explicit staged refit.
                clear = set(job.get("clear_stage_ids", []))
                if clear:
                    kept = [
                        adjustment
                        for adjustment in cfg.adjustments
                        if adjustment.stage_id not in clear
                    ]
                    cleared = len(kept) != len(cfg.adjustments)
                    cfg.adjustments = kept
                undo_steps.pop(name, None)
                redo_steps.pop(name, None)
            if cleared:
                revision += 1
                screenings.invalidate((session_id, project_id, revision))
                feature_selections.invalidate((session_id, project_id, revision))
            job.update(
                key=model_key(current, name),
                project=deepcopy(current),
                status="complete",
                message=(
                    "Unchanged main effects reused. Diagnostics and rate tables are ready."
                    if result.get("reuse", {}).get("main_effects")
                    else "Diagnostics and rate tables are ready."
                ),
                result=result,
            )
            jobs.remember_complete(name)

    jobs.on_complete = complete_fit

    def snapshot_pair_identity(name: str, source: Path) -> str | None:
        cfg = current.models[name]
        if not cfg.pair_stages:
            return None
        with (source / "fit.pkl").open("rb") as handle:
            run = pickle.load(handle)  # private fit artifact created by this app
        return pair_stage_fingerprint(cfg, run.rate_model)

    def scorer_fingerprint(stack: Any) -> str:
        """Hash the deployed score, excluding inspection and version-history metadata."""
        payload = stack.to_dict()
        payload.pop("virtual_components", None)

        def drop_non_scoring_metadata(value: Any) -> None:
            if isinstance(value, dict):
                # RateModel snapshots record creation timestamps.  They are useful
                # for inspection but do not change the current scorer or offset.
                value.pop("snapshots", None)
                value.pop("current_version", None)
                for child in value.values():
                    drop_non_scoring_metadata(child)
            elif isinstance(value, list):
                for child in value:
                    drop_non_scoring_metadata(child)

        drop_non_scoring_metadata(payload)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

    def fitted_prefix(name: str) -> Any:
        """Read an exact applicable source scorer for a guarded append."""
        from easy_glm.workflow.stack import stack_from_legacy

        source = jobs.artifact(current, name)
        with (source / "fit.pkl").open("rb") as handle:
            fitted = pickle.load(handle)  # private fit artifact
        if fitted.name != name:
            raise ValueError("The source fit identity does not match its model.")
        # A legacy artifact stores the raw fit.  Its accepted table amendments
        # and pair tables live in the saved configuration, so compile that
        # reviewed scorer before using it as any downstream offset.
        if fitted.stack is None:
            from easy_glm.workflow.run import rate_model_for

            fitted = deepcopy(fitted)
            fitted.rate_model = rate_model_for(current, fitted)
        return stack_from_legacy(current, fitted)

    def plan_ref_state(
        plan: ModelPlan | None = None,
    ) -> tuple[list[dict[str, Any]], PlanFitRef | None]:
        """Return the completed contiguous prefix and an optional live next job."""
        plan = current.model_plan if plan is None else plan
        if plan is None:
            return [], None
        statuses = jobs.status(current)
        completed: list[dict[str, Any]] = []
        pending: PlanFitRef | None = None
        for index, stage in enumerate(plan.stages):
            ref = next(
                (
                    item
                    for item in reversed(plan_refs)
                    if item.plan_id == plan.plan_id
                    and item.stage_id == stage.component_id
                ),
                None,
            )
            if ref is None:
                break
            job = statuses.get(ref.model_name, {})
            if job.get("id") != ref.fit_id:
                break
            expected = materialise_plan_prefix(current, plan, index + 1, None)
            actual = current.models.get(ref.model_name)
            if actual is None or (
                actual != expected and replace(actual, lineage=None) != expected
            ):
                break
            if job.get("status") in {"queued", "running"}:
                pending = ref
                break
            if not job.get("applicable"):
                break
            completed.append(
                {
                    "stage_id": stage.component_id,
                    "stage_name": stage.name,
                    "model_name": ref.model_name,
                    "fit_id": ref.fit_id,
                }
            )
        return completed, pending

    def plan_view() -> dict[str, Any]:
        """Server-authoritative state for the project-level model workflow."""
        plan = current.model_plan
        prepared_columns: set[str] = set()
        prepared_schema: dict[str, Any] = {}
        try:
            prepared = apply_variables(raw, current.data)
            prepared_columns = set(prepared.columns)
            prepared_schema = {
                name: str(dtype) for name, dtype in prepared.schema.items()
            }
        except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError):
            pass
        safe_columns: list[list[str]] = []
        if plan is not None:
            config = ModelConfig(
                family=plan.family,
                link=plan.link,
                target=current.target,
                weight=current.weight,
                offset=current.offset_column,
                divide_target_by_weight=plan.divide_target_by_weight,
                tweedie_power=plan.tweedie_power,
                components=deepcopy(plan.stages),
            )
            try:
                safe_columns = stack_factor_stages(
                    current, raw, prepared_columns, config
                )
            except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError):
                safe_columns = [[] for _stage in plan.stages]
        refs, pending = plan_ref_state(plan)
        ref_by_stage = {item["stage_id"]: item for item in refs}
        statuses: list[dict[str, Any]] = []
        next_stage_id: str | None = None
        for stage in plan.stages if plan else []:
            ref = ref_by_stage.get(stage.component_id)
            if ref is not None:
                state, message, fit = "fitted", "Fitted", ref
            elif pending is not None and pending.stage_id == stage.component_id:
                state, message, fit = (
                    "fitting",
                    "Fit in progress",
                    {
                        "stage_id": stage.component_id,
                        "model_name": pending.model_name,
                        "fit_id": pending.fit_id,
                    },
                )
            elif next_stage_id is None and pending is None:
                state, message, fit = "ready", "Ready to fit", None
                next_stage_id = stage.component_id
            else:
                state, message, fit = "waiting", "Fit the earlier stage first", None
            attempt_ref = (
                plan_attempts.get((plan.plan_id, stage.component_id)) if plan else None
            )
            attempt_job = (
                jobs.status(current).get(attempt_ref.model_name, {})
                if attempt_ref
                else {}
            )
            last_attempt = (
                {
                    "model_name": attempt_ref.model_name,
                    "fit_id": attempt_ref.fit_id,
                    "status": attempt_job.get("status"),
                    "message": attempt_job.get("message"),
                }
                if attempt_ref
                else None
            )
            statuses.append(
                {
                    "stage_id": stage.component_id,
                    "name": stage.name,
                    "kind": stage.kind,
                    "state": state,
                    "message": message,
                    "fit": fit,
                    "last_attempt": last_attempt,
                }
            )
        history = [
            {
                "name": name,
                "components": [item.name for item in config.components],
                "fit_id": job.get("id"),
                "status": job.get("status"),
                "can_adopt": config.family in {"poisson", "tweedie", "gaussian"}
                and config.link in {None, "log"},
            }
            for name, config in current.models.items()
            if (job := jobs.status(current).get(name, {})).get("applicable")
        ]
        catalogue_descriptors: dict[str, dict[str, Any]] = {}
        schema_at_stage = dict(prepared_schema)
        safe_column_details = []
        for index, columns in enumerate(safe_columns):
            safe_column_details.append(
                [
                    {
                        "name": name,
                        "dtype": schema_at_stage.get(name),
                        "role": current.data.roles.get(name, "unassigned"),
                    }
                    for name in columns
                ]
            )
            if plan is None or index >= len(plan.stages):
                continue
            stage = plan.stages[index]
            if stage.kind != "catalogue":
                continue
            descriptor = catalogue_descriptor(stage)
            if descriptor is None:
                continue
            catalogue_descriptors[stage.component_id] = descriptor
            dtype_by_name = {
                item["name"]: item["dtype"] for item in descriptor["columns"]
            }
            for attribute in stage.factors:
                if attribute in dtype_by_name:
                    schema_at_stage[attribute] = dtype_by_name[attribute]
        return {
            "plan": plan.to_dict() if plan is not None else None,
            "fit_refs": refs,
            "stage_statuses": statuses,
            "safe_columns": safe_columns,
            "safe_column_details": safe_column_details,
            "catalogue_descriptors": catalogue_descriptors,
            "project_roles": dict(current.data.roles),
            "design_defaults": current.to_dict()["design"]["defaults"],
            "next_stage_id": next_stage_id,
            "history": history,
        }

    def verified_append_source(name: str) -> tuple[str, Path] | None:
        """Return an exact saved parent scorer for the candidate's final addition.

        Lineage is durable project data, whereas ``append_sources`` used to be an
        in-memory convenience map.  Deriving the guard from lineage means that a
        saved draft can be reopened, amended and fitted as a suffix without
        accidentally dropping its frozen parent association.
        """
        lineage = current.models[name].lineage
        if lineage is None:
            return None
        source_name = lineage.parent_name
        source_path = jobs.artifact(current, source_name)
        if (
            source_path.name != lineage.parent_fit_id
            or model_key(current, source_name) != lineage.parent_key
        ):
            raise ValueError("The recorded starting fit is no longer available.")
        prefix = fitted_prefix(source_name)
        if (
            lineage.parent_name != source_name
            or lineage.parent_fit_id != source_path.name
            or lineage.parent_key != model_key(current, source_name)
            or lineage.parent_prefix_fingerprint != scorer_fingerprint(prefix)
        ):
            raise ValueError("The recorded starting checkpoint is no longer available.")
        return source_name, source_path

    def parent_fit_is_available(name: str) -> bool:
        """Whether this child can still revise its last addition against its parent."""
        if current.models[name].lineage is None:
            return True
        try:
            return verified_append_source(name) is not None
        except (KeyError, OSError, ValueError):
            return False

    def model_flow(
        saved: Project, saved_jobs: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Read-only durable provenance; it never supplies a scorer to a fit."""
        nodes = []
        edges = []
        for name, config in saved.models.items():
            job = saved_jobs.get(name, {})
            own_fit_available = bool(job.get("applicable"))
            missing_own_fit = bool(
                job.get("status") == "complete"
                and not job.get("artifact_available", False)
            )
            parent_fit_available = parent_fit_is_available(name)
            nodes.append(
                {
                    "name": name,
                    "kind": "stack" if config.components else "legacy",
                    "status": (
                        "missing_fit"
                        if missing_own_fit
                        else job.get("status", "not_fitted")
                    ),
                    # ``fitted`` remains for older clients.  The explicit flags
                    # stop one historical incoming edge from making a complete
                    # child scorer look unusable.
                    "fitted": own_fit_available,
                    "own_fit_available": own_fit_available,
                    "missing_own_fit": missing_own_fit,
                    "append_available": own_fit_available,
                    "parent_fit_available": parent_fit_available,
                    "revise_available": config.lineage is None or parent_fit_available,
                    "fit_id": job.get("id") if own_fit_available else None,
                    "component_count": len(config.components),
                    "full_refit_required": bool(
                        missing_own_fit
                        or (
                            not own_fit_available
                            and config.lineage is not None
                            and not parent_fit_available
                        )
                    ),
                }
            )
            lineage = config.lineage
            if lineage is None:
                continue
            parent_job = saved_jobs.get(lineage.parent_name)
            if job.get("full_refit"):
                # Its recipe was rebuilt independently; provenance remains a
                # navigational record but must not imply scorer reuse.
                edge_status = "historical"
            elif lineage.parent_name not in saved.models:
                edge_status = "missing"
            elif not parent_job or not parent_job.get("applicable"):
                edge_status = "historical"
            elif (
                parent_job.get("id") == lineage.parent_fit_id
                and model_key(saved, lineage.parent_name) == lineage.parent_key
            ):
                edge_status = "live"
            else:
                edge_status = "historical"
            edges.append(
                {
                    "from": lineage.parent_name,
                    "to": name,
                    "kind": "builds_on",
                    "parent_fit_id": lineage.parent_fit_id,
                    "prefix_length": lineage.prefix_length,
                    "edge_status": edge_status,
                    "historical": edge_status != "live",
                }
            )
        return {"nodes": nodes, "edges": edges}

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

    def complete_feature_selection(
        task: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """Keep the latest current feature-selection recipe for Python export.

        This is audit/export metadata, so it neither changes the applied roles
        nor advances the revision or invalidates any fitted model.
        """
        with lock, feature_selections.lock:
            generation = (session_id, project_id, revision)
            if (
                feature_selections.tasks.get(task["id"]) is not task
                or task["generation"] != generation
                or task["status"] != "completing"
            ):
                return
            screened_project = deepcopy(task["recipe_project"])
            screened_project["models"] = {}
            screened_project["champion"] = None
            screened_project["exploration"] = {}
            from easy_glm.workflow.selection_recipe import resolved_selection_options

            current.exploration["feature_selection"] = {
                "version": 1,
                "project": screened_project,
                "options": resolved_selection_options(task["recipe_options"], result),
                "result": deepcopy(result),
            }
            task.update(
                status="complete",
                message="Feature selection complete.",
                result=deepcopy(result),
            )

    feature_selections.on_complete = complete_feature_selection
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
            upload = request.url.path in {
                "/api/project/upload",
                "/api/catalogues/upload",
            }
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
                "binning": binning_setup(current, raw.columns),
            },
            "binning_columns": binning_columns(current, raw),
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
        edit: Edit | ScreeningRequest | FeatureSelectionRequest,
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
        has_binning = "binning" in setup
        requested_binning = setup.pop("binning", None)
        rows, errors = parse_variable_setup_json(
            current, raw.columns, json.dumps(setup)
        )
        if errors:
            raise HTTPException(422, errors)
        result = deepcopy(current)
        _, notices = apply_roles_grid(result, raw.columns, rows)
        try:
            if has_binning:
                apply_binning_setup(result, raw.columns, requested_binning)
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
        bin_changes = binning_changes(current, result, raw.columns)
        if bin_changes:
            try:
                validate_changed_linear_binning(current, result, raw)
            except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError) as exc:
                raise HTTPException(422, str(exc)) from exc
        changes.extend(bin_changes)
        if bin_changes:
            fitted = [
                name
                for name, job in jobs.jobs.items()
                if job.get("status") == "complete"
            ]
            if fitted:
                notices.append(
                    (
                        "warning",
                        "Applying binning will mark fitted models stale: "
                        + ", ".join(sorted(fitted))
                        + ". Fit them again to refresh rate tables.",
                    )
                )
            affected = [
                name
                for name, cfg in result.models.items()
                if cfg.adjustments or cfg.snapshots
            ]
            if affected:
                notices.append(
                    (
                        "warning",
                        "Models with saved adjustments or snapshots may refer to old bands: "
                        + ", ".join(sorted(affected))
                        + ". Review them after refitting.",
                    )
                )
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

    @app.post("/api/variables/feature-selection", status_code=202)
    def feature_selection_start(edit: FeatureSelectionRequest) -> dict[str, Any]:
        with lock:
            saved, _, _ = candidate(edit, validate_split=False)
            if not saved.target:
                raise HTTPException(
                    422, "Assign a target role before feature selection."
                )
            try:
                return feature_selections.start(
                    saved,
                    raw.clone(),
                    (session_id, project_id, revision),
                    edit.setup,
                    edit.options.model_dump(),
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.get("/api/feature-selections/{key}")
    def feature_selection_status(key: str) -> dict[str, Any]:
        with lock:
            try:
                return feature_selections.status(
                    key, (session_id, project_id, revision)
                )
            except KeyError as exc:
                raise HTTPException(404, str(exc.args[0])) from exc

    @app.post("/api/feature-selections/{key}/cancel")
    def feature_selection_cancel(key: str, edit: Revision) -> dict[str, Any]:
        with lock:
            check_revision(edit)
            try:
                feature_selections.cancel(key)
                return feature_selections.status(
                    key, (session_id, project_id, revision)
                )
            except KeyError as exc:
                raise HTTPException(404, str(exc.args[0])) from exc

    @app.post("/api/variables/preview")
    def preview(edit: Edit) -> dict[str, Any]:
        with lock:
            _, changes, notices = candidate(edit)
            return {"changes": changes, "notices": notices}

    @app.post("/api/variables/binning-preview")
    def variable_binning_preview(edit: BinningPreviewEdit) -> dict[str, Any]:
        with lock:
            saved, _, _ = candidate(edit, validate_split=False)
            frame = raw.clone()
        try:
            split_counts(saved, frame)
            return binning_preview(saved, frame, edit.column)
        except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/variables/apply")
    def apply(edit: Edit) -> dict[str, Any]:
        nonlocal current, revision
        with lock:
            result, changes, notices = candidate(edit)
            if changes:
                current = result
                revision += 1
                screenings.invalidate((session_id, project_id, revision))
                feature_selections.invalidate((session_id, project_id, revision))
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
        nonlocal current, raw, revision, session_id, project_id, token, jobs, reviews, exploration, plan_refs
        with lock:
            check_revision(edit)
            previous_jobs, previous_reviews = jobs, reviews
            jobs, reviews = FitJobs(), ReviewJobs()
            exploration = ExplorationCache()
            jobs.on_complete = complete_fit
            current, raw = loaded, loaded_raw
            plan_attempts.clear()
            plan_refs = [
                PlanFitRef(
                    loaded.model_plan.plan_id,
                    item["stage_id"],
                    item["model_name"],
                    item["fit_id"],
                )
                for item in (loaded.model_plan.fit_refs if loaded.model_plan else [])
            ]
            catalogue_sources.clear()
            catalogue_schema_cache.clear()
            register_catalogue_sources(loaded)
            revision = 0
            session_id = secrets.token_urlsafe(16)
            project_id = secrets.token_hex(32)
            token = secrets.token_urlsafe(32)
            screenings.invalidate((session_id, project_id, revision))
            feature_selections.invalidate((session_id, project_id, revision))
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

    @app.post("/api/catalogues/upload")
    async def upload_catalogue(
        request: Request,
        filename: str = Query(min_length=1, max_length=255),
        session_id: str = Query(min_length=1),
        revision: int = Query(ge=0),
    ) -> dict[str, Any]:
        """Pin a full master file under a server-issued path for a later recipe."""
        edit = Revision(session_id=session_id, revision=revision)
        with lock:
            check_revision(edit)
        suffix = Path(filename).suffix.lower()
        if suffix not in {".csv", ".parquet"}:
            raise HTTPException(422, "Choose a CSV or Parquet catalogue.")
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "The upload exceeds 512 MiB.")
        folder = new_input_folder()
        source = folder / upload_basename(filename)
        received = 0
        digest = hashlib.sha256()
        try:
            with source.open("xb") as destination:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "The upload exceeds 512 MiB.")
                    digest.update(chunk)
                    await run_in_threadpool(destination.write, chunk)
            if not received:
                raise ValueError("Choose a non-empty catalogue.")
            catalogue = await run_in_threadpool(
                pl.read_csv if suffix == ".csv" else pl.read_parquet,
                source,
                **({"infer_schema_length": 0} if suffix == ".csv" else {}),
            )
            if not catalogue.height or not catalogue.width:
                raise ValueError("The catalogue must contain rows and columns.")
            with lock:
                check_revision(edit)
                catalogue_sources[str(source.resolve())] = digest.hexdigest()
            return {
                "source_path": str(source.resolve()),
                "source_sha256": digest.hexdigest(),
                "filename": source.name,
                "rows": catalogue.height,
                "columns": [
                    {"name": name, "dtype": str(dtype)}
                    for name, dtype in catalogue.schema.items()
                ],
            }
        except HTTPException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(422, f"Could not read catalogue: {exc}") from exc

    @app.get("/api/workbench")
    def workbench() -> dict[str, Any]:
        with lock:
            saved, saved_raw = deepcopy(current), raw.clone()
            saved_revision, saved_session = revision, session_id
            saved_jobs = jobs.status(saved)
            saved_stage_statuses = jobs.stage_statuses(saved)
            saved_refit_previews = jobs.refit_previews(saved)
            saved_flow = model_flow(saved, saved_jobs)
            stack_sources = {}
            for source_name, job in saved_jobs.items():
                if not job.get("applicable"):
                    continue
                try:
                    stack = fitted_prefix(source_name)
                except (ValueError, KeyError, OSError):
                    continue
                stack_sources[source_name] = {
                    "components": [
                        item.recipe.to_dict()
                        for item in stack.components
                        if item.recipe.kind != "baseline"
                    ],
                    "source_fit_id": job["id"],
                }
        return {
            **setup_info(saved, saved_raw),
            "revision": saved_revision,
            "session_id": saved_session,
            "jobs": saved_jobs,
            "stage_statuses": saved_stage_statuses,
            "stage_refit_preview": saved_refit_previews,
            "stack_sources": stack_sources,
            "model_flow": saved_flow,
            "champion": saved.champion,
            "model_plan": plan_view(),
        }

    @app.get("/api/model-plan")
    def get_model_plan() -> dict[str, Any]:
        with lock:
            return {**plan_view(), "session_id": session_id, "revision": revision}

    @app.post("/api/model-plan")
    def save_model_plan(edit: ModelPlanEdit) -> dict[str, Any]:
        """Change the editable plan, never an immutable fitted model."""
        nonlocal current, revision, plan_refs
        with lock:
            check_revision(edit)
            if edit.plan is None and edit.operation != "adopt":
                raise HTTPException(422, "Choose a model structure to save.")
            try:
                if edit.operation == "adopt":
                    if not edit.adopt_model_name:
                        raise ValueError("Choose a saved model to adopt.")
                    status = jobs.status(current).get(edit.adopt_model_name, {})
                    source = current.models.get(edit.adopt_model_name)
                    if not status.get("applicable") or source is None:
                        raise ValueError(
                            "The saved model has no applicable fitted result."
                        )
                    stage_recipe = ComponentRecipe(
                        component_id="adopted-root",
                        name=edit.adopt_model_name,
                        kind="glm",
                        factors=tuple(source.predictors),
                        settings={
                            "alpha": source.penalty.alpha,
                            "cv": source.penalty.cv,
                            "n_alphas": source.penalty.n_alphas,
                            "l1_ratio": source.penalty.l1_ratio,
                            "min_alpha_ratio": source.penalty.min_alpha_ratio,
                            "base": source.base,
                        },
                        design=deepcopy(source.design),
                    )
                    plan = ModelPlan(
                        mode="sequential",
                        family=source.family,
                        link=source.link,
                        tweedie_power=source.tweedie_power,
                        divide_target_by_weight=source.divide_target_by_weight,
                        stages=[stage_recipe],
                        adopted_model_name=edit.adopt_model_name,
                    )
                else:
                    plan = ModelPlan.from_dict(edit.plan or {})
                # The browser cannot select a plan identity, references, or an
                # arbitrary adopted source.  Only the adopt operation creates it.
                old = current.model_plan
                if plan.adopted_model_name is not None and not (
                    edit.operation == "adopt"
                    or (
                        edit.operation in {"save", "revise"}
                        and old is not None
                        and plan.adopted_model_name == old.adopted_model_name
                    )
                ):
                    raise ValueError(
                        "Only Adopt saved model may choose an adopted source."
                    )
                if (
                    old is not None
                    and old.adopted_model_name is not None
                    and edit.operation == "save"
                    and plan.adopted_model_name != old.adopted_model_name
                ):
                    raise ValueError(
                        "The adopted scorer is frozen. Start a new independent chain to replace it."
                    )
                old_refs, pending = plan_ref_state(old)
                if edit.operation == "start":
                    plan.plan_id = new_plan_id()
                    plan_refs = []
                elif edit.operation == "revise":
                    if not old_refs:
                        raise ValueError("There is no fitted stage to revise.")
                    if pending is not None:
                        raise ValueError(
                            "A stage is fitting. Wait for it or start a new chain."
                        )
                    if (
                        old is not None
                        and old.adopted_model_name is not None
                        and (
                            not plan.stages
                            or not old.stages
                            or plan.stages[0] != old.stages[0]
                        )
                    ):
                        raise ValueError(
                            "The adopted scorer is frozen. Start a new independent chain to replace it."
                        )
                    root_same = old is not None and (
                        plan.mode,
                        plan.family,
                        plan.link,
                        plan.tweedie_power,
                        plan.divide_target_by_weight,
                    ) == (
                        old.mode,
                        old.family,
                        old.link,
                        old.tweedie_power,
                        old.divide_target_by_weight,
                    )
                    keep = 0
                    if root_same and old is not None:
                        for _ref in old_refs:
                            if (
                                keep >= len(plan.stages)
                                or plan.stages[keep] != old.stages[keep]
                            ):
                                break
                            keep += 1
                    plan.plan_id = new_plan_id()
                    plan_refs = [
                        PlanFitRef(
                            plan.plan_id,
                            ref["stage_id"],
                            ref["model_name"],
                            ref["fit_id"],
                        )
                        for ref in old_refs[:keep]
                    ]
                elif edit.operation == "adopt":
                    plan_refs = [
                        PlanFitRef(
                            plan.plan_id,
                            plan.stages[0].component_id,
                            plan.adopted_model_name or "",
                            status["id"],
                        )
                    ]
                else:
                    if old is not None:
                        plan.plan_id = old.plan_id
                        root_changed = (
                            plan.mode,
                            plan.family,
                            plan.link,
                            plan.tweedie_power,
                            plan.divide_target_by_weight,
                        ) != (
                            old.mode,
                            old.family,
                            old.link,
                            old.tweedie_power,
                            old.divide_target_by_weight,
                        )
                        if pending is not None and (
                            root_changed or plan.stages != old.stages
                        ):
                            raise ValueError(
                                "A stage is fitting. Wait for it or start a new chain."
                            )
                        if old_refs and (
                            root_changed
                            or plan.stages[: len(old_refs)]
                            != old.stages[: len(old_refs)]
                        ):
                            raise ValueError(
                                "A fitted stage is frozen. Choose Revise model plan to start a new version."
                            )
                prepared = apply_variables(raw, current.data)
                validate_model_plan(
                    current, plan, raw, set(prepared.columns), through_count=0
                )
                validate_plan_catalogues(plan.stages, require_source=False)
                candidate = deepcopy(current)
                candidate.model_plan = plan
                if candidate.to_dict() != current.to_dict():
                    current = candidate
                    persist_plan_refs()
                    revision += 1
                    screenings.invalidate((session_id, project_id, revision))
                    feature_selections.invalidate((session_id, project_id, revision))
                return {**plan_view(), "session_id": session_id, "revision": revision}
            except (ValueError, TypeError, KeyError, OSError) as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/model-plan/fit-next", status_code=202)
    def fit_next_plan_stage(edit: Revision) -> dict[str, Any]:
        """Fit exactly the next prefix, using its exact server-issued parent."""
        nonlocal current, revision, plan_refs
        with lock:
            check_revision(edit)
            plan = current.model_plan
            if plan is None or not plan.stages:
                raise HTTPException(422, "Define at least one model stage first.")
            try:
                refs, pending = plan_ref_state(plan)
                if pending is not None:
                    raise ValueError("This stage is already fitting.")
                count = len(refs) + 1
                if plan.adopted_model_name is not None and count == 1:
                    raise ValueError(
                        "The adopted scorer is unavailable. Adopt the saved model again."
                    )
                if count > len(plan.stages):
                    raise ValueError("Every declared stage is already fitted.")
                prepared = apply_variables(raw, current.data)
                validate_model_plan(
                    current, plan, raw, set(prepared.columns), through_count=count
                )
                validate_plan_catalogues(plan.stages[:count], require_source=True)
                name = plan_model_name(current, plan, count)
                config = materialise_plan_prefix(current, plan, count, name)
                candidate = deepcopy(current)
                candidate.models[name] = config
                source_path: Path | None = None
                source_name: str | None = None
                if refs:
                    previous = refs[-1]
                    source_name = previous["model_name"]
                    source_path = jobs.artifact(current, source_name)
                    source_stack = fitted_prefix(source_name)
                    config.lineage = ModelLineage(
                        parent_name=source_name,
                        parent_fit_id=previous["fit_id"],
                        parent_key=model_key(current, source_name),
                        parent_prefix_fingerprint=scorer_fingerprint(source_stack),
                        prefix_length=len(refs),
                    )
                    candidate.models[name] = config
                job = jobs.start(
                    candidate,
                    raw.clone(),
                    name,
                    source_fit=source_path,
                    frozen_source_name=source_name,
                )
                # Bind this queued worker to precisely this still-current plan.
                jobs.jobs[name]["model_plan_id"] = plan.plan_id
                jobs.jobs[name]["model_plan_stage_id"] = plan.stages[
                    count - 1
                ].component_id
                stage_id = plan.stages[count - 1].component_id
                plan_refs = [
                    item
                    for item in plan_refs
                    if not (item.plan_id == plan.plan_id and item.stage_id == stage_id)
                ]
                issued_ref = PlanFitRef(plan.plan_id, stage_id, name, job["id"])
                plan_refs.append(issued_ref)
                plan_attempts[(plan.plan_id, stage_id)] = issued_ref
                current = candidate
                persist_plan_refs()
                revision += 1
                return {
                    "job": job,
                    "model_name": name,
                    "stage_id": plan.stages[count - 1].component_id,
                    "snapshot": {
                        **plan_view(),
                        "revision": revision,
                        "session_id": session_id,
                    },
                }
            except (ValueError, TypeError, KeyError, OSError) as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/models/save")
    def save_model(edit: ModelEdit) -> dict[str, Any]:
        nonlocal current, revision
        with lock:
            check_revision(edit)
            try:
                existing = current.models.get(edit.name)
                existing_job = jobs.status(current).get(edit.name, {})
                if (
                    not edit.create
                    and existing is not None
                    and existing.components
                    and existing_job.get("applicable")
                ):
                    raise HTTPException(
                        409,
                        "This fitted model is read-only. Revise it as a new named model.",
                    )
                if not edit.create and existing is not None and existing.lineage:
                    if edit.source_name is not None or edit.source_fit_id is not None:
                        raise ValueError(
                            "A saved draft keeps its recorded parent automatically."
                        )
                    if (
                        edit.designs
                        or edit.n_bins is not None
                        or edit.min_level_share is not None
                    ):
                        raise ValueError(
                            "Bands are defined in Project & data and cannot be changed from this model draft."
                        )
                for component in edit.fields.get("components", []):
                    if component.get("kind") != "catalogue":
                        continue
                    settings = component.get("settings", {})
                    if settings.get("catalogue") is not None:
                        continue
                    source_path = settings.get("source_path")
                    source_hash = settings.get("source_sha256")
                    if not source_path or not source_hash:
                        raise ValueError(
                            "Upload a catalogue before saving its component."
                        )
                    canonical = str(Path(source_path).resolve())
                    if catalogue_sources.get(canonical) != source_hash:
                        raise ValueError(
                            "Catalogue source is not a pinned upload in this session."
                        )
                    actual_hash = _file_sha256(Path(canonical))
                    if actual_hash != source_hash:
                        raise ValueError("Catalogue contents changed since upload.")
                candidate_project = edit_model(current, edit)
                scope_start = 0
                if bool(edit.source_name) != bool(edit.source_fit_id):
                    raise ValueError(
                        "Choose a fitted source and its exact fit identity together."
                    )
                if edit.source_name:
                    from easy_glm.workflow.stack import _check_prefix

                    if edit.source_name == edit.name:
                        raise ValueError("Append into a new named candidate.")
                    if edit.source_name not in current.models:
                        raise ValueError("The starting model is unavailable.")
                    source_path = jobs.artifact(current, edit.source_name)
                    if source_path.name != edit.source_fit_id:
                        raise ValueError(
                            "The starting fit changed. Refresh before appending."
                        )
                    prefix_stack = fitted_prefix(edit.source_name)
                    config = candidate_project.models[edit.name]
                    prefix_recipes = [
                        item.recipe
                        for item in prefix_stack.components
                        if item.recipe.kind != "baseline"
                    ]
                    if (
                        config.components[: len(prefix_recipes)] != prefix_recipes
                        or len(config.components) != len(prefix_recipes) + 1
                    ):
                        raise ValueError(
                            "The append recipe must preserve the fitted prefix and add one component."
                        )
                    _check_prefix(config, prefix_stack)
                    config.lineage = ModelLineage(
                        parent_name=edit.source_name,
                        parent_fit_id=edit.source_fit_id,
                        parent_key=model_key(current, edit.source_name),
                        parent_prefix_fingerprint=scorer_fingerprint(prefix_stack),
                        prefix_length=len(prefix_recipes),
                    )
                    scope_start = len(prefix_recipes)
                elif not edit.create and existing is not None and existing.lineage:
                    # A draft child may change only the last component.  Its
                    # fitted parent, frozen prefix and response convention are
                    # deliberately copied from the existing saved definition.
                    config = candidate_project.models[edit.name]
                    prefix_length = existing.lineage.prefix_length
                    if (
                        len(existing.components) != prefix_length + 1
                        or len(config.components) != prefix_length + 1
                        or config.components[:prefix_length]
                        != existing.components[:prefix_length]
                    ):
                        raise ValueError(
                            "This draft can change only its final model component."
                        )
                    original = dict(current.to_dict()["models"][edit.name])
                    amended = dict(candidate_project.to_dict()["models"][edit.name])
                    original.pop("components", None)
                    original.pop("lineage", None)
                    amended.pop("components", None)
                    amended.pop("lineage", None)
                    if amended != original:
                        raise ValueError(
                            "This draft keeps its original settings; change only its final model component."
                        )
                    config.lineage = existing.lineage
                    scope_start = prefix_length
                elif edit.create:
                    # A root candidate has no implicit provenance, even when
                    # its recipes resemble an existing model.
                    candidate_project.models[edit.name].lineage = None
                if "components" in edit.fields:
                    prepared = apply_variables(raw, candidate_project.data)
                    validate_stack_factor_scope(
                        candidate_project,
                        raw,
                        set(prepared.columns),
                        candidate_project.models[edit.name],
                        start_index=scope_start,
                    )
                if candidate_project.models[edit.name].pair_stages:
                    prepared = apply_variables(raw, candidate_project.data)
                    groups = pair_candidate_groups(
                        candidate_project, raw, set(prepared.columns)
                    )
                    eligible = set(groups["predictors"] + groups["unassigned"])
                    for stage in candidate_project.models[edit.name].pair_stages:
                        if stage.a not in eligible or stage.b not in eligible:
                            raise ValueError(
                                "Interaction variables must be predictors or unassigned "
                                "columns from the source data."
                            )
                    errors = candidate_project.validate(
                        edit.name, columns=prepared.columns
                    )
                    if errors:
                        raise ValueError("; ".join(errors))
            except (ValueError, TypeError, KeyError) as exc:
                raise HTTPException(422, str(exc)) from exc
            if candidate_project.to_dict() != current.to_dict():
                current = candidate_project
                revision += 1
                screenings.invalidate((session_id, project_id, revision))
                feature_selections.invalidate((session_id, project_id, revision))
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
                feature_selections.invalidate((session_id, project_id, revision))
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
                feature_selections.invalidate((session_id, project_id, revision))
                jobs.invalidate(current)
            return snapshot()

    @app.post("/api/models/{name}/fit", status_code=202)
    def start_fit(name: str, edit: FitRequest) -> dict[str, Any]:
        with lock:
            check_revision(edit)
            if name not in current.models:
                raise HTTPException(404, "Save a model definition before fitting.")
            config = current.models[name]
            existing_job = jobs.status(current).get(name, {})
            if config.components and existing_job.get("applicable"):
                raise HTTPException(
                    409,
                    "This fitted model is read-only. Revise it as a new named model.",
                )
            if (
                config.components
                and existing_job.get("status") == "complete"
                and not existing_job.get("artifact_available", False)
            ):
                raise HTTPException(
                    409,
                    "The previous fitted model files are missing. Create a new named copy to fit this recipe again.",
                )
            if config.components and config.lineage is not None and edit.full_refit:
                raise HTTPException(
                    409,
                    "Create a new named copy to rebuild this saved model from scratch.",
                )
            problems = current.validate(name)
            if problems:
                raise HTTPException(422, "; ".join(problems))
            try:
                if config.components:
                    prepared = apply_variables(raw, current.data)
                    start_index = 0
                    if not edit.full_refit and config.lineage is not None:
                        start_index = config.lineage.prefix_length
                    validate_stack_factor_scope(
                        current,
                        raw,
                        set(prepared.columns),
                        current.models[name],
                        start_index=start_index,
                    )
                if edit.full_refit:
                    # Never let an in-memory append source turn this explicit
                    # recovery action into historical scorer reuse.
                    return jobs.start(
                        deepcopy(current), raw.clone(), name, full_refit=True
                    )
                source = verified_append_source(name)
                if source is not None:
                    source_name, source_path = source
                    return jobs.start(
                        deepcopy(current),
                        raw.clone(),
                        name,
                        source_fit=source_path,
                        frozen_source_name=source_name,
                    )
                if config.lineage is not None:
                    if not edit.full_refit:
                        raise HTTPException(
                            409,
                            "This draft's original fitted model is no longer available. "
                            "Create a new named copy to fit its complete recipe afresh.",
                        )
                    # This intentionally does not consult the lineage parent.
                    # It rebuilds the child's complete recorded recipe, so a
                    # historical graph edge can never supply a scorer.
                    return jobs.start(deepcopy(current), raw.clone(), name)
                return jobs.start(deepcopy(current), raw.clone(), name)
            except ValueError as exc:
                # Bad component scope is a validation error; stale provenance
                # is a conflict requiring a fresh save or full refit.
                status = (
                    422 if "Component " in str(exc) or "Catalogue " in str(exc) else 409
                )
                raise HTTPException(status, str(exc)) from exc

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
    def results(name: str, stack_view: bool = False) -> dict[str, Any]:
        with lock:
            try:
                result = jobs.result(current, name)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            if result.get("stack"):
                return result
            if not stack_view:
                return {
                    **{k: v for k, v in result.items() if k != "tables"},
                    "table_index": [
                        {"name": key, "rows": len(table["rows"]), "kind": "main"}
                        for key, table in result["tables"].items()
                    ]
                    + [
                        {
                            "name": key,
                            "stage_id": key,
                            "label": next(
                                (
                                    f"{stage.a} × {stage.b}"
                                    for stage in current.models[name].pair_stages
                                    if stage.stage_id == key
                                ),
                                key,
                            ),
                            "rows": len(table["rows"]),
                            "kind": "pair",
                        }
                        for key, table in result.get("pair_tables", {}).items()
                    ],
                }
        from easy_glm.desktop.fit_worker import json_safe
        from easy_glm.desktop.stack_views import stack_result

        saved, frame, fitted = stack_artifact(name)
        return json_safe(stack_result(saved, frame, fitted, result.get("warnings")))

    def stack_artifact(name: str) -> tuple[Project, pl.DataFrame, Any]:
        with lock:
            try:
                source = jobs.artifact(current, name)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            saved, saved_raw = deepcopy(current), raw.clone()
        with (source / "fit.pkl").open("rb") as handle:
            fitted = pickle.load(handle)  # private artifact from our fit worker
        if fitted.name != name:
            raise HTTPException(
                409, "The fitted artifact does not match the selected model."
            )
        from easy_glm.workflow.prep import prepare

        frame = prepare(saved, saved_raw)
        if getattr(fitted, "stack", None) is None:
            from easy_glm.workflow.run import rate_model_for
            from easy_glm.workflow.stack import stack_from_legacy

            # Expose a compatible single scorer for stack inspection without
            # changing its fitted artifact or the ordinary single-model result.
            fitted.rate_model = rate_model_for(saved, fitted)
            fitted.config = saved.models[name]
            try:
                fitted.stack = stack_from_legacy(saved, fitted)
            except ValueError as exc:
                raise HTTPException(
                    404, "This fitted model is not a compatible component stack."
                ) from exc
        return saved, frame, fitted

    @app.get("/api/stack/{name}/diagnostics")
    def stack_diagnostic_view(
        name: str,
        subset: Literal["train", "holdout"] = "train",
        variable: str | None = None,
        through: str | None = None,
    ) -> dict[str, Any]:
        from easy_glm.desktop.fit_worker import json_safe
        from easy_glm.desktop.stack_views import stack_diagnostics

        try:
            saved, frame, fitted = stack_artifact(name)
            return json_safe(
                stack_diagnostics(
                    saved,
                    frame,
                    fitted,
                    subset=subset,
                    variable=variable,
                    through=through,
                )
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/stack/{name}/compare")
    def stack_comparison(
        name: str, challenger: str, subset: Literal["train", "holdout"] = "train"
    ) -> dict[str, Any]:
        from easy_glm.desktop.diagnostic_views import compatible
        from easy_glm.desktop.fit_worker import json_safe
        from easy_glm.desktop.stack_views import stack_diagnostics
        from easy_glm.workflow.diagnostics import totals
        from easy_glm.workflow.prep import train_holdout

        try:
            saved, frame, fitted = stack_artifact(name)
            _other_project, _other_frame, other = stack_artifact(challenger)
            compatible(fitted, other)
            train, holdout = train_holdout(frame, saved.data.split)
            selected = train if subset == "train" else holdout
            if selected.is_empty():
                raise ValueError(f"No {subset} rows are available.")
            actual, expected, weight = totals(
                selected, fitted.config, fitted.stack.predict_unit(selected)
            )
            _actual, other_expected, _weight = totals(
                selected, other.config, other.predict(selected)
            )
            return json_safe(
                {
                    "subset": subset,
                    "model": name,
                    "challenger": challenger,
                    "metrics": stack_diagnostics(saved, frame, fitted, subset=subset)[
                        "metrics"
                    ],
                    "challenger_metrics": (
                        stack_diagnostics(saved, frame, other, subset=subset)["metrics"]
                        if getattr(other, "stack", None) is not None
                        else {
                            "rows": selected.height,
                            "exposure": float(weight.sum()),
                            "actual": float(actual.sum()),
                            "expected": float(other_expected.sum()),
                            "ae": (
                                float(actual.sum() / other_expected.sum())
                                if other_expected.sum() > 0
                                else None
                            ),
                        }
                    ),
                    "expected_change": (
                        float(expected.sum() / other_expected.sum() - 1)
                        if other_expected.sum() > 0
                        else None
                    ),
                }
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/stack/{name}/component/{component_id}/table")
    def stack_component_table(
        name: str,
        component_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=500),
    ) -> dict[str, Any]:
        from easy_glm.core.excel import pair_table_frames, rate_model_tables
        from easy_glm.desktop.stack_views import lookup_coverage
        from easy_glm.workflow.prep import train_holdout
        from easy_glm.workflow.stack_types import LookupScorer, RateModelScorer

        try:
            saved, frame, fitted = stack_artifact(name)
            item = fitted.stack.component(component_id)
            if isinstance(item.scorer, LookupScorer):
                tables = {"lookup": item.scorer.table}
            elif isinstance(item.scorer, RateModelScorer):
                tables = rate_model_tables(item.scorer.rate_model)
                if item.scorer.rate_model.pair_tables:
                    tables.update(pair_table_frames(item.scorer.rate_model))
            else:
                tables = {}
            visible = [
                *fitted.stack.virtual_components,
                *(
                    component
                    for component in fitted.stack.components
                    if component.kind != "legacy" or not fitted.stack.virtual_components
                ),
            ]
            position = visible.index(item)
            unit_label = (
                "Complete unit prediction"
                if position == 0
                and item.kind == "catalogue"
                and not fitted.stack.offset_col
                else (
                    "Multiplier on external offset"
                    if position == 0 and item.kind == "catalogue"
                    else "Incremental multiplier"
                )
            )
            return {
                "id": item.recipe.component_id,
                "name": item.name,
                "kind": item.kind,
                "editable": isinstance(item.scorer, LookupScorer),
                "unit_label": unit_label,
                "coverage": (
                    lookup_coverage(
                        item.scorer, train_holdout(frame, saved.data.split)[0]
                    )
                    if isinstance(item.scorer, LookupScorer)
                    else None
                ),
                "evidence": item.evidence,
                "tables": [
                    {
                        "name": key,
                        "columns": table.columns,
                        "rows": table.slice(offset, limit).to_dicts(),
                        "total": table.height,
                        "offset": offset,
                    }
                    for key, table in tables.items()
                ],
                "message": (
                    "Direct CatBoost trees have no editable rate table."
                    if not tables
                    else ""
                ),
            }
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/stack/{name}/export/{component_id}")
    def stack_component_export(name: str, component_id: str) -> Response:
        from easy_glm.workflow.project import safe_filename
        from easy_glm.workflow.stack_types import LookupScorer

        try:
            _saved, _frame, fitted = stack_artifact(name)
            item = fitted.stack.component(component_id)
            if not isinstance(item.scorer, LookupScorer):
                raise ValueError(
                    "Only a catalogue lookup can be exported as a rating file."
                )
            filename = safe_filename(f"{name}_{item.name}") + ".csv"
            position = fitted.stack.components.index(item)
            unit_label = (
                "Complete unit prediction"
                if position == 0 and not fitted.stack.offset_col
                else (
                    "Multiplier on external offset"
                    if position == 0
                    else "Incremental multiplier"
                )
            )
            return Response(
                item.scorer.table.write_csv(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "X-EasyGLM-Unit-Label": unit_label,
                },
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    def preview_stack_amendment(
        name: str, component_id: str, edit: StackAmendEdit
    ) -> tuple[Project, pl.DataFrame, Any, Any, dict[str, Any]]:
        from easy_glm.workflow.diagnostics import totals
        from easy_glm.workflow.prep import train_holdout
        from easy_glm.workflow.stack import amend_stack
        from easy_glm.workflow.stack_types import LookupScorer

        with lock:
            check_revision(edit)
            source = jobs.artifact(current, name)
            if edit.fit_id is not None and source.name != edit.fit_id:
                raise ValueError("The fitted model changed. Preview this edit again.")
        saved, frame, fitted = stack_artifact(name)
        component = fitted.stack.component(component_id)
        if not isinstance(component.scorer, LookupScorer):
            raise ValueError(
                "Caps and rounding are available for catalogue lookup files."
            )
        if edit.operation == "cap":
            if edit.lower is None and edit.upper is None:
                raise ValueError("Enter a lower or upper cap.")
            if edit.lower is not None and edit.lower <= 0:
                raise ValueError("The lower cap must be positive.")
            if edit.upper is not None and edit.upper <= 0:
                raise ValueError("The upper cap must be positive.")
            if (
                edit.lower is not None
                and edit.upper is not None
                and edit.lower > edit.upper
            ):
                raise ValueError("The lower cap cannot exceed the upper cap.")
        elif edit.decimals is None:
            raise ValueError("Choose the number of decimal places.")
        train, _holdout = train_holdout(frame, saved.data.split)
        amended = amend_stack(
            saved,
            train,
            fitted,
            component_id,
            edit.operation,
            lower=edit.lower,
            upper=edit.upper,
            decimals=edit.decimals,
            refit_later=False,
        )
        old_table = component.scorer.table
        new_table = amended.stack.component(component_id).scorer.table
        old_values = old_table["multiplier"].cast(pl.Float64).to_numpy()
        new_values = new_table["multiplier"].cast(pl.Float64).to_numpy()
        _actual, old_expected, _weight = totals(
            train, fitted.config, fitted.predict(train)
        )
        _actual, new_expected, _weight = totals(
            train, amended.config, amended.predict(train)
        )
        key = component.scorer.key
        changed = np.flatnonzero(~np.isclose(old_values, new_values, rtol=0, atol=0))
        detail = {
            "component_id": component_id,
            "changed_rows": int(changed.size),
            "expected_before": float(old_expected.sum()),
            "expected_after": float(new_expected.sum()),
            "sample_rows": [
                {
                    "key": old_table[key][int(index)],
                    "before": float(old_values[index]),
                    "after": float(new_values[index]),
                }
                for index in changed[:20]
            ],
            "refit_later": edit.refit_later,
            "preview_basis": (
                "Later components kept fixed; refit is performed after Apply"
                if edit.refit_later
                else "Complete accepted scorer on training rows"
            ),
        }
        return saved, frame, fitted, amended, detail

    @app.post("/api/stack/{name}/component/{component_id}/preview")
    def preview_stack_component(
        name: str, component_id: str, edit: StackAmendEdit
    ) -> dict[str, Any]:
        try:
            _saved, _frame, _fitted, _amended, detail = preview_stack_amendment(
                name, component_id, edit
            )
            return detail
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/stack/{name}/component/{component_id}/apply")
    def apply_stack_component(
        name: str, component_id: str, edit: StackAmendEdit
    ) -> dict[str, Any]:
        nonlocal current, revision
        from easy_glm.desktop.fit_worker import json_safe
        from easy_glm.desktop.stack_views import stack_result
        from easy_glm.workflow.project import validate_model_name

        try:
            saved, frame, _fitted, amended, detail = preview_stack_amendment(
                name, component_id, edit
            )
            if not edit.new_name:
                raise ValueError("Name the reviewed model checkpoint.")
            problem = validate_model_name(edit.new_name, saved.models)
            if problem:
                raise ValueError(problem)
            candidate = deepcopy(saved)
            candidate.models[edit.new_name] = amended.config
            amended.name = edit.new_name
            amended.project_snapshot = candidate.to_dict()
            with lock:
                check_revision(edit)
                if edit.refit_later:
                    source_fit = jobs.artifact(current, name)
                    queued = jobs.start(
                        deepcopy(candidate),
                        raw.clone(),
                        edit.new_name,
                        amendment={
                            "component_id": component_id,
                            "operation": edit.operation,
                            "lower": edit.lower,
                            "upper": edit.upper,
                            "decimals": edit.decimals,
                        },
                        source_fit=source_fit,
                    )
                    current = candidate
                    revision += 1
                    return {"snapshot": snapshot(), "job": queued, "preview": detail}
                result = json_safe(stack_result(candidate, frame, amended))
                published = jobs.publish_stack(
                    candidate, raw.clone(), edit.new_name, amended, result
                )
                current = candidate
                revision += 1
                return {"snapshot": snapshot(), "job": published, "preview": detail}
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/results/{name}/table")
    def result_table(
        name: str,
        variable: str = "",
        stage_id: str | None = None,
        offset: int = 0,
        limit: int = 200,
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
            tables = (
                result.get("pair_tables", {})
                if stage_id is not None
                else result["tables"]
            )
            key = stage_id if stage_id is not None else variable
            if key not in tables:
                raise HTTPException(404, "Unknown fitted variable.")
            table = tables[key]
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
                    feature_selections.invalidate((session_id, project_id, revision))
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
                    feature_selections.invalidate((session_id, project_id, revision))
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
                    feature_selections.invalidate((session_id, project_id, revision))
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
                            pair_stage_fingerprint=snapshot_pair_identity(name, source),
                        )
                    )
                    current.models[name].snapshots = cfg.snapshots
                    revision += 1
                    screenings.invalidate((session_id, project_id, revision))
                    feature_selections.invalidate((session_id, project_id, revision))
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
                        if (
                            cfg.pair_stages
                            and matched.pair_stage_fingerprint
                            != snapshot_pair_identity(name, source)
                        ):
                            raise ValueError(
                                "This snapshot was saved against different pair-stage axes or fitted prefix. "
                                "Fit the current sequence and save a new snapshot before restoring it."
                            )
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
                feature_selections.invalidate((session_id, project_id, revision))
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
                "pair_stages": [
                    {"stage_id": stage.stage_id, "a": stage.a, "b": stage.b}
                    for stage in current.models[name].pair_stages
                ],
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
                saved,
                saved_raw,
                sources,
                name,
                edit.format,
                edit.challenger,
                importance_sample_pct=edit.importance_sample_pct,
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
