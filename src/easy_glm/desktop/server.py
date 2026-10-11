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
import tempfile
import threading
import time
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
from starlette.background import BackgroundTask

from easy_glm.desktop.binning import (
    apply_binning_setup,
    binning_changes,
    binning_columns,
    binning_preview,
    binning_setup,
    validate_changed_linear_binning,
)
from easy_glm.desktop.core_interactions import (
    all_candidates,
    candidates_for,
    public_candidate,
    remove_candidate,
    replace_candidate,
)
from easy_glm.desktop.core_search import discover as discover_core_residuals
from easy_glm.desktop.datasets import ScoringDatasets
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
    materialise_plan_through_component,
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
from easy_glm.workflow.model_plan import AdoptedRoot, ModelPlan, new_plan_id
from easy_glm.workflow.prep import apply_variables
from easy_glm.workflow.project import (
    SINGLE_ROLES,
    ModelConfig,
    ModelLineage,
    Project,
    pair_stage_fingerprint,
    safe_filename,
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


class CoreInteractionCreate(Revision):
    """One unaccepted interaction candidate under a named GLM core."""

    model_id: str = Field(min_length=1)
    a: str = Field(min_length=1)
    b: str = Field(min_length=1)
    settings: dict[str, Any] = Field(default_factory=dict)
    design: dict[str, Any] | None = None


class CoreInteractionCandidateEdit(Revision):
    candidate_id: str = Field(min_length=1)


class CoreInteractionRecipeEdit(Revision):
    """A mutation of an already accepted table identified in the URL."""


class CoreInteractionMove(CoreInteractionRecipeEdit):
    before_component_id: str | None = None


class CoreSearchEdit(Revision):
    action: Literal["factors", "interactions"]
    n_bins: int = Field(default=8, ge=2, le=50)
    min_expected: float = Field(default=3.0, gt=0)
    min_cell_share: float = Field(default=0.0, ge=0, lt=1)
    top: int = Field(default=20, ge=1, le=200)


class ScoringDatasetFile(Revision):
    name: str = Field(min_length=1, max_length=255)
    path: str = Field(min_length=1)
    replace_id: str | None = None


class ScoringRequest(Revision):
    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    scope: Literal["full", "through", "model", "catalogue"] = "full"
    component_ids: list[str] = Field(default_factory=list)
    through: str | None = None
    exposure_col: str | None = None
    column_prefix: str = Field(default="easyglm_", max_length=100)


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
    scoring_frames: dict[str, pl.DataFrame] | None = None,
) -> FastAPI:
    """Serve one local, in-memory project and bundled assets."""
    jobs = FitJobs()
    if restore_folder is not None:
        jobs.restore(project, raw, restore_folder)
    reviews = ReviewJobs()
    exploration = ExplorationCache()
    screenings = ScreeningJobs()
    feature_selections = FeatureSelectionJobs()
    scoring_datasets = ScoringDatasets()
    for descriptor in project.scoring_sources:
        if isinstance(descriptor, dict):
            scoring_datasets.restore_descriptor(descriptor)
    for dataset_name, scoring_frame in (scoring_frames or {}).items():
        scoring_datasets.add_frame(dataset_name, scoring_frame)
    undo_steps = {}
    redo_steps = {}

    def persist_scoring_sources() -> None:
        """Only explicit local-file recipes survive an exported project."""
        current.scoring_sources = [
            item for item in scoring_datasets.descriptors() if item["source"] == "file"
        ]

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
        scoring_datasets.close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    current = deepcopy(project)
    revision = 0
    lock = threading.RLock()
    # These references are server-issued when a stage is queued.  They are
    # intentionally never reconstructed by comparing old saved recipes: a new
    # chain with identical settings must start cleanly.
    plan_attempts: dict[tuple[str, str], PlanFitRef] = {}

    def _plan_ref_from_saved(plan: ModelPlan, item: dict[str, str]) -> PlanFitRef:
        if plan.is_logical_models_plan:
            return PlanFitRef(
                plan.plan_id,
                item["through_component_id"],
                item["model_name"],
                item["fit_id"],
                model_id=item["model_id"],
                prefix_fingerprint=item["prefix_fingerprint"],
            )
        return PlanFitRef(
            plan.plan_id, item["stage_id"], item["model_name"], item["fit_id"]
        )

    plan_refs: list[PlanFitRef] = [
        _plan_ref_from_saved(current.model_plan, item)
        for item in (current.model_plan.fit_refs if current.model_plan else [])
    ]

    def _logical_models(plan: ModelPlan) -> list[Any]:
        """Read v2 logical models while leaving imported v1 plans untouched."""
        models = getattr(plan, "models", None)
        if isinstance(models, list):
            return models
        return []

    def _logical_model(plan: ModelPlan, model_id: str) -> Any:
        for model in _logical_models(plan):
            if getattr(model, "model_id", None) == model_id:
                return model
        raise ValueError("Choose a model in the current plan.")

    def _core_recipe(model: Any) -> ComponentRecipe:
        core = getattr(model, "core", None)
        if not isinstance(core, ComponentRecipe):
            raise ValueError("This imported plan does not use the core-model workflow.")
        if core.kind != "glm":
            raise ValueError("Only a GLM can own interaction tables.")
        return core

    def _pair_recipes(model: Any) -> list[ComponentRecipe]:
        pairs = getattr(model, "pair_corrections", [])
        return list(pairs) if isinstance(pairs, list | tuple) else []

    def _core_source_ref(plan: ModelPlan, model: Any) -> PlanFitRef:
        """The exact accepted scorer through this GLM's current pair table."""
        component_ids = {
            _core_recipe(model).component_id,
            *(item.component_id for item in _pair_recipes(model)),
        }
        for ref in reversed(plan_refs):
            if (
                ref.plan_id == plan.plan_id
                and ref.stage_id in component_ids
                and (ref.model_id is None or ref.model_id == model.model_id)
            ):
                return ref
        raise ValueError("Fit this model's main effects before adding an interaction.")

    def _checkpoint_for_pair_prefix(
        plan: ModelPlan, model: Any, pairs: tuple[ComponentRecipe, ...]
    ) -> PlanFitRef:
        """Locate the immutable historical checkpoint through an unchanged prefix."""
        from dataclasses import replace as dataclass_replace

        index = plan.models.index(model)
        temporary = deepcopy(plan)
        prefix_model = dataclass_replace(model, pair_corrections=pairs)
        temporary.models[index] = prefix_model
        temporary.stages = [
            component
            for item in temporary.models
            for component in item.physical_components
        ]
        through = pairs[-1].component_id if pairs else prefix_model.core.component_id
        expected = materialise_plan_through_component(
            current, temporary, index, through
        )
        for name, config in reversed(list(current.models.items())):
            status = jobs.status(current).get(name, {})
            if not status.get("applicable"):
                continue
            if config != expected and replace(config, lineage=None) != expected:
                continue
            try:
                fingerprint = scorer_fingerprint(fitted_prefix(name))
            except (KeyError, OSError, ValueError):
                continue
            return PlanFitRef(
                plan.plan_id,
                through,
                name,
                status["id"],
                model_id=model.model_id,
                prefix_fingerprint=fingerprint,
            )
        raise ValueError("The frozen checkpoint before this table is unavailable.")

    def _candidate_entry(candidate_id: str) -> dict[str, Any]:
        entry = next(
            (
                item
                for item in all_candidates(current)
                if item.get("candidate_id") == candidate_id
            ),
            None,
        )
        if entry is None:
            raise ValueError("This interaction candidate is unavailable.")
        return entry

    def _candidate_is_current(entry: dict[str, Any]) -> bool:
        """A reviewed table may only be kept against its exact frozen source."""
        source_name = entry.get("source_model_name")
        source_fit_id = entry.get("source_fit_id")
        if not isinstance(source_name, str) or not isinstance(source_fit_id, str):
            return False
        try:
            plan = current.model_plan
            if plan is None:
                return False
            source_ref = _core_source_ref(plan, _logical_model(plan, entry["model_id"]))
            if (
                source_ref.model_name != source_name
                or source_ref.fit_id != source_fit_id
                or source_ref.stage_id != entry.get("source_through_component_id")
            ):
                return False
            source = jobs.artifact(current, source_name)
            return source.name == source_fit_id and scorer_fingerprint(
                fitted_prefix(source_name)
            ) == entry.get("source_fingerprint")
        except (KeyError, OSError, ValueError):
            return False

    def _pair_table_payload(stack: Any, component_id: str) -> dict[str, Any] | None:
        """Return the actual fitted cells of one native interaction table."""
        from easy_glm.core.excel import pair_table_frames
        from easy_glm.desktop.fit_worker import json_safe
        from easy_glm.workflow.stack_types import RateModelScorer

        component = stack.component(component_id)
        if not isinstance(component.scorer, RateModelScorer):
            return None
        frame = pair_table_frames(component.scorer.rate_model).get(component_id)
        if frame is None:
            return None
        rows = frame.to_dicts()
        # ``fitted`` is the established table-review field used by the
        # workbench chart.  Keep ``relativity`` too: it is the exported pair
        # table column and remains useful in a raw table download.
        for row in rows:
            row["fitted"] = row["relativity"]
        return json_safe(
            {
                "columns": [*frame.columns, "fitted"],
                "rows": rows,
                "total": frame.height,
                "stage_id": component_id,
            }
        )

    def core_interaction_view(model_id: str) -> dict[str, Any]:
        """One logical core plus its accepted tables and isolated candidates."""
        plan = current.model_plan
        if plan is None:
            raise ValueError("Define the model plan first.")
        model = _logical_model(plan, model_id)
        core = _core_recipe(model)
        prepared = apply_variables(raw, current.data)
        groups = pair_candidate_groups(current, raw, set(prepared.columns))
        candidates = []
        for candidate in candidates_for(current, model_id):
            item = public_candidate(candidate)
            if item.get("status") in {"draft", "review"} and not _candidate_is_current(
                candidate
            ):
                item["status"] = "needs_rebuild"
                item["message"] = "The accepted source changed; rebuild this candidate."
            candidates.append(item)
        source: dict[str, Any] | None = None
        accepted_tables: dict[str, dict[str, Any]] = {}
        accepted_through: str | None = None
        try:
            reference = _core_source_ref(plan, model)
            source = {
                "model_name": reference.model_name,
                "fit_id": reference.fit_id,
                "fingerprint": scorer_fingerprint(fitted_prefix(reference.model_name)),
            }
            accepted_through = reference.stage_id
            _saved, _frame, fitted = stack_artifact(reference.model_name)
            if getattr(fitted, "stack", None) is not None:
                for pair in _pair_recipes(model):
                    table = _pair_table_payload(fitted.stack, pair.component_id)
                    if table is not None:
                        accepted_tables[pair.component_id] = table
        except (KeyError, OSError, ValueError):
            pass
        return {
            "model_id": model_id,
            "name": getattr(model, "name", core.name),
            "core": core.to_dict(),
            "pair_corrections": [item.to_dict() for item in _pair_recipes(model)],
            "pair_statuses": [
                {
                    "component_id": pair.component_id,
                    "state": (
                        "fitted"
                        if accepted_through is not None
                        and index
                        <= next(
                            (
                                position
                                for position, item in enumerate(_pair_recipes(model))
                                if item.component_id == accepted_through
                            ),
                            -1,
                        )
                        else "needs_refit"
                    ),
                }
                for index, pair in enumerate(_pair_recipes(model))
            ],
            "accepted_through_component_id": accepted_through,
            "accepted_tables": accepted_tables,
            "source": source,
            "eligible": {
                "main": [name for name in core.factors if name in prepared.columns],
                "other": groups["predictors"] + groups["unassigned"],
            },
            "candidates": candidates,
        }

    def persist_plan_refs() -> None:
        if current.model_plan is not None:
            if current.model_plan.is_logical_models_plan:
                current.model_plan.fit_refs = [
                    {
                        "model_id": item.model_id or item.stage_id,
                        "through_component_id": item.stage_id,
                        "model_name": item.model_name,
                        "fit_id": item.fit_id,
                        "prefix_fingerprint": item.prefix_fingerprint or "unknown",
                    }
                    for item in plan_refs
                    if item.plan_id == current.model_plan.plan_id
                ]
            else:
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
            candidate_id = job.get("core_interaction_candidate_id")
            if isinstance(candidate_id, str):
                # Candidate jobs deliberately have no entry in ``current.models``.
                # Publishing their result through the ordinary model collection
                # would make a never-reviewed table appear in A/E, export and
                # downstream offsets.  Keep only an explicitly labelled review
                # payload until the user chooses Keep.
                try:
                    entry = _candidate_entry(candidate_id)
                except ValueError:
                    job.update(status="stale", message="Candidate was discarded.")
                    return
                if not _candidate_is_current(entry):
                    entry.update(
                        status="needs_rebuild",
                        message="The accepted source changed while this candidate fitted.",
                    )
                    replace_candidate(current, entry)
                    job.update(status="stale", message=entry["message"])
                    revision += 1
                    return
                source_result = jobs.result(current, entry["source_model_name"])
                source_metrics = source_result.get("metrics", {})
                candidate_metrics = result.get("metrics", {})
                component_evidence = next(
                    (
                        component.get("evidence", {})
                        for component in reversed(result.get("components", []))
                        if component.get("id")
                        == ComponentRecipe.from_dict(entry["recipe"]).component_id
                    ),
                    {},
                )
                no_change = bool(component_evidence.get("no_change"))
                candidate_recipe = ComponentRecipe.from_dict(entry["recipe"])
                reviewed_table = None
                if not no_change:
                    # Candidate artifacts deliberately have no entry in
                    # ``current.models``.  Read this exact private job file,
                    # rather than going through the public scorer accessor.
                    artifact = Path(jobs.folder.name) / job["id"] / "fit.pkl"
                    with artifact.open("rb") as handle:
                        candidate_fit = pickle.load(handle)
                    if getattr(candidate_fit, "stack", None) is not None:
                        reviewed_table = _pair_table_payload(
                            candidate_fit.stack, candidate_recipe.component_id
                        )
                entry.update(
                    {
                        "status": "no_change" if no_change else "review",
                        "message": (
                            "This candidate made no fitted change."
                            if no_change
                            else "Candidate table is ready to review."
                        ),
                        "review": {
                            # Both figures use training rows only.  The
                            # incumbent is the exact frozen accepted scorer,
                            # rather than the bare GLM that started the chain.
                            "incumbent_train": deepcopy(
                                source_metrics.get("train", source_metrics)
                            ),
                            "candidate_train": deepcopy(
                                candidate_metrics.get("train", candidate_metrics)
                            ),
                            "metrics": deepcopy(candidate_metrics),
                            # Native stacks do not use the legacy
                            # ``result['pair_tables']`` shape.  Extract the
                            # reviewed component's actual cells so the table
                            # is visible before Keep.
                            "pair_tables": (
                                {candidate_recipe.component_id: reviewed_table}
                                if reviewed_table is not None
                                else {}
                            ),
                            "component_evidence": deepcopy(component_evidence),
                            "warnings": deepcopy(result.get("warnings", [])),
                        },
                    }
                )
                replace_candidate(current, entry)
                job.update(
                    status="complete",
                    message=entry["message"],
                    result=result,
                )
                revision += 1
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
            if job.get("model_plan_id") == (
                current.model_plan.plan_id if current.model_plan else None
            ):
                model_id = job.get("model_plan_model_id")
                through = job.get("model_plan_through_component_id")
                if isinstance(model_id, str) and isinstance(through, str):
                    fingerprint = fitted_scorer_fingerprint(name)
                    for index, ref in enumerate(plan_refs):
                        if (
                            ref.plan_id == current.model_plan.plan_id
                            and ref.model_id == model_id
                        ):
                            plan_refs[index] = PlanFitRef(
                                ref.plan_id,
                                through,
                                name,
                                job["id"],
                                model_id=model_id,
                                prefix_fingerprint=fingerprint,
                            )
                            break
                    persist_plan_refs()
                    # This job record is the immutable accepted checkpoint used
                    # by new-data scoring.  Persist the just-issued logical
                    # ownership reference with it; the worker's fit snapshot was
                    # written before the asynchronous completion callback.
                    job["project"] = deepcopy(current)
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

    def fitted_scorer_fingerprint(name: str) -> str:
        """Fingerprint an accepted scorer, including a direct non-log GLM.

        A direct binomial/logit GLM is a valid single logical model, but it is
        not a multiplicative offset source.  It still needs a durable plan
        reference for diagnostics and export.  Use its exact reviewed rate
        model for that reference; only an append needs ``fitted_prefix`` and
        its log-link stack adapter.
        """
        source = jobs.artifact(current, name)
        with (source / "fit.pkl").open("rb") as handle:
            fitted = pickle.load(handle)  # private fitted artifact
        if fitted.name != name:
            raise ValueError("The source fit identity does not match its model.")
        if fitted.stack is None:
            from easy_glm.workflow.run import rate_model_for

            fitted = deepcopy(fitted)
            fitted.rate_model = rate_model_for(current, fitted)
            if fitted.rate_model.metadata.link != "log":
                return scorer_fingerprint(fitted.rate_model)
        return scorer_fingerprint(fitted_prefix(name))

    def logical_ref_status(
        plan: ModelPlan, index: int, model: Any
    ) -> dict[str, Any] | None:
        """Read one valid v2 logical checkpoint, including a partial one."""
        ref = next(
            (
                item
                for item in reversed(plan_refs)
                if item.plan_id == plan.plan_id and item.model_id == model.model_id
            ),
            None,
        )
        if ref is None:
            return None
        job = jobs.status(current).get(ref.model_name, {})
        if job.get("id") != ref.fit_id:
            return None
        try:
            expected = materialise_plan_through_component(
                current, plan, index, ref.stage_id, None
            )
        except ValueError:
            return None
        actual = current.models.get(ref.model_name)
        if actual is None or (
            actual != expected and replace(actual, lineage=None) != expected
        ):
            return None
        if job.get("status") not in {"complete", "queued", "running"}:
            return None
        fingerprint: str | None = None
        if job.get("status") == "complete":
            if not job.get("applicable"):
                return None
            try:
                fingerprint = fitted_scorer_fingerprint(ref.model_name)
            except (KeyError, OSError, ValueError):
                return None
            if (
                ref.prefix_fingerprint is not None
                and ref.prefix_fingerprint != fingerprint
            ):
                return None
        return {
            "model_id": model.model_id,
            "stage_id": ref.stage_id,
            "through_component_id": ref.stage_id,
            "stage_name": model.name,
            "model_name": ref.model_name,
            "fit_id": ref.fit_id,
            "prefix_fingerprint": fingerprint or ref.prefix_fingerprint,
            "job_status": job.get("status"),
            "complete_model": (
                ref.stage_id == model.physical_components[-1].component_id
            ),
        }

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
        if plan.is_logical_models_plan:
            for index, model in enumerate(plan.models):
                status = logical_ref_status(plan, index, model)
                if status is None:
                    break
                if status["job_status"] in {"queued", "running"}:
                    pending = PlanFitRef(
                        plan.plan_id,
                        status["stage_id"],
                        status["model_name"],
                        status["fit_id"],
                        model_id=model.model_id,
                        prefix_fingerprint=status["prefix_fingerprint"],
                    )
                    break
                if not status["complete_model"]:
                    # A saved frozen prefix is valid, but this logical model
                    # still has pair tables to refit after a remove/reorder.
                    attempt = plan_attempts.get(
                        (plan.plan_id, model.physical_components[-1].component_id)
                    )
                    if attempt is not None:
                        attempt_job = statuses.get(attempt.model_name, {})
                        if attempt_job.get("id") == attempt.fit_id and attempt_job.get(
                            "status"
                        ) in {"queued", "running"}:
                            pending = attempt
                    break
                completed.append(status)
            return completed, pending
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
        ref_by_model = {item.get("model_id"): item for item in refs}
        partial_by_model = (
            {
                model.model_id: status
                for index, model in enumerate(plan.models)
                if (status := logical_ref_status(plan, index, model)) is not None
                and not status["complete_model"]
                and status["job_status"] == "complete"
            }
            if plan is not None and plan.is_logical_models_plan
            else {}
        )
        statuses: list[dict[str, Any]] = []
        next_stage_id: str | None = None
        displayed = (
            [model.core for model in plan.models]
            if plan is not None and plan.is_logical_models_plan
            else (plan.stages if plan else [])
        )
        for displayed_index, stage in enumerate(displayed):
            logical_model = (
                plan.models[displayed_index]
                if plan is not None and plan.is_logical_models_plan
                else None
            )
            display_id = (
                logical_model.model_id
                if logical_model is not None
                else stage.component_id
            )
            ref = (
                ref_by_model.get(display_id)
                if logical_model is not None
                else ref_by_stage.get(stage.component_id)
            )
            if ref is not None:
                state, message, fit = "fitted", "Fitted", ref
            elif logical_model is not None and display_id in partial_by_model:
                partial = partial_by_model[display_id]
                through_index = next(
                    index
                    for index, component in enumerate(logical_model.physical_components)
                    if component.component_id == partial["through_component_id"]
                )
                remaining = len(logical_model.physical_components) - 1 - through_index
                through_name = logical_model.physical_components[through_index].name
                if pending is not None and pending.model_id == display_id:
                    state, message, fit = (
                        "fitting",
                        "Fitting the remaining interaction tables; accepted "
                        + through_name
                        + " remains available for review.",
                        {
                            **partial,
                            "pending_model_name": pending.model_name,
                            "pending_fit_id": pending.fit_id,
                        },
                    )
                else:
                    state, message, fit = (
                        "needs_refit",
                        "Accepted through "
                        + through_name
                        + f"; refit {remaining} remaining interaction table"
                        + ("s" if remaining != 1 else ""),
                        partial,
                    )
                    if next_stage_id is None and pending is None:
                        next_stage_id = display_id
            elif pending is not None and (
                (logical_model is not None and pending.model_id == display_id)
                or (logical_model is None and pending.stage_id == stage.component_id)
            ):
                state, message, fit = (
                    "fitting",
                    "Fit in progress",
                    {
                        "stage_id": display_id,
                        "model_name": pending.model_name,
                        "fit_id": pending.fit_id,
                    },
                )
            elif next_stage_id is None and pending is None:
                state, message, fit = "ready", "Ready to fit", None
                next_stage_id = display_id
            elif partial_by_model:
                state, message, fit = (
                    "needs_refit",
                    "Refit the earlier model's remaining interaction tables first",
                    None,
                )
            else:
                state, message, fit = "waiting", "Fit the earlier stage first", None
            attempt_key = (
                logical_model.physical_components[-1].component_id
                if logical_model is not None
                else stage.component_id
            )
            attempt_ref = (
                plan_attempts.get((plan.plan_id, attempt_key)) if plan else None
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
                    "stage_id": display_id,
                    "model_id": display_id if logical_model is not None else None,
                    "through_component_id": (
                        ref.get("through_component_id")
                        if ref is not None
                        else stage.component_id
                    ),
                    "name": (
                        logical_model.name if logical_model is not None else stage.name
                    ),
                    "kind": stage.kind,
                    "pair_count": (
                        len(logical_model.pair_corrections)
                        if logical_model is not None
                        else 0
                    ),
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
                # Adoption creates a new v2 plan from this frozen scorer.
                # The browser must use its actual response convention rather
                # than guessing a family or link for the child plan.
                "family": config.family,
                "link": config.link,
                "tweedie_power": config.tweedie_power,
                "divide_target_by_weight": config.divide_target_by_weight,
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
        plan_payload = plan.to_dict() if plan is not None else None
        if (
            plan is not None
            and plan.is_logical_models_plan
            and plan_payload is not None
        ):
            # Existing setup clients consume ``stages``.  Keep that display
            # projection logical: interaction tables are nested under their GLM,
            # never another planned model card.
            plan_payload["stages"] = [model.core.to_dict() for model in plan.models]
        return {
            "plan": plan_payload,
            "fit_refs": refs,
            "partial_fit_refs": list(partial_by_model.values()),
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
                "/api/datasets/upload",
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

    @app.get("/api/datasets")
    def datasets() -> dict[str, Any]:
        with lock:
            return {
                "datasets": scoring_datasets.descriptors(),
                "session_id": session_id,
                "revision": revision,
            }

    @app.post("/api/datasets/file")
    def scoring_file(edit: ScoringDatasetFile) -> dict[str, Any]:
        with lock:
            check_revision(edit)
            try:
                item = scoring_datasets.add_file(
                    edit.name, edit.path, replace_id=edit.replace_id
                )
                persist_scoring_sources()
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            return {
                "dataset": item.descriptor(),
                "session_id": session_id,
                "revision": revision,
            }

    @app.post("/api/datasets/upload")
    async def scoring_upload(
        request: Request,
        filename: str = Query(min_length=1, max_length=255),
        name: str = Query(min_length=1, max_length=255),
        session_id: str = Query(min_length=1),
        revision: int = Query(ge=0),
        replace_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        edit = Revision(session_id=session_id, revision=revision)
        with lock:
            check_revision(edit)
        if (length := request.headers.get("content-length", "")).isdigit() and int(
            length
        ) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "The upload exceeds 512 MiB.")
        folder = new_input_folder()
        source = folder / upload_basename(filename)
        received = 0
        try:
            with source.open("xb") as destination:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "The upload exceeds 512 MiB.")
                    await run_in_threadpool(destination.write, chunk)
            if not received:
                raise ValueError("Choose a non-empty scoring file.")
            with lock:
                check_revision(edit)
                item = scoring_datasets.add_upload(
                    name, filename, source, replace_id=replace_id
                )
                persist_scoring_sources()
            shutil.rmtree(folder, ignore_errors=True)
            return {
                "dataset": item.descriptor(),
                "session_id": session_id,
                "revision": revision,
            }
        except HTTPException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(422, f"Could not read scoring data: {exc}") from exc

    @app.post("/api/datasets/{dataset_id}/remove")
    def remove_scoring_dataset(dataset_id: str, edit: Revision) -> dict[str, Any]:
        with lock:
            check_revision(edit)
            try:
                scoring_datasets.remove(dataset_id)
                persist_scoring_sources()
            except ValueError as exc:
                raise HTTPException(404, str(exc)) from exc
            return {
                "removed": dataset_id,
                "session_id": session_id,
                "revision": revision,
            }

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
                _plan_ref_from_saved(loaded.model_plan, item)
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
            minted_adopted_root = False
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
                    # An old client sends no plan and retains the v1 Adopt
                    # saved model flow.  A new logical-model client supplies
                    # its empty/named v2 plan; the server deliberately mints
                    # the adopted source identity instead of trusting it from
                    # the browser.
                    requested = (
                        ModelPlan.from_dict(edit.plan)
                        if edit.plan is not None
                        else None
                    )
                    if requested is not None and requested.is_logical_models_plan:
                        if requested.adopted_root is not None:
                            raise ValueError(
                                "Only Adopt saved model may choose an adopted source."
                            )
                        if (
                            requested.family,
                            requested.link,
                            requested.tweedie_power,
                            requested.divide_target_by_weight,
                        ) != (
                            source.family,
                            source.link,
                            source.tweedie_power,
                            source.divide_target_by_weight,
                        ):
                            raise ValueError(
                                "An adopted scorer keeps its family, link, Tweedie power and response scale."
                            )
                        source_stack = fitted_prefix(edit.adopt_model_name)
                        plan = ModelPlan(
                            format_version=2,
                            mode=requested.mode,
                            family=requested.family,
                            link=requested.link,
                            tweedie_power=requested.tweedie_power,
                            divide_target_by_weight=requested.divide_target_by_weight,
                            models=deepcopy(requested.models),
                            adopted_root=AdoptedRoot(
                                model_name=edit.adopt_model_name,
                                fit_id=status["id"],
                                scorer_fingerprint=scorer_fingerprint(source_stack),
                                physical_prefix=tuple(
                                    deepcopy(component.recipe)
                                    for component in source_stack.components
                                    if component.recipe.kind != "baseline"
                                ),
                            ),
                        )
                        minted_adopted_root = True
                    else:
                        # Keep the established Adopt saved model flow as v1;
                        # opening an old project must not silently regroup it.
                        if requested is not None:
                            raise ValueError(
                                "An adopted model plan must use the logical v2 format."
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
                if (
                    plan.is_logical_models_plan
                    and plan.adopted_root is not None
                    and not (
                        minted_adopted_root
                        or (
                            old is not None
                            and old.is_logical_models_plan
                            and old.adopted_root == plan.adopted_root
                            and edit.operation in {"save", "revise"}
                        )
                    )
                ):
                    raise ValueError(
                        "Only Adopt saved model may choose an adopted source."
                    )
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
                        if plan.is_logical_models_plan and old.is_logical_models_plan:
                            if plan.adopted_root != old.adopted_root:
                                keep = 0
                            else:
                                for ref in old_refs:
                                    if keep >= len(plan.models) or keep >= len(
                                        old.models
                                    ):
                                        break
                                    old_model = old.models[keep]
                                    new_model = plan.models[keep]
                                    if (
                                        old_model != new_model
                                        or ref.get("model_id") != old_model.model_id
                                    ):
                                        break
                                    keep += 1
                        else:
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
                            model_id=ref.get("model_id"),
                            prefix_fingerprint=ref.get("prefix_fingerprint"),
                        )
                        for ref in old_refs[:keep]
                    ]
                elif edit.operation == "adopt":
                    plan_refs = (
                        []
                        if plan.is_logical_models_plan
                        else [
                            PlanFitRef(
                                plan.plan_id,
                                plan.stages[0].component_id,
                                plan.adopted_model_name or "",
                                status["id"],
                            )
                        ]
                    )
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
                            root_changed
                            or (
                                plan.models != old.models
                                if plan.is_logical_models_plan
                                and old.is_logical_models_plan
                                else plan.stages != old.stages
                            )
                        ):
                            raise ValueError(
                                "A stage is fitting. Wait for it or start a new chain."
                            )
                        frozen_prefix_changed = (
                            plan.models[: len(old_refs)] != old.models[: len(old_refs)]
                            if plan.is_logical_models_plan
                            and old.is_logical_models_plan
                            else plan.stages[: len(old_refs)]
                            != old.stages[: len(old_refs)]
                        )
                        if old_refs and (root_changed or frozen_prefix_changed):
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
            if plan is None or not plan.logical_model_count:
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
                if count > plan.logical_model_count:
                    raise ValueError("Every declared stage is already fitted.")
                prepared = apply_variables(raw, current.data)
                validate_model_plan(
                    current, plan, raw, set(prepared.columns), through_count=count
                )
                physical_prefix = (
                    [
                        component
                        for model in plan.models[:count]
                        for component in model.physical_components
                    ]
                    if plan.is_logical_models_plan
                    else plan.stages[:count]
                )
                validate_plan_catalogues(physical_prefix, require_source=True)
                name = plan_model_name(current, plan, count)
                config = materialise_plan_prefix(current, plan, count, name)
                candidate = deepcopy(current)
                candidate.models[name] = config
                source_path: Path | None = None
                source_name: str | None = None
                logical = (
                    plan.models[count - 1] if plan.is_logical_models_plan else None
                )
                partial_ref = (
                    next(
                        (
                            item
                            for item in reversed(plan_refs)
                            if item.plan_id == plan.plan_id
                            and item.model_id == logical.model_id
                        ),
                        None,
                    )
                    if logical is not None
                    else None
                )
                source_ref: dict[str, Any] | PlanFitRef | None = (
                    partial_ref
                    if partial_ref is not None
                    else (refs[-1] if refs else None)
                )
                if (
                    source_ref is None
                    and plan.is_logical_models_plan
                    and plan.adopted_root is not None
                ):
                    adopted = plan.adopted_root
                    source_job = jobs.status(current).get(adopted.model_name, {})
                    if source_job.get("id") != adopted.fit_id or not source_job.get(
                        "applicable"
                    ):
                        raise ValueError(
                            "The adopted scorer is unavailable. Adopt the saved model again."
                        )
                    source_ref = {
                        "model_name": adopted.model_name,
                        "fit_id": adopted.fit_id,
                        "prefix_fingerprint": adopted.scorer_fingerprint,
                    }
                if isinstance(source_ref, PlanFitRef):
                    source_ref = {
                        "model_name": source_ref.model_name,
                        "fit_id": source_ref.fit_id,
                        "prefix_fingerprint": source_ref.prefix_fingerprint,
                    }
                if source_ref is not None:
                    source_name = source_ref["model_name"]
                    source_path = jobs.artifact(current, source_name)
                    source_stack = fitted_prefix(source_name)
                    expected_fingerprint = source_ref.get("prefix_fingerprint")
                    if (
                        expected_fingerprint is not None
                        and scorer_fingerprint(source_stack) != expected_fingerprint
                    ):
                        raise ValueError(
                            "The recorded starting checkpoint is no longer available."
                        )
                    config.lineage = ModelLineage(
                        parent_name=source_name,
                        parent_fit_id=source_ref["fit_id"],
                        parent_key=model_key(current, source_name),
                        parent_prefix_fingerprint=scorer_fingerprint(source_stack),
                        prefix_length=len(source_stack.components),
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
                stage_id = (
                    logical.physical_components[-1].component_id
                    if logical is not None
                    else plan.stages[count - 1].component_id
                )
                jobs.jobs[name]["model_plan_stage_id"] = stage_id
                if logical is not None:
                    jobs.jobs[name]["model_plan_model_id"] = logical.model_id
                    jobs.jobs[name]["model_plan_through_component_id"] = stage_id
                issued_ref = PlanFitRef(
                    plan.plan_id,
                    stage_id,
                    name,
                    job["id"],
                    model_id=logical.model_id if logical is not None else None,
                    prefix_fingerprint="pending" if logical is not None else None,
                )
                if partial_ref is None:
                    plan_refs = [
                        item
                        for item in plan_refs
                        if not (
                            item.plan_id == plan.plan_id
                            and (
                                (
                                    logical is not None
                                    and item.model_id == logical.model_id
                                )
                                or (logical is None and item.stage_id == stage_id)
                            )
                        )
                    ]
                    plan_refs.append(issued_ref)
                plan_attempts[(plan.plan_id, stage_id)] = issued_ref
                current = candidate
                persist_plan_refs()
                revision += 1
                return {
                    "job": job,
                    "model_name": name,
                    # v2 UI addresses one logical model card, whereas the
                    # worker/ref is pinned to its final physical component.
                    # Returning the physical ID here made the client poll a
                    # status that it never renders (notably a sole GLM whose
                    # model_id differs from its core component_id).
                    "stage_id": logical.model_id if logical is not None else stage_id,
                    "snapshot": {
                        **plan_view(),
                        "revision": revision,
                        "session_id": session_id,
                    },
                }
            except (ValueError, TypeError, KeyError, OSError) as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.get("/api/core-models/{model_id}/interactions")
    def core_interactions(model_id: str) -> dict[str, Any]:
        """Accepted tables and separately persisted candidate drafts for one GLM."""
        with lock:
            try:
                return {
                    **core_interaction_view(model_id),
                    "session_id": session_id,
                    "revision": revision,
                }
            except (ValueError, KeyError, OSError) as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/core-models/{model_id}/search")
    def search_core_residuals(model_id: str, edit: CoreSearchEdit) -> dict[str, Any]:
        """Search training residuals from one exact accepted logical checkpoint.

        This route deliberately does not alter roles, selected main effects or
        interaction candidates.  Discovery is a question about the scorer the
        actuary has accepted so far; deciding to add a factor or table remains
        a separate explicit action.
        """
        with lock:
            check_revision(edit)
            try:
                plan = current.model_plan
                if plan is None:
                    raise ValueError("Define the model plan first.")
                model = _logical_model(plan, model_id)
                source = _core_source_ref(plan, model)
                saved, frame, fitted = stack_artifact(source.model_name)
                if source.fit_id != jobs.artifact(current, source.model_name).name:
                    raise ValueError(
                        "The accepted model changed. Refresh before searching."
                    )
                result = discover_core_residuals(
                    saved,
                    frame,
                    fitted,
                    action=edit.action,
                    main_factors=_core_recipe(model).factors,
                    accepted_pairs=[pair.factors for pair in _pair_recipes(model)],
                    n_bins=edit.n_bins,
                    min_expected=edit.min_expected,
                    min_cell_share=edit.min_cell_share,
                    top=edit.top,
                )
                return {
                    **result,
                    "model_id": model_id,
                    "source_model_name": source.model_name,
                    "source_fit_id": source.fit_id,
                    "through_component_id": source.stage_id,
                    "session_id": session_id,
                    "revision": revision,
                }
            except (ValueError, KeyError, OSError) as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/core-models/{model_id}/interactions")
    def create_core_interaction(
        model_id: str, edit: CoreInteractionCreate
    ) -> dict[str, Any]:
        """Create a draft only.  It has no effect on any accepted score."""
        nonlocal current, revision
        with lock:
            check_revision(edit)
            if edit.model_id != model_id:
                raise HTTPException(
                    422, "The interaction model does not match the URL."
                )
            try:
                plan = current.model_plan
                if plan is None:
                    raise ValueError("Define the model plan first.")
                model = _logical_model(plan, model_id)
                if edit.a == edit.b:
                    raise ValueError(
                        "Choose two different variables for an interaction."
                    )
                prepared = apply_variables(raw, current.data)
                groups = pair_candidate_groups(current, raw, set(prepared.columns))
                eligible = set(groups["predictors"] + groups["unassigned"])
                if edit.a not in eligible or edit.b not in eligible:
                    raise ValueError(
                        "Interaction variables must be predictor or unassigned source columns."
                    )
                existing = {frozenset(item.factors) for item in _pair_recipes(model)}
                if frozenset((edit.a, edit.b)) in existing:
                    raise ValueError(
                        "This interaction is already accepted for the model."
                    )
                source = _core_source_ref(plan, model)
                source_path = jobs.artifact(current, source.model_name)
                if source_path.name != source.fit_id:
                    raise ValueError(
                        "The accepted model changed. Refresh before adding a pair."
                    )
                candidate_id = secrets.token_urlsafe(12)
                recipe = ComponentRecipe(
                    f"{model_id}:pair:{candidate_id}",
                    f"{edit.a} × {edit.b}",
                    "interaction",
                    (edit.a, edit.b),
                    {"tune": True, **deepcopy(edit.settings)},
                    design=deepcopy(edit.design),
                )
                entry = {
                    "candidate_id": candidate_id,
                    "model_id": model_id,
                    "a": edit.a,
                    "b": edit.b,
                    "recipe": recipe.to_dict(),
                    "source_model_name": source.model_name,
                    "source_fit_id": source.fit_id,
                    "source_through_component_id": source.stage_id,
                    "source_fingerprint": scorer_fingerprint(
                        fitted_prefix(source.model_name)
                    ),
                    "status": "draft",
                    "message": "Not included in the accepted score.",
                    "created_at": str(time.time_ns()),
                }
                replace_candidate(current, entry)
                revision += 1
                return {
                    **core_interaction_view(model_id),
                    "candidate": public_candidate(entry),
                    "session_id": session_id,
                    "revision": revision,
                }
            except (ValueError, KeyError, OSError) as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post(
        "/api/core-models/{model_id}/interactions/{candidate_id}/fit", status_code=202
    )
    def fit_core_interaction(
        model_id: str, candidate_id: str, edit: CoreInteractionCandidateEdit
    ) -> dict[str, Any]:
        """Fit one candidate against the exact accepted source scorer.

        The temporary worker job is intentionally not a saved project model.
        Its artefact is reachable only through this candidate until Keep.
        """
        nonlocal revision
        with lock:
            check_revision(edit)
            if edit.candidate_id != candidate_id:
                raise HTTPException(
                    422, "The interaction candidate does not match the URL."
                )
            try:
                entry = _candidate_entry(candidate_id)
                if entry.get("model_id") != model_id:
                    raise ValueError("This candidate belongs to a different model.")
                if entry.get("status") not in {"draft", "needs_rebuild", "failed"}:
                    raise ValueError(
                        "This candidate is already fitting or ready for review."
                    )
                if not _candidate_is_current(entry):
                    entry.update(
                        status="needs_rebuild",
                        message="The accepted source changed; create a fresh candidate.",
                    )
                    replace_candidate(current, entry)
                    revision += 1
                    raise ValueError(entry["message"])
                plan = current.model_plan
                assert plan is not None
                source_name = entry["source_model_name"]
                if source_name not in current.models:
                    raise ValueError(
                        "The accepted source configuration is unavailable."
                    )
                recipe = ComponentRecipe.from_dict(entry["recipe"])
                # A private worker name means the candidate cannot become a
                # default result, export or downstream offset by accident.
                job_name = f"__interaction_candidate__{candidate_id}"
                candidate_project = deepcopy(current)
                candidate_plan = candidate_project.model_plan
                assert candidate_plan is not None
                candidate_index = next(
                    index
                    for index, item in enumerate(candidate_plan.models)
                    if item.model_id == model_id
                )
                candidate_plan.models[candidate_index] = replace(
                    candidate_plan.models[candidate_index],
                    pair_corrections=(
                        *candidate_plan.models[candidate_index].pair_corrections,
                        recipe,
                    ),
                )
                candidate_plan.stages = [
                    component
                    for item in candidate_plan.models
                    for component in item.physical_components
                ]
                # Compile from the logical plan.  This keeps the accepted
                # checkpoint configuration identical to the normal plan-fit
                # path after Keep, including a direct GLM's one-time native
                # stack wrapping when it gains its first interaction.
                candidate_cfg = materialise_plan_prefix(
                    candidate_project,
                    candidate_plan,
                    candidate_index + 1,
                    job_name,
                )
                candidate_project.models[job_name] = candidate_cfg
                source_path = jobs.artifact(current, source_name)
                job = jobs.start(
                    candidate_project,
                    raw.clone(),
                    job_name,
                    source_fit=source_path,
                    frozen_source_name=source_name,
                )
                jobs.jobs[job_name]["core_interaction_candidate_id"] = candidate_id
                entry.update(
                    {
                        "status": "fitting",
                        "message": "Fitting on training rows against the accepted scorer.",
                        "job_name": job_name,
                        "job_id": job["id"],
                    }
                )
                replace_candidate(current, entry)
                revision += 1
                return {
                    "job": job,
                    "candidate": public_candidate(entry),
                    "session_id": session_id,
                    "revision": revision,
                }
            except (ValueError, KeyError, OSError) as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/core-models/{model_id}/interactions/{candidate_id}/discard")
    def discard_core_interaction(
        model_id: str, candidate_id: str, edit: CoreInteractionCandidateEdit
    ) -> dict[str, Any]:
        nonlocal revision
        with lock:
            check_revision(edit)
            try:
                entry = _candidate_entry(candidate_id)
                if entry.get("model_id") != model_id:
                    raise ValueError("This candidate belongs to a different model.")
                job_name = entry.get("job_name")
                if isinstance(job_name, str):
                    jobs.cancel(job_name, "Interaction candidate discarded")
                remove_candidate(current, candidate_id)
                revision += 1
                return {
                    **core_interaction_view(model_id),
                    "session_id": session_id,
                    "revision": revision,
                }
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/core-models/{model_id}/interactions/{candidate_id}/accept")
    def accept_core_interaction(
        model_id: str, candidate_id: str, edit: CoreInteractionCandidateEdit
    ) -> dict[str, Any]:
        """Promote the exact reviewed candidate artifact into a new checkpoint."""
        nonlocal current, revision, plan_refs
        with lock, jobs.lock:
            check_revision(edit)
            try:
                entry = _candidate_entry(candidate_id)
                if entry.get("model_id") != model_id:
                    raise ValueError("This candidate belongs to a different model.")
                if entry.get("status") != "review":
                    raise ValueError(
                        "Fit and review this interaction before keeping it."
                    )
                if not _candidate_is_current(entry):
                    entry.update(
                        status="needs_rebuild",
                        message="The accepted source changed; this reviewed table cannot be kept.",
                    )
                    replace_candidate(current, entry)
                    revision += 1
                    raise ValueError(entry["message"])
                job_name = entry.get("job_name")
                job_id = entry.get("job_id")
                if not isinstance(job_name, str) or not isinstance(job_id, str):
                    raise ValueError("The reviewed candidate fit is unavailable.")
                job = jobs.jobs.get(job_name)
                if (
                    job is None
                    or job.get("id") != job_id
                    or job.get("status") != "complete"
                ):
                    raise ValueError("The reviewed candidate fit is unavailable.")
                source = Path(jobs.folder.name) / job_id
                if not (source / "fit.pkl").is_file():
                    raise ValueError("The reviewed candidate artifact is unavailable.")
                plan = current.model_plan
                if plan is None:
                    raise ValueError("The model plan is unavailable.")
                model = _logical_model(plan, model_id)
                recipe = ComponentRecipe.from_dict(entry["recipe"])
                # A neutral candidate is review evidence, not an accepted table.
                with (source / "fit.pkl").open("rb") as handle:
                    fitted = pickle.load(handle)
                if fitted.stack is None or not fitted.stack.components:
                    raise ValueError("The reviewed interaction scorer is unavailable.")
                final_component = fitted.stack.components[-1]
                if final_component.recipe.component_id != recipe.component_id:
                    raise ValueError(
                        "The reviewed artifact does not match this candidate."
                    )
                if final_component.evidence.get("no_change"):
                    entry.update(
                        status="no_change",
                        message="This candidate made no fitted change.",
                    )
                    replace_candidate(current, entry)
                    revision += 1
                    raise ValueError(entry["message"])

                # ``PlannedModel`` is frozen so replace just this nested value;
                # older accepted checkpoints and every downstream recipe stay
                # intact.  They become a stale suffix through their mismatched
                # immutable prefix, never by mutating their scorer.
                from dataclasses import replace as dataclass_replace

                updated_model = dataclass_replace(
                    model,
                    pair_corrections=(*model.pair_corrections, recipe),
                )
                plan.models = [
                    updated_model if item.model_id == model_id else item
                    for item in plan.models
                ]
                plan.stages = [
                    component
                    for item in plan.models
                    for component in item.physical_components
                ]
                candidate_project = deepcopy(current)
                candidate_project.model_plan = plan
                accepted_index = plan.models.index(updated_model) + 1
                accepted_name = plan_model_name(current, plan, accepted_index)
                accepted_cfg = materialise_plan_prefix(
                    candidate_project, plan, accepted_index, accepted_name
                )
                candidate_project.models[accepted_name] = accepted_cfg

                # Re-label the same private artifact rather than re-fitting or
                # reconstructing it.  This is the exact reviewed scorer.
                fitted.name = accepted_name
                fitted.config = accepted_cfg
                fitted.project_snapshot = candidate_project.to_dict()
                with (source / "fit.pkl").open("wb") as handle:
                    pickle.dump(fitted, handle, protocol=5)
                accepted_fingerprint = scorer_fingerprint(fitted.stack)
                job["name"] = accepted_name
                job["key"] = model_key(candidate_project, accepted_name)
                job["project"] = deepcopy(candidate_project)
                job.pop("core_interaction_candidate_id", None)
                jobs.jobs.pop(job_name, None)
                jobs.jobs[accepted_name] = job
                jobs.last_complete[accepted_name] = deepcopy(job)

                through = recipe.component_id
                changed_index = plan.models.index(updated_model)
                affected = {item.model_id for item in plan.models[changed_index:]}
                plan_refs = [
                    item
                    for item in plan_refs
                    if not (item.plan_id == plan.plan_id and item.model_id in affected)
                ]
                plan_refs.append(
                    PlanFitRef(
                        plan.plan_id,
                        through,
                        accepted_name,
                        job_id,
                        model_id=model_id,
                        prefix_fingerprint=accepted_fingerprint,
                    )
                )
                current = candidate_project
                # Restore the now-updated logical plan into the accepted project.
                current.model_plan = plan
                persist_plan_refs()
                remove_candidate(current, candidate_id)
                revision += 1
                return {
                    **core_interaction_view(model_id),
                    "accepted": {
                        "model_name": accepted_name,
                        "fit_id": job_id,
                        "through_component_id": through,
                    },
                    "needs_refit": [
                        item.name for item in plan.models[changed_index + 1 :]
                    ],
                    "session_id": session_id,
                    "revision": revision,
                }
            except (ValueError, KeyError, OSError) as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/core-models/{model_id}/interactions/{component_id}/remove")
    def remove_accepted_core_interaction(
        model_id: str, component_id: str, edit: CoreInteractionRecipeEdit
    ) -> dict[str, Any]:
        """Revise a GLM's working recipe without altering historical scorers."""
        nonlocal revision, plan_refs
        with lock:
            check_revision(edit)
            try:
                plan = current.model_plan
                if plan is None:
                    raise ValueError("The model plan is unavailable.")
                model = _logical_model(plan, model_id)
                original_pairs = tuple(model.pair_corrections)
                removed_index = next(
                    (
                        index
                        for index, pair in enumerate(original_pairs)
                        if pair.component_id == component_id
                    ),
                    None,
                )
                if removed_index is None:
                    raise ValueError("Choose an accepted interaction table.")
                prefix_ref = _checkpoint_for_pair_prefix(
                    plan, model, original_pairs[:removed_index]
                )
                retained = tuple(
                    pair for pair in original_pairs if pair.component_id != component_id
                )
                from dataclasses import replace as dataclass_replace

                revised = dataclass_replace(model, pair_corrections=retained)
                changed_index = plan.models.index(model)
                plan.models = [
                    revised if item.model_id == model_id else item
                    for item in plan.models
                ]
                plan.stages = [
                    component
                    for item in plan.models
                    for component in item.physical_components
                ]
                # The source before the removed table is already an immutable
                # scored checkpoint.  Keep that prefix and invalidate only the
                # removed-table suffix and later logical models.
                affected = {item.model_id for item in plan.models[changed_index:]}
                plan_refs = [
                    item
                    for item in plan_refs
                    if not (item.plan_id == plan.plan_id and item.model_id in affected)
                ]
                plan_refs.append(prefix_ref)
                persist_plan_refs()
                revision += 1
                return {
                    **core_interaction_view(model_id),
                    "needs_refit": [item.name for item in plan.models[changed_index:]],
                    "session_id": session_id,
                    "revision": revision,
                }
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/core-models/{model_id}/interactions/{component_id}/move")
    def move_accepted_core_interaction(
        model_id: str, component_id: str, edit: CoreInteractionMove
    ) -> dict[str, Any]:
        """Reorder tables by creating a revised, explicitly-unfitted suffix."""
        nonlocal revision, plan_refs
        with lock:
            check_revision(edit)
            try:
                plan = current.model_plan
                if plan is None:
                    raise ValueError("The model plan is unavailable.")
                model = _logical_model(plan, model_id)
                original_pairs = tuple(model.pair_corrections)
                pairs = list(original_pairs)
                moved = next(
                    (pair for pair in pairs if pair.component_id == component_id), None
                )
                if moved is None:
                    raise ValueError("Choose an accepted interaction table.")
                pairs.remove(moved)
                if edit.before_component_id is None:
                    pairs.append(moved)
                else:
                    index = next(
                        (
                            i
                            for i, pair in enumerate(pairs)
                            if pair.component_id == edit.before_component_id
                        ),
                        None,
                    )
                    if index is None:
                        raise ValueError(
                            "Choose another accepted interaction as the destination."
                        )
                    pairs.insert(index, moved)
                first_changed = next(
                    (
                        index
                        for index, (before, after) in enumerate(
                            zip(original_pairs, pairs, strict=True)
                        )
                        if before.component_id != after.component_id
                    ),
                    len(pairs),
                )
                if first_changed == len(pairs):
                    raise ValueError("This move does not change the table order.")
                prefix_ref = _checkpoint_for_pair_prefix(
                    plan, model, original_pairs[:first_changed]
                )
                from dataclasses import replace as dataclass_replace

                revised = dataclass_replace(model, pair_corrections=tuple(pairs))
                changed_index = plan.models.index(model)
                plan.models = [
                    revised if item.model_id == model_id else item
                    for item in plan.models
                ]
                plan.stages = [
                    component
                    for item in plan.models
                    for component in item.physical_components
                ]
                affected = {item.model_id for item in plan.models[changed_index:]}
                plan_refs = [
                    item
                    for item in plan_refs
                    if not (item.plan_id == plan.plan_id and item.model_id in affected)
                ]
                plan_refs.append(prefix_ref)
                persist_plan_refs()
                revision += 1
                return {
                    **core_interaction_view(model_id),
                    "needs_refit": [item.name for item in plan.models[changed_index:]],
                    "session_id": session_id,
                    "revision": revision,
                }
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    def _amended_pair_checkpoint(
        model_id: str, component_id: str, edit: StackAmendEdit
    ) -> tuple[ModelPlan, Any, Any, Any, dict[str, Any], int]:
        """Preview an accepted pair edit and return its exact new prefix.

        ``amend_stack`` intentionally leaves later components in place for a
        generic stack preview.  A core-table amendment is different: its later
        pair tables are no longer fitted against the amended source.  Publish
        only the scorer through the amended table, then let the normal staged
        fit re-create that suffix.
        """
        plan = current.model_plan
        if plan is None:
            raise ValueError("The model plan is unavailable.")
        model = _logical_model(plan, model_id)
        pair_index = next(
            (
                index
                for index, pair in enumerate(_pair_recipes(model))
                if pair.component_id == component_id
            ),
            None,
        )
        if pair_index is None:
            raise ValueError("Choose an accepted interaction table.")
        source = _core_source_ref(plan, model)
        saved, frame, fitted, amended, detail = preview_stack_amendment(
            source.model_name, component_id, edit
        )
        amended_recipe = next(
            (
                recipe
                for recipe in amended.config.components
                if recipe.component_id == component_id
            ),
            None,
        )
        if amended_recipe is None:
            raise ValueError("The amended interaction table is unavailable.")
        updated_pairs = list(model.pair_corrections)
        updated_pairs[pair_index] = amended_recipe
        revised_model = replace(model, pair_corrections=tuple(updated_pairs))
        revised_plan = deepcopy(plan)
        model_index = plan.models.index(model)
        revised_plan.models[model_index] = revised_model
        revised_plan.stages = [
            component
            for item in revised_plan.models
            for component in item.physical_components
        ]
        # Keep exactly the amended prefix.  The untouched suffix remains in
        # the working plan, but has no accepted fit reference until refitted.
        partial_config = materialise_plan_through_component(
            current, revised_plan, model_index, component_id
        )
        amended = replace(
            amended,
            config=partial_config,
            stack=amended.stack.prefix_through(component_id),
        )
        return revised_plan, saved, frame, amended, detail, model_index

    @app.post("/api/core-models/{model_id}/interactions/{component_id}/preview")
    def preview_accepted_core_interaction(
        model_id: str, component_id: str, edit: StackAmendEdit
    ) -> dict[str, Any]:
        """Preview cap/rounding for a table without changing its checkpoint."""
        with lock:
            check_revision(edit)
            try:
                _plan, _saved, _frame, _amended, detail, _index = (
                    _amended_pair_checkpoint(model_id, component_id, edit)
                )
                return {
                    "preview": detail,
                    "model_id": model_id,
                    "component_id": component_id,
                    "session_id": session_id,
                    "revision": revision,
                }
            except (ValueError, KeyError, OSError) as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/core-models/{model_id}/interactions/{component_id}/apply")
    def apply_accepted_core_interaction(
        model_id: str, component_id: str, edit: StackAmendEdit
    ) -> dict[str, Any]:
        """Publish an amended pair prefix and invalidate only its suffix."""
        nonlocal current, revision, plan_refs
        from easy_glm.desktop.fit_worker import json_safe
        from easy_glm.desktop.stack_views import stack_result
        from easy_glm.workflow.project import validate_model_name

        with lock:
            check_revision(edit)
            try:
                if not edit.new_name:
                    raise ValueError("Name the reviewed model checkpoint.")
                plan, saved, frame, amended, detail, changed_index = (
                    _amended_pair_checkpoint(model_id, component_id, edit)
                )
                problem = validate_model_name(edit.new_name, current.models)
                if problem:
                    raise ValueError(problem)
                candidate = deepcopy(current)
                candidate.model_plan = plan
                candidate.models[edit.new_name] = amended.config
                amended.name = edit.new_name
                amended.project_snapshot = candidate.to_dict()
                result = json_safe(stack_result(candidate, frame, amended))
                published = jobs.publish_stack(
                    candidate, raw.clone(), edit.new_name, amended, result
                )
                fingerprint = scorer_fingerprint(amended.stack)
                changed_model = plan.models[changed_index]
                affected = {item.model_id for item in plan.models[changed_index:]}
                plan_refs = [
                    item
                    for item in plan_refs
                    if not (item.plan_id == plan.plan_id and item.model_id in affected)
                ]
                plan_refs.append(
                    PlanFitRef(
                        plan.plan_id,
                        component_id,
                        edit.new_name,
                        published["id"],
                        model_id=changed_model.model_id,
                        prefix_fingerprint=fingerprint,
                    )
                )
                current = candidate
                persist_plan_refs()
                revision += 1
                return {
                    **core_interaction_view(model_id),
                    "job": published,
                    "preview": detail,
                    "needs_refit": [item.name for item in plan.models[changed_index:]],
                    "session_id": session_id,
                    "revision": revision,
                }
            except (ValueError, KeyError, OSError) as exc:
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

    def frozen_artifact(fit_id: str) -> tuple[str, Project, Any]:
        """Read one immutable private fit without consulting the live plan/data."""
        with lock, jobs.lock:
            matches = [
                (name, job)
                for name, job in jobs.jobs.items()
                if job.get("id") == fit_id
                and job.get("status") == "complete"
                and jobs._artifact_available(job)
            ]
        if len(matches) != 1:
            raise ValueError("The selected saved fit is no longer available.")
        name, job = matches[0]
        source = Path(jobs.folder.name) / fit_id
        with (source / "fit.pkl").open("rb") as handle:
            fitted = pickle.load(handle)
        if fitted.name != name:
            raise ValueError("The fitted artifact does not match its saved identity.")
        recorded = job.get("project")
        snapshot = getattr(fitted, "project_snapshot", None)
        # ``job.project`` is updated at publication with server-issued logical
        # fit references.  It is otherwise the same fit-time copy as the worker
        # snapshot and remains immutable when the live draft changes.
        saved = (
            deepcopy(recorded)
            if isinstance(recorded, Project)
            else (Project.from_dict(snapshot) if isinstance(snapshot, dict) else None)
        )
        if not isinstance(saved, Project):
            raise ValueError("This older fit has no saved project snapshot.")
        if getattr(fitted, "stack", None) is None:
            from easy_glm.workflow.run import rate_model_for
            from easy_glm.workflow.stack import stack_from_legacy

            fitted = deepcopy(fitted)
            fitted.rate_model = rate_model_for(saved, fitted)
            fitted.config = saved.models.get(name, fitted.config)
            if fitted.rate_model.metadata.link == "log":
                fitted.stack = stack_from_legacy(saved, fitted)
        return name, saved, fitted

    def stack_artifact(name: str) -> tuple[Project, pl.DataFrame, Any]:
        with lock:
            try:
                jobs.artifact(current, name)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            # The result has to be inspected against the project that fitted
            # it.  The live workbench may have since moved a table, renamed a
            # stage or changed bands.  Reusing ``current`` here made a saved
            # checkpoint silently borrow those later settings.
            saved_raw = raw.clone()
        with lock, jobs.lock:
            fit_id = jobs.jobs[name]["id"]
        try:
            _frozen_name, saved, fitted = frozen_artifact(fit_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if getattr(fitted, "stack", None) is None:
            # A standalone non-log fit is scoreable through the new-data API as
            # a RateModel, but cannot be represented as a multiplicative stack
            # for the historical stack diagnostics/results view.
            raise HTTPException(
                404, "This fitted model is not a compatible component stack."
            )
        from easy_glm.workflow.prep import prepare

        frame = prepare(saved, saved_raw)
        return saved, frame, fitted

    def scoring_model(
        model_id: str,
    ) -> tuple[str, Project, Any, str, list[dict[str, Any]], str]:
        """Resolve an opaque public ID using the fit's own saved plan snapshot."""
        try:
            fit_id, logical_id = model_id.split(":", 1)
        except ValueError:
            fit_id, logical_id = model_id, ""
        name, saved, fitted = frozen_artifact(fit_id)
        stack = getattr(fitted, "stack", None)
        plan = saved.model_plan
        if logical_id and plan is not None and plan.is_logical_models_plan:
            logical = _logical_model(plan, logical_id)
            references = [
                item
                for item in plan.fit_refs
                if item.get("model_id") == logical_id
                and item.get("model_name") == name
                and item.get("fit_id") == fit_id
            ]
            if len(references) != 1:
                raise ValueError(
                    "This frozen logical model is not available from the saved fit."
                )
            through = references[0]["through_component_id"]
            physical = list(logical.physical_components)
            endpoint = next(
                (
                    index
                    for index, item in enumerate(physical)
                    if item.component_id == through
                ),
                None,
            )
            if endpoint is None:
                raise ValueError(
                    "The saved fit has an invalid logical ownership checkpoint."
                )
            return (
                name,
                saved,
                fitted,
                through,
                [item.to_dict() for item in physical[: endpoint + 1]],
                logical.name,
            )
        components = (
            []
            if stack is None
            else [item.recipe.to_dict() for item in stack.diagnostic_components()]
        )
        return name, saved, fitted, "", components, name

    @app.get("/api/scoring/models")
    def scoring_models() -> dict[str, Any]:
        with lock:
            scope_labels = [
                {
                    "id": "full",
                    "label": "Complete accepted score",
                    "description": "All accepted components.",
                },
                {
                    "id": "through",
                    "label": "Accepted checkpoint",
                    "description": "Score through a physical component.",
                },
                {
                    "id": "model",
                    "label": "One model component",
                    "description": "Inspect one accepted component.",
                },
                {
                    "id": "catalogue",
                    "label": "Catalogue candidate",
                    "description": "Native catalogue output for review only.",
                },
            ]
            models: list[dict[str, Any]] = []
            with jobs.lock:
                fit_ids = [
                    job["id"]
                    for job in jobs.jobs.values()
                    if jobs._artifact_available(job)
                ]
            for fit_id in fit_ids:
                try:
                    name, saved, fitted = frozen_artifact(fit_id)
                except ValueError:
                    continue
                plan = saved.model_plan
                choices: list[tuple[str, str, list[dict[str, Any]], str]] = []
                if plan is not None and plan.is_logical_models_plan:
                    for reference in plan.fit_refs:
                        if (
                            reference.get("model_name") != name
                            or reference.get("fit_id") != fit_id
                        ):
                            continue
                        try:
                            _n, _s, _f, through, components, label = scoring_model(
                                f"{fit_id}:{reference['model_id']}"
                            )
                        except ValueError:
                            continue
                        choices.append(
                            (reference["model_id"], through, components, label)
                        )
                if not choices:
                    choices = [
                        (
                            "",
                            "",
                            (
                                []
                                if getattr(fitted, "stack", None) is None
                                else [
                                    item.recipe.to_dict()
                                    for item in fitted.stack.diagnostic_components()
                                ]
                            ),
                            name,
                        )
                    ]
                for logical_id, through, components, label in choices:
                    public_scorer = getattr(fitted, "stack", None) or fitted.rate_model
                    public_link = (
                        public_scorer.link
                        if getattr(fitted, "stack", None) is not None
                        else fitted.rate_model.metadata.link
                    )
                    saved_config = saved.models.get(name)
                    preferred_exposure = (
                        saved_config.weight
                        if saved_config is not None and public_link != "logit"
                        else None
                    )
                    catalogue_components = [
                        item
                        for item in (
                            fitted.stack.components
                            if getattr(fitted, "stack", None) is not None
                            else []
                        )
                        if item.recipe.component_id
                        in {component["component_id"] for component in components}
                        and item.recipe.kind == "catalogue"
                    ]
                    native_available = any(
                        getattr(item.scorer, "native_scorer", None) is not None
                        for item in catalogue_components
                    )
                    scopes = scope_labels[:3] if components else scope_labels[:1]
                    if catalogue_components:
                        scopes = [*scopes, scope_labels[3]]
                    models.append(
                        {
                            "id": f"{fit_id}:{logical_id}" if logical_id else fit_id,
                            "name": label,
                            "fit_id": fit_id,
                            "kind": "logical" if logical_id else "physical",
                            "scopes": scopes,
                            "through": through or None,
                            "through_components": components,
                            "components": components,
                            "owned_component_ids": [
                                item["component_id"] for item in components
                            ],
                            "root_component_id": (
                                components[0]["component_id"] if components else None
                            ),
                            "external_offset_column": (
                                public_scorer.offset_col
                                if getattr(fitted, "stack", None) is not None
                                else fitted.rate_model.metadata.offset_col
                            ),
                            "exposure_column": preferred_exposure,
                            "allowed_exposure_columns": (
                                [preferred_exposure] if preferred_exposure else []
                            ),
                            "probability": public_link == "logit",
                            "output_basis": (
                                "probability" if public_link == "logit" else "rate"
                            ),
                            "catalogue_refresh_available": native_available,
                            "catalogue_refresh_unavailable": bool(catalogue_components)
                            and not native_available,
                        }
                    )
            return {"models": models, "session_id": session_id, "revision": revision}

    def score_request(
        edit: ScoringRequest,
    ) -> tuple[Any, pl.DataFrame, Project, dict[str, Any]]:
        """Pin all mutable choices before CPU work; callers never score a draft."""
        with lock:
            check_revision(edit)
            name, saved, fitted, accepted_through, components, label = scoring_model(
                edit.model_id
            )
            item, scoring_frame = scoring_datasets.get(edit.dataset_id)
            if edit.scope == "catalogue":
                known = {str(component["component_id"]) for component in components}
                if len(edit.component_ids) != 1 or edit.component_ids[0] not in known:
                    raise ValueError(
                        "Choose one component owned by this accepted model."
                    )
                if not any(
                    item.recipe.component_id == edit.component_ids[0]
                    and item.recipe.kind == "catalogue"
                    for item in (
                        fitted.stack.components
                        if getattr(fitted, "stack", None) is not None
                        else []
                    )
                ):
                    raise ValueError(
                        "Catalogue scoring requires an owned catalogue component."
                    )
            elif edit.scope == "model":
                known = {str(component["component_id"]) for component in components}
                logical_choice = ":" in edit.model_id
                if not edit.component_ids and logical_choice:
                    edit.component_ids = [
                        str(component["component_id"]) for component in components
                    ]
                if not edit.component_ids and not logical_choice:
                    raise ValueError("Choose one physical component to inspect.")
                ordered = [str(component["component_id"]) for component in components]
                selected = [
                    ordered.index(component_id)
                    for component_id in edit.component_ids
                    if component_id in known
                ]
                if (
                    not selected
                    or len(selected) != len(edit.component_ids)
                    or selected != list(range(min(selected), max(selected) + 1))
                    or (not logical_choice and len(selected) != 1)
                ):
                    raise ValueError("Choose accepted components owned by this model.")
            elif edit.component_ids:
                known = {str(component["component_id"]) for component in components}
                if not set(edit.component_ids).issubset(known):
                    raise ValueError(
                        "A selected component is not owned by this accepted model."
                    )
            if edit.scope == "through":
                if not edit.through:
                    raise ValueError("Choose the accepted component to score through.")
                if edit.through not in {
                    str(component["component_id"]) for component in components
                }:
                    raise ValueError(
                        "The selected checkpoint is not owned by this accepted model."
                    )
            elif edit.through is not None:
                raise ValueError(
                    "A checkpoint is only valid when scoring through a component."
                )
            scorer = getattr(fitted, "stack", None) or fitted.rate_model
            if not edit.column_prefix.strip() or edit.column_prefix == "easyglm_":
                suffix = (
                    "full_chain"
                    if edit.scope == "full"
                    else safe_filename(label, "model")
                    + ("_candidate" if edit.scope == "catalogue" else "")
                )
                edit.column_prefix = f"easyglm_{suffix}_"
            saved_config = saved.models.get(name)
            allowed_exposure = (
                saved_config.weight
                if saved_config is not None and getattr(scorer, "link", None) != "logit"
                else None
            )
            if edit.exposure_col is not None and edit.exposure_col != allowed_exposure:
                raise ValueError(
                    "Expected amounts can only use this saved model's exposure column."
                )
            pinned = {
                "dataset": item.descriptor(),
                "model_name": name,
                "fit_id": edit.model_id.split(":", 1)[0],
                "accepted_through": accepted_through or None,
                "preparation_id": hashlib.sha256(
                    json.dumps(saved.to_dict()["data"], sort_keys=True).encode()
                ).hexdigest(),
                "scorer_id": scorer_fingerprint(scorer),
            }
        return scorer, scoring_frame, saved, pinned

    def run_score(edit: ScoringRequest) -> tuple[Any, dict[str, Any]]:
        from easy_glm.workflow.scoring import score_data

        scorer, data, saved, pinned = score_request(edit)
        result = score_data(
            scorer,
            data,
            project=saved,
            scope=edit.scope,
            component_ids=tuple(edit.component_ids),
            through=edit.through,
            exposure_col=edit.exposure_col,
            column_prefix=edit.column_prefix,
        )
        return result, pinned

    @app.post("/api/scoring/check")
    def scoring_check(edit: ScoringRequest) -> dict[str, Any]:
        from easy_glm.workflow.scoring import validate_scoring_data

        try:
            scorer, data, saved, pinned = score_request(edit)
            check = validate_scoring_data(
                scorer,
                data,
                project=saved,
                scope=edit.scope,
                component_ids=tuple(edit.component_ids),
                through=edit.through,
                exposure_col=edit.exposure_col,
                column_prefix=edit.column_prefix,
            )
            return {
                **check,
                "pinned": pinned,
                "session_id": edit.session_id,
                "revision": edit.revision,
            }
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/scoring/preview")
    def scoring_preview(edit: ScoringRequest) -> dict[str, Any]:
        from easy_glm.desktop.fit_worker import json_safe

        try:
            result, pinned = run_score(edit)
            result.metadata["pinned"] = pinned
            return json_safe(
                {
                    "metadata": {**result.metadata, "pinned": pinned},
                    "issues": result.issues,
                    "columns": list(result.frame.columns),
                    "rows": result.preview(20).to_dicts(),
                    "total_rows": result.frame.height,
                    "warnings": result.warnings,
                    "session_id": edit.session_id,
                    "revision": edit.revision,
                }
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/scoring/export")
    def scoring_export(
        edit: ScoringRequest, format: Literal["csv", "parquet"] = "csv"
    ) -> Response:
        try:
            result, pinned = run_score(edit)
            result.metadata["pinned"] = pinned
            suffix = ".csv" if format == "csv" else ".parquet"
            folder = Path(tempfile.mkdtemp(prefix="easyglm_scoring_export_"))
            filename = f"{safe_filename(pinned['dataset']['name'], 'scoring')}_{safe_filename(pinned['model_name'])}{suffix}"
            path = folder / filename
            result.export(path)
            media = "text/csv" if format == "csv" else "application/vnd.apache.parquet"
            return FileResponse(
                path,
                media_type=media,
                filename=filename,
                background=BackgroundTask(shutil.rmtree, folder, ignore_errors=True),
                headers={"X-EasyGLM-Scoring-Metadata": json.dumps(result.metadata)},
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

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
        from easy_glm.workflow.stack_types import LookupScorer, RateModelScorer

        with lock:
            check_revision(edit)
            source = jobs.artifact(current, name)
            if edit.fit_id is not None and source.name != edit.fit_id:
                raise ValueError("The fitted model changed. Preview this edit again.")
        saved, frame, fitted = stack_artifact(name)
        component = fitted.stack.component(component_id)
        is_lookup = isinstance(component.scorer, LookupScorer)
        is_pair = (
            isinstance(component.scorer, RateModelScorer)
            and component.recipe.kind == "interaction"
        )
        if not (is_lookup or is_pair):
            raise ValueError(
                "Caps and rounding are available for catalogue and interaction tables."
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
        if is_lookup:
            old_table = component.scorer.table
            new_table = amended.stack.component(component_id).scorer.table
            old_values = old_table["multiplier"].cast(pl.Float64).to_numpy()
            new_values = new_table["multiplier"].cast(pl.Float64).to_numpy()
            keys = old_table[component.scorer.key].to_list()
        else:
            old_pair = component.scorer.rate_model.get_pair_table(component_id)
            new_pair = amended.stack.component(
                component_id
            ).scorer.rate_model.get_pair_table(component_id)
            old_values = np.asarray(
                [cell.relativity for cell in old_pair.cells], dtype=float
            )
            new_values = np.asarray(
                [cell.relativity for cell in new_pair.cells], dtype=float
            )
            keys = [f"{cell.axis_a_row}, {cell.axis_b_row}" for cell in old_pair.cells]
        _actual, old_expected, _weight = totals(
            train, fitted.config, fitted.predict(train)
        )
        _actual, new_expected, _weight = totals(
            train, amended.config, amended.predict(train)
        )
        changed = np.flatnonzero(~np.isclose(old_values, new_values, rtol=0, atol=0))
        detail = {
            "component_id": component_id,
            "changed_rows": int(changed.size),
            "expected_before": float(old_expected.sum()),
            "expected_after": float(new_expected.sum()),
            "sample_rows": [
                {
                    "key": keys[int(index)],
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
