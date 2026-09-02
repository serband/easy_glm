"""Session state for the workbench: one Project, cached frames and runs.

All pages go through these helpers so that
* the :class:`Project` in ``st.session_state`` is the single source of truth,
* expensive artefacts (raw frame, prepared frame, model runs, leakage report)
  are cached on a hash of the part of the spec they depend on, and
* the project is autosaved after every change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

from easy_glm.workflow import (
    ModelRun,
    Project,
    leakage_report,
    load_source,
    prepare,
    rebuild_rate_model,
    run_model,
)

PROJECT_SUFFIX = ".easyglm-project.json"


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------
def spec_hash(obj: Any) -> str:
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def data_hash(project: Project) -> str:
    d = project.to_dict()["data"]
    return spec_hash(d)


def source_hash(project: Project) -> str:
    d = project.to_dict()["data"]
    return spec_hash(
        {
            "source": d["source"],
            "sample_rows": d["sample_rows"],
            "sample_seed": d["sample_seed"],
        }
    )


def model_hash(project: Project, model: str) -> str:
    d = project.to_dict()
    cfg = dict(d["models"][model])
    cfg.pop("adjustments", None)  # applied post-fit, never require a refit
    cfg.pop("base_rate_override", None)
    cfg.pop("notes", None)
    design = {v: d["design"]["variables"].get(v) for v in cfg["predictors"]}
    return spec_hash(
        {
            "data": d["data"],
            "design_defaults": d["design"]["defaults"],
            "design": design,
            "model": cfg,
        }
    )


# --------------------------------------------------------------------------
# project
# --------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("project", Project(name="untitled"))
    ss.setdefault("project_path", None)
    ss.setdefault("raw", None)  # (hash, DataFrame)
    ss.setdefault("prepared", None)  # (hash, DataFrame)
    ss.setdefault("runs", {})  # model name -> (hash, ModelRun)
    ss.setdefault("leakage", None)  # (hash, DataFrame)
    ss.setdefault("errors", [])


def project() -> Project:
    init_state()
    return st.session_state.project


def set_project(p: Project, path: str | None = None) -> None:
    init_state()
    st.session_state.project = p
    if path is not None:
        st.session_state.project_path = path
    st.session_state.raw = None
    st.session_state.prepared = None
    st.session_state.runs = {}
    st.session_state.leakage = None


def touch() -> None:
    """Call after mutating the project: autosave and invalidate nothing
    (caches are hash-keyed, so stale entries simply stop matching)."""
    path = st.session_state.get("project_path")
    if path:
        try:
            project().to_json(path)
        except OSError as exc:  # pragma: no cover - surfaced in the UI
            st.session_state.errors.append(f"Autosave failed: {exc}")


def default_project_path(p: Project) -> str:
    src = Path(p.data.source.path) if p.data.source.path else Path.cwd()
    folder = src.parent if src.is_file() else Path.cwd()
    return str(folder / f"{p.name}{PROJECT_SUFFIX}")


# --------------------------------------------------------------------------
# cached artefacts
# --------------------------------------------------------------------------
def raw_frame(force: bool = False) -> pl.DataFrame | None:
    """The loaded (optionally sampled) source frame, or None if no source."""
    p = project()
    if not p.data.source.path:
        return None
    h = source_hash(p)
    cached = st.session_state.raw
    if cached is not None and cached[0] == h and not force:
        return cached[1]
    with st.spinner(f"Loading {Path(p.data.source.path).name} ..."):
        df = load_source(
            p.data.source, sample_rows=p.data.sample_rows, seed=p.data.sample_seed
        )
    st.session_state.raw = (h, df)
    st.session_state.prepared = None
    return df


def prepared_frame() -> pl.DataFrame | None:
    """Raw frame after renames / recodes / derived / filters / split."""
    p = project()
    raw = raw_frame()
    if raw is None:
        return None
    h = data_hash(p)
    cached = st.session_state.prepared
    if cached is not None and cached[0] == h:
        return cached[1]
    with st.spinner("Preparing data ..."):
        df = prepare(p, raw)
    st.session_state.prepared = (h, df)
    return df


def get_run(model: str) -> ModelRun | None:
    """The cached run for ``model`` if it matches the current spec."""
    p = project()
    if model not in p.models:
        return None
    cached = st.session_state.runs.get(model)
    if cached is not None and cached[0] == model_hash(p, model):
        return cached[1]
    return None


def stale_run(model: str) -> ModelRun | None:
    """A run for ``model`` that no longer matches the spec (needs refit)."""
    cached = st.session_state.runs.get(model)
    return cached[1] if cached is not None else None


def fit_model(model: str) -> ModelRun:
    p = project()
    df = prepared_frame()
    if df is None:
        raise ValueError("Load data first (Project page).")
    with st.spinner(f"Fitting {model} ..."):
        run = run_model(p, df, model)
    st.session_state.runs[model] = (model_hash(p, model), run)
    return run


def refresh_adjustments(model: str) -> ModelRun | None:
    """Re-apply the model's manual adjustments / base-rate override to its
    cached run without refitting."""
    run = get_run(model)
    df = prepared_frame()
    if run is None or df is None:
        return None
    return rebuild_rate_model(project(), run, df)


def current_runs() -> dict[str, ModelRun]:
    return {name: r for name, (_h, r) in st.session_state.get("runs", {}).items()}


def leakage(force: bool = False) -> pl.DataFrame | None:
    p = project()
    df = prepared_frame()
    if df is None or p.target is None:
        return None
    key = spec_hash(
        {"data": p.to_dict()["data"], "model": p.champion, "expl": p.exploration}
    )
    cached = st.session_state.leakage
    if cached is not None and cached[0] == key and not force:
        return cached[1]
    with st.spinner("Scanning for leakage (one small GLM per variable) ..."):
        rep = leakage_report(df, p)
    st.session_state.leakage = (key, rep)
    return rep


# --------------------------------------------------------------------------
# status helpers
# --------------------------------------------------------------------------
def status() -> dict[str, bool]:
    p = project()
    raw = st.session_state.get("raw")
    split_ok = p.data.split.mode == "random" or (
        raw is not None and p.data.split.column in raw[1].columns
    )
    return {
        "data": bool(p.data.source.path),
        "roles": p.target is not None and bool(p.predictors),
        "split": bool(p.data.source.path) and split_ok,
        "model": bool(p.models),
        "fitted": any(get_run(m) is not None for m in p.models),
    }
