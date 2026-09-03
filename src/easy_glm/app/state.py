"""Session state for the workbench: one Project, cached frames and runs.

All pages go through these helpers so that

* the :class:`Project` in ``st.session_state`` is the single source of truth,
* expensive artefacts are cached on a hash of the part of the spec they depend
  on, and
* the project is autosaved after every change.

Two data frames live in the session:

* the **full** prepared frame — every fit, diagnostic, rate table and the
  leakage report use it (``prepared_frame``); knots and levels are always
  derived from it;
* an **exploration sample** (``sample_frame`` / ``raw_sample``) — the Explore
  page, the Design-page previews and the Variables-page previews use it so
  large books stay interactive. ``Project.data.sample_rows`` / ``sample_seed``
  only ever size that sample; changing it never invalidates a fit.

Fitted runs are **persisted** next to the project file
(``<project>.easyglm-runs/<model>-<key>.pkl``) so a browser reload or reopening
the project restores them instead of refitting. The key combines the sample-free
spec hash, the identity of the data file (path, size, mtime) and the library
versions; anything that fails to load is treated as a cache miss and removed.
The folder holds pickles: trusted local content, like derived-column
expressions. The pickle stores the fitted run; adjustments and the base-rate
override are re-applied from the *current* project when it is loaded, so the
project file stays the truth.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

from easy_glm.workflow import (
    AdjustmentError,
    ModelRun,
    Project,
    build_design,
    leakage_report,
    load_source,
    prepare,
    rebuild_rate_model,
    run_model,
    train_holdout,
)

PROJECT_SUFFIX = ".easyglm-project.json"
RUNS_SUFFIX = ".easyglm-runs"
_SAMPLE_KEYS = ("sample_rows", "sample_seed")


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------
def spec_hash(obj: Any) -> str:
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _data_dict(project: Project, *, with_sample: bool) -> dict[str, Any]:
    d = dict(project.to_dict()["data"])
    if not with_sample:
        for k in _SAMPLE_KEYS:
            d.pop(k, None)
    return d


def source_hash(project: Project) -> str:
    """Key of the raw (full) frame: the source only."""
    return spec_hash({"source": project.to_dict()["data"]["source"]})


def data_hash(project: Project) -> str:
    """Key of the full prepared frame: every data step except the sample."""
    return spec_hash(_data_dict(project, with_sample=False))


def sample_hash(project: Project) -> str:
    """Key of the exploration sample: the data steps plus the sample settings."""
    return spec_hash(_data_dict(project, with_sample=True))


def model_hash(project: Project, model: str) -> str:
    """Key of a fit: data steps (sample-free), the design of the model's
    predictors and interactions, and the model config minus what is applied
    post-fit (adjustments, base-rate override, notes)."""
    d = project.to_dict()
    cfg = dict(d["models"][model])
    cfg.pop("adjustments", None)  # applied post-fit, never require a refit
    cfg.pop("base_rate_override", None)
    cfg.pop("notes", None)
    design = {v: d["design"]["variables"].get(v) for v in cfg["predictors"]}
    return spec_hash(
        {
            "data": _data_dict(project, with_sample=False),
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
    ss.setdefault("raw", None)  # (hash, DataFrame) — full source frame
    ss.setdefault("prepared", None)  # (hash, DataFrame) — full prepared frame
    ss.setdefault("sample", None)  # (hash, DataFrame) — exploration sample
    ss.setdefault("raw_sample", None)  # (hash, DataFrame) — sampled raw frame
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
    for key in ("raw", "prepared", "sample", "raw_sample", "leakage"):
        st.session_state[key] = None
    st.session_state.runs = {}


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
# cached frames
# --------------------------------------------------------------------------
def raw_frame(force: bool = False) -> pl.DataFrame | None:
    """The full source frame (never sampled), or None if no source."""
    p = project()
    if not p.data.source.path:
        return None
    h = source_hash(p)
    cached = st.session_state.raw
    if cached is not None and cached[0] == h and not force:
        return cached[1]
    with st.spinner(f"Loading {Path(p.data.source.path).name} ..."):
        df = load_source(p.data.source)
    st.session_state.raw = (h, df)
    for key in ("prepared", "sample", "raw_sample"):
        st.session_state[key] = None
    return df


def prepared_frame() -> pl.DataFrame | None:
    """Full frame after renames / recodes / derived / filters / split.
    Fits, diagnostics, tables and the leakage report use this frame."""
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


def _sample_of(df: pl.DataFrame, p: Project) -> pl.DataFrame:
    n = p.data.sample_rows
    if n is None or n <= 0 or n >= df.height:
        return df
    return df.sample(n=n, seed=p.data.sample_seed)


def sample_frame() -> pl.DataFrame | None:
    """Exploration sample of the prepared frame (the whole frame when no
    sample size is set). Never used for fitting."""
    p = project()
    df = prepared_frame()
    if df is None:
        return None
    h = sample_hash(p)
    cached = st.session_state.sample
    if cached is not None and cached[0] == h:
        return cached[1]
    s = _sample_of(df, p)
    st.session_state.sample = (h, s)
    return s


def raw_sample() -> pl.DataFrame | None:
    """Exploration sample of the raw frame, for previews before preparation
    (recode level counts, derived-column previews)."""
    p = project()
    raw = raw_frame()
    if raw is None:
        return None
    h = spec_hash({"source": source_hash(p), "sample": _data_dict(p, with_sample=True)})
    cached = st.session_state.raw_sample
    if cached is not None and cached[0] == h:
        return cached[1]
    s = _sample_of(raw, p)
    st.session_state.raw_sample = (h, s)
    return s


def is_sampled() -> bool:
    """True when the exploration sample is smaller than the data."""
    p = project()
    raw = st.session_state.get("raw")
    n = p.data.sample_rows
    return bool(n) and raw is not None and n < raw[1].height


def train_frame() -> pl.DataFrame | None:
    """Training rows of the full prepared frame (knots and levels come from here)."""
    df = prepared_frame()
    if df is None:
        return None
    return train_holdout(df, project().data.split)[0]


def train_sample() -> pl.DataFrame | None:
    """Training rows of the exploration sample (for preview charts only)."""
    df = sample_frame()
    if df is None:
        return None
    return train_holdout(df, project().data.split)[0]


# --------------------------------------------------------------------------
# run persistence
# --------------------------------------------------------------------------
def _versions() -> dict[str, str]:
    """Library versions that go into a persisted run's key (monkeypatchable)."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}"
    }
    for name in ("easy_glm", "glum", "polars", "numpy"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover
            out[name] = "unknown"
    return out


def _data_identity(p: Project) -> dict[str, Any]:
    path = Path(p.data.source.path) if p.data.source.path else None
    if path is None or not path.exists():
        return {"path": str(path), "size": None, "mtime_ns": None}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def run_key(p: Project, model: str) -> str:
    """Identity of a fit on disk: spec (sample-free), data file, library versions."""
    return spec_hash(
        {
            "model_hash": model_hash(p, model),
            "data": _data_identity(p),
            "versions": _versions(),
        }
    )


def runs_dir() -> Path | None:
    """Folder for persisted runs next to the project file; None when the project
    has not been saved anywhere."""
    path = st.session_state.get("project_path")
    if not path:
        return None
    path = Path(path)
    stem = (
        path.name[: -len(PROJECT_SUFFIX)]
        if path.name.endswith(PROJECT_SUFFIX)
        else path.stem
    )
    return path.parent / f"{stem}{RUNS_SUFFIX}"


def _run_files(folder: Path, model: str) -> list[Path]:
    return sorted(folder.glob(f"{model}-*.pkl"))


def persist_run(model: str, run: ModelRun, key: str | None = None) -> Path | None:
    """Pickle ``run`` for ``model`` (latest run per model only). Returns the
    file written, or None for an unsaved project."""
    folder = runs_dir()
    if folder is None:
        return None
    p = project()
    key = key or run_key(p, model)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{model}-{key}.pkl"
        tmp = target.with_suffix(".pkl.tmp")
        with tmp.open("wb") as fh:
            pickle.dump(run, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(target)
        sidecar = {
            "model": model,
            "key": key,
            "model_hash": model_hash(p, model),
            "data": _data_identity(p),
            "versions": _versions(),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_rows": run.train_rows,
            "holdout_rows": run.holdout_rows,
        }
        target.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
        for old in _run_files(folder, model):
            if old != target:
                _remove_run_file(old)
        return target
    except OSError as exc:  # pragma: no cover - surfaced in the UI
        st.session_state.errors.append(f"Could not persist the fit: {exc}")
        return None


def _remove_run_file(path: Path) -> None:
    for f in (path, path.with_suffix(".json")):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
        except OSError:  # pragma: no cover
            pass


def _design_matches(p: Project, model: str, run: ModelRun) -> bool:
    """Cheap sanity check on a loaded run: its design equals what the current
    spec would build on the current training data."""
    train = train_frame()
    if train is None:
        return False
    cfg = p.models[model]
    try:
        spec = build_design(
            p,
            train,
            cfg.predictors,
            weight_col=cfg.weight,
            interactions=cfg.interactions,
        )
    except Exception:  # noqa: BLE001 - any failure means "do not trust the file"
        return False
    return list(spec.feature_names) == list(run.spec.feature_names)


def load_persisted_run(model: str) -> ModelRun | None:
    """Load the persisted run for ``model`` when its key matches the current
    spec, data file and library versions; otherwise None. Unreadable or stale
    files are deleted."""
    folder = runs_dir()
    if folder is None or not folder.exists():
        return None
    p = project()
    if model not in p.models:
        return None
    key = run_key(p, model)
    target = folder / f"{model}-{key}.pkl"
    for f in _run_files(folder, model):
        if f != target:
            _remove_run_file(f)  # stale: spec, data or versions changed
    if not target.exists():
        return None
    try:
        with target.open("rb") as fh:
            run = pickle.load(
                fh
            )  # noqa: S301 - trusted local folder (module docstring)
        if not isinstance(run, ModelRun) or run.name != model:
            raise ValueError("not a persisted run for this model")
        if not _design_matches(p, model, run):
            raise ValueError("design no longer matches")
        df = prepared_frame()
        if df is None:
            raise ValueError("data not available")
        # the project is the truth for adjustments / base-rate override
        run = rebuild_rate_model(p, run, df)
    except Exception:  # noqa: BLE001 - any problem is a cache miss
        _remove_run_file(target)
        return None
    st.session_state.runs[model] = (model_hash(p, model), run)
    return run


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------
def get_run(model: str) -> ModelRun | None:
    """The run for ``model`` matching the current spec: from the session, else
    from the persisted folder, else None."""
    p = project()
    if model not in p.models:
        return None
    cached = st.session_state.runs.get(model)
    if cached is not None and cached[0] == model_hash(p, model):
        return cached[1]
    return load_persisted_run(model)


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
        while True:
            try:
                run = run_model(p, df, model)
                break
            except AdjustmentError as exc:
                _drop_refused_adjustment(p.models[model], exc)
    st.session_state.runs[model] = (model_hash(p, model), run)
    persist_run(model, run)
    return run


def _drop_refused_adjustment(cfg, exc: AdjustmentError) -> None:
    """Remove the adjustment the RateModel refused and tell the user, so a bad
    entry in the project can never lock a page."""
    bad = exc.adjustment
    cfg.adjustments = [a for a in cfg.adjustments if a is not bad]
    touch()
    st.error(f"Adjustment not applied and removed from the project: {exc}")


def refresh_adjustments(model: str) -> ModelRun | None:
    """Re-apply the model's manual adjustments / base-rate override to its
    cached run without refitting, and persist the result."""
    run = get_run(model)
    df = prepared_frame()
    if run is None or df is None:
        return None
    p = project()
    cfg = p.models[model]
    while True:
        try:
            run = rebuild_rate_model(p, run, df)
            break
        except AdjustmentError as exc:
            _drop_refused_adjustment(cfg, exc)
    persist_run(model, run)
    return run


def current_runs() -> dict[str, ModelRun]:
    return {name: r for name, (_h, r) in st.session_state.get("runs", {}).items()}


def leakage(force: bool = False) -> pl.DataFrame | None:
    """Leakage report on the full training rows (the report samples internally)."""
    p = project()
    df = prepared_frame()
    if df is None or p.target is None:
        return None
    key = spec_hash(
        {
            "data": _data_dict(p, with_sample=False),
            "model": p.champion,
            "expl": p.exploration,
        }
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


def persistence_note() -> str:
    """One line for the sidebar about where fits are kept."""
    folder = runs_dir()
    if folder is None:
        return (
            "Unsaved project — fits are not persisted; save the project to keep them."
        )
    return f"Fits persisted to `{folder.name}/` next to the project file."
