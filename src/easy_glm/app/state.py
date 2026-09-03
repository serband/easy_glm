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
(``<project>.easyglm-runs/<model-tag>-<key>.pkl``) so a browser reload or
reopening the project restores them instead of refitting. The key combines the
sample-free spec hash, the identity of the data file (path, size, mtime), the
library versions and :data:`PERSIST_FORMAT` (bumped whenever the shape *or the
meaning* of anything pickled changes); a file that cannot be loaded (a
corrupt pickle, a design that no longer matches *readable* data) is removed, a
file whose key merely differs — or whose data file cannot be read at this
moment — is left alone until a newer run of the same model is saved. The folder
holds pickles: trusted local content, like derived-column expressions. The
pickle stores the fitted run; adjustments and the base-rate override are
re-applied from the *current* project when it is loaded, so the project file
stays the truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sys
import uuid
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
#: Bump whenever the shape **or the meaning** of anything pickled (ModelRun,
#: GLMFit, RateModel, DesignSpec, their coefficients, ...) changes, so older
#: pickles are treated as cache misses even in a development checkout where the
#: installed version number does not move. A change of meaning that leaves the
#: shape alone is the dangerous case: it unpickles cleanly and is then read
#: wrongly.
#: 3 — B2: a ``LinearEncoder``'s coefficients became per-band slopes instead of
#: hinge (change-of-slope) coefficients. Nothing about the pickle's shape moved,
#: so a run cached by an earlier 0.4 development build would have been re-read
#: as if its numbers were slopes.
PERSIST_FORMAT = 3
#: Project-page widgets whose keyed value must not leak into another project.
# session-state keys that belong to the app itself; everything else is widget
# state (or a page's scratch result) and is dropped when another project is
# loaded, so no page carries the previous project's edits into the new one
_APP_STATE_KEYS = frozenset(
    {
        "project",
        "project_path",
        "project_token",
        "project_stamp",
        "conflict",
        "prep_error",
        "raw",
        "prepared",
        "sample",
        "raw_sample",
        "runs",
        "leakage",
        "errors",
        "load_error",
    }
)
_KEY_RE = re.compile(r"[0-9a-f]{16}")


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
    ss.setdefault("project_token", uuid.uuid4().hex[:8])
    ss.setdefault(
        "project_stamp", None
    )  # identity of the file as last read/written here
    ss.setdefault("conflict", None)  # path whose on-disk copy changed under us
    ss.setdefault("prep_error", None)
    ss.setdefault("raw", None)  # (hash, DataFrame) — full source frame
    ss.setdefault("prepared", None)  # (hash, DataFrame) — full prepared frame
    ss.setdefault("sample", None)  # (hash, DataFrame) — exploration sample
    ss.setdefault("raw_sample", None)  # (hash, DataFrame) — sampled raw frame
    ss.setdefault("runs", {})  # model name -> (hash, ModelRun)
    ss.setdefault("leakage", None)  # (hash, DataFrame)
    ss.setdefault("errors", [])
    ss.setdefault("load_error", None)


def project() -> Project:
    init_state()
    return st.session_state.project


def set_project(p: Project, path: str | None = None) -> None:
    """Make ``p`` the session's project. ``path`` is where it autosaves;
    ``None`` means an unsaved project (a new project never inherits the file
    of the one it replaces)."""
    init_state()
    st.session_state.project = p
    st.session_state.project_path = path
    st.session_state.project_stamp = _file_stamp(path)
    st.session_state.conflict = None
    st.session_state.prep_error = None
    st.session_state.load_error = None
    # keyed widgets keep their value across projects (Streamlit carries a
    # key's value even when the widget's default changes): give every project
    # its own widget keys (see ``widget_key``) and drop every other widget's
    # state, so a reload never re-applies this tab's stale edits
    st.session_state.project_token = uuid.uuid4().hex[:8]
    for key in ("raw", "prepared", "sample", "raw_sample", "leakage"):
        st.session_state[key] = None
    st.session_state.runs = {}
    for key in list(st.session_state.keys()):
        if key not in _APP_STATE_KEYS and not key.startswith("_"):
            st.session_state.pop(key, None)


def widget_key(name: str) -> str:
    """A widget key that changes whenever another project is loaded, so a
    text box never shows the previous project's value."""
    init_state()
    return f"{name}_{st.session_state.project_token}"


def _file_stamp(path: str | None) -> tuple[int, int, str] | None:
    """Identity of the project file: modification time, size and a hash of the
    bytes. The timestamp alone is not enough — NFS, SMB and FAT round it to a
    second or two, so a second write inside the same tick would look unchanged
    (project files are a few kB; hashing them costs nothing)."""
    if not path:
        return None
    try:
        info = os.stat(path)
        digest = hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size, digest)


def file_changed_on_disk() -> bool:
    """True when the project file was modified by someone else (another tab,
    another session) since this session last read or wrote it."""
    path = st.session_state.get("project_path")
    if not path:
        return False
    seen = st.session_state.get("project_stamp")
    now = _file_stamp(path)
    return seen is not None and now is not None and now != seen


def _clear_autosave_errors() -> bool:
    """Drop the "Autosave failed" entries after a successful save (the banner
    would otherwise keep saying "edits are not being saved" once they are
    again). True when something was actually dropped."""
    errors = st.session_state.get("errors", [])
    kept = [e for e in errors if not e.startswith("Autosave")]
    if len(kept) == len(errors):
        return False
    st.session_state.errors = kept
    return True


def _write_project(path: str) -> bool:
    """Write the project and record the file's identity. True when a stale
    autosave error was cleared (the caller redraws the page without it)."""
    project().to_json(path)
    st.session_state.project_stamp = _file_stamp(path)
    return _clear_autosave_errors()


def save_project(path: str, *, force: bool = False) -> str | None:
    """Write the project to ``path``; returns an error message instead of
    raising. A save to the current path that would overwrite another
    session's changes is refused unless ``force`` (see :func:`touch`)."""
    path = (path or "").strip()
    if not path:
        return "Give the project file a name first"
    if Path(path).is_dir():
        return f"{path} is a folder; give the project file a name"
    same_file = path == st.session_state.get("project_path")
    if same_file and not force and file_changed_on_disk():
        st.session_state.conflict = path
        return (
            f"{path} was changed by another session; reload it or overwrite it "
            "using the notice at the top of the page"
        )
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        project().to_json(path)
    except (OSError, TypeError, ValueError) as exc:
        # the project keeps autosaving to the previous file
        return f"Could not save the project to {path}: {exc}"
    st.session_state.project_path = path
    st.session_state.project_stamp = _file_stamp(path)
    st.session_state.conflict = None
    _clear_autosave_errors()
    return None


def resolve_conflict(action: str) -> str | None:
    """``"reload"`` replaces this session's project with the file on disk;
    ``"overwrite"`` writes this session's version over it."""
    path = st.session_state.get("conflict") or st.session_state.get("project_path")
    if not path:
        return None
    if action == "reload":
        try:
            set_project(Project.from_json(path), path)
        except Exception as exc:  # noqa: BLE001 - never a traceback
            return f"Could not reload {path}: {exc}"
        return None
    return save_project(path, force=True)


def touch() -> None:
    """Call after mutating the project: autosave (never raises; a failure is
    shown on every page) and invalidate nothing — caches are hash-keyed, so
    stale entries simply stop matching. While the file on disk has been changed
    by another session, autosave pauses until the user reloads or overwrites."""
    path = st.session_state.get("project_path")
    if not path:
        return
    if st.session_state.get("conflict"):
        return
    if file_changed_on_disk():
        st.session_state.conflict = path
        # show the notice straight away (it is drawn at the top of the page)
        st.rerun()
    try:
        recovered = _write_project(path)
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI on every page
        msg = f"Autosave failed: {exc}"
        errors = st.session_state.setdefault("errors", [])
        if msg not in errors:
            errors.append(msg)
            # the error strip is drawn at the top of the page, before the
            # widget that triggered this save: redraw so it shows straight away
            st.rerun()
    else:
        if recovered:
            # autosave works again; the strip at the top of this page still
            # shows the old failure, so redraw without it
            st.rerun()


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
    try:
        with st.spinner(f"Loading {Path(p.data.source.path).name} ..."):
            df = load_source(p.data.source)
    except Exception as exc:  # noqa: BLE001 - surfaced on the page, never a traceback
        st.session_state.raw = None
        if isinstance(exc, FileNotFoundError):
            reason = "the file does not exist"
        elif isinstance(exc, IsADirectoryError):
            reason = "the path is a folder, not a file"
        else:
            reason = str(exc)
        st.session_state.load_error = (
            f"Could not load {p.data.source.path}: {reason}. Check the path on the "
            "Project & data page."
        )
        return None
    st.session_state.load_error = None
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
    try:
        with st.spinner("Preparing data ..."):
            df = prepare(p, raw)
    except Exception as exc:  # noqa: BLE001 - a bad recode/derived/filter/split
        st.session_state.prepared = None
        st.session_state.prep_error = (
            f"The data steps fail: {exc}. Fix or remove the offending rename, "
            "recode, derived column, filter or split on the Variables / Split pages."
        )
        return None
    st.session_state.prep_error = None
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
    """Identity of a fit on disk: spec (sample-free), data file, library
    versions and the pickle format constant."""
    return spec_hash(
        {
            "model_hash": model_hash(p, model),
            "data": _data_identity(p),
            "versions": _versions(),
            "format": PERSIST_FORMAT,
        }
    )


def _model_tag(model: str) -> str:
    """Unambiguous, file-name-safe tag for a model name (names may contain any
    character; two names never share a tag prefix)."""
    return hashlib.sha1(model.encode()).hexdigest()[:10]


def run_file(folder: Path, model: str, key: str) -> Path:
    return folder / f"{_model_tag(model)}-{key}.pkl"


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
    """Persisted files of exactly this model (tag + 16-hex key), never of a
    model whose name merely shares a prefix."""
    prefix = f"{_model_tag(model)}-"
    return sorted(
        f
        for f in folder.iterdir()
        if f.suffix == ".pkl"
        and f.name.startswith(prefix)
        and _KEY_RE.fullmatch(f.stem[len(prefix) :]) is not None
    )


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
        target = run_file(folder, model, key)
        # unique temp name per process/write, then an atomic replace, so two
        # sessions persisting the same model never truncate each other's file
        tmp = folder / f"{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with tmp.open("wb") as fh:
                pickle.dump(run, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()
        sidecar = {
            "model": model,
            "key": key,
            "format": PERSIST_FORMAT,
            "model_hash": model_hash(p, model),
            "data": _data_identity(p),
            "versions": _versions(),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_rows": run.train_rows,
            "holdout_rows": run.holdout_rows,
        }
        target.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
        # "latest run per model" is enforced here, on a successful save only
        for old in _run_files(folder, model):
            if old != target:
                _remove_run_file(old)
        _remove_orphans(folder, p)
        return target
    except OSError as exc:  # pragma: no cover - surfaced in the UI
        st.session_state.errors.append(f"Could not persist the fit: {exc}")
        return None


def _remove_orphans(folder: Path, p: Project) -> None:
    """Drop persisted files of models that no longer exist in the project."""
    live = {_model_tag(m) for m in p.models}
    for f in folder.glob("*.pkl"):
        tag = f.name.split("-", 1)[0]
        if len(tag) == 10 and tag not in live:
            _remove_run_file(f)


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
    spec, data file, library versions and pickle format; otherwise None.

    Reading never deletes a file whose key merely differs (a transient spec
    edit must not erase the last fit); only a file that cannot be loaded is
    removed, and only while the data file *can* be read: a data file that is
    momentarily unreadable (a share that blips, a permission change) is a cache
    miss, never a reason to throw the fit away. Stale files go when a newer run
    is saved (:func:`persist_run`).
    """
    folder = runs_dir()
    if folder is None or not folder.exists():
        return None
    p = project()
    if model not in p.models:
        return None
    target = run_file(folder, model, run_key(p, model))
    if not target.exists():
        return None
    try:
        with target.open("rb") as fh:
            run = pickle.load(
                fh
            )  # noqa: S301 - trusted local folder (module docstring)
        if not isinstance(run, ModelRun) or run.name != model:
            raise ValueError("not a persisted run for this model")
    except Exception:  # noqa: BLE001 - a corrupt or foreign pickle: drop it
        _remove_run_file(target)
        return None
    df = prepared_frame()
    if df is None:
        # the data cannot be read *now* (unreadable file, a failing data step):
        # nothing can be verified, so this is a plain cache miss and the fit
        # stays on disk for when the data comes back
        return None
    if not _design_matches(p, model, run):
        _remove_run_file(target)
        return None
    # the project is the truth for adjustments / base-rate override; an entry
    # the model refuses is dropped with a message, like everywhere else
    cfg = p.models[model]
    while True:
        try:
            run = rebuild_rate_model(p, run, df)
            break
        except AdjustmentError as exc:
            _drop_refused_adjustment(cfg, exc)
        except Exception:  # noqa: BLE001 - shape mismatch etc.: cache miss
            _remove_run_file(target)
            return None
    st.session_state.runs[model] = (model_hash(p, model), run)
    return run


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------
def get_run(model: str) -> ModelRun | None:
    """The run for ``model`` matching the current spec: from the session, else
    from the persisted folder, else None. A run whose columns are no longer
    in the prepared data is never returned (nothing may look fitted while the
    model references a missing column)."""
    p = project()
    if model not in p.models:
        return None
    prepared = st.session_state.get("prepared")
    if prepared is not None and p.missing_columns(prepared[1].columns, model):
        return None
    cached = st.session_state.runs.get(model)
    if cached is not None and cached[0] == model_hash(p, model):
        return cached[1]
    return load_persisted_run(model)


def remove_model_runs(model: str) -> None:
    """Forget a deleted model's fit in the session and on disk."""
    st.session_state.runs.pop(model, None)
    folder = runs_dir()
    if folder is not None and folder.exists():
        for f in _run_files(folder, model):
            _remove_run_file(f)


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
    msg = f"Adjustment not applied and removed from the project: {exc}"
    st.error(msg)
    # the caller usually reruns straight after; keep the message for that run
    st.session_state.setdefault("_flash", []).append(("error", msg))


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
    loaded = bool(p.data.source.path) and raw is not None
    return {
        "data": loaded,
        "roles": p.target is not None and bool(p.predictors),
        "split": loaded and split_ok and not st.session_state.get("prep_error"),
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
