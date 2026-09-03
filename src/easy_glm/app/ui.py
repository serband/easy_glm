"""Small shared widgets and formatting helpers for the workbench pages."""

from __future__ import annotations

import io
import math
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import polars as pl
import streamlit as st

from easy_glm.workflow import ModelRun

from . import state as S

T = TypeVar("T")


def fmt(x: Any, *, pct: bool = False, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if pct:
        return "—" if abs(x) > 100 else f"{x:.1%}"  # ±10,000 % is noise, not a number
    if isinstance(x, int | float) and abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def metric_row(items: list[tuple[str, Any, str | None]]) -> None:
    """``[(label, value, help), ...]`` rendered as a row of ``st.metric``."""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items, strict=True):
        col.metric(label, value, help=help_text)


def flash(kind: str, text: str) -> None:
    """Queue a one-shot notice to show at the top of the *next* run.

    Streamlit discards anything drawn in a run that ends with ``st.rerun()``, so
    every message that precedes a rerun goes through here and is rendered by
    :func:`show_flash` (called from :func:`status_bar`)."""
    st.session_state.setdefault("_flash", []).append((kind, text))


def show_flash() -> None:
    notices = st.session_state.pop("_flash", [])
    for kind, text in notices:
        getattr(
            st, kind if kind in ("success", "warning", "error", "info") else "info"
        )(text)


def guarded(fn: Callable[[], T], what: str, *, default: T | None = None) -> T | None:
    """Run ``fn``; on any exception show a clear message naming ``what`` and
    return ``default``. The pipeline error boundary for every page: a bad
    rename, recode, derived column, filter or split must never be a traceback."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - that is the point
        st.error(f"{what} failed: {exc}")
        return default


def conflict_notice() -> None:
    """Persistent notice while the project file was changed by another
    session; autosave is paused until the user chooses."""
    path = st.session_state.get("conflict")
    if not path:
        return
    st.warning(
        f"**{path}** was changed by another browser tab or session since this tab "
        "last saved it. Autosave is paused so neither copy is lost: reload the "
        "file (this tab's unsaved edits are dropped) or overwrite it with this "
        "tab's version."
    )
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Reload from disk", key="conflict_reload", type="primary"):
        err = S.resolve_conflict("reload")
        if err:
            st.error(err)
        else:
            flash("success", f"Reloaded {path}")
            st.rerun()
    if c2.button("Overwrite with this tab's version", key="conflict_overwrite"):
        err = S.resolve_conflict("overwrite")
        if err:
            st.error(err)
        else:
            flash("success", f"Saved this tab's version to {path}")
            st.rerun()


def status_bar() -> None:
    show_flash()
    show_errors()
    conflict_notice()
    s = S.status()
    steps = [
        ("data", "Data"),
        ("roles", "Roles"),
        ("split", "Prepared"),
        ("model", "Model"),
        ("fitted", "Fitted"),
    ]
    chips = " ".join(
        (":green-badge[✓ " if s[k] else ":grey-badge[○ ") + label + "]"
        for k, label in steps
    )
    st.markdown(chips)


def require_data() -> pl.DataFrame | None:
    """The prepared frame, or None after showing why it is missing."""
    df = S.prepared_frame()
    if df is None:
        show_data_problem()
    return df


def show_data_problem() -> None:
    """The load or data-step error (if any), else a hint to load data."""
    err = st.session_state.get("load_error") or st.session_state.get("prep_error")
    if err:
        st.error(err)
    else:
        st.info("Load a data file on the **Project & data** page first.")


def require_raw() -> pl.DataFrame | None:
    """The raw frame, or None after showing why it is missing."""
    p = S.project()
    raw = S.raw_frame() if p.data.source.path else None
    if raw is None:
        err = st.session_state.get("load_error")
        if err:
            st.error(err)
        else:
            st.info("Load a data file on the **Project & data** page first.")
    return raw


def require_target() -> str | None:
    t = S.project().target
    if t is None:
        st.info("Assign a **target** role on the **Variables** page first.")
    return t


def run_selector(label: str = "Model", key: str = "run_select") -> ModelRun | None:
    runs = S.current_runs()
    fitted = {name: r for name, r in runs.items() if S.get_run(name) is not None}
    if not fitted:
        st.info("No fitted model yet — fit one on the **Model** page.")
        return None
    p = S.project()
    default = p.champion if p.champion in fitted else next(iter(fitted))
    names = list(fitted)
    name = st.selectbox(label, names, index=names.index(default), key=key)
    return fitted[name]


def frame_bytes(df: pl.DataFrame, kind: str = "csv") -> bytes:
    buf = io.BytesIO()
    if kind == "csv":
        df.write_csv(buf)
    else:
        df.write_parquet(buf)
    return buf.getvalue()


def safe_filename(name: str, fallback: str = "model") -> str:
    """A file-name-safe version of a model/project name (never a path)."""
    cleaned = re.sub(r"[^\w.\- ]+", "_", str(name)).strip(" ._")
    return cleaned[:80] if re.search(r"\w", cleaned) else fallback


def excel_bytes(run: ModelRun) -> bytes:
    """Workbook of the tables the scorer uses (manual adjustments included)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{safe_filename(run.name)}.xlsx"
        run.rate_model.to_excel(path)
        return path.read_bytes()


def easyglm_bytes(run: ModelRun) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{safe_filename(run.name)}.easyglm"
        run.rate_model.to_json(path)
        return path.read_bytes()


def show_errors() -> None:
    """Persistent problems (autosave / persistence failures) on every page;
    they clear when the cause is fixed (a successful save drops them)."""
    for e in st.session_state.get("errors", []):
        st.error(e)


def polars_table(df: pl.DataFrame, **kwargs: Any) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)
