"""Small shared widgets and formatting helpers for the workbench pages."""

from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

from easy_glm.core.easyglm import EasyGLM
from easy_glm.workflow import ModelRun

from . import state as S


def fmt(x: Any, *, pct: bool = False, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if pct:
        return f"{x:.1%}"
    if isinstance(x, int | float) and abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def metric_row(items: list[tuple[str, Any, str | None]]) -> None:
    """``[(label, value, help), ...]`` rendered as a row of ``st.metric``."""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items, strict=True):
        col.metric(label, value, help=help_text)


def status_bar() -> None:
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
    df = S.prepared_frame()
    if df is None:
        st.info("Load a data file on the **Project & data** page first.")
    return df


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


def excel_bytes(run: ModelRun) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{run.name}.xlsx"
        EasyGLM(run.fit, run.rate_model, run.tables).to_excel(path)
        return path.read_bytes()


def easyglm_bytes(run: ModelRun) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{run.name}.easyglm"
        run.rate_model.to_json(path)
        return path.read_bytes()


def show_errors() -> None:
    errs = st.session_state.get("errors", [])
    for e in errs:
        st.warning(e)
    st.session_state.errors = []


def polars_table(df: pl.DataFrame, **kwargs: Any) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)
