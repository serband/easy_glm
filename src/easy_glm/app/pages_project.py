"""Page 1 — Project & data."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from easy_glm.workflow import Project, column_summary, infer_source_type

from . import state as S
from . import ui


def _open_project(path: str) -> None:
    p = Project.from_json(path)
    S.set_project(p, path)
    ui.flash("success", f"Opened {path}")


def render() -> None:
    p = S.project()
    st.title("Project & data")
    ui.status_bar()
    ui.show_errors()

    # ------------------------------------------------------------ project
    with st.container(border=True):
        st.subheader("Project")
        c1, c2 = st.columns([2, 3])
        name = c1.text_input("Project name", p.name, key="proj_name")
        if name != p.name:
            p.name = name
            S.touch()
        path_default = st.session_state.project_path or S.default_project_path(p)
        proj_path = c2.text_input(
            "Project file (autosaved after every change)", path_default, key="proj_path"
        )
        b1, b2, b3 = st.columns(3)
        if b1.button("Save project", type="primary", width="stretch"):
            p.to_json(proj_path)
            st.session_state.project_path = proj_path
            st.success(f"Saved {proj_path}")
        if b2.button("Open project file", width="stretch"):
            if Path(proj_path).exists():
                _open_project(proj_path)
                st.rerun()
            else:
                st.error(f"{proj_path} does not exist")
        if b3.button("New empty project", width="stretch"):
            S.set_project(Project(name="untitled"), None)
            st.rerun()
        uploaded = st.file_uploader(
            "…or drop a project JSON here", type=["json"], key="proj_upload"
        )
        if uploaded is not None and st.button("Load uploaded project"):
            tmp = Path(tempfile.mkdtemp()) / uploaded.name
            tmp.write_bytes(uploaded.getvalue())
            _open_project(str(tmp))
            st.rerun()

    # --------------------------------------------------------------- data
    with st.container(border=True):
        st.subheader("Data source")
        st.caption(
            "Point at a local file (fastest for large data) or upload one. "
            "Supported: parquet, csv, sas7bdat, xlsx, arrow/ipc."
        )
        c1, c2 = st.columns([4, 1])
        src_path = c1.text_input("File path", p.data.source.path, key="src_path")
        kinds = ["parquet", "csv", "sas7bdat", "xlsx", "ipc"]
        guess = infer_source_type(src_path) if src_path else p.data.source.type
        src_type = c2.selectbox(
            "Type", kinds, index=kinds.index(guess if guess in kinds else "parquet")
        )
        c3, c4 = st.columns([1, 3])
        sample = c3.number_input(
            "Exploration sample (rows, 0 = all)",
            min_value=0,
            value=int(p.data.sample_rows or 0),
            step=10000,
            help=(
                "Rows used by the Explore page and the Design / Variables previews so "
                "large books stay interactive. Fits, diagnostics, rate tables and the "
                "leakage report always use the full data; changing this never "
                "invalidates a fit."
            ),
            key="sample_rows",
        )
        if (int(sample) or None) != p.data.sample_rows:
            p.data.sample_rows = int(sample) or None
            S.touch()
        up = c4.file_uploader(
            "…or upload a data file",
            type=["parquet", "csv", "sas7bdat", "xlsx"],
            key="data_upload",
        )
        if up is not None:
            tmp = Path(tempfile.mkdtemp()) / up.name
            tmp.write_bytes(up.getvalue())
            src_path = str(tmp)
            src_type = infer_source_type(src_path)
        if st.button("Load data", type="primary"):
            p.data.source.path = src_path
            p.data.source.type = src_type
            S.touch()
            try:
                S.raw_frame(force=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load: {exc}")
            else:
                st.rerun()

    raw = S.raw_frame() if p.data.source.path else None
    if raw is None:
        return

    with st.container(border=True):
        st.subheader("Preview")
        mem = raw.estimated_size("mb")
        ui.metric_row(
            [
                ("Rows", f"{raw.height:,}", None),
                ("Columns", str(raw.width), None),
                ("Memory", f"{mem:,.0f} MB", None),
                (
                    "Exploration sample",
                    (
                        f"{p.data.sample_rows:,} rows"
                        if S.is_sampled()
                        else "off (full data)"
                    ),
                    "Explore / preview charts only; fits use every row",
                ),
            ]
        )
        tab1, tab2 = st.tabs(["First rows", "Columns"])
        with tab1:
            ui.polars_table(raw.head(50))
        with tab2:
            ui.polars_table(column_summary(raw))
        st.caption(
            "Next: assign roles and clean up variables on the **Variables** page."
        )
