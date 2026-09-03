"""easy_glm Workbench — Streamlit entry point.

Run with ``python -m easy_glm.app [project.json]`` or
``streamlit run src/easy_glm/app/main.py -- --project=path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from easy_glm.app import (
    pages_compare,
    pages_diagnostics,
    pages_explore,
    pages_export,
    pages_model,
    pages_project,
    pages_split,
    pages_tables,
    pages_variables,
)
from easy_glm.app import state as S
from easy_glm.workflow import Project

st.set_page_config(
    page_title="easy_glm workbench",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _cli_project() -> str | None:
    for arg in sys.argv[1:]:
        if arg.startswith("--project="):
            return arg.split("=", 1)[1]
    return None


S.init_state()
if not st.session_state.get("_cli_loaded"):
    st.session_state._cli_loaded = True
    path = _cli_project()
    if path and Path(path).exists():
        S.set_project(Project.from_json(path), path)

pages = [
    st.Page(
        pages_project.render,
        title="Project & data",
        icon=":material/folder_open:",
        url_path="project",
        default=True,
    ),
    st.Page(
        pages_variables.render,
        title="Variables",
        icon=":material/view_column:",
        url_path="variables",
    ),
    st.Page(
        pages_explore.render,
        title="Explore",
        icon=":material/search_insights:",
        url_path="explore",
    ),
    st.Page(
        pages_split.render,
        title="Split",
        icon=":material/call_split:",
        url_path="split",
    ),
    st.Page(
        pages_model.render, title="Model", icon=":material/function:", url_path="model"
    ),
    st.Page(
        pages_diagnostics.render,
        title="Diagnostics",
        icon=":material/monitoring:",
        url_path="diagnostics",
    ),
    st.Page(
        pages_compare.render,
        title="Compare",
        icon=":material/compare_arrows:",
        url_path="compare",
    ),
    st.Page(
        pages_tables.render,
        title="Rate tables",
        icon=":material/table_chart:",
        url_path="tables",
    ),
    st.Page(
        pages_export.render, title="Export", icon=":material/code:", url_path="export"
    ),
]
nav = st.navigation({"Workflow": pages})

with st.sidebar:
    p = S.project()
    project_path = st.session_state.project_path
    st.markdown("### Current project")
    st.markdown(f"**{p.name}**")
    if project_path:
        st.caption(f"Saved setup · {Path(project_path).name}")
    else:
        st.warning("Not saved yet")
        st.caption("Name and save it on Project & data.")
        st.page_link(
            pages[0],
            label="Open Project & data",
            icon=":material/edit:",
            help="Name the project and choose where to save its setup.",
        )
    s = S.status()
    st.markdown("#### Setup progress")
    st.caption("Work down this checklist before reviewing results.")
    for label, ok in [
        ("Data loaded", s["data"]),
        ("Target and predictors chosen", s["roles"]),
        ("Data prepared and split", s["split"]),
        ("Model defined", s["model"]),
        ("Model fitted", s["fitted"]),
    ]:
        st.caption(f"{'✅' if ok else '⬜'} {label}")
    if p.models:
        st.caption(
            "Models: "
            + ", ".join(f"**{n}**" if n == p.champion else n for n in p.models)
        )
    # One comparison choice for the whole session: the Compare, Diagnostics and
    # Rate tables pages default to it (each can still have a page-level override).
    fitted = S.fitted_models()
    champion = p.champion if p.champion in fitted else (fitted[0] if fitted else None)
    options = ["(none)"] + [n for n in fitted if n != champion]
    if len(options) > 1:
        current = S.challenger()
        choice = st.selectbox(
            "Default comparison model",
            options,
            index=options.index(current) if current in options else 0,
            key=S.widget_key("sidebar_challenger"),
            help=(
                "The fitted incumbent or challenger used by Diagnostics, Compare, "
                "Rate tables and reports. Choose (none) for no model comparison; "
                "Diagnostics then uses a null-model benchmark where needed."
            ),
        )
        S.set_challenger(None if choice == "(none)" else choice)
    else:
        st.caption("Default comparison model")
        st.caption("Fit two models to compare them.")
    # With fewer than two fitted models the selector is not drawn; the stored
    # choice survives a momentary stale fit and is ignored once no longer valid.
    st.caption(
        "Saving keeps your setup and adjustments. Fitted models are stored "
        "separately and restored when their data and setup still match."
    )
    if st.button(
        "Save project setup",
        width="stretch",
        help="Save the project name, data setup, model definitions and adjustments.",
    ):
        path = project_path or S.default_project_path(p)
        err = S.save_project(path)
        if err:
            st.error(err)
        else:
            st.toast(f"Saved {path}")
    if st.session_state.get("conflict"):
        st.error("Autosave paused: the project file changed on disk (see the notice).")
    elif any(e.startswith("Autosave") for e in st.session_state.get("errors", [])):
        st.error("Autosave is failing — edits are not being saved.")
    st.caption(S.persistence_note())

nav.run()
