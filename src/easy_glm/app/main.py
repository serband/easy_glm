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
    pages_tables,
    pages_variables,
    ui,
)
from easy_glm.app import state as S
from easy_glm.workflow import Project

st.set_page_config(
    page_title="easy_glm workbench",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The workbench is a local modelling tool. Streamlit's hosting shortcut implies
# that the current project will be deployed, which is not an EasyGLM action and
# is misleading here.
st.html("""
    <style>
    [data-testid="stAppDeployButton"], .stAppDeployButton {
        display: none;
    }
    </style>
    """)


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


split_is_ready = S.split_ready()


def _after_split(title: str, render):
    """Keep a directly opened downstream URL behind the same workflow gate."""

    def gated() -> None:
        if not split_is_ready:
            st.title(title)
            ui.status_bar()
            ui.require_split()
            return
        render()

    return gated


downstream_visibility = "visible" if split_is_ready else "hidden"
model_page = st.Page(
    _after_split("Model", pages_model.render),
    title="Model",
    icon=":material/function:",
    url_path="model",
    visibility=downstream_visibility,
)

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
        _after_split("Explore", pages_explore.render),
        title="Explore",
        icon=":material/search_insights:",
        url_path="explore",
        visibility=downstream_visibility,
    ),
    model_page,
    st.Page(
        _after_split("Diagnostics", pages_diagnostics.render),
        title="Diagnostics",
        icon=":material/monitoring:",
        url_path="diagnostics",
        visibility=downstream_visibility,
    ),
    st.Page(
        _after_split("Compare", pages_compare.render),
        title="Compare",
        icon=":material/compare_arrows:",
        url_path="compare",
        visibility=downstream_visibility,
    ),
    st.Page(
        _after_split("Rate tables", pages_tables.render),
        title="Rate tables",
        icon=":material/table_chart:",
        url_path="tables",
        visibility=downstream_visibility,
    ),
    st.Page(
        _after_split("Export", pages_export.render),
        title="Export",
        icon=":material/code:",
        url_path="export",
        visibility=downstream_visibility,
    ),
]
# Callable pages can only be targeted through their registered Streamlit page
# object. Diagnostics uses this to update a model and take the user straight to
# its design without losing the current browser session.
st.session_state["_model_page"] = model_page
nav = st.navigation({"Workflow": pages})

with st.sidebar:
    if not split_is_ready:
        st.caption("Complete the train / holdout split on Variables to unlock:")
        for title in (
            "Explore",
            "Model",
            "Diagnostics",
            "Compare",
            "Rate tables",
            "Export",
        ):
            st.markdown(f":grey[○ {title}]")
        st.divider()
    p = S.project()
    project_path = st.session_state.project_path
    # Context only. Project actions live on Project & data; repeating save/open
    # controls here made the sidebar look like a second project editor.
    st.caption("Current project")
    st.markdown(f"**{p.name}**")
    if project_path:
        st.caption(f"Autosaved · {Path(project_path).name}")
    else:
        st.caption("Not saved")
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
    if st.session_state.get("conflict"):
        st.error("Autosave paused: the project file changed on disk (see the notice).")
    elif any(e.startswith("Autosave") for e in st.session_state.get("errors", [])):
        st.error("Autosave is failing — edits are not being saved.")
    st.caption(S.persistence_note())

nav.run()
