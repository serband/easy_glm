"""Page 1 — Project & data."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

import easy_glm
from easy_glm.workflow import Project, column_summary, infer_source_type

from . import state as S
from . import ui

DATA_SUFFIX = ".easyglm-data"
FRENCH_MOTOR_SAMPLE = "french_motor_sample.parquet"
FRENCH_MOTOR_ROWS = 50_000
SWEDISH_MOTORCYCLE_SAMPLE = "swedish_motorcycle_sample.parquet"


def open_project_file(path: str) -> str | None:
    """Load ``path`` as the session project. Returns an error message (and
    leaves the current project untouched) for anything that is not a valid
    easy_glm project file: missing, binary, truncated JSON, wrong shape,
    newer version, wrong field types."""
    p = Path(path)
    if not p.exists():
        return f"{path} does not exist"
    if p.is_dir():
        return f"{path} is a folder, not a project file"
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return f"Not a valid easy_glm project: {p.name} is not a text/JSON file"
    except json.JSONDecodeError as exc:
        return (
            f"Not a valid easy_glm project: {p.name} is not valid JSON "
            f"({exc.msg} at line {exc.lineno})"
        )
    except OSError as exc:
        return f"Could not read {path}: {exc}"
    if not isinstance(raw, dict):
        return (
            f"Not a valid easy_glm project: {p.name} does not contain a project object"
        )
    try:
        project = Project.from_dict(raw)
    except Exception as exc:  # noqa: BLE001 - version, types, shape: one message
        return f"Not a valid easy_glm project: {exc}"
    S.set_project(project, str(p))
    ui.flash("success", f"Opened {path}")
    return None


def _store_upload(upload, folder: Path) -> Path:
    """Write an uploaded data file into ``folder`` and return its path."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / Path(upload.name).name
    target.write_bytes(upload.getvalue())
    return target


def _data_folder() -> tuple[Path, bool]:
    """Where uploads go: ``<project>.easyglm-data/`` next to a saved project,
    else a temporary folder (second value: whether it is temporary)."""
    path = st.session_state.get("project_path")
    if path:
        p = Path(path)
        stem = (
            p.name[: -len(S.PROJECT_SUFFIX)]
            if p.name.endswith(S.PROJECT_SUFFIX)
            else p.stem
        )
        return p.parent / f"{stem}{DATA_SUFFIX}", False
    return Path(tempfile.mkdtemp(prefix="easy_glm_upload_")), True


def _load_french_motor_sample(p: Project) -> str | None:
    """Load the cached public sample and make it this project's file source.

    The workbench deliberately stores a Parquet copy because projects persist
    a file path, not an in-memory frame. It lets the project be saved, reopened
    and used by every workbench page without downloading again.
    """
    try:
        frame = easy_glm.load_external_dataframe()
        sample = frame.sample(n=min(FRENCH_MOTOR_ROWS, frame.height), seed=42)
        folder, _temporary = _data_folder()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / FRENCH_MOTOR_SAMPLE
        sample.write_parquet(path)
    except Exception as exc:  # noqa: BLE001 - show a friendly page message
        return (
            "Could not load the French motor sample. Check your internet connection "
            f"and try again. Details: {exc}"
        )

    p.data.source.path = str(path)
    p.data.source.type = "parquet"
    p.data.roles = {
        "ClaimNb": "target",
        "Exposure": "weight",
        "IDpol": "id",
        "DrivAge": "predictor",
        "Region": "predictor",
        "BonusMalus": "predictor",
        "Density": "predictor",
    }
    p.data.split.mode = "random"
    p.data.split.column = "traintest"
    p.data.split.fraction = 0.7
    p.data.split.seed = 42
    # Give first-time users a complete, editable model specification, but never
    # fit it here. ``new_model`` deliberately inherits the just-assigned roles
    # (target and weight); spelling out the other values keeps the sample's
    # starter configuration obvious and independent of future defaults.
    if not p.models:
        p.new_model(
            "frequency",
            family="poisson",
            divide_target_by_weight=True,
            predictors=["DrivAge", "Region", "BonusMalus", "Density"],
        )
        st.session_state.model_current = "frequency"
    S.touch()
    if S.raw_frame(force=True) is None:
        return st.session_state.get(
            "load_error", "Could not open the saved French motor sample."
        )
    return None


def _load_swedish_motorcycle_sample(p: Project) -> str | None:
    """Load the public motorcycle portfolio with a ready burn-cost setup."""
    try:
        frame = easy_glm.load_swedish_motorcycle_data()
        folder, _temporary = _data_folder()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / SWEDISH_MOTORCYCLE_SAMPLE
        frame.write_parquet(path)
    except Exception as exc:  # noqa: BLE001 - show a friendly page message
        return (
            "Could not load the Swedish motorcycle sample. Check your internet "
            f"connection and try again. Details: {exc}"
        )

    predictors = [
        "OwnerAge",
        "Gender",
        "Area",
        "RiskClass",
        "VehAge",
        "BonusClass",
    ]
    p.data.source.path = str(path)
    p.data.source.type = "parquet"
    p.data.roles = {
        "ClaimAmount": "target",
        "Exposure": "weight",
        "ClaimNb": "ignore",
        **dict.fromkeys(predictors, "predictor"),
    }
    p.data.filters = ["pl.col('Exposure') > 0"]
    p.data.split.mode = "random"
    p.data.split.column = "traintest"
    p.data.split.fraction = 0.7
    p.data.split.seed = 42
    if not p.models:
        p.new_model(
            "BurnCost",
            family="tweedie",
            tweedie_power=1.5,
            divide_target_by_weight=True,
            predictors=predictors,
        )
        st.session_state.model_current = "BurnCost"
    S.touch()
    if S.raw_frame(force=True) is None:
        return st.session_state.get(
            "load_error", "Could not open the saved Swedish motorcycle sample."
        )
    return None


def _new_project_button(
    p: Project, *, label: str, key: str, width: str = "stretch"
) -> None:
    """Render a two-click, loss-aware route back to an empty project."""
    has_work = bool(p.models or p.data.source.path or p.data.roles)
    confirm = st.session_state.get("confirm_new_project", False)
    button_label = "Confirm: start over" if confirm else label
    if not st.button(button_label, width=width, key=S.widget_key(key)):
        return
    if has_work and not confirm:
        st.session_state.confirm_new_project = True
        if st.session_state.get("project_path"):
            ui.flash(
                "warning",
                "This closes the current project. Its saved setup remains in its "
                "project file, and the new project starts empty. Click **Confirm: "
                "start over** to continue.",
            )
        else:
            ui.flash(
                "warning",
                "This project has not been saved. Save it first if you want to keep "
                "the current setup, or click **Confirm: start over** to discard it "
                "and return to the sample choices.",
            )
        st.rerun()
    st.session_state.confirm_new_project = False
    S.set_project(Project(name="untitled"), None)
    ui.flash("info", "New empty project — choose a sample or your own data below.")
    st.rerun()


def _project_section(p: Project) -> None:
    with st.container(border=True):
        st.subheader("Project")
        c1, c2 = st.columns([2, 3])
        name = c1.text_input("Project name", p.name, key=S.widget_key("proj_name"))
        if name != p.name and name.strip():
            p.name = name.strip()
            S.touch()
        path_default = st.session_state.project_path or S.default_project_path(p)
        proj_path = c2.text_input(
            "Project file (autosaved after every change)",
            path_default,
            key=S.widget_key("proj_path"),
        )
        b1, b2, b3 = st.columns(3)
        if b1.button("Save project", type="primary", width="stretch"):
            err = S.save_project(proj_path)
            if err:
                st.error(err)
            else:
                ui.flash("success", f"Saved {proj_path}")
                st.rerun()
        if b2.button("Open project file", width="stretch"):
            err = open_project_file(proj_path)
            if err:
                st.error(err)
            else:
                st.rerun()
        with b3:
            _new_project_button(
                p, label="Start over / new project", key="new_project_btn"
            )
        st.info(
            "Resuming previous work? Open an **EasyGLM project file** here. New to "
            "EasyGLM? You do not need one: use either built-in insurance sample "
            "below or choose a data file instead."
        )
        st.caption(
            "A project file saves the workbench setup: the data-file location, "
            "column roles and types, renames, recodes, derived columns and filters, "
            "the train/test split, design choices, model definitions and rate-table "
            "adjustments (including named snapshots). It does not contain the data "
            "itself or fitted results."
        )
        uploaded = st.file_uploader(
            "Open an existing EasyGLM project",
            type=["json"],
            key=S.widget_key("proj_upload"),
        )
        if uploaded is not None and st.button("Open EasyGLM project"):
            tmp = Path(tempfile.mkdtemp(prefix="easy_glm_project_")) / uploaded.name
            tmp.write_bytes(uploaded.getvalue())
            err = open_project_file(str(tmp))
            if err:
                st.error(err)
            else:
                # the uploaded copy lives in a temp folder: do not autosave there
                st.session_state.project_path = None
                st.session_state.project_stamp = None
                ui.flash(
                    "warning",
                    "Project loaded from an upload; choose a project file path and "
                    "save it so autosave has somewhere to go.",
                )
                st.rerun()


def _data_section(p: Project) -> None:
    with st.container(border=True):
        st.subheader("Data source")
        source_name = Path(p.data.source.path).name if p.data.source.path else ""
        if not p.data.source.path:
            st.info(
                "New here? Start with a public insurance sample. Each option loads "
                "the data, assigns sensible roles and prepares an editable starter "
                "model. Nothing is fitted automatically."
            )
            french, swedish = st.columns(2)
            with french:
                st.caption(
                    "50,000 motor policies with claim count, exposure and four "
                    "common rating factors."
                )
                french_clicked = st.button(
                    "French motor sample (Poisson frequency)",
                    type="primary",
                    width="stretch",
                    key=S.widget_key("french_motor_sample_btn"),
                )
            with swedish:
                st.caption(
                    "Motorcycle policies with claim amount, exposure and six rating "
                    "factors."
                )
                swedish_clicked = st.button(
                    "Swedish motorcycle sample (Tweedie burn cost)",
                    width="stretch",
                    key=S.widget_key("swedish_motorcycle_sample_btn"),
                )
            if french_clicked:
                err = _load_french_motor_sample(p)
                if err:
                    st.error(err)
                else:
                    ui.flash(
                        "success",
                        "French motor sample loaded. Review the preview, then visit "
                        "Variables and Split when you are ready. A Poisson frequency "
                        "model is prepared on the Model page; review it there, then "
                        "click Fit model when you want to run it.",
                    )
                    st.rerun()
            if swedish_clicked:
                err = _load_swedish_motorcycle_sample(p)
                if err:
                    st.error(err)
                else:
                    ui.flash(
                        "success",
                        "Swedish motorcycle sample loaded. Review the preview, roles "
                        "and split. A Tweedie burn-cost model is prepared on the Model "
                        "page; review it there, then click Fit model when you want to "
                        "run it.",
                    )
                    st.rerun()
        elif source_name == FRENCH_MOTOR_SAMPLE:
            st.caption(
                "French motor sample: ClaimNb is the claim count; Exposure is the "
                "weight; DrivAge, Region, BonusMalus and Density are predictors. "
                "A 70/30 random train/test split (seed 42) is ready to review."
            )
        elif source_name == SWEDISH_MOTORCYCLE_SAMPLE:
            st.caption(
                "Swedish motorcycle sample: ClaimAmount is incurred cost; Exposure "
                "is the weight; ClaimNb is ignored; six rating factors feed a Tweedie "
                "burn-cost model. Zero-exposure rows are filtered and a 70/30 random "
                "train/test split (seed 42) is ready to review."
            )
        if source_name in {FRENCH_MOTOR_SAMPLE, SWEDISH_MOTORCYCLE_SAMPLE}:
            st.caption(
                "Finished exploring this sample? Start over to return to both sample "
                "choices. Save the current project first if you want to keep it."
            )
            _new_project_button(
                p,
                label="Start over and choose another sample",
                key="choose_another_sample_btn",
            )
        st.caption(
            "Point at a local file (fastest for large data) or upload one. "
            "Supported: parquet, csv, sas7bdat, xlsx, arrow/ipc."
        )
        c1, c2 = st.columns([4, 1])
        src_path = c1.text_input(
            "File path", p.data.source.path, key=S.widget_key("src_path")
        )
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
            key=S.widget_key("sample_rows"),
        )
        if (int(sample) or None) != p.data.sample_rows:
            p.data.sample_rows = int(sample) or None
            S.touch()
        up = c4.file_uploader(
            "…or upload a data file",
            type=["parquet", "csv", "sas7bdat", "xlsx"],
            key=S.widget_key("data_upload"),
        )
        if up is not None:
            folder, temporary = _data_folder()
            src_path = str(_store_upload(up, folder))
            src_type = infer_source_type(src_path)
            st.caption(
                f"Upload stored at `{src_path}`"
                + (
                    " — a temporary folder because the project is unsaved; save the "
                    "project first to keep uploads next to it."
                    if temporary
                    else " (next to the project file)."
                )
            )
        if st.button("Load data", type="primary"):
            if not src_path.strip():
                st.error("Enter a file path or upload a file first")
            else:
                p.data.source.path = src_path.strip()
                p.data.source.type = src_type
                S.touch()
                S.raw_frame(force=True)
                st.rerun()
        err = st.session_state.get("load_error")
        if err and p.data.source.path:
            st.error(err)


def _preview(p: Project) -> None:
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
        prep_err = st.session_state.get("prep_error")
        if prep_err:
            st.error(prep_err)
        st.caption(
            "Next: assign roles and clean up variables on the **Variables** page."
        )


def render() -> None:
    p = S.project()
    st.title("Project & data")
    ui.status_bar()
    _project_section(p)
    _data_section(p)
    _preview(p)
