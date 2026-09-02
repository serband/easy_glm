"""Page 9 — Export: Python script, project JSON, artefacts."""

from __future__ import annotations

import json

import streamlit as st

from easy_glm.workflow import to_script

from . import state as S
from . import ui


def render() -> None:
    st.title("Export")
    ui.status_bar()
    p = S.project()
    if not p.models:
        st.info("Create and fit a model first.")
        return
    names = list(p.models)
    default = p.champion if p.champion in names else names[0]
    name = st.selectbox("Model", names, index=names.index(default), key="export_model")
    run = S.get_run(name)
    if run is None:
        st.warning(
            "This model is not fitted (or its spec changed). The script below derives knots and levels from the data at run time; fit it to get every knot and level written out explicitly."
        )
    try:
        src = to_script(p, name, run=run, output_prefix=f"{p.name}_{name}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cannot render the script: {exc}")
        return
    st.subheader("Python script")
    st.caption(
        "A linear, readable script using only the public easy_glm API. Re-running it rebuilds the model, the rate tables and the .easyglm scorer."
    )
    st.code(src, language="python", line_numbers=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button(
        "Download script (.py)",
        src,
        file_name=f"{p.name}_{name}.py",
        type="primary",
        key="dl_script",
    )
    c2.download_button(
        "Project JSON",
        json.dumps(p.to_dict(), indent=2, default=str),
        file_name=f"{p.name}.easyglm-project.json",
        key="dl_project",
    )
    if run is not None:
        c3.download_button(
            "Excel rate tables",
            ui.excel_bytes(run),
            file_name=f"{p.name}_{name}_rate_tables.xlsx",
            key="dl_xlsx2",
        )
        c4.download_button(
            "Scorer (.easyglm)",
            ui.easyglm_bytes(run),
            file_name=f"{p.name}_{name}.easyglm",
            key="dl_easyglm2",
        )
    with st.expander("Project JSON"):
        st.json(p.to_dict(), expanded=False)
