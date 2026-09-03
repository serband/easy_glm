"""Page 10 — Export: Python script, HTML report, project JSON, artefacts."""

from __future__ import annotations

import json

import streamlit as st

from easy_glm.workflow import to_report_html, to_script

from . import state as S
from . import ui


def render() -> None:
    st.title("Export")
    ui.status_bar()
    p = S.project()
    if ui.require_data() is None:
        return
    if not p.models:
        st.info("Create and fit a model first.")
        return
    names = list(p.models)
    default = p.champion if p.champion in names else names[0]
    name = st.selectbox(
        "Model", names, index=names.index(default), key=S.widget_key("export_model")
    )
    run = S.get_run(name)
    if run is not None and run.rate_model.relativity_label != "relativity":
        st.info(ui.relativity_note_markdown(run.rate_model))
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
        key=S.widget_key("dl_script"),
    )
    c2.download_button(
        "Project JSON",
        json.dumps(p.to_dict(), indent=2, default=str),
        file_name=f"{p.name}.easyglm-project.json",
        key=S.widget_key("dl_project"),
    )
    if run is not None:
        c3.download_button(
            "Excel rate tables",
            ui.excel_bytes(run),
            file_name=f"{p.name}_{name}_rate_tables.xlsx",
            key=S.widget_key("dl_xlsx2"),
        )
        c4.download_button(
            "Scorer (.easyglm)",
            ui.easyglm_bytes(run),
            file_name=f"{p.name}_{name}.easyglm",
            key=S.widget_key("dl_easyglm2"),
        )
    _report(name, run)
    with st.expander("Project JSON"):
        st.json(p.to_dict(), expanded=False)


def _report(name: str, run) -> None:
    """One self-contained HTML file: summary, every rating factor with its A/E
    on train and holdout, interactions, lift, the coefficients and this script —
    plus a comparison section when a challenger is chosen."""
    p = S.project()
    st.subheader("HTML report")
    if run is None:
        st.caption("Fit the model to get the report.")
        return
    fitted = [n for n in S.fitted_models() if n != name]
    sidebar = S.challenger()
    options = ["(none)"] + fitted
    default = sidebar if sidebar in fitted else "(none)"
    c1, c2 = st.columns([2, 3])
    chal = c1.selectbox(
        "Include a comparison with",
        options,
        index=options.index(default),
        key=S.widget_key(f"report_chal_{name}_{sidebar}"),
        help="Adds a section with the double lift and every relativity that "
        "differs between the two models.",
    )
    with c2:
        st.caption(
            "One file, nothing loaded from the internet: the summary and split, "
            "one block per rating factor (relativities, actual vs expected on "
            "train and on holdout, the rate table), interaction heatmaps, lift "
            "and Gini, the coefficients and the script above. Open it in any "
            "browser or attach it to a filing."
        )
    df = S.prepared_frame()
    runs = {
        n: r
        for n in [name] + ([chal] if chal != "(none)" else [])
        if (r := S.get_run(n)) is not None
    }
    if df is None or name not in runs:
        st.caption("The data must be loaded to build the report.")
        return
    html = ui.guarded(
        lambda: to_report_html(
            p,
            runs,
            df,
            champion=name,
            challenger=chal if chal != "(none)" and chal in runs else None,
        ),
        "Building the report",
    )
    if html is None:
        return
    st.download_button(
        "Download HTML report",
        html,
        file_name=f"{ui.safe_filename(p.name, 'project')}_{ui.safe_filename(name)}"
        "_report.html",
        mime="text/html",
        type="primary",
        key=S.widget_key("dl_report"),
        help=f"{len(html.encode()) / 1024:,.0f} kB",
    )
