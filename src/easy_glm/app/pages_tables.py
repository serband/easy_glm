"""Page 8 — Rate tables: inspect, adjust relativities, export."""

from __future__ import annotations

import pandas as pd
import polars as pl
import streamlit as st

from easy_glm.core.excel import rate_model_tables
from easy_glm.engine.models import level_label
from easy_glm.workflow import Adjustment, ae_by_variable, totals

from . import charts as C
from . import state as S
from . import ui


def _working_table(run, var: str) -> pl.DataFrame:
    return rate_model_tables(run.rate_model)[var]


def render() -> None:
    st.title("Rate tables")
    ui.status_bar()
    p = S.project()
    df = ui.require_data()
    if df is None:
        return
    c1, c2 = st.columns([2, 3])
    with c1:
        run = ui.run_selector("Model", key="tables_run")
    if run is None:
        return
    cfg = p.models[run.name]
    with c2:
        ui.metric_row(
            [
                (
                    "base rate",
                    ui.fmt(run.rate_model.base_rate, digits=6),
                    "prediction for the base risk (relativity 1.0 everywhere)",
                ),
                ("variables", str(len(run.rate_model.variables)), None),
                ("manual adjustments", str(len(cfg.adjustments)), None),
            ]
        )

    variables = list(run.rate_model.variables)
    var = st.selectbox("Variable", variables, key="tables_var")
    fitted = run.tables[var]
    working = _working_table(run, var)
    rows = run.rate_model.variables[var].table

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            C.relativity_chart(
                fitted,
                title=f"{var} — relativities",
                working=working if cfg.adjustments else None,
            ),
            width="stretch",
        )
        which = st.radio(
            "A/E rows", ["holdout", "train"], horizontal=True, key="tables_ae_rows"
        )
        frame = df.filter(
            pl.col(p.data.split.column) == (0 if which == "holdout" else 1)
        )
        if not frame.is_empty():
            actual, expected, w = totals(frame, cfg, run.predict(frame))
            knots = run.spec[var].knots if hasattr(run.spec[var], "knots") else None
            tbl = ae_by_variable(frame, var, actual, expected, w, knots=knots)
            st.plotly_chart(
                C.ae_chart(
                    tbl,
                    title=f"{var} — actual vs expected with current relativities ({which})",
                ),
                width="stretch",
            )
    with right:
        st.markdown(
            "**Edit relativities** (changes are saved as adjustments in the project and applied without refitting)"
        )
        grid = pd.DataFrame(
            {
                "bin": [level_label(r) for r in rows],
                "fitted": fitted["relativity"].to_list(),
                "working": [r.relativity for r in rows],
            }
        )
        edited = st.data_editor(
            grid,
            hide_index=True,
            width="stretch",
            height=min(38 * (len(rows) + 1) + 4, 560),
            disabled=["bin", "fitted"],
            column_config={
                "fitted": st.column_config.NumberColumn(format="%.4f"),
                "working": st.column_config.NumberColumn(
                    format="%.4f", min_value=0.0, step=0.01
                ),
            },
            key=f"rel_editor_{run.name}_{var}",
        )
        changed = False
        for i, r in enumerate(rows):
            new = float(edited["working"].iloc[i])
            if abs(new - r.relativity) > 1e-9:
                cfg.adjustments = [
                    a
                    for a in cfg.adjustments
                    if not (a.variable == var and a.from_ == r.from_ and a.to_ == r.to_)
                ]
                if abs(new - float(fitted["relativity"][i])) > 1e-9:
                    cfg.adjustments.append(Adjustment(var, r.from_, r.to_, new))
                changed = True
        if changed:
            S.touch()
            S.refresh_adjustments(run.name)
            st.rerun()
        b1, b2 = st.columns(2)
        if b1.button(
            "Reset this variable",
            key="tables_reset_var",
            disabled=not any(a.variable == var for a in cfg.adjustments),
        ):
            cfg.adjustments = [a for a in cfg.adjustments if a.variable != var]
            S.touch()
            S.refresh_adjustments(run.name)
            st.rerun()
        if b2.button("Reset all", key="tables_reset_all", disabled=not cfg.adjustments):
            cfg.adjustments = []
            S.touch()
            S.refresh_adjustments(run.name)
            st.rerun()
        if cfg.adjustments:
            with st.expander(f"{len(cfg.adjustments)} adjustment(s)"):
                ui.polars_table(
                    pl.DataFrame(
                        [
                            {
                                "variable": a.variable,
                                "from": str(a.from_),
                                "to": str(a.to_),
                                "relativity": a.relativity,
                            }
                            for a in cfg.adjustments
                        ]
                    )
                )

    st.subheader("Export")
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Excel rate tables (.xlsx)",
        ui.excel_bytes(run),
        file_name=f"{p.name}_{run.name}_rate_tables.xlsx",
        key="dl_xlsx",
    )
    c2.download_button(
        "Scorer (.easyglm)",
        ui.easyglm_bytes(run),
        file_name=f"{p.name}_{run.name}.easyglm",
        key="dl_easyglm",
    )
    c3.download_button(
        "This table (.csv)",
        ui.frame_bytes(working),
        file_name=f"{run.name}_{var}.csv",
        key="dl_csv",
    )
    if st.button(
        "Open the full relativity editor in a new tab",
        help="The stand-alone editor with snapshots and version history",
    ):
        run.rate_model.launch_editor(
            data=df,
            port=8502,
            formula=(
                "sum_over_weight" if cfg.divide_target_by_weight else "sum_weighted"
            ),
        )
        st.info("Editor starting on http://localhost:8502 …")
