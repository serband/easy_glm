"""Page 8 — Rate tables: inspect, adjust relativities (rows or cells), export."""

from __future__ import annotations

import pandas as pd
import polars as pl
import streamlit as st

from easy_glm.core.excel import rate_model_tables
from easy_glm.engine.models import level_label
from easy_glm.workflow import ae_by_pair, ae_by_variable, totals

from . import charts as C
from . import grids as G
from . import state as S
from . import ui


def _knots_and_levels(run) -> tuple[dict, dict]:
    knots: dict[str, list[float]] = {}
    levels: dict[str, list[str]] = {}
    for v in run.spec.main_effects:
        enc = run.spec[v]
        if hasattr(enc, "band_edges"):
            knots[v] = enc.band_edges()
        elif hasattr(enc, "levels"):
            levels[v] = list(enc.levels)
    return knots, levels


def _ae_frame(df: pl.DataFrame, which: str) -> pl.DataFrame:
    p = S.project()
    return df.filter(pl.col(p.data.split.column) == (0 if which == "holdout" else 1))


def _apply(run_name: str, changed: bool, errors: list[str]) -> None:
    for e in errors:
        st.error(e)
    if changed:
        S.touch()
        S.refresh_adjustments(run_name)
        st.rerun()


# --------------------------------------------------------------------------
# main-effect tables (step / categorical / linear)
# --------------------------------------------------------------------------
def _main_effect(run, var: str, df: pl.DataFrame) -> pl.DataFrame:
    p = S.project()
    cfg = p.models[run.name]
    rm = run.rate_model
    fitted = run.tables[var]
    working = rate_model_tables(rm)[var]
    rows = rm.variables[var].table
    kind = rm.variables[var].type
    is_linear = kind == "linear"

    left, right = st.columns([3, 2])
    with left:
        if is_linear:
            enc = run.spec[var]
            st.plotly_chart(
                C.linear_curve_chart(
                    fitted,
                    title=f"{var} — relativity curve (log-linear inside each band)",
                    working=working if cfg.adjustments else None,
                    clamp=(enc.lo, enc.hi),
                    x_base=rm.variables[var].x_base,
                ),
                width="stretch",
            )
        else:
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
        frame = _ae_frame(df, which)
        if not frame.is_empty():
            actual, expected, w = totals(frame, cfg, run.predict(frame))
            enc = run.spec[var]
            knots = enc.band_edges() if hasattr(enc, "band_edges") else None
            tbl = ae_by_variable(frame, var, actual, expected, w, knots=knots)
            st.plotly_chart(
                C.ae_chart(
                    tbl,
                    title=f"{var} — actual vs expected with current relativities ({which})",
                ),
                width="stretch",
            )
    with right:
        if is_linear:
            st.markdown(
                "**Edit the curve** — each row is a node: change the relativity at "
                "the band **start** and the slopes on either side follow so the curve "
                "stays continuous (the two flat end rows and the null row are steps). "
                "Saved as adjustments; no refit."
            )
            grid = pd.DataFrame(
                {
                    "band": [level_label(r) for r in rows],
                    "fitted": fitted["relativity"].to_list(),
                    "working": [r.relativity for r in rows],
                    "at band end": working["relativity_to"].to_list(),
                    "slope": working["slope"].to_list(),
                }
            )
            disabled = ["band", "fitted", "at band end", "slope"]
            col_cfg = {
                "fitted": st.column_config.NumberColumn(format="%.4f"),
                "working": st.column_config.NumberColumn(
                    "working (at band start)", format="%.4f", min_value=1e-4, step=0.01
                ),
                "at band end": st.column_config.NumberColumn(format="%.4f"),
                "slope": st.column_config.NumberColumn(
                    "slope (log per unit)", format="%.6f"
                ),
            }
        else:
            st.markdown(
                "**Edit relativities** (changes are saved as adjustments in the project "
                "and applied without refitting)"
            )
            grid = pd.DataFrame(
                {
                    "bin": [level_label(r) for r in rows],
                    "fitted": fitted["relativity"].to_list(),
                    "working": [r.relativity for r in rows],
                }
            )
            disabled = ["bin", "fitted"]
            col_cfg = {
                "fitted": st.column_config.NumberColumn(format="%.4f"),
                "working": st.column_config.NumberColumn(
                    format="%.4f", min_value=0.0, step=0.01
                ),
            }
        edited = st.data_editor(
            grid,
            hide_index=True,
            width="stretch",
            height=min(38 * (len(rows) + 1) + 4, 560),
            disabled=disabled,
            column_config=col_cfg,
            key=f"rel_editor_{run.name}_{var}",
        )
        changed, errors = G.apply_row_edits(
            cfg,
            var,
            rows,
            fitted["relativity"].to_list(),
            edited["working"].tolist(),
            require_positive=is_linear,
        )
        _apply(run.name, changed, errors)
    return working


# --------------------------------------------------------------------------
# interaction tables
# --------------------------------------------------------------------------
def _interaction(run, var: str, df: pl.DataFrame) -> pl.DataFrame:
    p = S.project()
    cfg = p.models[run.name]
    rm = run.rate_model
    grid = G.cell_grid(rm, var)
    a, b = grid["parents"]
    st.caption(
        f"Cells multiply the two main effects **{a}** and **{b}**; 1.00 means no "
        "adjustment — either the fit found none or the cell had too little exposure "
        "(hover shows the training exposure)."
    )
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            C.matrix_heatmap(
                grid["rows"],
                grid["cols"],
                grid["current"],
                title=f"{var} — cell adjustments (current)",
                row_name=a,
                col_name=b,
                hover={"exposure": grid["exposure"], "fitted": grid["fitted"]},
            ),
            width="stretch",
        )
        which = st.radio(
            "A/E rows", ["holdout", "train"], horizontal=True, key="tables_ae_rows"
        )
        frame = _ae_frame(df, which)
        if not frame.is_empty():
            actual, expected, w = totals(frame, cfg, run.predict(frame))
            knots, levels = _knots_and_levels(run)
            tbl = ae_by_pair(
                frame,
                a,
                b,
                actual,
                expected,
                w,
                knots_a=knots.get(a),
                knots_b=knots.get(b),
                levels_a=levels.get(a),
                levels_b=levels.get(b),
            )
            m = G.pair_matrices(tbl)
            st.plotly_chart(
                C.matrix_heatmap(
                    m["rows"],
                    m["cols"],
                    m["ae"],
                    title=f"{var} — actual / expected by cell ({which})",
                    row_name=a,
                    col_name=b,
                    hover={
                        "actual": m["actual"],
                        "expected": m["expected"],
                        "exposure": m["exposure"],
                    },
                ),
                width="stretch",
            )
    with right:
        st.markdown(
            f"**Edit cells** — rows are **{a}**, columns **{b}**. "
            "Saved as cell adjustments; no refit."
        )
        frame_grid = pd.DataFrame(
            grid["current"], index=grid["rows"], columns=grid["cols"]
        )
        edited = st.data_editor(
            frame_grid,
            width="stretch",
            height=min(38 * (len(grid["rows"]) + 1) + 4, 560),
            column_config={
                c: st.column_config.NumberColumn(
                    format="%.3f", min_value=0.0, step=0.01
                )
                for c in grid["cols"]
            },
            key=f"cell_editor_{run.name}_{var}",
        )
        changed, errors = G.apply_cell_edits(cfg, var, grid, edited.values.tolist())
        _apply(run.name, changed, errors)
        with st.expander("Fitted cells (before adjustments)"):
            st.dataframe(
                pd.DataFrame(grid["fitted"], index=grid["rows"], columns=grid["cols"]),
                width="stretch",
            )
    return rate_model_tables(rm)[var]


# --------------------------------------------------------------------------
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
    n_inter = sum(
        1 for c in run.rate_model.variables.values() if c.type == "interaction"
    )
    with c2:
        ui.metric_row(
            [
                (
                    "base rate",
                    ui.fmt(run.rate_model.base_rate, digits=6),
                    "prediction for the base risk: relativity 1.0 on every main "
                    "effect; interaction cells are adjustments on top (1.00 = none)",
                ),
                ("variables", str(len(run.rate_model.variables) - n_inter), None),
                ("interactions", str(n_inter), None),
                ("manual adjustments", str(len(cfg.adjustments)), None),
            ]
        )
    variables = list(run.rate_model.variables)

    def _display(v: str) -> str:
        kind = run.rate_model.variables[v].type
        return f"{v}  ({kind})" if kind in ("interaction", "linear") else v

    display = {_display(v): v for v in variables}
    chosen = st.selectbox("Variable", list(display), key="tables_var")
    var = display[chosen]
    missing = [c for c in run.spec.required_columns if c not in df.columns]
    if missing:
        st.error(
            "The prepared data no longer has columns the model needs: "
            + ", ".join(missing)
            + ". Refit on the Model page after fixing the Variables page."
        )
        return
    if run.rate_model.variables[var].type == "interaction":
        working = _interaction(run, var, df)
    else:
        working = _main_effect(run, var, df)

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
                            "row": (
                                f"{a.from_} – {a.to_}"
                                if not a.cell
                                else f"{a.from_} – {a.to_} | {a.from_b} – {a.to_b}"
                            ),
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
        help="Current (adjusted) tables; interactions get a long sheet and a matrix sheet",
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
        help="The stand-alone editor with snapshots and version history (main effects only)",
    ):
        run.rate_model.launch_editor(
            data=df,
            port=8502,
            formula=(
                "sum_over_weight" if cfg.divide_target_by_weight else "sum_weighted"
            ),
        )
        st.info("Editor starting on http://localhost:8502 …")
