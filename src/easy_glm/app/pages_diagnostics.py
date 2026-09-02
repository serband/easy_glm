"""Page 7 — Diagnostics: A/E by variable, lift, double lift, residual factors."""

from __future__ import annotations

import numpy as np
import polars as pl
import streamlit as st

from easy_glm.workflow import (
    ModelRun,
    ae_by_variable,
    alpha_path,
    double_lift,
    gini,
    lift_table,
    residual_factor_search,
    totals,
)

from . import charts as C
from . import state as S
from . import ui


def _subset(df: pl.DataFrame, which: str) -> pl.DataFrame:
    col = S.project().data.split.column
    if which == "train":
        return df.filter(pl.col(col) == 1)
    if which == "holdout":
        return df.filter(pl.col(col) == 0)
    return df


def _metrics(run: ModelRun, challenger: ModelRun | None) -> None:
    rows = []
    for r in [run] + ([challenger] if challenger else []):
        for subset, m in r.metrics.items():
            rows.append(
                {
                    "model": r.name,
                    "subset": subset,
                    "rows": int(m["rows"]),
                    "exposure": m["exposure"],
                    "A/E": m["ae"],
                    "Gini": m["gini"],
                    "deviance explained": m["deviance_explained"],
                    "mean deviance": m["mean_deviance"],
                }
            )
    st.dataframe(
        pl.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "exposure": st.column_config.NumberColumn(format="%.0f"),
            "A/E": st.column_config.NumberColumn(format="%.4f"),
            "Gini": st.column_config.NumberColumn(format="%.4f"),
            "deviance explained": st.column_config.NumberColumn(format="percent"),
            "mean deviance": st.column_config.NumberColumn(format="%.5f"),
        },
    )


def render() -> None:
    st.title("Diagnostics")
    ui.status_bar()
    p = S.project()
    df = ui.require_data()
    if df is None:
        return
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        run = ui.run_selector("Model", key="diag_run")
    if run is None:
        return
    fitted = [n for n in S.current_runs() if S.get_run(n) is not None and n != run.name]
    with c2:
        chal_name = st.selectbox(
            "Compare with (challenger)", ["(none)"] + fitted, key="diag_chal"
        )
    challenger = S.get_run(chal_name) if chal_name != "(none)" else None
    with c3:
        which = st.radio(
            "Rows", ["holdout", "train", "all"], horizontal=True, key="diag_subset"
        )
    frame = _subset(df, which)
    if frame.is_empty():
        st.warning("No rows in this subset.")
        return
    cfg = run.config
    pred = run.predict(frame)
    actual, expected, w = totals(frame, cfg, pred)
    exp_chal = (
        totals(frame, challenger.config, challenger.predict(frame))[1]
        if challenger
        else None
    )

    _metrics(run, challenger)
    tabs = st.tabs(
        [
            "A/E by variable",
            "Lift",
            "Double lift",
            "Residual factors",
            "Regularisation path",
        ]
    )

    with tabs[0]:
        reserved = {cfg.target, cfg.weight, cfg.offset, p.data.split.column} - {None}
        variables = [c for c in frame.columns if c not in reserved]
        in_model = [v for v in variables if v in cfg.predictors]
        others = [v for v in variables if v not in cfg.predictors]
        c1, c2 = st.columns([3, 1])
        var = c1.selectbox(
            "Variable",
            in_model + others,
            format_func=lambda v: v if v in cfg.predictors else f"{v} (not in model)",
            key="diag_var",
        )
        n_bins = c2.slider("Bands (numeric, not in model)", 5, 50, 20, key="diag_bins")
        knots = (
            run.spec[var].band_edges()
            if var in run.spec and hasattr(run.spec[var], "band_edges")
            else None
        )
        tbl = ae_by_variable(
            frame, var, actual, expected, w, n_bins=n_bins, knots=knots
        )
        cmp_tbl = (
            ae_by_variable(frame, var, actual, exp_chal, w, n_bins=n_bins, knots=knots)
            if exp_chal is not None
            else None
        )
        st.plotly_chart(
            C.ae_chart(
                tbl,
                title=f"{var} — actual vs expected ({which})",
                compare=cmp_tbl,
                compare_name=chal_name,
            ),
            width="stretch",
        )
        with st.expander("Table"):
            ui.polars_table(tbl)

    with tabs[1]:
        n = st.slider("Bins", 5, 20, 10, key="lift_bins")
        lt = lift_table(actual, expected, w, n_bins=n)
        g = gini(actual, expected, w)
        st.caption(
            f"Normalised Gini ({which}): **{g:.4f}**"
            + (
                f" · challenger: **{gini(actual, exp_chal, w):.4f}**"
                if exp_chal is not None
                else ""
            )
        )
        st.plotly_chart(
            C.lift_chart(
                lt, title=f"Lift ({which}) — equal-exposure bins by predicted rate"
            ),
            width="stretch",
        )
        with st.expander("Table"):
            ui.polars_table(lt)

    with tabs[2]:
        st.caption(
            "Sort policies by the ratio of two predictions; the model whose A/E stays closer to 1 across the bins wins."
        )
        options = (["challenger"] if exp_chal is not None else []) + [
            "a column (e.g. current premium)"
        ]
        pick = st.radio("Benchmark", options, horizontal=True, key="dl_pick")
        if pick == "challenger":
            exp_b, name_b = exp_chal, chal_name
        else:
            numeric_cols = [
                c
                for c, t in frame.schema.items()
                if t in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            ]
            col = st.selectbox(
                "Benchmark column (already on the same total scale, or per unit × weight)",
                numeric_cols,
                key="dl_col",
            )
            per_unit = st.checkbox(
                "Column is per unit of weight (multiply by weight)", True, key="dl_unit"
            )
            exp_b = frame[col].cast(pl.Float64).to_numpy() * (w if per_unit else 1.0)
            name_b = col
        if exp_b is not None and np.nansum(exp_b) > 0:
            dl = double_lift(actual, expected, exp_b, w, n_bins=10)
            st.plotly_chart(
                C.double_lift_chart(dl, name_a=run.name, name_b=name_b), width="stretch"
            )
            with st.expander("Table"):
                ui.polars_table(dl)

    with tabs[3]:
        st.caption(
            "Exposure-weighted spread of log(A/E) across the bands of each variable **not in the model**. "
            "Large values point at factors the model is missing."
        )
        reserved = {cfg.target, cfg.weight, cfg.offset, p.data.split.column} - {None}
        candidates = [
            c
            for c in frame.columns
            if c not in reserved
            and c not in cfg.predictors
            and p.data.roles.get(c) not in ("id",)
        ]
        if not candidates:
            st.info("Every available variable is already in the model (or is an id).")
        elif (
            st.button("Run residual search", key="rfs_go")
            or "rfs_result" in st.session_state
        ):
            res = residual_factor_search(frame, candidates, actual, expected, w)
            st.session_state.rfs_result = res
            st.dataframe(
                res,
                width="stretch",
                hide_index=True,
                column_config={
                    "signal": st.column_config.ProgressColumn(
                        "signal (sd of log A/E)",
                        min_value=0.0,
                        max_value=float(max(res["signal"].max() or 1.0, 1e-9)),
                        format="%.3f",
                    ),
                    "max_abs_log_ae": st.column_config.NumberColumn(
                        "max |log A/E|", format="%.3f"
                    ),
                },
            )
            if res.height:
                top = st.selectbox("Show", res["variable"].to_list(), key="rfs_show")
                t = ae_by_variable(frame, top, actual, expected, w, n_bins=10)
                st.plotly_chart(
                    C.ae_chart(t, title=f"{top} (not in model) — actual vs expected"),
                    width="stretch",
                )

    with tabs[4]:
        path = alpha_path(run.fit)
        if path.height > 1:
            st.plotly_chart(C.alpha_path_chart(path), width="stretch")
        ui.polars_table(path)
