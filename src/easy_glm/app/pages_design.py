"""Page 5 — Design: how each predictor becomes GLM features."""

from __future__ import annotations

import pandas as pd
import polars as pl
import streamlit as st

from easy_glm.core.design import (
    NUMERIC_DTYPES,
    CategoricalEncoder,
    LinearEncoder,
    StepEncoder,
)
from easy_glm.workflow import VariableDesign, encoder_for, univariate

from . import charts as C
from . import state as S
from . import ui

KNOT_OPTIONS = ["quantile", "integer", "custom"]
KIND_OPTIONS = ["auto", "step", "linear", "categorical"]
MONO_OPTIONS = ["none", "increasing", "decreasing"]


def _defaults() -> None:
    p = S.project()
    d = p.design.defaults
    with st.expander("Defaults for every predictor", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        n_bins = c1.number_input("Quantile knots (n_bins)", 2, 200, int(d.n_bins))
        share = c2.number_input(
            "Min level share", 0.0, 0.5, float(d.min_level_share), 0.0005, format="%.4f"
        )
        null_ind = c3.checkbox("Null indicator column", bool(d.null_indicator))
        max_int = c4.number_input(
            "Max integer knots", 10, 1000, int(d.max_integer_knots)
        )
        if (n_bins, share, null_ind, max_int) != (
            d.n_bins,
            d.min_level_share,
            d.null_indicator,
            d.max_integer_knots,
        ):
            d.n_bins, d.min_level_share, d.null_indicator, d.max_integer_knots = (
                int(n_bins),
                float(share),
                bool(null_ind),
                int(max_int),
            )
            S.touch()
            st.rerun()


def _grid(train: pl.DataFrame, predictors: list[str]) -> None:
    p = S.project()
    rows = []
    for v in predictors:
        vd = p.design.variables.get(v, VariableDesign())
        numeric = v in train.columns and train[v].dtype in NUMERIC_DTYPES
        rows.append(
            {
                "variable": v,
                "dtype": str(train[v].dtype) if v in train.columns else "?",
                "kind": vd.kind or "auto",
                "knots": "custom" if isinstance(vd.knots, list) else vd.knots,
                "n_bins": vd.n_bins or 0,
                "null col": (
                    vd.null_indicator
                    if vd.null_indicator is not None
                    else p.design.defaults.null_indicator
                ),
                "min share": (
                    vd.min_level_share
                    if vd.min_level_share is not None
                    else p.design.defaults.min_level_share
                ),
                "monotone": vd.monotone or "none",
                "inferred": "step" if numeric else "categorical",
            }
        )
    edited = st.data_editor(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        height=min(38 * (len(rows) + 1) + 4, 520),
        disabled=["variable", "dtype", "inferred"],
        column_config={
            "kind": st.column_config.SelectboxColumn(
                "kind", options=KIND_OPTIONS, required=True
            ),
            "knots": st.column_config.SelectboxColumn(
                "knots",
                options=KNOT_OPTIONS,
                required=True,
                help="quantile: n_bins quantiles · integer: every integer · custom: edit below",
            ),
            "n_bins": st.column_config.NumberColumn(
                "n_bins (0 = default)", min_value=0, max_value=200, step=1
            ),
            "null col": st.column_config.CheckboxColumn("null col"),
            "min share": st.column_config.NumberColumn(
                "min level share",
                min_value=0.0,
                max_value=0.5,
                step=0.0005,
                format="%.4f",
            ),
            "monotone": st.column_config.SelectboxColumn(
                "monotone", options=MONO_OPTIONS, required=True
            ),
        },
        key="design_grid",
    )
    changed = False
    for _, r in edited.iterrows():
        v = r["variable"]
        vd = p.design.variables.get(v, VariableDesign())
        new = VariableDesign(
            kind=None if r["kind"] == "auto" else r["kind"],
            knots=(
                vd.knots
                if (r["knots"] == "custom" and isinstance(vd.knots, list))
                else (vd.knots if r["knots"] == "custom" else r["knots"])
            ),
            n_bins=int(r["n_bins"]) or None,
            null_indicator=(
                None
                if bool(r["null col"]) == p.design.defaults.null_indicator
                else bool(r["null col"])
            ),
            min_level_share=(
                None
                if abs(float(r["min share"]) - p.design.defaults.min_level_share)
                < 1e-12
                else float(r["min share"])
            ),
            max_levels=vd.max_levels,
            levels=vd.levels,
            clamp=vd.clamp,  # detail-panel fields the grid does not show
            monotone=None if r["monotone"] == "none" else r["monotone"],
        )
        if r["knots"] == "custom" and not isinstance(new.knots, list):
            new.knots = []  # to be filled in the detail panel
        if new != vd:
            if new == VariableDesign():
                p.design.variables.pop(v, None)
            else:
                p.design.variables[v] = new
            changed = True
    if changed:
        S.touch()
        st.rerun()


def _detail(train: pl.DataFrame, preview: pl.DataFrame, predictors: list[str]) -> None:
    p = S.project()
    st.subheader("Variable detail")
    c1, c2 = st.columns([2, 1])
    var = c1.selectbox("Variable", predictors, key="design_detail_var")
    vd = p.design.variables.get(var, VariableDesign())
    weights = train[p.weight] if p.weight and p.weight in train.columns else None
    try:
        enc = encoder_for(var, train[var], vd, p, weights=weights)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Cannot build the design for {var} yet: {exc}")
        if train[var].dtype in NUMERIC_DTYPES:
            from easy_glm.core.design import quantile_knots

            suggestion = quantile_knots(
                train[var], vd.n_bins or p.design.defaults.n_bins
            )
            knots_txt = c2.text_area(
                "Knots (comma-separated)",
                ", ".join(f"{k:g}" for k in suggestion),
                height=90,
                key=f"knots_{var}",
            )
            if st.button("Apply knots", key=f"apply_knots_{var}"):
                try:
                    vd.knots = sorted(
                        {
                            float(x)
                            for x in knots_txt.replace("\n", ",").split(",")
                            if x.strip()
                        }
                    )
                except ValueError:
                    st.error("Knots must be numbers")
                else:
                    p.design.variables[var] = vd
                    S.touch()
                    st.rerun()
        return
    cfg = p.models.get(p.champion) if p.champion else None
    divide = cfg.divide_target_by_weight if cfg else bool(p.weight)
    if isinstance(enc, StepEncoder):
        knots_txt = c2.text_area(
            "Knots (comma-separated; editing switches to custom)",
            ", ".join(f"{k:g}" for k in enc.knots),
            height=90,
            key=f"knots_{var}",
        )
        if st.button("Apply knots", key=f"apply_knots_{var}"):
            try:
                knots = sorted(
                    {
                        float(x)
                        for x in knots_txt.replace("\n", ",").split(",")
                        if x.strip()
                    }
                )
            except ValueError:
                st.error("Knots must be numbers")
            else:
                vd.knots = knots
                p.design.variables[var] = vd
                S.touch()
                st.rerun()
        u = univariate(
            preview,
            var,
            target=p.target,
            weight=p.weight,
            divide_target_by_weight=divide,
            knots=enc.knots,
        )
        st.plotly_chart(
            C.exposure_rate_chart(
                u["table"],
                title=f"{var}: {len(enc.knots)} knots → {len(enc.knots) + 1} bins (+ null)",
            ),
            width="stretch",
        )
        st.caption(
            f"Design columns: {enc.n_features} · bins: {len(enc.bins())} · null indicator: {enc.null_indicator}"
        )
    elif isinstance(enc, LinearEncoder):
        c2.markdown(
            f"**Piecewise-linear** · clamp `{enc.lo:g}` – `{enc.hi:g}` "
            f"(flat outside) · {len(enc.knots)} interior knot(s)"
        )
        knots_txt = st.text_area(
            "Knots where the slope may change (comma-separated; editing switches to custom)",
            ", ".join(f"{k:g}" for k in enc.knots),
            height=90,
            key=f"knots_{var}",
        )
        clamp_txt = st.text_input(
            "Clamp range lo, hi (blank = training min/max)",
            ", ".join(f"{v:g}" for v in vd.clamp) if vd.clamp else "",
            key=f"clamp_{var}",
        )
        if st.button("Apply knots / clamp", key=f"apply_knots_{var}"):
            try:
                knots = sorted(
                    {
                        float(x)
                        for x in knots_txt.replace("\n", ",").split(",")
                        if x.strip()
                    }
                )
                clamp = [float(x) for x in clamp_txt.split(",") if x.strip()]
            except ValueError:
                st.error("Knots and clamp must be numbers")
            else:
                if clamp and (len(clamp) != 2 or not clamp[0] < clamp[1]):
                    st.error("Clamp must be two numbers, lo < hi")
                else:
                    lo_c, hi_c = clamp if clamp else (enc.lo, enc.hi)
                    outside = [k for k in knots if not lo_c < k < hi_c]
                    if outside:
                        st.warning(
                            f"Knots outside the clamp range are dropped: "
                            f"{', '.join(f'{k:g}' for k in outside)} "
                            f"(clamp {lo_c:g} – {hi_c:g})"
                        )
                    vd.knots = knots
                    vd.clamp = clamp or None
                    p.design.variables[var] = vd
                    S.touch()
                    st.rerun()
        u = univariate(
            preview,
            var,
            target=p.target,
            weight=p.weight,
            divide_target_by_weight=divide,
            knots=enc.band_edges(),
        )
        st.plotly_chart(
            C.exposure_rate_chart(
                u["table"],
                title=f"{var}: linear in {len(enc.knots) + 1} band(s) between {enc.lo:g} and {enc.hi:g}",
            ),
            width="stretch",
        )
        st.caption(
            f"Design columns: {enc.n_features} · rows in the rate table: {enc.n_rows} · "
            f"null indicator: {enc.null_indicator}"
        )
    elif isinstance(enc, CategoricalEncoder):
        c2.markdown(
            f"**Reference level:** `{enc.reference}`  \n**Kept levels:** {len(enc.levels)} (+ Other)"
        )
        levels_txt = st.text_area(
            "Levels (first = reference; others lumped into Other)",
            ", ".join(enc.levels),
            height=90,
            key=f"levels_{var}",
        )
        if st.button("Apply levels", key=f"apply_levels_{var}"):
            levels = [
                x.strip() for x in levels_txt.replace("\n", ",").split(",") if x.strip()
            ]
            vd.levels = levels or None
            p.design.variables[var] = vd
            S.touch()
            st.rerun()
        u = univariate(
            preview,
            var,
            target=p.target,
            weight=p.weight,
            divide_target_by_weight=divide,
            max_levels=len(enc.levels),
        )
        st.plotly_chart(
            C.exposure_rate_chart(
                u["table"], title=f"{var}: {len(enc.levels)} levels + Other"
            ),
            width="stretch",
        )


def render() -> None:
    st.title("Design")
    ui.status_bar()
    p = S.project()
    df = ui.require_data()
    if df is None:
        return
    predictors = p.predictors
    if not predictors:
        st.info("Assign **predictor** roles on the Variables page first.")
        return
    train = S.train_frame()  # knots and levels always come from the full training rows
    preview = (
        S.train_sample()
    )  # exposure / rate previews may use the exploration sample
    if train is None or preview is None:
        return
    if S.is_sampled():
        st.caption(
            f"Knots and levels are derived from all {train.height:,} training rows; "
            f"the preview charts use the exploration sample ({preview.height:,} rows)."
        )
    _defaults()
    st.caption(
        "Numeric predictors become step functions (one 0/1 column per knot, penalised increments → automatic banding); "
        "categoricals become one-hot with the most frequent level as reference and an **Other** bucket. "
        "Monotone constraints bound the step increments."
    )
    _grid(train, predictors)
    total = 0
    for v in predictors:
        try:
            total += encoder_for(
                v, train[v], p.design.variables.get(v, VariableDesign()), p
            ).n_features
        except Exception:  # noqa: BLE001
            pass
    st.caption(
        f"Design matrix: **{total}** columns across {len(predictors)} predictors on {train.height:,} training rows."
    )
    _detail(train, preview, predictors)
