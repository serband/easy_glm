"""Page 5 — Design: how each predictor becomes GLM features, and the
two-way interactions of the selected model."""

from __future__ import annotations

import math

import pandas as pd
import polars as pl
import streamlit as st

from easy_glm.core.design import (
    NUMERIC_DTYPES,
    CategoricalEncoder,
    InteractionEncoder,
    LinearEncoder,
    StepEncoder,
    quantile_knots,
    round_range_outward,
    row_label,
)
from easy_glm.workflow import Interaction, VariableDesign, encoder_for, univariate

from . import charts as C
from . import state as S
from . import ui

KNOT_OPTIONS = ["quantile", "integer", "custom"]
KIND_OPTIONS = ["auto", "step", "linear", "categorical"]
MONO_OPTIONS = ["none", "increasing", "decreasing"]


def _parse_numbers(text: str) -> list[float]:
    """Comma / newline separated numbers; raises ValueError naming the bad token."""
    out: list[float] = []
    for tok in text.replace("\n", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            value = float(tok)
        except ValueError as exc:
            raise ValueError(f"{tok!r} is not a number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{tok!r} is not a finite number")
        out.append(value)
    return sorted(set(out))


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
        key=S.widget_key("design_grid"),
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
        numeric = v in train.columns and train[v].dtype in NUMERIC_DTYPES
        if new.monotone and (not numeric or new.kind == "categorical"):
            ui.flash(
                "error",
                f"{v}: monotone constraints apply to numeric step designs only; "
                "the constraint was not saved",
            )
            new.monotone = vd.monotone if numeric and vd.kind != "categorical" else None
        if new != vd:
            if new == VariableDesign():
                p.design.variables.pop(v, None)
            else:
                p.design.variables[v] = new
            changed = True
    if changed:
        S.touch()
        st.rerun()


# --------------------------------------------------------------------------
# variable detail
# --------------------------------------------------------------------------
def _kind_selector(var: str, vd: VariableDesign, numeric: bool) -> None:
    """The kind of the selected variable (mirrors the grid column)."""
    p = S.project()
    current = vd.kind or "auto"
    kind = st.selectbox(
        "Kind",
        KIND_OPTIONS,
        index=KIND_OPTIONS.index(current),
        key=S.widget_key(f"kind_{var}"),
        help="auto = step for numbers, categorical for text",
    )
    if kind != current:
        if kind in ("step", "linear") and not numeric:
            st.error(f"{var} is not numeric; a {kind} design needs numbers")
            return
        vd.kind = None if kind == "auto" else kind
        if kind == "linear" and vd.monotone:
            vd.monotone = None
            ui.flash(
                "warning",
                f"Monotone constraint on {var} removed: not available for "
                "piecewise-linear terms",
            )
        p.design.variables[var] = vd
        S.touch()
        st.rerun()


def _knots_outside_the_data(var: str, knots: list[float], series: pl.Series) -> None:
    """Flash a warning for knots the training rows never reach. They are
    accepted (the actuary may be keeping room for next year's data) but they
    make an empty bin, so the page has to say which ones."""
    s = series.drop_nulls().cast(pl.Float64)
    if s.is_empty():
        return
    lo, hi = float(s.min()), float(s.max())
    for bad, where, edge in (
        ([k for k in knots if k > hi], "above the largest", hi),
        ([k for k in knots if k <= lo], "at or below the smallest", lo),
    ):
        if not bad:
            continue
        many = len(bad) > 1
        ui.flash(
            "warning",
            f"{var}: {'knots' if many else 'knot'} "
            + ", ".join(f"{k:g}" for k in bad)
            + f" {'are' if many else 'is'} {where} training value ({edge:g}); "
            + ("the bins they open have" if many else "the bin it opens has")
            + " no training rows, so the relativity there comes only from the "
            "penalty. Saved anyway.",
        )


def _step_detail(
    var: str,
    vd: VariableDesign,
    enc: StepEncoder,
    train: pl.DataFrame,
    preview: pl.DataFrame,
    divide,
) -> None:
    p = S.project()
    knots_txt = st.text_area(
        "Knots (comma-separated; editing switches to custom)",
        ", ".join(f"{k:g}" for k in enc.knots),
        height=90,
        key=S.widget_key(f"knots_{var}"),
    )
    if st.button("Apply knots", key=S.widget_key(f"apply_knots_{var}")):
        try:
            knots = _parse_numbers(knots_txt)
        except ValueError as exc:
            st.error(f"Knots: {exc}")
        else:
            if not knots:
                st.error("At least one knot is needed")
            else:
                _knots_outside_the_data(var, knots, train[var])
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
        f"Design columns: {enc.n_features} · bins: {len(enc.bins())} · "
        f"null indicator: {enc.null_indicator}"
    )


def _linear_detail(
    var: str,
    vd: VariableDesign,
    enc: LinearEncoder,
    train: pl.DataFrame,
    preview: pl.DataFrame,
    divide,
) -> None:
    p = S.project()
    d = p.design.defaults
    s = train[var].drop_nulls().cast(pl.Float64)
    tmin, tmax = float(s.min()), float(s.max())
    rlo, rhi = round_range_outward(tmin, tmax)
    st.markdown(
        f"**Piecewise-linear** — the relativity curve is continuous, log-linear "
        f"inside each band, and **flat outside the clamp range**. Training range "
        f"{tmin:g} – {tmax:g}; default clamp = that range rounded outward to a round "
        f"number → **{rlo:g} – {rhi:g}**."
    )
    strategy_now = "custom" if isinstance(vd.knots, list) else vd.knots
    c1, c2, c3 = st.columns([1, 1, 2])
    strategy = c1.radio(
        "Knot strategy",
        KNOT_OPTIONS,
        index=KNOT_OPTIONS.index(strategy_now),
        key=S.widget_key(f"lin_strategy_{var}"),
        help="Where the slope may change. quantile: n_bins quantiles · integer: every integer · custom: your list",
    )
    n_bins = c2.number_input(
        "n_bins (quantile)",
        2,
        200,
        int(vd.n_bins or d.n_bins),
        key=S.widget_key(f"lin_nbins_{var}"),
        disabled=strategy != "quantile",
    )
    knots_txt = c3.text_area(
        "Knots (custom)",
        ", ".join(f"{k:g}" for k in enc.knots),
        height=70,
        key=S.widget_key(f"lin_knots_{var}"),
        disabled=strategy != "custom",
    )
    use_default = st.checkbox(
        "Clamp to the training range (rounded outward)",
        vd.clamp is None,
        key=S.widget_key(f"lin_defaultclamp_{var}"),
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    lo = c1.number_input(
        "Clamp lo",
        value=float(vd.clamp[0]) if vd.clamp else rlo,
        key=S.widget_key(f"lin_lo_{var}"),
        disabled=use_default,
        format="%g",
    )
    hi = c2.number_input(
        "Clamp hi",
        value=float(vd.clamp[1]) if vd.clamp else rhi,
        key=S.widget_key(f"lin_hi_{var}"),
        disabled=use_default,
        format="%g",
    )
    c3.caption(
        "Values below lo / above hi get the relativity at the clamp. Knots must lie "
        "strictly inside the clamp range."
    )
    if st.button(
        "Apply linear design", key=S.widget_key(f"apply_lin_{var}"), type="primary"
    ):
        errors: list[str] = []
        clamp: list[float] | None
        if use_default:
            clamp = None
            lo_c, hi_c = rlo, rhi
        else:
            lo_c, hi_c = float(lo), float(hi)
            clamp = [lo_c, hi_c]
            if hi_c <= tmin or lo_c >= tmax:
                errors.append(
                    f"Clamp range {lo_c:g} – {hi_c:g} does not overlap the training "
                    f"range {tmin:g} – {tmax:g}; the term would be flat everywhere"
                )
            if not lo_c < hi_c:
                errors.append("Clamp lo must be below clamp hi")
        knots: list[float] = list(enc.knots)
        if strategy == "custom":
            try:
                knots = _parse_numbers(knots_txt)
            except ValueError as exc:
                errors.append(f"Knots: {exc}")
                knots = []
            outside = [k for k in knots if not lo_c < k < hi_c]
            if outside:
                errors.append(
                    "Knots outside the clamp range: "
                    + ", ".join(f"{k:g}" for k in outside)
                    + f" (clamp {lo_c:g} – {hi_c:g}); move them inside or widen the clamp"
                )
        if errors:
            for e in errors:
                st.error(e)
        else:
            vd.kind = "linear"
            vd.knots = knots if strategy == "custom" else strategy
            vd.n_bins = int(n_bins) if strategy == "quantile" else vd.n_bins
            vd.clamp = clamp
            vd.monotone = None
            p.design.variables[var] = vd
            S.touch()
            st.rerun()
    edges = enc.band_edges()
    u = univariate(
        preview,
        var,
        target=p.target,
        weight=p.weight,
        divide_target_by_weight=divide,
        knots=edges,
    )
    labels = u["table"]["label"].to_list()
    marks: dict[str, str] = {}
    if labels:
        marks[labels[0]] = f"clamp lo {enc.lo:g}"
        last = [lab for lab in labels if lab.startswith("≥")]
        if last:
            marks[last[-1]] = f"clamp hi {enc.hi:g}"
    st.plotly_chart(
        C.exposure_rate_chart(
            u["table"],
            title=f"{var}: linear in {len(enc.knots) + 1} band(s) between {enc.lo:g} and {enc.hi:g}",
            marks=marks,
        ),
        width="stretch",
    )
    st.caption(
        f"Design columns: {enc.n_features} · rows in the rate table: {enc.n_rows} · "
        f"null indicator: {enc.null_indicator} · knots: "
        + (", ".join(f"{k:g}" for k in enc.knots) or "none (one straight band)")
    )


def _categorical_detail(
    var: str, vd: VariableDesign, enc: CategoricalEncoder, preview, divide
) -> None:
    p = S.project()
    st.markdown(
        f"**Reference level:** `{enc.reference}`  \n**Kept levels:** {len(enc.levels)} (+ Other)"
    )
    levels_txt = st.text_area(
        "Levels (first = reference; others lumped into Other)",
        ", ".join(enc.levels),
        height=90,
        key=S.widget_key(f"levels_{var}"),
    )
    if st.button("Apply levels", key=S.widget_key(f"apply_levels_{var}")):
        levels = [
            x.strip() for x in levels_txt.replace("\n", ",").split(",") if x.strip()
        ]
        if len(set(levels)) != len(levels):
            st.error("Levels must be unique")
        else:
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


def _detail(train: pl.DataFrame, preview: pl.DataFrame, predictors: list[str]) -> None:
    p = S.project()
    st.subheader("Variable detail")
    c1, c2 = st.columns([2, 1])
    var = c1.selectbox("Variable", predictors, key=S.widget_key("design_detail_var"))
    vd = p.design.variables.get(var, VariableDesign())
    numeric = train[var].dtype in NUMERIC_DTYPES
    with c2:
        _kind_selector(var, vd, numeric)
    vd = p.design.variables.get(var, VariableDesign())
    weights = train[p.weight] if p.weight and p.weight in train.columns else None
    cfg = p.models.get(p.champion) if p.champion else None
    divide = cfg.divide_target_by_weight if cfg else bool(p.weight)
    try:
        enc = encoder_for(var, train[var], vd, p, weights=weights)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Cannot build the design for {var} yet: {exc}")
        if numeric and (vd.kind or "step") == "step":
            suggestion = quantile_knots(
                train[var], vd.n_bins or p.design.defaults.n_bins
            )
            knots_txt = st.text_area(
                "Knots (comma-separated)",
                ", ".join(f"{k:g}" for k in suggestion),
                height=90,
                key=S.widget_key(f"knots_{var}"),
            )
            if st.button("Apply knots", key=S.widget_key(f"apply_knots_{var}")):
                try:
                    knots = _parse_numbers(knots_txt)
                except ValueError as err:
                    st.error(f"Knots: {err}")
                else:
                    vd.knots = knots
                    p.design.variables[var] = vd
                    S.touch()
                    st.rerun()
        return
    if isinstance(enc, StepEncoder):
        _step_detail(var, vd, enc, train, preview, divide)
    elif isinstance(enc, LinearEncoder):
        _linear_detail(var, vd, enc, train, preview, divide)
    elif isinstance(enc, CategoricalEncoder):
        _categorical_detail(var, vd, enc, preview, divide)


# --------------------------------------------------------------------------
# interactions
# --------------------------------------------------------------------------
def _interaction_model_name() -> str | None:
    p = S.project()
    names = list(p.models)
    if not names:
        return None
    current = st.session_state.get("model_current")
    if current not in names:
        current = p.champion if p.champion in names else names[0]
    return st.selectbox(
        "Model",
        names,
        index=names.index(current),
        key=S.widget_key("design_inter_model"),
        help="Interactions belong to a model (its predictors must include both parents)",
    )


def _interactions(train: pl.DataFrame) -> None:
    p = S.project()
    st.subheader("Interactions")
    st.caption(
        "A two-way interaction A × B adds one adjustment per cell on top of the two "
        "main effects; cells with too little exposure get no adjustment (1.00). "
        "Add them here, fit on the Model page, see the cells on the Rate tables page."
    )
    name = _interaction_model_name()
    if name is None:
        st.info("Create a model on the Model page first.")
        return
    cfg = p.models[name]
    if cfg.interactions:
        for i, it in enumerate(list(cfg.interactions)):
            c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
            c1.markdown(f"**{it.name}**")
            c2.caption(f"min cell exposure {it.min_cell_exposure:.2%}")
            c3.caption(f"penalty weight {it.penalty_weight:g}")
            if c4.button("Remove", key=S.widget_key(f"rm_inter_{name}_{i}")):
                cfg.interactions.pop(i)
                cfg.adjustments = [a for a in cfg.adjustments if a.variable != it.name]
                S.touch()
                st.rerun()
    else:
        st.caption("No interactions in this model yet.")
    preds = list(cfg.predictors)
    if len(preds) < 2:
        st.info("The model needs at least two predictors before adding an interaction.")
        return
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
        a = c1.selectbox("First variable", preds, key=S.widget_key(f"inter_a_{name}"))
        b = c2.selectbox(
            "Second variable",
            preds,
            index=min(1, len(preds) - 1),
            key=S.widget_key(f"inter_b_{name}"),
        )
        share = c3.number_input(
            "Min cell exposure (%)",
            0.0,
            50.0,
            0.5,
            0.1,
            key=S.widget_key(f"inter_share_{name}"),
            help="Cells below this share of the pair's training exposure get no adjustment",
        )
        weight = c4.number_input(
            "Penalty weight", 0.1, 100.0, 1.0, 0.1, key=S.widget_key(f"inter_w_{name}")
        )
        errors: list[str] = []
        if a == b:
            errors.append("Pick two different variables.")
        if any({it.a, it.b} == {a, b} for it in cfg.interactions):
            errors.append(f"{a} × {b} is already in the model.")
        for v in (a, b):
            if v not in p.predictors:
                errors.append(f"{v} is not a predictor of the project.")
            elif v not in train.columns:
                errors.append(f"{v} is not in the prepared data.")
        preview_enc: InteractionEncoder | None = None
        if not errors:
            try:
                weights = (
                    train[cfg.weight]
                    if cfg.weight and cfg.weight in train.columns
                    else None
                )
                ea = encoder_for(
                    a,
                    train[a],
                    p.design.variables.get(a, VariableDesign()),
                    p,
                    weights=weights,
                )
                eb = encoder_for(
                    b,
                    train[b],
                    p.design.variables.get(b, VariableDesign()),
                    p,
                    weights=weights,
                )
                preview_enc = InteractionEncoder.from_data(
                    ea,
                    eb,
                    train,
                    weights=weights,
                    min_cell_exposure=float(share) / 100.0,
                    penalty_weight=float(weight),
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Cannot build {a} × {b}: {exc}")
        for e in errors:
            st.error(e)
        if preview_enc is not None:
            n_cells = preview_enc.a.n_rows * preview_enc.b.n_rows
            st.caption(
                f"Preview on all {train.height:,} training rows (never the exploration "
                f"sample — the fit decides cells on the same rows): "
                f"**{len(preview_enc.cells)} of {n_cells} cells** would get their own "
                f"adjustment at a {share:.1f}% threshold ({preview_enc.n_features} "
                "design columns)."
            )
            rows_a = [row_label(r) for r in preview_enc.a.rows()]
            rows_b = [row_label(r) for r in preview_enc.b.rows()]
            kept = [[0.0] * len(rows_b) for _ in rows_a]
            for i, j in preview_enc.cells:
                kept[i][j] = 1.0
            st.plotly_chart(
                C.matrix_heatmap(
                    rows_a,
                    rows_b,
                    preview_enc.exposure,
                    title=f"Training exposure by cell — {a} (rows) × {b} (columns)",
                    row_name=a,
                    col_name=b,
                    hover={"kept (1 = own adjustment)": kept},
                    centred=False,
                    height=380,
                ),
                width="stretch",
            )
        if st.button(
            "Add interaction",
            type="primary",
            key=S.widget_key(f"inter_add_{name}"),
            disabled=bool(errors),
        ):
            cfg.interactions.append(
                Interaction(
                    a,
                    b,
                    min_cell_exposure=float(share) / 100.0,
                    penalty_weight=float(weight),
                )
            )
            S.touch()
            ui.flash("success", f"Added {a} × {b} to {name}; fit on the Model page.")
            st.rerun()


# --------------------------------------------------------------------------
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
    if train.is_empty():
        st.error("There are no training rows; check the split on the Split page.")
        return
    missing = [v for v in predictors if v not in train.columns]
    if missing:
        st.error(
            "These predictors are not in the prepared data (renamed or removed?): "
            + ", ".join(missing)
            + ". Fix their roles on the Variables page."
        )
        predictors = [v for v in predictors if v in train.columns]
        if not predictors:
            return
    for m in [m for m in p.validate() if m.startswith("design[")]:
        st.error(m)
    if S.is_sampled():
        st.caption(
            f"Knots and levels are derived from all {train.height:,} training rows; "
            f"the preview charts use the exploration sample ({preview.height:,} rows)."
        )
    _defaults()
    st.caption(
        "Numeric predictors become step functions (one 0/1 column per knot, penalised increments → automatic banding) "
        "or piecewise-linear curves; categoricals become one-hot with the most frequent level as reference and an "
        "**Other** bucket. Monotone constraints bound the step increments (not available for linear terms)."
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
        f"Design matrix: **{total}** main-effect columns across {len(predictors)} predictors "
        f"on {train.height:,} training rows (interaction cells come on top)."
    )
    _detail(train, preview, predictors)
    _interactions(train)
