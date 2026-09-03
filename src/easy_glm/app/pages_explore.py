"""Page 3 — Explore: univariate views and the leakage report."""

from __future__ import annotations

import polars as pl
import streamlit as st

from easy_glm.workflow import univariate

from . import charts as C
from . import state as S
from . import ui


def _univariate(df: pl.DataFrame) -> None:
    p = S.project()
    if S.is_sampled():
        st.caption(
            f"Charts use the exploration sample of {df.height:,} rows "
            "(Project page); fits and diagnostics use the full data."
        )
    cfg = p.models.get(p.champion) if p.champion else None
    divide = cfg.divide_target_by_weight if cfg else bool(p.weight)
    reserved = {p.target, p.weight, p.offset, p.data.split.column} - {None}
    candidates = [c for c in df.columns if c not in reserved]
    c1, c2, c3 = st.columns([3, 1, 1])
    var = c1.selectbox("Variable", candidates, key=S.widget_key("explore_var"))
    n_bins = c2.slider("Bands", 5, 50, 20, key=S.widget_key("explore_bins"))
    subset = c3.radio(
        "Rows", ["train", "all"], horizontal=True, key=S.widget_key("explore_subset")
    )
    frame = (
        df.filter(pl.col(p.data.split.column) == 1)
        if subset == "train" and p.data.split.column in df.columns
        else df
    )
    u = univariate(
        frame,
        var,
        target=p.target,
        weight=p.weight,
        divide_target_by_weight=divide,
        n_bins=n_bins,
    )
    ui.metric_row(
        [
            ("Kind", u["kind"], None),
            ("Distinct", f"{u['n_unique']:,}", None),
            ("Null share", ui.fmt(u["null_share"], pct=True), None),
            ("Rows", f"{u['n']:,}", None),
        ]
    )
    st.plotly_chart(
        C.exposure_rate_chart(
            u["table"], title=f"{var}: exposure and observed {p.target or ''} rate"
        ),
        width="stretch",
    )
    with st.expander("Table"):
        ui.polars_table(u["table"])


def _leakage(df: pl.DataFrame) -> None:
    p = S.project()
    st.caption(
        "Every candidate predictor is scored on the training rows: single-factor deviance explained, "
        "rank correlation with the target, identifier-likeness, post-outcome naming and degeneracy. "
        "Anything that explains far too much is probably known only after the claim."
    )
    c1, c2 = st.columns([1, 4])
    if c1.button("Run / refresh scan", type="primary"):
        S.leakage(force=True)
    rep = S.leakage()
    if rep is None:
        return
    if rep.is_empty():
        st.info("No candidate predictors.")
        return

    def _colour(rec: str) -> str:
        return {
            "ignore": "🔴",
            "check": "🟠",
            "ok": "🟢",
            "ignored": "⚫",
            "acknowledged": "🔵",
        }.get(rec, "")

    show = rep.with_columns(
        (
            pl.col("recommendation").map_elements(_colour, return_dtype=pl.Utf8)
            + " "
            + pl.col("recommendation")
        ).alias("status")
    ).select(
        "status",
        "variable",
        "role",
        "score",
        "flags",
        "deviance_explained",
        "rank_corr",
        "unique_ratio",
        "null_share",
        "n_unique",
    )
    st.dataframe(
        show,
        width="stretch",
        hide_index=True,
        column_config={
            "score": st.column_config.ProgressColumn(
                "score", min_value=0, max_value=100, format="%.0f"
            ),
            "deviance_explained": st.column_config.NumberColumn(
                "dev. explained", format="percent"
            ),
            "rank_corr": st.column_config.NumberColumn("rank corr", format="%.2f"),
            "unique_ratio": st.column_config.NumberColumn(
                "unique ratio", format="%.2f"
            ),
            "null_share": st.column_config.NumberColumn("null share", format="percent"),
        },
        height=min(38 * (show.height + 1) + 4, 520),
    )
    flagged = rep.filter(pl.col("recommendation").is_in(["ignore", "check"]))[
        "variable"
    ].to_list()
    c1, c2, c3 = st.columns([3, 1, 1])
    chosen = c1.multiselect(
        "Variables",
        rep["variable"].to_list(),
        default=flagged,
        key=S.widget_key("leak_pick"),
    )
    leak = p.exploration.setdefault("leakage", {"ignored": [], "acknowledged": []})
    if (
        c2.button("Ignore selected", help="Set role = ignore and remember why")
        and chosen
    ):
        for v in chosen:
            p.set_role(v, "ignore")
            if v not in leak["ignored"]:
                leak["ignored"].append(v)
            if v in leak["acknowledged"]:
                leak["acknowledged"].remove(v)
            for m in p.models.values():
                if v in m.predictors:
                    m.predictors.remove(v)
        S.touch()
        st.rerun()
    if (
        c3.button("Acknowledge selected", help="Keep as predictor; stop flagging")
        and chosen
    ):
        for v in chosen:
            if v not in leak["acknowledged"]:
                leak["acknowledged"].append(v)
        S.touch()
        st.rerun()
    if leak["ignored"] or leak["acknowledged"]:
        st.caption(
            f"Ignored after review: {', '.join(leak['ignored']) or '—'} · "
            f"acknowledged: {', '.join(leak['acknowledged']) or '—'}"
        )


def render() -> None:
    st.title("Explore")
    ui.status_bar()
    df = ui.require_data()
    if df is None:
        return
    tab1, tab2 = st.tabs(["Univariate", "Leakage report"])
    with tab1:
        sample = S.sample_frame()
        if sample is not None:
            _univariate(sample)
    with tab2:
        if ui.require_target() is not None:
            _leakage(df)  # full training rows; the report samples internally
