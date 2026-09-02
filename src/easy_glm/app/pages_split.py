"""Page 4 — Split: train / holdout definition and balance."""

from __future__ import annotations

import polars as pl
import streamlit as st

from . import charts as C
from . import state as S
from . import ui


def render() -> None:
    st.title("Split")
    ui.status_bar()
    p = S.project()
    raw = S.raw_frame() if p.data.source.path else None
    if raw is None:
        st.info("Load a data file on the **Project & data** page first.")
        return
    sp = p.data.split
    mode = st.radio(
        "How is the holdout defined?",
        ["column", "random"],
        index=0 if sp.mode == "column" else 1,
        horizontal=True,
        format_func=lambda m: {
            "column": "Existing indicator column",
            "random": "Random split (seeded)",
        }[m],
        key="split_mode",
    )
    changed = mode != sp.mode
    sp.mode = mode
    if mode == "column":
        cols = list(raw.columns) + [
            c for c in p.data.renames.values() if c not in raw.columns
        ]
        c1, c2 = st.columns([2, 1])
        default = sp.column if sp.column in cols else cols[0]
        col = c1.selectbox(
            "Indicator column", cols, index=cols.index(default), key="split_col"
        )
        tv = c2.text_input(
            "Value meaning TRAIN", str(sp.train_value), key="split_train_value"
        )
        try:
            tv_parsed: object = int(tv)
        except ValueError:
            try:
                tv_parsed = float(tv)
            except ValueError:
                tv_parsed = tv
        if col != sp.column or tv_parsed != sp.train_value:
            sp.column, sp.train_value = col, tv_parsed
            p.set_role(col, "split")
            changed = True
    else:
        c1, c2, c3 = st.columns(3)
        frac = c1.slider(
            "Training fraction", 0.5, 0.95, float(sp.fraction), 0.05, key="split_frac"
        )
        seed = c2.number_input("Seed", 0, 10_000, int(sp.seed), key="split_seed")
        name = c3.text_input(
            "Split column name",
            sp.column if sp.column else "traintest",
            key="split_name",
        )
        if (frac, int(seed), name) != (sp.fraction, sp.seed, sp.column):
            sp.fraction, sp.seed, sp.column = float(frac), int(seed), name
            changed = True
    if changed:
        S.touch()
        st.rerun()

    df = S.prepared_frame()
    if df is None or sp.column not in df.columns:
        return
    w = pl.col(p.weight).sum() if p.weight else pl.len().cast(pl.Float64)
    cfg = p.models.get(p.champion) if p.champion else None
    divide = cfg.divide_target_by_weight if cfg else bool(p.weight)
    if p.target:
        rate = (
            (pl.col(p.target).sum() / pl.col(p.weight).sum())
            if (p.weight and divide)
            else (
                ((pl.col(p.target) * pl.col(p.weight)).sum() / pl.col(p.weight).sum())
                if p.weight
                else pl.col(p.target).mean()
            )
        )
    else:
        rate = pl.lit(None, dtype=pl.Float64)
    bal = (
        df.group_by(
            pl.when(pl.col(sp.column) == 1)
            .then(pl.lit("train"))
            .otherwise(pl.lit("holdout"))
            .alias("subset")
        )
        .agg(pl.len().alias("rows"), w.alias("exposure"), rate.alias("rate"))
        .sort("subset", descending=True)
    )
    ui.polars_table(bal)
    st.plotly_chart(C.split_balance_chart(bal), width="stretch")
    st.caption(
        "Design knots, level lumping and the fit all use **training rows only**; the holdout is used for diagnostics."
    )
