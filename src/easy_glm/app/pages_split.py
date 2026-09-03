"""Page 4 — Split: train / holdout definition and balance."""

from __future__ import annotations

import polars as pl
import streamlit as st

from . import charts as C
from . import state as S
from . import ui


def final_columns(raw: pl.DataFrame) -> list[str]:
    """Column names after renames and derived columns (what the split sees)."""
    p = S.project()
    names = [p.data.renames.get(c, c) for c in raw.columns]
    names += [d.name for d in p.data.derived if d.name not in names]
    return names


def _column_mode(raw: pl.DataFrame) -> bool:
    p = S.project()
    sp = p.data.split
    cols = final_columns(raw)
    c1, c2 = st.columns([2, 1])
    known = sp.column in cols
    col = c1.selectbox(
        "Indicator column",
        cols,
        index=cols.index(sp.column) if known else None,
        placeholder="choose the train/holdout indicator column",
        key=S.widget_key("split_col"),
    )
    if not known:
        st.error(
            f"The split column {sp.column!r} is not in the data. Pick the indicator "
            "column above or use a random split."
        )
    tv = c2.text_input(
        "Value meaning TRAIN",
        str(sp.train_value),
        key=S.widget_key("split_train_value"),
    )
    try:
        tv_parsed: object = int(tv)
    except ValueError:
        try:
            tv_parsed = float(tv)
        except ValueError:
            tv_parsed = tv
    changed = False
    if col is not None and (col != sp.column or tv_parsed != sp.train_value):
        previous = sp.column
        sp.column, sp.train_value = col, tv_parsed
        # the indicator gets the split role; the previous indicator simply loses
        # it (never demoted to "ignore" behind the user's back)
        for c, r in list(p.data.roles.items()):
            if r == "split" and c != col:
                p.data.roles.pop(c)
        p.data.roles[col] = "split"
        if previous != col:
            ui.flash("info", f"Split indicator is now {col!r} (was {previous!r})")
        changed = True
    return changed


def _random_mode(raw: pl.DataFrame) -> bool:
    p = S.project()
    sp = p.data.split
    c1, c2, c3 = st.columns(3)
    frac_value = min(max(float(sp.fraction), 0.5), 0.95)
    if frac_value != sp.fraction:
        st.warning(
            f"Training fraction {sp.fraction:g} in the project file is outside the "
            f"0.50–0.95 range; showing {frac_value:.2f}."
        )
    frac = c1.slider(
        "Training fraction", 0.5, 0.95, frac_value, 0.05, key=S.widget_key("split_frac")
    )
    seed = c2.number_input(
        "Seed",
        0,
        10_000,
        int(sp.seed),
        key=S.widget_key("split_seed"),
        help="0 – 10000",
    )
    name = c3.text_input(
        "Split column name",
        sp.column if sp.column else "traintest",
        key=S.widget_key("split_name"),
    ).strip()
    taken = set(final_columns(raw))
    if not name:
        st.error("The split column needs a name")
        name = sp.column
    elif name in taken and name != sp.column:
        st.error(
            f"{name!r} is already a column in the data; the random split would "
            "overwrite it. Choose another name."
        )
        name = sp.column
    elif sp.column in taken and name == sp.column:
        st.error(
            f"The split column name {name!r} is also a data column; the random "
            "split would overwrite it. Choose another name."
        )
    changed = False
    if (float(frac), int(seed), name) != (sp.fraction, sp.seed, sp.column):
        sp.fraction, sp.seed, sp.column = float(frac), int(seed), name
        changed = True
    return changed


def _balance(df: pl.DataFrame) -> None:
    p = S.project()
    sp = p.data.split
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


def render() -> None:
    st.title("Split")
    ui.status_bar()
    p = S.project()
    raw = ui.require_raw()
    if raw is None:
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
        key=S.widget_key("split_mode"),
    )
    changed = mode != sp.mode
    if changed:
        sp.mode = mode
        if mode == "random" and sp.column in final_columns(raw):
            # the old indicator column keeps its data; the random flag needs its own name
            existing = set(final_columns(raw))
            sp.column = "traintest" if "traintest" not in existing else "split_flag"
            ui.flash(
                "info", f"Random split will be written to a new column {sp.column!r}"
            )
    changed |= _column_mode(raw) if mode == "column" else _random_mode(raw)
    if changed:
        S.touch()
        st.rerun()

    df = S.prepared_frame()
    if df is None:
        ui.show_data_problem()
        return
    if sp.column not in df.columns:
        return
    ui.guarded(lambda: _balance(df), "Train / holdout balance")
    st.caption(
        "Design knots, level lumping and the fit all use **training rows only**; "
        "the holdout is used for diagnostics."
    )
