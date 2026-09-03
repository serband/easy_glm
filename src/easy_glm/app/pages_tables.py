"""Page 9 — Rate tables: inspect, adjust relativities (rows or cells), smooth /
cap / round them, undo, snapshot and compare snapshots, export."""

from __future__ import annotations

import copy
from datetime import datetime, timezone

import pandas as pd
import polars as pl
import streamlit as st

from easy_glm.core.excel import rate_model_tables, variable_frame
from easy_glm.engine import tooling as T
from easy_glm.engine.models import level_label
from easy_glm.workflow import (
    TableSnapshot,
    ae_by_pair,
    ae_by_variable,
    describe_diff,
    expected_claims,
    missing_variables,
    rate_model_diff,
    rate_model_for,
    totals,
)

from . import charts as C
from . import grids as G
from . import state as S
from . import ui

#: the tools offered above the editor, in the order they are listed
TOOLS = [
    "Smooth (moving average)",
    "Smooth (isotonic)",
    "Cap / floor",
    "Round",
]
#: what the two snapshot-diff selectors offer besides the named snapshots
FITTED_OPTION = "(fitted — no adjustments)"
CURRENT_OPTION = "(the tables now)"


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


def _challenger_selector(column, model: str):
    """The challenger whose expected line is overlaid on the A/E charts: the
    sidebar's "compare with" by default, overridable here (the widget's key
    carries the sidebar value, so moving that selector re-defaults this one)."""
    fitted = [n for n in S.fitted_models() if n != model]
    if not fitted:
        return None
    sidebar = S.challenger()
    options = ["(none)"] + fitted
    default = sidebar if sidebar in fitted else "(none)"
    with column:
        name = st.selectbox(
            "Compare with (challenger)",
            options,
            index=options.index(default),
            key=S.widget_key(f"tables_chal_{model}_{sidebar}"),
            help="Overlays the challenger's expected line on the A/E charts "
            "below. The full side-by-side view is the **Compare** page.",
        )
    return S.get_run(name) if name != "(none)" else None


def _apply(
    run_name: str,
    changed: bool,
    errors: list[str],
    before: S.EditStep | None = None,
) -> None:
    """Save an edit: record one undo step (``before`` = the adjustments *and*
    the base-rate override as they were), autosave, re-apply to the cached run
    and redraw."""
    for e in errors:
        if changed:
            ui.flash("error", e)  # the rerun below would discard it otherwise
        else:
            st.error(e + " (retype the cell to clear this message)")
    if changed:
        if before is not None:
            S.record_undo(run_name, before)
        S.touch()
        S.refresh_adjustments(run_name)
        st.rerun()


# --------------------------------------------------------------------------
# main-effect tables (step / categorical / linear)
# --------------------------------------------------------------------------
def _main_effect(run, var: str, df: pl.DataFrame, challenger=None) -> pl.DataFrame:
    p = S.project()
    cfg = p.models[run.name]
    rm = run.rate_model
    fitted = run.tables[var]
    working = rate_model_tables(rm)[var]
    rows = rm.variables[var].table
    other = rm.variables[var].other_label
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
                    log_x=enc.lo > 0 and enc.hi / enc.lo > 100,
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
            "A/E rows",
            ["holdout", "train"],
            horizontal=True,
            key=S.widget_key("tables_ae_rows"),
        )
        frame = _ae_frame(df, which)
        if not frame.is_empty():
            actual, expected, w = totals(frame, cfg, run.predict(frame))
            enc = run.spec[var]
            knots = enc.band_edges() if hasattr(enc, "band_edges") else None
            tbl = ae_by_variable(frame, var, actual, expected, w, knots=knots)
            cmp_tbl = None
            can_score = challenger is not None and not [
                c for c in challenger.spec.required_columns if c not in frame.columns
            ]
            if can_score and var in challenger.spec.encoders:
                exp_chal = totals(frame, challenger.config, challenger.predict(frame))[
                    1
                ]
                cmp_tbl = ae_by_variable(frame, var, actual, exp_chal, w, knots=knots)
            st.plotly_chart(
                C.ae_chart(
                    tbl,
                    title=f"{var} — actual vs expected with current relativities ({which})",
                    compare=cmp_tbl,
                    compare_name=challenger.name if challenger else "challenger",
                ),
                width="stretch",
            )
            if challenger is not None and cmp_tbl is None:
                st.caption(
                    f"{challenger.name} has no **{var}** term, so there is no "
                    "second expected line to draw."
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
                    "band": [level_label(r, other) for r in rows],
                    "exposure": [r.exposure for r in rows],
                    "fitted": fitted["relativity"].to_list(),
                    "working": [r.relativity for r in rows],
                    "at band end": working["relativity_to"].to_list(),
                    "slope": working["slope"].to_list(),
                }
            )
            disabled = ["band", "exposure", "fitted", "at band end", "slope"]
            col_cfg = {
                "exposure": st.column_config.NumberColumn(format="%.0f"),
                "fitted": st.column_config.NumberColumn(format="%.4f"),
                "working": st.column_config.NumberColumn(
                    "working (at band start)",
                    format="%.4f",
                    min_value=1e-4,
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
                    "bin": [level_label(r, other) for r in rows],
                    "exposure": [r.exposure for r in rows],
                    "fitted": fitted["relativity"].to_list(),
                    "working": [r.relativity for r in rows],
                }
            )
            disabled = ["bin", "exposure", "fitted"]
            col_cfg = {
                "exposure": st.column_config.NumberColumn(
                    format="%.0f",
                    help="Training exposure in this row — what the tools weight "
                    "a band by, and what tells 'no data' from 'no effect'.",
                ),
                "fitted": st.column_config.NumberColumn(format="%.4f"),
                "working": st.column_config.NumberColumn(format="%.4f", min_value=1e-4),
            }
        edited = st.data_editor(
            grid,
            hide_index=True,
            width="stretch",
            height=min(38 * (len(rows) + 1) + 4, 560),
            disabled=disabled,
            column_config=col_cfg,
            key=S.widget_key(f"rel_editor_{run.name}_{var}"),
        )
        before = S.edit_state(run.name)
        changed, errors = G.apply_row_edits(
            cfg,
            var,
            rows,
            fitted["relativity"].to_list(),
            edited["working"].tolist(),
            require_positive=is_linear,
            other_label=other,
        )
        _apply(run.name, changed, errors, before)
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
    n_all = sum(1 for row in grid["keys"] for k in row if k is not None)
    n_nodata = sum(1 for row in grid["current"] for v in row if v is None)
    n_thin = int(grid["n_below_threshold"])
    st.caption(
        f"Cells multiply the two main effects **{a}** and **{b}**; 1.00 means no "
        "adjustment. Blank cells had **no training exposure** "
        f"({n_nodata} of {n_all}) and cannot be edited; a further {n_thin} cells were "
        "below the exposure threshold and are 1.00 by construction (hover shows the "
        "training exposure and the fitted value)."
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
            "A/E rows",
            ["holdout", "train"],
            horizontal=True,
            key=S.widget_key("tables_ae_rows"),
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
                c: st.column_config.NumberColumn(format="%.4f", min_value=1e-4)
                for c in grid["cols"]
            },
            key=S.widget_key(f"cell_editor_{run.name}_{var}"),
        )
        before = S.edit_state(run.name)
        changed, errors = G.apply_cell_edits(cfg, var, grid, edited.values.tolist())
        _apply(run.name, changed, errors, before)
        with st.expander("Fitted cells (before adjustments)"):
            st.dataframe(
                pd.DataFrame(grid["fitted"], index=grid["rows"], columns=grid["cols"]),
                width="stretch",
            )
    return rate_model_tables(rm)[var]


# --------------------------------------------------------------------------
# tools: smooth / cap / floor / round
# --------------------------------------------------------------------------
def _tool_result(var_cfg, var: str, tool: str, kwargs: dict) -> T.ToolResult:
    """Run one tool on the variable's current table (nothing is changed)."""
    if tool == TOOLS[0]:
        return T.smooth_moving_average(var_cfg, var, **kwargs)
    if tool == TOOLS[1]:
        return T.smooth_isotonic(var_cfg, var, **kwargs)
    if tool == TOOLS[2]:
        return T.cap_floor(var_cfg, var, **kwargs)
    return T.round_relativities(var_cfg, var, **kwargs)


def _tool_parameters(tool: str, var_cfg, key) -> dict:
    """The parameter widgets of ``tool``, as keyword arguments for the engine."""
    c1, c2 = st.columns(2)
    kwargs: dict = {}
    if tool == TOOLS[0]:
        kwargs["window"] = int(
            c1.number_input(
                "Window (bands)",
                min_value=3,
                max_value=25,
                value=T.DEFAULT_WINDOW,
                step=2,
                key=key("window"),
                help="How many bands each average covers: 3 is the band and its "
                "two neighbours. Odd numbers only, so the average stays centred "
                "on the band it replaces.",
            )
        )
    elif tool == TOOLS[1]:
        kwargs["direction"] = c1.selectbox(
            "Direction",
            ["increasing", "decreasing"],
            key=key("direction"),
            help="The relativity may not turn back as the factor rises. Bands "
            "that break the direction are pooled with their neighbours.",
        )
    elif tool == TOOLS[2]:
        kwargs["floor"] = c1.number_input(
            "Floor (empty = none)",
            value=None,
            min_value=0.0001,
            step=0.05,
            format="%.4f",
            key=key("floor"),
        )
        kwargs["cap"] = c2.number_input(
            "Cap (empty = none)",
            value=None,
            min_value=0.0001,
            step=0.05,
            format="%.4f",
            key=key("cap"),
        )
    else:
        how = c1.radio(
            "Round to",
            ["decimals", "a step"],
            horizontal=True,
            key=key("round_to"),
        )
        if how == "decimals":
            kwargs["decimals"] = int(
                c2.number_input(
                    "Decimal places",
                    min_value=0,
                    max_value=6,
                    value=2,
                    key=key("decimals"),
                )
            )
        else:
            kwargs["step"] = float(
                c2.number_input(
                    "Step",
                    value=0.05,
                    min_value=0.0001,
                    step=0.01,
                    format="%.4f",
                    key=key("step"),
                    help="0.05 rounds 1.083 to 1.10 — the way a published rate "
                    "table is printed.",
                )
            )
    if var_cfg.type == "categorical" and tool in (TOOLS[0], TOOLS[1]):
        kwargs["ordered"] = st.checkbox(
            "The levels of this factor are in a meaningful order",
            value=False,
            key=key("ordered"),
            help="Levels are listed most-exposed first, which is not an order of "
            "the risk. Tick this only when the table really does read in order "
            "(a banded or graded factor), because smoothing averages each level "
            "with the ones next to it.",
        )
    return kwargs


def _claims_change(
    run, var: str, values: list[float], train: pl.DataFrame
) -> float | None:
    """``expected claims after this tool / expected claims now − 1`` on the
    training rows — the **true** effect of the tool on the book.

    Computed by scoring the two sets of tables, not from the table itself: the
    premium is the exposure-weighted mean of the *relativities* (and of every
    other factor's), which no average of logs can stand in for.
    """
    cfg = S.project().models[run.name]
    if train.is_empty() or cfg.target is None:
        return None
    now = expected_claims(run.rate_model, train, cfg)
    if not now > 0:
        return None
    after = expected_claims(T.preview_model(run.rate_model, var, values), train, cfg)
    return after / now - 1.0


def _pct(change: float | None, *, zero: str = "no change") -> str:
    """A percentage change for a metric tile. Only an exactly-zero change (to
    1e-9) reads as "no change" — a tool that moves the book by half a per cent
    must never say it moved nothing."""
    if change is None:
        return "—"
    return zero if abs(change) < 1e-9 else f"{change:+.3%}"


def _tools(
    run, var: str, fitted: pl.DataFrame, working: pl.DataFrame, df: pl.DataFrame
) -> None:
    """The Tools expander above the editor: pick a tool, see what it would do,
    apply it as ordinary adjustments."""
    cfg = S.project().models[run.name]
    var_cfg = run.rate_model.variables[var]
    is_linear = var_cfg.type == "linear"

    def key(name: str) -> str:
        return S.widget_key(f"tool_{name}_{run.name}_{var}")

    with st.expander("Tools — smooth, cap / floor, round"):
        st.caption(
            "Each tool works on the bands of **this** table, never on the "
            "*Other / Unknown* row, and is saved as ordinary adjustments (no "
            "refit). A smoothing keeps the exposure-weighted mean of the "
            "**log** relativities exactly where it is — that is the shape rule "
            "— but that is *not* the same as leaving the premium alone: the "
            "**expected claims** figure below is what the book actually does, "
            "and *Rebalance base rate*, under the table, puts it back to where "
            "the fitted model had it."
        )
        tool = st.selectbox("Tool", TOOLS, key=key("which"))
        kwargs = _tool_parameters(tool, var_cfg, key)
        try:
            result = _tool_result(var_cfg, var, tool, kwargs)
        except T.ToolingError as exc:
            st.info(str(exc))
            return
        train = _ae_frame(df, "train")
        claims_change = ui.guarded(
            lambda: _claims_change(run, var, result.values, train),
            "Pricing this tool",
        )
        ui.metric_row(
            [
                ("bands that would change", str(result.changed), None),
                (
                    "expected claims (training)",
                    _pct(claims_change),
                    "The real effect on the book: total expected claims on the "
                    "training rows with these tables against the tables now. "
                    "This is the money — a smoothing that keeps the mean log "
                    "relativity can still move it, and usually does.",
                ),
                (
                    "mean log relativity now",
                    ui.fmt(result.log_mean_before, digits=6),
                    "Exposure-weighted mean of the log relativities: the shape "
                    "rule a smoothing preserves. It is a geometric average, so "
                    "it is not the premium — read the expected-claims tile for "
                    "that.",
                ),
                (
                    "after this tool",
                    ui.fmt(result.log_mean_after, digits=6),
                    None,
                ),
            ]
        )
        preview = variable_frame(T.apply_values(var_cfg, result.values))
        if is_linear:
            enc = run.spec[var]
            chart = C.linear_curve_chart(
                working,
                title=f"{var} — now and after this tool",
                working=preview,
                clamp=(enc.lo, enc.hi),
                x_base=var_cfg.x_base,
                log_x=enc.lo > 0 and enc.hi / enc.lo > 100,
                name="now",
                working_name="after this tool",
            )
        else:
            chart = C.relativity_chart(
                working,
                title=f"{var} — now and after this tool",
                working=preview,
                name="now",
                working_name="after this tool",
            )
        st.plotly_chart(chart, width="stretch")
        st.caption(
            result.note
            + (
                ""
                if claims_change is None
                else " On the training rows this would change total expected "
                f"claims by **{_pct(claims_change, zero='nothing at all')}** — "
                "the base rate is not refitted, so that change stays until you "
                "rebalance it."
            )
        )
        if result.uniform_weights:
            st.warning(
                "This table carries no training exposure (it was built by hand "
                "or read from a file written before 0.4), so every band counted "
                "the same in the average above (the expected-claims figure is "
                "unaffected: it is measured on the data)."
            )
        if not result.changed:
            st.caption("Nothing would change: the table already looks like that.")
        if st.button(
            "Apply to the table",
            key=key("apply"),
            type="primary",
            disabled=not result.changed,
        ):
            before = S.edit_state(run.name)
            changed, errors = G.apply_row_edits(
                cfg,
                var,
                var_cfg.table,
                fitted["relativity"].to_list(),
                result.values,
                require_positive=is_linear,
                other_label=var_cfg.other_label,
            )
            if changed:
                ui.flash(
                    "success",
                    f"**{result.tool}** applied to **{var}**: "
                    f"{result.changed} band(s) changed, total expected claims "
                    f"{_pct(claims_change, zero='unchanged')}. {result.note}",
                )
            _apply(run.name, changed, errors, before)


# --------------------------------------------------------------------------
# off-balance: what the edits did to the book, and putting it back
# --------------------------------------------------------------------------
def _off_balance(run, df: pl.DataFrame) -> tuple[float, float] | None:
    """``(off-balance, the base rate that removes it)``.

    The off-balance is ``total expected claims now / total expected claims as
    fitted − 1`` on the training rows: what every edit to this model's tables —
    typed, smoothed, capped, rounded — has done to the book. Predictions are
    linear in the base rate, so the base rate that puts the book back is one
    ratio away.
    """
    cfg = S.project().models[run.name]
    target = S.fitted_expected_claims(run.name)
    train = _ae_frame(df, "train")
    if target is None or not target > 0 or train.is_empty() or cfg.target is None:
        return None
    current = expected_claims(run.rate_model, train, cfg)
    if not current > 0:
        return None
    return current / target - 1.0, run.rate_model.base_rate * target / current


def _rebalance(run, df: pl.DataFrame) -> None:
    """What the adjustments have done to the total expected claims, and a
    one-click base-rate override that takes it back to the fitted level."""
    p = S.project()
    cfg = p.models[run.name]
    if not cfg.adjustments and cfg.base_rate_override is None:
        return
    numbers = ui.guarded(lambda: _off_balance(run, df), "Measuring the off-balance")
    if numbers is None:
        return
    off, base = numbers
    balanced = abs(off) < 1e-9
    c1, c2 = st.columns([3, 1])
    c1.caption(
        "**The book is balanced**: with these tables the model expects the same "
        "total claims on the training rows as the fitted model did."
        if balanced
        else f"**Off-balance {off:+.3%}** — with these tables the model expects "
        f"{abs(off):.3%} "
        + ("more" if off > 0 else "less")
        + " in total claims on the training rows than the fitted model did. "
        "Editing a table does not refit the base rate, so this stays until you "
        "put it back. *Rebalance* sets the base-rate override to "
        f"**{ui.fmt(base, digits=6)}**, which restores the fitted total exactly; "
        "it changes no relativity and is one undo step."
    )
    if c2.button(
        "Rebalance base rate",
        key=S.widget_key("tables_rebalance"),
        disabled=balanced,
        help="Off-balance correction: keep the shape you have edited and put "
        "the overall level back where the fit had it.",
    ):
        before = S.edit_state(run.name)
        cfg.base_rate_override = base
        ui.flash(
            "success",
            f"Base rate rebalanced to {base:.6g}: the {abs(off):.3%} the edits "
            + ("added to" if off > 0 else "took off")
            + " the book is gone and total expected claims on the training rows "
            "are back where the fitted model had them. No relativity changed; "
            "*Undo* puts the old base rate back.",
        )
        _apply(run.name, True, [], before)


# --------------------------------------------------------------------------
# snapshots
# --------------------------------------------------------------------------
def _snapshot_version(cfg, choice: str) -> tuple[list, float | None] | None:
    """The ``(adjustments, base-rate override)`` a snapshot selector stands for:
    nothing at all for the model as fitted, the model's own for "the tables
    now", else the snapshot's. ``None`` when the name is not one of them."""
    if choice == FITTED_OPTION:
        return [], None
    if choice == CURRENT_OPTION:
        return list(cfg.adjustments), cfg.base_rate_override
    for snap in cfg.snapshots:
        if snap.name == choice:
            return list(snap.adjustments), snap.base_rate_override
    return None


def _snapshots(run) -> None:
    """Create, list, restore and compare named snapshots of the tables."""
    p = S.project()
    cfg = p.models[run.name]

    def key(name: str) -> str:
        return S.widget_key(f"snap_{name}_{run.name}")

    with st.expander(f"Snapshots ({len(cfg.snapshots)})"):
        st.caption(
            "A snapshot names the tables as they stand now — the fit plus this "
            "model's adjustments — and is kept in the project file, so it "
            "survives a reload and a refit. Restoring one puts those "
            "adjustments back; comparing two lists every band that moved "
            "between them."
        )
        c1, c2 = st.columns([3, 1])
        name = c1.text_input(
            "Name",
            key=key("name"),
            placeholder="before smoothing DrivAge",
            label_visibility="collapsed",
        )
        if c2.button("Snapshot as…", key=key("create")):
            clean = name.strip()
            if not clean:
                st.error("Give the snapshot a name first.")
            elif any(sn.name == clean for sn in cfg.snapshots):
                st.error(f"This model already has a snapshot called {clean!r}.")
            elif clean in (FITTED_OPTION, CURRENT_OPTION):
                st.error(
                    f"{clean!r} is the name of one of the built-in versions in the "
                    "comparison below; give the snapshot another name."
                )
            else:
                cfg.snapshots.append(
                    TableSnapshot(
                        name=clean,
                        created_at=datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        adjustments=copy.deepcopy(cfg.adjustments),
                        base_rate_override=cfg.base_rate_override,
                    )
                )
                S.touch()
                ui.flash(
                    "success",
                    f"Snapshot **{clean}** taken "
                    f"({len(cfg.adjustments)} adjustment(s)).",
                )
                st.rerun()
        if not cfg.snapshots:
            st.caption("No snapshots yet.")
            return
        ui.polars_table(
            pl.DataFrame(
                [
                    {
                        "snapshot": sn.name,
                        "taken": sn.created_at,
                        "adjustments": len(sn.adjustments),
                    }
                    for sn in cfg.snapshots
                ]
            )
        )
        names = [sn.name for sn in cfg.snapshots]
        c1, c2, c3 = st.columns([3, 1, 1])
        chosen = c1.selectbox(
            "Snapshot", names, key=key("chosen"), label_visibility="collapsed"
        )
        if c2.button("Restore", key=key("restore")):
            snap = next(sn for sn in cfg.snapshots if sn.name == chosen)
            gone = missing_variables(run.rate_model, snap.adjustments)
            if gone:
                # nothing is changed and nothing is saved: a snapshot older than
                # the design cannot be applied, and applying half of it would be
                # worse than applying none
                st.error(
                    f"Snapshot **{chosen}** cannot be restored: it adjusts "
                    + ", ".join(f"**{g}**" for g in gone)
                    + ", which this model no longer has. Nothing was changed. "
                    "Put the factor back on the Model page and refit, or take a "
                    "new snapshot and delete this one."
                )
            else:
                before = S.edit_state(run.name)
                cfg.adjustments = copy.deepcopy(snap.adjustments)
                cfg.base_rate_override = snap.base_rate_override
                ui.flash(
                    "success",
                    f"Restored the tables of snapshot **{chosen}** "
                    f"({len(snap.adjustments)} adjustment(s)). *Undo* puts back "
                    "the tables and the base rate you had.",
                )
                _apply(run.name, True, [], before)
        # deleting a snapshot is the one destructive action undo does not cover,
        # so it asks twice (the pattern the Project page uses for "New project")
        confirm_key = f"_confirm_delete_snapshot_{run.name}"
        pending = st.session_state.get(confirm_key)
        if pending is not None and pending != chosen:
            st.session_state.pop(confirm_key, None)  # another snapshot: ask again
        confirming = pending == chosen
        if c3.button(
            "Delete twice" if confirming else "Delete",
            key=key("delete"),
            help="A snapshot is not covered by Undo, so this asks twice.",
        ):
            if not confirming:
                st.session_state[confirm_key] = chosen
                # the button's own label changes on the next run, so redraw —
                # and the notice has to survive that rerun (ui.flash, not
                # st.warning, or Streamlit >= 1.63 drops it)
                ui.flash(
                    "warning",
                    f"Delete the snapshot **{chosen}** for good? Undo cannot bring "
                    "it back. The button now reads *Delete twice*; press it to "
                    "confirm, or pick another snapshot to forget it.",
                )
                st.rerun()
            else:
                cfg.snapshots = [sn for sn in cfg.snapshots if sn.name != chosen]
                st.session_state.pop(confirm_key, None)
                ui.flash("success", f"Snapshot **{chosen}** deleted.")
                S.touch()
                st.rerun()

        st.markdown("**Compare two snapshots**")
        options = [FITTED_OPTION, CURRENT_OPTION, *names]
        c1, c2, c3 = st.columns([2, 2, 2])
        left = c1.selectbox("Compare", options, index=0, key=key("diff_a"))
        right = c2.selectbox(
            "with", options, index=min(1, len(options) - 1), key=key("diff_b")
        )
        tol = c3.number_input(
            "Report a band when |Δ log relativity| exceeds",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.005,
            format="%.3f",
            key=key("diff_tol"),
        )
        if left == right:
            st.caption("Pick two different versions to see the differences.")
            return
        versions = [_snapshot_version(cfg, name) for name in (left, right)]
        if any(v is None for v in versions):  # a snapshot deleted in another tab
            st.info("Pick two versions that still exist.")
            return
        diff = ui.guarded(
            lambda: rate_model_diff(
                *[
                    rate_model_for(p, run, adj, base_rate_override=override)
                    for adj, override in versions
                ],
                tol,
            ),
            "Comparing the snapshots",
        )
        if diff is None:
            return
        if diff.is_empty():
            st.success("The two versions charge exactly the same premium.")
            return
        shown = describe_diff(diff, left, right)
        ui.polars_table(shown)
        st.download_button(
            "Download the differences (.csv)",
            ui.frame_bytes(shown),
            file_name=f"{ui.safe_filename(run.name)}_snapshot_diff.csv",
            key=key("diff_dl"),
        )


# --------------------------------------------------------------------------
def render() -> None:
    st.title("Rate tables")
    ui.status_bar()
    p = S.project()
    df = ui.require_data()
    if df is None:
        return
    c1, c2, c3 = st.columns([2, 2, 4])
    with c1:
        run = ui.run_selector("Model", key=S.widget_key("tables_run"))
    if run is None:
        return
    challenger = _challenger_selector(c2, run.name)
    cfg = p.models[run.name]
    n_inter = sum(
        1 for c in run.rate_model.variables.values() if c.type == "interaction"
    )
    with c3:
        ui.metric_row(
            [
                (
                    "base rate",
                    ui.fmt(run.rate_model.base_rate, digits=6),
                    "prediction for the base risk: relativity 1.0 on every main "
                    "effect. It comes from the main-effect fit alone (stage 1), so "
                    "it and the main tables are the same with and without an "
                    "interaction; cells are adjustments on top (1.00 = none)",
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
    chosen = st.selectbox("Variable", list(display), key=S.widget_key("tables_var"))
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
        if challenger is not None:
            st.caption(
                f"The challenger **{challenger.name}** is not overlaid on a "
                "heatmap — compare the two models cell by cell on the "
                "**Compare** page."
            )
    else:
        _tools(run, var, run.tables[var], rate_model_tables(run.rate_model)[var], df)
        working = _main_effect(run, var, df, challenger)

    b1, b2, b3, b4 = st.columns(4)
    if b1.button(
        "Undo",
        key=S.widget_key("tables_undo"),
        disabled=not S.can_undo(run.name),
        help="Step back through this session's edits to the tables of this "
        "model — one step per edit, tool or reset.",
    ):
        S.undo(run.name)
        st.rerun()
    if b2.button(
        "Redo", key=S.widget_key("tables_redo"), disabled=not S.can_redo(run.name)
    ):
        S.redo(run.name)
        st.rerun()
    if b3.button(
        "Reset this variable",
        key=S.widget_key("tables_reset_var"),
        disabled=not any(a.variable == var for a in cfg.adjustments),
    ):
        before = S.edit_state(run.name)
        cfg.adjustments = [a for a in cfg.adjustments if a.variable != var]
        _apply(run.name, True, [], before)
    if b4.button(
        "Reset all", key=S.widget_key("tables_reset_all"), disabled=not cfg.adjustments
    ):
        before = S.edit_state(run.name)
        cfg.adjustments = []
        _apply(run.name, True, [], before)
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

    _rebalance(run, df)
    _snapshots(run)

    st.subheader("Export")
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Excel rate tables (.xlsx)",
        ui.excel_bytes(run),
        file_name=f"{p.name}_{run.name}_rate_tables.xlsx",
        key=S.widget_key("dl_xlsx"),
        help="Current (adjusted) tables; interactions get a long sheet and a matrix sheet",
    )
    c2.download_button(
        "Scorer (.easyglm)",
        ui.easyglm_bytes(run),
        file_name=f"{p.name}_{run.name}.easyglm",
        key=S.widget_key("dl_easyglm"),
    )
    c3.download_button(
        "This table (.csv)",
        ui.frame_bytes(working),
        file_name=f"{run.name}_{var}.csv",
        key=S.widget_key("dl_csv"),
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
