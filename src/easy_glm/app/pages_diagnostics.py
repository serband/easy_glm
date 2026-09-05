"""Page 7 — Diagnostics: A/E by variable and by pair, lift, double lift,
residual factor and pair search, regularisation path."""

from __future__ import annotations

import numpy as np
import polars as pl
import streamlit as st

from easy_glm.core.design import NUMERIC_DTYPES
from easy_glm.workflow import (
    ModelRun,
    ae_by_pair,
    ae_by_variable,
    alpha_path,
    double_lift,
    gini,
    lift_table,
    null_model_predict,
    pearson_dispersion,
    predictions_effectively_equal,
    residual_factor_search,
    residual_pair_search,
    totals,
    train_holdout,
)

from . import charts as C
from . import grids as G
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


def _knots_and_levels(run: ModelRun) -> tuple[dict, dict]:
    knots: dict[str, list[float]] = {}
    levels: dict[str, list[str]] = {}
    for v in run.spec.main_effects:
        enc = run.spec[v]
        if hasattr(enc, "band_edges"):
            knots[v] = enc.band_edges()
        elif hasattr(enc, "levels"):
            levels[v] = list(enc.levels)
    return knots, levels


def _ae_grouping(run: ModelRun, variable: str) -> dict:
    """The model's own rows are the diagnostic groups for a fitted factor."""
    if variable not in run.spec.main_effects:
        return {}
    enc = run.spec[variable]
    knots = enc.band_edges() if hasattr(enc, "band_edges") else None
    levels = list(enc.levels) if hasattr(enc, "levels") else None
    var_cfg = run.rate_model.variables[variable]
    return {
        "knots": knots,
        "fitted_labels": run.tables[variable]["label"].to_list(),
        "fitted_levels": levels,
        "other_label": var_cfg.other_label,
    }


def _variable_ae(
    run: ModelRun,
    challenger: ModelRun | None,
    frame: pl.DataFrame,
    variable: str,
    n_bins: int,
) -> tuple[pl.DataFrame, pl.DataFrame | None, str | None]:
    """A/E tables on one common grouping, including an optional challenger."""
    actual, expected, weight = totals(frame, run.config, run.predict(frame))
    grouping = _ae_grouping(run, variable)
    table = ae_by_variable(
        frame, variable, actual, expected, weight, n_bins=n_bins, **grouping
    )
    if challenger is None:
        return table, None, None
    try:
        challenger_expected = totals(
            frame, challenger.config, challenger.predict(frame)
        )[1]
        compare = ae_by_variable(
            frame,
            variable,
            actual,
            challenger_expected,
            weight,
            n_bins=n_bins,
            **grouping,
        )
    except Exception as exc:  # noqa: BLE001 - a page message, never a traceback
        return table, None, f"Cannot score {challenger.name} on these rows: {exc}"
    return table, compare, None


def _pair_heatmap(
    frame, a, b, actual, expected, w, knots, levels, *, title, n_bins=10
) -> pl.DataFrame:
    tbl = ae_by_pair(
        frame,
        a,
        b,
        actual,
        expected,
        w,
        n_bins=n_bins,
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
            title=title,
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
    return tbl


def render() -> None:
    st.title("Diagnostics")
    ui.status_bar()
    p = S.project()
    df = ui.require_data()
    if df is None:
        return
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        run = ui.run_selector("Model", key=S.widget_key("diag_run"))
    if run is None:
        return
    missing = [c for c in run.spec.required_columns if c not in df.columns]
    if missing:
        st.error(
            "The prepared data no longer has columns the model needs: "
            + ", ".join(missing)
            + ". Refit on the Model page after fixing the Variables page."
        )
        return
    fitted = [n for n in S.fitted_models() if n != run.name]
    # the sidebar's "default comparison model" is the default; picking another one here
    # overrides it for this page (the widget's key carries the sidebar value, so
    # moving the sidebar selector re-defaults this one)
    sidebar = S.challenger()
    options = ["(none)"] + fitted
    default = sidebar if sidebar in fitted else "(none)"
    with c2:
        chal_name = st.selectbox(
            "Compare with (challenger)",
            options,
            index=options.index(default),
            key=S.widget_key(f"diag_chal_{sidebar}"),
            help="Defaults to the sidebar's *Default comparison model*; change it here to "
            "override it on this page. The full side-by-side view is the "
            "**Compare** page.",
        )
        unfitted = [
            name for name in p.models if name != run.name and name not in fitted
        ]
        if unfitted:
            names = ", ".join(unfitted)
            verb = "is" if len(unfitted) == 1 else "are"
            st.caption(
                f"{names} {verb} defined but not fitted; fit "
                f"{'it' if len(unfitted) == 1 else 'them'} before comparing models."
            )
    challenger = S.get_run(chal_name) if chal_name != "(none)" else None
    with c3:
        which = st.radio(
            "Rows",
            ["holdout", "train", "all"],
            horizontal=True,
            key=S.widget_key("diag_subset"),
            help="Holdout rows were not used to fit the model, so they are the best independent check.",
        )
    frame = _subset(df, which)
    no_global_rows = frame.is_empty()
    if no_global_rows:
        st.warning(
            "No rows in this subset. A/E by variable still shows every available "
            "train and holdout set below."
        )
    cfg = run.config
    train_frame, holdout_frame = train_holdout(df, p.data.split)
    train_pred = run.predict(train_frame)
    train_actual, train_expected, train_weight = totals(train_frame, cfg, train_pred)
    actual = expected = w = np.array([], dtype=float)
    if not no_global_rows:
        pred = run.predict(frame)
        actual, expected, w = totals(frame, cfg, pred)
    exp_chal = None
    chal_predictions = None
    if challenger is not None and not no_global_rows:
        chal_missing = [
            c for c in challenger.spec.required_columns if c not in df.columns
        ]
        if chal_missing:
            st.warning(
                f"Challenger {chal_name} cannot be scored: missing columns "
                + ", ".join(chal_missing)
            )
            challenger = None
        else:
            chal_predictions = challenger.predict(frame)
            exp_chal = totals(frame, challenger.config, chal_predictions)[1]
    knots, levels = _knots_and_levels(run)
    reserved = {cfg.target, cfg.weight, cfg.offset, p.data.split.column} - {None}
    variables = [c for c in frame.columns if c not in reserved]
    mains = list(run.spec.main_effects)
    in_model = [v for v in variables if v in mains]
    others = [v for v in variables if v not in mains]

    _metrics(run, challenger)
    tabs = st.tabs(
        [
            "A/E by variable",
            "A/E by pair",
            "Lift",
            "Double lift",
            "Residual factors",
            "Regularisation path",
        ]
    )

    with tabs[0]:
        c1, c2 = st.columns([3, 1])
        var = c1.selectbox(
            "Variable",
            in_model + others,
            format_func=lambda v: v if v in mains else f"{v} (not in model)",
            key=S.widget_key("diag_var"),
        )
        numeric = df.schema[var] in NUMERIC_DTYPES
        uses_fitted_groups = var in mains and numeric
        n_bins = 20
        if numeric and not uses_fitted_groups:
            n_bins = c2.slider(
                "Temporary groups",
                5,
                50,
                20,
                key=S.widget_key("diag_bins"),
                help="How many equal-frequency groups to use for this numeric column. "
                "These groups are for this diagnostic only; they do not change the model.",
            )
        elif numeric:
            c2.caption("Using this model's fitted bands.")

        ae_sets = [("train", train_frame)]
        if not holdout_frame.is_empty():
            ae_sets.append(("holdout", holdout_frame))
        else:
            st.info(
                "There are no holdout rows, so this view shows training experience only."
            )
        for subset_name, subset_frame in ae_sets:
            tbl, cmp_tbl, challenger_problem = _variable_ae(
                run, challenger, subset_frame, var, n_bins
            )
            st.plotly_chart(
                C.ae_chart(
                    tbl,
                    title=f"{var} — actual vs expected ({subset_name})",
                    compare=cmp_tbl,
                    compare_name=chal_name,
                ),
                width="stretch",
            )
            if challenger_problem:
                st.warning(challenger_problem)
            with st.expander(f"{subset_name.title()} table"):
                ui.polars_table(tbl)

    if no_global_rows:
        return

    with tabs[1]:
        st.caption(
            "Actual / expected in every **cell** of two variables. With both mains "
            "in the model, a block of red or blue cells with real exposure is an "
            "interaction the model is missing — add it on the Model page. This "
            "diagnostic follows the Rows choice above; it is not used by the "
            "automatic missing-interaction search."
        )
        options = in_model + others
        c1, c2, c3 = st.columns([2, 2, 1])
        a = c1.selectbox(
            "Rows",
            options,
            format_func=lambda v: v if v in mains else f"{v} (not in model)",
            key=S.widget_key("pair_a"),
        )
        b = c2.selectbox(
            "Columns",
            options,
            index=min(1, len(options) - 1),
            format_func=lambda v: v if v in mains else f"{v} (not in model)",
            key=S.widget_key("pair_b"),
        )
        nb = c3.slider(
            "Bands (not in model)",
            3,
            20,
            8,
            key=S.widget_key("pair_bins"),
            help="Temporary equal-frequency groups for a numeric factor not already banded by this model.",
        )
        if a == b:
            st.error("Pick two different variables.")
        else:
            tbl = _pair_heatmap(
                frame,
                a,
                b,
                actual,
                expected,
                w,
                knots,
                levels,
                title=f"{a} × {b} — actual / expected by cell ({which})",
                n_bins=nb,
            )
            with st.expander("Table"):
                ui.polars_table(tbl)

    with tabs[2]:
        n = st.slider(
            "Bins",
            5,
            20,
            10,
            key=S.widget_key("lift_bins"),
            help="Equal-exposure groups ordered from lowest to highest predicted rate. More bins show more detail but are noisier.",
        )
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

    with tabs[3]:
        st.caption(
            "Bins run from the selected model being cheapest relative to the incumbent "
            "or null model, to most expensive. A pattern where one A/E line stays "
            "closer to 1 is extra discrimination, not just a different overall level."
        )
        exp_b, name_b = None, ""
        if exp_chal is not None and challenger is not None:
            exp_b, name_b = exp_chal, chal_name
        else:
            train, _holdout = train_holdout(df, p.data.split)
            try:
                null_prediction = null_model_predict(p, cfg, train, frame)
                exp_b = totals(frame, cfg, null_prediction)[1]
                name_b = "null model"
                st.caption(
                    "No challenger selected — benchmark is an intercept-only model calibrated on training rows."
                )
            except Exception as exc:  # noqa: BLE001 - page errors are messages
                st.warning(
                    f"Could not fit the training-calibrated null benchmark: {exc}"
                )
        if exp_b is not None and np.nansum(exp_b) > 0:
            if (
                challenger is not None
                and chal_predictions is not None
                and predictions_effectively_equal(pred, chal_predictions)
            ):
                st.info(
                    "These models make the same predictions on the selected rows "
                    "(within numerical precision), so their A/E lines overlap."
                )
            dl = double_lift(actual, expected, exp_b, w, n_bins=10)
            st.plotly_chart(
                C.double_lift_chart(dl, name_a=run.name, name_b=name_b), width="stretch"
            )
            with st.expander("Table"):
                ui.polars_table(dl)
        elif exp_b is not None:
            st.warning("The benchmark column has no positive values.")

    with tabs[4]:
        st.markdown("**Missing factors** — variables not in the model")
        st.caption(
            "Scored on the **training rows only** — the holdout is for validation, "
            "not for choosing variables. Signal is a noise-adjusted Pearson excess "
            "z-score across each variable's bands; large values point at factors "
            "the model is missing."
        )
        candidates = [c for c in others if p.data.roles.get(c) not in ("id", "ignore")]
        if not candidates:
            st.info(
                "Every available variable is already in the model, an id, or "
                "explicitly ignored."
            )
        elif (
            st.button("Run residual search", key=S.widget_key("rfs_go"))
            or "rfs_training_result_v2" in st.session_state
        ):
            counts = cfg.family == "poisson"
            phi = (
                1.0
                if counts
                else pearson_dispersion(train_actual, train_expected, len(run.fit.coef))
            )
            res = residual_factor_search(
                train_frame,
                candidates,
                train_actual,
                train_expected,
                train_weight,
                dispersion=phi,
            )
            st.session_state.rfs_training_result_v2 = res
            st.session_state.rfs_result = res
            st.dataframe(
                res,
                width="stretch",
                hide_index=True,
                column_config={
                    "signal": st.column_config.NumberColumn(
                        "signal (excess z-score)", format="%.2f"
                    ),
                    "sd_log_ae": st.column_config.NumberColumn(
                        "sd of log A/E", format="%.3f"
                    ),
                    "max_abs_log_ae": st.column_config.NumberColumn(
                        "max |log A/E|", format="%.3f"
                    ),
                    "exposure_share": st.column_config.NumberColumn(
                        "exposure retained", format="percent"
                    ),
                },
            )
            if res.height:
                top = st.selectbox(
                    "Show", res["variable"].to_list(), key=S.widget_key("rfs_show")
                )
                t = ae_by_variable(
                    train_frame,
                    top,
                    train_actual,
                    train_expected,
                    train_weight,
                    n_bins=10,
                )
                st.plotly_chart(
                    C.ae_chart(
                        t,
                        title=f"{top} (not in model) — actual vs expected (train)",
                    ),
                    width="stretch",
                )
        st.markdown("**Missing interactions** — pairs of the model's predictors")
        counts = cfg.family == "poisson"
        phi = (
            1.0
            if counts
            else pearson_dispersion(train_actual, train_expected, len(run.fit.coef))
        )
        st.caption(
            "Scored on the **training rows only** — the holdout is for validation, "
            "not for choosing interactions. For every pair of the model's predictors: "
            "the cells' Pearson excess "
            "after re-fitting the two margins, as a z-score (numerics in 8 coarse "
            "bands, cells with fewer than 3 expected ignored; many small noisy cells "
            "do not outrank one large real effect). The top pairs are the "
            "interactions worth trying on the Model page."
            + (
                ""
                if counts
                else f" This is not a claim-count model, so the statistic is scaled by "
                f"the model's Pearson dispersion φ̂ = {phi:,.1f}; read it as a ranking "
                "rather than a calibrated z-score."
            )
        )
        existing = {frozenset((it.a, it.b)) for it in cfg.interactions}
        pairs = [
            (mains[i], mains[j])
            for i in range(len(mains))
            for j in range(i + 1, len(mains))
            if frozenset((mains[i], mains[j])) not in existing
        ]
        if len(mains) < 2:
            st.info("Pair search needs at least two predictors in the model.")
        elif not pairs:
            st.info("Every pair of predictors is already an interaction of this model.")
        elif (
            st.button("Search pairs", key=S.widget_key("rps_go"))
            or "rps_training_result_v2" in st.session_state
        ):
            with st.spinner(f"Scoring {len(pairs)} pairs ..."):
                res = residual_pair_search(
                    train_frame,
                    mains,
                    train_actual,
                    train_expected,
                    train_weight,
                    levels=levels,  # numerics in 8 coarse bands: enough claims per cell
                    pairs=pairs,
                    dispersion=phi,
                )
            st.session_state.rps_training_result_v2 = res
            st.session_state.rps_result = res
            if res.is_empty():
                st.info("No pair has enough populated cells to score.")
            else:
                st.dataframe(
                    res,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "signal": st.column_config.NumberColumn(
                            "signal (excess z-score)", format="%.1f"
                        ),
                        "sd_log_ae": st.column_config.NumberColumn(
                            "sd of log A/E", format="%.3f"
                        ),
                        "max_abs_log_ae": st.column_config.NumberColumn(
                            "max |log A/E|", format="%.3f"
                        ),
                    },
                )
                top_pair = st.selectbox(
                    "Show", res["pair"].to_list(), key=S.widget_key("rps_show")
                )
                row = res.filter(pl.col("pair") == top_pair).row(0, named=True)
                # the same 8-band grid the search scored, so worst_cell is visible
                _pair_heatmap(
                    train_frame,
                    row["a"],
                    row["b"],
                    train_actual,
                    train_expected,
                    train_weight,
                    {},
                    levels,
                    title=f"{top_pair} — actual / expected by cell (train), search bands",
                    n_bins=8,
                )

    with tabs[5]:
        path = alpha_path(run.fit)
        if path.height > 1:
            st.plotly_chart(C.alpha_path_chart(path), width="stretch")
        ui.polars_table(path)
