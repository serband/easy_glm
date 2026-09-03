"""Page 6 — Model: specify, fit, inspect."""

from __future__ import annotations

import polars as pl
import streamlit as st

from easy_glm.core.design import NUMERIC_DTYPES
from easy_glm.workflow import alpha_path, solve_base_rate, train_holdout
from easy_glm.workflow.project import FAMILIES

from . import charts as C
from . import pages_design as D
from . import state as S
from . import ui


def _column_pick(
    column, label: str, current: str | None, options: list[str], *, key: str, none: bool
) -> str | None:
    """Compatibility helper for the legacy private configuration routine."""
    choices = (["(none)"] if none else []) + options
    index = (
        0
        if current is None and none
        else choices.index(current) if current in choices else None
    )
    picked = column.selectbox(label, choices, index=index, key=key)
    if index is None:
        column.error(f"{label}: {current!r} is not a numeric column of the data")
        return current
    return None if picked == "(none)" else picked


def _fit_model_selector() -> str | None:
    """Choose an existing definition to fit without offering design edits."""
    p = S.project()
    names = list(p.models)
    if not names:
        st.info(
            "No model has been defined yet. Open the **Design** page to name a "
            "model and choose its family, target, weight and predictors."
        )
        return None
    current = st.session_state.get("model_current")
    if current not in names:
        current = p.champion if p.champion in names else names[0]
    name = st.selectbox(
        "Model to fit",
        names,
        index=names.index(current),
        key=S.widget_key("fit_model_select"),
    )
    st.session_state.model_current = name
    return name


def _fit_options(name: str) -> None:
    """Fitting and table-level options; the model definition lives on Design."""
    p = S.project()
    cfg = p.models[name]
    if st.session_state.pop("solved_base_rate", None) == name:
        st.session_state.pop(S.widget_key(f"bro_{name}"), None)
    with st.container(border=True):
        st.subheader("Fit settings")
        st.markdown("**Penalty**")
        c1, c2, c3, c4, c5 = st.columns(5)
        mode = c1.radio(
            "alpha",
            ["cross-validated", "fixed"],
            index=0 if cfg.penalty.alpha is None else 1,
            key=S.widget_key(f"pmode_{name}"),
            horizontal=True,
            help=(
                "The strength of regularisation. Cross-validated lets holdout folds "
                "choose it; fixed uses the number entered beside it."
            ),
        )
        alpha_value, alpha_problem = ui.repair_number(cfg.penalty.alpha, 0.001, "alpha")
        if alpha_problem:
            ui.flash("warning", alpha_problem)
        alpha = ui.number_in_range(
            c2,
            "alpha",
            value=alpha_value or 0.001,
            lo=0.0,
            hi=10.0,
            what="alpha",
            format="%.5f",
            key=S.widget_key(f"alpha_{name}"),
            disabled=mode != "fixed",
            help="Fixed regularisation strength. Larger values shrink more effects to 1.00.",
        )
        cv_value, cv_problem = ui.repair_number(
            cfg.penalty.cv, 5, "cv", integer=True, lo=2, hi=10
        )
        if cv_problem:
            ui.flash("warning", cv_problem)
        cv = c3.number_input(
            "CV folds",
            2,
            10,
            int(cv_value),
            key=S.widget_key(f"cv_{name}"),
            disabled=mode != "cross-validated",
            help="How many held-out folds are used to choose regularisation strength.",
        )
        n_alphas_value, n_alphas_problem = ui.repair_number(
            cfg.penalty.n_alphas, 20, "n_alphas", integer=True, lo=3, hi=100
        )
        if n_alphas_problem:
            ui.flash("warning", n_alphas_problem)
        n_alphas = c4.number_input(
            "alphas on path",
            3,
            100,
            int(n_alphas_value),
            key=S.widget_key(f"nalpha_{name}"),
            disabled=mode != "cross-validated",
            help="How many candidate regularisation strengths cross-validation tests.",
        )
        l1_value, l1_problem = ui.repair_number(
            cfg.penalty.l1_ratio, 1.0, "l1_ratio", lo=0.0, hi=1.0
        )
        if l1_problem:
            ui.flash("warning", l1_problem)
        l1 = c5.slider(
            "l1_ratio (1 = lasso)",
            0.0,
            1.0,
            l1_value,
            0.05,
            key=S.widget_key(f"l1_{name}"),
            help="1 keeps a sparse lasso model; lower values retain more, smaller effects.",
        )
        c1, c2, c3, c4 = st.columns(4)
        base = c1.radio(
            "Base risk for tables",
            ["modal", "reference"],
            index=0 if cfg.base == "modal" else 1,
            horizontal=True,
            key=S.widget_key(f"base_{name}"),
            help="The 1.00 point used to display the rate tables: most-exposed band or reference level.",
        )
        bro_value, bro_problem = ui.repair_number(
            cfg.base_rate_override, 0.0, "base_rate_override"
        )
        if bro_problem:
            ui.flash("warning", bro_problem)
        bro = ui.number_in_range(
            c2,
            "Base rate override (0 = exact)",
            value=bro_value or 0.0,
            lo=0.0,
            what="The base rate override",
            format="%.6f",
            key=S.widget_key(f"bro_{name}"),
            help="An overall multiplier on every prediction. Leave at 0 for the fitted level.",
        )
        run_now = S.get_run(name)
        if bro and run_now is not None:
            fitted_base = run_now.rate_model.base_rate
            snapshots = run_now.rate_model.snapshots
            fitted_base = (
                snapshots[0].metadata.get("base_rate", fitted_base)
                if snapshots
                else fitted_base
            )
            if fitted_base and not 0.01 <= bro / fitted_base <= 100:
                c2.warning(
                    f"Override is {bro / fitted_base:,.0f}× the fitted base rate "
                    f"({fitted_base:.6g}); every prediction is scaled by that much."
                )
        _target_loss_ratio(c3, name, cfg, run_now)
        notes = c4.text_input("Notes", cfg.notes, key=S.widget_key(f"notes_{name}"))

    penalty = {
        "alpha": float(alpha) if mode == "fixed" else None,
        "cv": int(cv) if mode == "cross-validated" else None,
        "n_alphas": int(n_alphas),
        "l1_ratio": float(l1),
    }
    values = {
        "base": base,
        "base_rate_override": float(bro) or None,
        "notes": notes,
    }
    changed = False
    rebuild_only = False
    for field, value in values.items():
        if getattr(cfg, field) != value:
            setattr(cfg, field, value)
            changed = True
            rebuild_only = True
    for field, value in penalty.items():
        if getattr(cfg.penalty, field) != value:
            setattr(cfg.penalty, field, value)
            changed = True
    if changed:
        S.touch()
        if rebuild_only and S.get_run(name) is not None:
            S.refresh_adjustments(name)
        else:
            st.rerun()


def _config(name: str) -> None:
    p = S.project()
    cfg = p.models[name]
    # the solver below writes a new base-rate override into the project;
    # Streamlit refuses to set a widget's key once the widget exists, so the key
    # is dropped here — before the box is drawn — and the box falls back to the
    # number the project now holds
    if st.session_state.pop("solved_base_rate", None) == name:
        st.session_state.pop(S.widget_key(f"bro_{name}"), None)
    df = S.prepared_frame()
    numeric_cols = (
        [c for c, t in df.schema.items() if t in NUMERIC_DTYPES]
        if df is not None
        else []
    )
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        family = c1.selectbox(
            "Family",
            list(FAMILIES),
            index=list(FAMILIES).index(cfg.family) if cfg.family in FAMILIES else 0,
            key=S.widget_key(f"fam_{name}"),
        )
        power, power_problem = ui.repair_number(cfg.tweedie_power, 1.5, "tweedie_power")
        if power_problem:
            ui.flash("warning", power_problem)
        if family == "tweedie":
            power = float(
                ui.number_in_range(
                    c1,
                    "Tweedie power",
                    value=power,
                    lo=1.001,
                    hi=1.999,
                    step=0.05,
                    what="The Tweedie power",
                    format="%.3f",
                    key=S.widget_key(f"tw_{name}"),
                    help="Between 1 (Poisson: counts) and 2 (Gamma: amounts). "
                    "1.5 is the usual starting point for a pure premium — a "
                    "mass of policies with no claim plus a skewed amount for "
                    "those that do. Nearer 1 puts more weight on how often "
                    "claims happen, nearer 2 on how large they are.",
                )
            )
        if family == "binomial":
            c1.caption(
                "Binomial: the target must be 0/1 (or a proportion). The tables "
                "are **odds relativities** and the scorer returns a probability, "
                "so predictions are never multiplied by exposure."
            )
        if df is not None:
            target = _column_pick(
                c2,
                "Target",
                cfg.target,
                numeric_cols,
                key=S.widget_key(f"tgt_{name}"),
                none=False,
            )
            weight = _column_pick(
                c3,
                "Weight",
                cfg.weight,
                numeric_cols,
                key=S.widget_key(f"wgt_{name}"),
                none=True,
            )
            offset = _column_pick(
                c4,
                "Offset (linear scale)",
                cfg.offset,
                numeric_cols,
                key=S.widget_key(f"off_{name}"),
                none=True,
            )
        else:
            target, weight, offset = cfg.target, cfg.weight, cfg.offset
        div_key = S.widget_key(f"div_{name}")
        if weight is None and st.session_state.get(div_key):
            # Streamlit keeps a key's value even when the default changes: an
            # unticked-and-disabled box must not stay ticked from when there
            # was a weight column (the project holds False either way)
            st.session_state.pop(div_key, None)
        divide = st.checkbox(
            "Divide target by weight (model a rate, e.g. claims / exposure)",
            cfg.divide_target_by_weight and weight is not None,
            key=div_key,
            disabled=weight is None,
        )
        missing_preds = [v for v in cfg.predictors if v not in p.predictors]
        if missing_preds:
            st.error(
                "Predictor(s) no longer available (role changed or column gone): "
                + ", ".join(missing_preds)
                + " — the model keeps them until you change the list below."
            )
        preds = st.multiselect(
            "Predictors",
            sorted(set(p.predictors) | set(cfg.predictors)),
            default=list(cfg.predictors),
            format_func=lambda v: v if v in p.predictors else f"{v} (missing)",
            key=S.widget_key(f"preds_{name}"),
        )
        if cfg.interactions:
            bad = sorted(
                {
                    parent
                    for it in cfg.interactions
                    for parent in (it.a, it.b)
                    if parent not in preds
                }
            )
            st.caption(
                "Interactions (edit on the Design page): "
                + ", ".join(
                    (
                        f"**{it.name}** (min cell {mce:.2%})"
                        if (mce := ui.safe_float(it.min_cell_exposure, None))
                        is not None
                        else f"**{it.name}** (min cell {it.min_cell_exposure!r})"
                    )
                    for it in cfg.interactions
                )
                + (
                    f" — ⚠ parents no longer among the predictors: {', '.join(bad)}"
                    if bad
                    else ""
                )
            )
        st.markdown("**Penalty**")
        c1, c2, c3, c4, c5 = st.columns(5)
        mode = c1.radio(
            "alpha",
            ["cross-validated", "fixed"],
            index=0 if cfg.penalty.alpha is None else 1,
            key=S.widget_key(f"pmode_{name}"),
            horizontal=True,
        )
        alpha_value, alpha_problem = ui.repair_number(cfg.penalty.alpha, 0.001, "alpha")
        if alpha_problem:
            ui.flash("warning", alpha_problem)
        alpha = ui.number_in_range(
            c2,
            "alpha",
            value=alpha_value or 0.001,
            lo=0.0,
            hi=10.0,
            what="alpha",
            format="%.5f",
            key=S.widget_key(f"alpha_{name}"),
            disabled=mode != "fixed",
        )
        cv_value, cv_problem = ui.repair_number(
            cfg.penalty.cv, 5, "cv", integer=True, lo=2, hi=10
        )
        if cv_problem:
            ui.flash("warning", cv_problem)
        cv = c3.number_input(
            "CV folds",
            2,
            10,
            int(cv_value),
            key=S.widget_key(f"cv_{name}"),
            disabled=mode != "cross-validated",
        )
        n_alphas_value, n_alphas_problem = ui.repair_number(
            cfg.penalty.n_alphas, 20, "n_alphas", integer=True, lo=3, hi=100
        )
        if n_alphas_problem:
            ui.flash("warning", n_alphas_problem)
        n_alphas = c4.number_input(
            "alphas on path",
            3,
            100,
            int(n_alphas_value),
            key=S.widget_key(f"nalpha_{name}"),
            disabled=mode != "cross-validated",
        )
        l1_value, l1_problem = ui.repair_number(
            cfg.penalty.l1_ratio, 1.0, "l1_ratio", lo=0.0, hi=1.0
        )
        if l1_problem:
            ui.flash("warning", l1_problem)
        l1 = c5.slider(
            "l1_ratio (1 = lasso)",
            0.0,
            1.0,
            l1_value,
            0.05,
            key=S.widget_key(f"l1_{name}"),
        )
        c1, c2, c3, c4 = st.columns(4)
        base = c1.radio(
            "Base risk for tables",
            ["modal", "reference"],
            index=0 if cfg.base == "modal" else 1,
            horizontal=True,
            key=S.widget_key(f"base_{name}"),
        )
        bro_value, bro_problem = ui.repair_number(
            cfg.base_rate_override, 0.0, "base_rate_override"
        )
        if bro_problem:
            ui.flash("warning", bro_problem)
        bro = ui.number_in_range(
            c2,
            "Base rate override (0 = exact)",
            value=bro_value or 0.0,
            lo=0.0,
            what="The base rate override",
            format="%.6f",
            key=S.widget_key(f"bro_{name}"),
        )
        run_now = S.get_run(name)
        if bro and run_now is not None:
            fitted_base = run_now.rate_model.base_rate
            adj_count = len(cfg.adjustments)
            snapshots = run_now.rate_model.snapshots
            fitted_base = (
                snapshots[0].metadata.get("base_rate", fitted_base)
                if snapshots
                else fitted_base
            )
            if fitted_base and not 0.01 <= bro / fitted_base <= 100:
                c2.warning(
                    f"Override is {bro / fitted_base:,.0f}× the fitted base rate "
                    f"({fitted_base:.6g}); every prediction is scaled by that much."
                )
            del adj_count
        _target_loss_ratio(c3, name, cfg, run_now)
        notes = c4.text_input("Notes", cfg.notes, key=S.widget_key(f"notes_{name}"))
        mono_default = {
            v: vd.monotone
            for v, vd in p.design.variables.items()
            if vd.monotone and v in preds
        }
        if mono_default:
            st.caption(
                "Monotone constraints from the Design page: "
                + ", ".join(f"{k} {v}" for k, v in mono_default.items())
            )

    new_pen = dict(
        alpha=float(alpha) if mode == "fixed" else None,
        cv=int(cv) if mode == "cross-validated" else None,
        n_alphas=int(n_alphas),
        l1_ratio=float(l1),
        min_alpha_ratio=cfg.penalty.min_alpha_ratio,
    )
    new_vals = dict(
        family=family,
        tweedie_power=power,
        target=target,
        weight=weight,
        offset=offset,
        divide_target_by_weight=bool(divide) and weight is not None,
        predictors=list(preds),
        base=base,
        base_rate_override=float(bro) or None,
        notes=notes,
    )
    changed = False
    rebuild_only = False
    for k, v in new_vals.items():
        if getattr(cfg, k) != v:
            setattr(cfg, k, v)
            changed = True
            if k in ("base_rate_override", "notes"):
                rebuild_only = True
    for k, v in new_pen.items():
        if getattr(cfg.penalty, k) != v:
            setattr(cfg.penalty, k, v)
            changed = True
    if changed:
        S.touch()
        if rebuild_only and S.get_run(name) is not None:
            S.refresh_adjustments(name)
        else:
            # the status chips and the sidebar were drawn before this change:
            # redraw, or they would say "Fitted" next to "refit to update"
            st.rerun()


def _target_loss_ratio(col, name: str, cfg, run) -> None:
    """Enter the loss ratio the book should be written at; the base rate that
    achieves it on the training rows is solved and stored as the override.

    The solve puts **actual ÷ expected** at the number typed. For a rate-change
    model (the offset is the current premium) the prediction is the price and
    the actual is the loss, so that ratio is the loss ratio and the base rate
    becomes the overall rate change. For an ordinary model, 1.00 balances it to
    the data.
    """
    p = S.project()
    premium = p.current_premium
    ratio = ui.number_in_range(
        col,
        "Target loss ratio",
        value=1.0,
        lo=0.0001,
        hi=100.0,
        step=0.05,
        what="The target loss ratio",
        format="%.4f",
        key=S.widget_key(f"tlr_{name}"),
        help=(
            "Solve sets the base-rate override so that, on the training rows, "
            "total actual ÷ total expected equals this number. "
            + (
                f"This model's prediction is a multiple of {premium}, so that "
                "ratio is the loss ratio the book would be written at and the "
                "base rate becomes the overall rate change."
                if premium
                else "1.00 balances the model to the data (overall A/E exactly "
                "1); 1.05 leaves the expected 5 % below the actual."
            )
            + " The relativities are untouched, and solving again from an "
            "existing override gives the same answer."
        ),
    )
    df = S.prepared_frame()
    blocked = run is None or df is None
    if col.button(
        "Solve base rate",
        disabled=blocked,
        key=S.widget_key(f"solve_{name}"),
    ):
        train, _ = train_holdout(df, p.data.split)
        value = ui.guarded(
            lambda: solve_base_rate(run, train, float(ratio)),
            "Solving the base rate",
        )
        if value is None:
            return
        cfg.base_rate_override = float(value)
        st.session_state["solved_base_rate"] = name
        ui.flash(
            "success",
            f"Base rate override set to {value:.6g} — actual ÷ expected is now "
            f"{float(ratio):.4g} on the training rows"
            + (
                f", i.e. the book is priced to a {float(ratio):.1%} loss ratio."
                if premium
                else "."
            ),
        )
        S.touch()
        st.rerun()
    if blocked:
        col.caption("Fit the model to solve for a base rate.")


def _explain_fit_error(exc: Exception) -> str:
    text = str(exc)
    if "No variation in y" in text:
        return (
            "the target has no variation on the training rows (is the target the "
            "same column as the weight, or constant?)"
        )
    if "singular" in text.lower():
        return (
            "the solver hit a singular matrix — usually alpha = 0 (an unpenalised "
            "fit) or a column that is constant; use a small alpha such as 1e-4"
        )
    if "Weights sum to zero" in text or "strictly positive" in text:
        return f"{text} (is the weight column an exposure with positive values?)"
    return text


def _fit_and_results(name: str) -> None:
    p = S.project()
    df = S.prepared_frame()
    problems = p.validate(name, columns=df.columns if df is not None else None)
    run = S.get_run(name) if not problems else None
    stale = S.stale_run(name) if run is None else None
    c1, c2, c3 = st.columns([1, 1, 3])
    if c1.button(
        "Fit model",
        type="primary",
        disabled=bool(problems),
        key=S.widget_key(f"fit_{name}"),
    ):
        try:
            run = S.fit_model(name)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Fit failed: {_explain_fit_error(exc)}")
            return
        if run.dropped_predictors:
            ui.flash(
                "warning",
                "Left out of the design because they are constant or all-null on "
                "the training rows: "
                + ", ".join(f"**{v}**" for v in run.dropped_predictors)
                + f". {name} was fitted without them.",
            )
        st.rerun()
    if c2.button(
        "Make champion", disabled=p.champion == name, key=S.widget_key(f"champ_{name}")
    ):
        p.champion = name
        S.touch()
        st.rerun()
    if problems:
        c3.error("; ".join(problems))
    elif run is not None:
        n_adj = len(p.models[name].adjustments)
        suffix = f" · metrics include {n_adj} manual adjustment(s)" if n_adj else ""
        c3.success(f"Fitted and up to date · {run.created_at}{suffix}")
    elif stale is not None:
        c3.warning(
            "Spec changed since the last fit — results below are from the previous fit. Refit to update."
        )
        run = stale
    else:
        c3.info("Not fitted yet.")
    if run is None:
        return
    if run.dropped_predictors:
        st.warning(
            "Not in this fit — constant or all-null on the training rows: "
            + ", ".join(f"**{v}**" for v in run.dropped_predictors)
            + ". Remove them from the predictor list, or check the data."
        )

    s = run.summary()
    two_stage = s["alpha_stage2"] is not None
    ui.metric_row(
        [
            (
                "alpha (mains)" if two_stage else "alpha",
                ui.fmt(s["alpha"], digits=5),
                "penalty strength used to fit the main effects",
            ),
            *(
                [
                    (
                        "alpha (cells)",
                        ui.fmt(s["alpha_stage2"], digits=5),
                        "penalty strength of the second stage, which fits the "
                        "interaction cells on top of the frozen mains",
                    )
                ]
                if two_stage
                else []
            ),
            ("non-zero / features", f"{s['non_zero']} / {s['features']}", None),
            ("train A/E", ui.fmt(s["train_ae"]), None),
            ("holdout A/E", ui.fmt(s["holdout_ae"]), None),
            (
                "holdout Gini",
                ui.fmt(s["holdout_gini"]),
                "normalised, exposure-weighted",
            ),
            (
                "holdout dev. explained",
                ui.fmt(s["holdout_dev_explained"], pct=True),
                "1 − deviance / null deviance",
            ),
        ]
    )
    if two_stage:
        stage2_model = run.fit.stage2.model
        stage2_cv = getattr(stage2_model, "cv", None)
        stage2_n_alphas = getattr(stage2_model, "n_alphas", None)
        selection = (
            f" Stage 2 independently selected that alpha by {stage2_cv}-fold CV "
            f"over {stage2_n_alphas} penalties on the interaction cells."
            if stage2_cv is not None
            else (
                " Stage 2 honoured a preserved legacy cell-alpha override."
                if any(it.alpha is not None for it in run.config.interactions)
                else " Stage 2 used the model's fixed alpha."
            )
        )
        st.info(
            f"**Fitted in two stages.** Stage 1 fitted the {len(run.spec.main_effects)} "
            f"main effects at alpha {ui.fmt(s['alpha'], digits=5)} — exactly the fit "
            "this model would get with no interaction — and those tables and the base "
            "rate are now frozen. Stage 2 fitted "
            + ", ".join(f"**{e.variable}**" for e in run.spec.interactions)
            + f" on top of them at alpha {ui.fmt(s['alpha_stage2'], digits=5)}: "
            f"{s['cells_kept']} cell(s) had enough exposure to be rated on their own. "
            "Each is an adjustment to the frozen mains (1.00 = none), including any "
            "small overall re-levelling stage 2 wants — with the mains fixed it has "
            "nowhere to put that but in the cells." + selection
        )
        if stage2_cv is not None and not (run.fit.stage2.coef != 0).any():
            st.info(
                "Stage 2's separate CV found no cell adjustment that improved its "
                "validation score, so every interaction cell is 1.00 and this model "
                "makes the same predictions as its frozen mains."
            )
    elif run.config.interactions and not run.cells_kept:
        st.info(
            "**No second stage.** No cell of "
            + ", ".join(f"**{it.name}**" for it in run.config.interactions)
            + " reached its exposure floor ("
            + ", ".join(f"{it.min_cell_exposure:.2%}" for it in run.config.interactions)
            + " of the pair's training exposure), so there was nothing to fit on top "
            "of the main effects: every cell of the matrix on the Rate tables page "
            "reads 1.00. Lower the floor on the Design page, or use coarser bands for "
            "the parents, if you want the pair rated."
        )
    tab1, tab2, tab3 = st.tabs(
        ["Coefficients kept", "Regularisation path", "All coefficients"]
    )
    with tab1:
        ui.polars_table(run.fit.coef_table(drop_zero=True))
    with tab2:
        path = alpha_path(run.fit)
        for stage in path["stage"].unique().sort().to_list():
            sub = path.filter(pl.col("stage") == stage)
            if two_stage:
                st.caption(
                    f"**Stage {stage}** — "
                    + ("main effects" if stage == 1 else "interaction cells")
                )
            if sub.height > 1:
                st.plotly_chart(
                    C.alpha_path_chart(sub),
                    width="stretch",
                    key=S.widget_key(f"alpha_path_{name}_{stage}"),
                )
            else:
                st.caption(
                    "Fixed alpha — switch the penalty to cross-validated to see "
                    "the path."
                )
            ui.polars_table(sub)
    with tab3:
        ui.polars_table(run.fit.coef_table())
    st.caption(
        f"Train rows {run.train_rows:,} · holdout rows {run.holdout_rows:,} · "
        f"design {run.spec!r}"
    )


def render() -> None:
    st.title("Model design and fit")
    ui.status_bar()
    name = D.render_contents()
    if name is None:
        return
    st.header("Fit and results")
    if S.get_run(name) is None:
        st.info(
            f"**{name}** is ready to fit. Review or adjust the settings below, then "
            "use the blue **Fit model** button to run it. Fitting is never automatic."
        )
    _fit_options(name)
    _fit_and_results(name)
