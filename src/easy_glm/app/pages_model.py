"""Page 6 — Model: specify, fit, inspect."""

from __future__ import annotations

import streamlit as st

from easy_glm.core.design import NUMERIC_DTYPES
from easy_glm.workflow import alpha_path
from easy_glm.workflow.project import FAMILIES, validate_model_name

from . import charts as C
from . import state as S
from . import ui


def _model_picker() -> str | None:
    p = S.project()
    names = list(p.models)
    # Create / Delete on the previous run asked for another model to be shown.
    # The selectbox keeps whatever its key holds, and Streamlit refuses to
    # touch a widget's key once the widget exists, so the keys are dropped here
    # — before the picker is drawn — and the boxes fall back to their defaults
    # (``model_current`` below). Without this, Create says "created" and the
    # page underneath goes on editing and fitting the model that was selected.
    if st.session_state.pop("model_pending", False):
        st.session_state.pop(S.widget_key("model_select"), None)
        st.session_state.pop(S.widget_key("model_new_name"), None)
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    current = st.session_state.get("model_current")
    if current not in names:
        current = p.champion if p.champion in names else (names[0] if names else None)
    sel = c1.selectbox(
        "Model",
        names or ["(none)"],
        index=names.index(current) if current in names else 0,
        key=S.widget_key("model_select"),
    )
    new_name = c2.text_input(
        "New model name", key=S.widget_key("model_new_name"), placeholder="freq_v2"
    ).strip()
    name_problem = validate_model_name(new_name, names) if new_name else None
    if c3.button("Create", disabled=not new_name or bool(name_problem)):
        p.new_model(new_name)
        if len(p.models) == 1:
            p.champion = new_name
        st.session_state.model_current = new_name
        st.session_state["model_pending"] = True  # show what was created
        S.touch()
        ui.flash("success", f"Model {new_name!r} created and selected")
        st.rerun()
    if name_problem:
        c2.caption(f"⚠ {name_problem}")
    if names and c4.button("Delete", disabled=len(names) == 0):
        p.models.pop(sel, None)
        kept = S.remove_model_runs(sel)
        if p.champion == sel:
            p.champion = next(iter(p.models), None)
        st.session_state.model_current = next(iter(p.models), None)
        st.session_state["model_pending"] = True
        # queued *before* the save: touch() reruns the moment it finds the file
        # changed on disk, and this sentence — "nothing was written" — is
        # exactly the one the user needs in that case
        ui.flash("warning" if kept else "info", kept or f"Model {sel!r} deleted")
        S.touch()
        st.rerun()
    if not names:
        st.info(
            "Create a model to start. It is pre-filled from the roles on the Variables page."
        )
        return None
    st.session_state.model_current = sel
    return sel


def _column_pick(
    col, label: str, current: str | None, options: list[str], *, key: str, none: bool
) -> str | None:
    """A column selector that never silently re-points a model: when the
    stored column is not among the options the box shows a placeholder and an
    error, and the stored value is kept until the user picks one."""
    opts = (["(none)"] if none else []) + options
    if current is None and none:
        index: int | None = 0
    elif current in opts:
        index = opts.index(current)
    else:
        index = None
    picked = col.selectbox(
        label, opts, index=index, key=key, placeholder="choose a numeric column"
    )
    if index is None:
        col.error(f"{label}: {current!r} is not a numeric column of the data")
        return current
    return None if picked == "(none)" else picked


def _config(name: str) -> None:
    p = S.project()
    cfg = p.models[name]
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
                    f"**{it.name}** (min cell {it.min_cell_exposure:.2%})"
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
        alpha = ui.number_in_range(
            c2,
            "alpha",
            value=float(cfg.penalty.alpha or 0.001),
            lo=0.0,
            hi=10.0,
            what="alpha",
            format="%.5f",
            key=S.widget_key(f"alpha_{name}"),
            disabled=mode != "fixed",
        )
        cv = c3.number_input(
            "CV folds",
            2,
            10,
            int(cfg.penalty.cv or 5),
            key=S.widget_key(f"cv_{name}"),
            disabled=mode != "cross-validated",
        )
        n_alphas = c4.number_input(
            "alphas on path",
            3,
            100,
            int(cfg.penalty.n_alphas),
            key=S.widget_key(f"nalpha_{name}"),
            disabled=mode != "cross-validated",
        )
        l1 = c5.slider(
            "l1_ratio (1 = lasso)",
            0.0,
            1.0,
            float(cfg.penalty.l1_ratio),
            0.05,
            key=S.widget_key(f"l1_{name}"),
        )
        c1, c2, c3 = st.columns(3)
        base = c1.radio(
            "Base risk for tables",
            ["modal", "reference"],
            index=0 if cfg.base == "modal" else 1,
            horizontal=True,
            key=S.widget_key(f"base_{name}"),
        )
        bro = ui.number_in_range(
            c2,
            "Base rate override (0 = exact)",
            value=float(cfg.base_rate_override or 0.0),
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
        notes = c3.text_input("Notes", cfg.notes, key=S.widget_key(f"notes_{name}"))
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
    ui.metric_row(
        [
            ("alpha", ui.fmt(s["alpha"], digits=5), "penalty strength used"),
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
    tab1, tab2, tab3 = st.tabs(
        ["Coefficients kept", "Regularisation path", "All coefficients"]
    )
    with tab1:
        ui.polars_table(run.fit.coef_table(drop_zero=True))
    with tab2:
        path = alpha_path(run.fit)
        if path.height > 1:
            st.plotly_chart(C.alpha_path_chart(path), width="stretch")
        else:
            st.caption(
                "Fixed alpha — switch the penalty to cross-validated to see the path."
            )
        ui.polars_table(path)
    with tab3:
        ui.polars_table(run.fit.coef_table())
    st.caption(
        f"Train rows {run.train_rows:,} · holdout rows {run.holdout_rows:,} · "
        f"design {run.spec!r}"
    )


def render() -> None:
    st.title("Model")
    ui.status_bar()
    if ui.require_data() is None or ui.require_target() is None:
        return
    name = _model_picker()
    if name is None:
        return
    _config(name)
    _fit_and_results(name)
