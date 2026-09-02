"""Page 6 — Model: specify, fit, inspect."""

from __future__ import annotations

import streamlit as st

from easy_glm.workflow import alpha_path
from easy_glm.workflow.project import FAMILIES

from . import charts as C
from . import state as S
from . import ui


def _model_picker() -> str | None:
    p = S.project()
    names = list(p.models)
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    current = st.session_state.get("model_current")
    if current not in names:
        current = p.champion if p.champion in names else (names[0] if names else None)
    sel = c1.selectbox(
        "Model",
        names or ["(none)"],
        index=names.index(current) if current in names else 0,
        key="model_select",
    )
    new_name = c2.text_input(
        "New model name", key="model_new_name", placeholder="freq_v2"
    )
    if c3.button("Create", disabled=not new_name or new_name in names):
        p.new_model(new_name)
        if len(p.models) == 1:
            p.champion = new_name
        st.session_state.model_current = new_name
        S.touch()
        st.rerun()
    if names and c4.button("Delete", disabled=len(names) == 0):
        p.models.pop(sel, None)
        st.session_state.runs.pop(sel, None)
        if p.champion == sel:
            p.champion = next(iter(p.models), None)
        S.touch()
        st.rerun()
    if not names:
        st.info(
            "Create a model to start. It is pre-filled from the roles on the Variables page."
        )
        return None
    st.session_state.model_current = sel
    return sel


def _config(name: str) -> None:
    p = S.project()
    cfg = p.models[name]
    df = S.prepared_frame()
    cols = list(df.columns) if df is not None else []
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        family = c1.selectbox(
            "Family",
            list(FAMILIES),
            index=list(FAMILIES).index(cfg.family) if cfg.family in FAMILIES else 0,
            key=f"fam_{name}",
        )
        target = (
            c2.selectbox(
                "Target",
                cols,
                index=cols.index(cfg.target) if cfg.target in cols else 0,
                key=f"tgt_{name}",
            )
            if cols
            else cfg.target
        )
        wopts = ["(none)"] + cols
        weight = c3.selectbox(
            "Weight",
            wopts,
            index=wopts.index(cfg.weight) if cfg.weight in wopts else 0,
            key=f"wgt_{name}",
        )
        offset = c4.selectbox(
            "Offset (linear scale)",
            wopts,
            index=wopts.index(cfg.offset) if cfg.offset in wopts else 0,
            key=f"off_{name}",
        )
        weight = None if weight == "(none)" else weight
        offset = None if offset == "(none)" else offset
        divide = st.checkbox(
            "Divide target by weight (model a rate, e.g. claims / exposure)",
            cfg.divide_target_by_weight,
            key=f"div_{name}",
            disabled=weight is None,
        )
        preds = st.multiselect(
            "Predictors",
            p.predictors,
            default=[v for v in cfg.predictors if v in p.predictors],
            key=f"preds_{name}",
        )
        st.markdown("**Penalty**")
        c1, c2, c3, c4, c5 = st.columns(5)
        mode = c1.radio(
            "alpha",
            ["cross-validated", "fixed"],
            index=0 if cfg.penalty.alpha is None else 1,
            key=f"pmode_{name}",
            horizontal=True,
        )
        alpha = c2.number_input(
            "alpha",
            0.0,
            10.0,
            float(cfg.penalty.alpha or 0.001),
            format="%.5f",
            key=f"alpha_{name}",
            disabled=mode != "fixed",
        )
        cv = c3.number_input(
            "CV folds",
            2,
            10,
            int(cfg.penalty.cv or 5),
            key=f"cv_{name}",
            disabled=mode != "cross-validated",
        )
        n_alphas = c4.number_input(
            "alphas on path",
            3,
            100,
            int(cfg.penalty.n_alphas),
            key=f"nalpha_{name}",
            disabled=mode != "cross-validated",
        )
        l1 = c5.slider(
            "l1_ratio (1 = lasso)",
            0.0,
            1.0,
            float(cfg.penalty.l1_ratio),
            0.05,
            key=f"l1_{name}",
        )
        c1, c2, c3 = st.columns(3)
        base = c1.radio(
            "Base risk for tables",
            ["modal", "reference"],
            index=0 if cfg.base == "modal" else 1,
            horizontal=True,
            key=f"base_{name}",
        )
        bro = c2.number_input(
            "Base rate override (0 = exact)",
            0.0,
            value=float(cfg.base_rate_override or 0.0),
            format="%.6f",
            key=f"bro_{name}",
        )
        notes = c3.text_input("Notes", cfg.notes, key=f"notes_{name}")
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


def _fit_and_results(name: str) -> None:
    p = S.project()
    problems = p.validate(name)
    run = S.get_run(name)
    stale = S.stale_run(name) if run is None else None
    c1, c2, c3 = st.columns([1, 1, 3])
    if c1.button(
        "Fit model", type="primary", disabled=bool(problems), key=f"fit_{name}"
    ):
        try:
            run = S.fit_model(name)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Fit failed: {exc}")
            return
        st.rerun()
    if c2.button("Make champion", disabled=p.champion == name, key=f"champ_{name}"):
        p.champion = name
        S.touch()
        st.rerun()
    if problems:
        c3.error("; ".join(problems))
    elif run is not None:
        c3.success(f"Fitted and up to date · {run.created_at}")
    elif stale is not None:
        c3.warning(
            "Spec changed since the last fit — results below are from the previous fit. Refit to update."
        )
        run = stale
    else:
        c3.info("Not fitted yet.")
    if run is None:
        return

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
