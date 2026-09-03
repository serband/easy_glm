"""Page 8 — Compare: champion against challenger, side by side.

Two fitted models, the same rows: the headline metrics next to each other, the
same A/E and lift charts with both models' expected lines, the double lift, and
the table of *which relativities actually differ*. Nothing is computed here —
every number comes from :mod:`easy_glm.workflow`.
"""

from __future__ import annotations

import polars as pl
import streamlit as st

from easy_glm.workflow import (
    ModelRun,
    ae_by_variable,
    describe_diff,
    double_lift,
    gini,
    lift_table,
    relativity_diff,
    totals,
)

from . import charts as C
from . import state as S
from . import ui

#: metric rows of the side-by-side table: ``(label, key, format)``
_ROWS = [
    ("rows", "rows", "{:,.0f}"),
    ("exposure", "exposure", "{:,.0f}"),
    ("actual", "actual", "{:,.1f}"),
    ("expected", "expected", "{:,.1f}"),
    ("A/E", "ae", "{:.4f}"),
    ("Gini (normalised)", "gini", "{:.4f}"),
    ("deviance explained", "deviance_explained", "{:.4f}"),
    ("mean deviance", "mean_deviance", "{:.5f}"),
]


def _fmt(value, spec: str) -> str:
    try:
        return spec.format(float(value))
    except (TypeError, ValueError):
        return "—"


#: the two label columns of the side-by-side table; a model called one of these
#: gets a suffix so its column is still its own
_LABEL_COLUMNS = ("rows used", "metric")


def _column_for(name: str) -> str:
    return f"{name} (model)" if name in _LABEL_COLUMNS else name


def _metrics_table(a: ModelRun, b: ModelRun) -> pl.DataFrame:
    """Metric per row, one column per model and subset, plus the model facts."""
    col_a, col_b = _column_for(a.name), _column_for(b.name)
    rows: list[dict[str, str]] = []
    for subset in ("train", "holdout"):
        for label, key, spec in _ROWS:
            rows.append(
                {
                    "rows used": subset,
                    "metric": label,
                    col_a: _fmt(a.metrics.get(subset, {}).get(key), spec),
                    col_b: _fmt(b.metrics.get(subset, {}).get(key), spec),
                }
            )

    def _facts(run: ModelRun) -> dict[str, str]:
        return {
            "alpha": f"{run.alpha:.6f}",
            "non-zero terms": f"{int((run.fit.coef != 0).sum()):,} of "
            f"{len(run.fit.coef):,}",
            "interactions": ", ".join(f"{i.a} × {i.b}" for i in run.config.interactions)
            or "none",
            "linear terms": ", ".join(
                v for v, c in run.rate_model.variables.items() if c.type == "linear"
            )
            or "none",
            "manual adjustments": str(len(run.config.adjustments)),
            "base rate": f"{run.rate_model.base_rate:.6f}",
        }

    fa, fb = _facts(a), _facts(b)
    for key in fa:
        rows.append(
            {"rows used": "model", "metric": key, col_a: fa[key], col_b: fb[key]}
        )
    return pl.DataFrame(rows)


def _subset(df: pl.DataFrame, which: str) -> pl.DataFrame:
    col = S.project().data.split.column
    if which == "train":
        return df.filter(pl.col(col) == 1)
    if which == "holdout":
        return df.filter(pl.col(col) == 0)
    return df


def _make_champion(name: str) -> None:
    p = S.project()
    p.champion = name
    S.touch()
    ui.flash("success", f"**{name}** is now the project champion.")
    st.rerun()


def render() -> None:
    st.title("Compare")
    ui.status_bar()
    p = S.project()
    df = ui.require_data()
    if df is None:
        return
    fitted = S.fitted_models()
    if len(fitted) < 2:
        st.info(
            "Compare needs **two fitted models**. Create a second model on the "
            "Model page (or clone the first and change something), fit it, and "
            "come back."
        )
        return

    default_a = p.champion if p.champion in fitted else fitted[0]
    others = [n for n in fitted if n != default_a]
    sidebar = S.challenger()
    default_b = sidebar if sidebar in others else S.latest_run(others) or others[0]

    c1, c2 = st.columns(2)
    with c1:
        name_a = st.selectbox(
            "Champion",
            fitted,
            index=fitted.index(default_a),
            key=S.widget_key("cmp_a"),
            help="The model you are comparing against — the project's champion "
            "unless you pick another one.",
        )
    rest = [n for n in fitted if n != name_a]
    if not rest:
        st.info("Pick two different models.")
        return
    with c2:
        name_b = st.selectbox(
            "Challenger",
            rest,
            index=rest.index(default_b) if default_b in rest else 0,
            key=S.widget_key(f"cmp_b_{name_a}_{sidebar}"),
            help="Defaults to the sidebar's *Compare with*, else the most "
            "recently fitted other model.",
        )
    run_a, run_b = S.get_run(name_a), S.get_run(name_b)
    if run_a is None or run_b is None:  # pragma: no cover - refitted meanwhile
        st.warning("One of the models is no longer fitted; refit it on the Model page.")
        return
    for run in (run_a, run_b):
        missing = [c for c in run.spec.required_columns if c not in df.columns]
        if missing:
            st.error(
                f"{run.name} cannot be scored: the prepared data no longer has "
                + ", ".join(missing)
                + ". Refit it on the Model page after fixing the Variables page."
            )
            return

    b1, b2, _b3 = st.columns([1, 1, 3])
    if b1.button(
        f"Make {name_a} champion",
        disabled=p.champion == name_a,
        key=S.widget_key("cmp_champ_a"),
    ):
        _make_champion(name_a)
    if b2.button(
        f"Make {name_b} champion",
        disabled=p.champion == name_b,
        key=S.widget_key("cmp_champ_b"),
        type="primary" if p.champion != name_b else "secondary",
    ):
        _make_champion(name_b)
    st.caption(f"Project champion: **{p.champion or '—'}**")

    st.subheader("Metrics side by side")
    st.caption(
        "The holdout rows are the ones to trust — neither model saw them. A/E "
        "near 1.00 means the model charges what happened in total; a higher Gini "
        "means the model orders the risks better; deviance explained is the share "
        "of the null deviance it removes."
    )
    ui.polars_table(_metrics_table(run_a, run_b))
    _snapshot_metrics(run_a, run_b)

    which = st.radio(
        "Rows",
        ["holdout", "train", "all"],
        horizontal=True,
        key=S.widget_key("cmp_subset"),
    )
    frame = _subset(df, which)
    if frame.is_empty():
        st.warning("No rows in this subset.")
        return
    actual, expected_a, w = totals(frame, run_a.config, run_a.predict(frame))
    expected_b = totals(frame, run_b.config, run_b.predict(frame))[1]

    tabs = st.tabs(
        ["A/E by variable", "Lift", "Double lift", "Relativities that differ"]
    )

    with tabs[0]:
        mains = list(run_a.spec.main_effects)
        reserved = {
            run_a.config.target,
            run_a.config.weight,
            run_a.config.offset,
            p.data.split.column,
        } - {None}
        others_vars = [c for c in frame.columns if c not in reserved and c not in mains]
        options = mains + others_vars
        c1, c2 = st.columns([3, 1])
        var = c1.selectbox(
            "Variable",
            options,
            format_func=lambda v: v if v in mains else f"{v} (not in model)",
            key=S.widget_key("cmp_var"),
        )
        n_bins = c2.slider(
            "Bands (numeric, not in model)", 5, 50, 20, key=S.widget_key("cmp_bins")
        )
        enc = run_a.spec[var] if var in run_a.spec.encoders else None
        knots = enc.band_edges() if hasattr(enc, "band_edges") else None
        tbl = ae_by_variable(
            frame, var, actual, expected_a, w, n_bins=n_bins, knots=knots
        )
        cmp_tbl = ae_by_variable(
            frame, var, actual, expected_b, w, n_bins=n_bins, knots=knots
        )
        st.plotly_chart(
            C.ae_chart(
                tbl,
                title=f"{var} — actual vs both models' expected ({which})",
                compare=cmp_tbl,
                compare_name=name_b,
            ),
            width="stretch",
        )
        st.caption(
            f"Blue is what happened, orange is **{name_a}**, green dashed is "
            f"**{name_b}**. Where the two model lines part with real exposure "
            "behind them, they disagree about that band."
        )
        with st.expander("Tables"):
            ui.polars_table(tbl)
            ui.polars_table(cmp_tbl)

    with tabs[1]:
        n = st.slider("Bins", 5, 20, 10, key=S.widget_key("cmp_lift_bins"))
        g_a = gini(actual, expected_a, w)
        g_b = gini(actual, expected_b, w)
        st.caption(
            f"Normalised Gini ({which}) — **{name_a}: {g_a:.4f}** · "
            f"**{name_b}: {g_b:.4f}**"
        )
        for name, exp in ((name_a, expected_a), (name_b, expected_b)):
            st.plotly_chart(
                C.lift_chart(
                    lift_table(actual, exp, w, n_bins=n),
                    title=f"Lift — {name} ({which})",
                ),
                width="stretch",
            )

    with tabs[2]:
        st.caption(
            "The book is sorted by how much cheaper the champion is than the "
            "challenger; the model whose A/E stays closer to 1.00 across the bins "
            "is the one getting those policies right."
        )
        dl = double_lift(actual, expected_a, expected_b, w, n_bins=10)
        st.plotly_chart(
            C.double_lift_chart(dl, name_a=name_a, name_b=name_b), width="stretch"
        )
        with st.expander("Table"):
            ui.polars_table(dl)

    with tabs[3]:
        tol = st.number_input(
            "Report a band when |Δ log relativity| exceeds",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.005,
            format="%.3f",
            key=S.widget_key("cmp_tol"),
            help="0.01 is a 1 % change in the relativity. Raise it to see only "
            "the differences that would move a premium.",
        )
        diff = relativity_diff(run_a, run_b, tol)
        st.caption(
            f"**{diff.height}** row(s). `log_diff` is log({name_b} / {name_a}): "
            f"+0.10 means {name_b} charges about 10 % more for that band. Bands "
            "are matched by their rate-table label, so a model whose knots or "
            "levels moved shows *band only in …* rows rather than false changes; "
            "a variable only one model has is listed once."
        )
        if diff.is_empty():
            st.success(
                "Every relativity the two models share is within the tolerance — "
                "they would charge the same premium."
            )
        else:
            shown = describe_diff(diff, name_a, name_b)
            ui.polars_table(shown)
            st.download_button(
                "Download the differences (.csv)",
                ui.frame_bytes(shown),
                file_name=f"{ui.safe_filename(name_a)}_vs_"
                f"{ui.safe_filename(name_b)}_relativity_diff.csv",
                key=S.widget_key("cmp_dl_diff"),
            )


def _snapshot_metrics(*runs: ModelRun) -> None:
    """Metrics recorded with each saved version of a model's rate tables."""
    rows = [
        {
            "model": run.name,
            "version": snap.version,
            "description": snap.description,
            "rows": subset,
            "A/E": m.get("ae"),
            "Gini": m.get("gini"),
        }
        for run in runs
        for snap in run.rate_model.snapshots
        if snap.metrics
        for subset, m in snap.metrics.items()
        if isinstance(m, dict)
    ]
    if not rows:
        return
    with st.expander("Saved versions of the rate tables"):
        st.caption(
            "A version is written when manual adjustments are applied; the "
            "metrics stored with it are the ones that version scored."
        )
        ui.polars_table(pl.DataFrame(rows))
