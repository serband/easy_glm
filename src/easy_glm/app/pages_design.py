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
from easy_glm.workflow.project import FAMILIES, validate_model_name

from . import charts as C
from . import state as S
from . import ui

KNOT_OPTIONS = ["quantile", "integer", "custom"]
KIND_OPTIONS = ["auto", "step", "linear", "continuous", "categorical"]
MONO_OPTIONS = ["none", "increasing", "decreasing"]
#: one line per kind, shown under the selector and in its help
KIND_HELP = {
    "auto": "auto — step for numbers, categorical for text (the default).",
    "step": "step — bands, each with its own relativity; the curve jumps at the "
    "band edges.",
    "linear": "linear — a continuous curve, straight between knots; flat unless "
    "the data insists on a slope.",
    "continuous": "continuous — one straight line over the whole range (no "
    "knots): a single slope on the value itself.",
    "categorical": "categorical — every value is its own level, plus an Other "
    "bucket.",
}
KIND_TOOLTIP = "\n\n".join(KIND_HELP.values())
#: kinds that build a LinearEncoder
LINEAR_KINDS = ("linear", "continuous")


# --------------------------------------------------------------------------
# model definition
# --------------------------------------------------------------------------
def _model_picker() -> str | None:
    """Create, select or delete the model whose design is being edited."""
    p = S.project()
    names = list(p.models)
    if st.session_state.pop("model_pending", False):
        st.session_state.pop(S.widget_key("model_select"), None)
        st.session_state.pop(S.widget_key("model_new_name"), None)
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    current = st.session_state.get("model_current")
    if current not in names:
        current = p.champion if p.champion in names else (names[0] if names else None)
    selected = c1.selectbox(
        "Model",
        names or ["(none)"],
        index=names.index(current) if current in names else 0,
        key=S.widget_key("model_select"),
    )
    new_name = c2.text_input(
        "New model name (required to enable Create)",
        key=S.widget_key("model_new_name"),
        placeholder="e.g. frequency",
    ).strip()
    name_problem = validate_model_name(new_name, names) if new_name else None

    def create_model() -> None:
        """Button callbacks run before the next script render.

        Updating the Model selectbox here matters: setting it after that
        selectbox has already been instantiated only changes the explanatory
        message, leaving the old model selected.  That made a newly created
        ``v2`` appear selected while **Fit model** silently fitted frequency.
        """
        candidate = str(
            st.session_state.get(S.widget_key("model_new_name"), "")
        ).strip()
        problem = validate_model_name(candidate, list(p.models)) if candidate else None
        if problem:
            return
        p.new_model(candidate)
        st.session_state.model_current = candidate
        st.session_state[S.widget_key("model_select")] = candidate
        st.session_state[S.widget_key("model_new_name")] = ""
        S.touch()
        ui.flash("success", f"Model {candidate!r} created and selected")

    c3.button(
        "Create",
        disabled=not new_name or bool(name_problem),
        help="Enter a valid new model name to enable Create.",
        on_click=create_model,
    )
    if name_problem:
        c2.caption(f"⚠ {name_problem}")
    if names and c4.button("Delete"):
        p.models.pop(selected, None)
        kept = S.remove_model_runs(selected)
        if p.champion == selected:
            p.champion = next(iter(p.models), None)
        st.session_state.model_current = next(iter(p.models), None)
        st.session_state["model_pending"] = True
        ui.flash("warning" if kept else "info", kept or f"Model {selected!r} deleted")
        S.touch()
        st.rerun()
    if not names:
        st.info(
            "Define a model to start: enter a valid model name above to enable "
            "Create. It is pre-filled from the roles on the Variables page."
        )
        return None
    st.session_state.model_current = selected
    return selected


def _column_pick(
    column,
    label: str,
    current: str | None,
    options: list[str],
    *,
    key: str,
    none: bool,
    help: str | None = None,
) -> str | None:
    """Select a numeric model column without silently changing an invalid one."""
    choices = (["(none)"] if none else []) + options
    if current is None and none:
        index: int | None = 0
    elif current in choices:
        index = choices.index(current)
    else:
        index = None
    picked = column.selectbox(
        label,
        choices,
        index=index,
        key=key,
        placeholder="choose a numeric column",
        help=help,
    )
    if index is None:
        column.error(f"{label}: {current!r} is not a numeric column of the data")
        return current
    return None if picked == "(none)" else picked


def _model_definition(name: str, df: pl.DataFrame) -> None:
    """Editable family, scale and predictors — the definition, not the fit."""
    p = S.project()
    cfg = p.models[name]
    numeric_cols = [
        column for column, dtype in df.schema.items() if dtype in NUMERIC_DTYPES
    ]
    with st.container(border=True):
        st.subheader("Model definition")
        c1, c2, c3, c4 = st.columns(4)
        family = c1.selectbox(
            "Family",
            list(FAMILIES),
            index=list(FAMILIES).index(cfg.family) if cfg.family in FAMILIES else 0,
            key=S.widget_key(f"fam_{name}"),
            help="The distribution for the outcome: Poisson for claim counts, Gamma for positive severity.",
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
                )
            )
        if family == "binomial":
            c1.caption(
                "Binomial models probabilities; tables are odds relativities and "
                "predictions are never multiplied by exposure."
            )
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
            help="A known adjustment already on the log scale, such as log(current premium).",
        )
        divide_key = S.widget_key(f"div_{name}")
        if weight is None and st.session_state.get(divide_key):
            st.session_state.pop(divide_key, None)
        divide = st.checkbox(
            "Divide target by weight (model a rate, e.g. claims / exposure)",
            cfg.divide_target_by_weight and weight is not None,
            key=divide_key,
            disabled=weight is None,
            help="For a count with exposure, fit claims divided by exposure so predictions are rates.",
        )
        missing = [
            predictor for predictor in cfg.predictors if predictor not in p.predictors
        ]
        if missing:
            st.error(
                "Predictor(s) no longer available (role changed or column gone): "
                + ", ".join(missing)
                + " — the model keeps them until you change the list below."
            )
        predictors = st.multiselect(
            "Predictors",
            sorted(set(p.predictors) | set(cfg.predictors)),
            default=list(cfg.predictors),
            format_func=lambda value: (
                value if value in p.predictors else f"{value} (missing)"
            ),
            key=S.widget_key(f"preds_{name}"),
            help="Rating factors included in this model. Their shared design is edited below.",
        )
        if cfg.interactions:
            missing_parents = sorted(
                {
                    parent
                    for interaction in cfg.interactions
                    for parent in (interaction.a, interaction.b)
                    if parent not in predictors
                }
            )
            st.info(
                "Interactions in this design: "
                + ", ".join(
                    f"**{interaction.name}**" for interaction in cfg.interactions
                )
                + ". Edit their cells and penalties in the Interactions section below."
            )
            if missing_parents:
                st.warning(
                    "Interaction parent(s) no longer among the predictors: "
                    + ", ".join(missing_parents)
                    + ". Put them back before fitting."
                )

    changed = False
    values = {
        "family": family,
        "tweedie_power": power,
        "target": target,
        "weight": weight,
        "offset": offset,
        "divide_target_by_weight": bool(divide) and weight is not None,
        "predictors": list(predictors),
    }
    for field, value in values.items():
        if getattr(cfg, field) != value:
            setattr(cfg, field, value)
            changed = True
    if changed:
        S.touch()
        st.rerun()


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
        n_bins = c1.number_input(
            "Quantile knots (n_bins)",
            2,
            200,
            int(d.n_bins),
            help="How many equal-exposure bands to propose for numeric predictors using quantile knots.",
        )
        share = c2.number_input(
            "Min level share",
            0.0,
            0.5,
            float(d.min_level_share),
            0.0005,
            format="%.4f",
            help="Categorical levels below this share of training exposure are grouped into Other.",
        )
        null_ind = c3.checkbox(
            "Null indicator column",
            bool(d.null_indicator),
            help="Lets the model give missing values their own effect instead of treating them as the reference.",
        )
        max_int = c4.number_input(
            "Max integer knots",
            10,
            1000,
            int(d.max_integer_knots),
            help="Above this many distinct integers, use quantile rather than one knot per integer.",
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
        displayed_bins = (
            len(vd.knots) + 1 if isinstance(vd.knots, list) else (vd.n_bins or 0)
        )
        rows.append(
            {
                "variable": v,
                "dtype": str(train[v].dtype) if v in train.columns else "?",
                "kind": vd.kind or "auto",
                "knots": "custom" if isinstance(vd.knots, list) else vd.knots,
                "n_bins": displayed_bins,
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
                "penalty": float(vd.penalty_weight),
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
                "kind", options=KIND_OPTIONS, required=True, help=KIND_TOOLTIP
            ),
            "knots": st.column_config.SelectboxColumn(
                "knots",
                options=KNOT_OPTIONS,
                required=True,
                help="quantile: n_bins quantiles · integer: every integer · custom: edit below",
            ),
            "n_bins": st.column_config.NumberColumn(
                "n_bins (0 = default)",
                min_value=0,
                max_value=200,
                step=1,
                help=(
                    "Number of quantile bins to request for this predictor; 0 uses "
                    "the default above. Repeated cut points can produce fewer bins. "
                    "For custom knots, this shows the exact number of resulting bins."
                ),
            ),
            "null col": st.column_config.CheckboxColumn(
                "null col",
                help="Give null values their own fitted effect.",
            ),
            "min share": st.column_config.NumberColumn(
                "min level share",
                min_value=0.0,
                max_value=0.5,
                step=0.0005,
                format="%.4f",
                help="Categorical levels below this training-exposure share become Other.",
            ),
            "monotone": st.column_config.SelectboxColumn(
                "monotone",
                options=MONO_OPTIONS,
                required=True,
                help="Require a numeric factor to only rise or only fall; it constrains band changes, not their size.",
            ),
            "penalty": st.column_config.NumberColumn(
                "penalty weight",
                min_value=0.0,
                max_value=100.0,
                step=0.25,
                format="%.2f",
                help="How hard the lasso shrinks this factor relative to the "
                "rest of the design. 1 = like everything else · 2 = twice as "
                "hard · 0 = not penalised at all, so every band or level is "
                "kept (use it for a factor you have decided to charge for, "
                "such as a territory table).",
            ),
        },
        key=S.widget_key("design_grid"),
    )
    changed = False
    for _, r in edited.iterrows():
        v = r["variable"]
        vd = p.design.variables.get(v, VariableDesign())
        requested_bins = int(r["n_bins"])
        selected_knots = r["knots"]
        if selected_knots == "custom":
            previous_displayed_bins = (
                len(vd.knots) + 1 if isinstance(vd.knots, list) else (vd.n_bins or 0)
            )
            if requested_bins != previous_displayed_bins:
                # Editing n_bins is a request to calculate a fresh quantile
                # design. Otherwise an old custom list silently wins and the
                # number appears to do nothing.
                selected_knots = "quantile"
            elif isinstance(vd.knots, list):
                selected_knots = vd.knots
            else:
                selected_knots = []  # to be filled in the detail panel
        new = VariableDesign(
            kind=None if r["kind"] == "auto" else r["kind"],
            knots=selected_knots,
            n_bins=requested_bins or None,
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
            penalty_weight=max(0.0, float(r["penalty"])),
        )
        numeric = v in train.columns and train[v].dtype in NUMERIC_DTYPES
        if new.monotone and (not numeric or new.kind == "categorical"):
            ui.flash(
                "error",
                f"{v}: monotone constraints apply to numeric designs (step, linear "
                "or continuous) only; the constraint was not saved",
            )
            new.monotone = (
                vd.monotone if numeric and new.kind != "categorical" else None
            )
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
        help=KIND_TOOLTIP,
    )
    st.caption(KIND_HELP[kind])
    if kind != current:
        if kind in ("step", *LINEAR_KINDS) and not numeric:
            st.error(f"{var} is not numeric; a {kind} design needs numbers")
            return
        vd.kind = None if kind == "auto" else kind
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
    target: str | None,
    weight: str | None,
    divide,
) -> None:
    p = S.project()
    custom = isinstance(vd.knots, list)
    requested_bins = vd.n_bins or p.design.defaults.n_bins
    actual_bins = len(enc.knots) + 1
    knots_txt = st.text_area(
        (
            "Custom knots (comma-separated)"
            if custom
            else "Calculated knots (edit and apply to override)"
        ),
        ", ".join(f"{k:g}" for k in enc.knots),
        height=90,
        key=S.widget_key(f"knots_{var}"),
        help=(
            "Band edges. Each band receives its own relativity. The values only "
            "become custom after you press Apply knots."
        ),
    )
    if custom:
        st.info(
            f"Custom knots are active. These {len(enc.knots)} knots define "
            f"{actual_bins} bins and replace automatic quantile binning. The "
            f"n_bins value in the table therefore reports {actual_bins}. Change "
            "the knots setting in the table to quantile to calculate them again."
        )
    elif vd.knots == "integer":
        st.info(
            f"These {len(enc.knots)} knots were calculated from the observed integer "
            f"values and define {actual_bins} bins. Edit the list and press Apply "
            "knots to replace them with exact custom values."
        )
    else:
        st.info(
            f"These {len(enc.knots)} unique knots were calculated from the requested "
            f"{requested_bins} quantile bins, giving {actual_bins} actual bins. "
            "Repeated quantile cut points are removed, so the actual count can be "
            "lower. Edit the list and press Apply knots to use exact custom values "
            "instead."
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
                vd.n_bins = len(knots) + 1
                p.design.variables[var] = vd
                S.touch()
                ui.flash(
                    "success",
                    f"{var}: using {len(knots)} custom knots ({len(knots) + 1} bins) "
                    "instead of automatic quantile binning",
                )
                st.rerun()
    u = univariate(
        preview,
        var,
        target=target,
        weight=weight,
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
    target: str | None,
    weight: str | None,
    divide,
    *,
    continuous: bool = False,
) -> None:
    """Editor for a linear term. ``continuous=True`` is the same term with no
    interior knots (one straight line), so the knot controls are hidden."""
    p = S.project()
    d = p.design.defaults
    s = train[var].drop_nulls().cast(pl.Float64)
    tmin, tmax = float(s.min()), float(s.max())
    rlo, rhi = round_range_outward(tmin, tmax)
    st.markdown(
        (
            "**Continuous** — one straight line on the log scale over the whole "
            "range (no knots, so the slope never changes), and **flat outside the "
            "clamp range**. "
            if continuous
            else "**Piecewise-linear** — the relativity curve is continuous, "
            "log-linear inside each band, and **flat outside the clamp range**. "
        )
        + f"Training range {tmin:g} – {tmax:g}; default clamp = that range rounded "
        f"outward to a round number → **{rlo:g} – {rhi:g}**."
    )
    strategy_now = "custom" if isinstance(vd.knots, list) else vd.knots
    strategy, n_bins, knots_txt = strategy_now, vd.n_bins or d.n_bins, ""
    if not continuous:
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
            help="Number of equal-exposure bands when using quantile knots.",
        )
        knots_txt = c3.text_area(
            "Knots (custom)",
            ", ".join(f"{k:g}" for k in enc.knots),
            height=70,
            key=S.widget_key(f"lin_knots_{var}"),
            disabled=strategy != "custom",
            help="Interior band edges where the fitted slope may change.",
        )
    use_default = st.checkbox(
        "Clamp to the training range (rounded outward)",
        vd.clamp is None,
        key=S.widget_key(f"lin_defaultclamp_{var}"),
        help="Keep extreme values at the fitted end-point instead of extending the curve beyond the training data.",
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    has_clamp = isinstance(vd.clamp, (list, tuple)) and len(vd.clamp) == 2
    lo = c1.number_input(
        "Clamp lo",
        value=ui.safe_float(vd.clamp[0], rlo) if has_clamp else rlo,
        key=S.widget_key(f"lin_lo_{var}"),
        disabled=use_default,
        format="%g",
    )
    hi = c2.number_input(
        "Clamp hi",
        value=ui.safe_float(vd.clamp[1], rhi) if has_clamp else rhi,
        key=S.widget_key(f"lin_hi_{var}"),
        disabled=use_default,
        format="%g",
    )
    c3.caption(
        "Values below lo / above hi get the relativity at the clamp. Knots must lie "
        "strictly inside the clamp range."
    )
    if st.button(
        "Apply continuous design" if continuous else "Apply linear design",
        key=S.widget_key(f"apply_lin_{var}"),
        type="primary",
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
        if not continuous and strategy == "custom":
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
            vd.kind = "continuous" if continuous else "linear"
            if not continuous:
                vd.knots = knots if strategy == "custom" else strategy
                vd.n_bins = int(n_bins) if strategy == "quantile" else vd.n_bins
            vd.clamp = clamp
            p.design.variables[var] = vd
            S.touch()
            st.rerun()
    edges = enc.band_edges()
    u = univariate(
        preview,
        var,
        target=target,
        weight=weight,
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
            title=(
                f"{var}: one straight line between {enc.lo:g} and {enc.hi:g}"
                if continuous
                else f"{var}: linear in {enc.n_bands} band(s) between "
                f"{enc.lo:g} and {enc.hi:g}"
            ),
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
    var: str,
    vd: VariableDesign,
    enc: CategoricalEncoder,
    preview,
    target: str | None,
    weight: str | None,
    divide,
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
        help="Levels not listed are grouped into Other; the first listed level is the 1.00 reference.",
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
        target=target,
        weight=weight,
        divide_target_by_weight=divide,
        max_levels=len(enc.levels),
    )
    st.plotly_chart(
        C.exposure_rate_chart(
            u["table"], title=f"{var}: {len(enc.levels)} levels + Other"
        ),
        width="stretch",
    )


def _detail(
    train: pl.DataFrame,
    preview: pl.DataFrame,
    predictors: list[str],
    *,
    target: str | None,
    weight: str | None,
    divide_target_by_weight: bool,
) -> None:
    p = S.project()
    st.subheader("Variable detail")
    c1, c2 = st.columns([2, 1])
    var = c1.selectbox("Variable", predictors, key=S.widget_key("design_detail_var"))
    vd = p.design.variables.get(var, VariableDesign())
    numeric = train[var].dtype in NUMERIC_DTYPES
    with c2:
        _kind_selector(var, vd, numeric)
    vd = p.design.variables.get(var, VariableDesign())
    weights = train[weight] if weight and weight in train.columns else None
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
                    vd.n_bins = len(knots) + 1
                    p.design.variables[var] = vd
                    S.touch()
                    ui.flash(
                        "success",
                        f"{var}: using {len(knots)} custom knots "
                        f"({len(knots) + 1} bins) instead of automatic quantile binning",
                    )
                    st.rerun()
        return
    if isinstance(enc, StepEncoder):
        _step_detail(
            var, vd, enc, train, preview, target, weight, divide_target_by_weight
        )
    elif isinstance(enc, LinearEncoder):
        _linear_detail(
            var,
            vd,
            enc,
            train,
            preview,
            target,
            weight,
            divide_target_by_weight,
            continuous=vd.kind == "continuous",
        )
    elif isinstance(enc, CategoricalEncoder):
        _categorical_detail(
            var, vd, enc, preview, target, weight, divide_target_by_weight
        )


# --------------------------------------------------------------------------
# interactions
# --------------------------------------------------------------------------
def _interactions(train: pl.DataFrame, name: str) -> None:
    p = S.project()
    st.subheader("Interactions")
    st.caption(
        "A two-way interaction A × B adds one adjustment per cell on top of the two "
        "main effects; cells with too little exposure get no adjustment (1.00). "
        "Add them here, fit on the Model page, see the cells on the Rate tables page."
    )
    cfg = p.models[name]
    if cfg.interactions:
        st.caption(
            "Main effects stay fixed; interaction cells are selected automatically by "
            "the model's CV setup (or its fixed alpha)."
        )
        for i, it in enumerate(list(cfg.interactions)):
            c1, c2, c3 = st.columns([3, 3, 1])
            c1.markdown(f"**{it.name}**")
            c2.caption(
                f"Minimum cell exposure: {it.min_cell_exposure:.2%}"
                if ui.safe_float(it.min_cell_exposure, None) is not None
                else f"Minimum cell exposure: {it.min_cell_exposure!r} (invalid)"
            )
            if c3.button("Remove", key=S.widget_key(f"rm_inter_{name}_{i}")):
                cfg.interactions.pop(i)
                # the cells are gone for good, so their adjustments go with them
                # — out of the working set *and* out of every snapshot, which
                # could otherwise never be restored again
                dropped = cfg.drop_adjustments_for(it.name)
                if dropped:
                    ui.flash(
                        "info",
                        f"{it.name} removed, with {dropped} cell adjustment(s) "
                        "of its own (snapshots included).",
                    )
                S.touch()
                st.rerun()
            with st.expander(f"Advanced interaction settings — {it.name}"):
                weight, problem = ui.repair_number(
                    it.penalty_weight, 1.0, "penalty weight", lo=0.0, hi=100.0
                )
                if problem:
                    st.warning(problem)
                new_weight = st.number_input(
                    "Penalty weight",
                    min_value=0.0,
                    max_value=100.0,
                    value=weight,
                    step=0.1,
                    key=S.widget_key(f"inter_w_{name}_{i}"),
                    help=(
                        "Relative L1 shrinkage for this interaction: 1 is normal, "
                        "higher values shrink it more, and 0 leaves its cells unpenalised."
                    ),
                )
                if new_weight != it.penalty_weight:
                    it.penalty_weight = float(new_weight)
                    S.touch()
                    st.rerun()
                if it.alpha is not None:
                    usable_alpha = (
                        isinstance(it.alpha, (int, float))
                        and not isinstance(it.alpha, bool)
                        and math.isfinite(it.alpha)
                        and it.alpha > 0
                    )
                    if not usable_alpha:
                        st.warning(
                            f"{it.name}: legacy cell-alpha override {it.alpha!r} "
                            "is not a usable number; using automatic model-level "
                            "selection instead."
                        )
                        it.alpha = None
                        S.touch()
                    else:
                        st.caption(
                            "Legacy cell-alpha override retained from this project file. "
                            "New interactions select their penalty automatically."
                        )
    else:
        st.info(
            "**No interactions are included in this model yet.** The controls and "
            "chart below only preview a possible interaction. Nothing will be fitted "
            "unless you press **Add interaction**."
        )
    preds = list(cfg.predictors)
    if len(preds) < 2:
        st.info("The model needs at least two predictors before adding an interaction.")
        return
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 1])
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
            help=(
                "Credibility threshold: cells below this share of the pair's training "
                "exposure keep a 1.00 adjustment."
            ),
        )
        with st.expander("Advanced interaction settings"):
            weight = st.number_input(
                "Penalty weight",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=0.1,
                key=S.widget_key(f"inter_w_{name}"),
                help=(
                    "Relative L1 shrinkage for this interaction: 1 is normal, higher "
                    "values shrink it more, and 0 leaves its cells unpenalised."
                ),
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
def render_contents() -> str | None:
    """Draw the definition and factor-design portions of the Model workflow."""
    p = S.project()
    df = ui.require_data()
    if df is None:
        return None
    name = _model_picker()
    if name is None:
        return None
    _model_definition(name, df)
    cfg = p.models[name]
    predictors = list(cfg.predictors)
    if not predictors:
        st.info(
            "Choose one or more predictor roles above to define this model's design."
        )
        return name
    train = S.train_frame()  # knots and levels always come from the full training rows
    preview = (
        S.train_sample()
    )  # exposure / rate previews may use the exploration sample
    if train is None or preview is None:
        return None
    if train.is_empty():
        st.error("There are no training rows; check the split on the Split page.")
        return None
    missing = [v for v in predictors if v not in train.columns]
    if missing:
        st.error(
            "These predictors are not in the prepared data (renamed or removed?): "
            + ", ".join(missing)
            + ". Fix their roles on the Variables page."
        )
        predictors = [v for v in predictors if v in train.columns]
        if not predictors:
            return None
    for m in [m for m in p.validate(name) if m.startswith("design[")]:
        st.error(m)
    if S.is_sampled():
        st.caption(
            f"Knots and levels are derived from all {train.height:,} training rows; "
            f"the preview charts use the exploration sample ({preview.height:,} rows)."
        )
    st.header("Factor design")
    _defaults()
    st.caption(
        "Numeric predictors default to **step** (one 0/1 column per knot, penalised increments → automatic banding); "
        "the explicit overrides are **linear** (a continuous curve whose slope may change at each knot, flat unless "
        "the data insists), **continuous** (one straight line, no knots) and **categorical** (each value a level). "
        "Categoricals become one-hot with the most frequent level as reference and an **Other** bucket. "
        "Monotone constraints bound the step increments or the band slopes; they are available for every numeric "
        "kind, not for categoricals. "
        "**Penalty weight** scales how hard the lasso shrinks one factor: 1 = like the rest of the design, "
        "2 = twice as hard, **0 = unpenalised**, so every band or level of that factor is kept."
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
    _detail(
        train,
        preview,
        predictors,
        # A hand-edited model may point target/weight at a text column. The
        # definition card reports that problem; previews must still render.
        target=(
            cfg.target
            if cfg.target in train.columns and train[cfg.target].dtype in NUMERIC_DTYPES
            else None
        ),
        weight=(
            cfg.weight
            if cfg.weight in train.columns and train[cfg.weight].dtype in NUMERIC_DTYPES
            else None
        ),
        divide_target_by_weight=(
            cfg.divide_target_by_weight
            and cfg.weight in train.columns
            and train[cfg.weight].dtype in NUMERIC_DTYPES
        ),
    )
    _interactions(train, name)
    return name


def render() -> None:
    """Compatibility route for an old direct ``/design`` URL.

    The sidebar now presents one combined Model workflow; keeping this small
    route avoids a broken saved browser URL without adding another navigation
    decision for new users.
    """
    st.title("Model design and fit")
    ui.status_bar()
    st.info("Model design now lives on the **Model** page in the sidebar.")
    render_contents()
