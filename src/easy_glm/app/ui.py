"""Small shared widgets and formatting helpers for the workbench pages."""

from __future__ import annotations

import io
import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import polars as pl
import streamlit as st

from easy_glm.workflow import ModelRun
from easy_glm.workflow.project import safe_filename as _safe_filename

from . import state as S

T = TypeVar("T")


def fmt(x: Any, *, pct: bool = False, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if pct:
        return "—" if abs(x) > 100 else f"{x:.1%}"  # ±10,000 % is noise, not a number
    if isinstance(x, int | float) and abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def relativity_note_markdown(rm: Any) -> str:
    """``rm.relativity_label`` / ``relativity_note`` as one markdown line.

    The engine's note is plain text (it also lands verbatim in Excel and the
    HTML report); only here, where Streamlit renders markdown, do "overall"
    and "differential" get bolded alongside the label."""
    note = rm.relativity_note.replace("overall", "**overall**").replace(
        "differential", "**differential**"
    )
    return f"**{rm.relativity_label}** — {note}"


def metric_row(items: list[tuple[str, Any, str | None]]) -> None:
    """``[(label, value, help), ...]`` rendered as a row of ``st.metric``."""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items, strict=True):
        col.metric(label, value, help=help_text)


def flash(kind: str, text: str) -> None:
    """Queue a one-shot notice to show at the top of the *next* run.

    Streamlit discards anything drawn in a run that ends with ``st.rerun()``, so
    every message that precedes a rerun goes through here and is rendered by
    :func:`show_flash` (called from :func:`status_bar`)."""
    st.session_state.setdefault("_flash", []).append((kind, text))


def show_flash() -> None:
    notices = st.session_state.pop("_flash", [])
    for kind, text in notices:
        getattr(
            st, kind if kind in ("success", "warning", "error", "info") else "info"
        )(text)


def safe_float(value: Any, default: float) -> float:
    """``float(value)``, or ``default`` when ``value`` is not a finite number.

    A hand-edited project file can put anything in a numeric field (``"abc"``,
    a list, ``NaN``). Every page that reads such a field straight into a
    widget's ``value=`` (an ``alpha``, a ``tweedie_power``, a base-rate
    override, a clamp bound) used to crash on ``float(...)`` before the user
    ever saw the message ``Project.validate`` names the problem with — this is
    the fallback that lets the page render so that message can be shown.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_int(value: Any, default: int) -> int:
    """``int(value)``, or ``default`` when ``value`` cannot be one — the
    integer counterpart of :func:`safe_float` (a hand-edited ``cv`` or
    ``n_alphas`` typed as text or a list must not crash the page it would be
    reported as a problem on)."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def repair_number(
    value: Any, default: float, label: str, *, integer: bool = False
) -> tuple[float, str | None]:
    """``(value or default, message)`` for a numeric project field that a
    page both displays *and* writes straight back to the project once the
    page has finished rendering (``pages_model._config``'s "did anything
    change" reconciliation, the Design page's interaction ``alpha`` box).

    ``safe_float``/``safe_int`` alone stop the crash, but the page would then
    silently save the fallback over whatever the file held next time it
    autosaves — the same shape of problem the seed box on the Split page
    already solves (``pages_split._seed_value``): say what was wrong and what
    replaced it, the way that box does, instead of rewriting the file with no
    trace of the original value. ``None`` is never a problem here — several of
    these fields (``base_rate_override``, ``penalty.alpha``, an interaction's
    ``alpha``) use it as "not set", and the caller's own ``default`` is the
    right value to show in that case.
    """
    if value is None:
        return default, None
    usable = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
    fixed = safe_int(value, int(default)) if integer else safe_float(value, default)
    if usable:
        return fixed, None
    return (
        fixed,
        f"{label} in the project file ({value!r}) is not a usable number; "
        f"using {fixed:g} instead.",
    )


def number_in_range(
    container: Any,
    label: str,
    *,
    value: Any,
    key: str,
    what: str,
    lo: float | None = None,
    hi: float | None = None,
    **kwargs: Any,
) -> Any:
    """A number box that can never show a number the project does not hold.

    ``st.number_input``'s ``min_value`` / ``max_value`` are enforced in the
    browser only: a pasted or typed out-of-range value stays on screen while
    the server quietly keeps the previous one, so the page ends up naming a
    number the fit did not use. This widget takes whatever is typed, and when
    it falls outside ``lo``–``hi`` it says so, puts the stored value back in the
    box and redraws the page.
    """
    # a correction from the previous run has to be applied before the widget
    # exists — Streamlit refuses to set a widget's key afterwards. Dropping the
    # key (rather than assigning to it) makes the box fall back to ``value``,
    # which is the number the project holds, and keeps the log clean.
    repair = f"_repair_{key}"
    if st.session_state.pop(repair, False):
        st.session_state.pop(key, None)
    typed = container.number_input(label, value=value, key=key, **kwargs)
    if typed is None or ((lo is None or typed >= lo) and (hi is None or typed <= hi)):
        return typed
    if lo is not None and hi is not None:
        allowed = f"between {lo:g} and {hi:g}"
    elif lo is not None:
        allowed = f"{lo:g} or more"
    else:
        allowed = f"{hi:g} or less"
    st.session_state[repair] = True
    flash(
        "error",
        f"{what} must be {allowed}: {typed:g} was not used, and the box is back "
        f"to {value:g}.",
    )
    st.rerun()


def guarded(fn: Callable[[], T], what: str, *, default: T | None = None) -> T | None:
    """Run ``fn``; on any exception show a clear message naming ``what`` and
    return ``default``. The pipeline error boundary for every page: a bad
    rename, recode, derived column, filter or split must never be a traceback."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - that is the point
        st.error(f"{what} failed: {exc}")
        return default


def conflict_notice() -> None:
    """Persistent notice while the project file was changed by another
    session; autosave is paused until the user chooses."""
    path = st.session_state.get("conflict")
    if not path:
        return
    st.warning(
        f"**{path}** was changed by another browser tab or session since this tab "
        "last saved it. Autosave is paused so neither copy is lost: reload the "
        "file (this tab's unsaved edits are dropped) or overwrite it with this "
        "tab's version."
    )
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Reload from disk", key="conflict_reload", type="primary"):
        err = S.resolve_conflict("reload")
        if err:
            st.error(err)
        else:
            flash("success", f"Reloaded {path}")
            st.rerun()
    if c2.button("Overwrite with this tab's version", key="conflict_overwrite"):
        err = S.resolve_conflict("overwrite")
        if err:
            st.error(err)
        else:
            flash("success", f"Saved this tab's version to {path}")
            st.rerun()


def interrupted_fit_notice() -> None:
    """Once per session: name any fit that was started and never saved (the
    page was reloaded, or the app stopped, part-way through)."""
    if st.session_state.get("interrupted_checked"):
        return
    st.session_state["interrupted_checked"] = True
    for model in S.interrupted_fits():
        st.warning(
            f"A fit of **{model}** was interrupted — the page was reloaded (or the "
            "app stopped) before it finished, so nothing was saved and the model is "
            "not fitted. Fit it again on the **Model** page."
        )


def status_bar() -> None:
    show_flash()
    show_errors()
    interrupted_fit_notice()
    conflict_notice()
    s = S.status()
    steps = [
        ("data", "Data"),
        ("roles", "Roles"),
        ("split", "Prepared"),
        ("model", "Model"),
        ("fitted", "Fitted"),
    ]
    chips = " ".join(
        (":green-badge[✓ " if s[k] else ":grey-badge[○ ") + label + "]"
        for k, label in steps
    )
    st.markdown(chips)


def require_data() -> pl.DataFrame | None:
    """The prepared frame, or None after showing why it is missing."""
    df = S.prepared_frame()
    if df is None:
        show_data_problem()
    return df


def show_data_problem() -> None:
    """The load or data-step error (if any), else a hint to load data."""
    err = st.session_state.get("load_error") or st.session_state.get("prep_error")
    if err:
        st.error(err)
    else:
        st.info("Load a data file on the **Project & data** page first.")


def require_raw() -> pl.DataFrame | None:
    """The raw frame, or None after showing why it is missing."""
    p = S.project()
    raw = S.raw_frame() if p.data.source.path else None
    if raw is None:
        err = st.session_state.get("load_error")
        if err:
            st.error(err)
        else:
            st.info("Load a data file on the **Project & data** page first.")
    return raw


def require_target() -> str | None:
    t = S.project().target
    if t is None:
        st.info("Assign a **target** role on the **Variables** page first.")
    return t


def run_selector(label: str = "Model", key: str = "run_select") -> ModelRun | None:
    runs = S.current_runs()
    fitted = {name: r for name, r in runs.items() if S.get_run(name) is not None}
    if not fitted:
        st.info("No fitted model yet — fit one on the **Model** page.")
        return None
    p = S.project()
    default = p.champion if p.champion in fitted else next(iter(fitted))
    names = list(fitted)
    name = st.selectbox(label, names, index=names.index(default), key=key)
    return fitted[name]


def frame_bytes(df: pl.DataFrame, kind: str = "csv") -> bytes:
    buf = io.BytesIO()
    if kind == "csv":
        df.write_csv(buf)
    else:
        df.write_parquet(buf)
    return buf.getvalue()


#: file-name-safe model / project names (defined in the workflow layer so the
#: command line can use it without importing Streamlit)
safe_filename = _safe_filename


def excel_bytes(run: ModelRun) -> bytes:
    """Workbook of the tables the scorer uses (manual adjustments included)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{safe_filename(run.name)}.xlsx"
        run.rate_model.to_excel(path)
        return path.read_bytes()


def easyglm_bytes(run: ModelRun) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{safe_filename(run.name)}.easyglm"
        run.rate_model.to_json(path)
        return path.read_bytes()


def show_errors() -> None:
    """Persistent problems (autosave / persistence failures) on every page;
    they clear when the cause is fixed (a successful save drops them)."""
    for e in st.session_state.get("errors", []):
        st.error(e)


def polars_table(df: pl.DataFrame, **kwargs: Any) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)
