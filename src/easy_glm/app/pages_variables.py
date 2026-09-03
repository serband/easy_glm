"""Page 2 — Variables: roles, renames, types, level recodes, derived columns, filters.

The roles grid is applied through :func:`apply_roles_grid`, a pure function
(no Streamlit) so its rules — a rename never collides with another column, a
cleared cell means "no rename", a rename carries roles and model references,
a role change keeps every model consistent — are unit-testable.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import polars as pl
import streamlit as st

from easy_glm.core.design import NUMERIC_DTYPES
from easy_glm.workflow import Derived, Project, Recode, apply_variables, eval_expr
from easy_glm.workflow.project import ROLES

from . import state as S
from . import ui

ROLE_OPTIONS = ["unassigned", *ROLES]
TYPE_OPTIONS = ["auto", "categorical", "numeric"]


def _guess_role(name: str, dtype: pl.DataType, n_unique: int, n: int) -> str:
    low = name.lower()
    if low in ("traintest", "train_test", "split", "is_train"):
        return "split"
    if low in ("exposure", "expo", "earned_exposure"):
        return "weight"
    if low.startswith("id") or low.endswith("id") or low.endswith("_id"):
        return "id"
    if n_unique > 0.9 * n and n > 100 and dtype not in NUMERIC_DTYPES:
        return "id"
    return "predictor"


def _cell_text(value: Any) -> str:
    """A text cell from the data editor: NaN / None / whitespace mean empty."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def apply_roles_grid(
    p: Project, raw_columns: list[str], rows: list[dict[str, Any]]
) -> tuple[bool, list[tuple[str, str]]]:
    """Apply the edited roles grid to ``p``.

    ``rows`` are ``{"column", "rename to", "role", "type"}`` per raw column.
    Returns ``(changed, notices)`` where notices are ``(kind, text)`` pairs for
    the user. Rules: a rename that would collide with another column's final
    name is refused; an emptied "rename to" cell undoes the rename (and its
    role follows the column back); a rename carries roles, types, recodes,
    design, row filters, derived formulas and every model reference; a role
    change keeps models consistent (a predictor leaving a model is reported,
    never silently).
    """
    notices: list[tuple[str, str]] = []
    changed = False
    # final names as they stand now, per raw column
    finals = {c: p.data.renames.get(c, c) for c in raw_columns}
    derived_names = {d.name for d in p.data.derived}
    for r in rows:
        raw_name = r["column"]
        if raw_name not in finals:
            continue
        wanted = _cell_text(r.get("rename to")) or raw_name
        current = finals[raw_name]
        if wanted != current:
            others = {v for c, v in finals.items() if c != raw_name} | derived_names
            if wanted in others:
                notices.append(
                    (
                        "error",
                        f"Cannot rename {raw_name!r} to {wanted!r}: another column "
                        "already has that name. Rename not saved.",
                    )
                )
            else:
                if wanted == raw_name:
                    p.data.renames.pop(raw_name, None)
                else:
                    p.data.renames[raw_name] = wanted
                expressions = p.expressions_using(current)
                touched = p.rename_column(current, wanted)
                finals[raw_name] = wanted
                changed = True
                if touched:
                    notices.append(
                        (
                            "info",
                            f"{current!r} renamed to {wanted!r} in model(s): "
                            + ", ".join(touched),
                        )
                    )
                if expressions:
                    notices.append(
                        (
                            "info",
                            f"{current!r} renamed to {wanted!r} in "
                            f"{len(expressions)} row filter / derived formula(s): "
                            + "; ".join(expressions),
                        )
                    )
        final = finals[raw_name]
        role = r.get("role") or "unassigned"
        if role == "unassigned":
            if final in p.data.roles:
                old_role = p.data.roles.pop(final)
                if old_role == "predictor":
                    notices.extend(("warning", n) for n in _drop_from_models(p, final))
                changed = True
        elif p.data.roles.get(final) != role:
            notices.extend(("warning", n) for n in p.apply_role_change(final, role))
            changed = True
        kind = r.get("type") or "auto"
        if kind == "auto":
            if final in p.data.types:
                p.data.types.pop(final)
                changed = True
        elif p.data.types.get(final) != kind:
            p.data.types[final] = kind
            changed = True
    return changed, notices


def _drop_from_models(p: Project, column: str) -> list[str]:
    """Remove a column that is no longer a predictor from every model."""
    notes: list[str] = []
    for name, cfg in p.models.items():
        if column in cfg.predictors:
            cfg.predictors = [v for v in cfg.predictors if v != column]
            notes.append(f"{column} was removed from model {name}: it is unassigned")
        dropped = [it for it in cfg.interactions if column in (it.a, it.b)]
        if dropped:
            cfg.interactions = [it for it in cfg.interactions if it not in dropped]
            notes.append(
                f"Interaction(s) {', '.join(it.name for it in dropped)} removed from "
                f"model {name}"
            )
        cfg.monotone.pop(column, None)
    return notes


def _roles_grid(raw: pl.DataFrame) -> None:
    p = S.project()
    rows = []
    n = max(raw.height, 1)
    for name, dtype in raw.schema.items():
        new = p.data.renames.get(name, name)
        rows.append(
            {
                "column": name,
                "rename to": new if new != name else "",
                "role": p.data.roles.get(new, "unassigned"),
                "type": p.data.types.get(new, "auto"),
                "dtype": str(dtype),
                "null %": round(100 * raw[name].null_count() / n, 1),
                "unique": raw[name].n_unique(),
            }
        )
    grid = pd.DataFrame(rows)
    c1, c2, c3 = st.columns([1, 1, 3])
    if c1.button(
        "Auto-assign roles",
        help="Guess split / weight / id / predictor from names and cardinality",
    ):
        for r in rows:
            new = p.data.renames.get(r["column"], r["column"])
            if p.data.roles.get(new, "unassigned") == "unassigned":
                p.data.roles[new] = _guess_role(
                    new, raw.schema[r["column"]], r["unique"], raw.height
                )
        S.touch()
        st.rerun()
    if c2.button("Unassigned → predictor"):
        for r in rows:
            new = p.data.renames.get(r["column"], r["column"])
            p.data.roles.setdefault(new, "predictor")
        S.touch()
        st.rerun()

    edited = st.data_editor(
        grid,
        hide_index=True,
        width="stretch",
        height=min(38 * (len(rows) + 1) + 4, 620),
        disabled=["column", "dtype", "null %", "unique"],
        column_config={
            "role": st.column_config.SelectboxColumn(
                "role", options=ROLE_OPTIONS, required=True
            ),
            "type": st.column_config.SelectboxColumn(
                "type", options=TYPE_OPTIONS, required=True
            ),
            "rename to": st.column_config.TextColumn("rename to"),
            "null %": st.column_config.NumberColumn("null %", format="%.1f"),
        },
        key=S.widget_key("roles_grid"),
    )
    changed, notices = apply_roles_grid(p, list(raw.columns), edited.to_dict("records"))
    if changed:
        for kind, text in notices:
            ui.flash(kind, text)
        S.touch()
        st.rerun()
    for kind, text in notices:  # e.g. a refused rename still sitting in the grid
        getattr(st, kind)(text)

    roles = p.data.roles
    summary = " · ".join(
        f"**{r}**: {', '.join(p.columns_with_role(r)) or '—'}"
        for r in ("target", "weight", "exposure", "offset", "split")
    )
    st.caption(
        summary
        + f" · **predictors**: {len(p.predictors)} · **ignored**: {len(p.columns_with_role('ignore'))}"
    )
    if roles and p.target is None:
        st.warning("No target assigned yet.")


def _recodes(raw: pl.DataFrame) -> None:
    p = S.project()
    sample = S.raw_sample()
    if sample is None:
        return
    if S.is_sampled():
        st.caption(
            f"Level counts from the exploration sample ({sample.height:,} rows)."
        )
    after_rename = ui.guarded(
        lambda: apply_variables(
            sample, _data_without(p, "recodes", "derived", "filters", "types")
        ),
        "Applying the renames",
    )
    if after_rename is None:
        return
    cat_cols = [
        c
        for c, t in after_rename.schema.items()
        if t not in NUMERIC_DTYPES or p.data.types.get(c) == "categorical"
    ]
    if not cat_cols:
        st.caption("No categorical columns.")
        return
    existing = list(p.data.recodes)
    col = st.selectbox(
        "Column",
        cat_cols,
        index=(
            cat_cols.index(existing[0]) if existing and existing[0] in cat_cols else 0
        ),
        key=S.widget_key("recode_col"),
    )
    rc = p.data.recodes.get(col, Recode())
    counts = (
        after_rename.select(pl.col(col).cast(pl.Utf8).alias("level"))
        .group_by("level")
        .agg(pl.len().alias("rows"))
        .sort("rows", descending=True)
        .head(200)
    )
    grid = pd.DataFrame(
        {
            "level": counts["level"].to_list(),
            "rows": counts["rows"].to_list(),
            "map to": [
                rc.mapping.get(lv, "") if lv is not None else ""
                for lv in counts["level"].to_list()
            ],
        }
    )
    c1, c2 = st.columns([3, 1])
    with c1:
        edited = st.data_editor(
            grid,
            hide_index=True,
            width="stretch",
            height=360,
            disabled=["level", "rows"],
            key=S.widget_key(f"recode_grid_{col}"),
        )
    with c2:
        policy = st.radio(
            "Unmapped levels",
            ["keep", "→ Other", "→ value"],
            index=0 if rc.default is None else (1 if rc.default == "Other" else 2),
            key=S.widget_key("recode_policy"),
        )
        literal = (
            st.text_input("value", rc.default or "", key=S.widget_key("recode_literal"))
            if policy == "→ value"
            else None
        )
        if st.button("Apply recode", type="primary", key=S.widget_key("recode_apply")):
            mapping = recode_mapping(edited.to_dict("records"))
            default = (
                None
                if policy == "keep"
                else ("Other" if policy == "→ Other" else (literal or "Other"))
            )
            if mapping or default is not None:
                p.data.recodes[col] = Recode(mapping=mapping, default=default)
            else:
                p.data.recodes.pop(col, None)
            S.touch()
            st.success(f"Recode saved for {col} ({len(mapping)} mapped levels)")
        if col in p.data.recodes and st.button(
            "Remove recode", key=S.widget_key("recode_remove")
        ):
            p.data.recodes.pop(col)
            S.touch()
            st.rerun()
    if p.data.recodes:
        st.caption(
            "Active recodes: "
            + ", ".join(
                f"{k} ({len(v.mapping)} levels, default={v.default or 'keep'})"
                for k, v in p.data.recodes.items()
            )
        )


def recode_mapping(rows: list[dict[str, Any]]) -> dict[str, str]:
    """``level -> new level`` from the recode grid; an empty, blank or NaN
    "map to" cell means "no mapping" (never a level called "nan")."""
    mapping: dict[str, str] = {}
    for r in rows:
        level = r.get("level")
        if level is None or (isinstance(level, float) and math.isnan(level)):
            continue
        target = _cell_text(r.get("map to"))
        if target and target != str(level):
            mapping[str(level)] = target
    return mapping


def _data_without(p, *fields):
    from copy import deepcopy

    d = deepcopy(p.data)
    for f in fields:
        setattr(d, f, {} if f in ("recodes", "types") else [])
    return d


def preview_derived(
    p: Project, raw: pl.DataFrame, name: str, expr: str
) -> tuple[pl.DataFrame | None, str | None]:
    """Evaluate a derived column on (a head of) ``raw`` the way the pipeline
    will; returns ``(preview, error)``. Used by Preview and by Add so that an
    expression that fails at run time (missing column, wrong types, a column
    referencing itself) is refused before it reaches the project."""
    name = (name or "").strip()
    expr = (expr or "").strip()
    if not name:
        return None, "Give the new column a name"
    if not expr:
        return None, "Enter an expression"
    try:
        e = eval_expr(expr)
    except ValueError as exc:
        return None, str(exc)
    try:
        base = apply_variables(raw.head(2000), p.data)
        prev = base.with_columns(e.alias(name)).select(name)
    except Exception as exc:  # noqa: BLE001 - polars errors, shown verbatim
        return None, f"{expr} fails: {exc}"
    return prev, None


def _derived(raw: pl.DataFrame) -> None:
    p = S.project()
    st.caption(
        "Polars expressions with `pl` and `np` available, e.g. "
        "`pl.when(pl.col('Lic') == 'Q').then(pl.col('Exp')).otherwise(0)` or "
        "`(pl.col('VehValue') / 1000).round(0)`. Earlier derived columns can be used by later ones."
    )
    for i, d in enumerate(list(p.data.derived)):
        c1, c2, c3 = st.columns([2, 6, 1])
        c1.code(d.name, language=None)
        c2.code(d.expr, language="python")
        if c3.button("✕", key=S.widget_key(f"del_derived_{i}"), help="Remove"):
            removed = p.data.derived.pop(i)
            p.data.roles.pop(removed.name, None)
            notes = _drop_from_models(p, removed.name)
            for note in notes:
                ui.flash("warning", note)
            S.touch()
            st.rerun()
    c1, c2 = st.columns([1, 3])
    name = c1.text_input("New column name", key=S.widget_key("derived_name"))
    expr = c2.text_input("Expression", key=S.widget_key("derived_expr"))
    b1, b2 = st.columns([1, 1])
    sample = S.raw_sample()
    base = raw if sample is None else sample
    if b1.button("Preview", key=S.widget_key("derived_preview")):
        prev, err = preview_derived(p, base, name, expr)
        if err:
            st.error(err)
        else:
            st.write(
                prev.describe()
                if prev[name].dtype in NUMERIC_DTYPES
                else prev[name].value_counts().head(20)
            )
    if b2.button("Add derived column", type="primary", key=S.widget_key("derived_add")):
        taken = {p.data.renames.get(c, c) for c in raw.columns} | {
            d.name for d in p.data.derived
        }
        clean = (name or "").strip()
        if clean in taken:
            st.error(f"A column named {clean!r} already exists")
            return
        _prev, err = preview_derived(p, base, clean, expr)
        if err:
            st.error(err)
            return
        p.data.derived.append(Derived(name=clean, expr=expr.strip()))
        p.data.roles.setdefault(clean, "predictor")
        S.touch()
        ui.flash("success", f"Derived column {clean!r} added (role: predictor)")
        st.rerun()


def _filters(raw: pl.DataFrame) -> None:
    p = S.project()
    for i, f in enumerate(list(p.data.filters)):
        c1, c2 = st.columns([8, 1])
        c1.code(f, language="python")
        if c2.button("✕", key=S.widget_key(f"del_filter_{i}")):
            p.data.filters.pop(i)
            S.touch()
            st.rerun()
    expr = st.text_input(
        "New filter (rows to keep)",
        placeholder="pl.col('Exposure') > 0",
        key=S.widget_key("filter_expr"),
    )
    if st.button("Add filter", key=S.widget_key("filter_add")) and expr:
        try:
            kept = apply_variables(raw, p.data).filter(eval_expr(expr)).height
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
        else:
            p.data.filters.append(expr)
            S.touch()
            ui.flash("success", f"Filter added — {kept:,} rows kept")
            st.rerun()


def _missing_role_columns(raw: pl.DataFrame) -> None:
    """Roles the project holds for columns this data file does not have (a new
    file, a removed derived column): kept, never re-pointed, but said out loud."""
    p = S.project()
    final = {p.data.renames.get(c, c) for c in raw.columns}
    final |= {d.name for d in p.data.derived}
    gone = [c for c, role in p.data.roles.items() if c not in final and role]
    if gone:
        st.warning(
            "Columns with a role that are not in this data file: "
            + ", ".join(f"**{c}** ({p.data.roles[c]})" for c in gone)
            + ". Their roles and any model using them are kept as they are; the "
            "Model page says what is missing and fitting waits until you rename a "
            "column to that name, add a derived column with it, or unassign it."
        )


def render() -> None:
    st.title("Variables")
    ui.status_bar()
    raw = ui.require_raw()
    if raw is None:
        return
    st.subheader("Roles, names and types")
    st.caption(
        "Exactly one **target**; **weight** = exposure or premium used as GLM weight; "
        "**split** = train/holdout indicator (or use a random split on the Split page); "
        "**id** and **ignore** are excluded from modelling. Renaming a column carries "
        "its role and every model reference with it."
    )
    _roles_grid(raw)
    _missing_role_columns(raw)
    tab1, tab2, tab3 = st.tabs(["Level recodes", "Derived columns", "Row filters"])
    with tab1:
        _recodes(raw)
    with tab2:
        _derived(raw)
    with tab3:
        _filters(raw)
    df = S.prepared_frame()
    if df is not None:
        st.caption(
            f"Prepared data: {df.height:,} rows × {df.width} columns after recodes, derived columns and filters."
        )
    else:
        ui.show_data_problem()
