"""Page 2 — Variables: roles, renames, types, level recodes, derived columns, filters."""

from __future__ import annotations

import pandas as pd
import polars as pl
import streamlit as st

from easy_glm.core.design import NUMERIC_DTYPES
from easy_glm.workflow import Derived, Recode, apply_variables, eval_expr
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
        key="roles_grid",
    )
    changed = False
    for _, r in edited.iterrows():
        raw_name = r["column"]
        new_name = (r["rename to"] or "").strip() or raw_name
        old_new = p.data.renames.get(raw_name, raw_name)
        if new_name != old_new:
            if new_name == raw_name:
                p.data.renames.pop(raw_name, None)
            else:
                p.data.renames[raw_name] = new_name
            for store in (
                p.data.roles,
                p.data.types,
                p.data.recodes,
                p.design.variables,
            ):
                if old_new in store:
                    store[new_name] = store.pop(old_new)
            changed = True
        role = r["role"]
        if role == "unassigned":
            if new_name in p.data.roles:
                p.data.roles.pop(new_name)
                changed = True
        elif p.data.roles.get(new_name) != role:
            p.set_role(new_name, role)
            changed = True
        kind = r["type"]
        if kind == "auto":
            if new_name in p.data.types:
                p.data.types.pop(new_name)
                changed = True
        elif p.data.types.get(new_name) != kind:
            p.data.types[new_name] = kind
            changed = True
    if changed:
        S.touch()
        st.rerun()

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
    after_rename = apply_variables(
        raw, _data_without(p, "recodes", "derived", "filters", "types")
    )
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
        key="recode_col",
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
            key=f"recode_grid_{col}",
        )
    with c2:
        policy = st.radio(
            "Unmapped levels",
            ["keep", "→ Other", "→ value"],
            index=0 if rc.default is None else (1 if rc.default == "Other" else 2),
            key="recode_policy",
        )
        literal = (
            st.text_input("value", rc.default or "", key="recode_literal")
            if policy == "→ value"
            else None
        )
        if st.button("Apply recode", type="primary", key="recode_apply"):
            mapping = {
                str(r["level"]): str(r["map to"]).strip()
                for _, r in edited.iterrows()
                if r["level"] is not None
                and str(r["map to"]).strip()
                and str(r["map to"]).strip() != str(r["level"])
            }
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
        if col in p.data.recodes and st.button("Remove recode", key="recode_remove"):
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


def _data_without(p, *fields):
    from copy import deepcopy

    d = deepcopy(p.data)
    for f in fields:
        setattr(d, f, {} if f in ("recodes", "types") else [])
    return d


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
        if c3.button("✕", key=f"del_derived_{i}", help="Remove"):
            p.data.derived.pop(i)
            S.touch()
            st.rerun()
    c1, c2 = st.columns([1, 3])
    name = c1.text_input("New column name", key="derived_name")
    expr = c2.text_input("Expression", key="derived_expr")
    b1, b2 = st.columns([1, 1])
    if b1.button("Preview", key="derived_preview") and name and expr:
        try:
            base = apply_variables(raw.head(2000), p.data)
            prev = base.with_columns(eval_expr(expr).alias(name)).select(name)
            st.write(
                prev.describe()
                if prev[name].dtype in NUMERIC_DTYPES
                else prev[name].value_counts().head(20)
            )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    if (
        b2.button("Add derived column", type="primary", key="derived_add")
        and name
        and expr
    ):
        try:
            eval_expr(expr)
        except ValueError as exc:
            st.error(str(exc))
        else:
            p.data.derived.append(Derived(name=name, expr=expr))
            p.data.roles.setdefault(name, "predictor")
            S.touch()
            st.rerun()


def _filters(raw: pl.DataFrame) -> None:
    p = S.project()
    for i, f in enumerate(list(p.data.filters)):
        c1, c2 = st.columns([8, 1])
        c1.code(f, language="python")
        if c2.button("✕", key=f"del_filter_{i}"):
            p.data.filters.pop(i)
            S.touch()
            st.rerun()
    expr = st.text_input(
        "New filter (rows to keep)",
        placeholder="pl.col('Exposure') > 0",
        key="filter_expr",
    )
    if st.button("Add filter", key="filter_add") and expr:
        try:
            kept = apply_variables(raw, p.data).filter(eval_expr(expr)).height
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
        else:
            p.data.filters.append(expr)
            S.touch()
            st.success(f"Filter added — {kept:,} rows kept")
            st.rerun()


def render() -> None:
    st.title("Variables")
    ui.status_bar()
    raw = S.raw_frame() if S.project().data.source.path else None
    if raw is None:
        st.info("Load a data file on the **Project & data** page first.")
        return
    st.subheader("Roles, names and types")
    st.caption(
        "Exactly one **target**; **weight** = exposure or premium used as GLM weight; "
        "**split** = train/holdout indicator (or use a random split on the Split page); "
        "**id** and **ignore** are excluded from modelling."
    )
    _roles_grid(raw)
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
