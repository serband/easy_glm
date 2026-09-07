"""Page 2 — Variables: roles, renames, types, level recodes, derived columns, filters.

The roles grid is applied through :func:`apply_roles_grid`, a pure function
(no Streamlit) so its rules — a rename never collides with another column, a
cleared cell means "no rename", a rename carries roles and model references,
a role change keeps every model consistent — are unit-testable.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any

import pandas as pd
import polars as pl
import streamlit as st

from easy_glm.core.design import NUMERIC_DTYPES
from easy_glm.workflow import Derived, Project, Recode, apply_variables, eval_expr
from easy_glm.workflow.project import ROLES, SINGLE_ROLES, premium_offset_column

from . import pages_split, ui
from . import state as S

ROLE_OPTIONS = ["unassigned", *ROLES]
TYPE_OPTIONS = ["auto", "categorical", "numeric"]
BULK_ROLE_GROUPS = ("predictor", "id", "unassigned", "ignore")
BULK_TYPE_GROUPS = ("categorical", "numeric", "auto")


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
    # Final names as they stand now, and as this edit wants them. Validate the
    # whole set before changing anything: a pasted bulk edit must never be
    # half-applied just because its last row contains a collision.
    finals = {c: p.data.renames.get(c, c) for c in raw_columns}
    wanted_finals = dict(finals)
    derived_names = {d.name for d in p.data.derived}
    rows_by_column = {r.get("column"): r for r in rows}
    for raw_name, r in rows_by_column.items():
        if raw_name not in finals:
            continue
        wanted_finals[raw_name] = _cell_text(r.get("rename to")) or raw_name

    collisions: dict[str, list[str]] = {}
    for raw_name, final in wanted_finals.items():
        collisions.setdefault(final, []).append(raw_name)
    duplicate_names = {name: cols for name, cols in collisions.items() if len(cols) > 1}
    if duplicate_names:
        for name, cols in duplicate_names.items():
            notices.append(
                (
                    "error",
                    f"Cannot use final name {name!r} for {', '.join(cols)}. "
                    "Another column already has that name; every column needs a "
                    "different final name. Nothing was changed.",
                )
            )
        return False, notices
    for raw_name, final in wanted_finals.items():
        if final in derived_names and final != finals[raw_name]:
            notices.append(
                (
                    "error",
                    f"Cannot rename {raw_name!r} to {final!r}: a derived column "
                    "already has that name. Nothing was changed.",
                )
            )
    if notices:
        return False, notices

    # Rename through unique temporary names. This makes a bulk swap such as
    # A -> B and B -> A well-defined, and lets us apply every validated rename
    # atomically rather than depending on JSON/grid row order.
    rename_plans: list[tuple[str, str, str, list[str], list[str]]] = []
    reserved = set(finals.values()) | set(wanted_finals.values()) | derived_names
    for index, raw_name in enumerate(raw_columns):
        current = finals[raw_name]
        wanted = wanted_finals[raw_name]
        if current == wanted:
            continue
        temporary = f"__easy_glm_bulk_rename_{index}__"
        while temporary in reserved:
            temporary += "_"
        reserved.add(temporary)
        expressions = p.expressions_using(current)
        p.data.renames[raw_name] = temporary
        touched = p.rename_column(current, temporary)
        rename_plans.append((raw_name, current, wanted, expressions, touched))
        finals[raw_name] = temporary

    for raw_name, current, wanted, expressions, touched in rename_plans:
        temporary = finals[raw_name]
        touched = sorted(set(touched) | set(p.rename_column(temporary, wanted)))
        if wanted == raw_name:
            p.data.renames.pop(raw_name, None)
        else:
            p.data.renames[raw_name] = wanted
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

    for raw_name, r in rows_by_column.items():
        raw_name = r["column"]
        if raw_name not in finals:
            continue
        final = finals[raw_name]
        role = r.get("role") or "unassigned"
        if role == "unassigned":
            if final in p.data.roles:
                old_role = p.data.roles.pop(final)
                if old_role == "predictor":
                    notices.extend(("warning", n) for n in _drop_from_models(p, final))
                elif old_role == "current_premium":
                    notices.extend(
                        ("warning", n) for n in _drop_premium_offset(p, final)
                    )
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


def variable_setup_json(p: Project, raw_columns: list[str]) -> str:
    """A compact, copy/paste-friendly snapshot of the variables grid.

    Column names in every section are deliberately the raw source names: they
    remain stable when ``renames`` changes and make pasted settings unambiguous.
    Ignored columns and automatic types are omitted because those are the bulk
    format's defaults. Unassigned columns are listed explicitly so regenerating
    and applying the current setup is lossless.
    """
    renames: dict[str, str] = {}
    assignments: dict[str, str] = {}
    roles: dict[str, list[str]] = {}
    types: dict[str, list[str]] = {}
    for raw_name in raw_columns:
        final = p.data.renames.get(raw_name, raw_name)
        if final != raw_name:
            renames[raw_name] = final
        role = p.data.roles.get(final, "unassigned")
        if role in SINGLE_ROLES:
            assignments[role] = raw_name
        elif role != "ignore":
            roles.setdefault(role, []).append(raw_name)
        kind = p.data.types.get(final, "auto")
        if kind != "auto":
            types.setdefault(kind, []).append(raw_name)
    setup = {
        "renames": renames,
        "assignments": assignments,
        "roles": roles,
        "types": types,
    }
    return json.dumps(setup, indent=2, ensure_ascii=False)


def parse_variable_setup_json(
    p: Project, raw_columns: list[str], text: str
) -> tuple[list[dict[str, str]], list[str]]:
    """Parse a section-based bulk variable setup without mutating ``p``.

    Omitted renames leave the source name unchanged; columns absent from both
    assignments and roles default to ignore; columns absent from types default
    to auto. Every column reference is a raw source-column name.
    """
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        return [], [f"Invalid JSON: {exc}"]
    if not isinstance(payload, dict):
        return [], ["The JSON must be an object containing the four sections."]

    raw_set = set(raw_columns)
    errors: list[str] = []
    allowed_sections = {"renames", "assignments", "roles", "types"}
    unknown_sections = sorted(set(payload) - allowed_sections)
    if unknown_sections:
        errors.append(
            "Unknown top-level section(s): "
            + ", ".join(repr(name) for name in unknown_sections)
        )

    sections: dict[str, dict[str, Any]] = {}
    for name in allowed_sections:
        section = payload.get(name, {})
        if not isinstance(section, dict):
            errors.append(f"{name!r} must be a JSON object.")
            sections[name] = {}
        else:
            sections[name] = section

    rows_by_column = {
        raw_name: {
            "column": raw_name,
            "rename to": "",
            "role": "ignore",
            "type": "auto",
        }
        for raw_name in raw_columns
    }

    def raw_column(value: Any, where: str) -> str | None:
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{where} must name one raw source column.")
            return None
        clean = value.strip()
        if clean not in raw_set:
            errors.append(f"{where} refers to unknown raw column {clean!r}.")
            return None
        return clean

    for raw_name, wanted in sections["renames"].items():
        source = raw_column(raw_name, "A renames key")
        if source is None:
            continue
        if wanted is None:
            continue
        if not isinstance(wanted, str) or not wanted.strip():
            errors.append(f"renames.{source} must be a non-empty name or null.")
            continue
        rows_by_column[source]["rename to"] = wanted.strip()

    assigned_roles: dict[str, str] = {}

    def assign_role(raw_name: str, role: str, where: str) -> None:
        previous = assigned_roles.get(raw_name)
        if previous is not None and previous != role:
            errors.append(
                f"Raw column {raw_name!r} appears in both {previous!r} and "
                f"{role!r} roles ({where})."
            )
            return
        assigned_roles[raw_name] = role
        rows_by_column[raw_name]["role"] = role

    for role, value in sections["assignments"].items():
        if role not in SINGLE_ROLES:
            errors.append(
                f"assignments.{role} is not a single-column assignment; use one "
                "of: " + ", ".join(SINGLE_ROLES)
            )
            continue
        source = raw_column(value, f"assignments.{role}")
        if source is not None:
            assign_role(source, role, f"assignments.{role}")

    for role, values in sections["roles"].items():
        if role not in BULK_ROLE_GROUPS:
            errors.append(
                f"roles.{role} is not valid here; use one of: "
                + ", ".join(BULK_ROLE_GROUPS)
            )
            continue
        if not isinstance(values, list):
            errors.append(f"roles.{role} must be a JSON list of raw columns.")
            continue
        for index, value in enumerate(values):
            source = raw_column(value, f"roles.{role}[{index}]")
            if source is not None:
                assign_role(source, role, f"roles.{role}")

    assigned_types: dict[str, str] = {}
    for kind, values in sections["types"].items():
        if kind not in BULK_TYPE_GROUPS:
            errors.append(
                f"types.{kind} is not valid; use one of: " + ", ".join(BULK_TYPE_GROUPS)
            )
            continue
        if not isinstance(values, list):
            errors.append(f"types.{kind} must be a JSON list of raw columns.")
            continue
        for index, value in enumerate(values):
            source = raw_column(value, f"types.{kind}[{index}]")
            if source is None:
                continue
            previous = assigned_types.get(source)
            if previous is not None and previous != kind:
                errors.append(
                    f"Raw column {source!r} appears in both {previous!r} and "
                    f"{kind!r} type groups."
                )
                continue
            assigned_types[source] = kind
            rows_by_column[source]["type"] = kind

    rows = [rows_by_column[name] for name in raw_columns]

    if errors:
        return rows, errors

    # Exercise the exact same rules as Apply, but only on a copy. This catches
    # final-name collisions and guarantees the real project remains untouched.
    candidate = deepcopy(p)
    _changed, notices = apply_roles_grid(candidate, raw_columns, rows)
    errors.extend(text for kind, text in notices if kind == "error")
    return rows, errors


def variable_setup_changes(
    p: Project, raw_columns: list[str], rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Plain preview rows for the settings that would actually change."""
    changes: list[dict[str, str]] = []
    for row in rows:
        raw_name = row["column"]
        if raw_name not in raw_columns:
            continue
        current_name = p.data.renames.get(raw_name, raw_name)
        wanted_name = _cell_text(row.get("rename to")) or raw_name
        current_role = p.data.roles.get(current_name, "unassigned")
        current_type = p.data.types.get(current_name, "auto")
        wanted_role = row["role"]
        wanted_type = row["type"]
        if (current_name, current_role, current_type) == (
            wanted_name,
            wanted_role,
            wanted_type,
        ):
            continue
        changes.append(
            {
                "raw column": raw_name,
                "name": (
                    current_name
                    if current_name == wanted_name
                    else f"{current_name} → {wanted_name}"
                ),
                "role": (
                    current_role
                    if current_role == wanted_role
                    else f"{current_role} → {wanted_role}"
                ),
                "type": (
                    current_type
                    if current_type == wanted_type
                    else f"{current_type} → {wanted_type}"
                ),
            }
        )
    return changes


def _drop_premium_offset(p: Project, column: str) -> list[str]:
    """The derived ``log(premium)`` column goes with the role, so a model still
    offsetting on it would fail at the next fit."""
    gone = premium_offset_column(column)
    notes: list[str] = []
    for name, cfg in p.models.items():
        if cfg.offset == gone:
            cfg.offset = None
            notes.append(
                f"Model {name} no longer offsets on {gone!r}: {column} is not the "
                "current premium any more"
            )
    return notes


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
        st.session_state[S.widget_key("bulk_roles_refresh")] = True
        st.rerun()
    if c2.button("Unassigned → predictor"):
        for r in rows:
            new = p.data.renames.get(r["column"], r["column"])
            p.data.roles.setdefault(new, "predictor")
        S.touch()
        st.session_state[S.widget_key("bulk_roles_refresh")] = True
        st.rerun()

    use_json = st.toggle(
        "Bulk edit with JSON",
        help=(
            "Switch from the table to a copy/paste editor for changing many "
            "column names, roles and types at once."
        ),
        key=S.widget_key("bulk_roles_toggle"),
    )
    if use_json:
        _bulk_roles_json(p, raw)
    else:
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
        changed, notices = apply_roles_grid(
            p, list(raw.columns), edited.to_dict("records")
        )
        if changed:
            for kind, text in notices:
                ui.flash(kind, text)
            S.touch()
            st.session_state[S.widget_key("bulk_roles_refresh")] = True
            st.rerun()
        for kind, text in notices:  # a refused rename remains visible in the grid
            getattr(st, kind)(text)

    roles = p.data.roles
    summary = " · ".join(
        f"**{r}**: {', '.join(p.columns_with_role(r)) or '—'}"
        for r in ("target", "weight", "exposure", "offset", "current_premium", "split")
    )
    st.caption(
        summary
        + f" · **predictors**: {len(p.predictors)} · **ignored**: {len(p.columns_with_role('ignore'))}"
    )
    if roles and p.target is None:
        st.warning("No target assigned yet.")
    if (premium := p.current_premium) is not None:
        st.caption(
            f"Rate change: `{premium_offset_column(premium)}` = log({premium}) is "
            "derived for you and pre-filled as the offset of new models, so a model "
            "fits the **change** from today's premium. Filter out rows with a "
            "premium of zero or less first."
        )


def _bulk_roles_json(p: Project, raw: pl.DataFrame) -> None:
    """Render the guarded copy/paste alternative to the variables grid."""
    raw_columns = list(raw.columns)
    editor_key = S.widget_key("bulk_roles_json_v2")
    refresh_key = S.widget_key("bulk_roles_refresh")
    if st.session_state.pop(refresh_key, False) or editor_key not in st.session_state:
        st.session_state[editor_key] = variable_setup_json(p, raw_columns)

    st.info(
        "This compact format has four sections. `renames` maps raw names to new "
        "names. `assignments` sets the single target, weight, exposure, offset, "
        "current premium and split columns. `roles` groups predictors and IDs. "
        "`types` groups explicit categorical or numeric overrides. All names "
        "refer to the raw source columns. A column omitted from assignments and "
        "roles becomes **ignored**; one omitted from types stays **auto**. "
        "Nothing is saved until you select **Apply JSON changes**."
    )
    if st.button(
        "Reset JSON from current setup",
        key=S.widget_key("bulk_roles_reset"),
        help="Discard text in this editor and regenerate it from the project.",
    ):
        st.session_state[refresh_key] = True
        st.rerun()

    text = st.text_area(
        "Variable setup JSON",
        height=460,
        key=editor_key,
        help=(
            "Use raw source-column names throughout. Omit unchanged renames, "
            "ignored columns and automatic types."
        ),
    )
    rows, errors = parse_variable_setup_json(p, raw_columns, text)
    changes = [] if errors else variable_setup_changes(p, raw_columns, rows)
    if errors:
        st.error("Fix the JSON before applying it:\n\n- " + "\n- ".join(errors))
    elif changes:
        st.caption(f"Proposed changes: {len(changes)} column(s).")
        st.dataframe(pd.DataFrame(changes), hide_index=True, width="stretch")
    else:
        st.caption("Valid JSON. It matches the current variable setup.")

    if st.button(
        "Apply JSON changes",
        type="primary",
        disabled=bool(errors) or not changes,
        key=S.widget_key("bulk_roles_apply"),
    ):
        changed, notices = apply_roles_grid(p, raw_columns, rows)
        unexpected = [text for kind, text in notices if kind == "error"]
        if not changed or unexpected:
            st.error(
                "The JSON could not be applied. "
                + (" ".join(unexpected) if unexpected else "No settings changed.")
            )
            return
        for kind, notice in notices:
            ui.flash(kind, notice)
        ui.flash("success", f"Applied JSON changes to {len(changes)} column(s).")
        S.touch()
        st.session_state[refresh_key] = True
        st.rerun()


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
        "**split** = an existing train/holdout indicator; you can instead create a "
        "seeded random split below; "
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
    st.divider()
    pages_split.render_contents(raw)
