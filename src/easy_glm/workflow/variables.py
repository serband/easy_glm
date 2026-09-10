"""Shared variable setup validation and atomic project edits."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any

from .project import SINGLE_ROLES, Project, premium_offset_column

BULK_ROLE_GROUPS = ("predictor", "id", "unassigned", "ignore")
BULK_TYPE_GROUPS = ("categorical", "numeric", "auto")


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
    Every column's role is explicit, including ignored and unassigned columns,
    so switching editors preserves the visible assignments. Empty singleton
    roles use null and empty role groups use lists to show every available role.
    Automatic types are omitted because those are the bulk format's default.
    """
    renames: dict[str, str] = {}
    assignments: dict[str, str | None] = dict.fromkeys(SINGLE_ROLES)
    roles: dict[str, list[str]] = {role: [] for role in BULK_ROLE_GROUPS}
    types: dict[str, list[str]] = {}
    for raw_name in raw_columns:
        final = p.data.renames.get(raw_name, raw_name)
        if final != raw_name:
            renames[raw_name] = final
        role = p.data.roles.get(final, "unassigned")
        if role in SINGLE_ROLES:
            assignments[role] = raw_name
        else:
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
    assignments and roles default to ignore; null singleton assignments name no
    column. Columns absent from types default to auto. Every column reference
    is a raw source-column name.
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
        if value is None:
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
        cfg.drop_adjustments_for(column)
        for interaction in dropped:
            cfg.drop_adjustments_for(interaction.name)
        cfg.monotone.pop(column, None)
    return notes
