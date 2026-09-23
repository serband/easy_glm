"""Actuarial workbook export for a pricing workflow checkpoint."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from easy_glm.core.excel import rate_model_tables, write_rate_tables_xlsx


def _audit_frame(rows: list[dict[str, Any]], columns: dict[str, Any]) -> pl.DataFrame:
    if rows:
        return pl.DataFrame(rows)
    return pl.DataFrame(schema=columns)


def _evidence_rows(model: Any) -> list[dict[str, str]]:
    rows = [
        {"item": str(key), "value": json.dumps(value, default=str)}
        for key, value in sorted(model._evidence.items())
    ]
    for order, artifact in enumerate(model._run.pair_stages, start=1):
        for key in (
            "stage_id",
            "parents",
            "status",
            "prefix_cv_loss",
            "table_cv_loss",
            "teacher_cv_loss",
            "approximation_loss",
            "training_table_loss",
            "fit_seconds",
        ):
            rows.append(
                {
                    "item": f"pair {order} {key}",
                    "value": json.dumps(getattr(artifact, key, None), default=str),
                }
            )
    return rows


def _scoring_rules(model: Any) -> list[str]:
    link = model._run.rate_model.metadata.link
    if link == "logit":
        return [
            "Start with the saved base odds.",
            "Multiply the main-factor odds relativities in the variable sheets.",
            "Multiply pair-stage odds tables in the order shown on Pair stages.",
            "Apply the saved offset once on the log-odds scale, when present.",
            "Convert odds to probability as odds / (1 + odds); never multiply the probability by exposure.",
        ]
    if link == "log":
        return [
            "Start with the saved base rate.",
            "Multiply the main-factor relativities in the variable sheets.",
            "Multiply pair-stage tables in the order shown on Pair stages.",
            "Apply the saved offset once on the log scale, when present.",
            "Multiply the final unit prediction by exposure only when an expected total is required.",
        ]
    return [
        f"This model uses the {link or 'saved'} link; score it with the frozen EasyGLM JSON scorer.",
        "Do not interpret these tables as multiplicative rate relativities.",
    ]


def _recorded_validation(model: Any) -> dict[str, dict[str, Any]]:
    """Current fit metrics plus subsets opened through explicit validation."""
    recorded = {
        str(subset): dict(metrics) for subset, metrics in model._run.metrics.items()
    }
    evidence = model._evidence.get("validation", {})
    if isinstance(evidence, dict):
        for subset, metrics in evidence.items():
            if isinstance(metrics, dict):
                recorded[str(subset)] = dict(metrics)
    return recorded


def export_excel(model: Any, path: str | Path) -> Path:
    """Export exactly scored tables plus amendments, evidence and scoring rules."""
    scorer = model._run.rate_model
    tables = rate_model_tables(scorer)
    amendments = []
    for item in model._run.config.adjustments:
        raw = asdict(item)
        amendments.append(
            {
                "variable": str(raw["variable"]),
                "stage_id": raw.get("stage_id") or "",
                "from": json.dumps(raw.get("from_"), default=str),
                "to": json.dumps(raw.get("to_"), default=str),
                "from_b": json.dumps(raw.get("from_b"), default=str),
                "to_b": json.dumps(raw.get("to_b"), default=str),
                "axis_a_row": raw.get("axis_a_row"),
                "axis_b_row": raw.get("axis_b_row"),
                "relativity": float(raw["relativity"]),
            }
        )
    tables["Amendments"] = _audit_frame(
        amendments,
        {
            "variable": pl.Utf8,
            "stage_id": pl.Utf8,
            "from": pl.Utf8,
            "to": pl.Utf8,
            "from_b": pl.Utf8,
            "to_b": pl.Utf8,
            "axis_a_row": pl.Int64,
            "axis_b_row": pl.Int64,
            "relativity": pl.Float64,
        },
    )
    tables["Evidence"] = _audit_frame(
        _evidence_rows(model), {"item": pl.Utf8, "value": pl.Utf8}
    )
    recorded_validation = _recorded_validation(model)
    validation_rows = [
        {"subset": subset, "metric": key, "value": json.dumps(value, default=str)}
        for subset, metrics in recorded_validation.items()
        for key, value in metrics.items()
    ]
    tables["Validation"] = _audit_frame(
        validation_rows,
        {"subset": pl.Utf8, "metric": pl.Utf8, "value": pl.Utf8},
    )
    history_rows = [
        {
            "step": index,
            "action": str(entry.get("action", "")),
            "details": json.dumps(entry, default=str),
        }
        for index, entry in enumerate(model._history, start=1)
    ]
    tables["History"] = _audit_frame(
        history_rows,
        {"step": pl.Int64, "action": pl.Utf8, "details": pl.Utf8},
    )
    rules = _scoring_rules(model)
    tables["Scoring rules"] = pl.DataFrame(
        {
            "order": [str(index) for index in range(1, len(rules) + 1)],
            "rule": rules,
        }
    )
    summary = {
        "model": model.name,
        "tables": "current deployed scoring values, including amendments",
        "base_rate": scorer.base_rate,
        "family": model._run.config.family,
        "link": scorer.metadata.link,
        "target": model._run.config.target,
        "weight": model._run.config.weight,
        "target divided by weight": model._run.config.divide_target_by_weight,
        "offset": model._run.config.offset,
        "main factors": list(scorer.variables),
        "ordered pair stages": [table.stage_id for table in scorer.pair_tables],
        "history entries": len(model._history),
        "recorded evidence": len(model._evidence),
        "recorded validation subsets": list(recorded_validation),
    }
    return write_rate_tables_xlsx(
        tables,
        path,
        summary=summary,
        pair_tables=scorer.pair_tables or None,
    )
