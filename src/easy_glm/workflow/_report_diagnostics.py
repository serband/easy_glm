"""Original-fit diagnostics at the start of the report's rating factors."""

from __future__ import annotations

import math
from html import escape
from typing import Any

import polars as pl

from ._diagnostic_svg import coefficient_path_chart, permutation_importance_chart
from .diagnostics import coefficient_path, permutation_importance
from .project import Project
from .run import ModelRun


def _number(value: Any) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    number = float(value)
    if number and abs(number) < 0.01:
        mantissa, exponent = format(number, ".3e").split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent):+d}"
    return f"{number:,.3f}".rstrip("0").rstrip(".")


def _importance_table(rows: list[dict[str, Any]]) -> str:
    body = "".join(
        f'<tr><th scope="row">{escape(row["variable"])}</th>'
        f'<td class="num">{_number(row["importance"])}</td>'
        f'<td class="num">{_number(row["std"])}</td></tr>'
        for row in rows
    )
    return (
        '<details class="diagnostic-values"><summary>Importance values</summary>'
        '<div class="scroll"><table><thead><tr><th scope="col">Predictor</th>'
        '<th scope="col" class="num">Mean deviance increase</th>'
        '<th scope="col" class="num">Shuffle SD</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div></details>"
    )


def _importance_section(
    project: Project,
    run: ModelRun,
    train: pl.DataFrame,
    importance: pl.DataFrame | None,
) -> str:
    out = [
        '<section id="variable-importance" class="report-diagnostic">',
        "<h3>Variable importance</h3>",
        '<p class="muted">Original fit · training data · five shuffles per predictor. '
        "Larger increases in mean deviance indicate greater importance.</p>",
    ]
    if train.height < 2:
        out.append('<p class="muted">At least two training rows are needed.</p>')
    else:
        protected = tuple(
            name
            for name, role in project.data.roles.items()
            if role
            in {
                "target",
                "weight",
                "exposure",
                "offset",
                "current_premium",
                "id",
                "split",
            }
        ) + (project.data.split.column,)
        if importance is None:
            importance = permutation_importance(
                run.fit, train, repeats=5, seed=42, protected_columns=protected
            )
        rows = (
            importance.sort(
                ["importance", "variable"], descending=[True, False]
            ).to_dicts()
            if not importance.is_empty()
            else []
        )
        if rows:
            out.append(
                permutation_importance_chart(
                    rows[:30], title="Training permutation importance — original fit"
                )
            )
            out.append(
                '<p class="muted diagnostic-caption">Whiskers show ±1 standard deviation '
                "across shuffles, not confidence intervals. Related predictors can share importance.</p>"
            )
            if len(rows) > 30:
                out.append(
                    f'<p class="muted">Top 30 of {len(rows):,} predictors shown; all values are below.</p>'
                )
            out.append(_importance_table(rows))
        else:
            out.append('<p class="muted">No eligible predictors in this fit.</p>')
    out.append("</section>")
    return "".join(out)


def _coefficient_section(run: ModelRun) -> str:
    out = [
        '<section id="coefficient-paths" class="report-diagnostic">',
        "<h3>Coefficients versus lambda</h3>",
        '<p class="muted">Stronger regularisation is to the right. Each line is one '
        "encoded coefficient; magnitudes depend on the variable's scale and encoding.</p>",
    ]
    path = coefficient_path(run.fit)
    if path.is_empty():
        out.append('<p class="muted">This fit has no encoded coefficients.</p>')
    else:
        for group in path.partition_by(["stage", "l1_ratio"], maintain_order=True):
            stage = group["stage"][0]
            l1_ratio = group["l1_ratio"][0]
            label = "Main effects" if stage == 1 else "Interaction cells"
            out.append(
                f'<div class="coefficient-panel"><h4>{label}'
                f' <span class="tag">L1 ratio {_number(l1_ratio)}</span></h4>'
            )
            source = group["source"][0]
            positive = group.filter(pl.col("alpha").is_finite() & (pl.col("alpha") > 0))
            if group["alpha"].n_unique() <= 1:
                alpha = group["alpha"][0]
                out.append(
                    f'<p class="muted">Fitted at a single lambda ({_number(alpha)}); '
                    "no coefficient path was recorded.</p>"
                )
            elif positive["alpha"].n_unique() <= 1:
                out.append(
                    '<p class="muted">At least two positive lambda values are needed for a logarithmic path chart.</p>'
                )
            else:
                selected = group.filter(pl.col("selected"))
                selected_alpha = (
                    float(selected["alpha"][0]) if selected.height else None
                )
                out.append(
                    coefficient_path_chart(
                        positive.to_dicts(),
                        selected_alpha=(
                            selected_alpha
                            if selected_alpha and selected_alpha > 0
                            else None
                        ),
                        title=f"{label}: coefficients versus lambda (L1 ratio {_number(l1_ratio)})",
                    )
                )
                basis = (
                    f'Mean coefficients across {group["folds"][0]} validation folds. The final full-training fit may differ.'
                    if source == "cv_fold_mean"
                    else "Coefficients recorded along the fitted path."
                )
                marker = (
                    f" Selected lambda: {_number(selected_alpha)}."
                    if selected_alpha is not None
                    else " This L1 ratio was not selected."
                )
                out.append(f'<p class="muted diagnostic-caption">{basis}{marker}</p>')
                if positive.height != group.height:
                    out.append(
                        '<p class="muted">Lambda zero is omitted from the logarithmic axis.</p>'
                    )
            out.append("</div>")
    out.append("</section>")
    return "".join(out)


def fitted_diagnostics_section(
    project: Project,
    run: ModelRun,
    train: pl.DataFrame,
    *,
    importance: pl.DataFrame | None = None,
) -> str:
    """Render original-fit importance and stored paths without modifying the run."""
    return _importance_section(project, run, train, importance) + _coefficient_section(
        run
    )


DIAGNOSTIC_CSS = """
.report-diagnostic { margin: 24px 0 32px; }
.report-diagnostic > h3 { margin-bottom: 6px; }
.report-diagnostic svg { display: block; width: 100%; height: auto; }
.diagnostic-caption { margin-top: 8px; font-size: 13px; }
.diagnostic-values { margin: 12px 0 20px; }
.diagnostic-values > summary { cursor: pointer; color: #334d66; font-weight: 600; }
.coefficient-panel { margin: 18px 0 28px; break-inside: avoid; }
@media print {
  #coefficient-paths { break-before: page; }
  .report-diagnostic > h3 { break-after: avoid; }
  .report-diagnostic > p { break-inside: avoid; }
}
"""
