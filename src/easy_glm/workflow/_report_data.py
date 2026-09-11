"""Compact data profiles for the standalone model report."""

from __future__ import annotations

import math
from html import escape
from typing import Any

import polars as pl

from easy_glm.core.design import CategoricalEncoder

from ._profile_svg import correlation_matrix, mini_histogram
from .data_summary import data_summary
from .run import ModelRun


def _number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    if number and (abs(number) < 0.001 or abs(number) >= 1e10):
        return f"{number:.3e}"
    return f"{number:,.3f}".rstrip("0").rstrip(".")


def _overview(
    rows: list[dict[str, Any]], roles: dict[str, str], dropped: set[str]
) -> str:
    body = []
    for row in rows:
        name = row["name"]
        notes = [roles.get(name, "Predictor"), row["kind"].capitalize()]
        if name in dropped:
            notes.append("Omitted from fit: no training variation")
        if row.get("note"):
            notes.append(row["note"])
        if row["kind"] == "numeric" and row["nonfinite"]:
            notes.append(f'{row["nonfinite"]:,} infinite values excluded')
        histogram = row["histogram"]
        chart = (
            mini_histogram(
                [item["label"] for item in histogram],
                [item["count"] for item in histogram],
                title=f"{name}: training distribution",
            )
            if histogram
            else '<span class="muted">'
            + ("Not profiled" if row["kind"] == "unsupported" else "No observed values")
            + "</span>"
        )
        body.append(
            f'<tr><th scope="row"><span class="profile-name">{escape(name)}</span>'
            f'<span class="profile-caption">{escape(" · ".join(notes))}</span></th>'
            f'<td class="num">{_number(row["missing_pct"])}%'
            f'<span class="profile-caption">{row["missing"]:,} rows</span></td>'
            f'<td class="num">{_number(row["unique"])}</td>'
            f'<td class="num">{_number(row["min"])}</td>'
            f'<td class="num">{_number(row["max"])}</td>'
            f'<td class="num">{_number(row["range"])}</td>'
            f'<td class="profile-distribution">{chart}</td></tr>'
        )
    return (
        '<div class="scroll"><table class="profile-overview">'
        "<caption>Selected predictors, target, weight and offset where present.</caption>"
        '<thead><tr><th scope="col">Variable</th><th scope="col" class="num">Missing</th>'
        '<th scope="col" class="num">Unique</th><th scope="col" class="num">Minimum</th>'
        '<th scope="col" class="num">Maximum</th><th scope="col" class="num">Range</th>'
        '<th scope="col">Distribution · rows</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def _shape_table(rows: list[dict[str, Any]]) -> str:
    numeric = [row for row in rows if row["kind"] == "numeric"]
    if not numeric:
        return ""
    columns = [
        ("Mean", "mean"),
        ("Median", "median"),
        ("Std. deviation", "std"),
        ("Skewness", "skewness"),
        ("Excess kurtosis", "kurtosis"),
    ]
    headers = "".join(
        f'<th scope="col" class="num">{label}</th>' for label, _ in columns
    )
    body = "".join(
        f'<tr><th scope="row">{escape(row["name"])}</th>'
        + "".join(f'<td class="num">{_number(row[key])}</td>' for _, key in columns)
        + "</tr>"
        for row in numeric
    )
    return (
        '<details class="profile-details" open><summary>Location, spread and shape</summary>'
        '<p class="muted">Skewness measures asymmetry; excess kurtosis measures tail '
        "weight relative to a normal distribution (0). Undefined values are shown as —.</p>"
        '<div class="scroll"><table class="profile-shape"><thead><tr>'
        f'<th scope="col">Variable</th>{headers}</tr></thead><tbody>{body}</tbody>'
        "</table></div></details>"
    )


def _correlations(result: dict[str, Any]) -> str:
    names = result["names"]
    out = [
        "<h3>Predictor correlations</h3>",
        '<p class="muted">Pearson correlation between numeric predictors, before '
        "matrix expansion. Each pair uses rows where both values are present and finite; "
        "each row counts equally.</p>",
    ]
    if result["sampled"]:
        out.append(
            f'<p class="profile-scope">Correlations use a reproducible sample of '
            f'{result["sample_rows"]:,} of {result["total_rows"]:,} training rows. '
            "The statistics and distributions above use all training rows.</p>"
        )
    if len(names) < 2:
        out.append('<p class="muted">At least two numeric predictors are needed.</p>')
    elif result["mode"] == "matrix":
        out.append(
            '<div class="scroll profile-correlations">'
            + correlation_matrix(
                names,
                result["matrix"],
                result["counts"],
                title="Numeric predictor correlations on training rows",
            )
            + '</div><p class="muted">Hover over a cell for the correlation and paired '
            "row count. A constant predictor or too few paired values gives —.</p>"
        )
    else:
        pairs = result["pairs"]
        out.append(
            f'<p class="muted">Strongest {len(pairs):,} of {result["valid_pairs"]:,} '
            f"valid pairs across {len(names):,} numeric predictors, ranked by absolute "
            "correlation. Positive values move together; negative values move oppositely.</p>"
        )
        body = "".join(
            f'<tr><th scope="row">{escape(pair["left"])}</th><td>{escape(pair["right"])}</td>'
            f'<td class="num">{_number(pair["correlation"])}</td>'
            f'<td class="num">{pair["rows"]:,}</td></tr>'
            for pair in pairs
        )
        out.append(
            '<div class="scroll"><table><thead><tr><th>First predictor</th>'
            '<th>Second predictor</th><th class="num">Correlation</th>'
            f'<th class="num">Paired rows</th></tr></thead><tbody>{body}</tbody></table></div>'
        )
    if result["excluded_names"]:
        out.append(
            '<p class="muted">Not included in numeric correlations: '
            + escape(", ".join(result["excluded_names"]))
            + ". Their distributions are shown above.</p>"
        )
    return "".join(out)


def data_summary_section(run: ModelRun, train: pl.DataFrame, holdout_rows: int) -> str:
    """Describe the champion's chosen columns without touching the fitted model."""
    predictors = list(dict.fromkeys(run.config.predictors + run.spec.required_columns))
    roles: dict[str, str] = {}
    for name, label in (
        (run.config.target, "Target"),
        (run.config.weight, "Weight / exposure"),
        (run.config.offset, "Offset"),
    ):
        if name:
            roles[name] = f"{roles[name]} · {label}" if name in roles else label
    variables = list(dict.fromkeys(predictors + list(roles)))
    categorical = {
        name
        for name, encoder in run.spec.encoders.items()
        if isinstance(encoder, CategoricalEncoder)
    }
    design = run.project_snapshot.get("design", {}).get("variables", {})
    categorical.update(
        name
        for name in run.dropped_predictors
        if design.get(name, {}).get("kind") == "categorical"
    )
    profile = data_summary(
        train, variables, categorical=categorical, correlation_variables=predictors
    )
    rows = profile["variables"]
    predictor_rows = [row for row in rows if row["name"] in predictors]
    missing = sum(row["missing"] for row in predictor_rows)
    possible = train.height * len(predictor_rows)
    missing_pct = 100 * missing / possible if possible else 0.0
    cards = [
        ("Training rows", f"{train.height:,}"),
        ("Selected predictors", f"{len(predictors):,}"),
        (
            "Predictors with missing values",
            str(sum(row["missing"] > 0 for row in predictor_rows)),
        ),
        ("Missing predictor values", f"{_number(missing_pct)}%"),
    ]
    overview = "".join(
        f"<div><span>{label}</span><strong>{value}</strong></div>"
        for label, value in cards
    )
    unavailable = (
        '<p class="muted">Columns unavailable in the supplied data: '
        + escape(", ".join(profile["unavailable_variables"]))
        + ".</p>"
        if profile["unavailable_variables"]
        else ""
    )
    return (
        '<section id="data-summary"><h2>2. Data summary</h2>'
        '<p class="muted">Training data after applied filters and recodes. '
        f"{holdout_rows:,} holdout rows are excluded. Statistics and distributions "
        "count each row equally, without exposure weighting.</p>"
        f'<div class="profile-kpis">{overview}</div><h3>Variables at a glance</h3>'
        + unavailable
        + _overview(rows, roles, set(run.dropped_predictors))
        + '<p class="muted">Missing includes null and NaN. Unique excludes missing; '
        "numeric summaries exclude infinite values. Range = maximum − minimum. "
        "Category charts show up to eight levels, with the rest grouped together.</p>"
        + _shape_table(rows)
        + _correlations(profile["correlations"])
        + "</section>"
    )


PROFILE_CSS = """
.profile-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px; margin: 18px 0 24px; }
.profile-kpis > div { background: #fff; border: 1px solid #e2e7ec;
  border-radius: 7px; padding: 12px 14px; }
.profile-kpis span { display: block; color: #5b6570; font-size: 12px; }
.profile-kpis strong { display: block; font-size: 23px; margin-top: 4px; color: #1f5f99; }
.profile-overview { width: 100%; }
.profile-overview caption { text-align: left; color: #5b6570; font-size: 13px; padding-bottom: 8px; }
.profile-overview th, .profile-overview td { vertical-align: middle; }
.profile-overview tbody th { min-width: 130px; max-width: 210px; white-space: normal; }
.profile-name { overflow-wrap: anywhere; }
.profile-caption { display: block; font-weight: normal; font-size: 11px;
  color: #5b6570; white-space: normal; margin-top: 3px; }
.profile-distribution { min-width: 210px; width: 230px; padding: 4px 6px; }
.profile-distribution svg { display: block; width: 100%; height: auto; }
.profile-details { margin-top: 24px; }
.profile-details summary { font-size: 17px; font-weight: 600; cursor: pointer; }
.profile-shape { width: 100%; }
.profile-scope { border-left: 3px solid #e07b39; padding: 6px 12px; font-size: 13px; }
.profile-correlations svg { display: block; max-width: 100%; height: auto; }
@media (max-width: 650px) { .profile-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media print { .profile-kpis, .profile-overview tr { break-inside: avoid; }
  .profile-distribution { min-width: 150px; width: 180px; }
  .profile-overview th, .profile-overview td { padding-left: 5px; padding-right: 5px; }
  .profile-correlations { break-inside: avoid; } }
"""
