"""One self-contained HTML report for a fitted model (and an optional challenger).

:func:`to_report_html` returns a **single string**: one file, no companion
folder, no network access. Everything it needs — the stylesheet, every chart,
the tables and the exported Python script — is inside it, so it can be emailed,
attached to a rate filing or opened from a share years later.

Why the charts are SVG and not Plotly
-------------------------------------
The report has a hard budget: it must stay small enough to email (the release
plan's test is "under 5 MB for the fixture project") *and* be self-contained.
Inlining ``plotly.min.js`` alone costs 4.84 MB, which breaks that budget before
a single chart is drawn, and a CDN link would break self-containment. So the
charts are written directly as SVG (:mod:`easy_glm.workflow._svg`): the report
of the French-motor fixture comes to a few hundred kB, opens instantly, prints,
and — having no JavaScript at all — cannot produce a console error. Hover text
is a native SVG ``<title>``, so exposure, actual and expected are still one
mouse-over away.
"""

from __future__ import annotations

import html
import math
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl

from easy_glm.core.excel import interaction_matrices, rate_model_tables

from . import _svg
from .diagnostics import (
    ae_by_pair,
    ae_by_variable,
    base_rate_change,
    describe_diff,
    double_lift,
    gini,
    lift_table,
    relativity_diff,
    totals,
)
from .export import to_script
from .prep import train_holdout
from .project import Project
from .run import ModelRun

#: default tolerance of the relativity diff: a band has to move by more than
#: 1 % (|log ratio| > 0.01) before it is listed
DEFAULT_DIFF_TOL = 0.01

#: metrics shown side by side, as ``(row label, key in ModelRun.metrics, digits)``
_METRIC_ROWS = [
    ("rows", "rows", 0),
    ("exposure", "exposure", 0),
    ("actual", "actual", 1),
    ("expected", "expected", 1),
    ("A/E", "ae", 4),
    ("Gini (normalised)", "gini", 4),
    ("deviance explained", "deviance_explained", 4),
    ("mean deviance", "mean_deviance", 5),
]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(name)).strip("-").lower()


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        if digits == 0:
            return f"{value:,.0f}"
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        if value != 0 and abs(value) < 10.0**-digits:
            return f"{value:.2e}"  # a tiny slope must not print as 0.0000
        return f"{value:.{digits}f}"
    return str(value)


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    for name in ("easy_glm", "glum", "polars", "numpy"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - source checkout
            out[name] = "unknown"
    return out


def _table(frame: pl.DataFrame, *, digits: int = 4, max_rows: int = 400) -> str:
    """A polars frame as a plain HTML table (numbers right-aligned)."""
    if frame.is_empty():
        return '<p class="muted">Nothing to show.</p>'
    shown = frame.head(max_rows)
    numeric = {
        c
        for c, t in shown.schema.items()
        if t
        in (
            pl.Float32,
            pl.Float64,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt32,
            pl.UInt64,
        )
    }
    head = "".join(
        f'<th class="{"num" if c in numeric else ""}">{_esc(c)}</th>'
        for c in shown.columns
    )
    body = []
    for row in shown.iter_rows(named=True):
        cells = []
        for c in shown.columns:
            v = row[c]
            if c in numeric:
                cells.append(f'<td class="num">{_esc(_num(v, digits))}</td>')
            else:
                cells.append(f"<td>{_esc(v)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    more = (
        f'<p class="muted">First {max_rows} of {frame.height} rows.</p>'
        if frame.height > max_rows
        else ""
    )
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>{more}"
    )


def _pairs_table(pairs: list[tuple[str, Any]]) -> str:
    rows = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in pairs if v != ""
    )
    return f'<div class="scroll"><table class="kv"><tbody>{rows}</tbody></table></div>'


class _Scored:
    """A subset of the data with one model's totals on it."""

    def __init__(self, frame: pl.DataFrame, run: ModelRun) -> None:
        self.frame = frame
        self.empty = frame.is_empty()
        if self.empty:
            self.actual = self.expected = self.weight = np.zeros(0)
            return
        self.actual, self.expected, self.weight = totals(
            frame, run.config, run.predict(frame)
        )


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def _metrics_table(runs: dict[str, ModelRun], names: list[str]) -> str:
    """Metric rows × model/subset columns — the side-by-side comparison."""
    cols = [
        (name, subset)
        for name in names
        for subset in ("train", "holdout")
        if subset in runs[name].metrics
    ]
    head = "".join(f'<th class="num">{_esc(n)} · {s}</th>' for n, s in cols)
    body: list[str] = []
    for label, key, digits in _METRIC_ROWS:
        row = "".join(
            f'<td class="num">{_esc(_num(runs[n].metrics[s].get(key), digits))}</td>'
            for n, s in cols
        )
        body.append(f"<tr><th>{_esc(label)}</th>{row}</tr>")
    extra = [
        ("alpha", lambda r: _num(r.alpha, 6)),
        ("non-zero terms", lambda r: f"{int((r.fit.coef != 0).sum()):,}"),
        ("terms in the design", lambda r: f"{len(r.fit.coef):,}"),
        ("interactions", lambda r: str(len(r.config.interactions))),
        (
            "linear terms",
            lambda r: str(
                sum(1 for c in r.rate_model.variables.values() if c.type == "linear")
            ),
        ),
        ("manual adjustments", lambda r: str(len(r.config.adjustments))),
    ]
    for label, fn in extra:
        cells = []
        for name in names:
            value = fn(runs[name])
            span = sum(1 for n, _s in cols if n == name) or 1
            cells.append(f'<td class="num" colspan="{span}">{_esc(value)}</td>')
        body.append(f'<tr class="span"><th>{_esc(label)}</th>{"".join(cells)}</tr>')
    return (
        f'<div class="scroll"><table><thead><tr><th>metric</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _summary_section(
    project: Project,
    run: ModelRun,
    df: pl.DataFrame,
    train: pl.DataFrame,
    holdout: pl.DataFrame,
    runs: dict[str, ModelRun],
    names: list[str],
) -> str:
    cfg = run.config
    split = project.data.split
    data_pairs = [
        ("project", project.name),
        ("data file", project.data.source.path or "—"),
        ("format", project.data.source.type),
        ("rows after preparation", f"{df.height:,}"),
        ("columns", f"{df.width:,}"),
        ("recodes", f"{len(project.data.recodes)}"),
        ("derived columns", f"{len(project.data.derived)}"),
        ("row filters", f"{len(project.data.filters)}"),
        (
            "split",
            (
                f"random, {split.fraction:.0%} train, seed {split.seed}"
                if split.mode == "random"
                else f"column {split.column!r} (1 = train)"
            ),
        ),
        ("training rows", f"{train.height:,}"),
        ("holdout rows", f"{holdout.height:,}"),
    ]
    model_pairs = [
        ("champion", run.name),
        ("fitted at", run.created_at),
        ("family", cfg.family),
        ("link", run.fit.link),
        ("target", cfg.target),
        ("weight", cfg.weight or "—"),
        ("target divided by weight", "yes" if cfg.divide_target_by_weight else "no"),
        ("offset", cfg.offset or "—"),
        ("predictors", ", ".join(cfg.predictors)),
        (
            "interactions",
            ", ".join(f"{i.a} × {i.b}" for i in cfg.interactions) or "none",
        ),
        ("alpha", _num(run.alpha, 6)),
        (
            "non-zero terms",
            f"{int((run.fit.coef != 0).sum()):,} of {len(run.fit.coef):,}",
        ),
        ("base rate", _num(run.rate_model.base_rate, 6)),
        ("manual adjustments", f"{len(cfg.adjustments)}"),
    ]
    return (
        '<section id="summary"><h2>1. Summary</h2>'
        '<div class="cols">'
        f"<div><h3>Data</h3>{_pairs_table(data_pairs)}</div>"
        f"<div><h3>Model</h3>{_pairs_table(model_pairs)}</div>"
        "</div>"
        "<h3>Metrics</h3>"
        '<p class="muted">A/E is actual over expected on totals — 1.00 means the '
        "model charges exactly what happened. Gini is normalised (1.00 = the best "
        "possible ordering of these rows). Deviance explained is the share of the "
        "null deviance the model removes. The holdout column is the one to trust: "
        "those rows were not used to fit.</p>"
        f"{_metrics_table(runs, names)}"
        f"{_snapshot_note(runs, names)}"
        "</section>"
    )


def _relativity_chart(run: ModelRun, var: str, table: pl.DataFrame) -> str:
    cfg = run.rate_model.variables[var]
    if cfg.type == "linear":
        xs: list[float] = []
        ys: list[float] = []
        bands = table.filter(pl.col("from").is_not_null() & pl.col("to").is_not_null())
        for row in bands.sort("from").iter_rows(named=True):
            x0, x1 = float(row["from"]), float(row["to"])
            y0 = float(row["relativity"])
            y1 = float(row.get("relativity_to", y0))
            for k in range(13):
                t = k / 12
                xs.append(x0 + t * (x1 - x0))
                ys.append(y0 * (y1 / y0) ** t if y0 > 0 and y1 > 0 else y0)
        marks = []
        if cfg.x_base is not None:
            marks.append((float(cfg.x_base), "base (1.00)"))
        if xs:
            return _svg.curve_chart(
                [("relativity", xs, ys, _svg.BLUE)],
                x_title=var,
                marks=marks,
                title=f"{var}: fitted relativity curve",
            )
    labels = table["label"].to_list()
    lines = [("relativity", table["relativity"].to_list(), _svg.BLUE)]
    if "fitted" in table.columns:
        fitted = table["fitted"].to_list()
        if any(
            abs(float(f) - float(r)) > 1e-12
            for f, r in zip(fitted, table["relativity"].to_list(), strict=True)
        ):
            lines.insert(0, ("fitted (before adjustments)", fitted, _svg.GREY))
    return _svg.category_chart(
        labels,
        bars=None,
        lines=lines,
        right_title="relativity",
        hline=1.0,
        title=f"{var}: fitted relativities by band",
    )


def _ae_chart(
    frame: pl.DataFrame,
    var: str,
    scored: _Scored,
    knots: list[float] | None,
    challenger: np.ndarray | None,
    challenger_name: str,
    subset: str = "",
) -> str:
    table = ae_by_variable(
        frame, var, scored.actual, scored.expected, scored.weight, knots=knots
    )
    # a fourth element (dashed) is optional: _svg.category_chart reads it when
    # a challenger line is present
    lines: list[tuple] = [
        ("actual", table["actual_rate"].to_list(), _svg.BLUE),
        ("expected", table["expected_rate"].to_list(), _svg.ORANGE),
    ]
    if challenger is not None:
        other = ae_by_variable(
            frame, var, scored.actual, challenger, scored.weight, knots=knots
        )
        lines.append(
            (
                f"expected ({challenger_name})",
                other["expected_rate"].to_list(),
                _svg.GREEN,
                True,  # dashed, so it never hides the champion's line
            )
        )
    return _svg.category_chart(
        table["label"].to_list(),
        bars=table["exposure"].to_list(),
        lines=lines,
        right_title="rate",
        title=f"{var}: actual vs expected by band ({subset})",
    )


def _variable_sections(
    run: ModelRun,
    tables: dict[str, pl.DataFrame],
    subsets: dict[str, _Scored],
    challenger_pred: dict[str, np.ndarray] | None,
    challenger_name: str,
) -> str:
    out = ['<section id="variables"><h2>2. Rating factors</h2>']
    out.append(
        '<p class="muted">One block per factor: the fitted relativities, then '
        "actual against expected on the training rows and on the holdout, and the "
        "rate table itself. Read the A/E charts first — where the two lines part "
        "company with real exposure behind them, the model is not charging what "
        "the data shows.</p>"
    )
    for var in run.rate_model.variables:
        cfg = run.rate_model.variables[var]
        if cfg.type == "interaction":
            continue
        enc = run.spec[var] if var in run.spec.encoders else None
        knots = (
            enc.band_edges() if enc is not None and hasattr(enc, "band_edges") else None
        )
        blocks = [
            f'<section class="variable" id="var-{_slug(var)}">',
            f'<h3>{_esc(var)} <span class="tag">{_esc(cfg.type)}</span></h3>',
            "<h4>Relativities</h4>",
            _relativity_chart(run, var, tables[var]),
        ]
        for label, scored in subsets.items():
            if scored.empty:
                continue
            blocks.append(f"<h4>Actual vs expected — {_esc(label)}</h4>")
            blocks.append(
                _ae_chart(
                    scored.frame,
                    var,
                    scored,
                    knots,
                    (challenger_pred or {}).get(label),
                    challenger_name,
                    label,
                )
            )
        blocks.append("<h4>Rate table</h4>")
        blocks.append(_table(tables[var]))
        blocks.append("</section>")
        out.append("".join(blocks))
    out.append("</section>")
    return "".join(out)


def _pivot_pairs(table: pl.DataFrame) -> dict[str, Any]:
    """``ae_by_pair`` long table → matrices in the parents' row order."""
    rows = (
        table.select("label_a", "order_a").unique().sort("order_a")["label_a"].to_list()
    )
    cols = (
        table.select("label_b", "order_b").unique().sort("order_b")["label_b"].to_list()
    )
    ia = {lab: i for i, lab in enumerate(rows)}
    ib = {lab: j for j, lab in enumerate(cols)}
    ae: list[list[float | None]] = [[None] * len(cols) for _ in rows]
    hover = {
        name: [[0.0] * len(cols) for _ in rows]
        for name in ("actual", "expected", "exposure")
    }
    for r in table.iter_rows(named=True):
        i, j = ia[r["label_a"]], ib[r["label_b"]]
        v = r["ae"]
        ae[i][j] = float(v) if v is not None and v == v and r["expected"] > 0 else None
        for name in hover:
            hover[name][i][j] = float(r[name])
    return {"rows": rows, "cols": cols, "ae": ae, "hover": hover}


def _interaction_sections(
    run: ModelRun,
    tables: dict[str, pl.DataFrame],
    subsets: dict[str, _Scored],
) -> str:
    names = [v for v, c in run.rate_model.variables.items() if c.type == "interaction"]
    if not names:
        return ""
    out = [
        '<section id="interactions"><h2>3. Interactions</h2>',
        '<p class="muted">Cells multiply the two main effects; white is 1.00 '
        "(no adjustment) and grey is a cell with nothing in it. The heatmaps "
        "that follow are actual over expected in the same cells with the "
        "current tables — a block of one colour with real exposure is structure "
        "the model has not taken up.</p>",
    ]
    knots, levels = _knots_and_levels(run)
    for var in names:
        cfg = run.rate_model.variables[var]
        if cfg.parents is None:  # an interaction always records its two parents
            continue
        a, b = cfg.parents
        rows_a, rows_b, rel, exp = interaction_matrices(run.rate_model, var)
        out.append(f'<section class="variable" id="var-{_slug(var)}">')
        out.append(f"<h3>{_esc(var)}</h3>")
        out.append("<h4>Cell relativities</h4>")
        out.append(
            _svg.heatmap(
                rows_a,
                rows_b,
                rel,
                row_name=a,
                col_name=b,
                hover={"training exposure": exp},
                title=f"{var}: cell relativities",
            )
        )
        for label, scored in subsets.items():
            if scored.empty or a not in scored.frame.columns:
                continue
            pair = ae_by_pair(
                scored.frame,
                a,
                b,
                scored.actual,
                scored.expected,
                scored.weight,
                knots_a=knots.get(a),
                knots_b=knots.get(b),
                levels_a=levels.get(a),
                levels_b=levels.get(b),
            )
            m = _pivot_pairs(pair)
            out.append(f"<h4>Actual / expected by cell — {_esc(label)}</h4>")
            out.append(
                _svg.heatmap(
                    m["rows"],
                    m["cols"],
                    m["ae"],
                    row_name=a,
                    col_name=b,
                    hover=m["hover"],
                    title=f"{var}: actual / expected by cell ({label})",
                )
            )
        out.append("<h4>Cells that carry an adjustment</h4>")
        adjusted = tables[var].filter((pl.col("relativity") - 1.0).abs() > 1e-9)
        out.append(
            f'<p class="muted">{adjusted.height} of {tables[var].height} cells; '
            "the rest are 1.00 (too little training exposure, or the penalty "
            "removed them).</p>"
        )
        out.append(_table(adjusted.sort("exposure", descending=True), max_rows=150))
        out.append("</section>")
    out.append("</section>")
    return "".join(out)


def _knots_and_levels(run: ModelRun) -> tuple[dict, dict]:
    knots: dict[str, list[float]] = {}
    levels: dict[str, list[str]] = {}
    for v in run.spec.main_effects:
        enc = run.spec[v]
        if hasattr(enc, "band_edges"):
            knots[v] = enc.band_edges()
        elif hasattr(enc, "levels"):
            levels[v] = list(enc.levels)
    return knots, levels


def _lift_section(subsets: dict[str, _Scored], number: int) -> str:
    out = [
        f'<section id="lift"><h2>{number}. Lift and Gini</h2>',
        '<p class="muted">Policies are put in ten equal-exposure bins ordered by '
        "the predicted rate, cheapest first. A model that separates risk shows an "
        "actual line that climbs with the expected line; where the two part, the "
        "model is mis-charging that end of the book.</p>",
    ]
    for label, scored in subsets.items():
        if scored.empty:
            continue
        table = lift_table(scored.actual, scored.expected, scored.weight)
        g = gini(scored.actual, scored.expected, scored.weight)
        out.append(
            f"<h3>{_esc(label)} — normalised Gini {_esc(_num(float(g), 4))}</h3>"
        )
        out.append(
            _svg.category_chart(
                [str(b) for b in table["bin"].to_list()],
                bars=table["exposure"].to_list(),
                lines=[
                    ("actual", table["actual_rate"].to_list(), _svg.BLUE),
                    ("expected", table["expected_rate"].to_list(), _svg.ORANGE),
                ],
                right_title="rate",
                title=f"Lift ({label}): actual and expected by predicted-rate bin",
            )
        )
        out.append(_table(table))
    out.append("</section>")
    return "".join(out)


def _compare_section(
    runs: dict[str, ModelRun],
    champion: str,
    challenger: str,
    subsets: dict[str, _Scored],
    challenger_pred: dict[str, np.ndarray],
    number: int,
) -> str:
    diff = describe_diff(
        relativity_diff(runs[champion], runs[challenger], DEFAULT_DIFF_TOL),
        champion,
        challenger,
    )
    out = [
        f'<section id="compare"><h2>{number}. {_esc(champion)} vs '
        f"{_esc(challenger)}</h2>",
        '<p class="muted">The metrics of both models are in section 1. Below: the '
        "double lift, which sorts the book by how much cheaper one model is than "
        "the other and asks which of the two got those policies right, and then "
        "every band whose relativity moved materially.</p>",
    ]
    for label, scored in subsets.items():
        if scored.empty or label not in challenger_pred:
            continue
        table = double_lift(
            scored.actual, scored.expected, challenger_pred[label], scored.weight
        )
        out.append(f"<h3>Double lift — {_esc(label)}</h3>")
        out.append(
            _svg.category_chart(
                [str(b) for b in table["bin"].to_list()],
                bars=table["exposure"].to_list(),
                lines=[
                    (f"A/E {champion}", table["ae_a"].to_list(), _svg.BLUE),
                    (f"A/E {challenger}", table["ae_b"].to_list(), _svg.ORANGE, True),
                ],
                right_title="A/E",
                hline=1.0,
                right_from_zero=False,
                title=f"Double lift ({label}): {champion} against {challenger}",
            )
        )
    out.append("<h3>Relativities that differ</h3>")
    out.append(_level_headline(runs[champion], runs[challenger], challenger))
    out.append(
        '<p class="muted">One row per band whose relativity moved by more than '
        f"{DEFAULT_DIFF_TOL:.0%} (|log ratio| &gt; {DEFAULT_DIFF_TOL:g}). "
        "<code>log_diff</code> is log(challenger / champion): +0.10 means "
        f"<em>{_esc(challenger)}</em> charges about 10 % more for that band "
        "<em>on top of</em> the overall level above — a band's premium change "
        "is its relativity change multiplied by that level change, so the two "
        "must be read together. Numeric factors are compared on the union of "
        "both models' band edges, so a moved knot (or the same factor banded in "
        "one model and a straight line in the other, where <code>kind</code> "
        "reads <em>numeric → linear</em>) is still compared like for like; "
        "levels and interaction cells are matched by name. Bands or variables "
        "only one model has are listed too.</p>"
    )
    out.append(_table(diff, max_rows=300))
    out.append("</section>")
    return "".join(out)


def _level_headline(run_a: ModelRun, run_b: ModelRun, challenger: str) -> str:
    """The overall level change on its own, above the band-by-band table: a
    band's premium change is its relativity change times this one."""
    change = base_rate_change(run_a, run_b)
    if change is None:
        return ""
    direction = "no change" if abs(change) < 5e-5 else f"{change:+.1%}"
    return (
        f'<p class="headline">Overall level (base rate): <strong>{direction}'
        f"</strong> — every risk pays that much more or less with "
        f"<em>{_esc(challenger)}</em> before its own bands move.</p>"
    )


def _no_comparison_section(
    other: ModelRun, challenger: str, df: pl.DataFrame, number: int
) -> str:
    """Stand in for the comparison when the challenger cannot be scored on
    these rows — a named challenger and no comparison would read as a bug."""
    missing = [c for c in other.spec.required_columns if c not in df.columns]
    why = (
        "the prepared data no longer has the columns it needs ("
        + ", ".join(missing)
        + ")"
        if missing
        else "there are no rows to score it on"
    )
    return (
        f'<section id="compare"><h2>{number}. No comparison with '
        f"{_esc(challenger)}</h2>"
        f'<p class="headline">This report names <strong>{_esc(challenger)}</strong> '
        f"as the challenger, but it could not be scored here: {_esc(why)}. Its "
        "metrics in section 1 are the ones recorded when it was fitted; there is "
        "no double lift and no relativity comparison in this file.</p></section>"
    )


def _appendix(project: Project, champion: str, run: ModelRun, number: int) -> str:
    try:
        script = to_script(project, champion, run=run)
    except Exception as exc:  # noqa: BLE001 - the report must still render
        script = f"# The script could not be rendered: {exc}"
    coefs = run.fit.coef_table()
    return (
        f'<section id="appendix"><h2>{number}. Appendix</h2>'
        "<h3>Coefficients</h3>"
        '<p class="muted">The fitted GLM coefficients on the log scale, one per '
        "design column. A zero coefficient is a term the penalty removed.</p>"
        f"{_table(coefs, digits=6, max_rows=1000)}"
        "<h3>The model as a Python script</h3>"
        '<p class="muted">Running this file rebuilds the model, its rate tables '
        "and the <code>.easyglm</code> scorer, using only the public easy_glm "
        "API — the reproducible record of what was fitted.</p>"
        f"<pre>{_esc(script)}</pre>"
        "</section>"
    )


_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f6f7f9; color: #1c2530;
  font: 15px/1.55 "Helvetica Neue", Helvetica, Arial, sans-serif; }
.page { max-width: 1080px; margin: 0 auto; padding: 32px 24px 80px; }
h1 { font-size: 27px; margin: 0 0 4px; }
h2 { font-size: 21px; margin: 40px 0 10px; padding-bottom: 6px;
  border-bottom: 2px solid #1f5f99; }
h3 { font-size: 17px; margin: 26px 0 8px; }
h4 { font-size: 14px; margin: 18px 0 4px; color: #5b6570;
  text-transform: uppercase; letter-spacing: .04em; }
p { margin: 8px 0; }
.muted { color: #5b6570; font-size: 13.5px; max-width: 78ch; }
.headline { background: #eef3f8; border-left: 3px solid #1f5f99; padding: 8px 12px;
  margin: 10px 0; max-width: 78ch; }
.sub { color: #5b6570; margin: 0 0 18px; }
section.variable { background: #fff; border: 1px solid #e2e7ec; border-radius: 8px;
  padding: 14px 18px 20px; margin: 18px 0; }
section.variable h3 { margin-top: 4px; }
.tag { font-size: 12px; font-weight: normal; color: #5b6570;
  border: 1px solid #d5dbe1; border-radius: 10px; padding: 1px 8px; }
.chart { display: block; background: #fff; margin: 6px 0 2px; }
.chart .ax { font: 11px "Helvetica Neue", Helvetica, Arial, sans-serif; fill: #5b6570; }
.chart .lg { font: 12px "Helvetica Neue", Helvetica, Arial, sans-serif; fill: #1c2530; }
.scroll { overflow-x: auto; margin: 8px 0 4px; }
table { border-collapse: collapse; font-size: 13px; background: #fff; }
th, td { border: 1px solid #e2e7ec; padding: 4px 9px; text-align: left;
  white-space: nowrap; }
thead th { background: #eef1f4; position: sticky; top: 0; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
table.kv th { background: #f4f6f8; font-weight: 600; width: 210px; }
table.kv td { white-space: normal; word-break: break-word; max-width: 460px; }
tr.span td { background: #fbfcfd; }
pre { background: #fff; border: 1px solid #e2e7ec; border-radius: 6px;
  padding: 14px; overflow-x: auto; font-size: 12.5px; line-height: 1.45; }
code { background: #eef1f4; padding: 1px 4px; border-radius: 3px; font-size: 12.5px; }
.cols { display: flex; gap: 24px; flex-wrap: wrap; }
.cols > div { flex: 1 1 380px; min-width: 0; }
nav.toc { background: #fff; border: 1px solid #e2e7ec; border-radius: 8px;
  padding: 10px 18px; margin: 18px 0 4px; }
nav.toc a { color: #1f5f99; text-decoration: none; margin-right: 18px;
  display: inline-block; padding: 2px 0; }
nav.toc a:hover { text-decoration: underline; }
footer { margin-top: 44px; padding-top: 14px; border-top: 1px solid #e2e7ec;
  color: #5b6570; font-size: 12.5px; }
@media print { body { background: #fff; } .page { max-width: none; padding: 0; }
  section.variable { break-inside: avoid; } }
"""


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
def to_report_html(
    project: Project,
    runs: dict[str, ModelRun],
    df: pl.DataFrame,
    *,
    champion: str,
    challenger: str | None = None,
) -> str:
    """One self-contained HTML page describing ``runs[champion]``.

    ``df`` is the prepared frame (with the split column); it is split into
    training and holdout rows exactly as the fit did. Sections: summary (project,
    data, split, metrics), one block per rating factor (relativities, actual vs
    expected on train and holdout, the rate table), interaction heatmaps, lift
    and Gini, an appendix with the coefficients and the exported Python script —
    and, when ``challenger`` names a second fitted model, a comparison section
    with the double lift and every relativity that differs.

    The returned string is the whole file: inline stylesheet, charts as SVG, no
    external requests (see the module docstring).
    """
    if champion not in runs:
        raise KeyError(f"No run for the champion {champion!r}")
    if challenger is not None and challenger not in runs:
        raise KeyError(f"No run for the challenger {challenger!r}")
    if challenger == champion:
        challenger = None
    run = runs[champion]

    train, holdout = train_holdout(df, project.data.split)
    subsets = {"train": _Scored(train, run), "holdout": _Scored(holdout, run)}
    names = [champion] + ([challenger] if challenger else [])

    challenger_pred: dict[str, np.ndarray] = {}
    if challenger:
        other = runs[challenger]
        missing = [c for c in other.spec.required_columns if c not in df.columns]
        if not missing:
            for label, scored in subsets.items():
                if scored.empty:
                    continue
                challenger_pred[label] = totals(
                    scored.frame, other.config, other.predict(scored.frame)
                )[1]

    tables = rate_model_tables(run.rate_model)
    inter = _interaction_sections(run, tables, subsets)
    number = 4 if inter else 3
    body = [
        _summary_section(project, run, df, train, holdout, runs, names),
        _variable_sections(
            run, tables, subsets, challenger_pred or None, challenger or ""
        ),
        inter,
        _lift_section(subsets, number),
    ]
    number += 1
    if challenger and challenger_pred:
        body.append(
            _compare_section(
                runs, champion, challenger, subsets, challenger_pred, number
            )
        )
        number += 1
    elif challenger:
        # the challenger is named in the subtitle and has a metrics column, so
        # the missing comparison must be explained rather than silently absent
        body.append(_no_comparison_section(runs[challenger], challenger, df, number))
        number += 1
    body.append(_appendix(project, champion, run, number))

    toc = [
        ("#summary", "Summary"),
        ("#variables", "Rating factors"),
        *([("#interactions", "Interactions")] if inter else []),
        ("#lift", "Lift and Gini"),
        *([("#compare", f"{champion} vs {challenger}")] if challenger else []),
        ("#appendix", "Appendix"),
    ]
    nav = "".join(f'<a href="{href}">{_esc(text)}</a>' for href, text in toc)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    versions = " · ".join(f"{k} {v}" for k, v in _versions().items())
    subtitle = f"Champion <strong>{_esc(champion)}</strong>"
    if challenger:
        subtitle += f" · challenger <strong>{_esc(challenger)}</strong>"
    return (
        "<!doctype html>\n"
        f'<html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(project.name)} — model report</title>"
        f'<style>{_CSS}</style></head><body><div class="page">'
        f"<h1>{_esc(project.name)} — model report</h1>"
        f'<p class="sub">{subtitle} · generated {_esc(generated)}</p>'
        f'<nav class="toc">{nav}</nav>'
        f"{''.join(body)}"
        f"<footer>Generated by easy_glm on {_esc(generated)}. "
        f"{_esc(versions)}.</footer>"
        "</div></body></html>"
    )


def _snapshot_note(runs: dict[str, ModelRun], names: list[str]) -> str:
    """Metrics recorded on the rate models' snapshots, when there are any."""
    rows: list[str] = []
    for name in names:
        for snap in runs[name].rate_model.snapshots:
            if not snap.metrics:
                continue
            for subset, m in snap.metrics.items():
                if not isinstance(m, dict):
                    continue
                rows.append(
                    f"<tr><td>{_esc(name)}</td><td>v{snap.version}</td>"
                    f"<td>{_esc(snap.description)}</td><td>{_esc(subset)}</td>"
                    f'<td class="num">{_esc(_num(m.get("ae"), 4))}</td>'
                    f'<td class="num">{_esc(_num(m.get("gini"), 4))}</td></tr>'
                )
    if not rows:
        return ""
    return (
        "<h3>Saved versions</h3>"
        '<p class="muted">Metrics recorded with each saved version of the rate '
        "tables (a version is written whenever manual adjustments are applied).</p>"
        '<div class="scroll"><table><thead><tr><th>model</th><th>version</th>'
        '<th>description</th><th>rows</th><th class="num">A/E</th>'
        f'<th class="num">Gini</th></tr></thead><tbody>{"".join(rows)}</tbody>'
        "</table></div>"
    )
