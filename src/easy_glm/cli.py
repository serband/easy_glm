"""``easy-glm`` — the command line for a workbench project.

Everything the workbench does to a project it can also do headlessly, which is
what a scheduled refit, a build server or a colleague without a browser needs::

    easy-glm run project.json                 # fit and write every artefact
    easy-glm run project.json --model freq_v2 --out artefacts/
    easy-glm export project.json --script     # just the Python script
    easy-glm export project.json --report --excel
    easy-glm validate project.json            # exit 1 and list the problems
    easy-glm workbench project.json           # open it in the browser

Every command that produces an artefact **fits the model first**, from the data
the project points at. That costs a fit even for ``--script``, and buys a script
with every knot, level and the resolved alpha written out — the same file the
Export page offers after a fit, rather than one that re-derives the design (and
re-runs cross-validation) whenever it is executed. Fits are not cached: the
workbench's ``.easyglm-runs`` folder belongs to a session, and a command line
that silently reused a fit from one would be the least predictable thing here.

Exit codes: ``0`` success, ``1`` a problem the user can act on (an invalid
project, an unreadable data file, a model that will not fit), ``2`` a usage
error from ``argparse``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import polars as pl

from easy_glm.workflow import (
    ModelRun,
    Project,
    prepare,
    run_model,
    to_report_html,
    to_script,
)
from easy_glm.workflow.project import safe_filename


class CliError(Exception):
    """A problem to report as a message and exit 1, never as a traceback."""

    def __init__(self, message: str, problems: list[str] | None = None) -> None:
        super().__init__(message)
        self.problems = problems or []

    def render(self) -> str:
        lines = [f"easy-glm: {self}"]
        lines += [f"  - {p}" for p in self.problems]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# loading and checking
# --------------------------------------------------------------------------
def open_project(path: str | Path) -> Project:
    """Read a project file, turning every failure into a :class:`CliError`."""
    p = Path(path)
    if not p.exists():
        raise CliError(f"no project file at {p}")
    try:
        return Project.from_json(p)
    except Exception as exc:  # noqa: BLE001 - the user gets the reason verbatim
        raise CliError(f"{p} is not a readable easy_glm project: {exc}") from exc


def prepared_frame(project: Project) -> pl.DataFrame:
    """The prepared frame (every data step, plus the split column)."""
    try:
        return prepare(project)
    except Exception as exc:  # noqa: BLE001
        raise CliError(f"cannot prepare the data: {exc}") from exc


def pick_model(project: Project, model: str | None) -> str:
    """``model``, or the champion, or the only model — else a clear error."""
    if not project.models:
        raise CliError("the project has no models")
    if model is not None:
        if model not in project.models:
            raise CliError(
                f"no model named {model!r} (the project has: "
                f"{', '.join(project.models)})"
            )
        return model
    if project.champion in project.models:
        return str(project.champion)
    return next(iter(project.models))


def check(project: Project, model: str | None = None) -> list[str]:
    """Everything wrong with the project, including its column references.

    The data is loaded so that a model naming a column the prepared frame does
    not have is reported here rather than at the first fit. A data file that
    cannot be read is itself one of the problems.
    """
    try:
        columns: list[str] | None = list(prepare(project).columns)
    except Exception as exc:  # noqa: BLE001
        return [f"the data cannot be prepared: {exc}"] + project.validate(model)
    return project.validate(model, columns=columns)


def fit(project: Project, model: str, df: pl.DataFrame) -> ModelRun:
    problems = check(project, model)
    if problems:
        raise CliError(f"model {model!r} cannot be fitted", problems)
    try:
        return run_model(project, df, model)
    except Exception as exc:  # noqa: BLE001
        raise CliError(f"fitting {model!r} failed: {exc}") from exc


def prefix(project: Project, model: str, out: Path) -> Path:
    """``<out>/<project>_<model>`` — the stem every artefact is named from."""
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_filename(project.name, 'project')}_{safe_filename(model)}"
    return out / stem


# --------------------------------------------------------------------------
# writing artefacts
# --------------------------------------------------------------------------
def write_easyglm(run: ModelRun, stem: Path) -> Path:
    # with_name, never with_suffix: a project called "motor v0.4" has a dot in
    # its stem and with_suffix would eat the "4"
    path = stem.with_name(f"{stem.name}.easyglm")
    run.rate_model.to_json(path)
    return path


def write_excel(run: ModelRun, stem: Path) -> Path:
    return run.rate_model.to_excel(stem.with_name(f"{stem.name}_rate_tables.xlsx"))


def write_script(project: Project, run: ModelRun, stem: Path) -> Path:
    path = stem.with_name(f"{stem.name}.py")
    path.write_text(to_script(project, run.name, run=run, output_prefix=stem.name))
    return path


def write_report(project: Project, run: ModelRun, df: pl.DataFrame, stem: Path) -> Path:
    path = stem.with_name(f"{stem.name}_report.html")
    path.write_text(
        to_report_html(project, {run.name: run}, df, champion=run.name),
        encoding="utf-8",
    )
    return path


def summary_lines(project: Project, run: ModelRun) -> list[str]:
    """The lines ``run`` prints: what was fitted and how well."""
    s = run.summary()
    rm = run.rate_model
    lines = [
        f"{project.name} · model {run.name} · {s['family']}"
        + (f" (link {rm.metadata.link})" if rm.metadata.link != "log" else ""),
        f"  rows          train {run.train_rows:,} · holdout {run.holdout_rows:,}",
        f"  penalty       alpha {s['alpha']:.6g} · {s['non_zero']} of "
        f"{s['features']} terms non-zero",
        f"  base rate     {rm.base_rate:.6g}  "
        f"(each table entry is a {rm.relativity_label})",
    ]
    if rm.metadata.offset_col:
        lines.append(f"  offset        {rm.metadata.offset_col}")
    for subset in ("train", "holdout"):
        m = run.metrics.get(subset)
        if not m:
            continue
        lines.append(
            f"  {subset:<13} A/E {_fmt(m.get('ae'))} · Gini {_fmt(m.get('gini'))} · "
            f"deviance explained {_pct(m.get('deviance_explained'))}"
        )
    if run.dropped_predictors:
        lines.append(
            "  left out      "
            + ", ".join(run.dropped_predictors)
            + " (constant or all-null on the training rows)"
        )
    return lines


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.2f}%"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    project = open_project(args.project)
    model = pick_model(project, args.model)
    df = prepared_frame(project)
    run = fit(project, model, df)
    stem = prefix(project, model, Path(args.out))
    written = [
        write_easyglm(run, stem),
        write_excel(run, stem),
        write_script(project, run, stem),
        write_report(project, run, df, stem),
    ]
    print("\n".join(summary_lines(project, run)))
    print("written:")
    for path in written:
        print(f"  {path}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    wanted = [
        name
        for name, on in (
            ("script", args.script),
            ("report", args.report),
            ("excel", args.excel),
        )
        if on
    ]
    if not wanted:
        raise CliError("pass at least one of --script, --report, --excel")
    project = open_project(args.project)
    model = pick_model(project, args.model)
    df = prepared_frame(project)
    run = fit(project, model, df)
    stem = prefix(project, model, Path(args.out))
    writers = {
        "script": lambda: write_script(project, run, stem),
        "report": lambda: write_report(project, run, df, stem),
        "excel": lambda: write_excel(run, stem),
    }
    for name in wanted:
        print(writers[name]())
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    project = open_project(args.project)
    problems = check(project, args.model)
    if problems:
        raise CliError(f"{args.project} has {len(problems)} problem(s)", problems)
    models = ", ".join(project.models) or "(none)"
    print(f"{args.project}: valid · models: {models}")
    return 0


def cmd_workbench(args: argparse.Namespace) -> int:
    if args.project is not None:
        open_project(args.project)  # fail here rather than in the browser
    try:
        from easy_glm.app import launch
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise CliError(
            f"the workbench needs Streamlit: pip install 'easy_glm[ui]' ({exc})"
        ) from exc
    proc = launch(
        args.project, port=args.port, block=True, headless=bool(args.headless)
    )
    return int(proc.returncode or 0)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="easy-glm",
        description="Fit, export and open easy_glm workbench projects.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def with_model(p: argparse.ArgumentParser) -> None:
        p.add_argument("project", help="path to the project JSON")
        p.add_argument(
            "--model",
            default=None,
            help="model to use (default: the project's champion)",
        )

    run_p = sub.add_parser(
        "run",
        help="fit a model and write the scorer, the Excel tables, the Python "
        "script and the HTML report",
    )
    with_model(run_p)
    run_p.add_argument("--out", default=".", help="output folder (default: .)")
    run_p.set_defaults(func=cmd_run)

    export_p = sub.add_parser("export", help="fit a model and write one artefact")
    with_model(export_p)
    export_p.add_argument("--script", action="store_true", help="the Python script")
    export_p.add_argument("--report", action="store_true", help="the HTML report")
    export_p.add_argument("--excel", action="store_true", help="the rate tables")
    export_p.add_argument("--out", default=".", help="output folder (default: .)")
    export_p.set_defaults(func=cmd_export)

    validate_p = sub.add_parser(
        "validate", help="check a project (and its data) without fitting"
    )
    with_model(validate_p)
    validate_p.set_defaults(func=cmd_validate)

    wb = sub.add_parser("workbench", help="open the project in the browser workbench")
    wb.add_argument("project", nargs="?", default=None, help="project JSON to open")
    wb.add_argument("--port", type=int, default=8501)
    wb.add_argument("--headless", action="store_true", help="do not open a browser tab")
    wb.set_defaults(func=cmd_workbench)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CliError as exc:
        print(exc.render(), file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("easy-glm: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    sys.exit(main())
