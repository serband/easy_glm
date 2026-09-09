"""On-demand diagnostics and table previews, using the original fit without refitting."""

from __future__ import annotations

import copy
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import polars as pl

from easy_glm.app.grids import apply_cell_edits, apply_row_edits, cell_grid
from easy_glm.desktop.fit_worker import result_for, table_payload, write_json
from easy_glm.engine import tooling
from easy_glm.engine.rate_model import RateModel
from easy_glm.workflow.diagnostics import (
    ae_by_pair,
    ae_by_variable,
    expected_claims,
    pearson_dispersion,
    residual_factor_search,
    residual_pair_search,
    totals,
)
from easy_glm.workflow.prep import prepare, train_holdout
from easy_glm.workflow.project import Project
from easy_glm.workflow.run import (
    ModelRun,
    rate_model_for,
    rebalance_override,
    rebuild_rate_model,
)


def grouping(run: ModelRun, variable: str) -> dict[str, Any]:
    if variable not in run.spec.main_effects:
        return {}
    enc = run.spec[variable]
    return {
        "knots": enc.band_edges() if hasattr(enc, "band_edges") else None,
        "fitted_levels": list(enc.levels) if hasattr(enc, "levels") else None,
        "fitted_labels": run.tables[variable]["label"].to_list(),
        "other_label": run.rate_model.variables[variable].other_label,
    }


def ae_detail(
    project: Project,
    run: ModelRun,
    frame: pl.DataFrame,
    variable: str,
    baseline: RateModel | None = None,
    challenger: ModelRun | None = None,
    n_bins: int = 20,
) -> list[dict[str, Any]]:
    actual, expected, weight = totals(frame, run.config, run.predict(frame))
    fitted = rate_model_for(project, run, [], base_rate_override=None)
    fitted_expected = totals(
        frame, run.config, fitted.predict(frame, exposure_col=None)
    )[1]
    kwargs = grouping(run, variable)
    cfg = run.rate_model.variables.get(variable)
    if cfg and cfg.type == "interaction":
        a, b = cfg.parents

        ga, gb = grouping(run, a), grouping(run, b)

        def aggregate(values: Any) -> pl.DataFrame:
            return ae_by_pair(
                frame,
                a,
                b,
                actual,
                values,
                weight,
                knots_a=ga.get("knots"),
                knots_b=gb.get("knots"),
                levels_a=ga.get("fitted_levels"),
                levels_b=gb.get("fitted_levels"),
            )

    else:

        def aggregate(values: Any) -> pl.DataFrame:
            return ae_by_variable(
                frame, variable, actual, values, weight, n_bins=n_bins, **kwargs
            )

    table = aggregate(expected)
    fitted_table = aggregate(fitted_expected)
    table = table.with_columns(
        pl.Series("fitted_rate", fitted_table["expected_rate"]),
        pl.Series("fitted_ae", fitted_table["ae"]),
        pl.Series("fitted_expected", fitted_table["expected"]),
    )
    if challenger is not None:
        from easy_glm.desktop.diagnostic_views import compatible

        compatible(run, challenger)
        other = totals(frame, challenger.config, challenger.predict(frame))[1]
        table = table.with_columns(
            pl.Series("challenger_rate", aggregate(other)["expected_rate"])
        )
    if baseline is not None:
        before = totals(frame, run.config, baseline.predict(frame, exposure_col=None))[
            1
        ]
        before_table = aggregate(before)
        table = table.with_columns(
            pl.Series("before_rate", before_table["expected_rate"]),
            pl.Series("before_ae", before_table["ae"]),
            pl.Series("before_expected", before_table["expected"]),
        )
    return table.to_dicts()


def review(
    project: Project,
    run: ModelRun,
    raw: pl.DataFrame,
    request: dict[str, Any],
    challenger: ModelRun | None = None,
) -> dict[str, Any]:
    frame = prepare(project, raw)
    rebuild_rate_model(project, run, frame)
    if challenger is not None:
        rebuild_rate_model(project, challenger, frame)
        from easy_glm.desktop.diagnostic_views import compatible

        compatible(run, challenger)
    train, holdout = train_holdout(frame, project.data.split)
    part = {"train": train, "holdout": holdout, "all": frame}[
        request.get("subset", "train")
    ]
    if request["action"] in ("lift", "double_lift", "path", "coefficients", "compare"):
        from easy_glm.desktop.diagnostic_views import view

        return view(project, run, frame, request, challenger)
    if part.is_empty():
        raise ValueError("The selected subset has no rows.")
    action = request["action"]
    variable = request.get("variable")
    available = [
        c
        for c in frame.columns
        if project.data.roles.get(c)
        not in (
            "target",
            "weight",
            "exposure",
            "offset",
            "current_premium",
            "id",
            "split",
            "ignore",
        )
        and c
        not in (
            project.data.split.column,
            run.config.target,
            run.config.weight,
            run.config.offset,
        )
    ]
    if action == "variable":
        if variable not in frame.columns and variable not in run.rate_model.variables:
            raise ValueError("Choose an available variable.")
        both = request.get("options", {}).get("both_subsets", False)
        selected_subset = request.get("subset", "train")
        if both and selected_subset == "all":
            part, selected_subset = train, "train"
        sets = []
        if both:
            for label, data in (("train", train), ("holdout", holdout)):
                if not data.is_empty() and label != selected_subset:
                    sets.append(
                        {
                            "title": f"{variable} · {label}",
                            "rows": ae_detail(
                                project,
                                run,
                                data,
                                variable,
                                challenger=challenger,
                                n_bins=request.get("n_bins", 20),
                            ),
                        }
                    )
        cfg = run.rate_model.variables.get(variable)
        kind = (
            cfg.type
            if cfg
            else ("numeric" if frame.schema[variable].is_numeric() else "categorical")
        )
        return {
            "rows": ae_detail(
                project,
                run,
                part,
                variable,
                challenger=challenger,
                n_bins=request.get("n_bins", 20),
            ),
            "book_impact": {
                "current": expected_claims(run.rate_model, train, run.config),
                "fitted": expected_claims(
                    rate_model_for(project, run, [], base_rate_override=None),
                    train,
                    run.config,
                ),
            },
            "ae_sets": sets,
            "kind": kind,
            "subset": selected_subset,
        }

    actual, expected, weight = totals(train, run.config, run.predict(train))
    if action == "pair":
        a, b = request["a"], request["b"]
        if a == b or a not in available or b not in available:
            raise ValueError("Choose two different available variables.")
        aa, ee, ww = totals(part, run.config, run.predict(part))
        ga, gb = grouping(run, a), grouping(run, b)
        search_preview = request.get("options", {}).get("search_preview", False)
        args = {
            "n_bins": 8 if search_preview else request.get("n_bins", 8),
            "knots_a": None if search_preview else ga.get("knots"),
            "knots_b": None if search_preview else gb.get("knots"),
            "levels_a": ga.get("fitted_levels"),
            "levels_b": gb.get("fitted_levels"),
        }
        rows = ae_by_pair(part, a, b, aa, ee, ww, **args)
        if challenger is not None:
            other = totals(part, challenger.config, challenger.predict(part))[1]
            other_table = ae_by_pair(part, a, b, aa, other, ww, **args)
            rows = rows.with_columns(
                pl.Series("challenger_ae", other_table["ae"]),
                pl.Series("challenger_expected", other_table["expected"]),
            )
        return {"rows": rows.to_dicts()}
    if action in ("factors", "interactions"):
        phi = (
            1.0
            if run.config.family == "poisson"
            else pearson_dispersion(actual, expected, len(run.fit.coef))
        )
        if action == "factors":
            candidates = [c for c in available if c not in run.config.predictors]
            rows = residual_factor_search(
                train, candidates, actual, expected, weight, dispersion=phi
            ).to_dicts()
        else:
            predictors = run.config.predictors
            existing = {frozenset((i.a, i.b)) for i in run.config.interactions}
            pairs = [
                (a, b)
                for i, a in enumerate(predictors)
                for b in predictors[i + 1 :]
                if frozenset((a, b)) not in existing
            ]
            rows = (
                residual_pair_search(
                    train,
                    predictors,
                    actual,
                    expected,
                    weight,
                    pairs=pairs,
                    levels={
                        v: list(run.spec[v].levels)
                        for v in run.spec.main_effects
                        if hasattr(run.spec[v], "levels")
                    },
                    dispersion=phi,
                ).to_dicts()
                if pairs
                else []
            )
        return {
            "rows": rows,
            "subset": "train",
            "note": "Ranked on training data only. Review the signal, then validate any refit on holdout data."
            + (
                " Non-Poisson model: signal is scaled by Pearson dispersion "
                + str(round(phi, 4))
                + "; interpret it as a ranking, not a calibrated z-score."
                if run.config.family != "poisson"
                else " Signal is a noise-adjusted Pearson excess z-score; 2 or higher is a review starting point."
            ),
        }
    if action == "compare_snapshots":
        from easy_glm.workflow.diagnostics import describe_diff, rate_model_diff

        options = request.get("options", {})

        def version(choice):
            cfg = project.models[run.name]
            if choice == "__fitted__":
                return rate_model_for(project, run, [], base_rate_override=None)
            if choice == "__current__":
                return run.rate_model
            saved = next((s for s in cfg.snapshots if s.name == choice), None)
            if saved is None:
                raise ValueError("Choose a saved table snapshot.")
            return rate_model_for(
                project,
                run,
                saved.adjustments,
                base_rate_override=saved.base_rate_override,
            )

        left, right = options.get("left", "__fitted__"), options.get(
            "right", "__current__"
        )
        first, second = version(left), version(right)
        diff = rate_model_diff(first, second, request.get("tolerance", 0.01))
        titles = {"__fitted__": "Original fitted", "__current__": "Current tables"}
        return {
            "tables": [
                {
                    "title": "Snapshot differences",
                    "rows": describe_diff(
                        diff, titles.get(left, left), titles.get(right, right)
                    ).to_dicts(),
                }
            ],
            "note": (
                "The versions charge the same premium within the selected tolerance."
                if diff.is_empty()
                else "Table versions compared without refitting. Band changes include the base-rate change."
            ),
        }
    before = run.rate_model.clone()
    original = rate_model_for(project, run, [], base_rate_override=None)
    config = project.models[run.name]
    before_adjustments = copy.deepcopy(config.adjustments)
    before_base = config.base_rate_override
    note = "Manual table adjustment."
    tool_details = None
    if action in ("edit", "moving", "isotonic", "cap", "round"):
        if variable not in run.rate_model.variables:
            raise ValueError("Choose a fitted table.")
        table = run.rate_model.variables[variable]
        values = [float(r.relativity) for r in table.table]
        if action == "edit":
            for key, value in request.get("edits", {}).items():
                index = int(key)
                if (
                    not 0 <= index < len(values)
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                ):
                    raise ValueError(
                        "Every edited relativity must be a finite positive number in an existing row."
                    )
                values[index] = float(value)
        else:
            options = request.get("options", {})
            functions = {
                "moving": tooling.smooth_trailing_average,
                "isotonic": tooling.smooth_isotonic,
                "cap": tooling.cap_floor,
                "round": tooling.round_relativities,
            }
            result = functions[action](
                original.variables[variable], variable, **options
            )
            values, note = list(result.values), result.note
            # Tools replace the factor overlay, but excluded Null / Other rows
            # retain any explicit manual adjustment.
            included = {i for group in tooling.groups(table) for i in group}
            for i, row in enumerate(table.table):
                if i not in included:
                    values[i] = float(row.relativity)
            tool_details = {
                "name": result.tool,
                "log_mean_before": result.log_mean_before,
                "log_mean_after": result.log_mean_after,
                "uniform_weights": result.uniform_weights,
            }
            if result.uniform_weights:
                note += " No training exposure is stored on this table; each band has equal smoothing weight."
        if table.type == "interaction":
            grid = cell_grid(run.rate_model, variable)
            edited = copy.deepcopy(grid["current"])
            lookup = {
                row.key: value for row, value in zip(table.table, values, strict=True)
            }
            current_values = {row.key: float(row.relativity) for row in table.table}
            for i, keys in enumerate(grid["keys"]):
                for j, key in enumerate(keys):
                    if key in lookup and lookup[key] != current_values[key]:
                        edited[i][j] = lookup[key]
            _, errors = apply_cell_edits(config, variable, grid, edited)
        else:
            original_rows = original.variables[variable].table
            indices = list(range(len(table.table)))
            edit_rows = table.table
            if action != "edit":
                indices = sorted(included)
                keys = {(original_rows[i].from_, original_rows[i].to_) for i in indices}
                config.adjustments = [
                    adj
                    for adj in config.adjustments
                    if not (
                        adj.variable == variable
                        and not adj.cell
                        and (adj.from_, adj.to_) in keys
                    )
                ]
                edit_rows = original_rows
            _, errors = apply_row_edits(
                config,
                variable,
                [edit_rows[i] for i in indices],
                [original_rows[i].relativity for i in indices],
                [values[i] for i in indices],
                require_positive=True,
                other_label=table.other_label,
            )
        if errors:
            raise ValueError("; ".join(errors))
    elif action == "rebalance":
        if run.fit.link == "logit":
            raise ValueError(
                "A probability model cannot be rebalanced by scaling its base rate."
            )
        config.base_rate_override = rebalance_override(project, run, frame)
        if config.base_rate_override is None:
            raise ValueError("No positive training total is available to rebalance.")
        note = "Restore fitted total expected claims by changing only the base rate."
    elif action == "restore":
        restored = Project.from_dict(request["restore_project"]).models[run.name]
        config.adjustments = restored.adjustments
        config.base_rate_override = restored.base_rate_override
        note = "Restore the selected table adjustments and base rate."
    else:
        raise ValueError("Unknown diagnostic or table action.")
    rebuild_rate_model(project, run, frame)
    after = expected_claims(run.rate_model, train, config)
    previous = expected_claims(before, train, config)
    fitted = expected_claims(
        rate_model_for(project, run, [], base_rate_override=None), train, config
    )
    changed = (
        config.adjustments != before_adjustments
        or config.base_rate_override != before_base
    )
    if not changed:
        note += " Nothing would change."
    from easy_glm.core.excel import rate_model_tables

    changes = []
    if variable:
        old_rows = table_payload(before, variable, rate_model_tables(before)[variable])[
            "rows"
        ]
        new_rows = table_payload(
            run.rate_model, variable, rate_model_tables(run.rate_model)[variable]
        )["rows"]
        for i, (old, new) in enumerate(zip(old_rows, new_rows, strict=True)):
            if old["relativity"] != new["relativity"] or old.get("slope") != new.get(
                "slope"
            ):
                changes.append(
                    {
                        "row": i + 1,
                        "label": old.get(
                            "label",
                            str(old.get("label_a", ""))
                            + " × "
                            + str(old.get("label_b", "")),
                        ),
                        "before": old["relativity"],
                        "after": new["relativity"],
                        "before_slope": old.get("slope"),
                        "after_slope": new.get("slope"),
                    }
                )
    return {
        "changes": changes,
        "tool_details": tool_details,
        "kind": run.rate_model.variables[variable].type if variable else None,
        "preview_table": (
            {
                "columns": list(new_rows[0]) if new_rows else [],
                "rows": [
                    dict(new, fitted=fit_row.relativity)
                    for fit_row, new in zip(
                        original.variables[variable].table, new_rows, strict=True
                    )
                ],
                "kind": (
                    "step"
                    if run.rate_model.variables[variable].type == "numeric"
                    else run.rate_model.variables[variable].type
                ),
                "offset": 0,
                "total": len(new_rows),
            }
            if variable
            else None
        ),
        "changed": changed,
        "project": project.to_dict(),
        "result": result_for(project, frame, run),
        "rows": ae_detail(project, run, part, variable, before) if variable else [],
        "note": note,
        "before_base_rate": before.base_rate,
        "after_base_rate": run.rate_model.base_rate,
        "before_expected": previous,
        "after_expected": after,
        "fitted_expected": fitted,
        "change": after / previous - 1 if previous else None,
    }


def main() -> None:
    folder = Path(sys.argv[1])
    request = json.loads((folder / "request.json").read_text())
    source = Path(
        sys.argv[2]
    )  # internal path supplied by the server, never an API path
    try:
        with (source / "fit.pkl").open("rb") as handle:
            run = pickle.load(handle)  # only our own private temporary worker artifact
        challenger = None
        if request.get("_challenger_source"):
            with (Path(request["_challenger_source"]) / "fit.pkl").open("rb") as handle:
                challenger = pickle.load(handle)
        result = review(
            Project.from_json(folder / "project.json"),
            run,
            pl.read_parquet(source / "raw.parquet"),
            request,
            challenger,
        )
    except Exception as exc:
        result = {"error": str(exc)}
    if "result" in result:
        result["result"]["warnings"] = json.loads(
            (source / "result.json").read_text()
        ).get("warnings", [])
    write_json(folder / "result.json", result)


if __name__ == "__main__":
    main()
