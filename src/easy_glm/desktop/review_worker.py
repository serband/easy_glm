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
from easy_glm.desktop.fit_worker import result_for, write_json
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
            return ae_by_variable(frame, variable, actual, values, weight, **kwargs)

    table = aggregate(expected)
    fitted_table = aggregate(fitted_expected)
    table = table.with_columns(pl.Series("fitted_rate", fitted_table["expected_rate"]))
    if baseline is not None:
        before = totals(frame, run.config, baseline.predict(frame, exposure_col=None))[
            1
        ]
        table = table.with_columns(
            pl.Series("before_rate", aggregate(before)["expected_rate"])
        )
    return table.to_dicts()


def review(
    project: Project, run: ModelRun, raw: pl.DataFrame, request: dict[str, Any]
) -> dict[str, Any]:
    frame = prepare(project, raw)
    rebuild_rate_model(project, run, frame)
    train, holdout = train_holdout(frame, project.data.split)
    part = train if request.get("subset", "train") == "train" else holdout
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
        return {
            "rows": ae_detail(project, run, part, variable),
            "subset": request.get("subset", "train"),
        }
    actual, expected, weight = totals(train, run.config, run.predict(train))
    if action == "pair":
        a, b = request["a"], request["b"]
        if a == b or a not in available or b not in available:
            raise ValueError("Choose two different available variables.")
        aa, ee, ww = totals(part, run.config, run.predict(part))
        return {"rows": ae_by_pair(part, a, b, aa, ee, ww).to_dicts()}
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
                    dispersion=phi,
                ).to_dicts()
                if pairs
                else []
            )
        return {
            "rows": rows,
            "subset": "train",
            "note": "Ranked on training data only. Review the signal, then validate any refit on holdout data.",
        }
    before = run.rate_model.clone()
    config = project.models[run.name]
    before_adjustments = copy.deepcopy(config.adjustments)
    before_base = config.base_rate_override
    note = "Manual table adjustment."
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
                "moving": tooling.smooth_moving_average,
                "isotonic": tooling.smooth_isotonic,
                "cap": tooling.cap_floor,
                "round": tooling.round_relativities,
            }
            result = functions[action](table, variable, **options)
            values, note = list(result.values), result.note
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
            fitted = rate_model_for(project, run, [], base_rate_override=None)
            _, errors = apply_row_edits(
                config,
                variable,
                table.table,
                [r.relativity for r in fitted.variables[variable].table],
                values,
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
        old_rows = rate_model_tables(before)[variable].to_dicts()
        new_rows = rate_model_tables(run.rate_model)[variable].to_dicts()
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
        "changed": changed,
        "project": project.to_dict(),
        "result": result_for(project, frame, run),
        "rows": ae_detail(project, run, part, variable, before) if variable else [],
        "note": note,
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
        result = review(
            Project.from_json(folder / "project.json"),
            run,
            pl.read_parquet(source / "raw.parquet"),
            request,
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
