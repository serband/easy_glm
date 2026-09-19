"""Runnable companion to LLM_USAGE_GUIDE.md, verified against easy-glm 0.471.

Uses synthetic data only. Writes the requested examples to --out.
Run: python llm_workflow.py --out guide-output
Add --with-pairs to exercise optional CatBoost/Optuna interaction training.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import polars as pl

from easy_glm import EasyGLM, RateModel, add_train_test_split
from easy_glm.workflow import (
    Adjustment,
    DataSource,
    Penalty,
    Project,
    Split,
    VariableDesign,
    ae_by_variable,
    prepare,
    rebalance_override,
    rebuild_rate_model,
    run_model,
    to_report_html,
    to_scoring_script,
    to_script,
    totals,
    train_holdout,
)
from easy_glm.workflow.feature_selection import select_variables
from easy_glm.workflow.project import PairSearchConfig, PairStageConfig


def synthetic_portfolio(size: int = 1200) -> pl.DataFrame:
    rng = np.random.default_rng(471)
    age = rng.integers(18, 81, size)
    vehicle_age = rng.integers(0, 16, size)
    region = rng.choice(["North", "South", "West"], size)
    exposure = rng.uniform(0.3, 1.0, size)
    frequency = (
        0.25
        * np.where(age < 30, 1.8, 1.0)
        * np.where(vehicle_age > 10, 1.3, 1.0)
        * np.where(region == "North", 1.4, 1.0)
        * np.where((age < 30) & (region == "North"), 1.8, 1.0)
    )
    raw = pl.DataFrame(
        {
            "PolicyId": np.arange(size),
            "Claims": rng.poisson(exposure * frequency),
            "Exposure": exposure,
            "DriverAge": age,
            "VehicleAge": vehicle_age,
            "Region": region,
            "Noise": rng.normal(size=size),
        }
    )
    return add_train_test_split(raw, seed=42)


def run_example(out: Path, *, with_pairs: bool = False) -> None:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = synthetic_portfolio()
    raw.write_parquet(out / "portfolio.parquet")

    # Short public API: predicts a rate, while the saved scorer can apply exposure.
    simple = EasyGLM.fit(
        raw,
        target="Claims",
        model_type="Poisson",
        predictors=["DriverAge", "VehicleAge", "Region"],
        weight_col="Exposure",
        divide_target_by_weight=True,
        cv=5,
        n_alphas=8,
        n_bins=6,
        knots={"DriverAge": [25, 35, 45, 55, 65]},
    )
    np.testing.assert_allclose(
        simple.predict(raw).to_numpy(),
        simple.rate_model.predict(raw, exposure_col=None),
    )
    np.testing.assert_allclose(
        simple.rate_model.predict(raw),
        simple.predict(raw).to_numpy() * raw["Exposure"].to_numpy(),
    )

    # Project API: explicit, shareable preparation and modelling settings.
    project = Project(name="Guide example")
    project.data.source = DataSource(
        type="parquet", path=str(out / "portfolio.parquet")
    )
    project.data.roles = {
        "PolicyId": "id",
        "Claims": "target",
        "Exposure": "weight",
        "DriverAge": "predictor",
        "VehicleAge": "predictor",
        "Region": "predictor",
        "Noise": "unassigned",
        "traintest": "split",
    }
    project.data.split = Split(
        mode="column", column="traintest", train_value=1, holdout_value=0
    )
    project.design.defaults.n_bins = 6
    project.design.variables["DriverAge"] = VariableDesign(knots=[25, 35, 45, 55, 65])
    project.design.variables["VehicleAge"] = VariableDesign(n_bins=5)
    project.design.variables["Region"] = VariableDesign(kind="categorical")

    # Keep the screening recipe BEFORE changing roles after review.
    screening_project = project.copy()
    options = {
        "family": "poisson",
        "divide_target_by_weight": True,
        "include_unassigned": True,
        "n_alphas": 8,
        "repeats": 2,
        "seed": 42,
    }
    screening = select_variables(screening_project, raw, **options)
    (out / "screening.json").write_text(
        json.dumps(screening, indent=2), encoding="utf-8"
    )
    assert screening["candidate_count"] == 4
    assert screening["training_rows"] == raw.filter(pl.col("traintest") == 1).height
    assert all(row["status"] != "failed" for row in screening["rows"]), screening
    project.exploration["feature_selection"] = {
        "version": 1,
        "project": screening_project.to_dict(),
        "options": copy.deepcopy(options),
        "result": copy.deepcopy(screening),
    }
    # A deliberate example decision, not an automatic consequence of the scores.
    project.apply_role_change("Noise", "ignore")
    cfg = project.new_model("Frequency", family="poisson", divide_target_by_weight=True)
    cfg.penalty = Penalty(cv=5, n_alphas=8, l1_ratio=1.0)
    prepared = prepare(project, raw)
    problems = project.validate("Frequency", columns=prepared.columns)
    assert not problems, problems
    run = run_model(project, prepared, "Frequency")
    train, holdout = train_holdout(prepared, project.data.split)
    actual, expected, weights = totals(holdout, run.config, run.predict(holdout))
    ae = ae_by_variable(holdout, "Region", actual, expected, weights)
    ae.write_csv(out / "holdout_ae_region.csv")

    # Project-level manual adjustments are rebuildable and included in exports.
    original_total = float(totals(train, run.config, run.predict(train))[1].sum())
    row = next(r for r in run.tables["DriverAge"].to_dicts() if r["from"] is not None)
    cfg.adjustments.append(
        Adjustment("DriverAge", row["from"], row["to"], float(row["relativity"]) * 1.05)
    )
    rebuild_rate_model(project, run, prepared)
    cfg.base_rate_override = rebalance_override(project, run, prepared)
    rebuild_rate_model(project, run, prepared)
    np.testing.assert_allclose(
        totals(train, run.config, run.predict(train))[1].sum(), original_total
    )

    if with_pairs:
        # Start from the fitted main model settings, without the example edits.
        pair_cfg = copy.deepcopy(cfg)
        pair_cfg.adjustments = []
        pair_cfg.base_rate_override = None
        pair_cfg.pair_method = "sequential_catboost"
        pair_cfg.pair_stages = [
            PairStageConfig(
                stage_id="driver_region",
                a="DriverAge",
                b="Region",
                search=PairSearchConfig(trials=2, prefix_trials=2),
            )
        ]
        project.models["Frequency_pairs"] = pair_cfg
        problems = project.validate("Frequency_pairs", columns=prepared.columns)
        assert not problems, problems
        run = run_model(project, prepared, "Frequency_pairs", progress=print)
        assert len(run.rate_model.pair_tables) == 1

    project.champion = run.name
    project.to_json(out / "project.json")
    run.rate_model.to_json(out / "model.easyglm")
    run.rate_model.to_excel(out / "rate_tables.xlsx")
    restored = RateModel.from_json(out / "model.easyglm")
    np.testing.assert_allclose(
        restored.predict(holdout, exposure_col=None), run.predict(holdout)
    )
    frozen_source = to_scoring_script(run, output_prefix="frozen")
    (out / "frozen_scoring.py").write_text(frozen_source, encoding="utf-8")
    namespace = {"__name__": "guide_frozen_scorer"}
    exec(compile(frozen_source, "frozen_scoring.py", "exec"), namespace)
    np.testing.assert_allclose(
        namespace["predict"](holdout, exposure_col=None), run.predict(holdout)
    )
    training_source = to_script(project, run.name, run=run, output_prefix="retrained")
    compile(training_source, "training_workflow.py", "exec")
    (out / "training_workflow.py").write_text(training_source, encoding="utf-8")
    (out / "report.html").write_text(
        to_report_html(project, {run.name: run}, prepared, champion=run.name),
        encoding="utf-8",
    )
    print(f"Validated workflow and exports in {out}")
    print(run.summary())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("guide-output"))
    parser.add_argument("--with-pairs", action="store_true")
    args = parser.parse_args()
    run_example(args.out, with_pairs=args.with_pairs)


if __name__ == "__main__":
    main()
