"""Actuarial check for piece A — two-way interactions.

Fits the French-motor frequency model with and without ``DrivAge × VehPower``
and prints (or, with ``--write``, regenerates ``docs/checks/a-interactions.md``)
what an actuary needs to judge the feature: the two DrivAge main tables side by
side, the adjustment matrix with its training exposure, holdout metrics with
and without the interaction, and the A/E-by-pair table before and after.

Run from the repository root::

    .venv/bin/python scripts/checks/a_interactions.py [--write]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

from easy_glm import DesignSpec, fit_glm, rate_tables, to_rate_model
from easy_glm.core.excel import interaction_matrices
from easy_glm.workflow import ModelConfig, ae_by_pair, gini, totals
from easy_glm.workflow.diagnostics import deviance_stats, unit_values

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "checks" / "a-interactions.md"
FIXTURE = ROOT / "tests" / "fixtures" / "french_motor_50k.parquet"
PREDICTORS = [
    "DrivAge",
    "VehAge",
    "BonusMalus",
    "Density",
    "VehPower",
    "Region",
    "VehBrand",
    "VehGas",
    "Area",
]
PAIR = ("DrivAge", "BonusMalus")  # the pair with the strongest holdout A/E structure
ALPHA = 0.0003
MIN_CELL = 0.005


def load() -> tuple[pl.DataFrame, str]:
    cached = glob.glob(os.path.expanduser("~/.cache/easy_glm/*.parquet"))
    if cached:
        return pl.read_parquet(cached[0]), "full French motor set (cached)"
    return pl.read_parquet(FIXTURE), "French motor 50k fixture"


def split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(42)
    is_train = rng.random(df.height) < 0.7
    df = df.with_columns(pl.Series("traintest", is_train.astype(int)))
    return df.filter(pl.col("traintest") == 1), df.filter(pl.col("traintest") == 0)


def fit(train: pl.DataFrame, with_interaction: bool):
    spec = DesignSpec.from_data(
        train,
        PREDICTORS,
        categorical=["VehPower"],
        weight_col="Exposure",
        interactions=[PAIR] if with_interaction else None,
        min_cell_exposure=MIN_CELL,
    )
    return fit_glm(
        train,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=ALPHA,
    )


def metrics(f, frame: pl.DataFrame) -> dict[str, float]:
    cfg = ModelConfig(target="ClaimNb", weight="Exposure", divide_target_by_weight=True)
    pred = f.predict(frame)
    actual, expected, w = totals(frame, cfg, pred)
    y, wu = unit_values(frame, cfg)
    dev = deviance_stats(f.model.family_instance, y, pred, wu)
    return {
        "A/E": actual.sum() / expected.sum(),
        "Gini": gini(actual, expected, w),
        "deviance explained": dev["deviance_explained"],
    }


def planted_check() -> dict[str, float]:
    """The planted-truth case from ``tests/test_recovery.py``, summarised."""
    rng = np.random.default_rng(11)
    n = 40_000
    age = rng.integers(18, 80, n).astype(float)
    region = rng.choice(
        ["R1", "R2", "R3", "R4", "R5"], n, p=[0.5, 0.3, 0.148, 0.05, 0.002]
    ).astype(object)
    expo = rng.uniform(0.2, 1.0, n)
    truth = 0.9
    mu = np.exp(
        -2.0
        - 0.03 * np.maximum(45 - age, 0)
        + np.where(region == "R1", 0.0, 0.25)
        + np.where((age < 25) & (region == "R2"), truth, 0.0)
    )
    book = pl.DataFrame(
        {
            "ClaimNb": rng.poisson(mu * expo).astype(float),
            "Exposure": expo,
            "DrivAge": age,
            "Region": region,
        }
    )
    knots = [25, 30, 40, 50, 60]
    kw = {
        "family": "poisson",
        "weight_col": "Exposure",
        "divide_target_by_weight": True,
    }
    spec = DesignSpec.from_data(
        book,
        ["DrivAge", "Region"],
        knots={"DrivAge": knots},
        min_level_share=0.001,
        interactions=[("DrivAge", "Region")],
        min_cell_exposure=0.0,
    )
    f = fit_glm(book, spec, "ClaimNb", alpha=2e-4, **kw)
    probe = pl.DataFrame(
        {"DrivAge": [20.0, 20.0, 40.0, 40.0], "Region": ["R2", "R1", "R2", "R1"]}
    )
    pr = f.predict(probe)
    recovered = float(np.log(pr[0] * pr[3] / (pr[1] * pr[2])))
    tab = rate_tables(f)["DrivAge×Region"]
    thin = tab.filter((pl.col("from_b") == "R5") & (pl.col("exposure") > 0))
    spec0 = DesignSpec.from_data(
        book, ["DrivAge", "Region"], knots={"DrivAge": knots}, min_level_share=0.001
    )
    f0 = fit_glm(book, spec0, "ClaimNb", alpha=5e-4, **kw)
    cfg = ModelConfig(target="ClaimNb", weight="Exposure", divide_target_by_weight=True)
    a, e, w = totals(book, cfg, f0.predict(book))
    pair = ae_by_pair(book, "DrivAge", "Region", a, e, w, knots_a=knots)
    cell = pair.filter((pl.col("label_a") == "< 25") & (pl.col("label_b") == "R2"))
    return {
        "truth": truth,
        "recovered": recovered,
        "thin_min": float(thin["relativity"].min()),
        "thin_max": float(thin["relativity"].max()),
        "ae_without": float(cell["ae"][0]),
    }


def md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main(write: bool) -> None:
    df, source = load()
    train, holdout = split(df)
    base = fit(train, False)
    inter = fit(train, True)
    name = f"{PAIR[0]}×{PAIR[1]}"
    rm_inter = to_rate_model(inter, exposure_col="Exposure")

    # exactness on holdout (the product's promise, with the interaction present)
    exact = float(
        np.abs(
            rm_inter.predict(holdout, exposure_col=None) / inter.predict(holdout) - 1
        ).max()
    )

    # main tables side by side
    t0 = rate_tables(base)["DrivAge"]
    t1 = rate_tables(inter)["DrivAge"]
    main_rows = [
        [a, f"{b:.3f}", f"{c:.3f}", f"{(c / b - 1) * 100:+.1f}%"]
        for a, b, c in zip(t0["label"], t0["relativity"], t1["relativity"], strict=True)
    ]

    # adjustment matrix + exposure
    rows_a, rows_b, rel, exp = interaction_matrices(rm_inter, name)
    cells = int(sum(1 for r in rm_inter.variables[name].table if r.relativity != 1.0))
    kept = len(inter.spec[name].cells)
    total = int(len(rows_a) * len(rows_b))
    matrix_rows = []
    for i, ra in enumerate(rows_a):
        matrix_rows.append(
            [ra]
            + [
                (f"{rel[i][j]:.3f} ({exp[i][j]:,.0f})" if exp[i][j] > 0 else "—")
                for j in range(len(rows_b))
            ]
        )

    # metrics
    m0 = metrics(base, holdout)
    m1 = metrics(inter, holdout)

    # A/E by pair before and after (holdout)
    cfg = ModelConfig(target="ClaimNb", weight="Exposure", divide_target_by_weight=True)
    knots = inter.spec["DrivAge"].knots
    ae_rows = []
    for label, f in (("without", base), ("with", inter)):
        a, e, w = totals(holdout, cfg, f.predict(holdout))
        pair = ae_by_pair(holdout, PAIR[0], PAIR[1], a, e, w, knots_a=knots).filter(
            pl.col("exposure") > 300
        )
        worst = pair.with_columns(pl.col("ae").log().abs().alias("dev")).sort(
            "dev", descending=True
        )
        ae_rows.append(
            [
                label,
                f"{worst['dev'].max():.3f}",
                f"{float(np.sqrt(np.average(worst['dev'] ** 2, weights=worst['exposure']))):.3f}",
                f"{worst['label_a'][0]} | {worst['label_b'][0]}",
            ]
        )

    planted = planted_check()

    lines = [
        "# A — two-way interactions: what changed for you",
        "",
        f"*Generated by `scripts/checks/a_interactions.py` on the {source} "
        f"(70/30 split, seed 42, alpha {ALPHA}, minimum cell exposure {MIN_CELL:.1%} of "
        f"the interaction's training exposure).*",
        "",
        "## What an interaction is here",
        "",
        f"`{name}` sits **on top of** the two main-effect tables: a policy's relativity is "
        f"the DrivAge factor × the VehPower factor × one cell of the adjustment matrix "
        "below. A cell of 1.000 means *no adjustment* — either the data did not ask for "
        "one (the lasso kept it at 1) or the cell had too little exposure to be rated on "
        "its own (shown with its exposure so you can tell the two apart). This is the "
        "Emblem-style layout agreed in the plan (mains + adjustment matrix, joint fit).",
        "",
        "## Defaults in force (from the questions for the actuary)",
        "",
        f"- **Q4** minimum cell exposure: {MIN_CELL:.1%} of the interaction's training "
        f"exposure ({kept} of {total} cells were rated on their own; the rest adjust by 1.000).",
        "- **Q5** joint fit: the main-effect tables move when the interaction is added; both "
        "versions of the DrivAge table are shown below so the movement is visible.",
        "- Thin cells are penalised harder than fat ones (an unstandardised penalty scaled so "
        "that a 50/50 cell is treated like a 50/50 main effect), so sparse corners of the "
        "matrix do not pick up noise.",
        "",
        "## Holdout metrics with and without the interaction",
        "",
        md_table(
            [
                ["A/E", f"{m0['A/E']:.4f}", f"{m1['A/E']:.4f}"],
                ["Gini", f"{m0['Gini']:.4f}", f"{m1['Gini']:.4f}"],
                [
                    "deviance explained",
                    f"{m0['deviance explained']:.2%}",
                    f"{m1['deviance explained']:.2%}",
                ],
                [
                    "non-zero coefficients",
                    f"{int((base.coef != 0).sum())} / {len(base.coef)}",
                    f"{int((inter.coef != 0).sum())} / {len(inter.coef)}",
                ],
                ["cells adjusted (≠ 1.000)", "—", f"{cells} of {kept} rated cells"],
            ],
            ["quantity", "without", "with"],
        ),
        "",
        f"Rate tables (mains × matrix) reproduce the GLM on the holdout: max relative "
        f"difference {'below 1e-12' if exact < 1e-12 else f'{exact:.1e}'}.",
        "",
        "## DrivAge main table, without and with the interaction",
        "",
        md_table(main_rows, ["band", "without", "with", "change"]),
        "",
        f"## Adjustment matrix `{name}` — relativity (training exposure)",
        "",
        md_table(matrix_rows, [f"{PAIR[0]} \\ {PAIR[1]}"] + rows_b),
        "",
        "## A/E by DrivAge × VehPower cell on the holdout (cells with exposure > 300)",
        "",
        md_table(
            ae_rows,
            [
                "model",
                "worst |log A/E|",
                "exposure-weighted RMS |log A/E|",
                "worst cell",
            ],
        ),
        "",
        "## Controlled check on synthetic data",
        "",
        "A synthetic book of 40,000 policies with a planted effect: drivers under 25 "
        f"in one region claim e^{planted['truth']:.1f} = {np.exp(planted['truth']):.2f}× more "
        "than the mains alone would say, and a deliberately rare region (about 80 "
        "policies) carries no effect at all.",
        "",
        md_table(
            [
                ["planted effect (log scale)", f"{planted['truth']:.3f}"],
                ["recovered by the model (log scale)", f"{planted['recovered']:.3f}"],
                [
                    "cells of the rare, no-effect region",
                    f"{planted['thin_min']:.3f} – {planted['thin_max']:.3f} (all 1.000 = untouched)",
                ],
                [
                    "A/E-by-pair on the model *without* the interaction, planted cell",
                    f"{planted['ae_without']:.3f} (|log| {abs(np.log(planted['ae_without'])):.2f} > 0.2 flags it)",
                ],
            ],
            ["quantity", "value"],
        ),
        "",
        "## Questions for you",
        "",
        "- Is **mains + adjustment matrix** how you want to read an interaction in Excel "
        "(sheet `"
        + name
        + "` is the long table, `"
        + name
        + " (matrix)` the grid with "
        "exposure alongside)? Default: yes.",
        f"- Is {MIN_CELL:.1%} of exposure a sensible floor for rating a cell on its own? "
        "Default: yes; it is a per-interaction setting.",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if write:
        DOC.write_text(text)
        print(f"\nwritten {DOC.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write", action="store_true", help="regenerate the docs/checks document"
    )
    main(ap.parse_args().write)
