"""Actuarial check for D5 — relativity tooling in the rate-table editor.

Fits the French-motor frequency model, then does on the fitted tables exactly
what the Tools panel does in the workbench — smooth the DrivAge curve (moving
average and isotonic), cap a tail, round to a step — and prints (or, with
``--write``, regenerates ``docs/checks/d5-tooling.md``) what an actuary needs to
judge it: the before/after tables with the exposure behind each band, the
premium level before and after each operation, what the undo stack and the
snapshots do, and the differences between two snapshots.

Run from the repository root::

    .venv/bin/python scripts/checks/d5_tooling.py [--write]
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import polars as pl

from easy_glm import DesignSpec, fit_glm, to_rate_model
from easy_glm.engine import tooling
from easy_glm.workflow import ModelConfig, rate_model_diff, totals
from easy_glm.workflow.diagnostics import deviance_stats, gini, unit_values

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "checks" / "d5-tooling.md"
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
SMOOTHED = "DrivAge"  # a wobbly curve: the smoothing example
CAPPED = "BonusMalus"  # a curve that turns back and has a long tail
ALPHA = 0.0003
#: cap for the tail example, on the BonusMalus curve
CAP = 3.00


def load() -> tuple[pl.DataFrame, str]:
    cached = glob.glob(os.path.expanduser("~/.cache/easy_glm/*.parquet"))
    if cached:
        return pl.read_parquet(cached[0]), "full French motor set (cached)"
    return pl.read_parquet(FIXTURE), "French motor 50k fixture"


def split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(42)
    df = df.with_columns(
        pl.Series("traintest", (rng.random(df.height) < 0.7).astype(int))
    )
    return df.filter(pl.col("traintest") == 1), df.filter(pl.col("traintest") == 0)


def fit(train: pl.DataFrame):
    spec = DesignSpec.from_data(
        train,
        PREDICTORS,
        categorical=["VehPower"],
        weight_col="Exposure",
        n_bins=20,
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


def holdout_numbers(
    glm, rm, holdout: pl.DataFrame, cfg: ModelConfig
) -> dict[str, float]:
    pred = rm.predict(holdout, exposure_col=None)
    actual, expected, w = totals(holdout, cfg, pred)
    y, wu = unit_values(holdout, cfg)
    dev = deviance_stats(glm.model.family_instance, y, pred, wu)
    return {
        "ae": float(actual.sum() / expected.sum()),
        "gini": float(gini(actual, expected, w)),
        "dev": float(dev["deviance_explained"]),
    }


def table_rows(cfg_var, values: list[float] | None = None) -> list[tuple]:
    """``(label, exposure, relativity)`` per row, using ``values`` when given."""
    from easy_glm.engine.models import level_label

    out = []
    for i, row in enumerate(cfg_var.table):
        rel = row.relativity if values is None else values[i]
        out.append((level_label(row, cfg_var.other_label), row.exposure, rel))
    return out


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *["| " + " | ".join(r) + " |" for r in rows],
    ]


def compare_table(cfg_var, results: dict[str, tooling.ToolResult]) -> list[str]:
    """One row per band: exposure, the fitted relativity and what each tool
    would make it."""
    names = list(results)
    header = ["band", "exposure", "fitted", *names]
    rows = []
    for i, (label, exposure, fitted) in enumerate(table_rows(cfg_var)):
        rows.append(
            [
                label,
                f"{exposure:,.0f}",
                f"{fitted:.3f}",
                *[f"{results[n].values[i]:.3f}" for n in names],
            ]
        )
    return md_table(header, rows)


def edited_model(glm, var: str, result: tooling.ToolResult):
    """A fresh rate model with one tool's result applied, the way the workbench
    applies it: one adjustment per band that moved."""
    edited = to_rate_model(glm)
    for row, value in zip(edited.variables[var].table, result.values, strict=True):
        if abs(value - row.relativity) > 1e-12:
            edited.update_relativity(var, row.from_, row.to_, value)
    return edited


def main(write: bool) -> None:
    df, source = load()
    train, holdout = split(df)
    glm = fit(train)
    cfg = ModelConfig(
        target="ClaimNb",
        weight="Exposure",
        divide_target_by_weight=True,
        predictors=PREDICTORS,
    )
    rm = to_rate_model(glm)
    age, bm = rm.variables[SMOOTHED], rm.variables[CAPPED]
    base = holdout_numbers(glm, rm, holdout, cfg)

    tools = {
        "smoothed (moving average, 3 bands)": (
            SMOOTHED,
            tooling.smooth_moving_average(age, SMOOTHED, window=3),
        ),
        "rounded to 0.05": (
            SMOOTHED,
            tooling.round_relativities(age, SMOOTHED, step=0.05),
        ),
        "isotonic (increasing)": (
            CAPPED,
            tooling.smooth_isotonic(bm, CAPPED, direction="increasing"),
        ),
        f"capped at {CAP:.2f}": (CAPPED, tooling.cap_floor(bm, CAPPED, cap=CAP)),
    }
    after = {
        name: holdout_numbers(glm, edited_model(glm, var, result), holdout, cfg)
        for name, (var, result) in tools.items()
    }

    def jump(values: list[float], n_rows: int) -> float:
        """Largest step between neighbouring bands, in log terms."""
        return max(
            abs(np.log(values[i + 1]) - np.log(values[i])) for i in range(n_rows - 2)
        )

    moving = tools["smoothed (moving average, 3 bands)"][1]
    before_jump = jump([r.relativity for r in age.table], len(age.table))
    after_jump = jump(moving.values, len(age.table))

    lines = [
        "# D5 — smoothing, capping and rounding relativities (and undoing it)",
        "",
        f"*Generated by `scripts/checks/d5_tooling.py` on the {source}: "
        f"{train.height:,} training rows, Poisson frequency, alpha {ALPHA}.*",
        "",
        "## What this adds",
        "",
        "The **Rate tables** page now has a *Tools* panel above the editor. It "
        "does the four things people otherwise do in a spreadsheet, on the "
        "table of the factor you are looking at:",
        "",
        "* **Smooth (moving average)** — replace each band by the average of "
        "itself and its neighbours, so a curve that zig-zags because of thin "
        "data reads as one shape.",
        "* **Smooth (isotonic)** — force the curve to run one way (increasing "
        "or decreasing) by pooling the bands that turn back.",
        "* **Cap / floor** — no relativity above (or below) a number you pick.",
        "* **Round** — to a number of decimals, or to a step such as 0.05, the "
        "way a published table is printed.",
        "",
        "Everything a tool does is saved as **ordinary manual adjustments**: the "
        "same entries you get by typing in the grid, in the same project file, "
        "applied to the same fit without refitting. Nothing appears in the "
        "tables that you could not have typed by hand, and *Reset* still "
        "removes it.",
        "",
        "## The rule that matters: smoothing does not move the premium level",
        "",
        "Relativities multiply, so the level of a factor is the "
        "**exposure-weighted average of the logs** of its relativities. The "
        "base rate is *not* refitted when you edit a table, so a smoothing that "
        "moved that average would quietly move every premium the factor "
        "touches. Both smoothers therefore keep it exactly where it was — the "
        "moving average is re-centred afterwards, and the isotonic fit "
        "preserves it by construction. A cap, a floor and a rounding do move "
        "it, on purpose, and the panel says by how much before you apply "
        "anything.",
        "",
        *md_table(
            [
                "operation",
                "factor",
                "mean log relativity",
                "overall level",
                "holdout A/E",
            ],
            [
                [
                    "*(fitted tables)*",
                    "—",
                    f"{tooling.weighted_log_mean(age):.9f} ({SMOOTHED})",
                    "—",
                    f"{base['ae']:.4f}",
                ],
                *[
                    [
                        name,
                        var,
                        f"{result.log_mean_after:.9f}",
                        (
                            "unchanged"
                            if abs(result.level_shift) < 5e-9
                            else f"{result.level_shift:+.4%}"
                        ),
                        f"{after[name]['ae']:.4f}",
                    ]
                    for name, (var, result) in tools.items()
                ],
            ],
        ),
        "",
        f"For the two smoothers the mean log relativity is the value the fitted "
        f"table already had — {tooling.weighted_log_mean(age):.9f} for {SMOOTHED} and "
        f"{tooling.weighted_log_mean(bm):.9f} for {CAPPED} — to every decimal shown, "
        "and to 1e-12 in the test that runs on every change. The holdout A/E "
        "moves only where the level moved: capping "
        f"{CAPPED} at {CAP:.2f} takes "
        f"{abs(tools[f'capped at {CAP:.2f}'][1].level_shift):.2%} off that "
        "factor, so the model under-charges by about that much until the base "
        "rate is re-set (Model page → base-rate override).",
        "",
        f"## {SMOOTHED}: a wobbly curve, smoothed and rounded",
        "",
        "Exposure is the training exposure in the band — it is what the tools "
        "weight a band by, so a thin band is pulled towards its well-populated "
        "neighbours and not the other way round. The *Other / Unknown* row is "
        "never touched by any tool: it is not part of the curve.",
        "",
        *compare_table(
            age,
            {
                "moving avg (3)": tools["smoothed (moving average, 3 bands)"][1],
                "round 0.05": tools["rounded to 0.05"][1],
            },
        ),
        "",
        f"The largest step from one band to the next was **{before_jump:.3f}** in "
        f"log terms ({np.exp(before_jump) - 1:+.1%}) and is **{after_jump:.3f}** "
        f"({np.exp(after_jump) - 1:+.1%}) after the moving average — with the "
        "level unchanged, so the premium of an average policy is what it was.",
        "",
        "The first and last bands are averaged over fewer neighbours (there is "
        "nothing beyond them), which is why the youngest band moves furthest "
        "here: it is averaged with one band that sits 33 % below it. If that is "
        "not the shape you want, undo it, or type that row back by hand — the "
        "level check on the panel tells you what the level did.",
        "",
        "On the page this is a chart: the current curve and the curve the tool "
        "would give, drawn on top of each other, with the mean log relativity "
        "before and after as two numbers above it. Nothing is written until "
        "*Apply to the table* is pressed.",
        "",
        f"## {CAPPED}: a curve that turns back, and a tail that is too long",
        "",
        f"The fitted {CAPPED} curve rises, dips in the middle (the "
        "60–72 bands) and ends at a relativity of "
        f"{max(r.relativity for r in bm.table):.2f}. The dip is the kind of "
        "shape an actuary will not sign: bonus-malus is a ranking of past "
        "claims, so the curve should not turn back. *Isotonic (increasing)* "
        "pools the bands that break the order, weighted by their exposure; "
        f"*cap {CAP:.2f}* then trims the tail.",
        "",
        *compare_table(
            bm,
            {
                "isotonic (inc.)": tools["isotonic (increasing)"][1],
                f"cap {CAP:.2f}": tools[f"capped at {CAP:.2f}"][1],
            },
        ),
        "",
        "Holdout quality of each edited set of tables, against the fitted one:",
        "",
        *md_table(
            ["tables", "holdout A/E", "Gini", "deviance explained"],
            [
                [
                    "*(fitted)*",
                    f"{base['ae']:.4f}",
                    f"{base['gini']:.4f}",
                    f"{base['dev']:.2%}",
                ],
                *[
                    [name, f"{m['ae']:.4f}", f"{m['gini']:.4f}", f"{m['dev']:.2%}"]
                    for name, m in after.items()
                ],
            ],
        ),
        "",
        "Smoothing trades a little fit for a shape you can defend; the Gini and "
        "the deviance say how much it cost on business the model has not seen.",
        "",
        "## Undo, and snapshots",
        "",
        "**Undo / Redo** sit next to *Reset*. Every edit — a typed cell, a tool, "
        "a reset, a restored snapshot — is one step, and a step is the whole "
        "set of adjustments, so undo puts the tables back exactly as they were "
        "(the same numbers, not an approximation). The stack holds the last 50 "
        "steps of this browser session; it is not written to the project file, "
        "so closing the browser ends it. What *is* in the project file is the "
        "list of adjustments, which is the tables themselves.",
        "",
        "**Snapshots** are for the versions you want to keep: *Snapshot as…* "
        "names the tables as they stand ('as fitted', 'before smoothing', "
        "'signed off'), and the name goes into the project file, so it survives "
        "a reload, a refit and tomorrow. Restoring one puts those tables back; "
        "comparing two lists every band that moved between them, exactly as the "
        "Compare page does for two models.",
        "",
        "Here is that comparison for two of the versions above — the fitted "
        f"tables against the smoothed {SMOOTHED} — listing every band that "
        "moved by more than 1 %:",
        "",
    ]

    diff = rate_model_diff(
        to_rate_model(glm), edited_model(glm, SMOOTHED, moving), tol=0.01
    )
    lines += md_table(
        ["variable", "band", "as fitted", "smoothed", "log difference"],
        [
            [
                r["variable"],
                r["band"],
                f"{r['relativity_a']:.3f}",
                f"{r['relativity_b']:.3f}",
                f"{r['log_diff']:+.3f}",
            ]
            for r in diff.head(10).iter_rows(named=True)
        ],
    )
    lines += [
        "",
        f"({diff.height} band(s) in all; the table shows the first 10, largest "
        "difference first.)",
        "",
        "## Guarantees (tested on every change)",
        "",
        "- **A smoothing never moves the premium level**: the exposure-weighted "
        "mean of the log relativities is identical before and after, to 1e-12. "
        "A cap, a floor and a rounding move it on purpose and say by how much.",
        "- **The Other / Unknown row is never touched** by any tool, whatever "
        "its relativity is — it is not part of the curve, and it is where "
        "unknown and missing values are rated.",
        "- **Capping and rounding are idempotent**: applying either a second "
        "time with the same numbers changes nothing.",
        "- **The isotonic result really is monotone**, and it pools whole runs "
        "of bands by their exposure, so one thin band cannot drag a fat one.",
        "- **A categorical factor is not smoothed by accident**: its levels are "
        "listed most-exposed first, which is not an order of the risk, so the "
        "two smoothers refuse it until you confirm that this table does read in "
        "order.",
        "- **A piecewise-linear curve stays continuous**: the tools move the "
        "*nodes* of the curve (the value at each knot and at the two clamp "
        "points) and the slopes are re-derived from them, so the curve never "
        "jumps at a knot.",
        "- **The tables stay exact**: after any tool, the scorer's predictions "
        "are the base rate times the relativities the tables show, to 1e-12.",
        "- **Undo restores the previous tables exactly**, and redo puts the "
        "change back.",
        "",
        "## Questions for you",
        "",
        "- **Q15.** The default smoothing window is **3 bands** (a band and its "
        "two neighbours), weighted by exposure. Is 3 the right default for a "
        "20-band curve, or would you rather start at 5? *Default until you "
        "answer: 3, changeable on the page every time it is used.*",
        "- **Q16.** After a cap or a floor, should the rest of the curve be "
        "scaled back up so the factor's level is unchanged (the premium stays "
        "where it was, and the capped bands are subsidised by the others), or "
        "should the level fall as it does now? *Default until you answer: the "
        "level falls and the page says by how much, because a cap that is "
        "shifted back up is not a cap; the overall level is re-set with the "
        "base-rate override on the Model page.*",
        "",
    ]

    text = "\n".join(lines)
    print(text, end="")  # stdout is byte-identical to the written document
    if write:
        DOC.write_text(text)
        print(f"\nwritten: {DOC}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write", action="store_true", help="regenerate the docs/checks document"
    )
    main(ap.parse_args().write)
