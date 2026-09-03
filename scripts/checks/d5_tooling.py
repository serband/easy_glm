"""Actuarial check for D5 — relativity tooling in the rate-table editor.

Fits the French-motor frequency model, then does on the fitted tables exactly
what the Tools panel does in the workbench — smooth the DrivAge curve (moving
average and isotonic), cap a tail, round to a step — and prints (or, with
``--write``, regenerates ``docs/checks/d5-tooling.md``) what an actuary needs to
judge it: the before/after tables with the exposure behind each band, the
true change in total expected claims each operation makes to the book (and
what *Rebalance base rate* does about it), what the undo stack and the
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
from easy_glm.workflow import (
    ModelConfig,
    expected_claims,
    rate_model_diff,
    totals,
)
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


def rebalanced(glm, rm, train: pl.DataFrame, cfg: ModelConfig, target: float):
    """``rm`` with its base rate scaled so total expected claims on the training
    rows are ``target`` again — what the *Rebalance base rate* button does
    (predictions are linear in the base rate, so it is one ratio)."""
    out = to_rate_model(glm)
    out.variables = rm.variables
    out.base_rate = rm.base_rate * target / expected_claims(rm, train, cfg)
    return out


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
    fitted_claims = expected_claims(rm, train, cfg)

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
    edited = {
        name: edited_model(glm, var, result) for name, (var, result) in tools.items()
    }
    after = {
        name: holdout_numbers(glm, model, holdout, cfg)
        for name, model in edited.items()
    }
    #: change in total expected claims on the training book, per tool
    claims = {
        name: expected_claims(model, train, cfg) / fitted_claims - 1.0
        for name, model in edited.items()
    }
    cap_name = f"capped at {CAP:.2f}"
    cap_balanced = rebalanced(glm, edited[cap_name], train, cfg, fitted_claims)
    cap_balanced_numbers = holdout_numbers(glm, cap_balanced, holdout, cfg)
    cap_balanced_claims = (
        expected_claims(cap_balanced, train, cfg) / fitted_claims - 1.0
    )

    def jump(values: list[float], n_rows: int) -> float:
        """Largest step between neighbouring bands, in log terms."""
        return max(
            abs(np.log(values[i + 1]) - np.log(values[i])) for i in range(n_rows - 2)
        )

    moving = tools["smoothed (moving average, 3 bands)"][1]
    before_jump = jump([r.relativity for r in age.table], len(age.table))
    after_jump = jump(moving.values, len(age.table))

    def biggest_move(cfg_var, result) -> tuple[str, float, float, float]:
        """``(band, before, after, gap to its nearest neighbour)`` of the band
        the tool moved most — computed, so the sentence about it cannot drift."""
        rows = table_rows(cfg_var)
        i = max(
            range(len(rows) - 1),
            key=lambda k: abs(np.log(result.values[k]) - np.log(rows[k][2])),
        )
        neighbours = [rows[k][2] for k in (i - 1, i + 1) if 0 <= k < len(rows) - 1]
        gaps = [n / rows[i][2] - 1.0 for n in neighbours]
        gap = min(gaps, key=abs) if gaps else 0.0
        return rows[i][0], rows[i][2], result.values[i], gap

    worst_move = biggest_move(age, moving)

    def turning_points(cfg_var) -> str:
        """The bands where the curve goes down after going up — the shape the
        isotonic fit removes."""
        rows = table_rows(cfg_var)[:-1]  # the null row is not on the curve
        down = [
            rows[i + 1][0]
            for i in range(len(rows) - 1)
            if rows[i + 1][2] < rows[i][2] - 1e-12
        ]
        return ", ".join(down) if down else "no band"

    turn_back = turning_points(bm)

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
        "Next to them: **Undo / Redo**, named **snapshots** of the tables with a "
        "difference table between any two, and **Rebalance base rate**, which "
        "puts the overall level back where the fit had it after you have "
        "changed the shape of a factor.",
        "",
        "Everything a tool does is saved as **ordinary manual adjustments**: the "
        "same entries you get by typing in the grid, in the same project file, "
        "applied to the same fit without refitting. Nothing appears in the "
        "tables that you could not have typed by hand, and *Reset* still "
        "removes it.",
        "",
        "## Two different questions: the shape, and the money",
        "",
        "**The shape rule.** A smoothing keeps the *exposure-weighted mean of "
        "the log relativities* exactly where it was — the moving average is "
        "re-centred to achieve it, the isotonic fit preserves it by "
        "construction. That is what stops a smoothing from quietly sliding a "
        "whole factor up or down while you are only trying to change its shape.",
        "",
        "**That is not the same as leaving the premium alone.** A premium is a "
        "product of relativities, and the book is the *sum* of those products "
        "over the policies — an average of logs is a geometric average, and "
        "smoothing a curve reshuffles exposure between bands. So every tool, "
        "smoothing included, changes what the model expects to pay out, and the "
        "base rate is not refitted when you edit a table. The panel therefore "
        "shows **the change in total expected claims on the training rows** "
        "next to the log figure, and that is the number to read as money.",
        "",
        *md_table(
            [
                "operation",
                "factor",
                "mean log relativity",
                "expected claims (training)",
                "holdout A/E",
            ],
            [
                [
                    "*(fitted tables)*",
                    "—",
                    f"{tooling.weighted_log_mean(age):.9f} ({SMOOTHED})",
                    "*(the reference)*",
                    f"{base['ae']:.4f}",
                ],
                *[
                    [
                        name,
                        var,
                        f"{result.log_mean_after:.9f}"
                        + (
                            " *(unchanged)*"
                            if abs(result.level_shift) < 5e-9
                            else f" *({result.level_shift:+.3%})*"
                        ),
                        f"{claims[name]:+.3%}",
                        f"{after[name]['ae']:.4f}",
                    ]
                    for name, (var, result) in tools.items()
                ],
                [
                    f"{cap_name} **+ rebalance base rate**",
                    CAPPED,
                    f"{tools[cap_name][1].log_mean_after:.9f}",
                    f"{cap_balanced_claims:+.3%}",
                    f"{cap_balanced_numbers['ae']:.4f}",
                ],
            ],
        ),
        "",
        "Read the two middle columns together. For the two smoothers the mean "
        "log relativity is exactly the number the fitted table already had "
        f"({tooling.weighted_log_mean(age):.9f} for {SMOOTHED}, "
        f"{tooling.weighted_log_mean(bm):.9f} for {CAPPED}, to 1e-12 in the test "
        "that runs on every change) — **and the book still moves**: the moving "
        f"average takes {abs(claims['smoothed (moving average, 3 bands)']):.3%} "
        "off total expected claims, the isotonic fit "
        f"{abs(claims['isotonic (increasing)']):.3%}. The cap is the big one: "
        f"{claims[cap_name]:+.3%}, which is why the holdout A/E goes from "
        f"{base['ae']:.4f} to {after[cap_name]['ae']:.4f} — claims come in "
        f"{after[cap_name]['ae'] - 1:.1%} above what the capped tables expect. "
        "The last row is the same capped tables after one click of **Rebalance "
        "base rate**: no relativity changes, the base rate absorbs the "
        f"{-claims[cap_name]:+.3%}, expected claims on the training rows are "
        "back to the fitted total to the last decimal, and the holdout A/E is "
        f"{cap_balanced_numbers['ae']:.4f} — the fitted model's "
        f"{base['ae']:.4f} up to the difference between the training rows the "
        "rebalance is measured on and the holdout it is read on.",
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
        f"({np.exp(after_jump) - 1:+.1%}) after the moving average. The shape is "
        "the point; the price of it is the "
        f"{claims['smoothed (moving average, 3 bands)']:+.3%} on the book in the "
        "table above, which *Rebalance base rate* removes.",
        "",
        "The first and last bands are averaged over fewer neighbours (there is "
        "nothing beyond them), which is why "
        f"**{worst_move[0]}** moves furthest here "
        f"({worst_move[1]:.3f} → {worst_move[2]:.3f}): its only neighbour sits "
        f"{abs(worst_move[3]):.0%} "
        + ("above" if worst_move[3] > 0 else "below")
        + " it. If that is not the shape you want, undo it, or type that row "
        "back by hand.",
        "",
        "On the page this is a chart: the current curve and the curve the tool "
        "would give, drawn on top of each other, with the change in expected "
        "claims and the mean log relativity above it. Nothing is written until "
        "*Apply to the table* is pressed.",
        "",
        f"## {CAPPED}: a curve that turns back, and a tail that is too long",
        "",
        f"The fitted {CAPPED} curve rises, turns back at "
        f"{turn_back} and ends at a relativity of "
        f"{max(r.relativity for r in bm.table):.2f}. Turning back is the kind of "
        "shape an actuary will not sign: bonus-malus is a ranking of past "
        "claims, so the curve should not go down as the score goes up. "
        "*Isotonic (increasing)* pools the bands that break the order, weighted "
        f"by their exposure; *cap {CAP:.2f}* then trims the tail.",
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
        "Two different things are mixed in that A/E column and it is worth "
        "separating them. The **Gini** and the **deviance explained** say what "
        "the edit cost in *discrimination* — how much worse the model now sorts "
        "risks on business it has not seen; that is the price of a shape you "
        "can defend, and it is small here. The **A/E** moving away from 1.00 is "
        "not that: it is the level, the same money as the expected-claims "
        "column, and it is put right by rebalancing the base rate (the last row "
        "of the first table) rather than by accepting it.",
        "",
        "## Undo, and snapshots",
        "",
        "**Undo / Redo** sit next to *Reset*. Every edit — a typed cell, a tool, "
        "a reset, a rebalance, a restored snapshot — is one step, and a step is "
        "the whole state of the tables: the adjustments **and the base rate**, "
        "so undo puts back exactly what you had (the same numbers, not an "
        "approximation, and not the tables with somebody else's level). The "
        "stack holds the last 50 steps of this browser session; it is not "
        "written to the project file, so closing the browser ends it. What *is* "
        "in the project file is the list of adjustments and the base rate, "
        "which together are the tables.",
        "",
        "**Snapshots** are for the versions you want to keep: *Snapshot as…* "
        "names the tables as they stand ('as fitted', 'before smoothing', "
        "'signed off') together with the base rate in force, and the name goes "
        "into the project file, so it survives a reload, a refit and tomorrow. "
        "Restoring one puts those tables and that base rate back — and it is an "
        "undo step, so one click takes you back again. A snapshot taken before "
        "a factor left the model can no longer be restored; the page says so, "
        "names the factors, and changes nothing. Deleting a snapshot asks "
        "twice: it is the one thing undo does not cover.",
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
        "- **A smoothing never moves the mean log relativity**: it is identical "
        "before and after, to 1e-12. That is the *shape* guarantee, and it is "
        "not a promise about the premium — every tool changes total expected "
        "claims, and the panel shows that change (measured on the training "
        "rows, by scoring both sets of tables) next to it. **Rebalance base "
        "rate** puts the total back to the fitted one exactly, without touching "
        "a relativity.",
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
        "- **Undo restores the previous tables exactly** — the relativities "
        "*and* the base rate, because a restored snapshot or a rebalance can "
        "change the second — and redo puts the change back.",
        "- **A snapshot that no longer fits the model is refused**, naming the "
        "factors it adjusts that the model no longer has, and nothing is "
        "changed or saved. Deleting a snapshot asks twice, because it is the "
        "one action undo cannot undo.",
        "",
        "## Questions for you",
        "",
        "- **Q15.** The default smoothing window is **3 bands** (a band and its "
        "two neighbours), weighted by exposure. Is 3 the right default for a "
        "20-band curve, or would you rather start at 5? *Default until you "
        "answer: 3, changeable on the page every time it is used.*",
        "- **Q16.** After **any** tool — a cap, a floor, a rounding or a "
        "smoothing — should the *other* bands be scaled to keep the factor's "
        "own level (so the capped bands are subsidised by the rest of the "
        "curve), or should the curve keep the shape you asked for and the "
        "difference be shown? *Default until you answer: the shape you asked "
        "for is kept, the change in expected claims is shown, and the level is "
        "put back with Rebalance (which moves the base rate, not the "
        "relativities).*",
        "- **Q17.** The bigger version of the same question: after smoothing or "
        "capping, should the tool **preserve total expected claims "
        "automatically** — moving the base rate for you, the way an off-balance "
        "correction does in a rate review — or leave the base rate alone and "
        "show you the change? *Default until you answer: leave it alone, show "
        "the change on the panel, and offer* **Rebalance base rate** *as one "
        "click. The engine's own rule (plan §R6) is that a smoothing preserves "
        "the exposure-weighted mean of the log relativities; if you would "
        "rather it preserved the book, say so and Rebalance becomes automatic.*",
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
