"""Actuarial check for pieces B / B2 — piecewise-linear (L-dummy) terms.

Fits the French-motor frequency model several times — with the factor as a step
function (0.3 behaviour), as a piecewise-linear term (B2 basis: one penalised
slope per band, so flat sections come out exactly flat), as a piecewise-linear
term with a monotone constraint, and as a single straight line
(``kind="continuous"``) — and prints (or, with ``--write``, regenerates
``docs/checks/b-linear.md``) what an actuary needs to judge the feature: the
curves side by side at round values, the clamp points, holdout deviance / Gini
for each, and the exactness of the rate tables at and beyond the clamp.

Run from the repository root::

    .venv/bin/python scripts/checks/b_linear.py [--write]
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import polars as pl

from easy_glm import DesignSpec, fit_glm, rate_tables, to_rate_model
from easy_glm.workflow import ModelConfig, gini, totals
from easy_glm.workflow.diagnostics import deviance_stats, unit_values

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "checks" / "b-linear.md"
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
LINEAR_VAR = "BonusMalus"  # the factor whose effect is closest to log-linear
SECOND_VAR = (
    "Density"  # a skewed factor where a single straight line is all the data asks for
)
ALPHA = 0.0003  # same penalty as the piece-A check; at 0.001 the linear BonusMalus term is over-shrunk
PROBE = [40, 50, 55, 60, 70, 80, 90, 100, 120, 150, 200, 230, 300]
PROBE_2 = [1, 25, 100, 500, 1_000, 2_500, 5_000, 10_000, 20_000, 27_000, 40_000]


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


def fit(
    train: pl.DataFrame,
    linear: list[str] | None,
    *,
    knots: dict[str, list[float]] | None = None,
    monotone: dict[str, str] | None = None,
):
    spec = DesignSpec.from_data(
        train,
        PREDICTORS,
        categorical=["VehPower"],
        weight_col="Exposure",
        linear=linear,
        knots=knots,
    )
    return fit_glm(
        train,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=ALPHA,
        monotone=monotone,
    )


def flat_bands(fitted, var: str) -> tuple[int, int]:
    """``(bands whose fitted slope is exactly zero, bands in total)``."""
    enc = fitted.spec[var]
    beta = fitted.coef[fitted.spec.slices()[var]][: enc.n_bands]
    return int((beta == 0.0).sum()), int(enc.n_bands)


def metrics(fitted, holdout: pl.DataFrame) -> dict[str, float]:
    cfg = ModelConfig(target="ClaimNb", weight="Exposure", divide_target_by_weight=True)
    pred = fitted.predict(holdout)
    actual, expected, w = totals(holdout, cfg, pred)
    y, wu = unit_values(holdout, cfg)
    dev = deviance_stats(fitted.model.family_instance, y, pred, wu)
    return {
        "ae": float(actual.sum() / expected.sum()),
        "gini": gini(actual, expected, w),
        "dev_explained": dev["deviance_explained"],
        "non_zero": int((fitted.coef != 0).sum()),
        "terms": {
            v: int(
                sum(
                    1
                    for f, c in zip(fitted.spec.features, fitted.coef, strict=True)
                    if f.variable == v and c != 0
                )
            )
            for v in (LINEAR_VAR, SECOND_VAR)
        },
    }


def curve(rm, var: str, values: list[float]) -> np.ndarray:
    """Relativity of one variable's table alone at ``values`` (mains only)."""
    cfg = rm.variables[var]
    from easy_glm.engine._scoring import score_linear, score_numeric

    x = np.asarray(values, dtype=float)
    if cfg.type == "linear":
        return score_linear(x, cfg)
    return score_numeric(x, cfg)


def main(write: bool) -> None:
    df, source = load()
    train, holdout = split(df)
    step_fit, lin_fit = fit(train, None), fit(train, [LINEAR_VAR])
    lin2_fit = fit(train, [SECOND_VAR])
    # B2: the same term with a monotone constraint (a sign bound on the slopes)
    mono_fit = fit(train, [LINEAR_VAR], monotone={LINEAR_VAR: "increasing"})
    # B2: kind="continuous" — the linear encoder with no interior knots
    cont_fit = fit(train, [SECOND_VAR], knots={SECOND_VAR: []})
    step_rm, lin_rm, lin2_rm = (
        to_rate_model(step_fit),
        to_rate_model(lin_fit),
        to_rate_model(lin2_fit),
    )
    m_step, m_lin, m_lin2 = (
        metrics(step_fit, holdout),
        metrics(lin_fit, holdout),
        metrics(lin2_fit, holdout),
    )
    m_mono, m_cont = metrics(mono_fit, holdout), metrics(cont_fit, holdout)
    mono_rm, cont_rm = to_rate_model(mono_fit), to_rate_model(cont_fit)
    zero_lin, n_lin = flat_bands(lin_fit, LINEAR_VAR)
    zero_lin2, n_lin2 = flat_bands(lin2_fit, SECOND_VAR)
    mono_curve = curve(mono_rm, LINEAR_VAR, PROBE)
    cont_curve = curve(cont_rm, SECOND_VAR, PROBE_2)
    mono_slopes = mono_fit.coef[mono_fit.spec.slices()[LINEAR_VAR]][
        : mono_fit.spec[LINEAR_VAR].n_bands
    ]
    enc = lin_fit.spec[LINEAR_VAR]
    lo, hi = enc.clamp
    enc2 = lin2_fit.spec[SECOND_VAR]
    step_curve, lin_curve = curve(step_rm, LINEAR_VAR, PROBE), curve(
        lin_rm, LINEAR_VAR, PROBE
    )
    step_curve2, lin_curve2 = (
        curve(step_rm, SECOND_VAR, PROBE_2),
        curve(lin2_rm, SECOND_VAR, PROBE_2),
    )
    lin_tab = rate_tables(lin_fit)[LINEAR_VAR]
    base_row = lin_tab.filter(pl.col("is_base"))

    # exactness of the linear rate tables at and beyond the clamp
    probe = holdout.head(200).with_columns(
        pl.Series(
            LINEAR_VAR,
            [lo - 1e6, lo, hi, hi + 1e6, None, float("inf"), -float("inf")]
            + holdout.head(200)[LINEAR_VAR].to_list()[7:],
            dtype=pl.Float64,
        )
    )
    exact = float(
        np.abs(
            lin_rm.predict(probe, exposure_col=None) / lin_fit.predict(probe) - 1
        ).max()
    )

    lines = [
        "# B — piecewise-linear terms: what changed for you",
        "",
        f"*Generated by `scripts/checks/b_linear.py` on the {source} "
        f"(70/30 split, seed 42, alpha {ALPHA}).*",
        "",
        "## What a piecewise-linear term is",
        "",
        "Until now every numeric rating factor was a **step** function: one relativity",
        "per band, jumping at the band edges. A **piecewise-linear** term lets the",
        "relativity change *smoothly* with the value — straight-line segments on the log",
        "scale between knots — which suits quantities like mileage, population density,",
        "sum insured or vehicle value.",
        "",
        "**Flat unless the data insists.** Following your answer to Q3, the model now",
        "fits one number per band — the *slope inside that band* — and the penalty pushes",
        "each of them to exactly zero. A band whose data does not demand a slope comes",
        "back perfectly flat, so the curve is a few sloped stretches joined by level ones",
        "rather than a line that is always drifting somewhere. (Before this change the",
        "fitted numbers were the *changes* of slope, so the penalty produced long straight",
        "runs instead of flat ones.) The curve is still continuous everywhere: the bands",
        "join up by construction.",
        "",
        "Three conventions, all from the plan review (questions Q1–Q3):",
        "",
        f"1. **Flat outside the data.** The curve is clamped at the training range — for "
        f"`{LINEAR_VAR}` here `{lo:g}` to `{hi:g}` — and stays level beyond it, so a value "
        "far outside anything seen in training gets the relativity at the nearer edge, never "
        "an extrapolated one. The default clamp is the training minimum and maximum "
        "**rounded outward to a round number** (each end moves by less than 1 % of the "
        "range: 17.65–29,857 becomes 0–29,900; 18–80 stays 18–80) and the end bands keep "
        "their fitted slope up to that number, so the curve does not stop exactly where the "
        "data stops. Set the clamp yourself on the Design page when the exact edge matters.",
        "2. **Relativity 1.00 sits at the lower edge of the most exposed band**, so the base "
        f"risk is a round, visible number (here `{LINEAR_VAR}` = {base_row['from'][0]:g}).",
        "3. **Few slopes, not few bends** (your Q3 answer). Each fitted number is the "
        "slope of one band and the penalty removes slopes, so flat stretches are the "
        f"norm: of the {n_lin} bands of the `{LINEAR_VAR}` curve below, {zero_lin} came "
        f"back exactly flat, and of the {n_lin2} bands of `{SECOND_VAR}`, {zero_lin2}. "
        "**Monotone constraints are available on these terms**: a direction bounds every "
        "band slope to one sign, which keeps the curve rising (or falling) throughout "
        "without forcing any particular shape on it.",
        "",
        f"## `{LINEAR_VAR}`: step versus piecewise-linear",
        "",
        f"Relativity of the `{LINEAR_VAR}` factor alone (base 1.00 at the base row of each "
        "table), at round values:",
        "",
        f"| {LINEAR_VAR} | step (0.3) | piecewise-linear |",
        "|---:|---:|---:|",
    ]
    for v, s_, l_ in zip(PROBE, step_curve, lin_curve, strict=True):
        note = ""
        if v < lo:
            note = " (below clamp → flat)"
        elif v > hi:
            note = " (above clamp → flat)"
        lines.append(f"| {v:,}{note} | {s_:.4f} | {l_:.4f} |")
    lines += [
        "",
        f"The linear curve has {len(enc.knots)} candidate knots, so {n_lin} bands; the "
        f"fit gave {n_lin - zero_lin} of them a slope and left {zero_lin} flat, against "
        f"{m_step['terms'][LINEAR_VAR]} non-zero step increments.",
        "",
        f"## Holdout comparison (everything else identical; only `{LINEAR_VAR}` changes)",
        "",
        "| | step | piecewise-linear |",
        "|---|---:|---:|",
        f"| A/E | {m_step['ae']:.4f} | {m_lin['ae']:.4f} |",
        f"| Gini | {m_step['gini']:.4f} | {m_lin['gini']:.4f} |",
        f"| Deviance explained | {m_step['dev_explained']:.2%} | {m_lin['dev_explained']:.2%} |",
        f"| Non-zero coefficients (whole model) | {m_step['non_zero']} | {m_lin['non_zero']} |",
        "",
        "The bonus-malus effect is close to log-linear, so a description made of sloped",
        f"stretches fits the holdout a little better (Gini {m_step['gini']:.4f} → "
        f"{m_lin['gini']:.4f}, deviance explained {m_step['dev_explained']:.2%} → "
        f"{m_lin['dev_explained']:.2%}) for about the same number of fitted numbers.",
        "That is the typical case for a linear term; it is not a general predictive gain.",
        "",
        "**One thing to look at before shipping such a curve.** Above about 120 the data",
        "is thin. The step design pooled everything from 100 upwards into one band; the",
        "linear term keeps its slope going through the thin region up to the clamp, so the",
        "table charges far more at 200–230 than the step table does. Within the training",
        "range the curve follows the data it has, however little. If that is not what you",
        "would charge, either set the clamp for this factor to where the data runs out",
        "(e.g. 150 — the curve is then flat above it) or keep the step design. See Q10.",
        "",
        f"## Keeping a curve monotone: `{LINEAR_VAR}` increasing",
        "",
        "Ask for a direction on the Design page and every band slope is bounded to that "
        "sign, so the curve can never turn round. Nothing else about the fit changes: "
        "flat bands are still allowed (a bound of zero), so the constraint costs you "
        "nothing where the data already agrees.",
        "",
        f"| {LINEAR_VAR} | piecewise-linear | same, forced increasing |",
        "|---:|---:|---:|",
        *[
            f"| {v:,} | {l_:.4f} | {m_:.4f} |"
            for v, l_, m_ in zip(PROBE, lin_curve, mono_curve, strict=True)
        ],
        "",
        f"Every band slope is at least zero ({int((mono_slopes >= 0).sum())} of "
        f"{len(mono_slopes)}, {int((mono_slopes == 0).sum())} of them exactly flat); "
        f"holdout Gini {m_lin['gini']:.4f} unconstrained vs {m_mono['gini']:.4f} "
        f"constrained, deviance explained {m_lin['dev_explained']:.2%} vs "
        f"{m_mono['dev_explained']:.2%}.",
        "",
        f"## A second example: `{SECOND_VAR}` (skewed) — and the **continuous** option",
        "",
        f"`{SECOND_VAR}` is heavily skewed (most policies below 5,000). Asked for a linear "
        f"term with {len(enc2.knots)} candidate knots, the fit left {zero_lin2} of its "
        f"{n_lin2} bands flat and gave {n_lin2 - zero_lin2} a slope, between "
        f"{enc2.lo:g} and {enc2.hi:g}, where the step design used "
        f"{m_step['terms'][SECOND_VAR]} increments. Holdout Gini {m_step['gini']:.4f} (step) "
        f"vs {m_lin2['gini']:.4f} (linear), deviance explained {m_step['dev_explained']:.2%} "
        f"vs {m_lin2['dev_explained']:.2%}: here the step design describes the low end "
        "better. Which shape to use is a judgement per factor; all of them are available.",
        "",
        "There is now a third choice for a numeric factor, **continuous**: one straight "
        "line over the whole range, no knots at all, so the relativity is a single rate "
        "per unit. It is the same machinery as a linear term with one band — same rate "
        "table, same editor, same Excel sheet, same exported script — and it is the "
        "shortest honest description of a factor you believe simply trends. Numeric "
        "factors still default to **step** (your Q9 answer); linear, continuous and "
        "categorical are the explicit overrides, one per factor, on the Design page.",
        "",
        f"| {SECOND_VAR} | step (0.3) | piecewise-linear | continuous |",
        "|---:|---:|---:|---:|",
        *[
            f"| {v:,}{' (above clamp → flat)' if v > enc2.hi else ''} | {s_:.4f} | "
            f"{l_:.4f} | {c_:.4f} |"
            for v, s_, l_, c_ in zip(
                PROBE_2, step_curve2, lin_curve2, cont_curve, strict=True
            )
        ],
        "",
        f"Holdout for the continuous version: Gini {m_cont['gini']:.4f}, deviance "
        f"explained {m_cont['dev_explained']:.2%}, A/E {m_cont['ae']:.4f}.",
        "",
        "## Guarantees (tested on every change)",
        "",
        f"- The rate tables reproduce the GLM at and beyond the clamp (including ±infinity) "
        "and on missing values: largest relative difference in this run below 1e-12 "
        f"({'yes' if exact < 1e-12 else 'NO — ' + format(exact, '.1e')}).",
        "- Inside a band the table is exactly log-linear (`relativity × exp(slope × distance)`);",
        "  the band's end value equals the next band's start value, so the curve is continuous.",
        "- Every row of the table is a point (node) of the curve. The '< lo' row and the first",
        "  band are **one number** (the value at the lower clamp) and move together; the",
        "  '≥ hi' row is the value at the upper clamp. Editing any row in the editor moves that",
        "  point and re-derives the slope of the band(s) touching it — one slope at either end,",
        "  two in the middle — so the curve never jumps, not even at the clamp points. The",
        "  missing-value row is not on the curve and edits on its own. Relativities must be",
        "  above 0 (the editor refuses 0 and says so).",
        "- A monotone direction bounds every band slope to one sign, so the curve cannot",
        "  turn round; the penalty may still flatten a band to zero, which both directions",
        "  allow. A **continuous** factor is a linear factor with a single band: identical",
        "  table type, editor, Excel sheet and exported script.",
        "- Excel and the exported script carry the slopes and the base point; a model rebuilt",
        "  from either scores identically. A table typed or rounded by hand (four decimals) reads",
        "  back as a continuous curve, because the slopes are re-derived from the row values;",
        "  a slope column that disagrees with them by more than 1 % is reported.",
        "",
        "## Questions for you",
        "",
        "- **Q10.** For a linear term, should the default upper clamp be the training "
        "maximum (the current default; the curve follows the data through thin tails) or a "
        "high quantile such as the 99.5th percentile (thin tails are pooled flat, as a step "
        "design would)? *Default until you answer: the training maximum, rounded outward; "
        "you can set the clamp per factor on the Design page.*",
        "- Q3 (few slopes rather than few bends) and Q9 (numeric factors default to step, "
        "with `linear` and `continuous` as explicit overrides) are answered and built; "
        "they need nothing further from you.",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if write:
        DOC.write_text(text)
        print(f"\nwritten: {DOC}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write", action="store_true", help="regenerate the docs/checks document"
    )
    main(ap.parse_args().write)
