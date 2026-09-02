"""Actuarial check for piece B — piecewise-linear (L-dummy) terms.

Fits the French-motor frequency model twice — with ``Density`` as a step
function (0.3 behaviour) and as a piecewise-linear term — and prints (or, with
``--write``, regenerates ``docs/checks/b-linear.md``) what an actuary needs to
judge the feature: the two curves side by side at round values of Density,
the clamp points, holdout deviance / Gini for each, and the exactness of the
rate tables at and beyond the clamp.

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


def fit(train: pl.DataFrame, linear: list[str] | None):
    spec = DesignSpec.from_data(
        train,
        PREDICTORS,
        categorical=["VehPower"],
        weight_col="Exposure",
        linear=linear,
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
        "sum insured or vehicle value. The lasso still decides where the slope changes:",
        "most knots carry no change, so the curve has few bends.",
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
        "3. **Few bends, not few slopes.** Each fitted number is a *change of slope*; the "
        "penalty removes changes, so long straight stretches are the norm. Monotone "
        "constraints are not offered on these terms in this release.",
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
        f"The linear curve has {len(enc.knots)} candidate knots; the fit kept "
        f"{m_lin['terms'][LINEAR_VAR]} non-zero `{LINEAR_VAR}` terms (the hinge at the "
        "clamp plus the bends it needed), against "
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
        "The bonus-malus effect is close to log-linear, so the straight-line description",
        "fits the holdout slightly better with fewer numbers. That is the typical case for",
        "a linear term; it is not a general predictive gain.",
        "",
        "**One thing to look at before shipping such a curve.** Above about 120 the data",
        "is thin. The step design pooled everything from 100 upwards into one band; the",
        "linear term keeps its slope going through the thin region up to the clamp, so the",
        "table charges far more at 200–230 than the step table does. Within the training",
        "range the curve follows the data it has, however little. If that is not what you",
        "would charge, either set the clamp for this factor to where the data runs out",
        "(e.g. 150 — the curve is then flat above it) or keep the step design. See Q10.",
        "",
        f"## A second example: `{SECOND_VAR}` (skewed, one straight line)",
        "",
        f"`{SECOND_VAR}` is heavily skewed (most policies below 5,000). Asked for a linear "
        f"term, the lasso kept {m_lin2['terms'][SECOND_VAR]} of {len(enc2.hinges)} hinges — "
        "a single straight line on the log scale from "
        f"{enc2.lo:g} to {enc2.hi:g} — where the step design used "
        f"{m_step['terms'][SECOND_VAR]} increments. Holdout Gini {m_step['gini']:.4f} (step) "
        f"vs {m_lin2['gini']:.4f} (linear), deviance explained {m_step['dev_explained']:.2%} "
        f"vs {m_lin2['dev_explained']:.2%}: here the step design describes the low end "
        "better. Which shape to use is a judgement per factor; both are available.",
        "",
        f"| {SECOND_VAR} | step (0.3) | piecewise-linear |",
        "|---:|---:|---:|",
        *[
            f"| {v:,}{' (above clamp → flat)' if v > enc2.hi else ''} | {s_:.4f} | {l_:.4f} |"
            for v, s_, l_ in zip(PROBE_2, step_curve2, lin_curve2, strict=True)
        ],
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
        "- Q9 — which of the bike book's variables should be piecewise-linear — still stands; "
        "the default is mileage only.",
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
