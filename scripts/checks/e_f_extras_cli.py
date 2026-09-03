"""Actuarial check for pieces E and F — the modelling extras and the CLI.

Builds a rate review on the French-motor fixture: a synthetic *current premium*
that is deliberately an incomplete tariff (it knows the driver's age and the
fuel type, and nothing else), a loss amount to price against, and a model fitted
with ``log(current premium)`` as its offset. It then prints, or with ``--write``
regenerates ``docs/checks/e-f-extras-cli.md``:

* what the multiplier table means when the offset is the premium charged today
  (the base rate is the overall rate change; each relativity is a differential
  change), with the factors the current tariff misses coming out large and the
  one it already charges for coming out near 1.00;
* the algebraic identity behind the setup (offset = log P against a
  P-weighted model) and the conditions it holds under;
* the target-loss-ratio solve: what the book earns today, what the base rate
  becomes at a target of 65 %, and what that does to the premium;
* an unpenalised factor and a binomial (lapse) model, both by the numbers;
* a real transcript of the ``easy-glm`` command line on the same project.

Usage: python scripts/checks/e_f_extras_cli.py [--write]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
DOC = ROOT / "docs" / "checks" / "e-f-extras-cli.md"
FIXTURE = ROOT / "tests" / "fixtures" / "french_motor_50k.parquet"

PREDICTORS = ["DrivAge", "VehAge", "BonusMalus", "Region", "VehGas"]
KNOTS = {
    "DrivAge": [25.0, 35.0, 45.0, 60.0],
    "BonusMalus": [55.0, 60.0, 70.0, 100.0],
    "VehAge": [2.0, 6.0, 12.0],
}
#: the factors today's tariff does not charge for — the gaps the review finds
NOT_PRICED_TODAY = ("BonusMalus", "Region")
#: everything else is already in the price, so its multiplier should be ~1.00
PRICED_TODAY = tuple(v for v in PREDICTORS if v not in NOT_PRICED_TODAY)
ALPHA = 0.0005
#: the loss ratio the synthetic tariff is set to earn today, and the one the
#: review is asked to price to
CURRENT_LOSS_RATIO = 0.80
TARGET_LOSS_RATIO = 0.65
#: deliberately heavy, so the lasso has visibly thinned the region table
PENALTY_DEMO_ALPHA = 0.5


# --------------------------------------------------------------------------
# the book
# --------------------------------------------------------------------------
def book() -> pl.DataFrame:
    """The fixture plus a loss amount and a current premium.

    ``Loss`` is a claim count times a Gamma severity, so it is a pure premium
    with the shape a real one has: a mass of zeros and a long tail.

    ``CurrentPremium`` is built as a real tariff would be — a model of that loss
    on every rating factor — and then **flattened in bonus-malus and region**,
    the two factors this book does not charge for. That is the situation a rate
    review is normally in: most of the price is right, one or two factors are
    missing, and the job is to find them. Building it from a fit rather than
    inventing coefficients matters: it makes "already in the price" mean exactly
    1.00 in the review, so the reader can tell a gap from a confirmation.
    The whole tariff is then scaled to earn a loss ratio of
    ``CURRENT_LOSS_RATIO`` today, so the target-loss-ratio solve has work to do.
    """
    from easy_glm import DesignSpec, fit_glm, to_rate_model

    df = pl.read_parquet(FIXTURE)
    rng = np.random.default_rng(20260903)
    n = df.height
    severity = rng.gamma(shape=2.0, scale=1200.0, size=n)
    df = df.with_columns(
        pl.Series("Loss", df["ClaimNb"].to_numpy() * severity),
        pl.Series("logExposure", np.log(np.maximum(df["Exposure"].to_numpy(), 1e-6))),
        pl.Series("traintest", (rng.random(n) < 0.7).astype(np.int64)),
    )
    spec = DesignSpec.from_data(df, PREDICTORS, knots=KNOTS)
    tariff = fit_glm(
        df,
        spec,
        "Loss",
        family="tweedie",
        tweedie_power=1.5,
        offset_col="logExposure",
        alpha=1e-5,
    )
    rm = to_rate_model(tariff)
    rm.metadata.offset_col = "logExposure"
    for var in NOT_PRICED_TODAY:  # the tariff is flat in these
        for row in list(rm.variables[var].table):
            rm.update_relativity(var, row.from_, row.to_, 1.0)
    flat = rm.predict(df, exposure_col=None)
    scale = float(df["Loss"].sum()) / (CURRENT_LOSS_RATIO * float(flat.sum()))
    return df.with_columns(pl.Series("CurrentPremium", flat * scale))


def project(data_path: Path, name: str = "rate-review"):
    from easy_glm.workflow import Project, VariableDesign

    p = Project(name=name)
    p.data.source.type = "parquet"
    p.data.source.path = str(data_path)
    p.data.roles = {
        "Loss": "target",
        "CurrentPremium": "current_premium",
        "Exposure": "ignore",
        "logExposure": "ignore",
        "ClaimNb": "ignore",
        "IDpol": "id",
        "Density": "ignore",
        "VehPower": "ignore",
        "VehBrand": "ignore",
        "Area": "ignore",
        **dict.fromkeys(PREDICTORS, "predictor"),
    }
    p.data.filters = ["pl.col('CurrentPremium') > 0"]
    p.data.split.mode = "column"
    p.data.split.column = "traintest"
    p.data.split.train_value = 1
    for var, knots in KNOTS.items():
        p.design.variables[var] = VariableDesign(knots=list(knots))
    p.new_model(
        "change",
        family="tweedie",
        tweedie_power=1.5,
        predictors=list(PREDICTORS),
    )
    p.models["change"].penalty.alpha = ALPHA
    p.models["change"].penalty.cv = None
    return p


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def _md_table(header: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(header) + " |"
    rule = "|" + "|".join("---" for _ in header) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, rule, *body])


def multiplier_tables(run, train: pl.DataFrame) -> str:
    """One markdown table per rating factor, with the training exposure of each
    band so a multiplier read on three policies is visible as such."""
    from easy_glm.engine.models import level_label

    out = []
    for var in PREDICTORS:
        cfg = run.rate_model.variables[var]
        rows = []
        for row in cfg.table:
            label = level_label(row, cfg.other_label)
            mask = _mask(train, var, row)
            rows.append(
                [
                    label,
                    f"{row.relativity:.3f}",
                    f"{100 * (row.relativity - 1):+.1f}%",
                    f"{int(mask.sum()):,}",
                ]
            )
        note = (
            " *(the current tariff already charges for this)*"
            if var in PRICED_TODAY
            else ""
        )
        out.append(
            f"**{var}**{note}\n\n"
            + _md_table(["band", "multiplier", "change", "policies"], rows)
        )
    return "\n\n".join(out)


def _mask(df: pl.DataFrame, var: str, row) -> np.ndarray:
    """Rows of ``df`` that fall in one rate-table row."""
    values = df[var]
    if row.from_ is None and row.to_ is None:
        return values.is_null().to_numpy()
    if values.dtype == pl.Utf8:
        return (values == row.from_).fill_null(False).to_numpy()
    x = values.cast(pl.Float64).to_numpy()
    lo = -np.inf if row.from_ is None else float(row.from_)
    hi = np.inf if row.to_ is None else float(row.to_)
    with np.errstate(invalid="ignore"):
        return (x >= lo) & (x < hi)


def spread(run, var: str) -> float:
    """Largest multiplier ÷ smallest, over the bands with real exposure."""
    values = [
        r.relativity
        for r in run.rate_model.variables[var].table
        if r.relativity > 0 and not (r.from_ is None and r.to_ is None)
    ]
    return max(values) / min(values) if values else float("nan")


# --------------------------------------------------------------------------
# the numbers
# --------------------------------------------------------------------------
def offset_identity() -> dict[str, float]:
    """The identity of plan §R6/S1, measured: an offset model and a
    premium-weighted one, Poisson, unscaled, alpha × sum(P)/n."""
    from easy_glm import DesignSpec, fit_glm
    from easy_glm.core.design import CategoricalEncoder, StepEncoder

    df = book().with_columns(pl.col("CurrentPremium").log().alias("logP"))
    spec = DesignSpec(
        {
            "BonusMalus": StepEncoder("BonusMalus", [55.0, 60.0, 70.0, 100.0]),
            "VehGas": CategoricalEncoder("VehGas", ["Regular", "Diesel"]),
        }
    )
    alpha = 1e-7  # small enough that the coefficients are not all lassoed away
    scale = float(df["CurrentPremium"].sum()) / df.height
    kw = {"scale_predictors": False, "gradient_tol": 1e-12, "max_iter": 100_000}
    a = fit_glm(
        df,
        spec,
        "ClaimNb",
        family="poisson",
        offset_col="logP",
        alpha=alpha * scale,
        **kw,
    )
    b = fit_glm(
        df,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="CurrentPremium",
        divide_target_by_weight=True,
        alpha=alpha,
        **kw,
    )
    # Gamma needs a strictly positive target: the claim-free policies are
    # dropped, which is how a severity model is fitted anyway
    claims = df.filter(pl.col("Loss") > 0)
    gamma_scale = float(claims["CurrentPremium"].sum()) / claims.height
    gamma_a = fit_glm(
        claims,
        spec,
        "Loss",
        family="gamma",
        offset_col="logP",
        alpha=alpha * gamma_scale,
        scale_predictors=False,
    )
    gamma_b = fit_glm(
        claims,
        spec,
        "Loss",
        family="gamma",
        weight_col="CurrentPremium",
        divide_target_by_weight=True,
        alpha=alpha,
        scale_predictors=False,
    )
    return {
        "scale": scale,
        "poisson": float(np.max(np.abs(a.coef - b.coef))),
        "gamma": float(np.max(np.abs(gamma_a.coef - gamma_b.coef))),
        "size": float(np.max(np.abs(a.coef))),
    }


def unpenalised_region(train: pl.DataFrame) -> dict[str, int]:
    """How many Region levels survive a heavy lasso with and without
    ``penalty_weight = 0``."""
    from easy_glm import DesignSpec, fit_glm

    out = {}
    for label, weights in (("penalised", {}), ("unpenalised", {"Region": 0.0})):
        spec = DesignSpec.from_data(
            train,
            ["Region", "BonusMalus"],
            knots={"BonusMalus": [55.0, 60.0, 70.0, 100.0]},
            penalty_weight=weights,
        )
        fit = fit_glm(
            train,
            spec,
            "Loss",
            family="tweedie",
            weight_col="Exposure",
            alpha=PENALTY_DEMO_ALPHA,
        )
        kept = sum(
            1
            for f, c in zip(spec.features, fit.coef, strict=True)
            if f.variable == "Region" and f.kind == "level" and c != 0
        )
        total = sum(
            1 for f in spec.features if f.variable == "Region" and f.kind == "level"
        )
        out[label] = kept
        out["levels"] = total
    return out


def lapse_model(df: pl.DataFrame) -> dict[str, float]:
    """A binomial model on a synthetic lapse flag: the odds relativity of the
    widest band and the probability range the scorer returns."""
    from easy_glm import DesignSpec, fit_glm, rate_tables, to_rate_model

    rng = np.random.default_rng(11)
    age = df["DrivAge"].to_numpy().astype(float)
    p = 1.0 / (1.0 + np.exp(-(-1.2 + 0.02 * (45.0 - age))))
    frame = df.with_columns(pl.Series("Lapsed", (rng.random(df.height) < p) * 1.0))
    spec = DesignSpec.from_data(
        frame, ["DrivAge", "Region"], knots={"DrivAge": [25.0, 35.0, 45.0, 60.0]}
    )
    fit = fit_glm(frame, spec, "Lapsed", family="binomial", alpha=0.001)
    rm = to_rate_model(fit)
    pred = rm.predict(frame, exposure_col=None)
    table = rate_tables(fit)["DrivAge"]
    odds = table["relativity"].to_numpy()
    return {
        "label": rm.relativity_label,
        "odds_min": float(odds.min()),
        "odds_max": float(odds.max()),
        "p_min": float(pred.min()),
        "p_max": float(pred.max()),
        "exact": float(np.max(np.abs(pred - fit.predict(frame)))),
        "base_odds": rm.base_rate,
    }


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------
def cli_transcript(project_path: Path, out: Path) -> str:
    """Run the real commands and capture exactly what they print."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    broken = project_path.with_name("broken.json")
    _break(project_path, broken)
    blocks = []
    for argv, shown in (
        (["validate", str(project_path)], "easy-glm validate rate-review.json"),
        (
            ["run", str(project_path), "--out", str(out)],
            "easy-glm run rate-review.json --out artefacts/",
        ),
        (["validate", str(broken)], "easy-glm validate broken.json"),
    ):
        proc = subprocess.run(
            [sys.executable, "-m", "easy_glm.cli", *argv],
            capture_output=True,
            text=True,
            env=env,
        )
        text = (proc.stdout + proc.stderr).replace(
            str(project_path), "rate-review.json"
        )
        text = text.replace(str(broken), "broken.json").replace(str(out), "artefacts")
        blocks.append(
            f"```console\n$ {shown}\n{text.rstrip()}\n"
            f"$ echo $?\n{proc.returncode}\n```"
        )
    files = "\n".join(
        f"  {f.name}  ({f.stat().st_size // 1024} kB)" for f in sorted(out.iterdir())
    )
    blocks.append(f"```console\n$ ls artefacts/\n{files}\n```")
    return "\n\n".join(blocks)


def _break(src: Path, dest: Path) -> None:
    from easy_glm.workflow import Project

    p = Project.from_json(src)
    p.models["change"].predictors = ["DrivAge", "NotAColumn"]
    p.models["change"].tweedie_power = 2.5
    p.to_json(dest)


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------
def build() -> str:
    from easy_glm.workflow import (
        prepare,
        run_model,
        solve_base_rate,
        totals,
        train_holdout,
    )

    tmp = Path(tempfile.mkdtemp(prefix="ef-check-"))
    data_path = tmp / "book.parquet"
    book().write_parquet(data_path)
    p = project(data_path)
    p.to_json(tmp / "rate-review.json")
    df = prepare(p)
    train, holdout = train_holdout(df, p.data.split)
    run = run_model(p, df, "change")

    premium_total = float(df["CurrentPremium"].sum())
    loss_total = float(df["Loss"].sum())
    _, expected, _ = totals(df, run.config, run.predict(df))
    fitted_lr = float(expected.sum()) / premium_total

    solved = solve_base_rate(run, train, TARGET_LOSS_RATIO)
    p.models["change"].base_rate_override = solved
    priced = run_model(p, df, "change")
    _, indicated, _ = totals(df, priced.config, priced.predict(df))
    overall_change = float(indicated.sum()) / premium_total
    achieved_lr = loss_total / float(indicated.sum())
    solved_again = solve_base_rate(priced, train, TARGET_LOSS_RATIO)

    identity = offset_identity()
    penalties = unpenalised_region(train)
    lapse = lapse_model(train)
    out = tmp / "artefacts"
    transcript = cli_transcript(tmp / "rate-review.json", out)

    return DOC_TEXT.format(
        rows=f"{df.height:,}",
        train_rows=f"{train.height:,}",
        holdout_rows=f"{holdout.height:,}",
        premium_total=f"{premium_total:,.0f}",
        loss_total=f"{loss_total:,.0f}",
        current_lr=f"{100 * loss_total / premium_total:.1f}%",
        fitted_lr=f"{100 * fitted_lr:.1f}%",
        base=f"{run.rate_model.base_rate:.4f}",
        tables=multiplier_tables(run, train),
        bm_spread=f"{spread(run, 'BonusMalus'):.2f}",
        region_spread=f"{spread(run, 'Region'):.2f}",
        age_spread=f"{spread(run, 'DrivAge'):.2f}",
        vehage_spread=f"{spread(run, 'VehAge'):.2f}",
        gas_spread=f"{spread(run, 'VehGas'):.2f}",
        target=f"{100 * TARGET_LOSS_RATIO:.0f}%",
        solved=f"{solved:.4f}",
        overall_change=f"{100 * (overall_change - 1):+.1f}%",
        achieved_lr=f"{100 * achieved_lr:.2f}%",
        solved_again=f"{solved_again:.4f}",
        alpha=f"{ALPHA:g}",
        identity_scale=f"{identity['scale']:,.0f}",
        identity_poisson=f"{identity['poisson']:.1e}",
        identity_gamma=f"{identity['gamma']:.3f}",
        identity_size=f"{identity['size']:.3f}",
        region_levels=str(penalties["levels"]),
        region_penalised=str(penalties["penalised"]),
        region_unpenalised=str(penalties["unpenalised"]),
        lapse_odds_min=f"{lapse['odds_min']:.3f}",
        lapse_odds_max=f"{lapse['odds_max']:.3f}",
        lapse_p_min=f"{lapse['p_min']:.3f}",
        lapse_p_max=f"{lapse['p_max']:.3f}",
        lapse_base=f"{lapse['base_odds']:.4f}",
        lapse_exact=f"{lapse['exact']:.1e}",
        transcript=transcript,
    )


DOC_TEXT = """# E and F — the rate review, the modelling extras, and the command line

*Regenerate with `python scripts/checks/e_f_extras_cli.py --write`. Every number
below is computed by that script; nothing here is typed in by hand.*

## What was built

Four small modelling features, and a command line.

1. **A rate review.** Tell easy_glm which column is *the premium you charge
   today* and the model fits the **change** from it, not the price from scratch.
2. **A target loss ratio.** Type the loss ratio you want to write the book at
   and the overall level that achieves it is solved for you.
3. **Per-factor penalty weights.** Tell the lasso to leave one factor alone, or
   to shrink it harder than the rest.
4. **Tweedie power and lapse models.** The Tweedie power is a box on the Model
   page; binomial models give odds relativities and predict probabilities.
5. **`easy-glm` on the command line**, so a refit does not need a browser.

---

## 1. The rate review

The book is the French-motor fixture ({rows} policies, {train_rows} training and
{holdout_rows} holdout) with two columns added. **Loss** is a claim count times
a Gamma severity — a pure premium with the shape a real one has. **Current
premium** is a tariff built the way a real one would be, a model of that loss on
every rating factor, and then deliberately **flattened in bonus-malus and
region**: this book does not charge for either. Everything else — driver age,
vehicle age, fuel — is already correctly priced. That is the position a rate
review is usually in, and it is what lets us tell a *gap* from a *confirmation*
further down.

The book earns **{premium_total}** of premium against **{loss_total}** of loss:
a loss ratio of **{current_lr}** today, against a target of {target}.

**How the setup is made.** On the Variables page the premium column is given the
role **current premium**. easy_glm derives one extra column,
`log_CurrentPremium`, and every new model uses it as its *offset*. Nothing else
changes: pick predictors and fit as usual. The derivation is written into the
exported Python script, so it is visible rather than implied by a role:

```python
# CurrentPremium is the premium charged today; its log is the model's
# offset, so the base rate is the overall rate change and every
# relativity is a multiplier on the current premium
df = df.with_columns(pl.col('CurrentPremium').cast(pl.Float64).log()
                     .alias('log_CurrentPremium'))
```

Rows with a premium of zero or less are filtered out first
(`CurrentPremium > 0`). If any are left, the tool refuses to prepare the data
and says how many there are, rather than letting a missing logarithm turn into
a fit that fails deep inside the solver.

### What the table means

Because the offset is the premium, every number in the rate tables is a
**multiplier on the premium you charge today**, and a relativity of 1.00 means
"this band changes by the same amount as the base risk and no more". The Rate
tables page, the Export page and the `Summary` sheet of the Excel workbook all
say so in as many words, so a multiplier cannot be mistaken for a rate.

Straight from the fit, before any target is applied, the model predicts the
**expected loss**: {fitted_lr} of the premium the book charges, against an
actual {current_lr}. The two are not the same because a Tweedie fit balances the
*deviance*, not the totals — only a Poisson one balances totals exactly — which
is one reason the level of a rate change is set separately rather than taken
from the fit. That is section 2.

{tables}

### Reading it

* **BonusMalus** spreads {bm_spread}x from its cheapest band to its dearest, and
  **Region** {region_spread}x. Neither is in the current tariff, so this is what
  the review is about: a policy at bonus-malus 100 or more is paying the same as
  one at 50 and should not be.
* **DrivAge** spreads only {age_spread}x, **VehAge** {vehage_spread}x and
  **VehGas** {gas_spread}x — the factors the current premium already charges
  for. Close to 1.00 is the right answer, and it is the check that the setup is
  working: the model is saying "what you already do here is about right". A
  from-scratch model would show those factors' full effect and you could not
  tell "already priced" from "no effect" at all.
* The exposure column is there to stop you reading a multiplier off three
  policies. The thinnest region bands here carry a few hundred; treat their
  numbers as direction, not as a price.

### With an interaction as well

Nothing about this changes if the model also has a two-way interaction. Such a
model is fitted in two stages (the answer to Q5): the main effects first, then
the interaction cells as pure adjustments on top of them. The premium offset
belongs to the first stage, so the multiplier tables above are the same numbers
whether or not an interaction is in the model, and the cells sit on top of them
as further adjustments. The scorer still reproduces the model exactly.

### The algebra behind it

Offsetting by `log(premium)` is the same model as fitting `loss / premium`
weighted by `premium` — that is why the setup is standard. It is an exact
identity only under conditions worth stating, because a builder who does not
state them ends up loosening a test until it passes:

* the **Poisson** deviance — the only one invariant under the swap;
* column standardisation **off**, because standardising uses the weights and the
  two models have different weights;
* the offset model's penalty multiplied by the mean premium (`sum(P)/n` =
  {identity_scale} here), because the solver divides its objective by the sum of
  the weights.

Measured on this book, where the largest coefficient is {identity_size}: the two
fits agree to **{identity_poisson}** on every coefficient. The same pair on a **Gamma** target differ by **{identity_gamma}**.
They are genuinely different models, and that is expected. If you fit severity
or pure premium against a current premium, the **offset** form is the one to
trust; the weighted form quietly changes the shape of the likelihood.

---

## 2. Pricing to a target loss ratio

The base rate multiplies every prediction, so the level that hits a target is
arithmetic, not a search. The box on the Model page asks for the loss ratio you
want and sets the base rate so that, on the training rows,

> total actual ÷ total expected = the number you typed.

For a rate-change model that sentence *is* the loss ratio, because the model's
prediction is the price and the actual is the loss. Asking for **{target}** on
this book gives:

> **base rate {solved}**, and the book as a whole moves **{overall_change}**.

Checking it on the whole book, including the holdout rows the solve never saw:
loss ÷ indicated premium is **{achieved_lr}**. Three things worth knowing:

* **The relativities do not move.** Only the overall level changes, so every
  differential you agreed stays exactly as it was.
* **Solving again is safe.** Re-solving from the answer gives {solved_again} —
  the same number. The solve reads the model's *current* base rate, so an
  override already in place cancels out. Type a different target, or come back
  tomorrow, without resetting anything first.
* **The base rate is the change for the base risk**, the policy that sits at
  relativity 1.00 on every factor — not the average policy. The overall move
  quoted above ({overall_change}) is the total indicated premium against the
  total premium charged today. The two differ because the base risk is not the
  average risk. Question 2 below asks whether you would rather the table were
  rebased so the two are the same number.

Where there is no current-premium column the same box balances the model against
its own target: 1.00 means total expected equals total actual, the ordinary
tidy-up after hand-editing relativities.

---

## 3. A factor the lasso may not touch

Every factor now has a **penalty weight** on the Design page: 1 is "like
everything else", 2 is "shrink twice as hard", **0 is "do not shrink at all"**.
The lasso exists to thin out factors the data does not support, but sometimes
you have decided to charge for something — a territory table you have committed
to, a factor a regulator expects — and you want its full signal.

Region has {region_levels} levels here. Under a deliberately heavy penalty the
lasso keeps **{region_penalised}** of them; with the penalty weight set to 0 it
keeps **{region_unpenalised}** — all of them. Nothing else about the model
changes.

---

## 4. Tweedie power, and lapse models

**Tweedie power** is a box on the Model page, shown when the family is Tweedie.
It runs between 1 (Poisson — all about how *often* claims happen) and 2 (Gamma —
all about how *large* they are); 1.5 is the default and the usual starting point
for a pure premium. It is saved with the model and written into the exported
script, so next quarter's refit uses the same one.

**Binomial models** — lapse, conversion, any yes/no — now produce rate tables
too. The numbers multiply the **odds**, not the probability, and every page and
the Excel summary label them "odds relativity" so they cannot be read as
probabilities. On a synthetic lapse model here the age bands run from
**{lapse_odds_min}** to **{lapse_odds_max}** times the base odds of
**{lapse_base}**, and the scorer converts back: it returns probabilities between
**{lapse_p_min}** and **{lapse_p_max}**, matching the GLM to {lapse_exact}. A
probability is not an amount, so such a model **refuses** to be multiplied by
exposure — asking for it is an error message, not a silently meaningless number.

---

## 5. The command line

Everything above can be done without a browser. This is a real transcript.

{transcript}

`run` writes four files: the `.easyglm` scorer, the Excel rate tables, a Python
script that rebuilds the model, and the self-contained HTML report. `export`
writes any subset (`--script`, `--report`, `--excel`), `validate` checks a
project without fitting, and `workbench` opens the browser tool on it. Anything
wrong is a message and a non-zero exit code — never a stack trace — so a
scheduled job can tell success from failure.

Two things worth knowing:

* **Every command fits the model afresh.** It does not reuse a fit made in the
  browser. That costs a fit, and buys a script with every knot, level and
  penalty written out explicitly, so it produces the same model wherever it runs.
* **Two runs are not byte-identical**, and that is not a fault. Each saved
  version of the tables records the time it was written, and the solver
  underneath adds numbers in whatever order its threads finish in, so the last
  floating-point digit of a relativity can differ (measured: 9 x 10^-16 between
  two fits in the *same* process). The predictions agree to twelve digits, which
  is what matters.

---

## Questions for you

1. **The multiplier table.** With the offset set to the current premium, each
   relativity is a differential change and the level is carried by the base
   rate. Is that how you want to read it, or would you rather every band showed
   its *total* change (overall x differential) with the split in a second column?
2. **Where the level sits.** The base rate is the change for the *base risk* —
   the policy at relativity 1.00 on every factor — so it is not the same number
   as the book's overall move ({solved} against {overall_change} here). Should a
   rate-change model rebase its tables so the base rate **is** the overall move,
   with every relativity divided through by the exposure-weighted average?
3. **Which rows the solve balances on.** It uses the **training** rows, because
   that is the data the model was fitted on, and the holdout is then an honest
   check ({achieved_lr} against a {target} target here). Would you rather it
   balanced on the whole book, or be asked each time?
4. **Already-priced factors.** A factor the current tariff already charges for
   comes back near 1.00 and the exact numbers are mostly noise. Should the
   workbench flag a factor whose multipliers are all within, say, +/-2 % as
   "already in the price", or leave you to see it?
5. **Unpenalised factors.** The penalty weight is one number per factor. Is
   0 / 1 / 2 the right granularity, or would named settings ("keep in full",
   "normal", "shrink hard") be easier to use?
6. **Tweedie power.** It is a number you type, defaulting to 1.5. Would you want
   easy_glm to *estimate* it as a starting point (a profile likelihood over a
   few values), or is a typed number the right level of control?
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(SRC))
    text = build()
    if args.write:
        DOC.parent.mkdir(parents=True, exist_ok=True)
        DOC.write_text(text)
        print(f"wrote {DOC} ({len(text.encode()) // 1024} kB)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
