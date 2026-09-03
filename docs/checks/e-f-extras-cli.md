# E and F — the rate review, the modelling extras, and the command line

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

The book is the French-motor fixture (50,000 policies, 34,975 training and
15,025 holdout) with two columns added. **Loss** is a claim count times
a Gamma severity — a pure premium with the shape a real one has. **Current
premium** is a tariff built the way a real one would be, a model of that loss on
every rating factor, and then deliberately **flattened in bonus-malus and
region**: this book does not charge for either. Everything else — driver age,
vehicle age, fuel — is already correctly priced. That is the position a rate
review is usually in, and it is what lets us tell a *gap* from a *confirmation*
further down.

The book earns **5,943,826** of premium against **4,755,061** of loss:
a loss ratio of **80.0%** today, against a target of 65%.

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
**expected loss**: 90.0% of the premium the book charges, against an
actual 80.0%. The two are not the same because a Tweedie fit balances the
*deviance*, not the totals — only a Poisson one balances totals exactly — which
is one reason the level of a rate change is set separately rather than taken
from the fit. That is section 2.

**DrivAge** *(the current tariff already charges for this)*

| band | multiplier | change | policies |
|---|---|---|---|
| < 25.0 | 1.128 | +12.8% | 1,521 |
| [25.0, 35.0) | 1.123 | +12.3% | 7,191 |
| [35.0, 45.0) | 1.184 | +18.4% | 8,925 |
| [45.0, 60.0) | 1.000 | +0.0% | 11,547 |
| ≥ 60.0 | 0.986 | -1.4% | 5,791 |
| Other / Unknown | 1.128 | +12.8% | 0 |

**VehAge** *(the current tariff already charges for this)*

| band | multiplier | change | policies |
|---|---|---|---|
| < 2.0 | 1.086 | +8.6% | 6,693 |
| [2.0, 6.0) | 0.956 | -4.4% | 9,690 |
| [6.0, 12.0) | 1.000 | +0.0% | 10,566 |
| ≥ 12.0 | 0.989 | -1.1% | 8,026 |
| Other / Unknown | 1.086 | +8.6% | 0 |

**BonusMalus**

| band | multiplier | change | policies |
|---|---|---|---|
| < 55.0 | 1.000 | +0.0% | 22,032 |
| [55.0, 60.0) | 1.622 | +62.2% | 1,838 |
| [60.0, 70.0) | 1.917 | +91.7% | 3,702 |
| [70.0, 100.0) | 2.260 | +126.0% | 6,014 |
| ≥ 100.0 | 4.748 | +374.8% | 1,389 |
| Other / Unknown | 1.000 | +0.0% | 0 |

**Region**

| band | multiplier | change | policies |
|---|---|---|---|
| Centre | 1.000 | +0.0% | 8,209 |
| Rhone-Alpes | 1.580 | +58.0% | 4,464 |
| Provence-Alpes-Cotes-D'Azur | 1.704 | +70.4% | 4,085 |
| Ile-de-France | 1.608 | +60.8% | 3,636 |
| Bretagne | 1.298 | +29.8% | 2,219 |
| Pays-de-la-Loire | 1.030 | +3.0% | 1,960 |
| Languedoc-Roussillon | 1.075 | +7.5% | 1,829 |
| Aquitaine | 1.188 | +18.8% | 1,552 |
| Nord-Pas-de-Calais | 0.878 | -12.2% | 1,389 |
| Poitou-Charentes | 1.539 | +53.9% | 994 |
| Midi-Pyrenees | 0.707 | -29.3% | 881 |
| Lorraine | 0.669 | -33.1% | 675 |
| Basse-Normandie | 1.418 | +41.8% | 554 |
| Bourgogne | 0.878 | -12.2% | 494 |
| Haute-Normandie | 0.936 | -6.4% | 466 |
| Picardie | 1.438 | +43.8% | 432 |
| Auvergne | 0.550 | -45.0% | 279 |
| Corse | 0.385 | -61.5% | 257 |
| Limousin | 0.544 | -45.6% | 241 |
| Champagne-Ardenne | 0.569 | -43.1% | 162 |
| Alsace | 1.528 | +52.8% | 114 |
| Other / Unknown | 1.360 | +36.0% | 0 |

**VehGas** *(the current tariff already charges for this)*

| band | multiplier | change | policies |
|---|---|---|---|
| Regular | 1.000 | +0.0% | 17,778 |
| Diesel | 1.011 | +1.1% | 17,197 |
| Other / Unknown | 1.000 | +0.0% | 0 |

### Reading it

* **BonusMalus** spreads 4.75x from its cheapest band to its dearest, and
  **Region** 4.43x. Neither is in the current tariff, so this is what
  the review is about: a policy at bonus-malus 100 or more is paying the same as
  one at 50 and should not be.
* **DrivAge** spreads only 1.20x, **VehAge** 1.14x and
  **VehGas** 1.01x — the factors the current premium already charges
  for. Close to 1.00 is the right answer, and it is the check that the setup is
  working: the model is saying "what you already do here is about right". A
  from-scratch model would show those factors' full effect and you could not
  tell "already priced" from "no effect" at all.
* The exposure column is there to stop you reading a multiplier off three
  policies. The thinnest region bands here carry a few hundred; treat their
  numbers as direction, not as a price.

### The algebra behind it

Offsetting by `log(premium)` is the same model as fitting `loss / premium`
weighted by `premium` — that is why the setup is standard. It is an exact
identity only under conditions worth stating, because a builder who does not
state them ends up loosening a test until it passes:

* the **Poisson** deviance — the only one invariant under the swap;
* column standardisation **off**, because standardising uses the weights and the
  two models have different weights;
* the offset model's penalty multiplied by the mean premium (`sum(P)/n` =
  119 here), because the solver divides its objective by the sum of
  the weights.

Measured on this book, where the largest coefficient is 0.862: the two
fits agree to **5.6e-12** on every coefficient. The same pair on a **Gamma** target differ by **0.281**.
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
prediction is the price and the actual is the loss. Asking for **65%** on
this book gives:

> **base rate 0.6809**, and the book as a whole moves **+22.3%**.

Checking it on the whole book, including the holdout rows the solve never saw:
loss ÷ indicated premium is **65.41%**. Three things worth knowing:

* **The relativities do not move.** Only the overall level changes, so every
  differential you agreed stays exactly as it was.
* **Solving again is safe.** Re-solving from the answer gives 0.6809 —
  the same number. The solve reads the model's *current* base rate, so an
  override already in place cancels out. Type a different target, or come back
  tomorrow, without resetting anything first.
* **The base rate is the change for the base risk**, the policy that sits at
  relativity 1.00 on every factor — not the average policy. The overall move
  quoted above (+22.3%) is the total indicated premium against the
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

Region has 20 levels here. Under a deliberately heavy penalty the
lasso keeps **8** of them; with the penalty weight set to 0 it
keeps **20** — all of them. Nothing else about the model
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
**0.709** to **1.722** times the base odds of
**0.2637**, and the scorer converts back: it returns probabilities between
**0.138** and **0.395**, matching the GLM to 5.6e-17. A
probability is not an amount, so such a model **refuses** to be multiplied by
exposure — asking for it is an error message, not a silently meaningless number.

---

## 5. The command line

Everything above can be done without a browser. This is a real transcript.

```console
$ easy-glm validate rate-review.json
rate-review.json: valid · models: change
$ echo $?
0
```

```console
$ easy-glm run rate-review.json --out artefacts/
rate-review · model change · tweedie
  rows          train 34,975 · holdout 15,025
  penalty       alpha 0.0005 · 33 of 37 terms non-zero
  base rate     0.501021  (each table entry is a multiplier on current premium)
  offset        log_CurrentPremium
  train         A/E 0.8834 · Gini 0.3927 · deviance explained 5.35%
  holdout       A/E 0.9019 · Gini 0.3713 · deviance explained 5.16%
written:
  artefacts/rate-review_change.easyglm
  artefacts/rate-review_change_rate_tables.xlsx
  artefacts/rate-review_change.py
  artefacts/rate-review_change_report.html
$ echo $?
0
```

```console
$ easy-glm validate broken.json
easy-glm: broken.json has 3 problem(s)
  - change: predictor(s) not in the data: ['NotAColumn']
  - change: tweedie_power must be strictly between 1 and 2 (1 = Poisson, 2 = Gamma), got 2.5
  - change: not predictor-role columns: ['NotAColumn']
$ echo $?
1
```

```console
$ ls artefacts/
  rate-review_change.easyglm  (12 kB)
  rate-review_change.py  (2 kB)
  rate-review_change_rate_tables.xlsx  (14 kB)
  rate-review_change_report.html  (116 kB)
```

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
   as the book's overall move (0.6809 against +22.3% here). Should a
   rate-change model rebase its tables so the base rate **is** the overall move,
   with every relativity divided through by the exposure-weighted average?
3. **Which rows the solve balances on.** It uses the **training** rows, because
   that is the data the model was fitted on, and the holdout is then an honest
   check (65.41% against a 65% target here). Would you rather it
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
