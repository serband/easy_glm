# Review of piece B2 — slope-penalised linear basis, monotone on linear terms, the `continuous` kind

*Reviewer: independent. Worktree `piece/b2`, commits `f38bd89`…`7efdd51` (`git diff f0508e8..HEAD`).
Contract: `docs/RELEASE_0.4_PLAN.md` §R2 (unchanged parts) and **R10** (Q3 → the basis penalises
slopes; monotone as sign bounds on slopes) plus the Q9 bullet (the `continuous` kind);
`docs/checks/00-questions-for-the-actuary.md` Q3 / Q9; the prior review `docs/reviews/b-linear.md`
(band-edit node rule, `x_base`, clamp rules — all of which must still hold). Date 2026-09-03.*

## 1. Verdict

**Changes requested — one blocking item, in the workbench's run cache, not in the mathematics.**

The mathematics is right and I could not break it. `beta_j` **is** band `j`'s slope — the
table's `slope` column equals the model's own coefficients to `0.0` (bit-identical), with
`scale_predictors=True` *and* with `False`, so glum's standardisation is correctly undone
before the table is read. The table `(from, to, relativity at from, slope)` reproduces
`exp(intercept + Σ βⱼ·overlapⱼ(x))` to **1.1e-16** on the coefficients and the rate model
matches the GLM to **1.4e-15** on a 40-value adversarial frame (`±1e12`, one ulp either side of
`lo`, of every knot and of `hi`, `±inf`, null, an unseen level, with an offset and a
`Mileage × Region` interaction), **1.8e-15** on a 3,000-row holdout and the same with Int64
input. Outside the clamp the curve is *exactly* flat (relative difference `0.0e+00`, not
merely small). `x_base` still holds: relativity 1.0000 at the base row and
`base_rate(fit) == fit.predict(base risk)` to 3e-16.

Monotone does what R10 asks: `increasing` puts `lower = 0` on exactly the seven band columns
of the constrained variable and on nothing else — not the `is null` column, not the step
variable, not the interaction cells — and it binds (a *decreasing* constraint on a truly
increasing planted curve returns all-zero slopes and a table that is 1.0000 everywhere).
`Project.validate`, the Design grid, the Design detail selector, the Model page and the
exported script all carry it, and it survives the project JSON round trip.

The `continuous` kind is a genuine one-band linear term: four table rows (two flat, one band,
null), spec JSON round-trip identical, `from_rate_tables` and Excel round-trip to `0.0` and
6.7e-16, the node edit rule intact on the single band, and **both** exported scripts (with and
without a run) execute in a subprocess and rebuild the model to 1.1e-15 / 1.3e-15.

The recovery test is honest: with the hinge basis monkeypatched back in, four of its six
assertions fail.

The blocking item is that a fit persisted by the *previous* build is still a cache hit, so the
workbench silently reinterprets hinge coefficients as band slopes. `PERSIST_FORMAT` must be
bumped — this is precisely the case the constant exists for.

## 2. Blocking

### B2-1. `PERSIST_FORMAT` was not bumped: a persisted run from the hinge build is loaded and silently re-read as band slopes

**What.** `src/easy_glm/app/state.py:69` still says `PERSIST_FORMAT = 2`. `run_key()` mixes
the model hash, the data file identity, library versions and that constant — none of which
moves across this change (`version("easy_glm")` is `0.2.2` here and `0.3.0` in
`pyproject.toml`; either way B2 does not move it). The pickled `ModelRun` stores a
`DesignSpec` whose `LinearEncoder` fields (`variable`, `knots`, `clamp`, `null_indicator`)
are unchanged, so unpickling produces a perfectly valid *new-class* encoder, and
`_design_matches` — which compares `spec.feature_names` — therefore compares the new band
names against the new band names and passes. `load_persisted_run` then calls
`rebuild_rate_model`, which re-derives the rate tables from `run.fit` with
`to_rate_model` / `rate_tables`, i.e. reads the old *hinge* coefficients as *band slopes*.

Reproduced end to end. I checked out `src/` at `f0508e8`, fitted a project there
(`Mileage` linear, knots 8,000 / 20,000), persisted the run, then opened the same project
and the same data file under the worktree through an AppTest:

```
old build : RUN_KEY 724a6cefe7a5081f   feature names ['max(Mileage-0,0)', 'max(Mileage-8000,0)', ...]
new build : RUN_KEY 724a6cefe7a5081f   (identical)   S.get_run("freq") -> cache HIT, no warning
  Mileage slopes  before  [0, 3.834e-05,  3.834e-05, -1.896e-05, 0, 0]
  Mileage slopes  after   [0, 3.834e-05,  0.000e+00, -5.730e-05, 0, 0]
  Mileage relativities after: [0.7359, 0.7359, 1.0, 1.0, 0.5638, 1.2471]
  run.rate_model vs run.fit : 1.32  (the exactness invariant, broken)
```

The middle band went flat and the top-of-range relativity fell from ≈1.30 to 0.5638 — a
curve the model never fitted, presented as the fitted one, with the "fitted" badge on and no
refit. Everything downstream inherits it: A/E and Gini on the Diagnostics page, the Rate
tables page, the Excel and `.easyglm` downloads, and the adjustments the actuary then makes
on top of it.

Note the two things that *do* migrate correctly, so the finding is only about the pickle
cache: a `.easyglm` written by the old build scores **bit-identically** under the new one
(max relative difference `0.0` on 3,000 holdout rows — the table format really is unchanged),
and an old project JSON loads and validates clean (`[]`).

**Failure scenario.** The owner fitted his frequency model in the workbench last week, saved
the project and closed the laptop. He pulls this branch, reopens the same project file on the
same data extract and goes straight to Rate tables to carry on editing. The mileage curve he
sees is not the one that was fitted, is not the one this build would fit, and nothing on the
screen says so. He edits relativities on top of it and exports.

**Fix.**

1. `PERSIST_FORMAT = 3` in `src/easy_glm/app/state.py`, with a one-line comment naming B2
   ("the linear basis changed meaning without changing the pickle's shape").
2. Widen the constant's docstring from "whenever the *shape* of a pickled class … changes"
   to "shape **or meaning**" — the shape is exactly what did not change here.
3. `docs/reviews/w1-state.md` S4 asked for the rule to be written into `AGENTS.md`
   ("bump `PERSIST_FORMAT` when …") and it never was (`grep -n PERSIST_FORMAT AGENTS.md` is
   empty). Add it under the workbench-state section, with this piece as the worked example.
4. Add the regression test: pickle a `ModelRun` under the current key, monkeypatch
   `PERSIST_FORMAT` to the *previous* value and assert `load_persisted_run` returns `None`
   and leaves the file alone (`test_app_state.py` already monkeypatches the constant to 99,
   so the mechanism is covered — what is missing is a test that pins the bump).
5. Mention it in the CHANGELOG next to the basis change: "fits cached in a
   `*.easyglm-runs` folder by an earlier 0.4 development build are ignored and refitted;
   `.easyglm` scorers and project files are unaffected."

## 3. Should fix

### S1. The penalty is ~24× cheaper in the thin tail than in the body, and the actuary document does not say so

The document tells the owner (correctly, and under Q10) that at `BonusMalus` 230 the linear
curve charges 89× while the step design charges 4.7×, and explains it as "the linear term
keeps its slope going through the thin region up to the clamp". That reads as *extrapolation
of a slope fitted elsewhere*. It is not: the top band fits its **own** slope, at a large
discount. With `scale_predictors=True` glum penalises `alpha·|beta_j|·sd_j`, so one unit of
log-relativity *rise across band j* costs `alpha·sd_j/w_j`, and `sd_j/w_j` collapses when few
rows reach the band. Measured on the check's own fit (full French motor set, alpha 3e-4):

```
band              width      sd   sd/width   exposure share >= lo    slope     rise (log)
[  50,   53)          3   1.412     0.4708               1.00000   0.13884      0.417
[  85,   95)         10   2.211     0.2211               0.08822   0.04454      0.445
[  95,  230)        135   2.650     0.0196               0.04672   0.02339      3.158
cost of one unit of rise, relative to the first band:
[1.000 0.967 0.932 0.884 0.792 0.728 0.613 0.470 0.042]
```

The top band buys its rise for **4.2 %** of what the first band pays, and duly fits a rise of
3.16 in log (23×) — that *is* the 89×. This is not a regression (the hinge build reached
101×, recorded in the previous `docs/checks/b-linear.md`) and it is not new to B2, but B2 is
where the tail became a per-band fitted number rather than a bend, so the explanation belongs
in the document now. Two things to do:

- One sentence in the "One thing to look at" paragraph: the penalty that keeps the rest of
  the curve flat is roughly 24× weaker in the top band because so few policies reach it, so
  the tail is the *least*-penalised part of the curve, not the most.
- Consider giving band columns the treatment `interaction_penalty_weights` already gives
  interaction cells for the identical reason ("a thin cell (small sd) buys a large raw effect
  for little penalty — the opposite of what a pricing model wants"): `P1 = 0.5·w_j/sd_j` on
  band columns would make one unit of rise cost the same everywhere on the curve. That is a
  behaviour change and a question for the owner — it belongs with Q10 in the check document,
  not in this piece silently.

### S2. For a `continuous` term the base point is always the lower clamp, which empties the Q2 convention

`_modal_bins` picks the most exposed row; a continuous term has exactly one band, so the base
is always `lo` and `x_base == lo` (confirmed: `x_base = 0.0` on both my books and
`Density` in the check). Convention 2 in the document — "Relativity 1.00 sits at the lower
edge of the most exposed band, so the base risk is a round, visible number" — is then
vacuous, and the base *rate* becomes the rate of a policy at the very bottom of the range: in
the check's own table the continuous `Density` column reads 1.0000 at `Density` = 1 while the
book's mass sits near 1,000, so the base rate describes a risk almost nobody is. Say so in
the `continuous` paragraph of `docs/checks/b-linear.md` ("with a single band the 1.00 point is
the lower clamp; read the base rate accordingly"), and consider whether the owner wants the
`base="reference"` option surfaced for these terms.

### S3. The `continuous` script round trip is asserted by string matching only

`test_continuous_kind_is_a_one_band_linear_term` checks
`"LinearEncoder('Mileage', [], clamp=(" in src` and `"knots={'Mileage': []}" in no_run`, but
never runs either script — the subprocess round trip
(`test_run_model_hash_and_exported_script`) still covers only the multi-knot case. I ran both
by hand: with a run, `rc=0` and the rebuilt `.easyglm` matches the workbench to **1.1e-15**;
without a run, `rc=0`, a four-row Mileage table and **1.3e-15**. Cheap to lock in — extend the
existing subprocess test with a `continuous` variant (and keep the monotone direction on it,
which I also verified is written into the script and reproduced).

### S4. The 0.25 curve bound in the planted test is a backstop, and its docstring should say which assertion is doing the work

Measured over six seeds of the planted book (150k rows, alpha 0.02): flat bands exactly 0 on
every seed, 9 of 15 bands non-zero on every seed, segment ratios 0.917–0.962, worst grid gap
**0.140–0.202** against the bound 0.25 (the true curve spans 2.40 in log, so the bound is
~10 % of the span). The gap is dominated by a systematic ~6 % shrinkage that accumulates
monotonically along the curve, which is exactly what assertion 3 already measures — so
assertion 4 catches nothing that 3 does not. Either tighten it to ~0.22 (the docstring's own
eight-seed worst case is 0.209, so 0.25 leaves one seed of headroom, which is thin for a
tolerance and generous for a claim), or compare after dividing out the common shrink factor,
which would make it a real shape test. Not blocking — assertions 1, 3 and 5 are the honest
ones and they are strong.

### S5. `docs/RELEASE_0.4_PLAN.md` §B still specifies the hinge basis

Lines 48, 160 ("Unit: hinge columns") and 353–356 still describe `max(x − k, 0)` as the
design of piece B. R10 supersedes them, but a reader who opens §B first will implement the
wrong thing. Add "(superseded by R10 — see §Revisions)" to those four places. `AGENTS.md`
already carries the "never reintroduce hinge columns" rule, which is the right belt.

### S6. `encoder_for` computes knots for a `continuous` term and throws them away

`src/easy_glm/workflow/run.py`: the knot strategy is evaluated (`integer_knots(series)` /
`quantile_knots(...)`) before the `kind == "continuous"` branch discards it. For
`knots="integer"` on a wide-range variable that is a real allocation for nothing. Move the
`continuous` short-circuit above the strategy block.

### S7. The CHANGELOG's headline improvement is measured against a number no longer in the repository

"the `BonusMalus` curve improves from Gini 0.3091 / 4.88 % deviance explained to 0.3106 /
4.97 %" — 0.3091 / 4.88 % is the *hinge* linear result, which now exists only in the git
history of `docs/checks/b-linear.md` (the current document's step column reads 0.3072 /
4.79 %, so a reader will pair the numbers wrongly). Add "(the hinge basis, earlier in this
release)". Both numbers are correct: I reproduced 0.3106 / 4.97 % and the old document
records 0.3091 / 4.88 %.

## 4. Nits

- **N1.** `_grid` (`pages_design.py`): the monotone rescue line reads
  `new.monotone = vd.monotone if numeric and vd.kind != "categorical" else None` — it tests
  the *old* `vd.kind`, not `new.kind`, so a single grid edit that sets kind → categorical and
  monotone → increasing keeps the previous monotone value instead of clearing it.
  Pre-existing; one-word fix while the line is being touched anyway.
- **N2.** Two knots that differ by less than the display precision produce a band of
  near-zero width whose column name is `x in [3, 3)` (`LinearEncoder("x", [3.0, 3.0000000001],
  (1, 9))` → widths `[2.0, 1e-10, 6.0]`); three such knots would give duplicate column names.
  Exact duplicates *are* deduplicated. `from_rate_tables` refuses a genuinely zero-width band
  but accepts this one. The Design page's custom-knot box can produce it. Consider refusing a
  band narrower than, say, 1e-9 of the clamp range in `__post_init__`.
- **N3.** The stated environment does not match the builder's note "1.63, app tests 167":
  with `/Users/serban/Documents/Projects/easy_glm/.venv` streamlit is **1.57.0** and the
  app-page tests (`test_app`, `test_app_state`, `test_w2_pages`, `test_w3_hardening`) are
  **124, all passing in 5.4 s**. `.venv-313` cannot import streamlit (`No module named
  'anyio'`). Whichever interpreter produced 167 should be named in the piece note, or the
  claim dropped.
- **N4.** `scripts/checks/b_linear.py` without `--write` reproduces `docs/checks/b-linear.md`
  **exactly** except for one trailing blank line from the final `print`. Harmless; strip it if
  `--write` is ever diffed in CI.
- **N5.** A monotone constraint bounds only the main-effect band columns; an interaction on
  the same variable is unconstrained, so `A × B` can still turn the combined curve round for
  some level of `B`. True for step terms too, so it is not new — but now that the constraint
  is offered on the smooth curves an actuary is most likely to constrain, one line in
  `monotone_bounds`' docstring and in the check document ("the direction binds the factor's
  own curve, not the interaction cells on top of it") is worth having.

## 5. Missing tests

- The persisted-run migration (B2-1): a run pickled under the previous `PERSIST_FORMAT` must
  be a cache miss. There is no test today that would have caught the missing bump.
- The `continuous` exported script executed in a subprocess, with a run and without (S3).
- `from_rate_tables` and the Excel round trip on a **one-band** (continuous) table — I ran
  them (`0.0` and 6.7e-16, `x_base` preserved, no warnings), nothing in the suite does.
- A term the constraint flattened completely (every slope 0, every relativity 1.0000): that is
  now a reachable workbench state (`decreasing` on a rising curve) and its table has no unique
  1.0 row. I verified `x_base` still survives Excel and `from_rate_tables` via the `is_base`
  column (12,500.0 both ways) and that editing one of its flat bands re-derives exactly two
  slopes; worth pinning.
- `monotone` on a linear term that is also an interaction parent (N5) — at least record the
  behaviour.
- An AppTest that the Model page draws the "Monotone constraints from the Design page: …"
  caption for a linear term (the code path is new for linear; only the absence of an *error*
  is asserted today).

## 6. What I re-ran

- **Full suite** in the worktree with the stated interpreter and `PYTHONPATH=<worktree>/src`
  (`easy_glm.__file__` confirmed under the worktree): **445 passed in 188 s**, and
  **445 passed in 172 s** again on a second run with the default plugin/ordering set.
  `ruff check .` clean; `black --check .` 87 files unchanged.
  `git diff f0508e8..HEAD -- tests/test_golden.py tests/fixtures` **empty**.
- **Exactness (adversarial).** Model: linear `Mileage` (clamp 0–30,000, six knots), step
  `DrivAge`, categorical `Region`, `Mileage × Region`, offset `logprem`, exposure 0.37.
  40-value frame (`±1e12`, `lo−1e6`, `lo−1`, `lo±ulp`, every knot `±ulp`, `hi±ulp`, `hi+1e6`,
  `±inf`, null; regions R1/R2/R3/null/NEW): `RateModel.predict` vs `fit.predict`
  **1.44e-15**, no NaN on either side. Holdout 3,000 rows **1.78e-15**; the same rows with
  `Mileage` cast to Int64 **1.78e-15**. Table read as `log rel = log(relativity_at_from) +
  slope·(x − from)` vs `Σ βⱼ·clip(x − startⱼ, 0, widthⱼ)`: **1.11e-16**. Flat outside the
  clamp with everything else held fixed: `0.0e+00` at `lo−1e6`, `lo−1`, `−inf`, `hi+ulp`,
  `hi+1e6`, `+inf`.
- **`slope_j = beta_j`.** `max |table slope − fit.coef|` = **0.0** with
  `scale_predictors=True` *and* `False` (the two fits give different coefficients, as they
  should; in each the table carries that fit's own unstandardised numbers). Rate model vs GLM
  6.7e-16 in both.
- **Base.** `rate_tables` base row relativity 1.0000, one row; `x_base = 25,000` = that row's
  `from`; `base_rate(fit)` **0.149183999607483** vs `fit.predict(base risk)`
  **0.14918399960748305** and `RateModel.predict(base risk)` **0.149183999607483**.
- **Monotone.** `monotone_bounds(spec, {"Mileage": "increasing"})` sets `lower = 0` on
  exactly the seven `Mileage in [·, ·)` columns and `upper = +inf` everywhere; `decreasing`
  mirrors it; `Mileage is null`, `DrivAge`, `Region` and the 5 interaction cells untouched.
  On a planted strictly-increasing book (60k rows, alpha 1e-8): free slopes all positive
  (0.0006–0.0737); `decreasing` → **all ten slopes exactly 0.0** and a table of 1.0000;
  `increasing` → the free solution back (max relative change 2e-3). Project JSON round trip
  of `VariableDesign(kind="continuous", monotone="increasing")`: `to_dict()` identical;
  `validate()` `[]`; exported script contains `monotone={'Mileage': 'increasing'}`;
  `run.fit.monotone == {'Mileage': 'increasing'}` and every table slope ≥ 0.
- **Design page (AppTest).** Kind selector options are
  `['auto','step','linear','continuous','categorical']` with one help line per kind; selecting
  `continuous` on a variable that had `kind="linear", knots=[8000, 20000],
  monotone="increasing"` gives `kind="continuous"` with the **monotone kept** (the old build
  dropped it), the knot radio gone, the "one straight line" caption drawn and an
  "Apply continuous design" button; no exception, no error.
- **`continuous` end to end.** `build_design` → `LinearEncoder('Mileage', [], (0, 30000))`,
  `n_features = 2`; rate table exactly four rows (`< 0`, `[0, 30000)`, `≥ 30000`,
  `Other / Unknown`); spec JSON round trip identical; rate model vs GLM **7.8e-16**;
  `from_rate_tables(rate_model_tables(rm))` **0.0** with no warnings; Excel round trip
  **6.7e-16** with `x_base` preserved; node edit rule on the single band — editing `(None, lo)`
  or the band moves rows {0,1} and one slope, `(hi, None)` moves row 2 and one slope, the null
  row moves nothing, and `predict(edge − ulp) / predict(edge)` is 1 at both clamp points for
  all four; exported script **with** a run `rc=0`, rebuilt model **1.11e-15**; **without** a
  run (`linear=['Mileage'], knots={'Mileage': []}, monotone={...}`) `rc=0`, four-row table,
  **1.33e-15**.
- **Recovery-test honesty.** Monkeypatched (outside the repo) `LinearEncoder.transform` back
  to hinge columns and re-ran the six assertions of
  `test_flat_stretch_is_exactly_flat_and_slopes_are_recovered`: **"flat bands exactly 0",
  "segment slopes", "curve ≤ 0.25" and "flat rows relativity_to == relativity" all FAIL**;
  with `tables._bin_rows` also reverted to the cumulative sum, "table slope == beta" fails as
  well (5 of 6). The test cannot pass with the old basis. Six seeds of the planted book: flat
  bands exactly 0 every time, 9/15 bands non-zero every time, segment ratios 0.917–0.962,
  worst grid gap 0.140–0.202 (bound 0.25).
- **Migration.** Built `src/` at `f0508e8` in a scratch tree, fitted and persisted a linear
  project there, then loaded the artefacts under the worktree: `.easyglm` scorer max relative
  difference **0.0** on 3,000 holdout rows; project JSON loads, `validate() == []`; the
  pickled run is a **cache hit** with an identical `run_key` and is silently re-read (B2-1,
  numbers above). `LinearEncoder.to_dict` never wrote a `hinges` key (it was `knots` / `clamp`
  / `null_indicator` at `f0508e8` too), so there is no old spec-JSON field to migrate.
- **Actuary document.** `scripts/checks/b_linear.py` (no `--write`) on the cached full French
  motor set reproduces `docs/checks/b-linear.md` **byte for byte** apart from one trailing
  newline — including the 89.4727 at `BonusMalus` 230, the 2-of-9 and 9-of-20 flat-band
  counts, Gini 0.3072 → 0.3106 and deviance explained 4.79 % → 4.97 %. The 89× tail is put in
  front of the owner with the remedy and Q10, as asked. Q3 and Q9 in the questions file are
  marked built and describe what was actually built.
- **Penalty geometry** (S1): band widths, weighted sd, exposure share and fitted rise per band
  on the check's own `BonusMalus` fit — table in S1.
- **Degenerate designs.** Duplicate knots are deduplicated; near-duplicate knots give a
  1e-10-wide band (N2); a design with such a band still fits, tables to 3.3e-16 and reads back.
  A fully flattened term (a `decreasing` constraint on a rising curve) round-trips through
  Excel and `from_rate_tables` with `x_base` intact and edits correctly.
