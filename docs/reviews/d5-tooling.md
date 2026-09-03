# D5 — relativity tooling: independent review

## Round 2 verdict

**APPROVED.** Round 2 reviewer, worktree `piece/d5` at `2950f70` (round 1's
fixes at `418b865`/`c7b5dcd`/`2cb35d2`, now merged with `release-0.4`/A2 two-stage
interactions). All four round-1 findings (B1, B2, S1, S2) are fixed and I
re-derived every number independently rather than trusting the builder's own
scripts or the document. No blocking defect remains. N1–N4 (nice-to-have) also
appear addressed in the diff, listed below as closed rather than as open
"known limitations".

Interpreters verified: `easy_glm.__file__` under the worktree on both the
main venv and the Streamlit-1.63 venv.

### B1 — "no change" is now the true change, and the document matches its own numbers

* The Tools panel's "expected claims (training)" tile scores the previewed
  tables via `_claims_change`/`workflow.expected_claims`
  (`src/easy_glm/app/pages_tables.py:456-475`) and reads "no change" only
  under 1e-9 (`_pct`, line 478-484). Independently recomputed off the raw
  `RateModel.predict` (not the app's own helper) on my own project via
  AppTest:

  ```
  tile ('expected claims (training)') for MA(3) on DrivAge: -1.125%
  workflow.expected_claims: now = 286.00297141566557  after MA(3) = 282.78449255511225  pct change = -1.1253%
  raw predict path:        now = 286.00297141566557  after MA(3) = 282.78449255511225  pct change = -1.1253%
  ```
  (`scratchpad/revd5r2/tile_check.py`) — the tile matches the fully
  independent computation to display precision.

* Re-ran the round-1 exactness/level script unmodified
  (`scratchpad/revd5/exact.py`) against the current tree: identical figures
  to round 1, including MA(3) on DrivAge at **−0.573347%** and the cap at
  **−4.860394%**:
  ```
  moving avg 3 DrivAge      1.221e-15   +0.000000%   -0.573347%   1.00574
  cap 3.0 BonusMalus        1.110e-15   -2.073695%   -4.860394%   1.05106
  ```
* `scripts/checks/d5_tooling.py` regenerated and byte-identical to
  `docs/checks/d5-tooling.md` (`diff ... && echo IDENTICAL` → IDENTICAL). The
  document now reads "Two different questions: the shape, and the money",
  states the true training-book changes (−0.573%, −4.860%) next to the
  unchanged mean-log-relativity, and adds a fifth row for
  "capped + rebalance base rate" showing expected claims back to +0.000% and
  holdout A/E back to 1.0184. The three sentences the round-1 review quoted as
  contradicted by the tables are gone.
* **Rebalance base rate** (`workflow.rebalance_override`,
  `src/easy_glm/workflow/run.py:493-511`) is exactly
  `base_rate × fitted_claims / current_claims`; independently verified on the
  French motor set with a real two-stage (interaction) fit to
  `2.220e-16` (budget 1e-10), one undo step (`_rebalance`, `pages_tables.py:676`
  calls `_apply(..., before)` once), relativities untouched.
* Q17 added, Q16 widened, both with the worked numbers
  (`docs/checks/00-questions-for-the-actuary.md:24-25`).

### B2 — undo now restores the base rate too, bit for bit

`EditStep` (`src/easy_glm/app/state.py:1099-1108`) is
`(adjustments, base_rate_override)`; `edit_state`/`_move` record and restore
both. Re-ran the round-1 AppTest script unmodified on **both** Streamlit
venvs:

```
main venv:  before restore: override=None base_rate=0.2645783804056859
            after  restore: override=0.5  base_rate=0.5
            after  undo   : override=None base_rate=0.2645783804056859
st163:      before restore: override=None base_rate=0.26457838040568576
            after  restore: override=0.5  base_rate=0.5
            after  undo   : override=None base_rate=0.26457838040568576
```

Base rate is back to the exact pre-restore float on both venvs. The official
suite's two-stage/interaction variant
(`test_the_page_tools_undo_and_rebalance_on_an_interaction_model`) additionally
asserts `np.testing.assert_array_equal` (not `approx`) after two undos on an
interaction model — passing on both venvs.

### S1 — stale snapshot restore now refuses cleanly; interaction removal strips snapshots too

* `_snapshots()` Restore now calls `missing_variables` first
  (`pages_tables.py:770-783`) and shows `st.error` naming the missing factor,
  changing and saving nothing. Re-ran the round-1 AppTest script: no
  exception, the error names `Region`, `adjustments now: []` (nothing was
  half-applied or autosaved).
* `apply_adjustments` raises `AdjustmentError` for an unknown variable, not a
  bare `KeyError` (`workflow/run.py:276-289`).
* Interaction removal now calls `cfg.drop_adjustments_for(it.name)`
  (`pages_design.py:622-632`), which strips the cell's adjustments from the
  working set **and every snapshot** (`workflow/project.py:298-313`). Wrote a
  fresh AppTest script (`scratchpad/revd5r2/interaction_strip.py`, not one of
  the builder's) that snapshots a cell adjustment, removes the interaction,
  and confirms: no exception, `interactions == []`,
  `snapshot adjustments == []` — the previously-crashing restore path is gone
  because the poisoned adjustment no longer exists to restore.

### S2 — delete is two clicks with a flash, verified via the project's own AppTest pattern

The button asks twice (`pages_tables.py:800-819`, same key, label flips on
`confirming`), the warning and success notices go through `ui.flash`. My own
minimal AppTest reproduction hit an unrelated AppTest harness quirk (a
selectbox default resetting across an internal `st.rerun()` when only one
snapshot exists — the same class of quirk round 1 already flagged in
`test_w2_pages`), so I fell back to the project's own two-snapshot AppTest
flow, which sidesteps it by re-asserting the selectbox value before each
click (`test_snapshot_create_restore_and_diff`,
`test_removing_an_interaction_cleans_its_snapshots_too`) — ran both in
isolation on **both** venvs, both green:
```
main venv: 2 passed, 59 deselected in 1.85s
st163:     2 passed, 59 deselected in 2.31s
```

### Merge checks (D5 × A2 two-stage interactions)

* `PERSIST_FORMAT = 6` in `state.py:111`, comment documents both reasons (D5's
  row-exposure field, A2's `TwoStageFit`).
* `TwoStageFit` (`core/fit.py:682`) now passes `row_exposure=dict(stage1.row_exposure)`
  through — confirmed independently on a real `DrivAge×BonusMalus` two-stage
  fit on the French motor set: `sum(row.exposure)` for DrivAge, BonusMalus and
  Density all equal the training exposure (250,951.35) to 1e-6.
* MA(3) applied to DrivAge (a parent of the interaction), independently
  scripted (`scratchpad/revd5r2/twostage_check.py`, not reused from round 1):
  interaction cells bit-identical before/after; `RateModel.predict` vs
  `fit.predict × adjustment` to **1.110e-15** (budget 1e-10); rebalance exact
  to **2.220e-16**. The official two-stage undo test additionally confirms
  bit-for-bit undo via `assert_array_equal`.

### Gates

```
631 passed, 1 skipped   (main venv, full tests/, -p no:randomly)
171 passed              (st163, test_d5_tooling.py test_w2_pages.py test_app.py test_w3_hardening.py)
black --check .         All done, 98 files unchanged
ruff check .            All checks passed
git diff release-0.4 -- tests/test_golden.py tests/fixtures   → empty
```

### Nice-to-haves (round 1) — appear closed, not carried forward as limitations

N1 (`groups(cfg)` empty-table refusal, `engine/tooling.py:127`), N2
(`app/grids.py:25` `TOL = TOOL_TOL`, single source of truth with
`engine/tooling.py:66`), N3 (`FITTED_OPTION`/`CURRENT_OPTION` sentinels
reserved, `pages_tables.py:41-42`) and N4 (the hand-written fit-specific
sentences are gone from the regenerated check document) all check out in the
diff. Not independently re-derived line-by-line since none is blocking either
way, but nothing contradicts the commit message's claims.

---


*Reviewer: not the builder. Worktree `piece/d5` at `f0bab84`, diff
`git diff release-0.4...HEAD`. Interpreter
`/Users/serban/Documents/Projects/easy_glm/.venv/bin/python` with
`PYTHONPATH=<worktree>/src` (`easy_glm.__file__` verified under the worktree);
Streamlit 1.63 venv used for the page tests as well as the 1.57 default.*

## Verdict

**Changes requested — two blocking, two should-fix.**

The engineering underneath is good and I could not break the arithmetic. Every
exactness property the piece claims holds, measured rather than asserted: after
any tool `RateModel.predict` equals `fit.predict` times the adjustments to
**1.2e-15** on the full French motor set (budget 1e-10); undo and snapshot
restore reproduce the previous predictions **bit for bit** (max relative error
`0.0`); JSON, Excel and the project file all carry the exposure column and the
adjustments and round-trip to `0.0` / `6.7e-16`; a piecewise-linear curve stays
continuous at the knots to **1.1e-16** with the slopes re-derived; the null /
Other row is untouched by every tool; the categorical refusal is in the engine,
not only on the page; a pre-D5 run pickle is ignored and refitted rather than
misread, even when forced onto the new key. `scripts/checks/d5_tooling.py`
reproduces `docs/checks/d5-tooling.md` byte-identically. 577 passed / 1 skipped
on 1.57; 148 passed on 1.63; golden untouched and green; `ruff check` clean.

What blocks is not the maths but **what the product tells the actuary about the
money**. The panel says a smoothing makes "no change" to the overall level while
it quietly takes 0.57 % off the book, and the check document contains three
sentences that its own tables contradict. Separately, Undo after a snapshot
restore silently keeps the snapshot's base rate, which is a 1.9× premium error
in the scenario the document itself recommends.

---

## Blocking

### B1. "Overall level: no change" is not true of the premium, and the check document says the opposite of its own numbers

The engine follows plan §R6 exactly: both smoothers preserve the
exposure-weighted mean of the **log** relativities to 1e-12. That is not the
same thing as leaving the premium alone, because the book premium is the
exposure-weighted mean of the **relativities** (times everything else), not of
their logs. Preserving a geometric mean while calling it "the premium level"
understates the real off-balance by a factor of 2–3 here, and reports zero where
the true figure is half a percent.

Measured on the full French motor set (`~/.cache/easy_glm/*.parquet`, 474,788
training rows, the check script's own fit), by comparing total expected claims
`Σ predict·Exposure` before and after each tool
(`scratchpad/revd5/exact.py`):

```
tool                           max|pred/expected-1|  level_shift  train exp claims chg  train A/E
moving avg 3 DrivAge                      1.221e-15   +0.000000%            -0.573347%    1.00574
moving avg 7 DrivAge                      1.221e-15   +0.000000%            -0.804615%    1.00808
isotonic inc BonusMalus                   9.992e-16   +0.000000%            -0.202079%    1.00200
cap 3.0 BonusMalus                        9.992e-16   -2.073695%            -4.860394%    1.05106
round 0.05 DrivAge                        1.110e-15   -0.290726%            -0.222332%    1.00220
```

(The first column is the exactness check and is fine — see the verdict.)

So the page's metric row shows

* **overall level → "no change"** for an operation that takes **0.573 %** off
  every premium in aggregate (0.805 % at window 7);
* **overall level → −2.07 %** for a cap whose real cost is **−4.86 %**.

The check document repeats this three times, and each time its own table
disproves it:

* "*The holdout A/E moves only where the level moved*" — the table two lines
  above shows fitted A/E **1.0191** and smoothed (moving average, level
  "unchanged") **1.0247**, i.e. the model under-charges by 0.55 % after a
  smoothing that reported no change.
* "*capping BonusMalus at 3.00 takes 2.07 % off that factor, so the model
  under-charges by about that much*" — the holdout table shows
  1.0191 → **1.0704**, an under-charge of **5.03 %**, not 2.07 %.
* "*with the level unchanged, so the premium of an average policy is what it
  was*" and the guarantee "**A smoothing never moves the premium level**" — true
  of the geometric-mean policy, false of the book.

An actuary reading this document cannot read the code; "smoothing does not move
the premium level" is the sentence they will rely on, and after smoothing five
factors they would be several percent light with every panel saying "no change".

Evidence for the reproduction of the document itself:

```
$ PYTHONPATH=$PWD/src .venv/bin/python scripts/checks/d5_tooling.py > /tmp/out.md
$ diff /tmp/out.md docs/checks/d5-tooling.md && echo IDENTICAL
IDENTICAL
```

**What I would accept.** Any of these closes it; the first is best.

1. Report the **true** number on the panel. The page already holds the prepared
   frame and already predicts for A/E, so `Σ predict·w` before and after the
   previewed values is a couple of lines and is exact. Keep the log-mean figure
   if you like, but label it "mean log relativity" only, and give
   "overall level" the real change in expected claims.
2. If the number must stay a pure function of the table, weight the
   *relativities* rather than the logs — the exposure-weighted relativity mean
   already tracks the truth far better than the log mean
   (cap 3.0: −5.22 % proxy vs −4.86 % true vs −2.07 % as shipped) — and say in
   the caption that it is an approximation because the band weights are
   exposure, not premium.
3. At minimum: delete the three sentences above, stop printing "no change" for
   the smoothers, and state plainly that a smoothing preserves the mean *log*
   relativity, that this is not the same as preserving the premium, and that the
   book falls by roughly half a percent in the worked example.

**And add a question.** Q16 asks the actuary about re-centring after a *cap*.
The bigger question is not asked at all: **which invariant should a smoother
preserve — the mean log relativity (as built, per §R6) or the book premium
(re-balanced so total expected claims are unchanged)?** That is a real
actuarial choice with a standard answer in rate reviews (off-balance
correction), and it belongs next to Q15/Q16 rather than being settled by a plan
revision the actuary has not seen in these terms.

### B2. Undo after restoring a snapshot does not restore the base rate

`_snapshots()` restore sets **both** `cfg.adjustments` and
`cfg.base_rate_override` from the snapshot
(`src/easy_glm/app/pages_tables.py:648-651`), but the undo step it records is
`before = list(cfg.adjustments)` only, and `state._move()` puts back only
`cfg.adjustments` (`src/easy_glm/app/state.py:1096-1108`). Undo therefore
reverts the tables and leaves the snapshot's base rate in force.

`scratchpad/revd5/undo_bro.py` (Streamlit 1.63, AppTest on the real page):

```
before restore: override = None  base_rate = 0.2645783804056858
after  restore: override = 0.5   base_rate = 0.5
Undo enabled: True
after  undo   : override = 0.5   base_rate = 0.5
```

After Undo the model charges **1.89× the premium it charged before the restore**,
with no message, while the check document states "*undo puts the tables back
exactly as they were (the same numbers, not an approximation)*". This is
reachable in exactly the workflow D5 recommends: the document tells the user to
re-set the level with the base-rate override after a cap, so an override and a
snapshot will routinely coexist.

Fix: make an undo step the pair `(adjustments, base_rate_override)` — record
both in `_apply`/`record_undo` and restore both in `_move`. A test that restores
a snapshot carrying an override, undoes, and asserts `rate_model.base_rate` is
back would have caught it; the existing
`test_apply_a_tool_then_undo_and_redo_it` only exercises adjustments.

---

## Should-fix

### S1. Restoring a snapshot whose variable no longer exists crashes the page, after autosaving the bad list

`apply_adjustments` raises a bare `KeyError` when an adjustment names a variable
the rate model does not have (`src/easy_glm/workflow/run.py:267-271`), and
`refresh_adjustments` catches only `AdjustmentError`
(`src/easy_glm/app/state.py:1034-1041`). `_apply` has already called
`S.touch()`, so the project file is saved with the poisoned list before the
traceback.

`scratchpad/revd5/page_stale.py` (snapshot taken with a `Region` adjustment,
`Region` then removed from the model and refitted, Restore clicked):

```
snapshots: ['with Region'] predictors: ['DrivAge', 'Density']
restore button present: True
EXCEPTION after Restore: ['"Adjustment refers to \'Region\', which is not a variable of the model (known: [\'DrivAge\', \'Density\'])"']
errors on page: []
adjustments now: [('Region', 1.5)]
```

I checked whether this is D5's doing: it is **not new** — the same traceback
happens on `release-0.4` if you simply drop a predictor that carries an
adjustment (`scratchpad/revd5/page_stale2.py`, run against both trees, identical
message). But D5 turns it from "an odd sequence on the Model page" into a
one-click button whose caption promises that snapshots "survive a reload, a
refit and tomorrow", and it half-fixed the same class already:
`Project.rename_column` was extended to rewrite snapshot adjustments
(`project.py:468`), while `pages_design.py:596`, which strips `cfg.adjustments`
when an interaction is removed, still ignores `cfg.snapshots[*].adjustments`, so
restoring such a snapshot is the same crash.

The good half works: a *band* that no longer exists raises `AdjustmentError`,
which is dropped and reported, not applied silently —
`AdjustmentError: No row found with from=33.0, to=41.0 in variable 'DrivAge'`
(`scratchpad/revd5/stale.py`). Only the missing-variable case escapes.

Fix: treat the missing-variable case the same way (catch it in
`refresh_adjustments` and report-and-drop, or raise `AdjustmentError` from
`apply_adjustments`), and strip snapshot adjustments wherever `cfg.adjustments`
are stripped.

### S2. Deleting a snapshot is one click, unconfirmed, and outside the undo stack

`Delete` sits in the same three-column row as `Restore`, both driven by the same
selectbox, and a mis-click destroys the only durable record of a table version —
snapshots are deliberately *not* covered by undo (the stack holds adjustments,
not the snapshot list), so there is no way back. The rest of the page is careful
about this (every destructive edit is one undo step); this one is not. A
confirmation, or putting the snapshot list into the undo step, would do.

---

## Nice-to-have

* **N1.** A table with only the null / Other row gives `nan` for both log means
  and a `RuntimeWarning: invalid value encountered in scalar divide`
  (`tooling.py:186`) from `cap_floor` / `round_relativities`; the page would
  print "nan" in the level check. Not reachable from a fit, but
  `groups(cfg) == []` deserves the same refusal `_smoothable` already gives.
* **N2.** `grids.TOL = 1e-9` against the tools' `1e-12` change threshold: a
  result whose changes all fall between the two enables "Apply to the table" and
  then writes nothing, with no message. Use one constant.
* **N3.** A snapshot named `(the tables now)` or `(fitted — no adjustments)` is
  shadowed in the compare selectors — `_snapshot_version` matches the sentinels
  before the snapshot list — so it silently compares the wrong thing. Reserve
  the two strings when a snapshot is created.
* **N4.** Four sentences of the generated document are hand-written facts about
  this particular fit ("*sits 33 % below it*", "*dips in the middle (the 60–72
  bands)*"); everything around them is computed, so they will drift silently if
  the fixture, alpha or `n_bins` change. Derive them or drop them.

---

## What I verified (and would not re-litigate)

* **Exactness.** `max |RateModel.predict / (fit.predict × adjustments) − 1|` ≤
  **1.3e-15** for all six tool applications on the French motor set (table in
  B1). Baseline fit-vs-rate-model agreement 1.0e-15.
* **Undo and snapshot restore are bit-exact.** `scratchpad/revd5/roundtrip.py`:
  restoring "smoothed" after a further cap, and restoring "fitted", both give
  `max rel err 0.0` against the predictions recorded at the time; the AppTest in
  the suite asserts list equality of relativities, not `approx`.
* **Round-trips.** JSON `0.0`; Excel `6.7e-16` and `rate_model_diff` empty at
  tol 1e-9; exposure identical through JSON, `create_snapshot`/`switch_to`, and
  Excel (`from_rate_tables` reads the column by name, so column order is not
  load-bearing); reloaded project file reproduces predictions to `0.0` and keeps
  both snapshots with their adjustment counts and overrides.
* **Cap / floor and round act on relativities, not log relativities**
  (`np.clip(v, …)`, `np.round(v, …)`), which is what an actuary means by "cap at
  3.00" and what the document says. Both idempotent; neither re-centred, which
  is right and is stated.
* **Linear tables.** Smoothing moves nodes and re-derives slopes: continuity
  max gap at the knots **1.1e-16**, the two open end bands keep slope 0, null
  row untouched (`scratchpad/revd5/linear.py`).
* **The null / Other row** is in no group for every table type and is unchanged
  by every tool, including when its relativity is far off the curve.
* **The categorical refusal is in the engine.** `smooth_moving_average` /
  `smooth_isotonic` raise `ToolingError` regardless of the page; `cap_floor` and
  `round` are (correctly) allowed.
* **Break-it.** window 25 on 5 bands → all bands equal, level preserved; window
  1, 0, 2, 4 → refused with a clear message; one band → "nothing to smooth";
  cap below floor → refused; round to 0 dp works and is refused when it would
  zero a band; a negative or zero relativity is refused by all three tools;
  NaN or net-negative exposure falls back to uniform weights and says so; an
  interaction has no tools; a 2-row linear table is refused, a 3-row one works.
* **Workbench.** Apply / Undo / Redo / Snapshot / Restore / Delete / diff all
  work on 1.57 and 1.63 (148 tests green on 1.63). Every new widget key goes
  through `S.widget_key` (grep for a raw `key=` in `pages_tables.py` returns
  nothing), and switching project clears the undo stack and the adjustments:
  token changes, `undo_stacks` → `{'freq': {'past': [], 'future': []}}`, Undo
  disabled. Success notices go through `ui.flash` so they survive the rerun.
* **Runs folder / PERSIST_FORMAT 5.** A run written by `release-0.4`
  (format 4) is **left on disk and ignored** — `get_run` returns None, the app
  refits, the old files are untouched. Forced onto the D5 key it is detected and
  removed as a cache miss, not misread
  (`scratchpad/revd5/loadold.py`, both cases).
* **Tests.** 43 real assertions in `tests/test_d5_tooling.py`, not smoke: hand
  computable tables, longhand recomputation of the weighted log mean,
  idempotence by re-application, monotonicity by `np.diff`, a hand-written
  `_lookup` that reads the exported frame the way a human would and compares it
  to the scorer. The `test_w2_pages` edit (a fresh selectbox node per run) is a
  genuine AppTest mechanic, justified in a comment, and does not weaken the
  assertions. The two changed column assertions are the new `exposure` column
  and are tightened, not loosened — they now also assert the column sums to the
  training exposure. `tests/test_golden.py` untouched and green (7 passed).
* **Q15/Q16** are well posed and honestly defaulted as far as they go; see B1
  for the question they are missing.

---

## Re-check

On the next round I will re-run exactly this and nothing else:

1. `PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q -p no:randomly` from the
   worktree root — expect ≥ 577 passed, and new tests for B1/B2.
2. `PYTHONPATH=$PWD/src <st163>/bin/python -m pytest -q -p no:randomly tests/test_d5_tooling.py tests/test_w2_pages.py tests/test_app.py tests/test_w3_hardening.py`
   — expect all green.
3. `scripts/checks/d5_tooling.py` piped to a file and `diff`ed against
   `docs/checks/d5-tooling.md` — must stay byte-identical, and the three
   sentences named in B1 must be gone or corrected.
4. **B1**: `scratchpad/revd5/exact.py` — the exactness column must stay ≤ 1e-12,
   and whatever the panel now calls "overall level" must agree with the
   `train exp claims chg` column (or be labelled so that it cannot be read as
   the premium change). I will also re-read the Tools caption and the check
   document's level section end to end as an actuary who cannot read code.
5. **B2**: `scratchpad/revd5/undo_bro.py` — after Undo, `base_rate` must be back
   to `0.2645783804056858`.
6. **S1**: `scratchpad/revd5/page_stale.py` — Restore must produce a message and
   a usable page, never a traceback, and the project must not be left holding an
   adjustment that cannot be applied; plus the interaction-removal variant.
7. **S2**: the Delete path on the page.
8. Spot re-run of the round-trip and linear scripts
   (`scratchpad/revd5/roundtrip.py`, `linear.py`) to confirm the B1/B2 fixes did
   not disturb exactness, continuity or the restore paths.
