# Review of piece B — piecewise-linear (L-dummy) terms (core, engine, workflow spec, export)

*Reviewer: independent. Branch `release-0.4`, commits `75d0289`…`f606661` (`git diff 220f38c..HEAD`).
Contract: `docs/RELEASE_0.4_PLAN.md` §B + R2, `docs/reviews/00-plan-review.md` B2 and the §5 lines tagged B,
`docs/checks/00-questions-for-the-actuary.md` Q1–Q3 (+ Q10). Date 2026-09-03.*

## 1. Verdict

**Changes requested — three blocking items, all small to fix.**

The core and engine do what R2 asks and I could not break the exactness: clip before the
hinges, a hinge at `lo`, exactly flat outside `[lo, hi]`, nulls on their own row, table
rows `(from, to, relativity_at_from, slope)` with the two flat end rows and the null row,
scoring by `searchsorted` + one `exp`, base 1.00 at `x_base`. On a 44-value adversarial
frame (values at `±1e12`, one ulp either side of `lo`, of every knot and of `hi`, null,
NaN, an unseen and a null region, with an offset and a `Mileage × Region` interaction
whose parent is the linear variable) `RateModel.predict` matches `fit.predict` to
2.0e-15; integer-dtype input 2.2e-15; `from_rate_tables(rate_tables(fit), base_rate(fit))`
equals `to_rate_model(fit)` exactly; Excel and JSON round-trip to 1e-15; monotone on a
linear term is refused at design and project level with a message naming the variable.
Suite 328 green, ruff and black clean, golden test and fixtures untouched, the planted
test is deterministic over three runs.

The blocking items are (1) the band-edit rule breaks the curve at the two clamp points
— the very property the feature exists for — and the reader does not notice; (2) the
Design page silently throws away a clamp the user typed; (3) typing 0 in the rate-table
editor crashes and locks the page. (2) and (3) are data-loss / crash findings in the
workbench part of this piece and fall under the plan's merge rule.

## 2. Blocking

### B1. Editing either flat end row makes the curve jump at `lo` / `hi`; the node at `hi` cannot be moved continuously at all

**What.** `RateModel._update_linear` (`src/easy_glm/engine/rate_model.py`) treats the
`(None, lo)` and `(hi, None)` rows as steps: the row's value changes and no slope is
re-derived. But those two rows *are* the curve's values at `lo` and at `hi` — the last
sloped band's slope is derived towards the `(hi, None)` value, and the first sloped band
mirrors its start into `(None, lo)`. So:

- Editing `(hi, None)` ×1.25 (my run, no interaction): the prediction one ulp below
  30,000 divided by the prediction at 30,000 is **0.80**. The last band still ends at
  0.772 while the row after it now says 0.965; Excel shows `relativity_to = 0.7723` on
  one line and `relativity = 0.9654` on the next.
- Editing `(None, lo)` ×1.25: ratio just-below/at 0 is **1.25**; the two rows that
  showed the same number (0.4627) now differ, although editing the *other* of the pair
  moves both (`test_first_band_edit_moves_the_lower_flat_row`).
- There is no row an actuary can edit to raise or lower the curve's value *at `hi`*
  without a jump: the last sloped band's edit moves its start node, the top row's edit is
  a step.

`from_rate_tables` checks continuity only between interior sloped bands, so both broken
tables read back without a word (verified: "discontinuous at lo" and "discontinuous at
hi" tables ACCEPTED), and the exported script replays the adjustment. The plan (§B:
"re-derives the slope (keeps continuity by default)"; §5-B: "editing one band-end
relativity changes exactly the two adjacent slopes") does not allow a jump inside a linear
table. The test `test_flat_and_null_rows_edit_as_steps` locks the wrong behaviour in and
never probes `x = hi` against `x = hi − ε`.

**Failure scenario.** The actuary caps the mileage curve by typing 1.30 into the
"≥ 30000" row. Every policy at 29,999 miles pays 0.77 × and every policy at 30,000 pays
1.30 × — a 69 % cliff the table, the Excel and the round-trip all present as a valid
piecewise-linear curve.

**Fix.** Make every row's value a *node* and re-derive the adjacent slopes on every edit:

- `(None, lo)` and the first sloped band share the node at `lo`: editing either sets both
  values and re-derives slope 1 (one slope changes).
- `(hi, None)` is the node at `hi`: editing it re-derives the last sloped band's slope
  (one slope changes).
- Interior bands as now (two slopes change). Null row as now (a step — it is not on the
  curve).
- In `_config_from_linear_table`, also check `rows[0].relativity == rows[1].relativity`
  and `rows[-2].relativity_to == rows[-1].relativity` (same tolerance) so a hand-edited
  table with a cliff is refused like an interior one.
- Rewrite `test_flat_and_null_rows_edit_as_steps` into "end rows edit as nodes": after
  each of the six row kinds is edited, assert `predict(hi − ε) / predict(hi)` and
  `predict(lo − ε) / predict(lo)` are 1 within 1e-12, and that exactly the expected
  slopes changed (0 for the null row, 1 for the two end nodes, 2 for interior bands).
- Actuary doc: replace "the two flat end rows and the null row edit as plain steps" with
  the node rule, and say explicitly that the "< lo" row and the first band are one
  number.

If the builder wants to keep step edits at the ends, that is a domain question for the
owner and must be put to him in `docs/checks/b-linear.md` with the cliff shown; it
cannot be the silent default.

### B2. The Design page drops a user-set clamp on the next render (data loss)

**What.** `src/easy_glm/app/pages_design.py`, the grid loop (`for _, r in
edited.iterrows()`), rebuilds every row's `VariableDesign(...)` without `clamp=`. The loop
runs on every rerun, `new != vd` is then true for any variable with a clamp, and the
project's entry is overwritten. Verified with an AppTest: project saved with
`VariableDesign(kind="linear", knots=[8000, 20000], clamp=[100, 25000])`; after one
render of the Design page `clamp` is `None` (knots survive). The "Apply knots / clamp"
button therefore works for exactly one rerun; the model is then fitted with the training
min/max, and the exported script writes that clamp as if the user had chosen it.
`Project.to_json`/`from_json` round-trip the clamp correctly, so the loss is only in the
page.

**Failure scenario.** The owner follows the check document's own advice ("set the clamp
for this factor to where the data runs out, e.g. 150"), clicks Apply, fits — and gets the
230 clamp and the 101× relativity again with no message.

**Fix.** Pass `clamp=vd.clamp` in that constructor (one line, next to `levels=vd.levels`),
and add an AppTest that loads a project with a clamp, renders `pages_design` twice and
asserts the clamp is still there. Check `max_levels`/`levels` are the only other
fields carried and that no other new `VariableDesign` field is missing from this loop.

### B3. Typing 0 in the rate-table editor for a linear band crashes the page and locks it

**What.** `pages_tables.py` allows `min_value=0.0` in the "working" column; for a linear
table `update_relativity` refuses 0 (`relativities must be > 0`), the `ValueError`
propagates out of `S.refresh_adjustments` → `rebuild_rate_model` → `apply_adjustments`
and the page raises. Verified with an AppTest: `ValueError: Linear variable 'Mileage':
relativities must be > 0, got 0.0` uncaught in `pages_tables.render`. The bad adjustment
was appended to `cfg.adjustments` and touched *before* the refresh, so it is in the
project and every rerun of the Rate tables page for that model raises again; the "Reset
this variable" button is rendered after the failing call and is unreachable. Step tables
accept 0, which is why this never showed before.

**Fix.** For linear tables use `min_value=1e-4` (or refuse `<= 0` in the loop with
`st.error`), and make `refresh_adjustments` catch `ValueError`, show it with `st.error`
and drop the offending adjustment (or not append it until the refresh succeeded). Add an
AppTest with an `Adjustment(..., 0.0)` on a linear band that asserts no exception and an
error message.

## 3. Should fix

### S1. `from_rate_tables` reads a linear table without its `slope` column as a step table

Dispatch is `"slope" in columns`. Drop the column (a downstream system, or someone tidying
the sheet) and the same rows are accepted as `"numeric"`: a staircase holding each band's
*start* value — for the fitted Mileage table that is 23 % too low at the top of the last
band — with no error. Verified ACCEPTED, also with `relativity_to` still present. Fix:
if `relativity_to` is present (or any `from`/`to` pair is open at both ends of the table
the way a linear table is) without `slope`, raise "linear table … needs a slope column";
say in the docstring that `slope` is what makes a table linear. In the same function,
refuse zero-width bands (`from == to`): verified accepted, and a later edit on the band
after it produces `slope = inf` with a `RuntimeWarning` and writes `Infinity` into the
JSON. Warn when the null row is missing, as the categorical reader does (nulls then raise
at scoring time, which the actuary will meet in production, not here).

### S2. The 1e-6 continuity check rejects any rounded table

Relativities rounded to 4 dp, or the slope alone rounded to 6 dp (Excel's usual display,
and any rate manual), fail with "not continuous at 5000.0: the band before ends at
0.554194 but the next band starts at 0.5541". The slope is redundant once the curve is
continuous; derive interior slopes from consecutive start values (and the last one from
the `(hi, None)` value) and treat a supplied `slope` as a cross-check with a loose
tolerance (1e-3 relative on the band-end value) that warns rather than raises. Then a
hand-typed table with four decimals reads back and is continuous by construction.

### S3. `inf` scores as NaN in the RateModel but as the clamped value in the GLM

`score_linear` computes `slope * (x − start)` with slope 0 on the end rows: `0 × inf =
NaN`, silenced by `errstate(invalid="ignore")`, so a premium comes out NaN while
`fit.predict` gives 0.0958 (verified). The contract says clip → searchsorted → exp; do the
clip in the scorer (`np.clip(values, lo, hi)` before the exp) and the two agree for every
float.

### S4. Rounding the clamp outward changes fitted numbers and extends the sloped bands beyond the data — say so

`round_range_outward` is a reasonable convenience, but it is not neutral: with training
mileage 17.65–29,857 the clamp becomes (0, 29,900); the last sloped band continues its
fitted slope from 29,857 to 29,900 (0.36 % lower relativity at 29,900 than the raw clamp
gives), the first band from 17.65 down to 0, and the training predictions themselves move
by up to 8.8e-6 (the null coefficient changes: 0.41898 → 0.41963). Integer variables are
not left alone either: 3–4,995 becomes (0, 5,000); 18–80 stays. None of this is in the
actuary document — its "clamped at the training range" and "the clamp points are shown
in the table" (Q1) let the owner believe the clamp *is* the data range. State the rule in
one sentence in the doc and in the CHANGELOG ("the training range rounded outward to a
round number; the end bands keep their slope up to that number"), and consider skipping
the rounding when the rounding step exceeds, say, 1 % of the range. With a fitted run the
exported script writes the clamp explicitly, so the design is data-independent there
(good); without a run it calls `from_data(linear=[...])` and drops a user clamp — see S7.

### S5. `x_base` is not carried by the table, is lost by `from_rate_tables`, can be `None`, and goes stale

`x_base` lives only on `VariableConfig`/JSON. `rate_tables` marks it with `is_base`,
`rate_model_tables` (Excel) has neither, and `from_rate_tables(rate_tables(fit), …)`
returns `x_base=None` (verified) so JSON written from a rebuilt model differs from the
original. When the null row is the most exposed (verified with 70 % nulls: modal bin 5 of
6) the base is the null row and `x_base` is `None` — the plan's "relativity 1.00 at a
stated `x_base`" then has no `x`. After editing the base band's value, `x_base` still
claims 1.00 there. Fix: default the base to the most exposed *non-null* band for linear
terms; write `x_base` into the Excel Summary sheet (or a `base` column); in
`from_rate_tables` set `x_base` from the unique row with relativity 1.0 when there is one;
clear or recompute it on an edit of the base band.

### S6. Actuary document: three omissions and one non-reproducible number

`docs/checks/b-linear.md` is honest where it matters (the 101× at BonusMalus 230 is put
in front of the owner with the remedy and Q10), reads without code, and the script
reproduces it — except the exactness line, which printed 1.1e-15 in my run against 1.4e-15
in the file (solver threading noise). Print it as "below 1e-14" so `--write` is
idempotent. Add: (a) that the "< lo" row and the first band are one number and move
together; (b) what editing the end rows does (after B1: moves the end node); (c) the
rounding rule of S4 (the doc says the Density clamp is "0 to 27000" — true here only
because the data happen to end on round numbers). The "Guarantees" bullet "the two flat
end rows and the null row edit as plain steps" must change with B1.

### S7. The no-run exported script drops a user clamp (and per-variable knots)

`to_script(project)` without a run emits `DesignSpec.from_data(..., linear=['Mileage'])`
with no `clamp=` and no `knots=`; a project whose Design page set `clamp=[50, 150]`
produces a script that fits with the training range. Knots were already dropped before
this piece, but for a linear term the clamp changes the curve where the owner most cares
(Q10). Emit `clamp={var: (lo, hi)}` and `knots={...}` from the `VariableDesign`s, or at
least a loud comment.

### S8. Planted-truth test: the ±4,000-mile window is chosen to fit the noise; assert the curve instead

`tests/test_recovery.py::TestPlantedLinear` is deterministic (three runs, 2 passed each,
0.8 s) and does test something real (the average slope of each true segment within 10 %
from `fit.predict`; the flat-beyond test would catch a missing clip on either side). But
the "bends are sparse" part is weak: at alpha 3e-5 eleven of fourteen knots carry a
non-zero change; the 20,000 bend is smeared over 18,000–24,000 with a sign flip
(+2.65e-5 at 22,000, −4.02e-5 at 24,000), and with a ±2,000 window (one knot spacing, as
the docstring says) that bend sums to 47 % of the truth and the test fails — hence
±4,000, which also leaves only four "far" knots for the sparsity inequality. A broken
table (slopes not cumulated, wrong start value) would be caught by the invariants, not by
this test, since the table check here compares the table with the coefficients it was
built from. Add the assertion that matters to an actuary and is robust to where the lasso
puts the bends: `max |log rel_fitted(x) − log rel_true(x)| ≤ 0.03` over a 100-point grid
of `[0, 30000]`, relative to `x = 0`; state the window choice in the docstring as an
observed property of the lasso, not as the tolerance the feature promises.

## 4. Nits

- N1. `rm.diff` reports two rows for one edit of the first band (the `< lo` mirror
  and the band); after B1 this is by design — say so in the CHANGELOG, or collapse the
  mirror into one `Change`.
- N2. `_update_linear` indexes `rows[idx ± 1]`, so it depends on table order; `from_dict`
  neither sorts nor validates linear tables (a JSON with the null row first scores
  correctly but the neighbours are wrong for an edit), and a `"linear"` JSON whose rows
  lack `slope` dies with `AttributeError: 'FromToRow' object has no attribute 'slope'`
  instead of a format error. Route `from_dict` through the same validation as
  `_config_from_linear_table`.
- N3. A null `slope` on an interior band is reported as "not continuous"; say "slope
  missing".
- N4. The standalone editor (`ui/app.py`, `ui/charts.py`) and the workbench chart draw a
  linear table as bars / points at band starts (the categorical branch); no crash
  (verified load, select, edit), cosmetic.
- N5. `AGENTS.md` tree line for `design.py` is garbled ("LinearEncoder (hinges,
  InteractionEncoder (A×B cells), … clamp)").
- N6. Design detail panel: knots typed outside the clamp vanish silently on Apply (the
  data-driven builder drops them); show a caption.
- N7. The workbench editor grid shows only the start value; slope and `relativity_to`
  exist only in Excel. Fine for this piece, but the D-workstream editor should show both
  ends.
- N8. CHANGELOG: "refuses discontinuous curves" is true only at interior knots until B1
  lands.

## 5. Missing tests

- Edit of the `(None, lo)` row; scored continuity at `lo` and `hi` (`x` vs `x − ε`) after
  every row kind is edited (B1).
- `from_rate_tables` on: missing `slope` column, zero-width band, 4-dp rounded table,
  discontinuity at `lo`/`hi`, missing null row (S1, S2).
- `±inf` input through `RateModel.predict` vs `fit.predict` (S3).
- Design page keeps `clamp` across two renders (B2); Rate-tables page with an adjustment
  of 0 on a linear band (B3).
- `x_base` when the null row is modal and after `from_rate_tables` (S5).
- Integer-dtype input for the linear variable in the exactness probe (passes today,
  2.2e-15 — cheap to lock in).
- A `round_range_outward` case where the clamp moves (e.g. 17.65–29,857) asserting the
  stated effect on the end bands (S4).

## 6. What I re-ran

- Full suite: `328 passed in 165 s`; `ruff check .` clean; `black --check .` 73 files
  unchanged. `git diff 220f38c..HEAD -- tests/test_golden.py tests/fixtures` empty.
  `tests/test_app.py` untouched by the piece and green in the suite.
- `tests/test_recovery.py -k PlantedLinear` three times: 2 passed each, 0.81–0.82 s,
  identical.
- `scripts/checks/b_linear.py` (no `--write`) on the cached full French motor set: output
  identical to `docs/checks/b-linear.md` except the exactness figure (1.1e-15 vs 1.4e-15).
- Adversarial exactness (model: linear Mileage with 6 knots, step DrivAge, categorical
  Region, `Mileage × Region` with 40 kept cells, offset `logprem`): 44-value probe
  (`−1e12`, `lo − 1e6`, `lo − 1`, `lo − ulp`, `lo`, `lo + ulp`, each knot ± ulp, `hi ± ulp`,
  `hi + 1`, `hi + 1e6`, `1e12`, null; regions R1/R2/R3/null/NEW): max relative difference
  `RateModel.predict` vs `fit.predict` **2.0e-15**; with exposure 0.37 exact; NaN and null
  score identically; Int64 Mileage on 3,000 holdout rows 2.2e-15; float holdout 2.4e-15;
  `±inf` → RateModel NaN vs GLM 0.0958 / 0.0743 (S3).
- `from_rate_tables(rate_tables(fit), base_rate(fit))` vs `to_rate_model(fit)` on 3,000
  holdout rows with an unseen level, offset applied by hand: **0.0**; `x_base` None vs
  25,000 (S5). Excel (`to_excel` → `read_excel` → `from_rate_tables`): 1.1e-15; the sheet
  shows `slope` and `relativity_to`; `from`/`to` come back as Int64 and are accepted. JSON
  round trip 0.0.
- Band edits ×1.25 on each row kind (no-interaction model, rows `(None,0) 0.4576 |
  (0,5000) | (5000,8000) | … | (25000,30000) 1.0 | (30000,None) 0.6556 | null 0.6952`):
  lower flat → rows changed [0], slopes changed [], scored ratio at 0 = **1.25**; first
  sloped → rows [0, 1], slopes [1], continuous; interior → rows [2, 3], slopes [2, 3],
  continuous; last sloped → rows [6, 7], slopes [6, 7], continuous; upper flat → rows [8],
  slopes [], scored ratio at 30,000 = **0.80**; null → rows [9], continuous. Snapshot →
  JSON → `switch_to(1)` / `switch_to(v2)` restores every (relativity, slope) exactly for
  all six; `diff` lists the edit (two rows for the first-band edit). Two edits on adjacent
  bands in either order give the same table (order independence True). Table edited on
  the upper flat row read back by `from_rate_tables` without error (B1).
- `from_rate_tables` adversarial: relativity 0 / null, non-zero end slope, gap between
  bands, two null rows, only end rows — all refused with clear messages; missing null row,
  missing `slope`, missing `slope`+`relativity_to`, reversed, shuffled, zero-width band,
  discontinuity at `lo`, discontinuity at `hi`, string `from` column — all ACCEPTED; zero
  width + edit → `slope = inf` with `RuntimeWarning`; relativities rounded to 4 dp or
  slope to 6 dp → refused as "not continuous".
- `round_range_outward`: (12, 49873) → (0, 49900); (17, 81) → (17, 81); (1, 4997) →
  (0, 5000); (0.5, 1.5) unchanged; (−3.2, 3.7) unchanged; (1e5, 1,987,654) → (1e5,
  1,990,000). Fit raw vs rounded clamp on shifted mileage (17.65–29,857 → 0–29,900):
  train predictions differ ≤ 8.8e-6; at 29,900 by 0.36 %; null coefficient 0.41898 vs
  0.41963.
- 70 %-null Mileage: modal bin is the null row, `x_base = None`, JSON omits it, exactness
  6.7e-16.
- AppTests (linear project, fitted): `pages_design`, `pages_model`, `pages_tables`,
  `pages_diagnostics`, `pages_export` render with no exception, also with an adjustment
  of 1.3 on the first band (lower flat row follows: both 1.3) and with an adjustment of
  2.0 on the `(hi, None)` row; adjustment 0.0 on a linear band → uncaught `ValueError`
  in `pages_tables` (B3); Design page render turns `clamp=[100, 25000]` into `None`
  (B2). Standalone editor `ui/app.py` with a linear `.easyglm` in the working directory:
  loads, lists the variable, edits a band through `update_relativity` without exception.

## 7. Re-check (commits `05643c2`…`e45f913`, `git diff f606661..e45f913`) — 2026-09-03

### Final verdict: **Approved.**

All three blocking items and the eight should-fix items are addressed in the way the
review asked, the two deliberate deviations are justified, and every probe I built for
the first round now passes.

### Blocking items

- **B1 — fixed.** Every row is a node (`_update_linear`): editing `(None, lo)` or the
  first band sets both and re-derives one slope; editing `(hi, None)` re-derives the last
  band's slope; interior edits re-derive two; the null row edits alone. Re-run of my
  six-row-kind probe (no-interaction model, ×1.25 on each row): scored ratio
  `predict(edge − ulp) / predict(edge)` at every band edge including 0 and 30,000 is 1 for
  **all six** edits (was 1.25 at `lo` and 0.80 at `hi`). Rows/slopes changed: lower flat
  [0, 1]/[1], first band [0, 1]/[1], interior [2, 3]/[2, 3], last band [6, 7]/[6, 7],
  upper flat [7, 8]/[7], null [9]/[]. Snapshot → JSON → `switch_to` restores exactly for
  all six; `diff` lists the edit (two rows for the shared `lo` node, now documented).
  `from_rate_tables` refuses a mismatch between the `< lo` row and the first band ("they
  are the same point of the curve") and derives every slope from the node values, so a
  table edited at `hi` reads back as the continuous curve it now is.
- **B2 — fixed.** `clamp=vd.clamp` carried through the Design grid loop. AppTest: project
  with `clamp=[100, 25000]` still has it after one and two renders of `pages_design`.
- **B3 — fixed.** Editor refuses `≤ 0` for linear tables before saving (`min_value=1e-4`,
  `st.error`), and a refused adjustment already in the project is dropped by
  `_drop_refused_adjustment` via the new `AdjustmentError`. AppTest with an
  `Adjustment(..., 0.0)` on a linear band: no exception, message "Adjustment not applied
  and removed from the project: … relativities must be > 0, got 0.0", other pages render.

### Should-fix items

- **S1** — `relativity_to` without `slope` refused with a clear message; zero-width band
  refused ("band 5000.0–5000.0 has zero width"); missing null row warns; a table with
  neither `slope` nor `relativity_to` is read as a step table on purpose (documented).
- **S2** — slopes derived from consecutive node values; relativities rounded to 3, 4 or
  6 dp and a slope rounded to 6 dp all read back (were refused). *Deviation accepted:* the
  cross-check tolerance on the supplied `slope` is 1 % of the band-end value, not 0.1 %.
  The arithmetic holds — a slope shown to 6 dp on a 5,000-wide band is off by up to
  0.5e-6 × 5,000 = 0.25 % at the band end, so 0.1 % would warn on every Excel-displayed
  table — and the check only warns; the slopes actually used come from the nodes, so the
  tolerance changes no number.
- **S3** — clip before the exp: `±inf` now scores 0.0958 / 0.0743 in both the RateModel
  and the GLM (was NaN vs finite); NaN still routes to the null row.
- **S4** — rounding rule stated in the actuary doc (with the 17.65–29,857 → 0–29,900 and
  18–80 examples and "the end bands keep their fitted slope up to that number") and in the
  CHANGELOG; the "< 1 % of the range per end" claim is correct by construction (step ≤
  range/100).
- **S5** — null row never the base of a linear term (`_modal_bins`): the 70 %-null case
  now gives `x_base = 0.0` and JSON carries it; `is_base` column in Excel and
  `x_base (var)` in the Summary sheet; `from_rate_tables(rate_tables(fit), …)` recovers
  `x_base` (was `None`).
- **S6** — doc updated (node rule, one-number lower row, rounding rule, ±infinity);
  exactness line no longer prints a run-dependent digit. `scripts/checks/b_linear.py`
  without `--write` now reproduces `docs/checks/b-linear.md` **byte for byte**.
- **S7** — the no-run exported script now emits `knots={...}` and `clamp={...}` from the
  `VariableDesign`s.
- **S8** — curve-level assertion added (`max |log rel_fitted − log rel_true|` on a
  100-point grid). *Deviation accepted:* bound 0.06 rather than 0.03. I reproduced the
  evidence recorded in the test: 0.0525 at alpha 3e-5 and **0.0581 at alpha 1e-6**
  (near-unpenalised), so the residual is sampling noise of the planted book, not
  shrinkage or a scoring error; 0.06 is set from that measurement and the docstring says
  so. The window comment is reworded as an observed property, as asked.

### Nits

N2 (JSON validated and ordered on load; a `"linear"` table whose rows lack `slope` now
fails with "every row needs a 'slope' … is this a step table saved as linear?"), N3
(null slope on a band is simply derived), N5 (AGENTS.md line), N6 (knots outside the
clamp reported on Apply), N8 (CHANGELOG wording) are done. N1 is documented rather than
collapsed (fine). N4 (bar chart for a linear table in the standalone editor) and N7
(workbench grid shows only the start value) remain cosmetic and are for the D
workstream.

### What I re-ran on `e45f913`

- Full suite **337 passed** (168 s); `ruff check .` clean; `black --check .` 73 files
  unchanged. `tests/test_recovery.py -k PlantedLinear` ×3: 2 passed each, 0.78–0.80 s.
- Check script vs `docs/checks/b-linear.md`: identical.
- 44-value adversarial probe with offset and `Mileage × Region`: 1.4e-15; Int64 input
  2.1e-15; holdout 2.1e-15; `from_rate_tables(rate_tables(fit), base_rate(fit))` vs
  `to_rate_model(fit)` 6.7e-16; Excel round trip 1.0e-15; JSON 0.0; `±inf` equal to the
  GLM.
- Adversarial `from_rate_tables` (16 tables): missing `slope` with `relativity_to`,
  zero-width band, relativity 0 / null, non-zero end slope, gap, two null rows, only end
  rows, cliff at `lo` — all refused with the messages quoted above; reversed, shuffled,
  string `from`, rounded to 3/4/6 dp, null `slope` on a band, cliff at `hi` (re-read as a
  slope, with a warning that the `slope` column disagrees) — accepted; missing null row
  accepted with a warning.
- Malformed JSON: null row first → reordered, edit of the first band now touches rows 0
  and 1 together; rows without `slope` → clear `ValueError`.
- AppTests: Design page keeps the clamp; Rate tables page with a 0.0 adjustment shows the
  error and drops it; Tables/Export/Diagnostics/Design/Model render with an adjustment on
  the `(hi, None)` row; standalone editor loads, selects and edits a linear table without
  exception.

### New tests read

`test_every_row_edits_as_a_node_and_the_curve_stays_continuous`,
`test_null_row_edit_does_not_touch_the_curve`,
`test_from_rate_tables_shapes_rounding_and_x_base`,
`test_inf_and_integer_input_score_like_the_glm`,
`test_json_orders_and_validates_linear_rows`,
`test_null_row_is_never_the_base_of_a_linear_term`,
`test_round_outward_clamp_extends_the_end_bands_at_their_slope`,
`test_apply_adjustments_names_the_refused_entry`,
`test_design_page_keeps_a_user_clamp_across_renders`,
`test_tables_page_survives_a_zero_adjustment_on_a_linear_band` — they cover every item
in §5 "Missing tests".
