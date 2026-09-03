# Review of piece A — two-way interactions (core, engine, workflow spec, export)

*Reviewer: independent. Branch `release-0.4`, commits `051db8d`…`f67ce57` (`git diff 7bea74e..HEAD`).
Contract: `docs/RELEASE_0.4_PLAN.md` §A + R3, `docs/reviews/00-plan-review.md` B3, S7, §5 (A lines).
Date 2026-09-02.*

## 1. Verdict

**Approve with one blocking fix (documentation only) and a short should-fix list.**

The code does what R3 asks: cells are indexed by the parents' rate-table rows, thin cells
are lumped to 1.0 with their exposure recorded, the RateModel reproduces the GLM to
machine precision on a frame with nulls in both parents, a NaN, an unseen level and an
unseen integer code at once, and the P1 decision is mathematically sound and empirically
necessary (without it, thin no-signal cells drift to 0.70–3.47 at CV-chosen alpha; with it
they sit at exactly 1.000). Engine, workflow spec, export and Excel all round-trip. The
suite (283), ruff and black are green; the golden test and fixtures are untouched; the
0.3 fixture still loads and scores identically.

The one blocking item is the actuary document, which names the wrong second parent
(VehPower) in two places while the matrix is DrivAge×**BonusMalus**, bands the A/E-by-pair
table differently from the matrix, and does not say why alpha 3e-4 was used. Since that
document is the owner's sign-off artefact and cannot be cross-checked against code by the
owner, it must be right before A is closed. The fix is a few lines in the check script.

## 2. Blocking

### B1. Actuary document `docs/checks/a-interactions.md` is internally inconsistent

**What.** `scripts/checks/a_interactions.py` hard-codes "VehPower" in the prose while
`PAIR = ("DrivAge", "BonusMalus")`:

- "a policy's relativity is the DrivAge factor × the **VehPower** factor × one cell of the
  adjustment matrix" (script line 240) — the matrix is DrivAge×BonusMalus.
- Section title "A/E by DrivAge × **VehPower** cell on the holdout" (line 288) — the table
  underneath is DrivAge × BonusMalus.
- The module docstring says the same.

Two further problems in the same document:

- The A/E-by-pair table bands BonusMalus by holdout quantiles (`knots_b` not passed), so its
  "worst cell" labels (`[72, 85)`, `[53, 60)`) are cells that do not exist in the matrix
  (`[72.0, 76.0)`, `[76.0, 85.0)`, …). The actuary cannot look the worst cell up in the
  matrix. DrivAge labels also differ in format (`< 25` vs `< 25.0`; see S3).
- Alpha 3e-4 is stated in the header but not justified. The builder's reason (at the plan's
  default 1e-3 the lasso kept no cells, so the check would have shown nothing) is exactly
  the sort of thing the owner must be told plainly: the alpha was chosen so that the feature
  is visible, it was not chosen by CV, and at a CV-chosen alpha the interaction would be
  smaller. (My CV experiment below: at CV alpha the planted log effect is recovered at
  0.59–0.75 of 0.90, versus 0.889 at the hand-picked 2e-4 the document reports.)

**Failure scenario.** The owner reads "DrivAge × VehPower", signs off on a
DrivAge×BonusMalus matrix, and later cannot reconcile the A/E "worst cell" with any matrix
cell; or takes 0.889/0.900 as the recovery to expect from the workbench (which fits by CV).

**Exact fix.** In `scripts/checks/a_interactions.py`: replace the two literal "VehPower"
strings (and the docstring) with `PAIR[1]`; pass
`knots_b=inter.spec[PAIR[1]].knots` to both `ae_by_pair` calls; add one sentence under
"Defaults in force": *"alpha 0.0003 was fixed by hand for this check — at the plan's default
0.001 the penalty kept no cells; the workbench chooses alpha by cross-validation, and at a
CV-chosen alpha the same planted effect comes back at roughly 65–85% of its true size (the
rest shows up in the A/E-by-pair table, which is why that table is shown)."*; re-run with
`--write` and commit the regenerated document.

## 3. Should-fix

### S1. Vacuous assertion in the export round-trip test
`tests/test_workflow.py:499–501`: `assert (... .cell_matrix.max() >= 1.25 or True)` always
passes. The subprocess comparison a few lines later does check predictions, but nothing
checks that the 1.25 cell adjustment was applied in `run`. Replace with
`assert next(r for r in run.rate_model.variables["DrivAge×Region"].table if r.key == kept.key).relativity == 1.25`.

### S2. Matrix sheet name is unrecognisable with long parent names
With parents `DriverAgeAtInceptionYears` and `GeographicalRegionCode` the workbook gets
`DriverAgeAtInceptionYears×Geogr` and `DriverAgeAtInceptionYears×G (2)` — the "(matrix)"
suffix is truncated away and the collision suffix takes its place. The Index sheet does map
it, but the actuary should not need it. Fix in `write_rate_tables_xlsx`: build the matrix
name from the *truncated* key, e.g.
`sheet_name(f"{str(key)[: _MAX_SHEET_LEN - len(' (matrix)')]} (matrix)", used)`, and add a
test with long names.

### S3. `ae_by_pair` / `band_expr` labels do not match rate-table row labels
Even with the model's knots passed, `band_expr` formats `< 25`, `[25, 30)`, `null`, while the
table rows read `< 25.0`, `[25.0, 30.0)`, `Other / Unknown`. My check with identical knots:
label sets differ on every row. The workbench heatmap (A-workbench piece) will have to line
up with the matrix; unify now (use `row_label` / `format_knot` for both, and one name for
the null row) so the diagnostic and the table are joinable by label.

### S4. The planted-truth test only runs at a hand-picked alpha
`test_recovery.py` fixes alpha 2e-4. The plan's §5 line ("thin non-signal cells must stay
within [0.98, 1.02]") is about the product as used, i.e. CV-chosen alpha. My run (cv=5,
25 alphas, seeds 11/12/13) passes — thin cells exactly 1.000, planted cell 1.44–2.13 — so
add a CV variant (or a second fixed alpha near the CV choice, ~1e-3) so a regression in the
P1 rule cannot hide behind the low alpha.

### S5. Dense cell block and the per-cell loop (hand-over to G, not a bug)
`InteractionEncoder.transform_frame` allocates a dense float64 block of `n_rows × n_cells`
and loops once per cell. Measured: 1.02 M rows × 66 cells = 0.41 s and 0.54 GB; at 5 M rows
× 210 cells (the French-motor matrix) that is 8.4 GB for the cells alone. Piece G must give
this block a categorical/sparse representation (tabmat `CategoricalMatrix` over the flat
cell index is the natural fit); the loop can also become a single lookup
(`lookup[flat] → column`). Record in G's brief.

### S6. Second implementation of the row rule in the engine
R3 asks for one shared helper. Core has one (`Encoder.row_index`, used by the cells,
`_modal_bins`, and — through `rows()` — the table builder). The engine necessarily has its
own (`engine/_scoring.row_index` on `VariableConfig`, since a `.easyglm` file carries no
encoder). Two implementations, guarded by `test_encoder_and_engine_agree` and by the
exactness tests. Acceptable; say so in the module docstring of `_scoring.py` so nobody
"simplifies" one without the other, and keep the agreement test.

### S7. Small robustness items
- `DesignSpec.from_data(interactions=[("A","B")])` with a parent not in `predictors` raises
  a bare `KeyError: 'A'`; give the message the encoder gives elsewhere.
- `apply_adjustments`: an `Adjustment(cell=True)` naming a main effect silently goes down
  the main-effect path (and a non-cell adjustment on an interaction errors with "pass
  from_b"). Validate `cell` against the variable's type and say which is wrong.
- The separator `×` is spelled in three places (`core/design.INTERACTION_SEP`,
  `engine/rate_model.INTERACTION_SEP`, `workflow.Interaction.name`). `from_rate_tables`
  splits the table name on it and `_interaction_table` splits labels on `" | "`; a main
  variable containing `×` or a level containing `" | "` breaks both. Cheap guards.
- When the caller passes their own `P1` in `glum_kwargs`, the cell rule is skipped without
  notice; document it in `fit_glm`'s docstring (one line).

## 4. Nits

- `docs/checks/a-interactions.md`: the DrivAge "Other / Unknown" row moves by exactly the
  same −21.1% as "< 25.0" (nulls sit in the lowest bin and the null column was zeroed). One
  clause would spare the actuary a question. The French data has no null ages, so the row
  is empty anyway.
- `Feature.level` for a cell holds `"< 25.0 | R2"`; `coef_table` shows it, fine, but
  `Feature.cell` is not in `coef_table` — add the two indices as columns when convenient.
- `InteractionEncoder.rows()` returns nested tuples; `row_label` on one would print garbage.
  It is only used for `n_rows`; either document that or return the flat label.
- `pages_tables` caption "the interaction editor arrives with the workbench update" is fine
  for now; make sure the A-workbench piece removes it.
- CHANGELOG entry is accurate; the AGENTS.md tree line is updated.

## 5. Missing tests (each one line; all were run by hand and pass unless noted)

- Excel long sheets read back with `pl.read_excel` → `RateModel.from_rate_tables` reproduces
  an *adjusted* model with two interactions (the S4 product path) — passes, add it.
- AppTest on `pages_design`, `pages_model`, `pages_diagnostics`, `pages_tables`,
  `pages_export` with an interaction in the config — all render, add to `test_app.py`.
- `model_hash` changes with `min_cell_exposure`, `penalty_weight`, removal of an
  interaction, and does **not** change with a cell adjustment — passes, add it.
- Long parent names → matrix sheet still carries "(matrix)" — **fails today** (S2).
- `ae_by_pair` sums of `actual`/`expected` equal the overall totals (the current test checks
  exposure only) — passes.
- Zero kept cells (`min_cell_exposure` above every share): fit, exactness, all-1.0 table,
  script emits `cells=[]` — passes.
- Numeric parent with `null_indicator=False` and enough nulls to keep the (null, ·) cells —
  the cell carries the null effect and scoring is exact — passes.
- Planted truth at CV-chosen alpha (S4) — passes for three seeds.
- An interaction parent edited after the fit: predictions of that row's policies scale by
  the factor and every other policy is unchanged (cells untouched) — passes.

## 6. Contract compliance (R3, B3, S7, §5-A) — findings by reading and by test

| Item | Finding |
|---|---|
| Cell index = parent rate-table row index | Yes. `InteractionEncoder.cell_index` calls the parents' `row_index`; `StepEncoder.row_index` = `searchsorted(knots, side="right")` with NaN → last row; `CategoricalEncoder.row_index` = level position, unseen/null → Other. `_modal_bins` uses the same. Engine mirrors it (S6). |
| Nulls in both parents, unseen level, same frame | 3 000-row frame with null and NaN ages, null and `NEVER_SEEN` regions, unseen VehPower code 99, two interactions incl. a categorical parent: `RateModel.predict` vs `fit.predict` max relative difference **8.9e-16**. Nine rows null in both parents land in (null row, Other row) and score exactly. |
| Kept-cell rule | `keep = share ≥ min_cell_exposure & exposure > 0`, share of the interaction's whole training exposure; 32 of 35 kept in my case; not-kept cells 1.0 with exposure recorded; exposure sums to training exposure. |
| Spec round trip without data | `DesignSpec.from_dict(json)` rebuilds identical design matrices and feature names; parents resolved by name; unknown parent → clear error. |
| "All cells 1.0 == GLM with cell slice zeroed" | Tested by the builder (`test_all_cells_one_equals_zeroed_slice`) and holds because cells are never re-based and `base_rate` sums mains only. |
| `ModelConfig.interactions`, `validate`, `model_hash` | Present; validate flags a=b, (b,a) duplicates, non-predictor parents, bad threshold/weight; `model_hash` sensitive to all interaction fields (via `asdict`). |
| Cell adjustments `(from_a,to_a,from_b,to_b)` | `Adjustment(cell=True)`, `Change(is_cell)`, `update_relativity(from_b=,to_b=)`, `_mask_for_row` (parents separately) — all present and round-tripped. |
| Format version | Stays 2; the unknown-type tests were re-pointed from `"interaction"` to `"spline"`, which is the right change now that the type is known; 0.3 fixture scores identically. |
| Strict dispatch | `"interaction"` handled by an explicit branch before `_SCORERS`; `KNOWN_TYPES` gates `_from_dict`; `"spline"` still refused. Interaction whose parent is missing → *"Interaction 'DrivAge×Region' needs its parent 'Density' in the model"*. |

## 7. Mathematics of the P1 decision

glum 3.4.1 (`_utils.standardize`) rescales `P1` by `1/sd` **only** when
`estimate_as_if_scaled_model` is False; with `scale_predictors=True` the penalty is
`α Σ P1_j · sd_j · |β_j|` on the raw coefficient. I verified `fit.model.P1` equals the
vector we pass. So:

- A 0/1 column of share *p* has `sd = √(p(1−p))`. With the default `P1 = 1` a thin cell's raw
  effect is almost free (penalty ∝ √p); that is S7's complaint and my experiment confirms it.
- With the builder's `P1 = penalty_weight · 0.5 / sd`, every cell pays `α · 0.5 · |β|`
  irrespective of *p*. Relative to a main-effect step column of share *q* the cell is
  penalised `0.5/√(q(1−q)) ≥ 1` times as hard: equal for a 50/50 step, 2.3× for a 5% step.
  In noise units (s.e. ∝ 1/√p) thin cells are shrunk ∝ 1/√p harder — the credibility-like
  direction an actuary wants. Sound, and it is exactly "`scale_predictors=False` semantics
  for the cell block" as R3 asks, with a sensible normalisation.
- The knob is exposed per interaction (`penalty_weight`, in the spec, the project file and
  the script). Do not expose the 0.5 separately.

Experiment (40 000 policies, planted log effect 0.9 in <25 × R2, region R5 at 0.2% = 82
rows spread over 6 cells; `cv=5`, 25 alphas, 3 seeds):

| rule | CV alpha | recovered log dd (truth 0.9) | planted cell | thin cells | non-zero cells |
|---|---|---|---|---|---|
| builder `0.5/sd`, scaled | 7.2e-4 – 1.2e-3 | 0.59 / 0.75 / 0.74 | 1.44 / 2.13 / 2.09 | **1.000 – 1.000** (all seeds) | 4–9 of 30 |
| plain scaled lasso `P1=1` | 1.2e-3 – 2.3e-3 | 0.82 / 0.71 / 0.72 | 1.49 / 2.07 / 1.52 | **0.70 – 3.47** | 8–19 of 30 |
| unscaled, `P1=1` | 3.3e-4 – 3.5e-4 | 0.76 / 0.72 / 0.74 | 1.51 / 2.12 / 1.81 | 1.000 – 1.000 | 5–11 of 30 |

Conclusion: keep the builder's rule. The one thing to say to the actuary (B1) is that at a
CV-chosen alpha an interaction is recovered at roughly 65–85% of its size — ordinary lasso
shrinkage, shared by all three rules — and that the remainder is visible in A/E by pair.

## 8. What I re-ran, with numbers

- `pytest tests -q`: **283 passed**, 17 warnings, 161.6 s. `ruff check src tests scripts`:
  clean. `black --check`: 65 files unchanged.
- `git diff 7bea74e..HEAD -- tests/test_golden.py tests/fixtures`: empty.
- `scripts/checks/a_interactions.py` (full French set, cached): output identical to the
  committed `docs/checks/a-interactions.md` apart from a trailing blank line. Exactness on
  the holdout below 1e-12; 28 of 210 cells rated; 7 adjusted.
- Adversarial exactness (see §6): 8.9e-16. Encoder vs engine `row_index`: identical on
  DrivAge, Density, Region, VehPower over that frame.
- `P1` alignment with two interactions and a categorical parent: every non-cell column 1.0,
  every cell column `0.5/sd` to 1e-9; `fit.model.P1` equals it.
- Engine: `from_rate_tables(rate_tables(fit), base_rate(fit))` and
  `from_rate_tables(rate_model_tables(rm), rm.base_rate)` equal the RateModel to 1e-12;
  omitting the 20 rows at 1.0 changes nothing; a cell pointing at `from_a = 999.0` → "does not
  match a row"; a duplicated cell → "lists cell … twice"; JSON with a parent deleted → clear
  error; an interaction stored under an arbitrary name still loads (the `parents` field
  drives it); `clone` shares nothing; `diff` reports one `is_cell` change; `switch_to` after
  JSON restores the matrix bit-for-bit and predictions with `atol=0`; editing a parent row
  scales only that row's policies; offset × interaction exact to 1e-10; 0.3 fixture scores
  identically to `v0_3_0_predictions.parquet`.
- Excel: both matrix sheets read back with `pl.read_excel` equal `cell_matrix` and the
  recorded exposure to 1e-9; row/column headers equal the parents' table labels; lumped
  cells 1.0 with share below threshold. Long names → S2 failure.
- Workflow: `model_hash` sensitivities as in §5; `validate` on a deliberately bad config
  returns all three complaints; `run_model` with two interactions and 54 null ages in the
  holdout exact to 1e-10; `rebuild_rate_model` keeps a 1.77 cell adjustment and a 0.5 main
  adjustment; project → JSON → `rebuild_rate_model` re-applies identically (`atol=0`); a
  cell adjustment on a non-existent cell → "No cell (None, 25.0, 'ZZ', 'ZZ') in interaction".
  The builder's subprocess round-trip test (`test_exported_script_reproduces_the_model`)
  ran green inside the suite.
- `ae_by_pair`: actual/expected/exposure sums equal the overall totals to 1e-9 for a
  numeric×categorical and a categorical×categorical pair; null ages get a `null` row;
  labels differ from table rows (S3).
- AppTest with `Interaction("DrivAge","Region")` in the config: `pages_design`,
  `pages_model`, `pages_diagnostics`, `pages_tables`, `pages_export` render with no
  exception; the export page emits `InteractionEncoder(`; the tables page shows the
  "1 interaction table(s) are included in the Excel download" caption.
- Edge cases: zero kept cells (exact, all 1.0, script emits `cells=[]`);
  `null_indicator=False` parent with kept null-row cells (exact); `transform_frame` on
  1.02 M rows × 66 cells: 0.41 s, 0.54 GB.

## 9. Planted-truth test — is it meaningful?

Yes. With the interaction coefficients zeroed the fitted model is additive in log space,
so the double difference is 0, not 0.9 ± 0.1 — the test fails. If cells were indexed against
the wrong rows (the B3 failure mode), the lasso would still find *a* column for the young×R2
policies, the double difference could still pass, but the test also asserts the cell
labelled `(to_a = 25, R2)` is > 1.2 and is the largest kept adjustment — a mislabelled cell
fails that. Thin-cell bounds [0.98, 1.02] would pass trivially if all cells were dropped,
but then the double difference fails. The three assertions together are not passable by a
broken implementation. The only weakness is the fixed low alpha (S4).

## 10. Actuary document (item 8)

Clear and, apart from B1, honest: the mains-move table (Q5) shows DrivAge shifting up to
+17% at ≥72 and −21% below 25 when the interaction enters, which is the right thing to put
in front of the owner; the threshold (Q4) is stated with the count of rated cells; no code.
Alpha 3e-4 is *not* explained (B1). Suggest also one sentence saying that "1.000 (14,759)"
and "1.000 (20)" mean different things and that the second is "too thin to rate", which the
introduction says but the matrix does not mark visually.

## 11. Re-check of the follow-up commit `5988c13` (2026-09-02)

Scope: `git diff f67ce57..5988c13` only (`005eb30` is the coordinator's plan note and was
ignored). Read every source hunk; re-ran the check script, my adversarial scripts, the
suite, ruff and black.

### Verdict

**Approved. Piece A is closed from the reviewer's side.** B1 is fixed and the regenerated
document is correct; S1–S4, S6 and S7 are addressed as asked; S5 is recorded in the
CHANGELOG as a known limitation handed to G, which is what I recommended.

### What the diff does (by reading)

| Item | Fix in `5988c13` | Judgement |
|---|---|---|
| B1 | `scripts/checks/a_interactions.py` uses `PAIR[0]`/`PAIR[1]` everywhere, passes `knots_b` (the model's BonusMalus knots) to both `ae_by_pair` calls, adds the plain alpha sentence ("fixed by hand … at 0.001 the penalty kept no cells … at a CV-chosen alpha … roughly 65–85% of its true size"), the `1.000 (14,759)` vs `1.000 (20)` explanation, and the one-line note on the `Other / Unknown` row. | Correct. The A/E "worst cell" is now a matrix cell (`[28.0, 30.0) | [60.0, 64.0)`), so the actuary can look it up. |
| S1 | Vacuous `or True` replaced. | Done (suite covers it). |
| S2 | New `suffixed_sheet_name(key, " (matrix)", used)` shortens and numbers the *stem* and keeps the suffix. | Correct; verified below. |
| S3 | One `NULL_LABEL = "Other / Unknown"` and one `INTERACTION_SEP` in `engine/models.py`, imported by core and workflow; `band_expr`/`band_labels` build labels with `row_label`, so diagnostics use `< 25.0`, `[25.0, 30.0)`, `Other / Unknown`; `ae_by_pair` gains `levels_a/levels_b` so categorical parents are banded onto the table's rows too. | Correct; verified below. NaN is now routed to the null label as well. |
| S4 | `test_recovery.py` gains a CV-alpha variant. | Present in the suite. |
| S6 | `_scoring.py` module docstring names the deliberate duplicate and the test that binds them. | As asked. |
| S7 | Clear error for a non-predictor parent in `DesignSpec.from_data`; `apply_adjustments` rejects an unknown variable and a `cell` flag that disagrees with the variable's type; parent names containing `×` are refused at encoder construction; `from_rate_tables` resolves `A×B` by trying every split point and refusing ambiguity; cell labels come from `InteractionEncoder.cell_labels()` instead of splitting on `" | "`; `fit_glm` docstring says a user `P1` replaces the cell rule. | All as asked. |

One remark, not a finding: `apply_adjustments` now raises for an adjustment on a variable
that is not in the model, where before it fell through to `update_relativity`'s `KeyError`.
Same exception type, better message; nothing upstream catches it specifically.

### What I re-ran, with numbers

- `pytest tests -q`: **299 passed**, 17 warnings, 166.9 s. `ruff check`: clean.
  `black --check`: 65 files unchanged.
- `scripts/checks/a_interactions.py` (no `--write`, full French set): output identical to the
  committed `docs/checks/a-interactions.md` apart from a trailing blank line.
  `grep -c VehPower docs/checks/a-interactions.md` → **0**. The A/E-by-pair section is now
  banded on the matrix's own rows and columns.
- Adversarial exactness frame (null and NaN ages, null and unseen regions, unseen integer
  VehPower code, two interactions incl. a categorical parent): `RateModel.predict` vs
  `fit.predict` max relative difference **6.7e-16**; encoder and engine `row_index`
  identical on all four variables; null/null rows land in (null row, Other row) and score
  exactly; every engine, JSON, clone/diff/switch_to, offset and 0.3-fixture check from §8
  still passes.
- Long parent names: sheets are now `DriverAgeAtInceptionYears×Geogr` (long table) and
  **`DriverAgeAtInceptionYe (matrix)`** — 31 characters, unique, recognisable. Passes.
- Workflow script: `model_hash` sensitivities, `validate`, run exactness with 54 null ages in
  the holdout, `rebuild_rate_model` keeping a 1.77 cell and a 0.5 main adjustment, project
  JSON re-apply (`atol=0`), bad cell → "No cell …", `ae_by_pair` totals equal to overall,
  Excel long sheets → `from_rate_tables` reproducing the adjusted two-interaction model —
  all pass. `ae_by_pair` labels now equal the rate-table row labels exactly (`< 20.0`,
  `[20.0, 23.0)`, …, `Other / Unknown`); the one line my script reports as a failure is my
  own stale expectation that null ages be labelled `"null"` — they are now labelled
  `Other / Unknown`, which is the S3 fix working as intended.

### Left open (non-blocking, already recorded elsewhere)

- S5 (dense cell block, per-cell loop) — CHANGELOG "Known limitations" and the plan note for
  G; G's brief should carry the numbers from §8 (1.02 M × 66 cells = 0.54 GB; French-motor
  scale ≈ 8.4 GB).
- The nits in §4 that were not part of the fix list (`Feature.cell` in `coef_table`,
  removing the `pages_tables` caption when the interaction editor lands).
