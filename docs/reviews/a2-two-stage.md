# Review of piece A2 — two-stage interactions with frozen mains (Q5)

Reviewer: independent. Two rounds; **round 2 is the current verdict** and round 1
is kept below it for the record.

---

# Round 2 — re-check of `ee330a2`, `7335656`, `6660305` (2026-09-03)

## Verdict

**Not a blanket approval: one blocking item remains, and it is one line.**
Everything else is done. Both round-1 blockers are fixed and fixed properly, all
seven should-fixes are addressed, the seven missing tests are present and are the
right tests, and nits N2–N6 are closed. `515 passed` on Streamlit 1.57 and on
1.63; `ruff check` and `black --check` clean; the golden files are still
untouched; and `scripts/checks/a_interactions.py --write` now regenerates
`docs/checks/a-interactions.md` **byte for byte identical** to the committed
file, which it did not in round 1.

The one thing left is a crash the piece introduced, in the control the piece
added:

### R2-B1. The Design page dies on a cells alpha outside `[0, 10]`

**What.** `pages_design.py::_interactions` renders each existing interaction's
alpha as `st.number_input("Cells alpha (0 = as mains)", 0.0, 10.0,
float(it.alpha or 0.0), ...)`. `Project.validate` puts no upper bound on
`Interaction.alpha` (it only checks `> 0`), so a project can legitimately hold a
value the widget refuses:

* `alpha = 12.0` → `StreamlitValueAboveMaxError: The value 12.0 is greater than
  the max_value 10.0.` — and I confirmed `validate("freq")` returns `[]` and
  `run_model` fits it happily at `alpha_stage2 = 12.0`. So the project is valid,
  it fits, and the page that owns interactions is the one page that cannot show
  it.
* `alpha = -1.0` → `StreamlitValueBelowMinError`. `validate` *does* reject a
  negative alpha, but the Design page is where the user would go to correct it,
  and it crashes before they can. The project can only be repaired by hand-editing
  the JSON.

**Failure scenario.** A project file written by hand, by a script, or by a future
build (the field is public, round-trips through `to_dict`/`from_dict` and is part
of `model_hash`) carries a cells alpha outside the widget's range. The Design
page raises for that model until someone edits the JSON. It is narrow — the UI
itself cannot produce such a value — but it is a hard crash on a valid project,
which is the round-2 bar.

**Exact fix**, either one:

```python
# pages_design.py — let the widget show whatever the project holds
current = float(it.alpha or 0.0)
new_alpha = c4.number_input(
    "Cells alpha (0 = as mains)", 0.0, max(10.0, current), current, 0.0001, ...
)
```

or bound the field where it is validated, so the widget's range is the contract:

```python
# project.py::validate
if it.alpha is not None and not 0 < it.alpha <= 10:
    problems.append(f"{name}: {it.name} alpha must be between 0 and 10 ...")
```

The first is safer on its own (it cannot make an existing project invalid);
doing both is better still. Add the regression test alongside
`test_design_page_offers_the_cells_alpha`: render the page with `alpha=12.0` and
assert no exception.

If the coordinator judges a hand-edited project out of scope, this is a clean
"Known limitations" line instead — but by the letter of the round-2 rule it is a
crash, so I am naming it rather than deciding that for you.

## The 0.38 % vs 0.61 % point — the builder is right

I re-measured on **the check's own fit** (full French motor set from the cache,
677 991 rows, alpha 3e-4, `DrivAge × BonusMalus`), independently of their script:

| quantity | measured |
|---|---|
| intercept the identical stage 2 chooses when allowed one | **+0.003785** |
| as a relativity | **0.3792 %** |
| adjusted cells | 9 |
| per-cell `(no-intercept − with-intercept)` | 0.00378479 … 0.00378490 |
| max deviation from the common level | **3.41e-07** |
| support identical between the two fits | yes |

So 0.38 % is correct for the model the document ships, and my round-1 0.61 % was
the same quantity on my own 6 000-row fixture at alpha 1e-3 (there the shift was
tighter, 8e-16, which is why I called it exact). Both numbers are right for their
own data; theirs is the one that belongs in the actuary's document. Their caveat
— that it is not an exact constant because policies in unrated cells have no cell
to carry a level shift for them — is the correct explanation of the 3.4e-07 gap
and of why my fixture's was smaller. The document reports the level, the gap and
the percentage from the fit it ships, so it will stay honest if the data changes.

## Round-1 items, re-checked one by one

| item | status | evidence |
|---|---|---|
| **B1** export emits an unrunnable stage 2 | **fixed** | `two_stage = isinstance(run.fit, TwoStageFit)`; the lumped run now exports a plain `fit_glm` script, rc 0. Test `test_exported_script_runs_when_no_cell_was_rated` reproduces the run to 1e-10. |
| **B2** array offset lost by stage 2 | **fixed** | array route vs `offset_col` route: stage-2 coefficients agree to **7.8e-16** (round 1: 6.28 in log space), stage 1 to 1.1e-15. `P1` is also dropped for stage 2 now. |
| **S1** `EasyGLM.save` unloadable bundle | **fixed** | bundle v3 writes `glm_model_stage2.joblib`; `load` returns a `TwoStageFit` with 69 coefficients against a 69-column spec and predictions identical (`0.0`). |
| **S2** cells alpha invisible | **fixed** (see R2-B1) | number input per interaction + one for the interaction being added; `alpha=0.25` renders and round-trips; `alpha=0.5` reaches the fit and the exported script (`stage2_alpha=0.5` in the no-run script, both scripts rc 0, with-run script reproduces the run to 1.1e-15). |
| **S3** vacuous stage-1 CV bound | **fixed** | now `assert fit.alpha < 1e-4`; measured 3.56e-06, so two orders of room and it would fail on a regression. |
| **S4** CHANGELOG "penalty rule unchanged" | **fixed** | now states the unstandardised branch moved from `penalty_weight` to `penalty_weight × 0.5`, that no fitted model on any product path moves, and that a hand caller should halve their alpha to reproduce an earlier fit. |
| **S5** cost of freezing under-reported | **fixed** | the document now carries both paragraphs — holdout A/E 1.0191 → 1.0223 (joint 1.0194) with the base-rate override named as the remedy, and the level-in-cells paragraph above. The Model page's info box says the same in one clause. |
| **S6** document not reproducible / undrived claim | **fixed** | `--write` output is **identical** to the committed file. Both sentences are derived, and each has a ⚠ branch that says the document came from a buggy build if the promise ever breaks — better than what I asked for. |
| **S7** silence when no cell was rated | **fixed** | "**No second stage.**" info box naming the pair and its floor; asserted by `test_pages_with_an_interaction_that_kept_no_cell`. |
| **N2** offset dtype in the script | fixed | script writes `.cast(pl.Float64).to_numpy()`; offset+interaction round trip 1.3e-15. |
| **N3** `TwoStageFit.__eq__` | fixed | explicit `__eq__` over both stages; still unhashable, as `GLMFit` already was, so nothing regressed. |
| **N4** no-run script drops the cell floor | fixed, and better than asked | each interaction is now written as `spec.add_interaction(InteractionEncoder.from_data(..., min_cell_exposure=..., penalty_weight=...))`. Verified three variants all run: floor 2 % → 8 non-unit cells, floor 99 % → 0 non-unit cells and no crash, penalty weight 5 → 3 non-unit cells. |
| **N5** return annotation | fixed | docstring now tells the caller to narrow with `isinstance`. |
| **N6** `P1` forwarded to stage 2 | fixed | in `dropped`. |
| **N1** plan sentence about the pair search | not mine to fix; the coordinator owns it. Behaviour unchanged and still correct (the page excludes already-added pairs). |

All seven missing tests from round 1 are present:
`test_exported_script_runs_when_no_cell_was_rated`,
`test_an_offset_array_reaches_both_stages`,
`test_save_load_round_trip_with_an_interaction`,
`test_the_same_penalty_through_glum_not_only_through_arithmetic`,
`test_stage2_alpha_is_the_largest_any_interaction_asks_for`,
`test_pickles_and_comes_back_whole`,
`test_a_cell_also_carries_the_level_stage_two_cannot_put_anywhere_else`
— plus two I had not asked for
(`test_pages_with_an_interaction_that_kept_no_cell`,
`test_script_without_a_run_lets_the_data_decide_the_stages`).

## Round-2 nits (not blocking; "Known limitations" candidates)

* **R2-N1.** The cells-alpha box is `format="%.5f"`, so any alpha below 5e-6
  displays as `0.00000` — which in that box *means* "as mains". The stored value
  is preserved (I rendered `alpha=1e-06` and read `1e-06` back off the widget),
  so nothing is lost, but the display is misleading at the small end. `%.6g` or a
  caption showing the raw value would fix it.
* **R2-N2.** `test_a_cell_also_carries_the_level...` passes if the deviation is
  under `0.01·|level| + 1e-3`, i.e. effectively 1e-3 absolute. My measurement on
  the check data supports 1e-6. As written the test would not notice the
  relationship degrading by three orders of magnitude; tightening the absolute
  term to ~1e-5 would make it a real pin.
* **R2-N3.** `fit_two_stage(..., offset=<array>)` is now fitted correctly, but a
  `RateModel` compiled from such a fit still silently omits the offset, as it
  does for `fit_glm`. The docstring now says so explicitly and points at
  `offset_col`; this is the right shape for a "Known limitations" line rather
  than more code.
* **R2-N4.** `Interaction.alpha` still has no upper bound in `validate` (see
  R2-B1) and no lower bound other than `> 0`, so a value like 1e-12 is accepted
  and effectively means "no penalty on the cells". Harmless, but the Design page
  cannot express it either.

## What I re-ran in round 2, with numbers

Everything in §6 of round 1, on the new HEAD.

* **Suites.** `515 passed` on the project venv (Streamlit 1.57), 183 s.
  `515 passed` on the `st163` venv (Streamlit 1.63.0), 210 s. `ruff check .` — all
  checks passed. `black --check .` — 89 files unchanged.
  `git diff --stat 043b69c..HEAD -- tests/test_golden.py tests/data examples/` —
  empty.
* **Actuary document.** `scripts/checks/a_interactions.py --write` on the cached
  full French motor set → `diff` against the committed file: **IDENTICAL**.
  Working tree restored with `git checkout`; the only change in the worktree is
  this review file.
* **Frozen mains** (same fixture and settings as round 1):

  | table | two-stage vs mains-only | joint vs mains-only |
  |---|---|---|
  | DrivAge | 4.4e-16 | 1.4e-01 |
  | Density | 8.9e-16 | 1.6e-02 |
  | VehPower | 2.2e-16 | 9.1e-03 |
  | Region | 2.2e-16 | 2.4e-01 |
  | base rate | 2.2e-16 | 1.8e-01 |

  Stage-2 intercept exactly `0.0`; `linear_predictor − (η₁+η₂)` 4.4e-16;
  `RateModel.predict / fit.predict − 1` **6.7e-16** on 1000 rows with nulls in
  both parents and an unseen level. Piecewise-linear parent: mains ≤ 8.9e-16,
  RateModel 6.7e-16. Two interactions: mains ≤ 7.8e-16, RateModel 6.7e-16.
* **Penalty equivalence through glum.** Cell block fitted with an intercept,
  standardised vs unstandardised: coefficients agree to **5.8e-16**, intercepts
  0.00613174958112397**1** vs …**8**.
* **Thin cells and CV.** Planted book, fixed alpha 2e-4: all seven R5 cells
  exactly 1.000000. `cv=5, n_alphas=20`: stage-1 alpha **3.56e-06**, stage-2
  alpha **8.87e-04**, R5 cells all exactly 1.000000, both 20-point paths finite
  with no null CV deviance.
* **Consumers.** `run.predict` vs `fit.predict` 6.7e-16; RateModel 6.7e-16;
  `.easyglm` JSON round trip **0.0**; Excel written; snapshot metrics
  `{"alpha": 0.002, "alpha_stage2": 0.002, "stages": 2}` and `"stages": 1` for
  the mains-only twin; `rebuild_rate_model` with a cell adjustment keeps the same
  `TwoStageFit` object, reprices 141 holdout rows and keeps `stages: 2`;
  `alpha_path` gives `stage ∈ {1, 2}`; pickle round trip identical (`0.0`);
  `model_hash` still moves with `Interaction.alpha` and returns; `PERSIST_FORMAT`
  5.
* **Exported scripts, all executed in a subprocess.** With a run + cell
  adjustment: rc 0, reproduces the run to 1e-10. With an offset column: rc 0,
  1.3e-15. With `Interaction.alpha = 0.5`: `stage2_alpha=0.5` in the no-run
  script, both scripts rc 0, with-run script reproduces the run to 1.1e-15. Run
  with no rated cell: rc 0 (round 1: two consecutive `ValueError`s). No-run
  scripts for fixed alpha, CV and the lumped floor: all rc 0.
* **Add / remove the interaction.** Adding it moves the mains by ≤ 1.1e-15 and
  the base rate by 6.7e-16; removing it returns the run key to the mains-only
  key and the tables to 8.9e-16 of the original.
* **Pages (AppTest, 1.57).** Model / Tables / Export / Diagnostics render for a
  two-stage run and for the no-rated-cell run, the latter now showing
  "**No second stage.** No cell of **DrivAge×Region** reached its exposure floor
  (99.00 % …)". Design page renders the cells alpha at 0.25 and 1e-06, and
  **raises** at 12.0 and −1.0 (R2-B1).

---

# Round 1 — original review (2026-09-03, commits `0215c3c`…`7c599ab`)

Reviewer: independent. Branch `piece/a2`, 6 commits on `043b69c`
(`0215c3c` core, `a9f5114` workflow/export, `c402fc3` workbench, `7febd65` tests,
`13448ad`/`7c599ab` actuary doc). Contract: `docs/RELEASE_0.4_PLAN.md` §R3 and
§R10 (Q5 → A2), `docs/checks/00-questions-for-the-actuary.md` Q5, and the prior
review `docs/reviews/a-interactions.md`.

---

## 1. Verdict

**The central promise is kept, and kept exactly.** On every design I could build
— step parents, categorical parents, a piecewise-linear parent, a one-band
`continuous` parent, two interactions at once, with and without an offset — the
main-effect rate tables and the base rate produced *with* an interaction are the
ones produced *without* it, to between 0 and 1.7e-15 on a relativity. The joint
fit the same designs used to get moves them by up to 24 %. Stage 2 carries no
intercept, `η = η₁ + η₂` to 4.4e-16, and `RateModel.predict == fit.predict` to
8.9e-16 (budget 1e-10) in every case, including nulls in both parents, unseen
levels, offsets and linear parents. The stage-2 penalty rule is right: I fitted
the cells standardised and unstandardised on an intercept-carrying variant and
the coefficients agree to **8.0e-16**, so `penalty_weight · 0.5` really is the
same penalty as `penalty_weight · 0.5 / sd` under standardisation. 504 tests pass
on Streamlit 1.57 and on 1.63; `ruff check` and `black --check` are clean; the
golden files are untouched; `scripts/checks/a_interactions.py --write`
regenerates the committed actuary document byte for byte apart from one
non-deterministic 1e-15 rounding number (S6).

**Not ready to merge as it stands: two blocking items.** One is a product path
an actuary can reach from the workbench — if an interaction is added whose cells
are all below the exposure floor, the **exported script no longer runs**. The
other is a new public argument that silently produces a wrong model. Both fixes
are a few lines. Everything else below is should-fix or smaller.

---

## 2. Blocking

### B1. The exported script is broken when an interaction keeps no cells

**What.** `workflow/export.py::to_script` decides it is writing a two-stage
script from `run.spec.interactions` — the *encoders* — but `fit_two_stage`
decides from `spec.interactions_spec().n_features` — the *columns*. When every
cell of an interaction is below `min_cell_exposure`, the encoder exists with
zero cells, `run_model` correctly returns a plain `GLMFit`, `run.alpha_stage2`
is `None`, and `to_script` nonetheless emits a stage-2 block. With
`alpha2 = None` the emitted call is `fit_glm(..., cv=None, n_alphas=20, ...)`.

**Failure scenario.** The actuary adds `DrivAge × Region` on the Design page with
the cell floor set high (or the pair is genuinely thin), fits, sees an all-1.000
matrix, and downloads the script from the Export page to hand to IT. The script
dies on the first run:

```
ValueError: Pass alpha=<penalty strength> or cv=<n_folds>. (Without either,
glum would silently return the least-regularised end of its path.)
```

and, once the alpha is supplied by hand, dies again on the same line:

```
ValueError: Found array with 0 feature(s) (shape=(3951, 0)) while a minimum
of 1 is required by GeneralizedLinearRegressor.
```

The workbench itself is fine (all four pages render; `run_model`, `alpha_path`,
`rate_tables`, `to_rate_model` all behave). Only the exported artefact is
broken, and nothing in the suite executes an exported script for this case.

**Exact fix.** In `to_script`, take the fitted run as the authority on stages the
same way `fit_two_stage` does:

```python
from easy_glm.core.fit import TwoStageFit
...
two_stage = (
    isinstance(run.fit, TwoStageFit) if run is not None else bool(cfg.interactions)
)
```

That is enough for the run branch, which is the one the Export page uses. Add a
test that executes the script for a run whose interaction kept no cells (see
§5). For the no-run branch the same hazard exists in principle; the cleanest
guard there is to emit `fit = fit_two_stage(train, spec, ...)`, which already
returns a plain `GLMFit` when there are no cell columns. (It does not bite today
only because the no-run script drops `min_cell_exposure` altogether — see N4.)

### B2. `fit_two_stage(..., offset=<array>)` silently fits the wrong stage 2

**What.** `fit_two_stage` drops `offset` from the stage-2 keyword set
(`dropped = {"monotone", "offset_col", "offset", "alpha", "cv", "fit_intercept"}`)
and builds stage 2's offset as

```python
eta1 = fit1.linear_predictor(data)
if fit1.offset_col:
    eta1 = eta1 + data[fit1.offset_col]...
```

`GLMFit.linear_predictor` never reads a stored offset, and an array offset is not
stored anywhere, so when the caller passes `offset=<array>` instead of
`offset_col=` the user's offset is in stage 1's fit but **not** in stage 2's
offset, and not in `TwoStageFit.predict` either (`offset_col` is `None`).

**Failure scenario.** A rate-change model (Q6: offset = log of current premium)
built by a script rather than the workbench, with the offset supplied as an
array. Stage 1 is correct (its coefficients match the `offset_col` route to
1.4e-15), but stage 2 sees a residual that still contains the whole log-premium
and dumps it into the cells: on my fixture the stage-2 coefficients differ from
the `offset_col` route by up to **6.28 in log space** (a cell relativity of ~535
instead of ~1). Nothing raises; the rate tables look plausible until someone
reads the matrix.

**Exact fix.** One of the two, in `fit_two_stage`, before the stage-1 fit:

```python
if kwargs.get("offset") is not None:
    raise ValueError(
        "fit_two_stage takes the model's offset as offset_col=<column>: an "
        "offset array cannot be carried into the second stage."
    )
```

or, if the array form should be supported, keep it out of `dropped`, add it to
`eta1` (`eta1 = eta1 + np.asarray(kwargs["offset"], float)`) and store it so
`TwoStageFit.predict` re-applies it. The raise is the safer of the two and
matches how `run_model` and the exported script already work.

---

## 3. Should-fix

### S1. `EasyGLM.save` writes a bundle that cannot be loaded back

`EasyGLM.save` persists `self.glm.model`, which for a `TwoStageFit` is stage 1's
glum estimator, together with `self.spec`, which is the composed mains+cells
spec. `save` succeeds; `load` returns a plain `GLMFit` with 38 coefficients
against a 69-column spec, and the first `predict` raises
`ValueError: X has 69 features, but GeneralizedLinearRegressor is expecting 38`.
It is loud rather than silently wrong, and `EasyGLM.fit` cannot build
interactions today, so this needs hand-construction to reach — but `EasyGLM` is
public and the class now advertises `TwoStageFit` in the package docstring.
Either refuse in `save` (`raise TypeError` naming `fit_two_stage` /
`to_rate_model` as the supported route) or persist both estimators and rebuild
the pair in `load`. The `.easyglm` RateModel JSON route is unaffected and exact
(round-trip difference 0.0).

### S2. `Interaction.alpha` exists but is invisible in the workbench

The new field validates, round-trips through the project JSON, changes
`model_hash` (so a run is correctly invalidated) and drives stage 2 — I confirmed
all four. But `pages_design.py` neither offers it nor displays it: the
interaction row shows `min cell exposure` and `penalty weight` only. A project
file that sets `alpha: 0.5` therefore changes the fit with no sign of it on the
page that owns interactions. At minimum add a caption next to `penalty weight`
when `it.alpha is not None`; better, a number input beside the other two, since
the Model page already surfaces the resulting "alpha (cells)".

### S3. The stage-1 half of the CV recovery assertion was made vacuous

`test_recovery.py::test_recovery_at_cv_chosen_alpha` had
`assert 2e-4 < fit.alpha < 3e-3` and now has `assert fit.alpha != 2e-4 and
2e-4 < fit.alpha_stage2 < 3e-3`. The stage-2 bound is the meaningful one and is
correct (I measured 8.87e-4). The stage-1 clause is near-vacuous: any float but
one passes. The reason for dropping the old bound is real and is written in the
comment — with the cells removed, stage 1's own CV wants far less penalty; I
measured **3.56e-06** — but R7 asks for a bound, not for none. Replace with
something that would fail on a regression, e.g. `assert fit.alpha < 1e-4`.

### S4. The CHANGELOG says the cell penalty rule is "unchanged"; the
unstandardised branch changed by a factor of two

`penalty_weights` went from `p1[sl] = enc.penalty_weight` to
`enc.penalty_weight * 0.5` in the `scale_predictors=False` branch. For the
product path (standardised) nothing moves, and the change is a *correction* —
the two branches disagreed by 2× before, and the new value is what makes stage 2
match a joint fit. But anyone who called `fit_glm(spec_with_cells,
scale_predictors=False)` before now gets cells penalised half as hard, and the
CHANGELOG bullet reads "**The cell penalty rule is unchanged**". Per R7 this
needs one sentence saying what moved and why (the previous unstandardised value
was inconsistent with the standardised one, which is the rule R3 specified).

### S5. The actuary document under-reports what freezing the mains costs

Two things the document shows in a table but does not say in words:

1. **Overall calibration gets slightly worse, not better.** Holdout A/E is
   1.0191 without the interaction, **1.0223** with the two stages, 1.0194 with
   the joint fit. The prose paragraph after the table explains the Gini and
   deviance-explained give-up carefully and never mentions A/E. An actuary reads
   A/E first.
2. **A cell is a pure adjustment to stage 1, but not purely an interaction.**
   Because stage 2 has no intercept, any overall re-levelling stage 2 would like
   has nowhere to go but the cells. I fitted the identical stage 2 *with* an
   unpenalised intercept: it chose intercept 0.006132, and **every non-zero cell
   coefficient in the no-intercept fit is the with-intercept coefficient plus
   exactly that 0.006132** (max deviation 8e-16). So on that fixture ~0.61 % of
   each adjusted cell is level, not interaction. This is an unavoidable
   consequence of the design the actuary asked for, and the design is right —
   but "each is an adjustment to the mains (1.00 = none)" (Model page) and
   "every cell is a *pure adjustment*" (check doc) would be more honest with a
   clause such as "including any small overall re-levelling stage 2 wants, which
   it cannot put in the base rate".

### S6. The check document contains one number it cannot reproduce, and one
sentence it does not derive

Re-running `scripts/checks/a_interactions.py --write` on the cached full French
motor set reproduces the committed `docs/checks/a-interactions.md` **exactly**
except for one line: "largest change 1e-15" becomes "2e-15". The script prints
`f"{moved_two_stage:.0e}"`, i.e. an observed glum-noise value, so the committed
document is not reproducible run to run and a reviewer's `--write` always
produces a diff. Print a threshold instead (`"below 1e-13"`, matching the test
tolerance). Separately, "**Every change is 0.00%**" is a hard-coded string; if
the promise ever broke the document would still assert it. Derive it from
`moved_two_stage` (or have the script assert on it) the way `7c599ab` already
derived the main-effect count.

### S7. A model with an interaction but no rated cell says nothing on the Model
page

`two_stage = s["alpha_stage2"] is not None` is `False` in that case, so the "Fitted
in two stages" box, the "alpha (cells)" metric and `cells_kept` all disappear;
the Rate tables page still says "interactions: 1" and shows a matrix of 1.000s.
The actuary is given no reason. One `st.info` when `cfg.interactions and not
run.cells_kept` ("no cell of *A×B* reached the 2.0 % exposure floor, so there was
no second stage and every cell reads 1.00") would close it, and it pairs
naturally with the B1 fix since both hinge on the same condition.

---

## 4. Nits

* **N1.** R10 says "the pair search runs on the stage-1 residuals (already the
  case)". It does not: `pages_diagnostics.py` computes `pred = run.predict(frame)`,
  the *composed* prediction. The outcome is nevertheless right, because the page
  removes pairs that are already interactions of the model from the candidate
  list, so a fitted pair is never scored. Measured on the fixture holdout for
  `DrivAge × Region` after fitting it: signal −0.293 against stage-1 expected,
  −0.714 against the composed expected — i.e. the composed number is the one you
  want when hunting for the *next* interaction. Correct the plan sentence rather
  than the code.
* **N2.** The exported script writes `eta1 = stage1.linear_predictor(train) +
  train['logprem'].to_numpy()`; the library writes
  `data[offset_col].cast(pl.Float64).to_numpy()`. An `Int64` or `Float32` offset
  column would take a different path in the script than in the workbench. (The
  end-to-end round trip still matched to 1.4e-15 on a `Float64` column.)
* **N3.** `TwoStageFit` inherits `GLMFit`'s generated dataclass `__eq__`, which
  compares the declared fields only; `stage1` / `stage2` are ordinary attributes,
  so two two-stage fits sharing a stage-1 model compare equal regardless of their
  cells. Nothing uses it; `eq=False` or an explicit `__eq__` would be tidier.
* **N4.** The no-run exported script still drops `min_cell_exposure` and
  `interaction_penalty_weight` (pre-existing, `_spec_from_data_code` writes only
  `interactions=[('A', 'B')]`). It matters slightly more now that the presence of
  a stage 2 depends on the cell floor.
* **N5.** `fit_two_stage` is annotated `-> GLMFit`. Honest (the no-cells case
  returns a plain fit) but it means callers lose `alpha_stage2` to the type
  checker. A `GLMFit | TwoStageFit` alias or a one-line note would help.
* **N6.** `fit_two_stage` forwards user `glum_kwargs` such as an explicit `P1` to
  both stages, where the stage-2 length will not match. Low value to guard, but
  the docstring claims "``kwargs`` are :func:`fit_glm`'s", which is what invited
  B2 as well.

---

## 5. Missing tests

Each was run by hand for this review; all pass except where noted.

1. **The exported script for a run whose interaction kept no cells.** Would have
   caught B1 outright. The existing export test only asserts strings for that
   shape and executes the script only for the ordinary case.
2. **`fit_two_stage(..., offset=<array>)`.** Would have caught B2. Assert it
   raises (after the fix), or assert stage-2 coefficients match the `offset_col`
   route.
3. **`EasyGLM.save` / `load` of a `TwoStageFit`** (S1) — currently silently
   writes a bundle that raises on load.
4. **The 0.5 equivalence *through glum*, not only through arithmetic.**
   `test_the_cell_rule_is_the_same_penalty_in_stage_two` asserts
   `std * sd == raw`, which is a property of `penalty_weights`, not of glum.
   Add the end-to-end version: fit the cell block with an intercept twice, once
   `scale_predictors=True` and once `False`, and compare coefficients (I measured
   max |Δ| = **8.0e-16**). That is the assertion that would fail if glum ever
   stopped multiplying a standardised column's `P1` by its `sd`.
5. **`stage2_alpha` largest-wins with two interactions asking different alphas.**
   Only the single-interaction override is covered. (Verified by hand:
   `[0.1, 0.7, None] → 0.7`.)
6. **Pickle round-trip of a `TwoStageFit` run**, which is what the
   `.easyglm-runs` folder does. Verified by hand: type preserved,
   `alpha_stage2` preserved, predictions bit-identical — but nothing asserts it,
   and `TwoStageFit` is not a dataclass, so its pickling is not covered by the
   pattern the other fits rely on.
7. **A stage-2 cell coefficient carrying the level shift** (S5). A test pinning
   the "no-intercept fit = with-intercept fit + intercept on the non-zero cells"
   relationship would keep the documentation honest.

---

## 6. What I re-ran, with numbers

**Suites and tooling**
* Full suite, project venv (Streamlit 1.57): **504 passed**, 184 s.
* Full suite, `st163` venv (Streamlit 1.63.0): **504 passed**, 186 s.
* `ruff check .` — all checks passed. `black --check .` — 89 files unchanged.
  (`ruff format --check` flags 6 files, all pre-existing and none touched here;
  the project's formatter is black per `AGENTS.md`.)
* `git diff --stat 043b69c..HEAD -- tests/test_golden.py tests/data examples/` —
  empty. No golden number or tolerance in the golden path was touched.
* `scripts/checks/a_interactions.py --write` on the cached full French motor set:
  the regenerated document is identical to the committed one except
  "largest change 1e-15" → "2e-15" (S6). Every table — the DrivAge main table
  with its 21 `0.00%` rows, the adjustment matrix, the metrics table with
  Gini 0.3072 / 0.3083 / 0.3105, the A/E-by-pair table and the planted check —
  reproduces byte for byte. Working tree restored with `git checkout`.

**Frozen mains** (fixture from `tests/test_interactions.py`, alpha 0.001, Poisson,
`Exposure` weight, target divided by weight; mains DrivAge / Density / VehPower /
Region, interaction DrivAge×Region)

| table | two-stage vs mains-only | joint vs mains-only |
|---|---|---|
| DrivAge | 2.2e-16 | 1.4e-01 |
| Density | 6.7e-16 | 1.6e-02 |
| VehPower | 2.2e-16 | 9.1e-03 |
| Region | 2.2e-16 | 2.4e-01 |
| base rate | 7.8e-16 | 1.8e-01 |

Stage-2 intercept exactly `0.0`. `linear_predictor − (η₁+η₂)` max abs
**4.4e-16**. `RateModel.predict / fit.predict − 1` max **8.9e-16** (budget 1e-10),
on 1000 scoring rows containing nulls in both parents and an `UNSEEN` region.
Stage-1 coefficients against the standalone mains-only fit: max abs **1.0e-15**
(glum run-to-run noise — the 1e-13 test tolerance is justified, and "bit for bit"
in the docstrings is a hair strong; "to floating-point noise", as the tests say,
is right).

Other shapes, same measurements:
* **piecewise-linear (`linear`) parent** `Density × Region`: mains move ≤ 1.7e-15,
  RateModel exact to 7.8e-16.
* **two interactions** `DrivAge×Region` + `VehPower×Region`: mains move ≤ 7.8e-16,
  RateModel exact to 8.9e-16.
* **offset column** (`logprem`): stage 2's `offset_col` is `None` and the offset
  is inside η₁ exactly once; exported script round trip **1.4e-15**.

**Workflow (`run_model` on the `test_workflow` project, alpha 0.002)**
* `type(run.fit) == TwoStageFit`, `alpha = alpha_stage2 = 0.002`, 12 cells kept.
* `run.predict` vs `fit.predict` 7.8e-16; `RateModel` vs `fit` 7.8e-16;
  `.easyglm` JSON round trip **0.0**; Excel written (19 KB).
* Snapshot metrics carry `{"alpha": 0.002, "alpha_stage2": 0.002, "stages": 2}`;
  the mains-only twin carries `"stages": 1`.
* `rebuild_rate_model` with a **cell adjustment** of 1.25: no refit
  (`again.fit is run.fit`), `TwoStageFit` preserved, 141 holdout rows repriced,
  metrics entry still `stages: 2`.
* Exported script executed in a subprocess with an interaction **and** a cell
  adjustment: rc 0, rebuilt `.easyglm` reproduces the run to 1e-10 (this is the
  committed test; I re-ran it plus the offset variant above).
* `alpha_path` returns `stage ∈ {1, 2}` in both the fixed-alpha (2 rows) and CV
  cases (20 + 20 rows), CV deviance finite with no nulls in either stage.
* Pickle round-trip of the run: type and `alpha_stage2` preserved, predictions
  identical to 0.0.
* `model_hash` changes when `Interaction.alpha` is set and returns to its old
  value when unset; `PERSIST_FORMAT == 5`; `Interaction.alpha` survives
  `to_dict`/`from_dict`.

**Add / remove the interaction after the mains were fitted**
Fitted mains-only, added `DrivAge×Region`, refitted: every main table moves
≤ 1.6e-15 and the base rate 3.3e-16. Removed it again: the run key returns to the
mains-only key, the fit is a plain `GLMFit` with `alpha_stage2 is None`, and the
tables match the original mains-only fit to 1.7e-15. No spurious refit warning
in either direction.

**Stage-2 penalty**
Fitted the cell block twice with an intercept and `offset = η₁`, once
`scale_predictors=True` and once `False`: identical support and coefficients to
**8.0e-16**, intercepts 0.0061317495811239**12** vs …**586**. Against the actual
no-intercept stage 2, every non-zero coefficient differs by exactly the intercept
0.006132 and every zero stays zero (S5).

Planted-truth book (40 000 rows, `PLANTED_LOG_EFFECT = 0.9`, R5 ≈ 0.2 % of rows,
`min_cell_exposure = 0`):
* fixed alpha 2e-4 — all seven R5 cells exactly **1.000000**, largest |log rel|
  over all cells 0.375.
* `cv=5, n_alphas=20` — stage 1 alpha **3.56e-06**, stage 2 alpha **8.87e-4**
  (inside the 2e-4–3e-3 band the joint fit used to land in), all seven R5 cells
  exactly 1.000000, largest |log rel| 0.307. Stage 2's CV runs cleanly on the
  residual offset: 20 alphas from 3.8e-09 to 3.8e-03, no null or non-finite CV
  deviance, 8.4 non-zero coefficients at the selected point.

**Pages** (AppTest, Streamlit 1.57). Model / Tables / Export / Diagnostics all
render for a two-stage run (covered by the committed tests, which also assert the
"Fitted in two stages" box and the two alpha metrics) and for the no-kept-cells
run — in the latter the page shows a single "alpha" metric, no info box and no
cell count (S7). No Compare page or HTML report exists in this build (D3 / D4 are
later pieces), so there was nothing to check there; the Diagnostics page's
challenger comparison renders with a two-stage run.
