# Breaker session 3 — Compare, HTML report, Tools/Undo/Redo/Snapshots/Rebalance, rate-change, penalties, Tweedie/binomial, cells alpha, CLI, compact-matrix (release-0.4 @ `piece/breaker3`)

Third breaker session, after D3/D4 (Compare, report), A2 (two-stage
interactions), D5 (tooling/undo/snapshots/rebalance) and E+F (rate-change,
penalties, Tweedie/binomial, CLI) all merged to `release-0.4`. This session
targets the surfaces those pieces added, plus the compact-matrix path (piece
G) run through the workbench rather than just the core.

Method: read the code for every surface named in the brief, then drove it two
ways — direct calls into `easy_glm.workflow`/`easy_glm.engine` for the fast,
wide sweep (dozens of edge-case models: zero holdout rows, binomial, an
interaction with no kept cell, mismatched champion/challenger, a 250k-row
book through fit → diagnostics → tools → report → exported script), and
Streamlit's `AppTest` for anything that only shows up on a page (widget
state, deleted-model handling, the Tools/Snapshot panel, a hand-edited
project file rendering on the Model/Design pages). `docs/reviews/w2-breakage.md`
and `w3-breakage-2.md`'s bug classes (uncaught exceptions from engine errors,
state leaking, autosave of a bad state) were the starting list; the one class
this session actually found belongs to the first of those.

## 1. Summary

**Findings: 1 crash class (8 call sites) plus 1 follow-on "autosave of a bad
state" it uncovered once fixed, both severity crash / data-loss and both
fixed, with a reproducing test per site.** No wrong-number findings; a broad
sweep of Compare, the HTML report, the Tools panel, rate-change, penalty
weights, Tweedie/binomial, cells alpha, the CLI and the 250k-row compact path
found nothing else (§3).

The one class: **a hand-edited project file can put anything in a numeric
field, and both `Project.validate()` and several Model/Design page widgets
assumed a real number.** `"alpha": "abc"` (or a boolean, `NaN`, the wrong
shape) raised `TypeError`/`ValueError`/`OverflowError` straight out of a
`<=`/`<`/format-spec comparison — in `validate()` itself, which the CLI and
two pages call unguarded, and independently in the Model page's `alpha`,
`tweedie_power`, `l1_ratio`, `cv`, `n_alphas`, `base_rate_override` and
per-interaction `alpha`/`penalty_weight`/`min_cell_exposure` widgets, and the
Design page's clamp box and its own copy of the interaction widgets. Every
one of these is reachable simply by **opening the Model or Design page**, or
running `easy-glm validate|run|export`, on such a file — no clicking required
— which is worse than the usual "typed a bad value in the browser" case the
existing hardening (`ui.number_in_range`) already covers, because that
widget's own safety net runs *after* the crash: the unguarded `float(...)` /
`int(...)` / f-string format happens while building the `value=` the widget
would have shown.

This violates two standing rules at once: AGENTS.md's "`cli.py` never raises
through to the user: every failure is a `CliError`" and the workbench
convention "Errors are messages, never tracebacks." Both were already true
for a bad value **typed in the browser**; they were not true for the same
value arriving via a hand-edited file, because nothing between reading the
file and drawing the widget converted the value.

## 2. Findings

| # | Surface | What I did | What happened | Severity | Fixed |
|---|---|---|---|---|---|
| 1 | `Project.validate()` | Hand-edit a project's `models.<m>.penalty.alpha` to `"abc"` (also: `tweedie_power` to `"abc"` with `family="tweedie"`; an interaction's `penalty_weight`, `min_cell_exposure` or `alpha` to `"abc"`; `design.variables.<v>.clamp` to `["abc", 10]` or to `5` (not a list); `data.split.fraction` to `"abc"` with `mode="random"`). Call `project.validate("m1")`. | `TypeError: '<=' not supported between instances of 'str' and 'int'` (alpha, interaction fields), `ValueError: could not convert string to float: 'abc'` (tweedie_power, clamp), `TypeError: object of type 'int' has no len()` (clamp as a bare number) — raised out of `validate()` itself instead of returning one more problem string. | **crash** | yes |
| 2 | CLI (`validate`/`run`/`export`) | Same file, `easy-glm validate project.json` as a subprocess. | `main()` only catches `CliError`/`ProjectFileError`/`OSError`; the `TypeError`/`ValueError` from finding 1 propagated as a raw Python traceback on stderr with exit code 1 — a traceback the docstring ("every failure is a `CliError`") explicitly promises never happens. | **crash** | yes |
| 3 | Model page (`pages_model.py`) | Open the Model page on a project with `alpha`, `tweedie_power`, `l1_ratio`, `cv`, `n_alphas`, `base_rate_override`, or an interaction's `alpha`/`penalty_weight`/`min_cell_exposure` set to `"abc"`. | `ValueError: could not convert string to float: 'abc'` (or `Unknown format code '%'/'g' for object of type 'str'`) raised while building the widgets — a raw traceback in the browser, before the user ever sees `validate()`'s message naming the problem. Nine distinct expressions crashed this way: `float(cfg.tweedie_power)` (unconditional, even for a non-Tweedie family), the `alpha`/`l1_ratio`/`cv`/`n_alphas`/`base_rate_override` widgets' `value=`, and the interaction row's `min_cell_exposure:.2%` / `penalty_weight:g` captions and `alpha` widget. | **crash** | yes |
| 4 | Design page (`pages_design.py`) | Same file, open the Design page (the per-variable clamp box, and its own copy of the interaction alpha/caption widgets). | `float(vd.clamp[0])`/`float(vd.clamp[1])`, and the same `min_cell_exposure`/`penalty_weight`/`alpha` expressions duplicated on this page, raised the same way. | **crash** | yes |
| 5 | Model page, with a project path (autosave live) | Fix findings 3–4 first, so the page renders past the bad `alpha`/interaction `alpha` instead of crashing. Re-open the same hand-edited file, this time with `S.set_project(project, path)` — an ordinary session, not a read-only smoke test. | The page's own "did anything change?" reconciliation (`_config`'s `new_pen`/`new_vals` loop, and the Design page's per-interaction "save if the box's value differs from the stored one") compares the *fallback* number the widget now shows against the stored `"abc"`/`"xyz"`, finds them different, and **autosaves the fallback over the original value with no message at all** — the moment the page renders, before the user touches a control. This is finding 3's crash turned into a silent file rewrite: the hand-edited mistake vanishes from the project file with no record of what it was or why it changed. | **data loss** (silent, unexplained overwrite of a project field) | yes |

Eight call sites for the crash (project.py: 6 comparisons; pages_model.py: 6
expressions; pages_design.py: 3 expressions — some fields are read in more
than one place), one root cause, one shared fix pattern; finding 5 is the
same root cause's second symptom, found by fixing the first and then asking
"what does the now-rendering page *do* with that fallback number".

**Fix.** `easy_glm/workflow/project.py` gained `_is_finite_number(value)`
(true only for a real, non-boolean, finite `int`/`float`) and every
comparison in `validate()` that could receive a hand-edited field now checks
it first — so a non-numeric value becomes one more line in the problem list
instead of an exception, using the *same* message text the numeric out-of-
range case already used (`alpha must be > 0`, `tweedie_power must be
strictly between 1 and 2`, …), which is why no existing test needed to
change. `easy_glm/app/ui.py` gained `safe_float(value, default)` and
`safe_int(value, default)` (a value that cannot convert, or converts to a
non-finite float, returns `default`), and `pages_model.py`/`pages_design.py`
now build every at-risk widget value through them, clamping the couple of
widgets whose own `min_value`/`max_value` would otherwise raise on an
out-of-range-but-numeric value (`l1_ratio`, `cv`) rather than a non-numeric
one. `NaN` is caught by the same fix (it does not raise, but silently passed
`validate()`'s numeric checks before — now it is reported like any other bad
alpha).

For finding 5, `ui.py` gained `repair_number(value, default, label)`, which
returns `(value or default, message)` — the same shape as the Split page's
`_seed_value` helper this mirrors exactly, down to a value that is legitimately
`None` (an unset `base_rate_override`, an interaction with no `alpha`
override) never being reported as a problem. `pages_model.py` and
`pages_design.py` now `ui.flash("warning", ...)` that message when one comes
back, before the reconciliation loop that would otherwise save the fallback
with no trace — so opening the page on a bad file now says, in the same
banner the seed box already uses for the same situation, *"alpha in the
project file ('abc') is not a usable number; using 0.001 instead"*, and only
then is 0.001 the number saved.

Tests: `tests/test_workflow.py::TestProject::test_a_non_numeric_field_is_reported_not_raised`
(direct, all six `validate()` sites, `NaN` included), a new parametrised case
in `tests/test_e_f_extras_cli.py::TestCliValidate` (four project-file
mutations through the actual `easy-glm` subprocess), and a new
`tests/test_w5_breakage.py` with the same mutations exercised through
`Project.validate()`, the CLI, `AppTest` against both the Model and Design
pages with a **read-only** project (findings 1–4: 18 page-render cases × the
two pages, plus the clamp case), and `AppTest` against the Model/Design pages
with a project that has a **path**, so autosave actually runs (finding 5: the
warning appears and names the old and new value, the file ends up holding the
repaired number, and a legitimately-unset `alpha` never triggers the
message) — 30 new tests there, all failing on the pre-fix tree (12 of them:
the eight crash sites still reachable through the page-render cases, plus
finding 5's three autosave tests) and passing after. Every one of these was
verified against the pre-fix code first (`git stash` the three touched `src`
files and re-run) to confirm it reproduces the reported bug, not a difference
of opinion about wording.

## 3. What was tried and came back clean

No crash, no data loss and no wrong number in any of the following — kept
brief since "nothing happened" earns no table row, only a note of what was
covered:

* **Compare page.** Fewer than two fitted models → the documented message,
  never a crash; the picker cannot select the same model twice (the
  challenger list excludes the champion by construction); deleting the
  challenger's model out from under an open Compare tab → the page falls back
  to "needs two fitted models" once only one remains, no stale-widget crash;
  champion vs. challenger with **different targets and families** (Poisson
  weight="Exposure" vs. binomial weight=None) rendered every tab (A/E by
  variable, Lift, Double lift, Relativities that differ) with no exception.
* **HTML report (`to_report_html`).** A model with **zero holdout rows**
  (100% training split) — alone and paired with a normal challenger; a
  **binomial** champion — alone and vs. a Poisson challenger with a different
  target; an interaction that **kept no cell** (min exposure 99%) — alone and
  vs. a challenger whose same-pair interaction kept cells; the champion
  passed as its own challenger (silently not a comparison, matching the
  documented rule). All produced valid, appropriately-sized HTML with no
  exception.
* **Tools / Undo / Redo / Snapshots / Rebalance.** `docs/reviews/` and
  `tests/test_d5_tooling.py` already cover window=1/even/wider-than-the-table,
  cap<floor, round-to-zero-decimals, ten repeated applications, undo past the
  start, redo after a fresh edit, a tool refused on an interaction's cell
  table, and rebalance being exact and one undo step — this session re-drove
  those plus: an **empty** and a **duplicate** snapshot name (already
  refused, unchanged); comparing/restoring after **deleting the snapshot
  currently shown in the diff selectors** — the selectors fall back to their
  default option rather than crash on a stale widget value, even with two
  snapshots present and one deleted mid-comparison; **rebalance on zero
  training exposure** — `rebalance_override` returns `None` (guarded by
  `not current > 0`) and the panel shows nothing rather than dividing by
  zero; restoring a snapshot whose adjustments no longer match the model's
  *bands* (not just a missing variable) — `apply_adjustments` raises
  `AdjustmentError`, and `state.refresh_adjustments` already drops the
  refused adjustment and reports it rather than crashing.
* **Rate-change / `current_premium`.** Zero, negative, null and infinite
  values in the premium column are refused by name and count before the
  offset is derived (existing, re-verified); renaming the premium column
  mid-session carries the derivation (existing, re-verified).
* **Compact-matrix path (piece G) through the workbench, not just the
  core.** A synthetic 250k-row book (above `SPARSE_ROW_THRESHOLD`, with a
  two-stage interaction and a linear term) driven through
  `prepare → run_model → ae_by_variable → to_report_html → tooling.smooth →
  to_script → exec(script)`, and separately through `AppTest` for every page
  (Model, Diagnostics, Tables, Compare, Export, Explore) — all completed in
  well under a second each, no exception, and the exported script reproduced
  the fit.
* **CLI.** Four `easy-glm run` processes launched concurrently against the
  same project and `--out` folder: all four exited 0, and the resulting
  report/script/scorer files were intact (well-formed HTML, non-truncated) —
  the CLI's "every artefact command fits afresh, nothing is cached or shared
  across processes" design has no shared mutable state for a race to corrupt
  here, unlike the workbench's runs folder (W3's findings 1–2, out of scope
  for this session since nothing changed there).

## 4. Test counts

| | before | after |
|---|---|---|
| `pytest -q tests` (repo venv, Streamlit 1.57) | 776 passed, 1 skipped, 1 deselected | 811 passed, 1 skipped, 1 deselected |
| `pytest -q tests/test_app*.py tests/test_w*.py tests/test_d*.py` (Streamlit 1.63 venv) | 339 passed, 1 skipped | 370 passed, 1 skipped |
| `tests/e2e` (Playwright, `EASY_GLM_E2E=1`) | 3 passed | 3 passed |

Gates: `black .`, `ruff check .`, `mypy src/easy_glm/core src/easy_glm/workflow
--ignore-missing-imports` all clean; `git diff release-0.4 -- tests/test_golden.py
tests/fixtures` empty (no golden number touched).

## 5. Left open

Nothing from this session is left open — every finding was a crash, all
eight call sites were fixed, and every fix has a reproducing test that fails
on the pre-fix tree. Two smaller, related observations that are **not** filed
as findings because they are pre-existing, deliberately-scoped behaviour, not
new breakage:

* `design.min_level_share` and a handful of other `VariableDesign` fields
  still have no type check in `validate()` at all (not just "crashes on a bad
  type" — they are simply not validated), unlike `penalty_weight`, `clamp`,
  `alpha` and `tweedie_power`, which this session made robust because they
  were the ones that actually crashed a page. Worth a follow-up pass if
  another hand-edited-file scenario surfaces a crash through one of them.
* `l1_ratio` and `cv` values that are numeric but out of the widget's declared
  range (e.g. `l1_ratio: 5`) are now clamped into range for display rather
  than crashing `st.slider`/`st.number_input` — a slightly different fallback
  from `ui.number_in_range`'s "show the number, name the problem, keep the
  old value" pattern used elsewhere, chosen here because these two widgets
  have no free-standing "problem" message of their own to attach to; `Fit`
  still reads the *project's* stored value, not the clamped display value, so
  no number silently changes.
