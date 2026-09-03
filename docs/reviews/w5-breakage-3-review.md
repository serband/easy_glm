# Independent review — Breaker session 3 fixes (`piece/breaker3` vs `release-0.4`)

## Round 2 final verdict: **APPROVED**

Commit `26bb80c` ("Round 2: fix out-of-range penalty fields silently
autosaving, and the number_in_range infinite-rerun hang") closes both items
this review raised. Re-verified by running code, not by reading the diff or
trusting the breaker's counts:

- **Blocking item (§2 finding A) — fixed.** Rebuilt the exact repro from
  round 1 with fresh scripts (not just the new checked-in tests) against a
  project with a live `project_path` (autosave on): `l1_ratio: 5.0`,
  `n_alphas: -5`, `cv: 999` (mode `cross-validated`) each now produce a
  `ui.flash("warning", ...)` naming the field, the bad stored value and the
  replacement —
  `"l1_ratio in the project file (5) must be between 0 and 1; using 1
  instead."`,
  `"n_alphas in the project file (-5) must be between 3 and 100; using 3
  instead."`,
  `"cv in the project file (999) must be between 2 and 10; using 10
  instead."` — and the project file on disk after the run holds exactly the
  repaired value in each case (`1.0`, `3`, `10`), matching the warning. No
  regression case reproduces the old silent overwrite.
- **Should-fix item (§2 finding B) — fixed.** The hang no longer reproduces.
  Re-ran a fresh (not the checked-in test) `alpha: 1e9` / `base_rate_override:
  -1.0` project through `pages_model.render()` with `AppTest(...,
  default_timeout=8)` on **both** venvs: repo venv (Streamlit 1.57) returned
  in 0.07s, `st163` venv (Streamlit 1.63) in 0.18s — both with no exception
  and the same two-field warning message, and the saved values repaired
  (`alpha` → `10.0`, `base_rate_override` → `None`, i.e. unset since 0 is the
  "exact" default and the repaired value clamped up to `lo=0`).
- **Split page** — rendered without exception on both venvs with a normal
  `seed=42`, random-split project (the surface the commit message flags as
  at risk of `StreamlitMixedNumericTypesError` from `repair_number` widening
  an int to a float). No mixed-type crash on either version.
- **Regression check.** Reproduced the round-1→round-2 diff's own claim
  directly: archived the pre-round-2 source (`git archive 34d974f`) into a
  side directory and ran the 6 new `test_w5_breakage.py` tests against it —
  all 6 fail there (the three silent-autosave cases and the two hang cases
  time out with `RuntimeError: AppTest script run timed out after 8(s)`,
  confirming the hang is real and not a test artifact) — and all 6 pass
  against this branch, 36/36 for the whole file.
- **No number moved.** `git diff <merge-base 790d028c> HEAD -- src/easy_glm/core
  src/easy_glm/engine` is still 0 lines; `tests/test_golden.py`/`tests/fixtures`
  diff is still 0 lines; no assertion removed anywhere in `tests/`.
- **No false positives.** Direct `Project.validate()` sweep: `cv` ∈
  `{None, 2, 5, 10}`, `n_alphas` ∈ `{2, 3, 20, 100}`, `l1_ratio` ∈
  `{0.0, 0.5, 1.0}` — all clean (`[]`).
- **Gates and full suites, re-run clean on this commit:** `black --check .`,
  `ruff check .`, `mypy src/easy_glm/core src/easy_glm/workflow
  --ignore-missing-imports` all clean; `pytest -q tests` (repo venv,
  Streamlit 1.57) — **817 passed, 1 skipped, 1 deselected**, matching the
  round-2 report exactly; `pytest -q tests/test_app*.py tests/test_w*.py
  tests/test_d*.py` (`st163` venv, Streamlit 1.63) — **376 passed, 1
  skipped**, matching exactly.

No new blocking or should-fix items found in this pass. The one remaining
open item from round 1 — no CHANGELOG entry yet for this session, unlike W3/
W4's own sections — still applies (not re-checked this round; ranked
should-fix, not blocking, in the original review below) and should be closed
before the 0.4.0 release commit, not before this piece merges.

---

# Round 1 review (superseded above; kept for the record)

## Verdict: **one more round needed — not clean, but not a full re-fight**

The five findings in `docs/reviews/w5-breakage-3.md` are real, are fixed, and
are fixed correctly: every one reproduces on `release-0.4` and is clean on
this branch, `Project.validate()` now returns problem strings instead of
raising, the CLI turns them into a clean exit-1 message for `validate`/`run`/
`export`, valid values still validate clean, no fitted number moved (`core`/
`engine` diff against the merge-base is empty; golden/invariants/recovery all
pass), and no test was weakened. Gates are clean on both venvs.

But the fix is **incomplete in exactly the way its own report claims it is
not**: on the Model page, a hand-edited `cv`, `n_alphas` or `l1_ratio` that is
a real, finite number but outside the widget's declared UI range is silently
clamped and the clamped fallback is **autosaved over the original value with
no message at all** — the same "autosave of a bad state" class findings 1–5
exist to close, on fields this piece touched, verified by running the actual
code (§2, finding A). The breaker's own `docs/reviews/w5-breakage-3.md` "Left
open" section explicitly asserts this cannot happen ("Fit still reads the
project's stored value, not the clamped display value, so no number silently
changes") — that sentence is factually wrong for `l1_ratio` and is a
regression: on `release-0.4`, the same hand-edited `l1_ratio: 5.0` neither
crashes nor changes on disk (`st.slider` does not enforce its bounds); on this
branch it silently becomes `1.0`. That is data loss, the review brief's own
blocking category, on a surface this piece specifically set out to hardened
and reports as fully closed ("Nothing from this session is left open").

This is a narrow, mechanical gap — the same `repair_number`/`ui.flash`
pattern already used four lines above each of these three widgets just needs
to also cover the range-clamp, the way `ui.number_in_range` already does for
`alpha`/`base_rate_override` — not a design problem, and not grounds to
re-open the five original findings or re-run the whole breaker sweep. One
follow-up round, scoped to §2 finding A, is enough.

Separately (§2, finding B): `ui.number_in_range` itself — untouched by this
diff, so **not a regression** — hangs indefinitely (confirmed via Streamlit's
own `AppTest` timeout, reproduces identically on `release-0.4` and on this
branch) when the *stored* value it is asked to show is a valid number outside
`lo`/`hi` (e.g. a hand-edited `alpha: 50` or `base_rate_override: -5`, both
legal per `Project.validate()`). It always resets to the same invalid stored
value, which retriggers the same out-of-range check, forever. This is squarely
in the bug family this session was chartered to hunt (the pages_design.py
interaction-alpha widget carries a comment about exactly this scenario) and
is worse than any of the five findings it fixed — a crash fails cleanly, this
does not fail at all. It does not block this piece (no line here was touched
by `piece/breaker3`), but it should not wait for someone to hit it in
production.

---

## 1. Do the five findings reproduce on `release-0.4` and get fixed here?

Confirmed by running code, not by reading the diff. `release-0.4` and
`piece/breaker3` have diverged (`git merge-base release-0.4 HEAD` =
`790d028c`; `release-0.4` alone carries 12 unrelated commits — the README
rewrite, the compose test, etc. — that are not on this branch), so
`git diff release-0.4...HEAD` (three dots — merge-base diff) is the only
correct isolation of this piece's own changes; it is exactly the 8 files /
757+/21− lines the task brief describes, and `git diff release-0.4 HEAD`
(no dots) or `git log release-0.4...HEAD` both pull in that unrelated
history and would give a false picture.

- Ran `tests/test_w5_breakage.py` (30 tests) from `<worktree>/tests` with
  `PYTHONPATH=/Users/serban/Documents/Projects/easy_glm/src` (the real
  `release-0.4` checkout's source, read-only, never modified) against the
  same, unmodified test file: **24 failed** — every finding-1/2/4 case, with
  the exact `TypeError`/`ValueError` the breaker's report quotes. The 6 that
  passed are the CLI subprocess test (it hard-codes `parents[1]/"src"`, i.e.
  always spawns against *this worktree's* fixed CLI regardless of the
  `PYTHONPATH` set for the parent process — verified separately below) and
  the "a legitimate `None` is never reported" test, which needs no fix to
  pass.
- Same 30 tests on this worktree's own fixed source: **30 passed**.
- CLI finding verified directly (not through the test's hard-coded path):
  built a project with `penalty.alpha: "abc"` and ran
  `python -m easy_glm.cli validate` with `PYTHONPATH` pointed at each source
  tree in turn.
  - `release-0.4`: raw `TypeError` traceback on stderr, exit 1.
  - this worktree: `easy-glm: proj.json has 1 problem(s)` /
    `- m1: alpha must be > 0 (...)`, exit 1.
  - `run` and `export` on the same file (fixed source): both refuse cleanly
    with the same message and exit 1 (item 4 below).

## 2. No number changed

- `git diff <merge-base> HEAD -- src/easy_glm/core src/easy_glm/engine` — **0
  lines**. The fix cannot have moved a fitted coefficient; it never touches
  the code that computes one.
- `pytest -q tests/test_golden.py tests/test_invariants.py
  tests/test_recovery.py` — **57 passed**.
- `git diff <merge-base> HEAD -- tests/test_golden.py tests/fixtures` — **0
  lines**.

**But** — checking the same "does a hand-edited field silently change"
question this piece exists to answer, beyond its own 5 findings, turned up a
real gap, run directly against the fixed worktree:

**Finding A (blocking — data loss, in scope).** `pages_model.py`'s `_config`
builds `cv`, `n_alphas` and `l1_ratio` display values through
`ui.repair_number` (which only flags a value that is *not a real number* —
`"abc"`, `NaN`) and then, separately, clamps that value into the widget's
declared range with a bare `min(hi, max(lo, ...))` — no problem message, even
though the reconciliation loop at the bottom of `_config` unconditionally
autosaves whatever the widget shows:

```python
n_alphas = c4.number_input(
    "alphas on path", 3, 100,
    min(100, max(3, int(n_alphas_value))),
    ...
)
...
new_pen = dict(..., n_alphas=int(n_alphas), l1_ratio=float(l1), ...)
...
for k, v in new_pen.items():
    if getattr(cfg.penalty, k) != v:
        setattr(cfg.penalty, k, v)
        changed = True
...
if changed:
    S.touch()  # writes the project to disk unconditionally
```

Verified with `AppTest` against a project with a live `project_path` (so
autosave runs), on the fixed worktree:

| hand-edited value | saved after opening Model page | warning shown |
|---|---|---|
| `penalty.l1_ratio: 5.0` | **`1.0`** | none |
| `penalty.n_alphas: -5` | **`3`** | none |
| `penalty.cv: 999` (mode `cross-validated`) | **`10`** | none |

`l1_ratio` is a genuine **regression**, not just an incomplete fix: on
`release-0.4`, `st.slider`'s `min_value`/`max_value` are not actually
enforced against `value=` (confirmed directly — `st.slider("x", 0.0, 1.0,
5.0, 0.05)` returns `5.0` unmodified, no exception), so the same hand-edited
`l1_ratio: 5.0` neither crashes nor changes on disk pre-fix — it round-trips
exactly. Post-fix it is silently rewritten to `1.0`. `n_alphas`/`cv` use
`st.number_input`, which *does* raise (`StreamlitValueBelowMinError` /
`StreamlitValueAboveMaxError`) for an out-of-range `value=` — confirmed on
`release-0.4` with `n_alphas: -5` (crash) and `cv: 999` (crash) — so those two
did trade a crash for a silent overwrite, which is progress, but not the
"repair and say so" the piece's own summary claims for every field it
touched, and not what the "Left open" section says happens ("Fit still reads
the project's stored value, not the clamped display value, so no number
silently changes" — demonstrably false for all three: the *stored* value is
overwritten, not just the display).

The new test suite does not catch this because it never exercises this case:
`tests/test_w5_breakage.py` only ever sets `l1_ratio`/`cv`/`n_alphas` to the
string `"abc"` (lines 173–175), never to an in-range-type-but-out-of-UI-range
number, so the "repaired" path this finding hits is never reached by any
existing assertion.

**Finding B (not blocking — pre-existing, same bug family, worth a
follow-up).** `ui.number_in_range` (used for `alpha` and
`base_rate_override`, and untouched by this diff) hangs — not crashes,
*hangs* — when the value it is told to show is a real, finite, positive
number outside `lo`/`hi`. Its repair path resets the widget to the same
`value` it was given (the project's stored number), which fails the same
range check again, which reruns again, forever. Reproduced with
`AppTest(..., default_timeout=8)` raising `RuntimeError: AppTest script run
timed out after 8(s)` for a hand-edited `alpha: 50.0` (legal per
`Project.validate()`, which only requires `alpha > 0`) — **identically on
`release-0.4` and on this worktree**, and again for `base_rate_override:
-5.0` on this worktree. This is not a regression and touches no line in this
diff, so it does not block this piece, but it is exactly the class of bug
this session's brief describes ("both `Project.validate()` and several
Model/Design page widgets assumed a real number"), it is worse than any of
the five crashes that were fixed (a crash fails cleanly; this consumes the
session in an endless rerun loop), and the Design page's own interaction-alpha
widget already has a code comment acknowledging the identical scenario for a
different field — it should not be left for a fourth breaker session to find.

## 3. Repair-and-say-so (alpha = `"abc"`)

Ran directly (not just the checked-in test) against both Streamlit versions:

- **Streamlit 1.63** (`st163` venv), Model page, `penalty.alpha: "abc"`,
  project given a path (autosave live): page rendered
  (`at.exception == ElementList()`, empty), warning text was exactly
  `"alpha in the project file ('abc') is not a usable number; using 0.001
  instead."`, and the file on disk held `0.001` after the run completed.
- Same venv, Design page, interaction `alpha: "xyz"`: rendered, warning
  `"DrivAge×Region: alpha in the project file ('xyz') is not a usable number;
  using 0 instead."`, saved interaction `alpha` became `null` (0 → unset, as
  documented).
- **Streamlit 1.57** (repo venv): covered by the full `pytest -q tests` run
  (811 passed includes the `AppTest`-based finding-4 tests on this venv).

**Ordering — the design autosaves; does the message precede the save?**
Yes, in the sense that matters: `ui.flash(...)` is called and the notice is
queued in session state before `_config`'s reconciliation loop reaches
`S.touch()` (the write) later in the same script run. The write and the
visible banner do not land in the same *frame* — `flash()` explicitly queues
for the next run because a pending `st.rerun()` (fired by the same
reconciliation loop, right after `touch()`, since `alpha` is not in the
`("base_rate_override", "notes")` rebuild-only list) discards anything drawn
so far — but that rerun is internal and immediate; nothing the user did
triggers it, and by the time the page settles the warning is showing and the
file holds the repaired number. `AppTest.run()` (which drives reruns to
completion) confirms both are true after one `.run()` call. **The CHANGELOG
does not say any of this** — `git diff <merge-base> HEAD -- CHANGELOG.md` is
empty; unlike the two previous breaker sessions (`### The persisted-run
folder is shared state (W4) — the second breaker session`, `### Workbench
hardening (W3) — the break-it review's blocking findings`), this session has
no CHANGELOG section at all yet. Given the precedent, that reads like it is
deferred to a later consolidation/release commit (as `dc3092f`, the 0.4.0
release commit, did for the README piece) rather than an omission — but it
is worth confirming that assumption before release, since the brief asks
this question explicitly. Ranked should-fix, not blocking.

## 4. `Project.validate()` / CLI contract

- `validate()` returns a list of problem strings for every bad field
  (verified directly — no exception for any of the 7 mutated fields).
- CLI `validate`: clean message, exit 1 (verified above).
- CLI `run` and `export --script`: both refuse with
  `easy-glm: model 'm1' cannot be fitted` / the same problem line, exit 1
  (verified directly with a hand-edited `alpha: "abc"` project).

## 5. No false positives on valid values

Ran `Project.validate()` directly on a project built one field at a time:
`alpha=0.001` → `[]`; `alpha=1` (int) → `[]`; `tweedie_power=1.5` (family
`tweedie`) → `[]`; `design.variables[v].penalty_weight=0` → `[]` (the
*design-level* `penalty_weight` rule is `>= 0`, unlike the *interaction*
`penalty_weight` rule, which is `> 0` and correctly rejects `0` —
`interactions[i].penalty_weight=0` → `["... penalty_weight must be > 0"]`,
which is the existing, intentional, unchanged rule, not a false positive);
`interactions[i].min_cell_exposure=0` → `[]`; `clamp=None` → `[]`;
`clamp=[1.0, 10.0]` → `[]`; `split.fraction=0.7` → `[]`. All clean.

## 6. Spot-check three "no other findings" probes (Streamlit 1.63)

- **Compare with an unfitted model.** Project with one fitted + one
  unfitted model, `pages_compare.render()`: no exception, `st.info`
  "Compare needs **two fitted models**. ..." — matches the report.
- **HTML report, zero holdout rows.** 100%-train split (`traintest` all
  `1`), `to_report_html(p, {"m1": run}, prepared, champion="m1")`: no
  exception, 52 KB of valid HTML (`<html` present). Matches.
- **Snapshot delete while its diff is shown.** Two snapshots taken (no
  adjustments between them, so the diff was trivially empty either way — a
  limit of this quick probe, not of the finding being checked), both
  selected in the "Compare"/"with" diff boxes, then the first deleted via
  the delete-twice flow: no exception before or after the delete
  (`at.exception == ElementList()` throughout), page kept rendering. Matches
  "no crash"; did not independently confirm the exact fallback message since
  the two versions happened to be identical regardless.

## 7. Gates

- `black --check .` — clean (106 files).
- `ruff check .` — clean.
- `mypy src/easy_glm/core src/easy_glm/workflow --ignore-missing-imports` —
  clean (19 files).
- `pytest -q tests` (repo venv, Streamlit 1.57): **811 passed, 1 skipped, 1
  deselected** — matches the report's claimed after-count exactly.
- `pytest -q tests/test_app*.py tests/test_w*.py tests/test_d*.py` (`st163`
  venv, Streamlit 1.63): **370 passed, 1 skipped** — matches exactly.
- `git diff <merge-base> HEAD -- tests/test_golden.py tests/fixtures` —
  empty.
- `git diff <merge-base> HEAD -- tests | grep '^-' | grep -i assert` —
  empty: no existing assertion was touched or weakened, only new ones added.

---

## Ranking

**Blocking (send back for one round):**
- §2 Finding A — `cv`/`n_alphas`/`l1_ratio` on the Model page silently
  autosave a clamped fallback over a numeric-but-out-of-range hand-edited
  value with no message, contradicting this piece's own "repair and say so"
  pattern and the explicit (incorrect) claim in its own report that this
  cannot happen; `l1_ratio` is a genuine regression from `release-0.4`'s
  behavior (previously preserved verbatim, now silently changed). Fix: run
  the same value through `ui.repair_number`/`ui.flash` *after* the range
  clamp too (or extend `repair_number` to take `lo`/`hi` and report a
  clamp the same way it reports a non-numeric value), and add a test that
  sets each of the three fields to an in-range-type value that is merely
  out of the widget's UI bounds (e.g. `l1_ratio: 5`, `n_alphas: -5`,
  `cv: 999`) and asserts both a warning and that the saved value is not
  silently different from a value the user would recognize as "what I had,
  fixed" — mirroring the existing finding-4 tests exactly.

**Should-fix (not blocking this piece, but don't let it wait for release):**
- §2 Finding B — `ui.number_in_range` hangs indefinitely on a legal,
  in-range-per-`validate()` but UI-out-of-range stored value (`alpha`,
  `base_rate_override`). Pre-existing, reproduces on `release-0.4` too, no
  line here touched it — but it is the same bug family, worse in kind than
  anything this session fixed, and squarely inside what this session's
  brief asked it to hunt for.
- CHANGELOG has no entry for this session (unlike W3/W4's own sections);
  confirm whether that is deliberately deferred to a later release commit,
  and if so make sure it lands before 0.4.0 ships, since the review brief
  specifically asks whether the CHANGELOG documents the autosave-ordering
  behavior in §3.

**Nice-to-have:**
- `design.min_level_share` and other untyped `VariableDesign` fields
  remain unvalidated for type (per the report's own "Left open" note) —
  no evidence they crash anything today, but the same class of bug as
  finding 1 if a page ever reads one straight into a widget.
