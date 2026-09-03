# Review: piece R11 — the README release gate (round 1)

## Verdict

**Not yet approved — one blocking fix required.** The gate itself is real (it
genuinely runs the page and fails when the page breaks — verified by
deliberately breaking a block and watching the test go red), every quality
gate is clean, the full suite passes, and 20 of the 21 literal `# →` output
values I checked against an actual run match to the precision shown. But one
of them does not: the reloaded-model prediction printed in §8
(`[0.00153043 0.00649728 0.08165957]`) is off by roughly 12% from what the
exact code in that block actually prints on a clean run
(`[0.00172092 0.00686527 0.08666724]`), reproducibly, not as floating-point
noise. That is precisely the "the README said one thing, running it gives
another" problem 0.4 exists to close, and it sits inside the very section
that makes the page's headline "exact scoring" promise. Fix that one number
(regenerate it from a real run, the way `docs/checks/readme-gate.md` already
is) and this is a clean approve — everything else is should-fix or
nice-to-have.

## What I verified

Interpreter: `/Users/serban/Documents/Projects/easy_glm/.venv/bin/python`,
`PYTHONPATH=<worktree>/src` on every command; confirmed
`easy_glm.__file__` resolved under the worktree
(`/private/tmp/.../wt-readme/src/easy_glm/__init__.py`) before running
anything.

1. **The gate really tests the page.** `tests/test_readme.py` extracts every
   ```` ```python ```` fence with the same regex I ran standalone
   (`r"```python( skip-test)?\n(.*?)```"`, `re.DOTALL`): **20 blocks total, 1
   `skip-test`** (`eglm.rate_model.launch_editor(...)`, which opens a browser
   tab — the only kind of block the rule allows to be exempted), well under
   the `MAX_SKIPPED_BLOCKS = 3` cap. The 19 runnable blocks execute in one
   shared namespace, in document order, in a temp cwd with `tests/fixtures`
   symlinked in; every `examples/*.py` runs standalone via `subprocess`,
   exit-code checked. Fresh run: `pytest -q tests/test_readme.py` →
   **11 passed in 23.19s** (README blocks alone: 19 blocks in ~3.5s, printed
   by the test itself).
2. **Broke it on purpose.** Backed up README.md, inserted
   `raise RuntimeError('deliberate-break-for-review')` right after the
   `PREDICTORS = [...]` line in §3, reran
   `pytest tests/test_readme.py::test_readme_code_blocks_all_run` — it failed
   with a clear `AssertionError` naming the block index and showing the block
   source. Restored the file with `git checkout -- README.md`; confirmed
   byte-identical to the backup and `git status` clean afterward. The gate
   does what it claims.
3. **Fresh-clone realism.** Extracted all 19 runnable blocks into one script,
   ran it from an empty directory with only `tests/fixtures/*.parquet` copied
   in (not symlinked — a genuine copy, as a clone would have) and
   `PYTHONPATH` pointed at the worktree's `src`, cwd at that empty directory.
   It ran to completion with no hidden setup the test provides that a reader
   wouldn't have — the data path the README names
   (`tests/fixtures/french_motor_50k.parquet`) exists at the repository root
   in a plain clone.
4. **Honesty spot-check, all 27 `# →` literal-value comments in README.md**
   (not just two): 26 match an actual run to the precision shown, including
   the trickier ones — the `Gini champion/challenger` 4-decimal values, the
   two-stage base-rate identity described as "run-to-run noise", the
   two-stage vs. mains invariant (`True`), the `holdout_ae`/`holdout_gini` in
   `run.summary()`, the lapse-probability vector, and the g-scale benchmark
   table (200k/1M/5M rows → 1s/4s/21s, 0.37/0.86/2.59 GB), which I
   cross-checked against `docs/checks/g-scale.md` and it reproduces exactly.
   The one mismatch is documented below (blocking #1).
5. **Rules.** `git diff release-0.4...HEAD | grep -ri bike` → no matches (an
   earlier grep pass appeared to hit `docs/HANDOVER.md`'s own statement of
   the rule — re-ran against a saved copy of the full diff and confirmed
   zero matches; `docs/HANDOVER.md` has no diff in this range at all). Every
   numeric predictor in every README/example `DesignSpec.from_data` call
   (`DrivAge`, `BonusMalus`, `Region` categorical) defaults to step; the one
   opt-in (`Density`, via `linear=[...]`) is explicitly called out as such,
   twice. No private/proprietary variable names — everything is standard
   `freMTPL2freq` column names (`DrivAge`, `Region`, `BonusMalus`, `Density`,
   `ClaimNb`, `Exposure`). All 17 in-page anchor links resolve against the
   actual (GitHub-slugified) headings; all 8 relative file links
   (`docs/checks/*`, `AGENTS.md`, `CONTRIBUTING.md`, `LICENSE`,
   `docs/WORKBENCH_PLAN.md`, `tests/fixtures/french_motor_50k.parquet`, two
   PNGs) resolve on disk; external links (glum, aglm, CASdatasets) look
   correct.
6. **Release metadata.** `pyproject.toml`: `version = "0.4.0"`,
   `readme = "README.md"`, classifiers and `[project.urls]` present and
   correctly pointed at `serband/easy_glm`. `CHANGELOG.md`: heading
   `## 0.4.0 (2026-09-03)` (matches today), with a full R11 entry describing
   the gate mechanism and the README rewrite accurately — nothing overstated
   there. Two 0.4 CHANGELOG features (`fit_glm(..., aggregate=True)`,
   `fit_glm(..., progress=callable)`, `penalty_weight`, `tweedie_power`) are
   not demonstrated on the front page but are covered in the linked
   `docs/checks/g-scale.md` / `e-f-extras-cli.md` — a reasonable scope
   choice for a page that is already a 17-section walkthrough, not an
   omission I'd block on (nice-to-have below).
7. **Examples**: all 9 run standalone (`pytest`'s own parametrized
   `test_example_runs`, and independently via the full suite run below);
   `docs/checks/readme-gate.md` (generated by
   `scripts/checks/readme_gate.py --write`, which actually runs and times
   things rather than being hand-typed — I read the script to confirm) times
   them individually, 1.1–8.9s each, and each prints something a reader
   learns (an A/E, a rate table, a refusal message, a prediction vector) —
   not just "done."
8. **Gates**: `black --check .` → clean (110 files). `ruff check .` → clean.
   `mypy src/easy_glm/core src/easy_glm/workflow --ignore-missing-imports` →
   clean (19 files). Full `pytest -q tests` → **787 passed, 1 skipped, 1
   deselected in 233.67s** (the deselected one is the `-m slow` 5M-row
   benchmark, correctly excluded by `addopts`; nothing failed).

## Blocking

**B1 — §8's printed reload-prediction vector does not match what the block
actually produces.**

README.md line 339, inside the "Exact scoring, `.easyglm`, and the one-line
invariant" section (the section that *is* the page's central trust claim):

```
print(reloaded.predict(holdout.head(3)))
# → [0.00153043 0.00649728 0.08165957]
```

Running exactly this block (in the pytest-driven gate, in three independent
repeat runs, and again in a from-scratch fresh-clone script) consistently
prints:

```
[0.00172092 0.00686527 0.08666724]
```

That's a ~12–13% relative difference on every one of the three values, not a
trailing-digit rounding difference — and it reproduces identically every
time I ran it, so it isn't run-to-run noise on my end either; the value in
the README simply appears to predate the current state of
`RateModel`/`tooling.cap_floor`/`tooling.smooth_moving_average`/
`update_relativity` and was never regenerated against the code that shipped.
Every other adjustment-affected number in the same section (the
`-4.04%` total-claims move, the `Rebalanced: True`, the invariant `True`) is
correct — it's specifically this one printed vector.

This is exactly the class of defect R11 exists to catch: the test only
asserts the block *runs*, not that its documented output is what it prints,
so a comment can silently go stale. Fix: regenerate this one line from an
actual run (ideally via the same `scripts/checks/readme_gate.py` machinery,
or add it to the invariant-checked set) before merge.

## Should-fix

**S1 — the two-stage base-rate line overstates precision it doesn't have.**
Line 250:
```
print("base rate:", base_rate(fit), base_rate(two_stage_fit))
# → 0.04340918722561671 0.04340918722561671  (identical to glum's own run-to-run noise)
```
The parenthetical correctly warns the two numbers aren't bit-identical, but
printing the *literal same digit string twice* undercuts that warning — a
reader who runs it and sees two visibly different 18-digit floats (I got
`0.043409187225616715` and `0.043409187225616694`) may read that as
something broken, which is the opposite of what the hedge is trying to
prevent. Suggest either showing the two real (differing) values with the
hedge, or collapsing to something like `# → 0.0434091872... (both fits, to
~1e-13 — see the note above)`.

**S2 — §17's "2.5e-16" is a specific-looking number for something that is
only "about 1e-16."** Line 713. My run gave `2.220446049250313e-16` — same
order of magnitude, consistent with the prose ("not approximately, to about
1e-16"), but the six-significant-figure "2.5e-16" invites exactly the kind
of literal comparison that doesn't hold up. Minor next to B1/S1, but the
same family of problem: round it to `~1e-16` or `on the order of 1e-16`
rather than a specific figure that machine-precision arithmetic won't
reproduce.

## Nice-to-have

- `elastic-net` (line 148, the design-yourself table) and `tabmat` (§10) are
  used without a gloss; a first-time-actuary reader will not know either
  term and both are skippable on a first read, but a three-word aside would
  help ("elastic-net (L1 + a bit of L2)").
- `fit_glm(..., aggregate=True)` and `fit_glm(..., progress=callable)` (both
  new in 0.4, both documented in `docs/checks/g-scale.md`) aren't mentioned
  on the front page at all. Not a gap I'd block on — the page is already
  long and links to the doc that covers them — but a one-line mention in
  §10 ("`progress=` for a status line on long fits; `aggregate=True` to
  collapse identical design rows") would close the loop for a reader who
  only ever reads the README.
- `penalty_weight` and `tweedie_power` (also 0.4, covered in
  `docs/checks/e-f-extras-cli.md`) are likewise absent from the page.
  Same call: fine to omit, could be one more table row in §4 if the owner
  wants completeness over length.

## Re-check list (round 2)

- [ ] B1 fixed: §8's `reloaded.predict(holdout.head(3))` comment matches an
      actual fresh run (verify by re-running the exact block, not just
      trusting the new number).
- [ ] Re-run `pytest -q tests/test_readme.py` and the full `pytest -q tests`
      once more after the fix, to confirm nothing else moved.
- [ ] Optional: S1/S2 addressed (hedge language, not a re-run requirement).
- [ ] Optional: nice-to-haves addressed or explicitly deferred to 0.4.1
      "Known limitations."
