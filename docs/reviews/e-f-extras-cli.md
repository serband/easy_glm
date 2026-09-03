## Round 2 verdict

**APPROVED.** Fix commit `7f0d91c` closes B1 and all four should-fixes (S1–S4).
Everything below was re-run independently (own probes, not the builder's
tests) against the worktree at `7f0d91c`; nothing in the diff touches
`core/`, `fit.py`, `design.py` or `engine/rate_model.py` / `_scoring.py`, so
round 1's numeric findings (V1, V3, V5, V7, V8) carry forward unchanged — and
were spot-checked again below to confirm.

* **B1 (was blocking) — fixed.** `pyproject.toml` gained
  `[tool.mypy]` with `follow_imports = "silent"` (and
  `ignore_missing_imports = true`). Re-ran the exact CI command:

  ```
  $ mypy src/easy_glm/core src/easy_glm/workflow --ignore-missing-imports
  Success: no issues found in 18 source files
  exit=0
  ```

  `.github/workflows/ci.yml` still invokes the same command verbatim (config
  now comes from `pyproject.toml`, which is what CI reads). `AGENTS.md`
  updated to say so. One residual nit: `CHANGELOG.md`'s "It found 30 problems;
  all are fixed rather than silenced" line was not touched, so it still
  doesn't say the followed layers (`engine`/`ui`) are exempted rather than
  clean — documentation-only, not a number/crash/data-loss issue, so it is a
  **known-limitation candidate**, not blocking.

* **S1 — fixed.** Own subprocess probes, not the builder's tests:
  `--out` at an existing file → `easy-glm: [Errno 17] File exists: '...'`,
  exit 1, exactly one stderr line, no `Traceback`. `--out` under a
  `chmod 500` parent → `easy-glm: [Errno 13] Permission denied: '...'`, exit
  1, no traceback. Both match `S1`'s promise; `main()`'s new
  `except OSError` clause is exercised.

* **S2 — fixed.** Own probes: a `.easyglm` scorer passed to `validate` →
  `easy-glm: <path> is a rate-table scorer (.easyglm), not a workbench
  project — pass the project.json this model was fitted from`, exit 2, no
  traceback. A directory → `... is a directory, not a project JSON file`,
  exit 2. A zip archive saved under a `.json` name → `... looks like a zip
  archive (e.g. an .xlsx export), not a project JSON file`, exit 2 — content
  sniffing (`PK` magic bytes), not extension, so the `.json` disguise doesn't
  fool it. All three name the actual mistake instead of the old data-glob
  error.

* **S3 — fixed.** `engine.models.relativity_note` is plain text now; a fresh
  rate-change run's Excel `Summary` sheet contains "multiplier on current
  premium" and "overall"/"differential" with **no** `**` anywhere in the
  sheet, and the HTML report likewise contains the sentence with no `**`.
  The two Streamlit pages (`pages_export.py`, `pages_tables.py`) route through
  the new `app.ui.relativity_note_markdown()`, so the bold styling still
  renders in the workbench — verified by reading the diff (not exercised
  live under Streamlit, but the function is pure string substitution and is
  covered by the app test suite that passed on both venvs).

* **S4 — fixed.** Own spy on `easy_glm.cli.prepare` (not the builder's test):
  `cli.main(["run", ...])` → exactly 1 call; `cli.main(["export", "--script",
  ...])` → exactly 1 call. `check()` now accepts a `columns=` list and `fit()`
  passes `list(df.columns)` instead of re-preparing.

* **Numbers unmoved (own probes).** A fresh rate-change model with a
  `DrivAge × Region` interaction (two-stage fit): `RateModel.predict` vs
  `fit.predict` max relative diff = **9.2e-16** (budget 1e-10, well inside).
  `solve_base_rate` on the same run/frame hit targets 0.65 / 1.0 / 1.25 with
  achieved A/E errors of **6.7e-16 / 2.2e-16 / 0.0** (budget 1e-10). Diffing
  `4ce322c...7f0d91c` confirms the fix commit touches no file under
  `core/`, `engine/rate_model.py`, or `engine/_scoring.py` — only `cli.py`,
  `engine/models.py` (the note string), `app/ui.py`,
  `app/pages_export.py`, `app/pages_tables.py`, `pyproject.toml`,
  `AGENTS.md`, and the test file.

* **Gates.** `black --check .` — all done, 98 files unchanged, exit 0.
  `ruff check .` — all checks passed, exit 0. Full `pytest -q tests`: **662
  passed, 1 skipped** on the main venv (204.9 s) and **662 passed, 1 skipped**
  again on the Streamlit-1.63 venv (223.8 s) — both a clean 8 more than round
  1's 654/229-subset, matching the 8 tests the fix commit adds (2 for S1, 3
  for S2, 1 for S3, 2 for S4). No new failures on either interpreter.
  `git diff release-0.4 -- tests/test_golden.py tests/fixtures` — empty.
  `scripts/checks/e_f_extras_cli.py --write` reproduces the check page with
  one difference: the binomial-scorer round-trip figure moved from
  **1.1e-16** to **5.6e-17** — a last-digit float drift in one probability
  figure, exactly the known glum noise flagged in the task brief, not a
  regression from this fix (nothing in the diff touches the binomial path).
  Reverted that regeneration (`git checkout -- docs/checks/e-f-extras-cli.md`)
  so the worktree is left clean; only this review file is modified.

No blocking items remain. Round 1's nice-to-haves (N1–N6) are unaddressed and
unaffected by this fix; they remain **known-limitation candidates**, not
blockers, under this round's stricter bar (numbers/lost-work/crashes only).

---

# Independent review — piece E+F (rate-change models, modelling extras, CLI, mypy)

*Reviewer: independent (did not write the code). Branch `piece/e-f` @ `4ce322c`
(merge of `release-0.4` @ `20f6ef4`, which carries A2), diff
`git diff release-0.4...HEAD`. Worktree
`/private/tmp/claude-501/-Users-serban-Sabre-Onceoff/26b6a938-4cfc-4149-bb2f-e039456cde3c/scratchpad/wt-ef`,
interpreter `/Users/serban/Documents/Projects/easy_glm/.venv/bin/python` with
`PYTHONPATH=<worktree>/src` (verified `easy_glm.__file__` resolves inside the
worktree), Streamlit-1.63 venv for the workbench pages. Everything below was run,
not read off.*

---

## Verdict

**Accept with one blocking fix and four should-fixes.** The modelling work is
right and the numbers hold up under every test I could think of. Rate-change
exactness is 1.3e-15 (budget 1e-10) for step, categorical, linear, null, unseen
level and two-stage-with-an-interaction models; the offset identity claim is
exactly true for Poisson and provably false for Gamma, and the doc says so
correctly; per-variable penalty weights multiply the right columns and *nothing*
else, with the interaction P1 vector **bit-identical** to release-0.4;
`solve_base_rate` hits its target to machine precision on ordinary and two-stage
fits and never moves a relativity; the binomial path is exact, labelled, and
refuses exposure; the merge reverts nothing from A2; the check page regenerates
byte-for-byte. 654 tests pass in the main venv, and the app tests also pass under
Streamlit 1.63.

The one blocking item is not a number: **the CI mypy step this piece adds fails**
(exit 1, 20 errors) — zero of them inside `core/` or `workflow/`, all in modules
mypy follows through imports. The piece's own acceptance gate is "ships behind
green CI", and this step is red for every future PR until it is fixed. It is a
one-flag change and I verified the fix.

Two of the should-fixes are broken promises rather than broken code (the CLI
prints tracebacks it says it never prints; markdown bold leaks into Excel and the
HTML report), one is the break-it case the brief asked about (`.easyglm` handed
to the CLI), and one is a doubled data load in the headless path that matters at
workstream G's row counts.

**Nothing in E+F produces a wrong number, loses data, crashes the workbench, or
silently misreads a file.**

---

## Blocking

### B1. The CI `mypy` step this piece adds is red

`.github/workflows/ci.yml` gains

```yaml
- name: Type check (mypy) — the layers the exported script and the CLI rely on
  run: mypy src/easy_glm/core src/easy_glm/workflow --ignore-missing-imports
```

There is no `[tool.mypy]` section in `pyproject.toml`, so that is the whole
configuration. mypy follows imports by default and reports errors in every
module it can find source for — including `engine/` and `ui/`, which are not on
the command line.

```console
$ cd <worktree>
$ .venv/bin/python -m mypy src/easy_glm/core src/easy_glm/workflow --ignore-missing-imports >/dev/null 2>&1; echo "exit=$?"
exit=1

$ .venv/bin/python -m mypy src/easy_glm/core src/easy_glm/workflow --ignore-missing-imports 2>&1 | tail -1
Found 20 errors in 3 files (checked 18 source files)

$ ... | awk -F: '{print $1}' | sort | uniq -c
   3 src/easy_glm/engine/_scoring.py
  12 src/easy_glm/engine/rate_model.py
   9 src/easy_glm/ui/metrics.py

$ ... | grep -c "^src/easy_glm/\(core\|workflow\)/"
0
```

The builder's claim "mypy clean on core+workflow" is *literally* true — zero
errors live in those two directories, and the piece took the followed-import
count from 52 to 20 (release-0.4, same command: `Found 52 errors in 14 files`).
The CI command simply does not express that claim. mypy in this venv is 2.3.1;
`pyproject.toml` pins `mypy>=1.8.0`, so CI installs the current major and sees
the same thing. Several of the errors are version-independent core checks (e.g.
`rate_model.py:594 Incompatible default for parameter "exposure_col" (default
has type "object", ...)` — the `_UNSET = object()` sentinel).

**Fix, verified:**

```console
$ .venv/bin/python -m mypy src/easy_glm/core src/easy_glm/workflow \
      --ignore-missing-imports --follow-imports=silent; echo "exit=$?"
Success: no issues found in 18 source files
exit=0
```

Either add `--follow-imports=silent` to the CI step, or add a `[tool.mypy]`
section with per-module overrides for `easy_glm.engine.*` / `easy_glm.ui.*`, or
fix the 20. Whichever — the step has to be green, and the CHANGELOG's "It found
30 problems; all are fixed rather than silenced" should say what the followed
layers still owe.

---

## Should-fix

### S1. The CLI prints raw Python tracebacks for filesystem errors on `--out`

`cli.py`'s module docstring, the CHANGELOG ("Nothing ever prints a traceback")
and `docs/checks/e-f-extras-cli.md` ("Anything wrong is a message and a non-zero
exit code — never a stack trace — so a scheduled job can tell success from
failure") all make the same promise. `prefix()` calls `out.mkdir(parents=True,
exist_ok=True)` and the writers call `path.write_text(...)` outside any handler,
so any `OSError` escapes `main`.

```console
$ touch afile
$ easy-glm run proj/rate-review.json --out afile
  File ".../pathlib/__init__.py", line 1011, in mkdir
    os.mkdir(self, mode)
    ~~~~~~~~^^^^^^^^^^^^
FileExistsError: [Errno 17] File exists: 'afile'
$ echo $?
1

$ mkdir -p ro2 && chmod 500 ro2
$ easy-glm export proj/rate-review.json --script --out ro2/sub
    os.mkdir(self, mode)
    ~~~~~~~~^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: 'ro2/sub'
$ echo $?
1

# an output file that already exists and is read-only: same shape, exit 1
```

The exit code happens to be 1 because that is Python's default for an uncaught
exception, not because the CLI decided anything. Fix: one more clause in
`main()` —

```python
except OSError as exc:
    print(f"easy-glm: {exc}", file=sys.stderr)
    return 1
```

(`CliError` already renders exactly like that.) Worth a test alongside
`TestCliRun::test_the_output_folder_is_created`.

### S2. A `.easyglm` handed to the CLI is accepted as a project and fails obscurely

This is the break-it case in the brief. `Project.from_json` tolerates unknown
keys, so a scorer file loads as an empty project with an empty
`data.source.path`, and `prepare` then globs the *current working directory*:

```console
$ easy-glm validate artefacts/rate-review_change.easyglm
.../cli.py:68: UserWarning: Ignoring unknown project keys ['base_rate',
 'column_mapping', 'current_version', 'format_version', 'metadata',
 'snapshots', 'variables'] in project (written by a newer easy_glm?) ...
easy-glm: artefacts/rate-review_change.easyglm has 1 problem(s)
  - the data cannot be prepared: directory contained paths with different file
    extensions: first path: ./shots.py, second path: ./s2_density.png. Please
    use a glob pattern ...
$ echo $?
1

$ easy-glm run artefacts/rate-review_change.easyglm --out /tmp/zz
easy-glm: the project has no models
```

Exit code and "no stack trace" are fine; the *message* points at a directory
glob in the user's cwd and never at the real mistake. Two lines in
`open_project` fix it: if the parsed JSON has `format_version` / `variables` /
`base_rate` and no `models`, raise
`CliError(f"{p} is a rate-table scorer (.easyglm), not a workbench project")`.

Related, and worth a glance even though it predates this piece: an empty
`data.source.path` makes `prepare` scan the working directory. The CLI is the
first surface where that is reachable without a browser.

### S3. Markdown bold leaks literally into the Excel `Summary` sheet and the HTML report

`engine/models.py::relativity_note` is written for Streamlit's markdown, but it
is also `_esc`-ed into a `<p>` by `workflow/report.py::_summary_section` and
written raw into an Excel cell. Both are what the actuary actually receives, and
both show the asterisks:

```console
$ unzip -p artefacts/..._rate_tables.xlsx xl/sharedStrings.xml | ...
'Multipliers on the current premium: this model was fitted with
 log_CurrentPremium ... as an offset, so the base rate is the **overall** rate
 change and each relativity is the **differential** change for that band. ...'

$ grep -o 'Multipliers on the current premium[^<]*' artefacts/..._report.html
... the base rate is the **overall** rate change and each relativity is the
**differential** change for that band. ...
```

Since this string exists precisely to answer Q6 in the actuary's own
deliverables, the asterisks undercut it. Drop them (Streamlit renders the plain
sentence perfectly well), or keep a `_md` and a `_plain` variant.

### S4. `easy-glm run` / `export` prepare the data twice

`cmd_run` calls `prepared_frame(project)`, then `fit(project, model, df)` calls
`check(project, model)`, which calls `prepare(project)` a second time to get the
column list. Every data step — read, rename, recode, derive, filter, the premium
log — runs twice on every artefact command. Invisible on the 50k fixture (1.2 s
total), but the CLI is the scheduled-refit surface and workstream G targets 5M
rows in memory, where this doubles the read and the peak.

`check` already accepts `columns=`; passing `list(df.columns)` from the caller
(and keeping the load-and-report path only for `cmd_validate`, which has no
frame yet) removes it.

---

## Nice-to-have

* **N1.** The Excel `Summary` line reads `current multiplier on current premium
  per band (manual adjustments included)` — "current … current premium" is a
  stumble. `rate_model.py` builds it as `f"current {self.relativity_label} per
  band"`; a per-label phrasing would read better.
* **N2.** `solve_base_rate` on a frame with nulls in the target reports *"The
  total of the model's target (Loss) on these rows is not a positive number"*.
  True of NaN, but it points the user at the level rather than at the nulls.
  `np.nansum` is wrong here; naming the null count would be right.
* **N3.** There is no `easy-glm score` (score new business from a `.easyglm` plus
  a data file). Plan §F does not ask for one — it asks for `run` and `export`,
  both delivered — but the review brief expected it and R11 promises "reload the
  scorer and score new business" only as a README Python example. Worth a line in
  the handover so it is a decision, not an omission.
* **N4.** Other §F items are not in this piece: the CI matrix is still
  3.10–3.13 (§F asks for 3.14), there is no mkdocs site, and the version is not
  yet 0.4.0. Presumably scheduled for the release gate; confirm.
* **N5.** In the offset-identity section, the Poisson leg fits `ClaimNb` on the
  whole book with `gradient_tol=1e-12, max_iter=100_000`; the Gamma leg fits
  `Loss` on claim-bearing rows at default tolerances. "The same pair on a **Gamma**
  target differ by 0.281" is true of the offset/weighted pair, but the reader may
  take "the same pair" to mean the same data too. One clause would fix it. (The
  *conclusion* is correct — see V2 below; the Gamma difference is structural, not
  a convergence artefact.)
* **N6.** The Model page's box is labelled "Target loss ratio" even for an
  ordinary frequency model, where the number is an A/E target rather than a loss
  ratio. The help text explains it; a conditional label would be kinder.

---

## What I verified (evidence)

### V1. Rate-change exactness — 1.3e-15, budget 1e-10

`RateModel.predict` vs `fit.predict`, 50k rows, premium offset, scored both on
the training frame and on a frame with 50 null `DrivAge` and 50 unseen `Region`
levels injected:

```
step+cat, premium offset (train)       max rel diff = 1.348e-15   n=50000  finite=True
step+cat, nulls+unseen (score)         max rel diff = 1.348e-15   n=50000  finite=True
  offset_is_premium = True | label: multiplier on current premium
with linear term (train)               max rel diff = 1.363e-15
with linear term (score)               max rel diff = 1.363e-15
two-stage interaction (train)          max rel diff = 1.282e-15
two-stage interaction (score)          max rel diff = 1.282e-15
mains frozen (max abs table diff)      = 8.882e-16
base rate no-int 0.6683047898854604  two-stage 0.6683047898854603
```

The mains-frozen line is the A2 × E1 composition invariant: the five main tables
are the same numbers with and without the `DrivAge × Region` interaction, and the
base rate agrees to the last bit.

### V2. The offset identity — exactly true for Poisson, structurally false for Gamma

The doc's claim is right and the explanation is right. Rather than trust two
fits, I compared the two *objectives* at the same coefficient vector: if the
offset form and the premium-weighted form differ by a constant, they are the same
model.

```
poisson  offset-obj minus weighted-obj at 3 betas: [-0.0, -0.0, 0.0]         -> constant => same model? True
gamma    offset-obj minus weighted-obj at 3 betas: [-1433411.12, -1106147.93, -1202769.42] -> constant => same model? False
```

So "the Poisson deviance — the only one invariant under the swap" is correct, and
"they are genuinely different models, and that is expected" is the right thing to
say about Gamma. (I could not re-run the Gamma pair at `gradient_tol=1e-12` to
rule convergence in or out — glum hits line-search failures and does not finish
in two minutes — but the objective test above settles it without fitting.) The
5.6e-12 Poisson figure and every other number on the page regenerate exactly; see
V7.

### V3. Penalty weights — a main's weight touches only its own columns; the interaction P1 is bit-identical to release-0.4

P1 with `scale_predictors=True`, per-variable min/max:

```
no weights, no linear/inter -> None                     (glum default, unchanged)
with Region=0:   DrivAge (1,1) VehAge (1,1) BonusMalus (1,1) Region (0,0) VehGas (1,1)
with Region=2.5: DrivAge (1,1) VehAge (1,1) BonusMalus (1,1) Region (2.5,2.5) VehGas (1,1)
```

With a linear term and an interaction present, ratio of P1 with
`{BonusMalus: 4.0, DrivAge: 0.0}` to P1 without:

```
  ratio DrivAge                -> [0.]
  ratio VehAge                 -> [1.]
  ratio BonusMalus             -> [4.]      (multiplies the per-band rule, does not replace it)
  ratio Region                 -> [1.]
  ratio VehGas                 -> [1.]
  ratio DrivAge×Region         -> [1.]      (cells untouched by a parent's main weight)
```

Same fit run under this branch and under release-0.4 extracted with `git archive`
(`penalty_weight` unset, interaction weight 2.0, two-stage, Poisson, offset):

```
stage1_coef    identical=False  maxdiff=8.049e-16
stage2_coef    identical=False  maxdiff=1.818e-15
p1_stage2      identical=True   maxdiff=0.000e+00      <-- A2's number, bit for bit
p1_stage1 both None
intercept identical: True
```

The coefficients are not bit-identical, but neither are two fits of the same
model in the same process — I measured `8.049e-16` across three consecutive
`run_model` calls, which is the builder's documented 9e-16 and the justification
for the 1e-12 comparison in `test_the_easyglm_file_is_the_workbench_model`. The
P1 vector, which is the thing the refactor could have changed, is exact.

Monotone survives an unpenalised factor (`BonusMalus`, `penalty_weight=0.0`,
`monotone="increasing"`):

```
BonusMalus rels: [1.0, 1.5298, 2.0931, 2.3029, 5.0093, 1.0]  non-decreasing: True
Region levels kept non-1: 21 of 22        (unpenalised: nothing thinned out)
```

`penalty_weight` reaches `DesignSpec.from_data` and both export paths — the
fitted script writes `StepEncoder(..., penalty_weight=0.0)` /
`CategoricalEncoder(..., penalty_weight=2.5)`, the unfitted one writes
`penalty_weight={'BonusMalus': 0.0, 'Region': 0.0, 'VehGas': 2.5}`; both execute
(exit 0).

### V4. Tweedie power reaches both stages; the binomial path is exact and labelled

```
stage1 glum family obj: TweedieDistribution  power: 1.7
stage2 glum family obj: TweedieDistribution  power: 1.7
```

and the exported script carries `tweedie_power=1.7` in *both* `fit_glm` calls.

Binomial (synthetic lapse, logit), end to end through `run_model`:

```
link: logit | label: odds relativity
base rate (odds): 0.4566495115555206
exact max rel diff: 3.740843194439391e-16
pred range: 0.1961 .. 0.4444   all in (0,1): True
nulls+unseen exact: 3.740843194439391e-16   finite: True
exposure refused: This model predicts a probability (logit link), which cannot be multiplied by an ...
exposure_for: binomial -> None
```

Excel `Summary`: `'current odds relativity per band ...'` and `'Odds
relativities: the tables multiply the odds, not the probability. ...'`. So Q7's
label is on every surface. A non-0/1 target is refused by glum with `Binomial
target must lie in [0, 1].` for both a continuous `Loss` target and a `ClaimNb`
count.

### V5. `solve_base_rate` — machine-precision, idempotent, relativities frozen

Definition confirmed as **total actual ÷ total expected = target**, which for a
rate-change model *is* the loss ratio (actual = loss, expected = indicated
premium). The brief's inverted version would have been premium ÷ loss. The
builder's deviation is correct, and it is stated in the docstring and on the
check page.

```
ordinary   target=0.65  base=0.607130 achieved A/E=0.650000000000 err=0.00e+00 rel-moved=0.0e+00 resolve-diff=0.0e+00
ordinary   target=1.0   base=0.394635 achieved A/E=1.000000000000 err=0.00e+00 rel-moved=0.0e+00 resolve-diff=0.0e+00
ordinary   target=1.25  base=0.315708 achieved A/E=1.250000000000 err=0.00e+00 rel-moved=0.0e+00 resolve-diff=0.0e+00
two-stage  target=0.65  base=0.604696 achieved A/E=0.650000000000 err=1.11e-16 rel-moved=0.0e+00 resolve-diff=3.7e-16
two-stage  target=1.0   base=0.393052 achieved A/E=1.000000000000 err=2.22e-16 rel-moved=0.0e+00 resolve-diff=1.4e-16
two-stage  target=1.25  base=0.314442 achieved A/E=1.250000000000 err=0.00e+00 rel-moved=0.0e+00 resolve-diff=0.0e+00
```

`rel-moved` is the largest change in any relativity of any table: zero. Target
1.0 gives overall A/E exactly 1. Re-solving from the answer returns the same
number.

Degenerate inputs:

```
empty frame          -> ValueError: No rows to balance the base rate on
all-zero actual      -> ValueError: The total of the model's target (Loss) ... is not a positive number
negative / zero / nan target -> ValueError: target_ratio must be a positive number
against= / weight= missing column -> KeyError naming the column
binomial             -> ValueError: solve_base_rate needs a multiplicative (log-link) model ...
```

(Nulls in the target land in the "not a positive number" branch — see N2.)

**UI path.** `pages_model._target_loss_ratio` calls the same
`workflow.solve_base_rate` on the training rows. The risky part is the widget-key
dance around `bro_{name}`, so I drove it under Streamlit 1.63 with `AppTest`,
solving twice at the same target and then at a different one:

```
solve1 override=0.02044796470254865 box=0.02044796470254865
solve2 override=0.02044796470254865 box=0.02044796470254865  idempotent=0.00e+00
solve3(1.0) override=0.012677738115580163 box=0.012677738115580163
```

The visible box always agrees with the stored override, and no exception is
raised on the repeat.

### V6. The CLI

Exit codes are exactly as documented — 0 success, 1 actionable, 2 usage:

```
exit=1 : validate nosuch.json           easy-glm: no project file at nosuch.json
exit=1 : validate broken.txt            ... is not a readable easy_glm project: Expecting property name ...
exit=1 : validate bad.json              3 problem(s): predictor not in the data / tweedie_power must be strictly between 1 and 2 / not predictor-role columns
exit=1 : validate notarget.json         change: no target column
exit=1 : run notarget.json              easy-glm: model 'change' cannot be fitted \n  - change: no target column
exit=1 : validate nodata.json           the data cannot be prepared: /nowhere/x.parquet
exit=1 : run nomodels.json              the project has no models
exit=1 : export proj/... (no flags)     pass at least one of --script, --report, --excel
exit=2 : validate ... --nope            argparse usage error
exit=2 : badcmd x                       argparse usage error
exit=0 : validate proj/rate-review.json rate-review.json: valid · models: change
exit=0 : run proj/rate-review.json
```

`--help` and per-subcommand `--help` are correct. An output path far outside cwd
is created and written (`--out .../deep/a/b/c` → exit 0, file present).

**No Streamlit at import.** With an import hook that raises on `streamlit`,
`import easy_glm.cli` succeeds, `'streamlit' in sys.modules` is `False`, and
`main(['validate', ...])` returns 0. The only Streamlit import is inside
`cmd_workbench`, behind an `ImportError` handler that names `pip install
'easy_glm[ui]'`.

**Round trip.** `easy-glm run` on a rate-change project wrote all four artefacts;
the emitted script executed standalone and its `.easyglm` matched the CLI's to
**1.03e-15**. Repeated for the harder project (premium offset + `DrivAge ×
Region` interaction + three penalty weights + `tweedie_power=1.7` + a monotone
constraint): **1.69e-15**.

`easy-glm score` does not exist; §F does not ask for it (see N3).

### V7. Persistence, run keys, and old files

```
PERSIST_FORMAT = 6
format-6 key: e526f9ebfccc1cf0   -> 7550b672e1-e526f9ebfccc1cf0.pkl
format-5 key: c4dcc3aafef24d38   -> 7550b672e1-c4dcc3aafef24d38.pkl
different: True
```

`PERSIST_FORMAT` is part of the run key, i.e. part of the *filename*, so a
format-5 pickle is never opened — ignored, not misread.

Run key sensitivity:

```
tweedie 1.5 -> 1.6            key changes: True
Region penalty_weight 1 -> 0  key changes: True
current_premium role removed  key changes: True
restored                      key back to base: True
```

A project JSON written by release-0.4 (no `penalty_weight`, no `tweedie_power`)
loads, defaults to 1.0 / 1.5, validates clean and fits. `.easyglm`
`FORMAT_VERSION` stays 2; `_metadata_from_dict` already tolerates unknown keys
with a warning, so an older reader will not choke on `offset_is_premium`.

`offset_is_premium` really is labels-only:

```
offset_is_premium changes predictions? False | base rates equal: True
labels: relativity | multiplier on current premium
no offset col -> offset_is_premium: False | label: relativity     (guarded)
round-trip flag: True
```

### V8. Merge sanity

`git diff release-0.4...HEAD -- src/` deletes no A2 code. `fit_two_stage` itself
is untouched (the only `fit.py` hunk mentioning it is a docstring line);
`to_script`'s two-stage branch gains lines and loses none of its own; the bundle
`"version": 3` in `core/easyglm.py` is intact (that file's only changes are
`blueprint` handling a non-categorical encoder and an `isinstance` narrowing).
Every deleted line I inspected is either replaced in place or is a docstring.

The check page regenerates exactly:

```console
$ python scripts/checks/e_f_extras_cli.py > regen.md
$ diff docs/checks/e-f-extras-cli.md regen.md
348a349
>
```

— one trailing newline from the shell redirect. Every number on the page,
including the 5.6e-12 identity, the 8-of-20 / 20-of-20 penalty demo, the 0.6809
base rate and the CLI transcript, reproduces.

### V9. Break-it

| attempt | result |
|---|---|
| premium column with zeros / negatives / nulls | `prepare` refuses the frame naming the count (`has 7 row(s) that are not a positive number …`) and the filter to add |
| the same, plus `pl.col('CurrentPremium') > 0` | prepares; the offset is derived **after** the filter, so the filter is the fix and not a trap |
| `current_premium` role on the target column | duplicate single-role caught by `validate`; target becomes `None` |
| take the premium role away after a fit | offset cleared with the notice *"Model change no longer offsets on 'log_CurrentPremium'…"*; the refit then works (base rate 93.9, no offset) |
| rename the premium column | offset follows: `log_Prem2024`, role preserved |
| `tweedie_power` = 1.0 / 2.0 / 0.5 / 2.5 / NaN | `ValueError` naming the open interval |
| `tweedie_power` on a non-Tweedie family | `ValueError: tweedie_power is only meaningful for the tweedie family` |
| `penalty_weight` = 0 / −1 / NaN / inf | 0 accepted (unpenalised); the rest `ValueError` naming the encoder and variable |
| binomial with a continuous or count target | refused (`Binomial target must lie in [0, 1].`) |
| binomial + exposure | refused in `to_rate_model`, in `RateModel.predict`, and never offered by `exposure_for` |
| a `.easyglm` given to the CLI | non-zero exit, but a confusing message — **S2** |
| `--out` at an existing file / read-only parent | traceback — **S1** |
| a project whose filter leaves no training rows | `easy-glm: cannot prepare the data: No row of 'traintest' equals the TRAIN value 1; check the value on the Split page` |

### V10. Tests

```
654 passed, 1 skipped, 17 warnings in 214.11s        (main venv, PYTHONPATH=<worktree>/src)
229 passed in 41.37s                                 (Streamlit 1.63: test_e_f_extras_cli, test_app,
                                                      test_w2_pages, test_w3_hardening, test_app_state, test_ui)
```

`tests/test_e_f_extras_cli.py` collects **83** tests; the
`TestRateChangeWithAnInteraction` block holds the **9** composition tests. Both
match the builder's report.

---

## Re-check

Round 2 should show me:

1. **B1** — `mypy src/easy_glm/core src/easy_glm/workflow --ignore-missing-imports
   [+ whatever you add]` exiting 0 in a paste, and the CHANGELOG line adjusted if
   the followed layers stay unchecked. I will re-run the CI step verbatim.
2. **S1** — `easy-glm run <project> --out <an existing file>` and `--out
   <read-only>/sub` each printing one `easy-glm: …` line, exit 1, no traceback,
   with a test.
3. **S2** — `easy-glm validate <a .easyglm>` naming the actual mistake.
4. **S3** — no `**` in `xl/sharedStrings.xml` of a rate-change workbook or in the
   HTML report, for both the premium and the odds note.
5. **S4** — `prepare` called once per `run` / `export` (a counter or the changed
   call site is enough).

I will re-run V1, V3 (the release-0.4 P1 comparison), V5, V6 and V8's check-page
regeneration to confirm nothing moved, and the full suite plus the Streamlit-1.63
app tests. Nothing else needs to be re-derived; the rest of the piece is sound.
