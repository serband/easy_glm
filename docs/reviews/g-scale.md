# Review: piece G — scale (round 1)

## Verdict

**Approve with two should-fix items.** No blocking findings. Every correctness,
memory and exactness claim I independently re-measured held (matrix ops to
1e-12, fits to 1e-10/1e-12, aggregation to 1e-12, memory within the stated
budgets, `docs/checks/g-scale.md` reproduces from the committed script). The
two should-fix items are a real (but narrow-blast-radius) crash on 0-row sparse
builds and a docstring that promises a safety check the code does not have.
Neither changes a number, loses work, or is reachable from the workbench today.

## What I verified

Interpreter: `/Users/serban/Documents/Projects/easy_glm/.venv/bin/python`,
`PYTHONPATH=<worktree>/src` on every command; confirmed `easy_glm.__file__`
resolved under the worktree before and after each run. Machine: 24 GB RAM;
free memory checked before every run above 1M rows (8.3–9.0 GB free
throughout, well under the 60 %/15.4 GB abort threshold; 5M-row run peaked at
2.56 GB, nowhere near it).

1. **Matrix correctness** (independent script, not the repo's own tests):
   `StepMatrix.matvec`, `transpose_matvec`, `sandwich`, `_cross_sandwich`
   (against `CategoricalMatrix` and `DenseMatrix`), `getcol`,
   `standardize`/`unstandardize`, and `rows`/`cols` subsets — including
   **duplicated and out-of-order row indices** and **unsorted column
   indices** — all matched a from-scratch dense numpy reimplementation to
   1e-8/1e-12. Edge cases: one-bin step term, all-reference-level
   categorical, zero-kept-cell interaction, no-step-terms design, 0-row
   `StepMatrix` — all correct. One edge case failed: see should-fix #2.
2. **Fit equivalence beyond the builder's cases**: monotone bounds, Gamma and
   Tweedie(1.7) with weights, CV `n_alphas` choosing matching alphas
   (6.039995e-4 dense vs 6.039995e-4 compact), a 350k-row synthetic book
   through `workflow.run_model` (245,008 training rows after a 70/30 split —
   confirmed the compact path is picked by default, predictions agree to
   1e-10, identical non-zero set), and stage 2 of a two-stage fit compact vs
   dense (coefficients agree to 1e-10). All passed.
3. **Scoring**: monkeypatched `DesignSpec.build`/`build_dense`/`build_sparse`
   to raise — both `GLMFit.predict` and `TwoStageFit.predict` scored without
   calling any of them. Chunk boundaries `n = chunk±1`, `n < chunk`, and
   `chunk_rows` from 1 to 10M all reproduced the unchunked result exactly.
   0-row `predict` returns a 0-length array. A wrong-length offset raises a
   clear numpy broadcast `ValueError`. Unseen categorical level + null
   numeric in the same scoring frame produced finite predictions.
4. **Memory**: ran `scripts/bench_scale.py` myself at 200k, 1M and 5M rows
   (sparse), plus 200k/1M dense for comparison:

   | rows | rep | design bytes | formula | peak RSS |
   |---:|---|---:|---:|---:|
   | 200,000 | sparse | 27 MB | 27 MB | 0.37 GB |
   | 200,000 | dense | 346 MB | 346 MB | 0.65 GB |
   | 1,000,000 | sparse | 134 MB | 134 MB | 0.83 GB |
   | 1,000,000 | dense | 1,732 MB | 1,732 MB | 2.25 GB |
   | 5,000,000 | sparse | 668 MB | 668 MB | **2.56 GB** |

   Design bytes matched the formula exactly at every size (script's own
   `--check-budget` also passed at 3 GB for 5M). These numbers are within a
   few percent of the committed `docs/checks/g-scale.md` (0.37/0.86/2.59 GB) —
   the small differences are ordinary machine-to-machine noise, not drift.
   Separately measured `workflow.run_model` + `predict` on a fresh 1M-row
   synthetic book end to end: **peak RSS 0.68 GB**, confirming no hidden dense
   copy in prep or scoring.
5. **Shim and pin**: `install_glum_shim()` wraps
   `glum._validation.check_array_tabmat_compliant` (and the same name in
   `glum._glm`) with one `isinstance(mat, StepMatrix)` branch; confirmed the
   wrapped signature `(mat, drop_first=False, **kwargs)` matches glum 3.4.1's
   real signature exactly (read via `inspect`). Confirmed it patches at
   **import/call time, globally, in-process** — verified this has no effect on
   an unrelated plain-numpy glum fit run in the same process before and after
   installing the shim (identical coefficients). Confirmed it fails loudly
   (`RuntimeError`, not a silent densification) when I deleted
   `check_array_tabmat_compliant` from the module to simulate glum removing
   it. `glum>=3.4.0,<3.5` is in `pyproject.toml`; the shim, the pin and the
   canary test are documented in `CHANGELOG.md`.
6. **Aggregation**: independently re-verified Poisson, Gamma and Tweedie(1.6),
   **all three with weights and an offset together** (the builder's own tests
   only combine weights+offset for Poisson) — coefficients and intercept
   matched the row-level fit to 1e-12, predictions matched exactly, for every
   family. All-unique-row data produced a harmless near-no-op (a handful of
   floating-point coincidental collisions out of 50k rows, not a bug).
7. **Progress**: read `app/state.py`'s `fit_progress_callback` and
   `core/fit.py`'s `_ElapsedProgress`. Directly instrumented
   `_ElapsedProgress` to record which thread called back and when relative to
   `__exit__`: the background thread's last tick lands **before** the stop
   event is processed, the thread is fully joined (`is_alive() == False`)
   before `__exit__` returns, and the one call that happens exactly at exit
   time runs on the *calling* thread, not the background one — so no stray
   callback can land after `fit_glm` has returned control. A failing callback
   was already covered by the repo's own test and still passed. The existing
   `AppTest`-based page tests (which fit a real model through the Model page)
   passed on **both** Streamlit 1.57 and 1.63 venvs (622 and 264 tests
   respectively — see Gates below), which exercises this code path, though I
   did not do a live-browser (Playwright) check of a genuinely long fit's
   progress line — see Re-check.
8. **Persistence**: built a `TwoStageFit` under the actual `release-0.4`
   source (extracted via `git archive release-0.4 -- src`, not by touching the
   main checkout or another worktree), pickled it, then unpickled that file
   under piece/g's source: it deserialized as `easy_glm.core.fit.TwoStageFit`
   and `.predict()` on the French-motor fixture agreed with the original,
   pre-pickle predictions to **2.8e-16 max absolute / 1.0e-15 max relative**
   — comfortably inside 1e-10. `PERSIST_FORMAT` is unchanged at 5, with a
   comment history that stops at A2 (5) and does not mention piece G — correct,
   since nothing about a persisted object's *shape* changed. `sparse` and
   `aggregate` are **not** in `run_key`/`model_hash`, and correctly so: they
   are not exposed anywhere in `ModelConfig`/`workflow.run_model` at all
   (confirmed by grep — the workbench never sets either), they are decided
   automatically by row count, and the exactness invariant guarantees they
   cannot change any number even if they were forced.
9. **Docs**: regenerated `docs/checks/g-scale.md` myself — ran
   `scripts/checks/g_scale.py` (without `--write`, feeding it my own
   `bench_scale.py` JSON via `--results` so the committed file was never
   touched) and diffed the output against the committed file. Every line
   matched **byte-for-byte except the machine-dependent numbers** (fit
   seconds, peak RSS, µs/row) — in particular the "Is it the same model?"
   table's non-zero counts (108/108), CV alpha (0.00074476 both), and the
   `9e-15`/`7e-15`-scale exactness numbers reproduced from the same
   deterministic fixture. This confirms the doc is genuinely generated, not
   hand-edited to look right. README's performance paragraph
   (0.4/0.8/2.6 GB at 200k/1M/5M) matches what I measured. The linear-band
   cost (`8 * n_rows` bytes/band, ~0.8 GB for a 20-band term at 5M) is stated
   in both `build_sparse`'s docstring and the check doc.
10. **Gates**: `black --check .` — clean (99 files). `ruff check .` — clean.
    `pytest -q tests` on the repo venv (Streamlit 1.57) — **622 passed, 1
    skipped, 1 deselected** in 178s, matching the builder's claimed count
    exactly. `pytest -q -m slow tests/test_scale.py` — the 5M-row budget test
    passed on its own (29.5s). App/workbench tests
    (`test_app_state.py test_app.py test_d3_d4_compare_report.py
    test_w2_pages.py test_w3_hardening.py test_w4_runs_folder.py
    test_scale.py`) on the Streamlit 1.63 venv — **264 passed, 1 skipped, 1
    deselected**. `git diff release-0.4...HEAD -- tests/test_golden.py
    tests/fixtures docs/RELEASE_0.4_PLAN.md` — empty, as required.

## Findings

### Should-fix

**S1. `DesignSpec.build_sparse` crashes with a confusing third-party error on
0-row data with any categorical or interaction term.**

```
>>> spec.build(book.head(0), sparse=True)
  File ".../tabmat/categorical_matrix.py", line 377, in __init__
    if max(indices) >= len(categories):
ValueError: max() iterable argument is empty
```

Reproduced with a plain categorical (`DesignSpec.from_data(book, ["DrivAge",
"Region"], ...)`), not just an interaction. `build_dense` on the same 0-row
frame works fine (returns a `(0, 24)` array), and a `StepMatrix`-only sparse
design (no categorical/interaction) also works fine on 0 rows — the bug is
specific to `tabmat.CategoricalMatrix`'s constructor choking on an empty
`indices` array, which `easy_glm.core.design.build_sparse` does not guard
against.

*Blast radius*: not reachable through `fit_glm` (which raises a clear "No
training rows." before ever calling `build`) or through
`predict`/`linear_predictor` (which never build a matrix at all — verified in
check 3 above). I grepped `src/easy_glm/app/` and `src/easy_glm/workflow/` for
any direct call to `.build(`, `.build_sparse(`, `.build_dense(` or
`.design_matrix(` outside `core/fit.py` and found none, so today's workbench
cannot hit this. It is reachable by any direct caller of the public
`DesignSpec.build(..., sparse=True)` / `GLMFit.design_matrix(..., sparse=True)`
API on an empty (e.g., fully-filtered) frame — plausible for a future
diagnostics feature or a user script, and squarely inside the "0-row data"
edge case this piece was asked to cover.

*Fix*: special-case `n == 0` in `build_sparse` (skip straight to an empty
`SplitMatrix`/return early) or wrap the `CategoricalMatrix` construction to
raise an easy_glm-level message. Either way, this should never surface a raw
tabmat traceback, per the project's own rule for user-facing errors.

**S2. `fit_glm`'s `aggregate` docstring claims a refusal that does not
exist.**

`core/fit.py`, the `aggregate` parameter docstring:

> Off by default, and refused with `cv=` (folds must be assigned to rows, not
> groups) and **when the fit has a piecewise-linear term with many distinct
> values**.

Only the `cv=` refusal is implemented (`fit_glm` raises `ValueError` when
`aggregate and cv is not None`). I grepped `fit.py` for any check tied to
`LinearEncoder`/distinct-value counts under the `aggregate` path and found
none — `aggregate_rows` happily includes the clamped linear value in the
grouping key (as documented elsewhere, correctly) and simply gets little or no
compression when that value is nearly always distinct. Nothing crashes or
gives a wrong number; the docstring is just wrong, and a reader relying on it
would believe there's a guard rail that isn't there. Fix the sentence (e.g.,
"a piecewise-linear term with many distinct values gets little or no
compression, since its clamped value has to be part of the grouping key") —
this is a one-line doc fix, no behaviour change needed.

### Nice-to-have

- `install_glum_shim()`'s `_SHIM_INSTALLED` check-then-set is not
  lock-protected. In a benign race (two fits starting in the same process at
  once) the wrapper could end up double-wrapped (patched-around-patched); it
  would still be correct (the passthrough chain still terminates at the real
  original), just with one extra call frame. Not worth blocking on.
- No live-browser (Playwright) confirmation that the elapsed-time progress
  caption actually renders and updates during a genuinely long (multi-second)
  fit in the deployed app, on both Streamlit versions — I relied on the
  `AppTest`-based page tests (which do exercise `fit_progress_callback`) and a
  direct unit-level thread-timing check instead, for time reasons.

## Re-check

For round 2, re-run only what changes:

1. Re-run my `check_edge_designspec.py`-style 0-row + categorical/interaction
   probe against the fix for S1 (0-row `build_sparse` with a categorical, an
   interaction, and a combination of both) and confirm it either returns a
   valid empty matrix or raises a clear `easy_glm`-level error, plus that
   `build_dense` and the `fit_glm`/`predict` paths are unaffected.
2. Confirm the `aggregate` docstring in `core/fit.py` no longer claims a
   refusal that isn't implemented (S2) — diff review only, no re-run needed
   unless the fix adds real behaviour.
3. Re-run the full gate list (`black --check .`, `ruff check .`, `pytest -q
   tests` on the repo venv, the app/workbench subset on the Streamlit 1.63
   venv, `pytest -q -m slow tests/test_scale.py`) to confirm nothing else
   moved.
4. If time allows: a live Playwright pass on the Model page during a
   multi-second fit (1M+ rows) on both Streamlit versions, to close the one
   nice-to-have gap above.
