# Independent review — D3 + D4: Compare page and self-contained HTML report

*Branch `piece/d3-d4`, 6 commits on `f0508e8`. Reviewed 2026-09-03 against
`docs/RELEASE_0.4_PLAN.md` §D items 3–4 and their test rows (lines 169–170),
`docs/reviews/00-plan-review.md` §5 D3/D4 (lines 339–340) and AGENTS.md.*

## Verdict

**Approve with one blocking fix and nine should-fixes.**

The engine is right. `relativity_diff` is exact, exactly symmetric, has the
tolerance semantics the plan asked for, and its label-matching decision for
moved knots is the honest one. The report is genuinely self-contained (one
`file:` request, zero `<script>` tags, no external `src`/`href`), renders every
predictor, opens with no console errors, is 345–383 kB, and the script in its
appendix runs and rebuilds the model. The SVG-instead-of-Plotly decision is
**correct and should stand** — see §3. Core is untouched, the suite is green on
both Streamlit versions, ruff/black are clean, the e2e passes, and the check
script reproduces its markdown byte for byte.

The one blocking item is not in the code: the owner-facing check document
describes a fixture that the check script does not build, and promises the
reader a row that is not in the table it points at.

---

## Blocking

### B1. The check document describes two models the script does not build

**What.** `docs/checks/d3-d4-compare-report.md` (generated from `DOC_TEXT` in
`scripts/checks/d3_d4_compare_report.py`) opens with

> `freq_v1` uses plain banded (step) terms; `freq_v2` adds a DrivAge × VehPower
> interaction and **treats Density as a straight line in log space**

and later tells the reader

> A factor one model has and the other does not (**here Density is linear in one
> and banded in the other**, and the interaction exists only in `freq_v2`) is
> listed once as *only in …*.

Both statements are false. `DesignConfig` lives on `Project`, not on
`ModelConfig` (`src/easy_glm/workflow/project.py:176`), so the script's
`{"design": {"variables": {"Density": {"kind": "linear"}}}}` makes Density
**piecewise-linear in both models**. I built the script's exact project and
fitted it:

```
freq_v1 Density kind: linear
freq_v2 Density kind: linear
diff height: 26
(base rate)       changed     1
DrivAge           changed    15
DrivAge×VehPower  only_in_b   1
Region            changed     3
VehPower          changed     6
Density: no rows at all
```

The only *only in* row in the whole table is the interaction. There is no
Density row, linear or banded, and nothing in the fixture exercises the moved-knot
path the surrounding bullet is teaching.

**Failure scenario.** The owner cannot read code, so this page is the whole
specification of what the feature does. He reads that Density is banded in one
model and linear in the other, scrolls the difference table looking for the
*only in* row it promises, and finds neither. Either he concludes the diff is
dropping factors (it is not), or — the more damaging reading — he concludes that
a factor which is **banded in one model and linear in the other is safely
reported as "only in"**, and stops checking. It is not: I fitted that case
directly (Density step in one project, linear in the other, same knots) and 19
of ~22 band labels collide, so the tool silently reports them as ordinary
`changed` rows carrying only model A's `kind` (see S4).

**Exact fix.** Correct the prose in `scripts/checks/d3_d4_compare_report.py`,
then regenerate with `--write`:

* header → "Two frequency models on the same six factors, both with Density as a
  piecewise-linear term; `freq_v2` adds a DrivAge × VehPower interaction."
* the *only in* bullet → "A factor one model has and the other does not (here the
  DrivAge × VehPower interaction, which exists only in `freq_v2`) is listed once
  as *only in …*."
* the moved-knot bullet is fine as written — it describes real behaviour — but it
  should not claim the fixture demonstrates it.

While regenerating: `docs/checks/img/d3_compare_metrics.png` is cut off at the
last **train** row, and the paragraph beside it opens with "**Holdout first.**"
Scroll the capture (or collapse the caption) so the holdout block is visible.

---

## Should-fix

### S2. The report names a challenger it then silently refuses to compare

`to_report_html` skips the comparison when the challenger's columns are missing
from `df` (`report.py:757–765`), but the subtitle and the metrics table still
announce it. Reproduced by dropping a column only the challenger needs:

```
built ok: 111 kB
compare section present: False
subtitle: Champion freq_a · challenger freq_b · generated ...
TOC: Summary | Rating factors | Lift and Gini | Appendix
metrics header cols: freq_a · train, freq_a · holdout, freq_b · train, freq_b · holdout
```

A reader gets a report that says "challenger freq_b", shows freq_b's metrics, and
has no comparison — indistinguishable from a bug. **Fix:** when `challenger` is
set and `challenger_pred` is empty, emit one line where the compare section would
have been ("*freq_b could not be scored on these rows (missing: Density), so
there is no comparison section*"), or drop the challenger from the subtitle and
the metrics columns as well. One `if` in `to_report_html`.

### S3. The champion's expected line is hidden under the challenger's

`report.py::_ae_chart` appends the challenger as a **solid** green line, drawn
after the orange champion line, so wherever the two models agree the champion
disappears entirely. In the fixture's Density block the orange line is in the
legend and nowhere on the chart. The workbench does not have this problem:
`app/charts.py:113` draws the challenger `dash="dash"` — and the check doc even
tells the actuary to look for "green **dashed**". **Fix:** give
`_svg.category_chart`'s `lines` an optional dash and pass it for the challenger,
so the report matches the app and the doc.

### S4. A factor that is step in one model and linear in the other is compared silently

`_relativity_rows` takes `kind` from run A only, and band labels for a step term
and a linear term with the same knots are identical, so they match and come back
as ordinary `changed` rows:

```
Density rows: 23   statuses: ['band_only_in_a','band_only_in_b','changed']
kind column values: ['numeric','linear']   (whichever model was A)
shared labels: 19 of 21 / 23
```

Two problems. The reader is never told the two models represent the factor
differently. And the comparison is not like for like: a step band's relativity is
flat across the band, a linear band's is the value at the **band start**, so a
row reading "changed −0.21" can overstate or understate what the two models
actually charge across that band. **Fix (minimal):** when
`a_rows[var][0] != b_rows[var][0]`, write `kind` as `f"{kind_a} → {kind_b}"` (or
add a `kind_b` column), so every such row is self-explaining and the reader knows
to treat it as an approximation.

### S5. Two relativities that are both exactly zero are reported as a change

`_log_ratio` returns `None` when either value is ≤ 0, and a `None` log diff is
always emitted as `changed`. With both models at 0.0 for the same band:

```
both zero -> rows: 1   [{'variable':'Region','band':'R2','status':'changed','log_diff':None}]
```

`update_relativity(..., 0.0)` is accepted, so an actuary who floors a band to
zero in both models gets a phantom row. **Fix:** `if rel_a == rel_b: continue`
before computing the log ratio (the `a > 0 and b > 0` guard then only fires for
genuine sign/zero changes, which do deserve a row).

### S6. Every chart in the report is an unlabelled `role="img"`

`_svg._frame` emits `role="img"` with no `<title>` child and no `aria-label`; a
screen reader announces 25 anonymous images. Per-band `<title>` tooltips exist
(757 of them in the fixture report) but they hang off inner `<rect>`s, not the
root. **Fix:** add a `title` parameter to `_frame` and emit it as the SVG's first
child; pass the chart heading through from `_relativity_chart` / `_ae_chart` /
`heatmap`. Two lines, and it also makes the printed PDF's bookmarks useful.

### S7. The Compare page's metrics table is the wrong shape for a side-by-side read

`pages_compare._metrics_table` produces 22 rows × 2 model columns with `rows used`
and `metric` repeated on every row, so train and holdout for one model are 8 rows
apart. The report's `_metrics_table` already has the right shape — metric rows ×
`model · subset` columns — which puts the four numbers a reader compares on one
line. Consider using that layout on the page too; it would also stop the
screenshot problem in B1.

### S8. The stated report size is understated

README and CHANGELOG both say "about 250 kB". Measured on the French-motor
fixture: **345 kB** champion only, **383 kB** with a challenger (the D3/D4 test
fixture, which is smaller, gives 212 kB). Say "a few hundred kB" or use 350–400.

### S9. D4's plan text also asks for the report "from the CLI"

`RELEASE_0.4_PLAN.md:73` — "From the Export page **and the CLI**". Only the Export
page ships (there is no `easy-glm` console script yet; `pyproject.toml:37` has
only `easy-glm-workbench`). That is genuinely §F work, but it is currently
invisible: add a line to the CHANGELOG saying the CLI half of D4 lands with F so
it is not lost when 0.4.0 is cut.

### S10. Only the opt-in e2e actually proves D4's browser criterion

`test_it_opens_in_a_headless_browser_without_console_errors` is `importorskip`-ed
and is the one skip in the default suite (`472 passed, 1 skipped`). The real
coverage is `tests/e2e/test_persona_data_scientist.py`, which does download the
report and open it — good, and the right place for it. But
`docs/reviews/00-plan-review.md:340` states the criterion as a unit-level test.
Note in AGENTS.md's test table which lane proves it, so CI cannot pass D4 with
that assertion silently skipped.

---

## Judgement on SVG instead of Plotly

**Acceptable — keep it.** I rendered the fixture report at 1280 px and as A4 PDF
and read every section.

* **Readability.** The relativity, A/E and lift charts are as legible as a static
  Plotly export: labelled dual axes, round ticks, exposure bars behind the rate
  lines, rotated band labels with a `…` truncation, a dashed reference line at
  1.00. The heatmaps carry a colour key and are centred on 1.00 on a log scale.
* **Hover.** Native `<title>` tooltips are on every band column and every heatmap
  cell (757 in the fixture report), carrying exposure, actual, expected and the
  value. That is the information a Plotly hover would give.
* **Print.** The PDF is 579 kB and correct; `@media print` drops the background
  and sets `break-inside: avoid` on each factor block. Plotly's WebGL/canvas
  charts print far worse.
* **Budget.** Inlining `plotly.min.js` is 4.84 MB before a single chart; the
  acceptance is 5 MB. Any Plotly path either breaks the budget or breaks
  self-containment via a CDN — the plan review's own D4 line forbids the latter.

I looked for a lighter alternative and do not recommend one. `uPlot` (~48 kB) or
`Chart.js` (~200 kB) would fit the budget, but they buy interactivity the report
does not need at the cost of the property that makes this file trustworthy for a
filing: **it contains no JavaScript at all, so it cannot error, cannot be blocked
by an enterprise CSP, and cannot rot.** That is worth more than a zoomable axis
in an artefact whose job is to still open in five years. The residual gaps are
S3 (dash the challenger) and nit N12 (no log x-axis), not the technology choice.

---

## Nits

* **N11.** `_svg.category_chart([], ...)` raises `ValueError: zip() argument 2 is
  longer than argument 1` (`_x_labels` uses `strict=True` while `n = max(len, 1)`
  manufactures one centre). Unreachable today — I checked that `ae_by_variable`
  returns a one-row "Other / Unknown" table even for an all-null column, and
  `lift_table` always returns bins — but it is a one-line guard.
* **N12.** `curve_chart` has no log x-axis, while the Rate tables page switches to
  one when `hi / lo > 100` (`pages_tables.py:95`). For a heavy-tailed linear
  factor the report's curve compresses the informative range into the left ~8 %
  of the chart. Also, `report._relativity_chart`'s interpolation loop is a verbatim
  copy of `charts.py::linear_curve_chart`'s `_polyline` — worth sharing. (The
  geometric interpolation itself is *correct*: scoring is
  `relativity * exp(slope * (x - from))`, `rate_model.py:291`.)
* **N13.** In a factor block the relativity chart includes the "Other / Unknown"
  band as the last point of the age curve — visually joined to `≥ 72.0` as if it
  were the next band — while the A/E chart beneath it omits that band, so the two
  x-axes in the same block do not line up.
* **N14.** Lift and double-lift charts draw exposure bars over equal-exposure bins,
  so the bars are all the same height, fill the plot and hide nothing but
  themselves. Pass `bars=None` for those two.
* **N15.** `main.py` calls `S.set_challenger(None)` whenever fewer than two models
  are fitted, so editing the design (which stales the fits) silently forgets the
  challenger even after both are refitted.
* **N16.** The check doc's diff section does not say that a band's **premium**
  change is its relativity change *times* the base-rate change. The fixture shows
  both moving (base rate −2.4 %, top band +9.4 %), so one sentence would stop a
  reader quoting +9.4 % as the premium impact.

---

## Missing tests

1. **`easy_glm.workflow._svg` has no tests at all** — 466 lines of hand-rolled
   tick/axis arithmetic reached only through the report's HTML assertions. At
   minimum: `nice_ticks` on degenerate ranges (`lo == hi`, non-finite), a chart
   whose values are all `None`, a one-band chart, `<script>` in a label is
   escaped (it is — I checked), and the N11 empty-label guard.
2. **Symmetry of `relativity_diff`.** Not pinned anywhere. It holds exactly — I
   verified 24 rows each way, every status swapped and every `log_diff` negated to
   1e-12 — which is worth a regression test given how easy it is to break.
3. **A step-in-one-model / linear-in-the-other variable** (S4). No test.
4. **A single edit to a *linear* band gives exactly one row.** Only step,
   categorical, interaction and base-rate edits are covered. I confirmed it does
   (one row, `log_diff = 0.2624` for a ×1.30 edit) — the band-start decision is
   sound because the linear table chains `relativity_to[i] == relativity[i+1]`, so
   the band starts determine the whole curve.
5. **Tolerance boundary and sign.** `|log diff| == tol` is *not* reported (strict
   `>`, matching the plan) and a negative tolerance is `abs()`-ed. Both verified,
   neither pinned.
6. **The report with a challenger that cannot be scored** (S2).
7. **The Compare page's metrics table equals `run.metrics`.** I wrote it: all 32
   metric cells match `_fmt(run.metrics[subset][key], spec)` exactly. Cheap to add.

---

## What I re-ran, with numbers

| Check | Result |
|---|---|
| `easy_glm.__file__` under the worktree | ✔ `…/wt-d3/src/easy_glm/__init__.py`, Python 3.14.7 |
| Full suite, Streamlit 1.57 | **472 passed, 1 skipped**, 177 s (the skip is the Playwright report test) |
| `test_app.py` + `test_w2_pages.py` + `test_d3_d4_compare_report.py`, Streamlit **1.63.0** | **94 passed, 1 skipped**, 12 s |
| `ruff check .` / `black --check .` | All checks passed / 93 files unchanged |
| `git diff f0508e8..HEAD -- src/easy_glm/core` | **empty** — core untouched, as claimed |
| Golden | `tests/test_golden.py` and `tests/fixtures/` unchanged in the diff; 7 passed |
| e2e, documented command, Playwright venv | **3 passed, 104 s** |
| `scripts/checks/d3_d4_compare_report.py` vs the committed markdown | identical but for `print`'s trailing newline |
| Screenshots | 7 new files, largest 138 kB, budget 300 kB ✔ |
| Report size, French-motor fixture | **345 kB** champion only, **383 kB** with challenger (limit 5 MB) |
| Report build time, 50k rows | 0.08 s / 0.12 s — the Export page's eager build is not a performance problem |
| Self-containment | 0 `<script>` tags, 0 external `src`/`href`; all 6 `href`s are in-page anchors and all resolve |
| Headless Chromium (both reports) | **0** console errors/warnings, 0 page errors, 0 failed requests, **1** request total (the file); no horizontal body overflow |
| `<section class="variable">` count | **7** = 6 predictors + 1 interaction; headings `DrivAge/VehAge/BonusMalus/Density/VehPower/Region/DrivAge×VehPower` |
| Compare section | present with a challenger, absent without (and absent when champion == challenger) |
| Appendix script | extracted from the `<pre>`, unescaped, run in a subprocess → **rc 0**, 74-row coefficient table, `holdout A/E: 1.0116` |
| Accessibility | 25/25 `svg[role="img"]` have no accessible name (S6); 757 per-element `<title>` tooltips present |
| `relativity_diff` symmetry | A↔B: 24 rows each way, statuses swapped, `log_diff` negated — **perfectly symmetric** |
| Tolerance semantics | `|log diff| == tol` → 0 rows; `tol = 0.0099` → 1 row; identical runs at `tol = 0` → 0 rows; negative tol `abs()`-ed |
| Single adjustment | ×1.25 on one Region level → exactly 1 row, `log_diff = 0.223144 = log(1.25)` |
| Single linear-band edit | ×1.30 on one Density node → exactly 1 row, `log_diff = 0.262364` |
| Base rate | ×1.10 → 1 row, `(base rate)`, `log_diff = log(1.10)`; in the fixture report −0.0242 = log(0.0530/0.0543) ✔ |
| Moved knots | reported as `band_only_in_a` / `band_only_in_b`, no false `changed` |
| AppTest: metrics table vs `run.metrics` | all 32 cells match exactly |
| AppTest: CSV download | "Download the differences (.csv)" present |
| AppTest: sidebar → page default | sidebar `(none)` → key `cmp_b_freq_a_None`; sidebar `freq_b` → key `cmp_b_freq_a_freq_b`, both defaulting correctly |
| AppTest: project isolation | after `set_project`, token changes (`98ffa7a2` → `34e19d7b`), `S.challenger()` → `None`, and **zero** `cmp_*`/`diag_chal*`/`tables_chal*`/`report_chal*`/`challenger` keys survive — no leak |
| AppTest: only one model fitted | "Compare needs **two fitted models**" info, no exception |
| AppTest: challenger deleted while selected | 3 models, `freq_c` selected then removed → no exception, selector falls back to `freq_v… freq_b`, options shrink cleanly; same on Diagnostics |
| All of the above on Streamlit 1.63 | 8 probes passed |

---

## The four questions the builder raised

All four are genuine domain questions with sensible defaults already implemented,
and none of them blocks the merge. My reading of each:

1. **Tolerance — 1 % or 5 %?** Genuine. The 1 % default is right for the *first*
   look (it is roughly where two separately-fitted models stop differing by
   noise) and the page already exposes the box, so the cost of a wrong default is
   one keystroke. Worth asking; do not pre-empt it.
2. **Base rate as a row, or as a headline?** Genuine, and the more consequential
   of the four — the fixture shows the base rate moving −2.4 % while bands move up
   to +9.4 %, and a reader who quotes a band's `log_diff` as the premium impact
   will be wrong by the base-rate amount. Whichever way he answers, N16's sentence
   should go in.
3. **Report ordering — configuration first, or a one-page filing summary first?**
   Genuine and purely editorial; the current order (data → model → metrics) is
   the defensible default for an internal document, and a filing front page is a
   real, separable request.
4. **Moved knots — refuse, or compare both curves at the union of band edges?**
   Genuine and the most interesting. The current refusal is the honest default and
   I would keep it as the default. Note that the answer also settles S4: the same
   "evaluate both curves at common points" machinery is what would let a
   step-vs-linear factor be compared properly instead of matched by label.
