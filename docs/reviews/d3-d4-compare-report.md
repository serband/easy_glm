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

---

# Re-check (2026-09-03)

*Six further commits, `git diff 1a8219f..HEAD` = 20 files, +1245 / −274. Only
the new diff was re-read; every number below was re-measured on it.*

## Final verdict

**Approve — merge.** The blocking item is resolved and all nine should-fixes are
done, several of them better than I asked. `git diff f0508e8..HEAD -- src/easy_glm/core`
is still empty. Three chart nits (N12–N14) remain open by choice; none of them
blocks, and none of them is in the owner's path.

## B1 — resolved

The check document now describes the models the script actually builds:

> Two frequency models on the same six factors, both treating Density as a
> straight line in log space (**the band design belongs to the project, so every
> model shares it**); `freq_v2` adds a DrivAge × VehPower interaction, which is
> the only structural difference between them.

and the *only in* bullet now names the interaction rather than Density. I fitted
the script's exact project again and confirmed the diff it describes: 26 rows,
`(base rate)` 1, `DrivAge` 15, `Region` 3, `VehPower` 6, `DrivAge×VehPower`
`only_in_b` 1 — no Density row, exactly as the corrected prose now says.

The document's new worked example is arithmetically right, which I checked
against the fixture rather than taking on trust:

| doc says | measured |
|---|---|
| overall level +2.4 % | `base_rate_change(freq_v1, freq_v2)` = **+2.4498 %** (base rate 0.052996 → 0.054294) |
| largest band move −8.9 %, DrivAge `[28.0, 30.0)` | **−8.94 %** (0.5883 → 0.5357, `log_diff` −0.0936) |
| premium falls about 6.7 % | **−6.71 %** |

`scripts/checks/d3_d4_compare_report.py` (no `--write`) reproduces
`docs/checks/d3-d4-compare-report.md` byte for byte apart from `print`'s trailing
newline. The two screenshots I complained about were re-anchored:
`d3_compare_metrics.png` now shows the whole metrics table including the holdout
columns, and `d3_compare_diff.png` now frames the tolerance box, the new level
headline and the table together.

## The common-grid rewrite — the substantive change, and it is right

The builder went further than S4 and replaced label-matching for **numeric and
piecewise-linear** factors with a comparison on the union of both models' band
edges (`_curve` / `_common_grid` / `_value_at` / `_grid_point`). This pre-empts
open question 4, so I checked it on its merits rather than only for regressions.

It is a clear improvement, and the headline case is the proof:

* **Moved knot.** Shifting one DrivAge knot from 28.0 to 28.5 used to give four
  uninformative `band_only_in_*` rows. It now gives **one** row, `[28.0, 28.5)`,
  `log_diff` −0.0447 — precisely the ages that would be charged differently.
* **Two models with genuinely different knot counts** (9 vs 21 DrivAge bands):
  **20** common-grid rows, all `changed`, top row `< 21.0` at +0.307. Under the
  old rule that would have been ~30 rows saying only "these do not line up".
* **Correctness of the interval-start rule.** Exact for step-vs-step (flat inside
  an interval) and — less obviously — exact for linear-vs-linear: the grid uses
  the union of both models' knots, and continuity makes each interval's end the
  next interval's start, so agreeing at every grid point means agreeing at both
  endpoints of every interval, and two exponentials that agree at both endpoints
  are identical on it.
* **The one approximation, and it is disclosed.** For a *mixed* pair the step is
  flat while the linear slopes, so reading the start can understate the
  difference inside the interval. The ceiling is one band's drift of the linear
  curve; on this fixture's Density that is **+55.1 %** on the widest band
  (`[7313, 27000)`, 1.1502 → 1.5424), median 0.00065. Every such row is now
  labelled `kind = "numeric → linear"` so it cannot be mistaken for a
  like-for-like comparison, and rewritten question 4 asks the owner exactly this
  ("read each interval at its **start** … or the exposure-weighted average?").
  That is the honest handling. *One sentence for the next pass:* say it in the
  bullet too, not only in the question — a reader who skips the questions should
  still know a `numeric → linear` row is the value at the band start.

## Every relativity_diff case re-run

| case | result |
|---|---|
| identical runs (A vs A, B vs B) | **0 rows** each; the eight documented columns survive on an empty frame |
| single categorical adjustment ×1.25 | **1 row**, `Region / R2`, `log_diff` 0.22314355131420976 = log(1.25) |
| single numeric step band ×1.25 | **1 row**, `DrivAge [28.0, 31.0)`, same `log_diff` |
| single linear node ×1.30 | **1 row**, `Density [16.63, 25.56)`, `log_diff` 0.26236426446749106 |
| moved knot (28.0 → 28.5) | **1 row**, `[28.0, 28.5)`, −0.0447 (was 2 + 2 `band_only`) |
| step vs linear, same factor | **21 rows**, `kind` = `numeric → linear`; reversed run reads `linear → numeric` |
| both relativities 0.0 | **0 rows** — S5 fixed |
| both relativities −1.5 | **0 rows** (identical values are never a change, sign irrelevant) |
| 1.42 → 0.0 | **1 row**, `log_diff` null, sorted last — a real crossing is still listed |
| symmetry, A vs B | 24 vs 24 rows, every status swapped, every `log_diff` negated |
| symmetry, step vs linear | 47 vs 47, symmetric |
| symmetry, moved knot | 1 vs 1, symmetric |
| tolerance `|d| == tol` | **0 rows** (strict `>`); `tol = 0.0099` → 1; negative tol `abs()`-ed → 0; `tol = 0` on identical runs → 0 |
| `base_rate_change` | +10 % → 0.10000000000000009; reversed −9.09 % (a ratio−1 is correctly asymmetric, not a bug); zero base rate → `None`, and the diff still emits the `(base rate)` row with a null `log_diff` |

## The other should-fixes

* **S2** — a challenger that cannot be scored now gets its own section under the
  same `#compare` anchor, so the TOC link still resolves: *"This report names
  freq_b as the challenger, but it could not be scored here: the prepared data no
  longer has the columns it needs (Density). Its metrics in section 1 are the
  ones recorded when it was fitted; there is no double lift and no relativity
  comparison in this file."* Exactly the fix asked for.
* **S3** — the challenger's line is dashed (`stroke-dasharray="7 4"`) in the chart
  *and* in the legend swatch. I re-rendered the Density block: the champion's
  orange line now shows through the green dashes on both A/E charts.
* **S4/S5** — see above.
* **S6** — **25 of 25** `svg[role="img"]` now carry a `<title>` first child; the
  browser reports **0** unnamed. Names are specific ("DrivAge: actual vs expected
  by band (holdout)"), so the PDF outline is usable too.
* **S7** — done as recommended: `metric | freq_a · train | freq_b · train |
  freq_a · holdout | freq_b · holdout`, with the model facts split into their own
  table below. The fixture screenshot now makes the real story readable at a
  glance — freq_v2 wins on train Gini (0.3523 vs 0.3352) and *loses* on holdout
  (0.2883 vs 0.2916).
* **S8** — README and CHANGELOG now say "350–400 kB"; measured 346 kB and 386 kB.
* **S9** — CHANGELOG now records that the CLI half of D4 lands with F.
* **S10** — AGENTS.md now states which lane proves the browser criterion. Verified:
  in the Playwright venv `tests/test_d3_d4_compare_report.py` runs **38 passed,
  12 skipped** (the skips are the AppTest classes) — the headless-browser test
  runs and passes there, and the static half (no `<script>`, no external
  `src`/`href`) is now asserted on every run in every venv.
* **N11** (empty labels), **N15** (`set_challenger(None)` no longer wipes the
  choice while fits are stale) and **N16** (the premium-multiplies sentence) also
  fixed.

## Missing tests — closed

All seven are now covered; the file went from 29 to **50** tests. New:
`test_the_diff_is_symmetric`, `test_a_step_and_a_linear_term_are_compared_on_a_common_grid`,
`test_one_edited_linear_band_is_exactly_one_row`, `test_the_tolerance_boundary_is_strict`,
`test_two_identical_relativities_are_never_a_change`, `test_base_rate_change_is_the_overall_level`,
`test_a_challenger_that_cannot_be_scored_is_explained`, `test_the_metrics_table_is_exactly_the_runs_metrics`,
`test_it_carries_no_javascript_at_all`, `test_every_chart_has_an_accessible_name`,
`test_the_challengers_line_is_dashed`, plus a whole `TestSvg` class (ticks,
degenerate charts, escaping, dashes, heatmap naming). `test_moved_bands_are_only_in_rows_not_false_changes`
was correctly *replaced* by `test_a_moved_knot_is_compared_on_the_common_grid`
rather than deleted.

## Still open (nits, by choice — not blocking)

* **N12** — `curve_chart` still has no log x-axis (the Rate tables page switches
  to one at `hi / lo > 100`), and `report._relativity_chart`'s interpolation loop
  still duplicates `charts.py::linear_curve_chart`'s `_polyline`.
* **N13** — the numeric relativity chart still joins "Other / Unknown" onto the
  end of the age curve, while the A/E chart beneath it omits that band, so the
  two x-axes in one block do not line up.
* **N14** — lift and double-lift still draw equal-height exposure bars over
  equal-exposure bins.
* One sentence recommended above, so a `numeric → linear` row's number is
  explained in the bullet and not only in question 4.

## Re-run, with numbers

| Check | Before | Now |
|---|---|---|
| Full suite, Streamlit 1.57 | 472 passed, 1 skipped, 177 s | **493 passed, 1 skipped, 173.22 s** |
| App tests, Streamlit 1.63.0 | 94 passed, 1 skipped | **115 passed, 1 skipped, 11.89 s** |
| `test_d3_d4_compare_report.py` in the Playwright venv | — | **38 passed, 12 skipped** (browser test runs here) |
| e2e, documented command | 3 passed, 104 s | **3 passed, 105.54 s** |
| `ruff check .` / `black --check .` | clean | **clean** / 93 files unchanged |
| `git diff f0508e8..HEAD -- src/easy_glm/core` | empty | **empty** |
| Golden + fixtures since `1a8219f` | — | **untouched** |
| Check script vs committed markdown | identical | **identical** (trailing newline only) |
| Report size (French motor) | 345 / 383 kB | **346 / 386 kB**; build 0.09 s / 0.14 s |
| Report self-containment | 0 scripts, 0 external | **0 `<script>`, 0 external `src`/`href`** |
| Headless Chromium, both reports | 0 errors | **0 problems, 1 request, 7 `section.variable`, 0 unnamed `role="img"`, TOC all resolve, no h-overflow** |
| Appendix script in a subprocess | rc 0 | **rc 0**, `holdout A/E: 1.011592477159294` |
| Compare page metrics vs `run.metrics` | 32 cells exact | **32 metric cells + 12 fact cells exact** |
| Level headline on the page | — | present, `+2.0%` on the test fixture, matching `base_rate_change` |
| CSV download | present | **present** |
| Project isolation | no leak | **no leak** — token changes, `S.challenger()` → `None`, zero leftover widget keys |
| Challenger deleted while selected (3 models) | graceful | **graceful**, no exception, selector falls back |
| Reviewer AppTest probes | 7 | **9 passed** (1.57), rewritten for the new metrics shape |
