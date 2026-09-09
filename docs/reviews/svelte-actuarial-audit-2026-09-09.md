# Svelte actuarial and UI audit — 9 September 2026

**Verdict: functional progress, but no clean UI sign-off.** The independent
walkthrough found a misleading partial interaction heatmap and substantial layout
and export-parity gaps. Correct mathematics alone is not workflow parity.

Scope: the isolated `codex/svelte-workbench` checkout. No release or change to the
user's live project, fitted model, adjustments or history is part of this audit.

## Findings corrected

| Finding | Correction | Evidence |
| --- | --- | --- |
| CV interaction coefficients used out-of-fold main predictions, while the composed model scores the final full-training mains. | Preserve CV penalty selection, then refit the selected cells against the final frozen mains. | Independent Poisson reproduction differed by up to 2.581% in predictions before correction; corrected predictions match the direct oracle to floating-point precision. Gamma, binomial and Tweedie oracles also pass. |
| An L1-ratio grid crashed during fixed-alpha out-of-fold main refits. | Use stage 1's selected scalar L1 ratio for those refits. | Regression exercises `[0.5, 1.0]`, column and array offsets. |
| Changing model-definition controls could raise `field is not defined` in the browser. | Replace nested dynamic Svelte bindings with explicit immutable field updates. | Reproduced before correction; family, Tweedie power, link, target, weight, offset, divide-by-weight and table-base controls now pass a browser regression. |
| An untouched factor said “Saved adjustments” after another factor was edited. | Derive the state from that factor's displayed saved values versus its original fit. | Explicit Apply regression now checks that untouched Region still says “Original fit”. |
| Late variable/table responses could clear a new tool choice or show stale rows. | Bind table requests to their exact model, fit, factor, page and revision; remount/reset busy state on factor changes; ignore obsolete results/errors. | Held-response browser regression covers old review completion, delayed tables, A→B→A ordering, no unintended commits or refits, and no browser errors. |
| Importance-axis labels collided near zero. | Keep zero visible and omit endpoint labels too close to it. | Browser bounding-box regression and a refreshed French motor screenshot confirm no overlap at 884 pixels. |
| Regularisation charts labelled fold-average retained counts as final counts and displayed long decimal ticks. | Label the CV mean and consistently format right-axis ticks. | Browser parity check covers combined axes, labels, selected penalty and narrow-screen tick spacing. |

Existing fitted models stay frozen. An explicit refit is required to obtain the
corrected CV interaction coefficients. Coefficient encoding and stored-model
interpretation have not changed, so no persistence-format bump is needed.

## Independent mathematical review

23 independent checks passed, including:

- Trailing arithmetic averages for windows 1, 2, 3 and 6, available initial points,
  equal weights and an unchanged missing-value row.
- Increasing and decreasing isotonic smoothing against an exhaustive weighted
  log-error optimum over 120 seeded cases.
- Piecewise-linear node values, slopes, midpoint scoring and cap/floor.
- Interaction exposure thresholds at equality, zero exposure and missing values.
- Training permutation importance against independently expanded matrices and
  explicit Poisson, Gamma, binomial and Tweedie deviance formulas. Step, linear,
  categorical and interaction-parent shuffles were covered. Target, weights and
  offsets remain fixed; production scoring does not refit or build a matrix.
- Final interaction coefficients, unchanged mains, CV paths, selected-alpha
  estimator predictions, pickle reload and rate-table scoring.

A separate read-only reviewer found no blocking defect in the interaction fix.
The underlying glum paths at non-selected alphas retain their out-of-fold search
provenance. Supported EasyGLM scoring uses only the final selected coefficients;
the UI's CV diagnostics use the preserved fold paths.

## Automated workflow checks

17 browser tests passed across 12 isolated configurations: model creation/fitting,
explicit adjustment application and races, all table tools, diagnostics and
comparison parity, permutation importance, interaction editing, automatic pair
A/E, A/E caching, review requests, session recovery and wide schemas.

After the fixes, the full parity workflow and interaction editor were rerun and
passed. Further model-controls and delayed table/context regressions passed. Fourteen frontend unit tests
and 224 focused Python regressions passed; five CV regressions were rerun after
the final estimator-state correction. One slow scale test was deselected.

Final Svelte checking reported zero errors and warnings; the production bundle,
formatting, lint, types and source diff checks passed. The generated minified
Svelte bundle contains intentional whitespace inside a runtime string literal,
which Git flags as trailing whitespace; it is preserved exactly as built.

These are targeted regression results, not a new full-suite result. The browser
checks use installed Chrome on macOS, at desktop and 884 × 773 viewports.

## Independent visual walkthrough

Completed in a separate French 50k session on port 8790, using installed Chrome
at 884 × 773 and 1440 × 1000. Two models were built through the UI: a three-fold
CV model with seven alphas, DrivAge step, BonusMalus piecewise linear, Region
categorical and DrivAge × Region; then a fixed-alpha challenger.

The reviewer exercised moving windows 1/2/3, both isotonic directions, cap/floor,
manual Apply and Discard; inspected numeric, categorical and interaction charts,
training permutation importance and both regularisation charts; compared two
models and downloaded project JSON. The original fit stayed visible. No browser
page errors occurred in the completed diagnostics/comparison/export flow.

History/race behavior and residual-search-to-refit coverage also come from the
automated workflows; the independent visual walkthrough was not exhaustive in
those areas. The numerical audit separately covers all four tested families.

Evidence: `/private/tmp/easyglm-actuary-review-20260909/REVIEW.md`, numbered
screenshots and reproducible scripts in that directory.

## Open findings blocking full UI/workflow sign-off

1. **P1 — Partial interaction heatmap misrepresents exposure.** The chart renders
   the first 200 of 462 cells and labels unloaded cells as if they have no exposure.
   `[42,44) × Provence-Alpes-Cotes-D'Azur` displays an em dash despite exposure
   87.844 and relativity 1 in row 200. Ile-de-France and Bretagne also have positive
   exposure outside that page. A complete matrix or an honest unloaded-cell state
   is required. Evidence: screenshot 08 and `race-results.json`.
2. **Rate editing layout.** The relativity chart expands with width and pushes the
   adjustment controls out of the first screen, even at 1440 × 1000. At 884 × 773
   the reviewer must scroll between a parameter and its resulting curve. Totals,
   history/reset controls and explanatory text compete with the main decision.
3. **Export parity.** Project JSON works. Excel tables, HTML report, generated
   script and scorer exports still require Streamlit; fitted runs and undo history
   are not included in the JSON download.
4. **Minor display noise.** A constant challenger displays a Gini around -3e-14
   rather than zero. This is floating-point noise, not meaningful discrimination.

The late-variable-response and importance-label defects found in this walkthrough
were subsequently corrected and checked as described above. The untouched-factor
label was independently rechecked after refreshing the review session.

## Evidence and limits

A final read-only live-state check matched the original project, fit identifier,
all nine rate tables and five undo steps exactly. Derived summary differences
were only floating-point summation noise (maximum relative difference 3.4e-15).


- Mathematical oracle scripts and results: `/tmp/easyglm-math-audit/`.
- Browser batch results and logs: `/private/tmp/easyglm-ui-audit-20260909/`.
- Regression sources: `tests/test_interactions.py`, `frontend/tests/model-controls.spec.js`
  and the existing frontend workflow specifications.
- Training permutation importance measures training performance degradation under
  shuffling. It is not causal importance or a replacement for holdout validation.
- Smoothing can change total expected claims; rebalancing remains a separate action.
- Windows, Positron, corporate browsers and an exhaustive accessibility review are
  outside this pass. Passing these checks does not establish that no UI bugs exist.
