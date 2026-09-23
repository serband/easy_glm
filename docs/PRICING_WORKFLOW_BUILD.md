# Interactive pricing workflow: build checklist

Approved workflow: `examples/modelling_workflow_proposal.md`.
Branch: `codex/interactive-pricing`. Release remains paused.

## How work is judged

The user is a pricing actuary making one modelling decision at a time. A step is
complete only when the public walkthrough performs it with short calls, its
results are readable, and the corresponding checks pass. Internal functions alone
do not satisfy a step.

| Actuary's action | Required result | Status |
| --- | --- | --- |
| Set data, roles, split and bands | Training-only boundaries, explicit overrides, useful preview | Verified |
| Fit main effects | Named model, rates, exposure and A/E; earlier models preserved | Verified |
| Investigate and add missing variables | Full current training residuals; explicit additions and refit | Verified |
| Fit first CatBoost interaction | Fixed GLM baseline, deployed table and support visible | Verified |
| Fit the next interaction | Baseline includes all previous tables; earlier tables unchanged | Verified |
| Inspect and amend rates | Before/after preview, explicit freeze/refit choice, separate rebalance | Verified |
| Compare and validate | Honest training/CV labels; holdout only on explicit request | Verified |
| Export and reopen | Current Excel tables and saved scorer reproduce reviewed predictions | Verified |

## Responsibilities

- Astra planning: reviewed current APIs and statistical constraints.
- Independent Sol critique: reviewed usability and reuse of existing engines.
- Separate actuarial reviewer: specified lifecycle and leakage checks.
- Sol builders: session/fitting; inspection/export; amendments/dependent refits.
- Primary agent: integration, executable French motor guide, this checklist.
- Astra validation, adversarial testing and actuarial use: completed; findings fixed and rechecked.

## Scope decisions

- Add `PricingSession` rather than change the meaning of the existing `EasyGLM`
  constructor. Existing callers remain supported.
- Reuse the GLM, sequential CatBoost and table-scoring engines. No new statistical
  selection method is being introduced.
- Ordinary GLM alpha tuning currently learns a design before its internal folds.
  A separate fold-local evaluation must support honest model comparisons; do not
  rename tuning scores as independent performance.
- Preserve the complete edited table when refitting later interactions. Ambiguous
  edits on fold-dependent automatic bands must fail clearly rather than move to a
  different band.
- Current scoring metrics must be recalculated after editing. Pre-edit CV evidence
  cannot be presented as evaluation of the new rates.
- No README, package version, release tag or publishing changes.

## Completion evidence

- Latest check, 23 September 2026: all 31 current Markdown Python blocks passed
  in a clean Python 3.13 notebook installation, with the real loader and displays.
  Saved-model predictions matched exactly; a scorer rebuilt from Excel matched
  to 3.33e-16. See [the verification record](PRICING_WALKTHROUGH_VERIFICATION.md)
  for the environment, checks and retained outputs.
- The complete Markdown ran on 50,000 French motor policies: main fits, searches,
  two ordered interactions, both amendment choices, holdout, Excel and saved-model
  scoring. No tutorial helper functions or workbench session were needed.
- Built and installed the wheel into a separate directory. The guide and rate-review
  tests passed against that installed package: 13 tests in 85.90 seconds.
- Astra closed the statistical and evidence checks, including exact upstream-table
  preservation and invalidating old validation after edits.
- A separate actuarial reviewer exercised two non-neutral interactions and both
  edit choices, then checked the display and naming corrections.
- Adversarial checks covered reordered/changed saved data and models reopened
  without analysis data. Policy-ID splits survive row reordering.
- Rendered and inspected the actual A/E and interaction charts. Corrected title
  clipping and legend overlap; unsupported interaction cells are marked.
- Ruff, Black, mypy (43 source files), and whitespace checks passed.
- Broad regression suite: 1,525 passed, one skipped and one slow test deselected.
  Its nine failures were all sandbox-denied localhost binds. Rerunning both launch
  test files with port permission passed all 10 tests.
- Final pricing tests: 27 passed. Final installed-wheel chart checks: two passed.
  No remaining findings from the independent reviews.

## Review locations

- Human walkthrough: `examples/pricing_walkthrough.md`.
- Implementation: `/private/tmp/easyglm-pricing-workflow`, branch
  `codex/interactive-pricing`.
- A review copy of the guide and this checklist is also in the normal project at
  `/Users/serban/Documents/Projects/easy_glm`.
- Nothing has been released, tagged or pushed. The README and version are unchanged.
