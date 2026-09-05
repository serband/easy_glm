# Future releases

This is the running list of improvements identified while using EasyGLM on
real modelling work. Items belong here until they are scheduled for a release.

## Model examples

- Add a binomial example built around a genuine yes/no outcome.
- **Implemented on the feature branch:** a Tweedie incurred-claims example and
  workbench walkthrough using the Swedish motorcycle portfolio, including
  zero-claim policies and exposure filtering.

## Tweedie power selection

- Keep manual selection as the current default.
- Add an optional training-only search across candidate powers between 1 and
  2, using profile likelihood to select the power.
- Show the powers tested, the comparison, and the selected power so the choice
  can be reviewed.
- Keep the holdout data completely separate for final model validation.

## Data setup

- When a dataframe is launched from Python, detect an existing train/test
  column and make its values and interpretation clear instead of initially
  presenting the data as broken.
- Support a spreadsheet round trip for datasets with many columns: download
  the detected variable types and proposed roles, edit them in Excel, validate
  the file, and upload it to apply the mapping.
- Ensure the exported Python workflow contains the final column roles,
  renames, recodes, derived columns, and split definition.

## Statistical improvements — implementation status

These items change model or variable selection rather than the fitted tables
at a given alpha.

- **Implemented on the feature branch:** run the *Missing factors* and *Missing interactions* searches (and the
  pair heatmap used to find candidates) on the training rows regardless of
  the Diagnostics page's *Rows* selector, say so on the page, and add a test
  that a factor planted on holdout rows only is not found. (Blocking: the
  page currently defaults to holdout.)
- **Implemented on the feature branch:** cross-validate on shuffled, seeded folds (`KFold(cv, shuffle=True,
  random_state=<split seed>)`) instead of glum's contiguous blocks; test that
  the CV alpha does not depend on row order.
- **Implemented on the feature branch:** score the missing-factor search with the same Pearson-excess z-score the
  pair search uses (bands with `expected >= min_expected × φ`, rare levels
  pooled), keeping the zero-claim bands; keep sd of log A/E as a secondary
  column.
- **Implemented on the feature branch:** compute "deviance explained" against an intercept-plus-offset null model
  (`null_model_predict`, fitted on train) on both subsets.
- **Implemented on the feature branch:** stage 2 cross-validation assembles an
  out-of-fold stage-1 offset (k extra main-effect fits) using the same seeded,
  shuffled folds as the two CV stages.
- Options to consider: an AGLM-style raw linear column next to a step
  factor's O-dummies (trend shrinkage); a "holdout locked" mode that shows
  train and CV numbers during design and reveals the holdout on request; a
  split-by-policy-key mode; the one-standard-error rule for the CV alpha;
  the family's variance function (not `φ E`) in the pair search for
  gamma/Tweedie, and an estimated φ for Poisson.
- **Implemented on the feature branch:** a step term's modal base cannot be the
  null row. Remaining guards: a gamma or inverse-Gaussian holdout row with zero
  loss should be excluded from the holdout deviance and counted; document that
  Gaussian uses a log link.
