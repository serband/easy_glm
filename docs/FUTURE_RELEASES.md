# Roadmap

This is a list of proposed work. It does not promise a release date, priority
or delivery order. For released changes, see the [changelog](../CHANGELOG.md).

## Modelling methods

- **Hierarchical factors:** allow sparse codes to borrow strength from their
  parent groups. Show support and fallback rules and export leaf-level lookup
  files alongside component detail.
- **Spatial smoothing:** assess geographic smoothing and local residual effects,
  with validation suited to new areas and future periods.
- **Tweedie power selection:** investigate an optional training-only
  profile-likelihood search. Keep manual selection and show the powers and
  comparison before accepting a choice.
- **Model selection:** assess a one-standard-error rule for the GLM penalty and
  a raw linear trend alongside binned step effects.
- **Residual searches:** review family-specific variance and dispersion in
  Gamma, Tweedie and Poisson searches. Test ranking and calibration on known
  examples before changing the method.

## Workflow and assurance

- Make time-based and grouped validation equally clear across the Python API,
  exports and workbench.
- Check that exported Python retains final roles, renames, recodes, derived
  columns and split settings together.
- Make leakage review treat absent, unassigned and ignored roles consistently,
  while keeping reserved-column exclusions and any use of an unassigned column
  explicit.
- Improve messages for invalid Gamma and inverse-Gaussian evaluation targets
  so affected rows and their treatment are visible.

## Worked examples

- Add an external-offset workflow.
- Add examples for wide datasets, predictor screening, comparisons of step
  and linear effects, and monotonic constraints.
- Add a keyed lookup workflow using public data.

Each example should use short public calls, make its modelling decisions
explicit and run from a standard installation. Exported scoring should reproduce
the accepted model on new data.
