# Pricing walkthrough verification

Verified on 23 September 2026. Release remains paused.

The current [walkthrough](../examples/pricing_walkthrough.md) completed in a real
Jupyter kernel using a clean installation of the unreleased v0.472 candidate.
All 31 Python blocks ran unchanged. Charts and tables were displayed normally;
the loader and display methods were not replaced with test substitutes.

## Results

- **Data:** 50,000 French motor rows loaded through `load_external_dataframe`;
  35,000 training rows and 15,000 holdout rows. A separate uncached download also
  succeeded and returned the complete 677,991-row dataset.
- **Setup:** ignored `VehGas` stays excluded; saved settings restore the same
  split memberships, numeric cuts and categorical treatment.
- **Fitting:** the two-factor GLM, expanded four-factor GLM, residual searches
  and two ordered CatBoost interaction tables all completed.
- **Amendments:** both keeping later tables fixed and refitting them completed.
  Earlier models kept their original tables. Every checkpoint retained the same
  disjoint training and holdout groups.
- **Validation:** explicit holdout checks completed. Predictions were finite
  and positive for every checkpoint.
- **Saved model:** reopening reproduced all 50,000 predictions exactly.
- **Excel:** a scorer rebuilt from the workbook's stored base rate, metadata,
  main tables and pair tables reproduced predictions to floating-point precision.
  The largest absolute difference was **3.33 × 10⁻¹⁶**.
- **Displays:** 57 HTML outputs were captured without errors. The actual A/E
  chart and interaction heatmap were also opened and inspected in a browser.
- **Tests:** 27 pricing API tests passed against the clean installed wheel.
  Both strengthened walkthrough tests passed against source, including Markdown
  and companion-script consistency. Black, Ruff and whitespace checks passed.

Independent Astra and actuarial reviews found no remaining blockers. The optional
examples for an existing split, changed bands, removing a factor, restoring
settings and rebalancing were also exercised on a smaller dataset.

No package implementation change was needed during this verification. The Excel
test reader was corrected to preserve numeric precision and handle blank cells.

## Environment and evidence

- macOS, Python **3.13.15**, isolated virtual environment without system packages.
- Installed the candidate wheel with its normal dependencies; added Jupyter and
  pytest separately as verification tools. `pip check` passed.
- The notebook imported the installed package, not the development source tree.
- Focused source tests also passed in the existing Python 3.14 environment.
- Complete notebook and artifact checks: **75.01 seconds** on this machine.

Walkthrough SHA-256:
`cc9b924c2958b029b670f97f61707b28ef4b96811132267ea4ebf7fc4e6d945c`

Candidate wheel SHA-256:
`18eee8abef4acaacdd25a5f888ad462d9578f20d2a494e338bc00a030c33c31a`

Local evidence is retained under
`/private/tmp/easyglm-walkthrough-check-ymc24ord/`:

- `executed_walkthrough.ipynb`: executed blocks and their real outputs.
- `run_metadata.json`: guide hash, block count, duration and zero execution errors.
- `notebook-output/verification.json`: row counts and scoring differences.
- `notebook-output/motor_pricing_tables.xlsx`: actual exported workbook.
- `notebook-output/motor_pricing_model.easyglm`: actual saved model.
- `notebook-output/motor_settings.json`: actual saved settings.
- `notebook-output/driver_age_ae.html` and `interaction_rates.html`: inspected charts.
- `pricing-tests.xml`, `installed-versions.txt` and install logs.

This verifies the local release candidate on macOS. It is not a Windows execution
test or a check of a published v0.472 installation. Nothing was released or pushed.
