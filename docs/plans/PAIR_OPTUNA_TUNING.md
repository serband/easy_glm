# Automatic tuning for pair corrections

The pair editor should ask for two predictors, then choose CatBoost parameters
automatically. The main GLM remains frozen and every later pair is fitted against
the earlier deployed tables. This change does not alter scoring or table axes.

## Search and validation

- Use a seeded, serial Optuna TPE study with eight trials by default and three
  startup trials. Search depth 2–5, 40–160 trees in steps of 20, learning rate
  0.03–0.15 and L2 regularisation 0.1–20 (the last two on logarithmic scales).
- Score each trial on the same five training-data folds, using the deployed
  table's weighted deviance. Evaluate no correction separately and prefer it on
  numerical ties. Do not select on the raw CatBoost teacher's performance.
- Prepare each fold's earlier-stage prediction once and reuse it across trials.
  When earlier automatic pairs exist, choose their complete parameter configuration in an
  independent four-trial study within that outer training partition, using five
  inner folds and two startup trials. Include an all-neutral baseline. Prefixes
  containing only fixed-candidate stages retain their bounded fixed search.
- Never initialise fold-local studies from full-training winners, losses or study
  histories. Each inner fit creates its own axes and main fitting procedure from
  its training rows. This is bounded nested validation, without recursive tuning.
- Adaptive trials use validation scores to choose subsequent parameters. These
  are training-CV selection scores, not unbiased estimates of final performance.
  The untouched holdout remains the assessment set.

## Compatibility and runtime

Existing projects with explicit fixed candidates retain that fitting procedure.
New UI stages use automatic tuning. The selected numerical parameters, trial
results and fold-prefix parameters are plain records; scoring artefacts need
neither Optuna nor CatBoost. Python workflow exports retain the search request.

Search settings, search-space version, seeds and Optuna version participate in
fit/cache identity.

Two raw inputs keep individual teacher fits small, but five-fold validation and
earlier-stage tuning multiply the workload. Keep bounded trial budgets, resource
preflight and the cooperative fit deadline. Measure actual default workloads
before quoting runtime; previous fixed-candidate timings do not establish Optuna
timings.

## Acceptance checks

1. Automatic search round-trips through project JSON and Python workflow export;
   fixed-candidate projects continue to work.
2. Every completed trial is evaluated through deployed tables on five folds;
   the neutral candidate remains eligible.
3. Changing holdout rows changes no tuning trace or table. Poisoning an outer
   validation fold leaves that fold's prior-prefix study unchanged. Current-stage
   adaptive trial suggestions may change because they legitimately use CV scores.
4. Identical seeds reproduce search parameters; unchanged stages reuse their
   tables and changing a stage invalidates only its suffix.
5. Scoring in a fresh process imports neither Optuna nor CatBoost and reproduces
   saved predictions. Failed or cancelled studies publish no partial model.
6. The UI shows automatic tuning and useful progress without numbered candidate
   editors. Validate both narrow and wide layouts and measure a two-stage fit.
