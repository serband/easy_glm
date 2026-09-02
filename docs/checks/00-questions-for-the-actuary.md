# Questions for the actuary (from the 0.4 plan review)

Each question has the default we are building with. Answer any of them at any
time; the defaults are parameters, not architecture, so changing an answer later
is cheap.

| # | Question | Default we are using |
|---|----------|----------------------|
| Q1 | For a piecewise-linear factor (e.g. mileage), beyond the smallest and largest values seen in training: should the curve stay **flat**, or keep its last slope? | **Flat** (clamped at the training range). The clamp points are shown in the table. |
| Q2 | Where should relativity 1.00 sit on a continuous curve? | At the lower knot of the band with the most exposure, so it is a round, visible number. |
| Q3 | Should the lasso prefer curves with few **bends** (long straight sloped sections) or few **slopes** (flat wherever the data does not insist)? | Few bends (the AGLM hinge basis). Monotone constraints are therefore not offered on linear terms in 0.4. |
| Q4 | Minimum exposure for an interaction cell to get its own adjustment? | 0.5% of training exposure per interaction, editable. Cells below it show 1.00 with their exposure alongside. |
| Q5 | When `A × B` is added, the main-effect tables for A and B change (the split between mains and cells is not unique). Acceptable, or should mains be frozen and the interaction fitted as a second stage? | Joint fit; the before/after main tables are shown in the actuarial check. |
| Q6 | Rate-change setup (offset = log of current premium): should the export read as "multiplier on current premium" (base rate = overall change, relativities = differential changes)? | Yes. |
| Q7 | Binomial (e.g. lapse) models: export odds relativities with a label, or probabilities by band? | Odds relativities; the scorer returns probabilities. |
| Q8 | For frequency models, actual = Σ claims / Σ exposure and expected = Σ fitted claims / Σ exposure — confirm. | Yes; the rate/count flag is stored in the model file so the editor stops guessing. |
| Q9 | Which bike variables should be piecewise-linear rather than step? | Mileage only, as in the original script; everything else step. |
| Q10 | For a piecewise-linear factor, should the default **upper clamp** be the training maximum (the curve follows the data through thin tails) or a high quantile such as the 99.5th percentile (thin tails pooled flat, as a step design would)? See `docs/checks/b-linear.md` (BonusMalus 120–230). | Training maximum, rounded outward; the clamp is editable per factor on the Design page. |
