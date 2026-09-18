# Independent Astra implementation review

Reviewed 18 September 2026 in `/private/tmp/easyglm-feature-selection`.
This is a scoped technical acceptance record for the working diff, not release
approval or a claim that every proposed acceptance gate has passed. README was
not changed by this reviewer. Production corrections were made by Sol builders.

## Executed independent evidence

- Re-ran the numerical contract suite: 8 tests passed at the first review.
- Independently demonstrated CatBoost `allow_const_label=True` accepts constant
  zero and one targets with varying upstream baselines; the builder adopted it.
- Poisoned both outcomes and raw pair input A on outer validation fold zero of a
  two-pair model. Every recorded matching outer-fold training baseline, pair axis
  and table remained bit-identical (five records). Temporary driver:
  `/private/tmp/pair-review-leakage.py`.
- Changed the bins of pair-only input A with existing full/fold caches. After the
  correction, all 50 affected inner-prefix constructions were repeated instead
  of reusing old axes. Temporary driver: `/private/tmp/pair-review-cache.py`.
- Demonstrated that reversed numeric pair-axis intervals were initially accepted
  and mis-scored; subsequent validation rejects non-finite/non-increasing cuts.
- Re-ran focused engine, desktop and fitter regression tests after those fixes:
  14 tests passed in 10.24 seconds at that checkpoint. The later targeted fitter,
  desktop and numeric-axis suite passed 20 tests in 15.11 seconds.
- Executed `/private/tmp/pair-review-suffix.py` after the final desktop correction:
  changing C cuts when C occurs only in the second pair keeps mains and the first
  pair current, refits only the second pair, and lists only second-pair manual
  edits for clearing. Upstream pair edits remain intact.
- A separate adversarial actor executed an actual CatBoost-fold cancellation
  (`/private/tmp/pair-adversarial/cancel_mid_catboost.py`). The root agent reported
  HTTP cancellation in 0.002 seconds, worker exit in 0.057 seconds, preservation
  of the prior completed result and HTTP 409 for inapplicable/stale results. This
  reviewer read the driver; these timings are the other actor's execution, not
  a second independently repeated timing measurement.

## Reviewed corrections and contracts

Fold cache keys now include relevant pair-axis configuration, preparation/split
settings, algorithm identity and dependency signatures. Fitted tables carry
immutable input-prefix provenance, so same-axis refits cannot masquerade as the
same snapshot lineage. Completed desktop results are detached from mutable job
records, preserving the last completed prefix across invalidation/failure.

The frozen scorer and exported scoring script contain independent ordered pair
tables and do not require CatBoost for scoring. Training export explicitly uses
`replay_pair_adjustments=True`, applying stored exact-coordinate manual edits at
their own stages before later teachers are trained. Ordinary fresh refits refuse
to silently reuse the stage's own manual overrides. Desktop refit previews list
suffix edits to clear, worker input clears those edits, and published project
state removes them only after successful current-revision completion.

Pair edit requests now identify `stage_id` explicitly instead of overloading the
main-variable name. The synthetic `main` stage ID is reserved. Variables-role
removal and rename have focused backend/API regressions for pair-only parents.

Resource preflight runs before the main fit. It bounds stages/cells/candidates,
counts teacher and distinct fold-main fits, reports the original main CV/alpha
search, and estimates main-design and table bytes. The 3 GiB estimate is a
refusal gate, not an operating-system-enforced peak guarantee. The 900-second
limit is cooperative between native fits, starting before the main fit. Measured
workloads and those limitations are recorded separately in `RESULTS.md`.

## Scoped acceptance

The last desktop suffix-identity blocker is corrected: `_main_key` hashes only
main-relevant design, and each pair compares its own parent design to locate the
first affected stage. The independent executable probe above passed.

The reviewed statistical, cache, scoring, snapshot-identity and explicit-refit
contracts have no remaining blocker identified by this reviewer at this
checkpoint. The root agent owns final full-suite/browser evidence and review of
concurrent late UI/method-discriminator changes. This is scoped technical
acceptance, not blanket release approval. Million-row/eight-stage resource
guarantees remain outside the executed evidence; estimated memory and cooperative
time budgets must continue to be described accurately.
