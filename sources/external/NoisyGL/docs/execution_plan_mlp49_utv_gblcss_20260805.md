# MLP-49: robust-teacher-verified LOO-GB-LCSS

## Hypothesis

The fixed GRACE granular balls contain useful but uneven local label consensus. A robust NRGNN teacher can identify which ball-level corrections are trustworthy by checking agreement on non-training nodes, while leaving the granular-ball target itself as the correction content.

## Granular-ball role

The ball remains the fixed, label-free, mutually exclusive local semantic unit. Teacher predictions neither create, split, merge, nor relabel balls. They only attenuate the formal leave-one-out correction weight.

## Motivation and prior evidence

- MLP-46/48: direct noisy-label guidance of ball construction failed, including negligible clean-oracle partition headroom.
- MLP-47: NRGNN teacher knowledge is complementary at high noise.
- MgCNL and BNO-GB motivate separating local consensus content from reliability estimation, but their specific algorithms are not copied.

## Changes

- New isolated target module and tests.
- New standalone Phase A runner and output directory.
- No modification to the formal GB-LCSS predictor, NRGNN training algorithm, GRACE embeddings, split, noise generator, or paper results.

## Success and failure

The exact Phase A gate is frozen in the design document and runner before any formal result is generated. Passing permits a downstream seed-0 smoke. Failing closes UTV-GB-LCSS and records the teacher verifier as non-complementary to formal LOO confidence.

## Outputs

`runs/mlp49_utv_gblcss_20260805/` must contain raw target metrics, teacher reliability summaries, protocol audits, completeness, teacher posterior caches, the gate decision, command manifest, and a detailed report.

## Result and decision

- Phase A completed all `3 datasets x 2 noise rates x 6 target arms = 36` rows with six NRGNN teacher caches.
- All split, train-only-noise, probability, cache-hash, and completeness audits passed.
- UTV was better than both formal LOO and shuffled UTV in Brier on `0/6` cells; mean target-accuracy delta versus formal LOO was `-1.335pp`.
- UTV had worse Brier than formal LOO on every cell (`+0.1454` to `+0.3474`). Cora u0.4 and CiteSeer u0.4 lost `6.470pp` and `2.256pp` target accuracy.
- Real UTV balls beat shuffled UTV balls in Brier on `6/6`, so the failure is not absence of geometric information.
- Clean-oracle verification improved formal LOO Brier on `0/6`, showing that the multiplicative attenuation rule lacks verifier headroom.
- Decision: `abandon` this verifier formulation. Do not run downstream Phase B or tune another threshold/product. Preserve direct NRGNN teacher knowledge as a separate distillation signal.

Evidence: `runs/mlp49_utv_gblcss_20260805/mlp49_report.md` and `phase_a_gate.json`.
