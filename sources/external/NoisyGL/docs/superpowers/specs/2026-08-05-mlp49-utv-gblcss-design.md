# MLP-49 UTV-GB-LCSS design

## Question

Can a robust graph teacher improve LOO-GB-LCSS by verifying when its fixed granular-ball consensus should be trusted, without changing the frozen representation, partition, local target content, or X-only student?

MLP-46/48 closed direct noisy-label guidance of ball construction. MLP-47 nevertheless showed that an NRGNN teacher carries complementary information at high noise. MLP-49 therefore leaves Stage 2 label-free and uses the teacher only as a reliability signal in Stage 3.

## Frozen components

- frozen, row-normalized GRACE representation `H`;
- recursive 2-means partition with `max_balls=64`, `min_ball_size=32`, and `split_improvement=0.05`;
- leave-one-out voting and `tie_mode=distribution_only`;
- `target_beta=0.5`, multiclass concentration alpha, train-only noise, and the existing X-only MLP;
- percent split `0.2/0.1/0.7` and downstream seed pairing.

## Candidate target

For training node `v` in ball `B_j`, the formal method computes a leave-one-out distribution `p_{j,-v}`, a tie-safe semi-hard local target `q_{j,-v}`, and concentration `alpha_{j,-v}`.

An independently trained NRGNN provides `M=8` MC-dropout posterior samples for every node on its validation-selected graph. Reliability of non-training node `u` is

`c_u = (1 - H(mean(pi_u))/log(C)) * (1 - mean_m JSD(pi_u^m, mean(pi_u))/log(2))`.

Only non-training nodes in the same ball are aggregated:

`r_j = sum_{u in B_j minus V_L} c_u mean(pi_u) / sum c_u`.

The teacher verifies, but does not replace, ball consensus:

`a_{j,-v} = 1 - JSD(q_{j,-v}, r_j)/log(2)`,

`alpha'_{j,-v} = alpha_{j,-v} * a_{j,-v} * (1 - H(r_j)/log(C))`,

`t_v = (1-alpha'_{j,-v}) onehot(noisy_y_v) + alpha'_{j,-v} q_{j,-v}`.

If a ball has no reliable non-training teacher mass, `alpha'=0`. This conservative fallback prevents unverified correction.

## Controls

1. formal LOO-GB-LCSS;
2. UTV-GB-LCSS;
3. size-matched shuffled-ball UTV using the same teacher;
4. teacher posterior target on training nodes;
5. teacher-confidence-weighted noisy one-hot control;
6. clean-oracle verifier, diagnostic only.

## Phase A protocol

- datasets: Cora, CiteSeer, CoraFull;
- noise: uniform 0.4 and 0.6, train only;
- seed: 0;
- no downstream MLP training;
- metrics: target accuracy, Brier score, NCR, CDR, alpha/verification coverage, teacher entropy and stochastic stability;
- the NRGNN teacher is retrained in every dataset-rate cell and never reads clean labels for training.

## Preregistered gate

Phase B is triggered only if all conditions hold:

1. all 6 cells pass split/noise/leakage and probability integrity checks;
2. UTV has lower Brier than both formal LOO and shuffled UTV in at least 4/6 cells;
3. mean UTV target-accuracy delta over formal LOO is at least +0.5 percentage points;
4. no cell loses more than 1 percentage point in target accuracy;
5. UTV has lower Brier than shuffled UTV in at least 4/6 cells and nonzero verified correction coverage in every cell;
6. clean-oracle verifier improves Brier over formal LOO in at least 3/6 cells, establishing verifier headroom.

Failure closes this verifier formulation. It does not authorize threshold tuning, teacher replacement, partition changes, or a second Phase A rule.
