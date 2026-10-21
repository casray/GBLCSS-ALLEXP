# MLP-25 Coarse-Ball Holdout Smoke

## Research hypothesis

The default small-graph groups contain only about ten noisy train-label votes. A moderate support size near 20 votes may reduce voting variance without collapsing semantic classes. This is a final smoke on the granularity axis; it must not block the frozen paper mainline.

## Granular-ball role

Groups remain label-blind partitions of frozen GRACE H. Only train noisy labels are counted after construction. Clean labels remain diagnostic-only.

## Five preregistered arms

1. `fixed`: `max_balls=64, min_ball_size=32, split_improvement=0.05`.
2. `vote_target`: `max_balls=min(64,ceil(|V_L|/20))`, with `32/0.05` unchanged.
3. `vote_target_shuffled`: size-matched random control for arm 2.
4. `candidate`: frozen MLP-13 candidate `128/64/0.01`.
5. `candidate_shuffled`: size-matched random control for arm 4.

The two shuffled controls cannot be shared because the candidates induce different group-size distributions.

## Protocol

- Holdouts: CiteSeerFull, Cora_ML, Attributed-Wiki.
- Split: percent 0.2/0.1/0.7.
- Noise: train-only uniform 0.4.
- Seed: 0 smoke only.
- Student: frozen SoftDist-MLP beta=0.5 implementation.
- GPU: ROCm `cuda:0`.

Vote-target caps are fixed by the split before execution: CiteSeerFull 43, Cora_ML 30, Attributed-Wiki 25.

## Seed-0 advancement gate

Each candidate is judged independently against fixed and its own shuffled arm. It advances only if all conditions hold:

- at least 2/3 datasets improve over fixed by at least 1pp;
- no dataset regresses by more than 0.5pp;
- at least 2/3 datasets beat the candidate's own shuffled arm by at least 1pp;
- no collapse, mean train votes per group at least 2, and nonzero target coverage;
- all split/noise hashes match and val/test labels remain unmodified.

Passing seed0 means only “advance to a preregistered five-seed confirmation.” It does not authorize changing the default. If both candidates fail, the granularity axis closes and fixed 64 remains unchanged.

## Required outputs

All artifacts go to `runs/mlp25_coarse_holdout_smoke_20260712/`: raw rows, aggregate, gate decisions, report, stdout/stderr, and protocol audit evidence.
