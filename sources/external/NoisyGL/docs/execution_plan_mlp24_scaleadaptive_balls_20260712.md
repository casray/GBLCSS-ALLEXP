# MLP-24 Scale-Adaptive Ball Count

## Hypothesis

The fixed `max_balls=64` cap may make large-graph local consensus groups too coarse. The single preregistered candidate is

\[
B_{\mathrm{eff}}=\max(64,\lceil\sqrt{|V|}\rceil).
\]

Only the cap changes. Frozen GRACE embeddings, `min_ball_size=32`, `split_improvement=0.05`, `beta=0.5`, MLP training, split, noise construction, and evaluation remain fixed.

## Role of granular balls

Granular balls remain label-blind local groups in frozen GRACE space. No clean label is used to build a group or a training target. No historical fixed-64 configuration is overwritten.

## Phases and preregistered decisions

1. Geometry: compare fixed, adaptive, and size-matched shuffled groups on all 11 datasets. Physics must exceed 64 groups, reduce mean group size by at least 25%, preserve at least two train votes per group, and remain within the runtime limit.
2. Seed-0 downstream: adaptive must improve Physics by 0.5pp over fixed, improve the seven affected datasets by 0.25pp on average, avoid a loss over 1pp on any current core large dataset, and beat shuffled on Physics by 1pp.
3. Five seeds: Physics must improve fixed and JoCoR by 1pp with at least four positive seeds; the affected-dataset mean improvement must be 0.5pp; at least five of seven datasets must be nonnegative; core losses may not exceed 0.5pp; at least six of seven datasets must beat shuffled by 1pp with four positive seeds.
4. Full spectrum runs only if all five-seed gates pass. The candidate replaces fixed 64 only if the final cross-noise gates pass.

## Required records

All outputs are stored under `runs/mlp24_scaleadaptive_balls_20260712/`, including geometry, raw downstream rows, aggregates, paired deltas, completeness, gate decisions, runtime, and the final report. Failed gates close the direction without tuning a second formula.
