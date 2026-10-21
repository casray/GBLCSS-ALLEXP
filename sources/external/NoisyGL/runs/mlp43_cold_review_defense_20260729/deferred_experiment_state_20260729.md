# MLP-43 Deferred Experiment Closure

Started: 2026-07-29

Completed: 2026-07-30

## Original reason for pause

The user requested completion of the non-experimental manuscript revision before
continuing new training runs. The running representation-source job was stopped
after flushing its completed rows. No partial result is used in the manuscript.

## Representation-source experiment

### Research hypothesis

The local-consensus effect may depend on graph-derived GRACE geometry rather
than raw feature geometry alone.

### Granular-ball role

The granular-ball constructor and downstream target rule remain fixed. Only the
representation used to form ball membership changes.

### Frozen protocol

- uniform 0.4 train-only noise
- percent split 0.2/0.1/0.7
- `tie_mode=distribution_only`
- `max_balls=64`, `min_ball_size=32`, `split_improvement=0.05`
- `target_beta=0.5`

### Final completed rows

- Seed-0 geometry and downstream rows exist for raw X and two-step adjacency
  propagation on Cora, CiteSeer, CoraFull, Coauthor-CS, and Amazon-Photo.
- Seed-0 supervised-GCN-hidden diagnostics exist for Cora and CiteSeer.
- The non-collapsed two-step adjacency arm has completed seeds 0-4 on
  Coauthor-CS and Amazon-Photo.
- The raw file contains 20 audited rows. The pre-registered geometric gate
  stops one-ball label-free sources at seed 0 and expands non-collapsed
  sources to five seeds.

### Final decision

Complete. The split/noise-hash audit passed. Raw X collapses to one ball on all
five datasets. Two-step adjacency propagation also collapses on Cora,
CiteSeer, and CoraFull. It remains non-collapsed on Coauthor-CS and
Amazon-Photo but trails GRACE by 1.37 and 0.92 percentage points,
respectively. The result is reported as a comparison among tested
representation sources, not as proof that GRACE is uniquely optimal.

## Corrected kNN-teacher control

### Research hypothesis

The Table 5 kNN-teacher control should remain competitive after removing the
query node from its own neighborhood.

### Code and configuration state

- `exclude_self=true` has been added to all `mlp_distill_knn_xh` configs.
- The focused configuration test passes.
- The corrected downstream grid contains 30/30 successful rows.

### Final decision

Complete. Cora, CiteSeer, and CoraFull were rerun under uniform 0.4 and 0.6
for seeds 0-4. Every corrected row records `exclude_self=1`,
`target_beta=0`, train-only noise, and zero validation/test modifications.
The other four arms were reused only after per-seed split/noise hashes matched.
The corrected kNN teacher exceeds the ball teacher in all six aggregate cells
by 0.25--1.01 percentage points. The five-arm comparison is restored in v36.
