# Posterior-Calibrated SoftDist Experiment Design

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-06-22
- Verification Status: UNVERIFIED
- Version Label: posterior_softdist_design_v1

## Research Question

Does the current SoftDist confidence over-trust granular balls with very few labeled
training nodes, and can a fixed Jeffreys-prior posterior shrinkage improve target
quality and downstream robustness without reducing effective smoothing coverage?

## Hypothesis

The current confidence depends only on the largest observed class proportion. It
therefore assigns maximal confidence to a `2:0` vote although that evidence is less
reliable than a larger vote. If low-support balls have worse clean-label damage or
clean-mass behavior, shrinking the ball distribution with a fixed `1/2` pseudo-count
per class should reduce false confidence while converging to the current SoftDist as
support grows.

## Scope

### Included

- Datasets: Cora, CiteSeer, Coauthor-CS.
- Split: percent `0.2/0.1/0.7`.
- Noise: train-only uniform noise `0.4`.
- Seeds: `0,1,2,3,4` for the diagnostic and any downstream confirmation.
- Frozen GRACE embeddings and the existing dataset-specific ball parameters.
- Train-mask-only votes; clean labels are diagnostic-only.
- Existing GCN `[X||H]` downstream protocol.

### Excluded

- Anchor nodes, graph rewiring, LCC, model-prediction feedback, pseudo-label
  expansion, adaptive purity-driven splitting, and hyperparameter sweeps.
- Pair noise, instance noise, additional datasets, and alternative backbones in the
  first experiment.

## Stage A: Support-Bin Diagnostic

For every eligible geometric ball, group its training-label support into:

- `2-3`
- `4-7`
- `8+`

For training nodes in each bin, report:

- number of balls and training nodes;
- noisy-node count and clean-node count;
- pseudo-label NCR and CDR;
- mean alpha;
- total and per-node clean-mass delta;
- changed-node clean-mass delta;
- true majority accuracy as a diagnostic;
- mean observed margin.

### Go/No-Go Rule

Proceed only if the pooled five-seed evidence shows that the `2-3` support bin is
materially less reliable than `8+` on at least two of the three datasets, using either:

1. CDR at least `0.05` higher; or
2. per-node clean-mass delta at least `0.02` lower.

The direction must hold in at least `4/5` paired seeds on each passing dataset.
If the `8+` bin is absent, compare `2-3` with `4-7` using the same thresholds.

## Stage B: Posterior-Calibrated Target

Use a fixed Jeffreys prior:

```text
posterior_count_bc = count_bc + 0.5
posterior_dist_bc = posterior_count_bc / (n_b + C / 2)
posterior_alpha_b =
    clip((max(posterior_dist_b) - 1/C) / (1 - 1/C), 0, 1)
q_i = (1 - posterior_alpha_b) * one_hot(noisy_y_i)
      + posterior_alpha_b * posterior_dist_b
```

No parameter is tuned. The current SoftDist remains unchanged and serves as the
anchor.

## Downstream Comparisons

1. `SoftDist`: current residual mixture.
2. `Posterior-SoftDist`: Jeffreys-prior residual mixture.
3. `Posterior-SoftDist shuffled`: size-matched random-ball control.
4. `DirectDist`: GBmooer-style complete replacement by the empirical ball
   distribution (`alpha=1` for eligible balls).

All methods use the same split, noisy labels, frozen `H`, ball parameters, GCN
architecture, optimizer, and seeds.

## Success Criteria

- CiteSeer: Posterior-SoftDist minus SoftDist is at least `+0.008` mean accuracy and
  positive in at least `4/5` paired seeds.
- Cora and Coauthor-CS: neither drops by more than `0.003` mean accuracy.
- Posterior-SoftDist exceeds its shuffled control with positive mean delta and at
  least `4/5` positive paired seeds on all three datasets.
- Effective smoothing coverage does not fall by more than `0.02` absolute relative
  to SoftDist.
- Mean alpha may decrease, but the number of eligible/alpha-positive training nodes
  must remain unchanged unless posterior alpha is exactly zero by construction.

## Failure Criteria

- Stage A gate fails.
- CiteSeer gain is below `0.008` or has fewer than four positive paired seeds.
- Either Cora or Coauthor-CS drops by more than `0.003`.
- Real posterior balls do not consistently beat shuffled balls.
- Apparent improvement is explained mainly by removing smoothing coverage.

## Outputs

- `runs/posterior_softdist_20260622/support_bin_raw.csv`
- `runs/posterior_softdist_20260622/support_bin_aggregate.csv`
- `runs/posterior_softdist_20260622/support_bin_report.md`
- If Stage A passes:
  - `runs/posterior_softdist_20260622/downstream_raw.csv`
  - `runs/posterior_softdist_20260622/downstream_aggregate.csv`
  - `runs/posterior_softdist_20260622/paired_deltas.csv`
  - `runs/posterior_softdist_20260622/posterior_softdist_report.md`

## Claim Boundary

A successful result supports only that support-aware posterior shrinkage improves
the current SoftDist target under the tested graph, split, noise, representation,
and downstream protocol. It does not establish Bayesian optimality, universal
model independence, superiority to NRGNN, or validity under pair/systematic noise.

