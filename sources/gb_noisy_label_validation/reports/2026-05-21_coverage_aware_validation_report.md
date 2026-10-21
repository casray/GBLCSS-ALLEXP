# Coverage-aware GB Noisy-label Validation Report

## Experiment Type

This is a validation experiment, not a new downstream task. The task remains noisy-label node classification on frozen GRACE embeddings. The experiment tests whether granular balls in a stable GCL representation provide a better local smoothing signal than CE, global smoothing, kmeans smoothing, and topk matched neighbor smoothing.

## Hypothesis

Coverage-aware granular-ball smoothing should help if noisy train labels can be corrected by reliable local semantic regions in the frozen GRACE embedding space. The key question is whether confidence-weighted GB targets outperform ordinary local smoothers under the same noisy labels, splits, seeds, and classifier protocol.

## Granular-Ball Role

Granular balls are used only as classifier-stage local reliability units. They are not used as hard positive labels, hard negative filters, pretraining constraints, prototype pulling targets, or split/evaluation modifiers.

## Configuration

```text
datasets = Cora, CiteSeer
split_mode = ratio
train_ratios = 0.1, 0.2
val_ratio = 0.1
noise_type = symmetric
noise_rates = 0.2, 0.4
seeds = 0, 1, 2, 3, 4
mode = frozen
alpha = 0.3
methods = ce global kmeans topk_matched gb gb_prior gb_ca_ce gb_ca_global
```

Raw result rows are under `results/gate_runs/**/result_rows.csv`. The formal run wrote 40 result files, 64 summary rows, and 128 paired-delta rows.

## Mean Test Accuracy

| Dataset | Train | Noise | Best method | CE | kmeans | topk | gb | gb_prior | gb_ca_ce | gb_ca_global |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 0.1 | 0.2 | gb_prior | 0.7989 | 0.7974 | 0.7896 | 0.7970 | 0.8025 | 0.8004 | 0.7982 |
| Cora | 0.1 | 0.4 | kmeans | 0.7546 | 0.7567 | 0.7539 | 0.7542 | 0.7504 | 0.7563 | 0.7443 |
| Cora | 0.2 | 0.2 | gb_ca_ce | 0.8113 | 0.8110 | 0.8105 | 0.8108 | 0.8108 | 0.8115 | 0.8046 |
| Cora | 0.2 | 0.4 | gb | 0.7935 | 0.7945 | 0.7919 | 0.7974 | 0.7930 | 0.7950 | 0.7834 |
| CiteSeer | 0.1 | 0.2 | kmeans | 0.7048 | 0.7076 | 0.7045 | 0.7044 | 0.7059 | 0.7054 | 0.7053 |
| CiteSeer | 0.1 | 0.4 | global | 0.6876 | 0.6878 | 0.6891 | 0.6872 | 0.6878 | 0.6872 | 0.6894 |
| CiteSeer | 0.2 | 0.2 | gb_ca_global | 0.7064 | 0.7068 | 0.7001 | 0.7009 | 0.7067 | 0.7009 | 0.7093 |
| CiteSeer | 0.2 | 0.4 | kmeans | 0.6979 | 0.6987 | 0.6960 | 0.6976 | 0.6980 | 0.6979 | 0.6954 |

## Paired Delta Summary

`gb_ca_ce` is usually close to CE and kmeans on Cora, but its gains are small and not consistently positive across seeds. On CiteSeer, it fails against global or kmeans in several settings:

```text
CiteSeer train=0.1 noise=0.2: gb_ca_ce - kmeans = -0.0022, positive seeds = 1/5
CiteSeer train=0.1 noise=0.4: gb_ca_ce - global = -0.0026, positive seeds = 0/5
CiteSeer train=0.2 noise=0.2: gb_ca_ce - global = -0.0073, positive seeds = 1/5
CiteSeer train=0.2 noise=0.4: gb_ca_ce - kmeans = -0.0009, positive seeds = 2/5
```

`gb_ca_global` has one positive CiteSeer setting but is not stable:

```text
CiteSeer train=0.2 noise=0.2: gb_ca_global - global = +0.0011, positive seeds = 2/5
CiteSeer train=0.2 noise=0.4: gb_ca_global - kmeans = -0.0033, positive seeds = 2/5
Cora train=0.2 noise=0.4: gb_ca_global - kmeans = -0.0111, positive seeds = 1/5
```

## Diagnostics

The coverage-aware variants reached full train-node coverage, but the actual confidence weight was small:

```text
Cora gb_ca_ce alpha_mean_all_train: 0.0593 to 0.0774
Cora gb_ca_global alpha_mean_all_train: 0.0270 to 0.0440
CiteSeer gb_ca_ce alpha_mean_all_train: 0.0560 to 0.0678
CiteSeer gb_ca_global alpha_mean_all_train: 0.0231 to 0.0341
```

This means the new methods are mostly fallback targets with a weak GB correction. The result does not support the stronger claim that coverage-aware GB local smoothing is better than generic local smoothing.

## Interpretation

Supported by this experiment:

- Frozen GRACE embeddings plus GB smoothing can occasionally match or slightly improve CE/local baselines.
- Strict or prior GB still shows a Cora-positive signal in some medium-label high-noise settings.
- Coverage-aware fallback avoids catastrophic behavior but does not create a stable cross-dataset advantage.

Reasonable but unproven explanation:

- The confidence gate is too conservative, so GB evidence is diluted.
- CiteSeer local label distributions remain too noisy or fragmented for current GB construction.
- Ordinary kmeans/topk smoothers already capture much of the useful local smoothing signal.

Weakened direction:

- Presenting coverage-aware GB smoothing as a main method is not justified by this validation gate.
- The earlier Experiment 7 pattern is not solved: Cora has local positive signals, but CiteSeer remains unstable or baseline-competitive.

## Final Decision

Do not continue this exact `gb_ca_ce` / `gb_ca_global` design as the main method. Keep it as a negative validation experiment and use it to narrow the paper claim:

> In frozen GCL embedding space, granular balls can provide weak local reliability signals, but simple confidence-weighted label smoothing is not consistently stronger than generic local smoothing baselines.

The next experiment should not be another small alpha/fallback target variant. If continuing the noisy-label direction, the next test should either isolate when strict GB helps on Cora through diagnostics, or move GB from target replacement to sample weighting / confidence estimation where the mechanism is weaker and better aligned with current evidence.
