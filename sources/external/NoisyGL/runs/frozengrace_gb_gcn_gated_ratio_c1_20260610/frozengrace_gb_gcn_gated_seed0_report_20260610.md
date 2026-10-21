# Frozen-GRACE GB-GCN Gated Seed0 Smoke

- Date: 2026-06-10
- Protocol: NoisyGL percent split 0.2/0.1/0.7, uniform noise 0.4, seed 0, CPU.
- Gated variant: hard replacement only when `min_train_per_ball=3` and ball noisy-label majority confidence `>=0.7`.
- Boundary: single-seed smoke only; clean labels are used only for diagnostics, not training.

## Research Record

- Hypothesis: hard ball-majority correction loses on Cora mainly because CDR is too high; raising `min_train_per_ball` and adding a majority-confidence gate should reduce CDR while keeping high-confidence NCR.
- Granular ball role: frozen-GRACE embedding space provides local semantic neighborhoods; balls are used only as a label-preprocessing teacher for a fresh end-to-end GCN.
- Code/config change: added `min_majority_confidence` to frozen-GRACE target construction and added `frozengrace_gb_gcn_gated` / `frozengrace_gb_gcn_gated_shuffled` configs.
- Expected mechanism: skip ambiguous balls, avoid overwriting many originally clean train labels.
- Success standard for this smoke: test accuracy should at least match GCN+CE while CDR drops clearly from the hard-replacement variant.
- Failure standard: NCR collapses so much that the model becomes equivalent to plain GCN+CE.

## Test Accuracy

| Dataset | GCN+CE | Hard Real | Hard Shuffled | Gated Real | Gated Shuffled |
|---|---:|---:|---:|---:|---:|
| Cora | 0.8100 | 0.7947 | 0.3953 | 0.8137 | 0.8100 |
| CiteSeer | 0.6327 | 0.6830 | 0.3222 | 0.6314 | 0.6327 |

## Diagnostics

| Dataset | Variant | Changed Train | NCR | CDR | Eligible Train | Conf-Pass Train | Conf-Pass Balls |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | hard real | 246 | 0.6878 | 0.2244 | 541 | 541 | 52 |
| Cora | gated real | 22 | 0.0794 | 0.0057 | 541 | 107 | 11 |
| Cora | gated shuffled | 0 | 0.0000 | 0.0000 | 541 | 0 | 0 |
| CiteSeer | hard real | 358 | 0.5911 | 0.2980 | 661 | 661 | 62 |
| CiteSeer | gated real | 10 | 0.0112 | 0.0101 | 661 | 39 | 6 |
| CiteSeer | gated shuffled | 0 | 0.0000 | 0.0000 | 665 | 0 | 0 |

Notes:

- Cora hard correction changed 246 / 541 train labels and had high NCR, but CDR was 22.44%, so it finished 1.53pp below GCN+CE.
- Cora gated correction changed only 22 labels; CDR dropped to 0.57%, and seed0 test accuracy moved to 0.8137, 0.37pp above GCN+CE.
- CiteSeer hard correction was the strongest seed0 result because its NCR gain covered the high CDR.
- CiteSeer gated correction became too conservative: only 39 train nodes passed the confidence gate and only 10 labels changed, so the previous +5.03pp gain disappeared.
- Gated shuffled control made no corrections and matched GCN+CE, which is expected under this strict gate.

## Decision

The diagnosis is confirmed but the `min_train_per_ball=3, tau=0.7` gate is too strict for CiteSeer. It solves the Cora CDR problem but also collapses NCR, especially on CiteSeer. Do not expand this exact setting to 5 seeds yet.

Recommended next smoke grid:

| Setting | Purpose |
|---|---|
| `min_train_per_ball=3, tau=0.60` | recover more NCR while keeping some CDR control |
| `min_train_per_ball=3, tau=0.65` | middle point between hard and current strict gate |
| `min_train_per_ball=4, tau=0.60` | test whether support count is more useful than high purity |

Only move to 5 seeds if one setting keeps Cora at or above GCN+CE and keeps CiteSeer close to the hard-correction gain.
