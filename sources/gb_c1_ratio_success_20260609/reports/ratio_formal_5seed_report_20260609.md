# Formal Ratio-Split C1 Validation

Date: 2026-06-09

Setting:

- Datasets: Cora, CiteSeer
- Split: stratified ratio split
- Train/val/test: 0.2 / 0.1 / 0.7
- Noise: symmetric 0.4 on train labels only
- Seeds: 0, 1, 2, 3, 4
- Methods: `C1`, `C1_noball`, `C1_topk`, `C1_shuffled`
- Base config: `configs/stage_c_smoke.yaml`
- Results dir: `results_ratio_formal_20260609`

## Main Results

| Dataset | Method | Test Acc Mean | Test Acc Std | Linear Eval Mean | Linear Eval Std | Label Purity Mean | Correction Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | C1 | 0.7475 | 0.0178 | 0.8453 | 0.0078 | 0.8123 | 0.3861 |
| Cora | C1_noball | 0.5548 | 0.0274 | 0.8039 | 0.0062 | 0.7128 | 0.0000 |
| Cora | C1_topk | 0.5491 | 0.0183 | 0.8239 | 0.0077 | 0.7350 | 0.4870 |
| Cora | C1_shuffled | 0.5935 | 0.0310 | 0.8364 | 0.0036 | 0.7620 | 0.0759 |
| CiteSeer | C1 | 0.5713 | 0.0263 | 0.6824 | 0.0129 | 0.6867 | 0.2707 |
| CiteSeer | C1_noball | 0.4235 | 0.0105 | 0.6106 | 0.0105 | 0.5970 | 0.0000 |
| CiteSeer | C1_topk | 0.4108 | 0.0190 | 0.6595 | 0.0082 | 0.6321 | 0.3789 |
| CiteSeer | C1_shuffled | 0.4473 | 0.0181 | 0.6669 | 0.0102 | 0.6494 | 0.0511 |

## Paired Deltas: C1 Minus Controls

| Dataset | Control | Mean Delta | Std Delta | Positive Seeds | Min Delta | Max Delta |
|---|---|---:|---:|---:|---:|---:|
| Cora | C1_noball | 0.1927 | 0.0176 | 5/5 | 0.1740 | 0.2177 |
| Cora | C1_topk | 0.1984 | 0.0152 | 5/5 | 0.1740 | 0.2114 |
| Cora | C1_shuffled | 0.1540 | 0.0240 | 5/5 | 0.1270 | 0.1887 |
| CiteSeer | C1_noball | 0.1478 | 0.0367 | 5/5 | 0.1013 | 0.1845 |
| CiteSeer | C1_topk | 0.1605 | 0.0328 | 5/5 | 0.1356 | 0.2146 |
| CiteSeer | C1_shuffled | 0.1240 | 0.0306 | 5/5 | 0.0936 | 0.1657 |

## C1 Ball Vote Coverage

Coverage is computed on final C1 embeddings and reliable balls.

| Dataset | Reliable Balls Mean | Avg Train / Reliable Ball | Reliable Balls With 0 Train | With 1 Train | With >=2 Train | With >=3 Train | With >=5 Train |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 175.4 | 2.77 | 25.0 | 47.0 | 103.4 | 65.4 | 27.0 |
| CiteSeer | 228.4 | 2.59 | 39.4 | 53.4 | 135.6 | 85.8 | 35.8 |

## Direct Findings

1. C1 wins all controls on both datasets.
   - Cora: 0.7475 vs 0.5548 / 0.5491 / 0.5935.
   - CiteSeer: 0.5713 vs 0.4235 / 0.4108 / 0.4473.

2. C1 has positive paired deltas for every seed.
   - Cora: 5/5 positive vs each control.
   - CiteSeer: 5/5 positive vs each control.

3. Real GB structure matters.
   - C1 beats `C1_shuffled` by +15.4pp on Cora and +12.4pp on CiteSeer.
   - This supports a real spatial GB contribution rather than generic regularization or random grouping.

4. kNN correction is not enough.
   - `C1_topk` has high correction metrics in some settings, but its classifier accuracy is weak.
   - C1 beats topk by +19.8pp on Cora and +16.1pp on CiteSeer.

5. The 20% ratio split provides enough local train-label coverage for C1 to function.
   - Reliable balls have about 2.6 to 2.8 train nodes on average.
   - Many reliable balls still have sparse labels, but the coverage is far better than the Planetoid standard split failure case.

## Interpretation

The medium-label protocol validates the central mechanism:

`GRACE representation -> granular balls -> train-only ball majority correction + GB structure regularization`

works when local labeled coverage is sufficient.

This does not contradict the earlier Planetoid standard split failure. Instead, it explains it:

- In the standard low-label split, most balls have no or only one train label, so majority voting cannot estimate a useful ball label.
- In the 20% ratio split, enough balls contain multiple train labels, so ball-level denoising becomes meaningful.

## Claim Boundary

Supported claim:

> Granular balls can serve as local noisy-label correction units in graph contrastive representations when local labeled coverage is sufficient.

Not supported:

> C1 solves the standard Planetoid low-label split.

Not supported yet:

> C1 is broadly robust across noise rates, split ratios, or larger graph datasets.

## Recommended Next Step

Keep `train_ratio=0.2, val_ratio=0.1` as the primary medium-label protocol and add:

1. noise rates 0.2 and 0.4;
2. optional 0.1 ratio as a coverage sensitivity setting;
3. concise comparison against external noisy-label baselines only if they are rerun under the same split/noise/seed protocol.

