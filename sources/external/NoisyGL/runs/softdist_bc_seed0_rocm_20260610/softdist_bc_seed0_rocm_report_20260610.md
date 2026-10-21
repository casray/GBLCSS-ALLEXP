# SoftDist + Ball Consistency Seed0 ROCm Report

- Date: 2026-06-10
- Python: `.venv-rocm/Scripts/python.exe` under the GRACE-master ROCm environment.
- Device: `cuda:0` reported by ROCm PyTorch, AMD Radeon RX 7700 XT.
- Protocol: NoisyGL percent split 0.2/0.1/0.7, uniform noise 0.4, seed 0.
- Variant: SoftDist target plus ball consistency regularization over all nodes.
- Config: `lambda_bc=0.1`, `bc_min_ball_size=2`.
- Boundary: seed0 smoke only.

## Method

The training objective is:

```text
L_total = L_soft_target + lambda_bc * L_ball_consistency
```

`L_ball_consistency` penalizes the variance of GCN prediction distributions inside each frozen GRACE ball:

```text
preds_B = softmax(logits[members_of_B])
mean_B = mean(preds_B)
L_B = mean(sum((preds_B - mean_B)^2))
```

The regularizer uses node predictions and ball membership only. It does not use clean labels.

## Seed0 Test Accuracy

| Dataset | GCN+CE | NRGNN | SoftDist Static | SoftDist BC | SoftDist BC Shuffled |
|---|---:|---:|---:|---:|---:|
| Cora | 0.7974 | 0.8311 | 0.8042 | 0.8032 | 0.8016 |
| CiteSeer | 0.6435 | 0.7113 | 0.6748 | 0.6748 | 0.6456 |

## Readout

- `lambda_bc=0.1` does not improve the real-ball model on seed0.
- Cora slightly drops from static SoftDist `0.8042` to `0.8032`.
- CiteSeer stays exactly at static SoftDist `0.6748`.
- The shuffled controls drop compared with static shuffled, especially on CiteSeer, so the regularizer is not simply helping every assignment. But it is not adding positive signal to the real-ball model yet.
- This setting should not be expanded to 5 seeds.

## Decision

The ball-consistency idea remains plausible, but the first plain variance regularizer with `lambda_bc=0.1` is not enough.

Useful next checks, if continuing this line:

| Candidate | Reason |
|---|---|
| lower `lambda_bc` such as `0.01` | avoid over-smoothing confident class predictions inside noisy balls |
| entropy-gated BC | apply consistency only to low-entropy balls or high-confidence ball distributions |
| train-node-only BC vs all-node BC | isolate whether unlabeled/test-node smoothing is helping or hurting |
| pair BC with graph-neighbor consistency | compare semantic-ball smoothing against topology smoothing under the same loss form |
