# NRGNN + BallAux Seed0 ROCm Report

- Date: 2026-06-10
- Python: `.venv-rocm/Scripts/python.exe` under the GRACE-master ROCm environment.
- Device: `cuda:0` reported by ROCm PyTorch, AMD Radeon RX 7700 XT.
- Protocol: NoisyGL percent split 0.2/0.1/0.7, uniform noise 0.4, seed 0.
- Variant: keep NRGNN unchanged and add `lambda_ball * CE_soft(model_logits, frozen_GRACE_ball_target)` on train nodes.
- Ball target: SoftDist target with multiclass alpha and ball label distribution.

## Seed0 Accuracy

| Dataset | NRGNN | lambda | NRGNN+BallAux | Shuffled BallAux | Real vs NRGNN | Real vs Shuffled |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.8311 | 0.01 | 0.8311 | 0.8311 | +0.0000 | +0.0000 |
| Cora | 0.8311 | 0.05 | 0.8306 | 0.8375 | -0.0005 | -0.0069 |
| Cora | 0.8311 | 0.10 | 0.8322 | 0.8369 | +0.0011 | -0.0047 |
| CiteSeer | 0.7113 | 0.01 | 0.7109 | 0.7113 | -0.0004 | -0.0004 |
| CiteSeer | 0.7113 | 0.05 | 0.7131 | 0.7109 | +0.0018 | +0.0022 |
| CiteSeer | 0.7113 | 0.10 | 0.7135 | 0.7113 | +0.0022 | +0.0022 |

## Readout

- The auxiliary ball target does not give a clear gain over NRGNN on Cora. At larger lambda, shuffled ball targets outperform real ball targets, so the Cora gain is not attributable to meaningful GRACE-ball structure.
- CiteSeer has a small positive signal at `lambda=0.05` and `0.10`, but the gain is only about `+0.2pp` on seed0.
- This is not strong enough to justify a 5-seed expansion yet.
- The result suggests NRGNN already captures most useful denoising signal; simply adding frozen ball soft targets as an auxiliary train-node loss is too weak or too redundant.

## Decision

Do not promote `NRGNN+BallAux` as the next main method in its current form.

If continuing route B, the next version should use balls to modulate NRGNN's online mechanism rather than adding a small independent auxiliary loss. Candidate changes:

| Candidate | Reason |
|---|---|
| use ball confidence to reweight NRGNN pseudo-label loss `loss_add` | directly affects NRGNN's denoising component |
| use ball agreement to filter `idx_add` pseudo-label nodes | avoids adding unreliable pseudo edges |
| fuse ball target only for NRGNN-selected high-confidence unlabeled nodes | targets the online propagation path instead of train CE |
| compare against shuffled at every step | required because Cora shuffled can outperform real in this auxiliary setup |
