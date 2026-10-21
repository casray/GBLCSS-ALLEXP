# SoftDist Dynamic Alpha Seed0 ROCm Report

- Date: 2026-06-10
- Python: `.venv-rocm/Scripts/python.exe` under the GRACE-master ROCm environment.
- Device: `cuda:0` reported by ROCm PyTorch, AMD Radeon RX 7700 XT.
- Protocol: NoisyGL percent split 0.2/0.1/0.7, uniform noise 0.4, seed 0.
- Boundary: seed0 smoke only.

## Variants

| Variant | Warmup | Refresh | Rule |
|---|---:|---:|---|
| SoftDist static | none | none | multiclass alpha + ball distribution |
| DynAlpha | 20 epochs | every 20 epochs | if GCN prediction agrees with ball majority, multiply alpha by `1.5`; otherwise multiply by `0.5` |
| DynAlpha Fast | 1 epoch | every epoch | same rule, but active before early checkpoint selection |

## Seed0 Test Accuracy

| Dataset | GCN+CE | SoftDist Static | DynAlpha | DynAlpha Shuffled | DynAlpha Fast | DynAlpha Fast Shuffled |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.7974 | 0.8042 | 0.8042 | 0.8047 | 0.7968 | 0.8058 |
| CiteSeer | 0.6435 | 0.6748 | 0.6748 | 0.6491 | 0.6710 | 0.6491 |

## Readout

- The conservative DynAlpha setting did not change the selected checkpoint. Best epochs were before the first dynamic refresh, so its final accuracy exactly matches static SoftDist.
- DynAlpha Fast forces the feedback into training early, but it does not improve results. It hurts Cora and slightly hurts CiteSeer.
- The naive agreement rule is too self-reinforcing: once GCN starts agreeing with a possibly noisy ball majority, it increases confidence in that same signal. The shuffled control staying high on Cora confirms that this rule is not isolating reliable ball structure.
- This experiment does not support expanding the current dynamic-alpha rule to 5 seeds.

## Decision

Keep SoftDist static as the current best fixed-ball mechanism. Do not use the current DynAlpha rule as the main method.

If continuing dynamic feedback, the next version should not directly multiply alpha from hard agreement. Better candidates:

| Candidate | Reason |
|---|---|
| confidence-gated agreement | only adjust alpha when model confidence is high and ball entropy is low |
| validation-calibrated alpha schedule | avoid early self-reinforcement from noisy predictions |
| dynamic reweighting loss term | add a small consistency regularizer instead of rewriting soft targets |
| rebuild balls from GCN hidden after warmup | test the real dynamic-ball idea separately from alpha self-training |
