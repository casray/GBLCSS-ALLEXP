# MLP-8 P2 Ball-Parameter Rescue Diagnostic

## Goal

Check whether the MLP-8 P2 failure was caused by overly strict granular-ball construction parameters rather than by unusable P2 representations.

This is a diagnostic-only experiment. It does not change the main method, beta, split, noise protocol, or default ball parameters.

## Hypothesis

P2a and P2b may have failed because the default construction rule collapsed the candidate representation into one ball. If looser construction parameters recover both a normal ball count and GRACE-like purity, the direction deserves a downstream seed0 re-test. If only ball count recovers while purity remains far below frozen GRACE, the failure is representation-level and P2 remains closed.

## Scope

- Dataset and seed:
  - P2a: Cora seed0.
  - P2b: Cora and CiteSeer seed0.
- Noise:
  - `uniform`, rate `0.4`.
  - Percent split: train `0.2`, val `0.1`, test `0.7`.
- Representations:
  - P2a anchor: frozen GRACE `H`.
  - P2a candidate: `GRACE + 0.1 noisy CE` saved by MLP-8.
  - P2b anchor: frozen GRACE `H`.
  - P2b candidate: `Z1`, reconstructed from the original F0 MLP.

## Parameter Grid

- `max_balls = 64`
- `min_ball_size in {16, 32, 64}`
- `split_improvement in {0.05, 0.01, 0.005, 0.001, 0.0}`

The default point is `64 / 32 / 0.05`.

## Metrics

For each representation and parameter setting, report:

- `num_balls`
- `mean_ball_size`
- `mean_train_per_ball`
- `eligible_balls`
- `eligible_train_nodes`
- `alpha_positive_train_count`
- `mean_alpha_train`
- `noisy_train_purity_weighted`
- `true_train_purity_weighted`
- `true_all_purity_weighted`

Clean-label purity is diagnostic only and must not enter training.

## Gate

For each P2 candidate, compare against the matching frozen-GRACE anchor under default parameters.

The candidate is considered geometrically rescued only if a single parameter point satisfies:

- `num_balls >= 0.75 * anchor_default_num_balls`
- `true_all_purity_weighted >= anchor_default_true_all_purity_weighted - 0.05`

If both conditions pass, a later downstream seed0 run is allowed. If not, P2 remains closed and no downstream matrix should be launched.

## Output

Run directory:

```text
runs/mlp8_p2_ballparam_rescue_20260703/
```

Files:

- `p2_ballparam_raw.csv`
- `p2_ballparam_best.csv`
- `p2_ballparam_decisions.csv`
- `p2_ballparam_report.md`
- reconstructed `Z1` embeddings under `embeddings/`

## Command

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" scripts\diagnose_mlp8_p2_ballparam_rescue.py --device cuda:0
```

