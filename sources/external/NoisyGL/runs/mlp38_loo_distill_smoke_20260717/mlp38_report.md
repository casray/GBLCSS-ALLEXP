# MLP-38 LOO-LCSS Distillation Seed0 Smoke

| Dataset | GCN[X||H] CE | Default LCSS teacher | LOO-LCSS teacher | LOO shuffled | LOO-default | LOO-shuffled |
|---|---:|---:|---:|---:|---:|---:|
| cora | 79.42 | 80.90 | 81.37 | 76.89 | +0.47pp | +4.49pp |
| citeseer | 69.46 | 69.12 | 69.72 | 69.54 | +0.60pp | +0.17pp |
| corafull | 58.61 | 57.81 | 57.90 | 58.29 | +0.09pp | -0.39pp |

## Preregistered gate

- Decision: `stop_seed0`
- Positive LOO-default datasets: 3/3
- Mean LOO-default: +0.39pp
- No regression over 0.5pp: 1
- LOO real-shuffled >=1pp datasets: 1/3

Seed0 is a directional smoke only; it is not evidence of statistical stability.
