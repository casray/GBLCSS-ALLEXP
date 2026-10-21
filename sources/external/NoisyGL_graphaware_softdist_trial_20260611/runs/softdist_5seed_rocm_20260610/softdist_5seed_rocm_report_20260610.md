# SoftDist 5-Seed ROCm Report

- Date: 2026-06-10
- Python: `.venv-rocm/Scripts/python.exe` under the GRACE-master ROCm environment provided by the user.
- Device: `cuda:0` reported by ROCm PyTorch, AMD Radeon RX 7700 XT
- Protocol: NoisyGL percent split 0.2/0.1/0.7, uniform noise 0.4, seeds 0-4.
- Methods: GCN+CE, NRGNN, SoftDist, SoftDist shuffled control.
- Boundary: ROCm results are internally comparable with each other; do not mix their exact seed0 values with previous CPU-run values.

## Mean Test Accuracy

| Dataset | GCN+CE | NRGNN | SoftDist | SoftDist Shuffled | SoftDist vs GCN | SoftDist vs Shuffled | SoftDist vs NRGNN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 0.7842 +/- 0.0151 | 0.8107 +/- 0.0230 | 0.7954 +/- 0.0094 | 0.7832 +/- 0.0167 | +0.0113 | +0.0122 | -0.0152 |
| Citeseer | 0.6530 +/- 0.0121 | 0.7027 +/- 0.0108 | 0.6737 +/- 0.0113 | 0.6578 +/- 0.0116 | +0.0207 | +0.0159 | -0.0289 |

## Per-Seed Test Accuracy

### Cora

| Seed | GCN+CE | NRGNN | SoftDist | SoftDist Shuffled |
|---:|---:|---:|---:|---:|
| 0 | 0.7974 | 0.8311 | 0.8042 | 0.8047 |
| 1 | 0.7995 | 0.8169 | 0.7894 | 0.7937 |
| 2 | 0.7625 | 0.7747 | 0.7826 | 0.7609 |
| 3 | 0.7826 | 0.8026 | 0.8042 | 0.7799 |
| 4 | 0.7789 | 0.8280 | 0.7968 | 0.7768 |

### Citeseer

| Seed | GCN+CE | NRGNN | SoftDist | SoftDist Shuffled |
|---:|---:|---:|---:|---:|
| 0 | 0.6435 | 0.7113 | 0.6748 | 0.6491 |
| 1 | 0.6594 | 0.7161 | 0.6843 | 0.6619 |
| 2 | 0.6503 | 0.7010 | 0.6817 | 0.6718 |
| 3 | 0.6705 | 0.6916 | 0.6555 | 0.6632 |
| 4 | 0.6413 | 0.6933 | 0.6723 | 0.6430 |

## SoftDist Diagnostics Mean

| Dataset | Variant | Mean Alpha | Soft Shift Mass | Pseudo NCR | Pseudo CDR | Clean-Mass Delta |
|---|---|---:|---:|---:|---:|---:|
| Cora | SoftDist | 0.4488 | 97.7327 | 0.6871 | 0.2018 | -29.4284 |
| Cora | SoftDist Shuffled | 0.2069 | 70.8886 | 0.1554 | 0.6574 | -44.9437 |
| Citeseer | SoftDist | 0.3847 | 112.6817 | 0.5826 | 0.2996 | -45.9721 |
| Citeseer | SoftDist Shuffled | 0.2100 | 86.0574 | 0.1465 | 0.6615 | -55.2261 |

## Decision

- Cora: SoftDist beats GCN+CE by `+0.0113` mean accuracy and beats shuffled by `+0.0122`, but the gain is within the observed seed variance. This is a positive but not yet strong Cora signal.
- CiteSeer: SoftDist is useful and more stable than the seed0-only read suggested. Mean delta is `+0.0207` vs GCN+CE and `+0.0159` vs shuffled, but it is still `-0.0289` below NRGNN.
- NRGNN remains the strongest method on both datasets in this ROCm 5-seed run.
- SoftDist should remain a candidate mechanism, but the current frozen one-shot version is not yet strong enough as the paper main method. The next useful step is a dynamic-alpha GCN-Ball feedback smoke, not another fixed alpha tweak.
