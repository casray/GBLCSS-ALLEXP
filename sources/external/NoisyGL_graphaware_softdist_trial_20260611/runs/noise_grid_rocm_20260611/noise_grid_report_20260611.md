# Phase 6: Noise Grid Regime Analysis

Date: 2026-06-11

## Protocol

- Runtime: `C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`
- Device: `cuda:0`
- Split: `split_type=percent`, `train_percent=0.2`, `val_percent=0.1`, `test_percent=0.7`
- Datasets: Cora, CiteSeer
- Seeds: 0-4
- New regimes: `uniform 0.2`, `uniform 0.6`, `pair 0.2`, `pair 0.4`
- Reused anchor regime: `uniform 0.4`
- Methods: `gcn`, `gcn_xh_ce`, `frozengrace_gb_gcn_softdist_xh`, `frozengrace_gb_gcn_softdist_xh_shuffled`, `nrgnn`

The same frozen GRACE embedding H is reused across all noise regimes. H is label-free, so the representation stage is not re-exposed to label noise when the noise setting changes.

## Main Table

### uniform 0.2

| Method | Cora | CiteSeer |
|---|---:|---:|
| GCN+CE | 0.8350 ± 0.0094 | 0.7111 ± 0.0115 |
| GCN+CE [X\|\|H] | **0.8444 ± 0.0045** | 0.7154 ± 0.0085 |
| SoftDist [X\|\|H] | 0.8407 ± 0.0120 | 0.7208 ± 0.0092 |
| SoftDist [X\|\|H] shuffled | 0.8432 ± 0.0027 | 0.7221 ± 0.0064 |
| NRGNN | 0.8303 ± 0.0230 | **0.7289 ± 0.0064** |

### uniform 0.4

| Method | Cora | CiteSeer |
|---|---:|---:|
| GCN+CE | 0.7842 ± 0.0151 | 0.6530 ± 0.0121 |
| GCN+CE [X\|\|H] | 0.8027 ± 0.0069 | 0.6761 ± 0.0181 |
| SoftDist [X\|\|H] | **0.8137 ± 0.0067** | 0.6891 ± 0.0159 |
| SoftDist [X\|\|H] shuffled | 0.7921 ± 0.0097 | 0.6696 ± 0.0234 |
| NRGNN | 0.8107 ± 0.0230 | **0.7027 ± 0.0108** |

### uniform 0.6

| Method | Cora | CiteSeer |
|---|---:|---:|
| GCN+CE | 0.6579 ± 0.0376 | 0.4968 ± 0.0552 |
| GCN+CE [X\|\|H] | 0.6443 ± 0.0924 | 0.5578 ± 0.0444 |
| SoftDist [X\|\|H] | 0.6509 ± 0.1006 | 0.5801 ± 0.0462 |
| SoftDist [X\|\|H] shuffled | 0.6868 ± 0.0378 | 0.5572 ± 0.0343 |
| NRGNN | **0.7629 ± 0.0141** | **0.6513 ± 0.0375** |

### pair 0.2

| Method | Cora | CiteSeer |
|---|---:|---:|
| GCN+CE | 0.8072 ± 0.0305 | 0.6884 ± 0.0209 |
| GCN+CE [X\|\|H] | 0.8070 ± 0.0385 | 0.6997 ± 0.0141 |
| SoftDist [X\|\|H] | 0.8261 ± 0.0184 | 0.6816 ± 0.0296 |
| SoftDist [X\|\|H] shuffled | 0.8088 ± 0.0268 | 0.6957 ± 0.0154 |
| NRGNN | **0.8292 ± 0.0159** | **0.7172 ± 0.0131** |

### pair 0.4

| Method | Cora | CiteSeer |
|---|---:|---:|
| GCN+CE | 0.6003 ± 0.0612 | 0.4987 ± 0.0423 |
| GCN+CE [X\|\|H] | 0.5955 ± 0.0498 | 0.5246 ± 0.0247 |
| SoftDist [X\|\|H] | **0.6139 ± 0.0564** | 0.5378 ± 0.0378 |
| SoftDist [X\|\|H] shuffled | 0.5783 ± 0.0438 | 0.5182 ± 0.0302 |
| NRGNN | 0.6088 ± 0.0417 | **0.5802 ± 0.0326** |

## Ours vs NRGNN

Win definition: `softdist_xh - nrgnn` mean delta > 0 and at least 4/5 paired seeds positive. Loss is symmetric. Otherwise tie.

| Regime | Cora | CiteSeer |
|---|---|---|
| uniform 0.2 | tie (+1.05pp, 3/5) | loss (-0.81pp, 0/5) |
| uniform 0.4 | tie (+0.31pp, 3/5) | loss (-1.36pp, 1/5) |
| uniform 0.6 | loss (-11.20pp, 0/5) | loss (-7.12pp, 0/5) |
| pair 0.2 | tie (-0.32pp, 2/5) | loss (-3.56pp, 0/5) |
| pair 0.4 | tie (+0.52pp, 3/5) | loss (-4.23pp, 0/5) |

Decision: there is no strict regime win over NRGNN. The strongest claim remains: on Cora, SoftDist [X||H] is consistently competitive with NRGNN and often has lower variance; on CiteSeer, NRGNN remains stronger.

## Ball Increment Trend

Paired delta `softdist_xh - gcn_xh_ce`:

| Regime | Cora | CiteSeer |
|---|---:|---:|
| uniform 0.2 | -0.37pp (2/5) | +0.54pp (3/5) |
| uniform 0.4 | +1.10pp (4/5) | +1.30pp (5/5) |
| uniform 0.6 | +0.66pp (4/5) | +2.22pp (4/5) |
| pair 0.2 | +1.91pp (3/5) | -1.81pp (2/5) |
| pair 0.4 | +1.84pp (4/5) | +1.32pp (3/5) |

The "higher noise gives larger ball gain" hypothesis is only partially supported. For uniform noise, CiteSeer shows a monotonic increase in ball increment from 0.2 to 0.6, and Cora improves from 0.2 to 0.4 but does not further increase at 0.6. Under pair noise, the trend is not monotonic and depends strongly on dataset.

## Real vs Shuffled Ball Structure

Paired delta `softdist_xh - softdist_xh_shuffled`:

| Regime | Cora | CiteSeer |
|---|---:|---:|
| uniform 0.2 | -0.25pp | -0.13pp |
| uniform 0.4 | +2.16pp | +1.95pp |
| uniform 0.6 | -3.59pp | +2.29pp |
| pair 0.2 | +1.73pp | -1.41pp |
| pair 0.4 | +3.57pp | +1.96pp |

The ball structure contribution is strongest and most consistent at uniform 0.4 and pair 0.4. It is not universally positive: low-noise uniform and pair 0.2 CiteSeer show that ball correction can be redundant or slightly harmful when labels are already relatively reliable or when pair-noise class coupling conflicts with ball-local voting.

## Interpretation

The noise grid does not support a broad claim that SoftDist [X||H] beats NRGNN across regimes. It supports a narrower and more defensible claim:

- The method is much simpler than NRGNN and remains competitive on Cora across multiple regimes.
- Its strongest setting is uniform 0.4 Cora, where it matches NRGNN mean performance with lower variance.
- The method provides a stable feature-level gain from frozen GRACE H and a conditional ball-level gain from local soft label correction.
- CiteSeer remains the main weakness, especially against NRGNN.
- Very high uniform noise (0.6) is unfavorable for this offline target-correction pipeline; NRGNN's online graph-native mechanism is clearly stronger there.

## Decision

Do not claim regime-level superiority over NRGNN. Use the grid as an applicability-boundary analysis:

1. Main method: SoftDist [X||H].
2. Main advantage: simple non-iterative pipeline, stable Cora competitiveness, strong attribution chain.
3. Limitation: weaker on CiteSeer and high uniform noise.
4. Closed mechanisms remain closed: unlabeled expansion, ball consistency, dynamic alpha, adaptive granularity, BallAux, and early-learning arbitration.

## Artifacts

- Raw summary: `runs/noise_grid_rocm_20260611/noise_grid_raw_summary_20260611.csv`
- Aggregate: `runs/noise_grid_rocm_20260611/noise_grid_aggregate_20260611.csv`
- Paired deltas: `runs/noise_grid_rocm_20260611/noise_grid_paired_deltas_20260611.csv`
- Regime win matrix: `runs/noise_grid_rocm_20260611/noise_grid_regime_win_matrix_20260611.csv`
- Ball increment trend: `runs/noise_grid_rocm_20260611/noise_grid_ball_increment_trend_20260611.csv`
- Logs: `runs/noise_grid_rocm_20260611/logs/`
