# DBLP selected ellipsoid downstream 5-seed report

Scope: DBLP only, seeds 0-4. This follows the seed0 sanity result; PubMed, AmazonCom, and AmazonPhoto were not advanced in this report.

## Protocol

- Dataset: DBLP
- Seeds: 0, 1, 2, 3, 4
- Noise: uniform 0.4, train scope
- Split: percent 0.2 / 0.1 / 0.7
- Selected GB setting: max_balls=128, split_improvement=0.001
- Device: cuda:0

## Integrity checks

- Raw rows: 25; expected rows: 25
- Status ok rows: 25 / 25
- Duplicate conflicts: 0
- seed 0: rows=5, status_ok=True, same_split_hash=True, same_noise_hash=True
- seed 1: rows=5, status_ok=True, same_split_hash=True, same_noise_hash=True
- seed 2: rows=5, status_ok=True, same_split_hash=True, same_noise_hash=True
- seed 3: rows=5, status_ok=True, same_split_hash=True, same_noise_hash=True
- seed 4: rows=5, status_ok=True, same_split_hash=True, same_noise_hash=True

## Test accuracy mean/std

| Method | Mean | Std | ok seeds |
|---|---:|---:|---:|
| GCN[XH] | 0.8391 | 0.0032 | 5 |
| Current SoftDist | 0.8391 | 0.0036 | 5 |
| Current shuffled | 0.8391 | 0.0036 | 5 |
| Selected ellipsoid | 0.8392 | 0.0031 | 5 |
| Selected shuffled | 0.8374 | 0.0041 | 5 |

## Paired deltas

| Comparison | Mean delta | Std | Positive seeds |
|---|---:|---:|---:|
| selected minus selected shuffled | +0.19pp | +0.49pp | 3 / 5 |
| selected minus current softdist | +0.01pp | +0.09pp | 2 / 5 |
| selected minus current shuffled | +0.01pp | +0.09pp | 2 / 5 |
| selected minus gcn xh | +0.01pp | +0.19pp | 2 / 5 |
| current softdist minus current shuffled | +0.00pp | +0.00pp | 0 / 5 |

## Per-seed selected vs selected shuffled

| Seed | Selected ellipsoid | Selected shuffled | Delta |
|---:|---:|---:|---:|
| 0 | 0.8424 | 0.8320 | +1.04pp |
| 1 | 0.8416 | 0.8414 | +0.02pp |
| 2 | 0.8397 | 0.8403 | -0.06pp |
| 3 | 0.8376 | 0.8391 | -0.15pp |
| 4 | 0.8349 | 0.8340 | +0.08pp |

## Decision

STOP: selected ellipsoid does not provide stable downstream value over shuffled.

Interpretation boundary: this is DBLP-only downstream evidence. It does not rescue PubMed/AmazonCom from the seed0 sanity failure, and it does not establish a cross-dataset paper claim.

## Output files

- runs/ellipsoid_selected_downstream_seed0_20260629/dblp_5seed_combined_raw.csv
- runs/ellipsoid_selected_downstream_seed0_20260629/dblp_5seed_summary.csv
- runs/ellipsoid_selected_downstream_seed0_20260629/dblp_5seed_paired_deltas.csv
