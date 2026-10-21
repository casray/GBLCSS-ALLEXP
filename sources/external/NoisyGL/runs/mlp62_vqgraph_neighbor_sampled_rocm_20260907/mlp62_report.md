# MLP-62 VQGraph neighbor-sampled ROCm report

## Research question

Can the official VQGraph GraphSAGE execution semantics be transferred to the current AMD ROCm host, removing the full-batch graph-reconstruction bottleneck and supporting the manuscript noise protocol?

## Method and protocol

- VQGraph role: graph-teacher comparison; granular balls are not used by this baseline.
- Source: official VQGraph commit `60aa1c99782d542c316a37f0394eba934bcf97b0`.
- Identity: faithful PyTorch/ROCm protocol reimplementation, not an official DGL run.
- Teacher: sampled local GraphConv, official VectorQuantize, feature reconstruction, local adjacency reconstruction, and the existing VQGraph soft-label/soft-token student path.
- Sampling: CPU inbound-neighbor sampling with fanouts [5,5]; all trainable operations run on ROCm `cuda:0`.
- Evaluation: percent 0.2/0.1/0.7 splits, train-only uniform noise at 0.4 and 0.6, seeds 0-4, clean-validation checkpoint selection.
- Hyperparameters were frozen before the runs. No post-result tuning was performed.

## Code and configuration changes

- Added a deterministic CPU inbound-neighbor sampler and a PyTorch implementation of DGL GraphConv with `norm='both'`.
- Added a separate sampled VQGraph predictor; the existing full-batch predictor was not changed.
- Reused the vendored official VectorQuantize implementation and the existing VQGraph student objective.
- Added five sampled-path configurations. Published settings were used where available; CoraFull and Coauthor-CS used the pre-registered nearest-node-count mapping.

## Expected mechanism and decision criteria

Neighbor sampling was expected to replace the global dense graph-reconstruction tensor with bounded local tensors while preserving the tokenizer and distillation objectives. Feasibility required finite outputs, at least one active code, ROCm teacher/student parameters, paired split/noise hashes, and no global N-by-N reconstruction. Method ranking was not a feasibility criterion. If the smoke checks passed, every retained dataset-rate-seed row had to be reported without post-result tuning.

## Formal four-dataset results

| noise | dataset | VQGraph (%) | ball teacher (%) | delta (pp) | positive seeds |
|---|---|---:|---:|---:|---:|
| 0.4 | amazonpho | 38.88+/-18.01 | 90.40+/-0.77 | -51.53 | 0/5 |
| 0.4 | citeseer | 51.28+/-9.99 | 69.27+/-1.57 | -17.99 | 0/5 |
| 0.4 | coauthorcs | 74.22+/-4.50 | 93.83+/-0.21 | -19.61 | 0/5 |
| 0.4 | cora | 60.17+/-4.31 | 81.35+/-1.05 | -21.18 | 0/5 |
| 0.6 | amazonpho | 37.43+/-21.42 | 86.81+/-1.94 | -49.38 | 0/5 |
| 0.6 | citeseer | 37.84+/-11.34 | 58.93+/-4.62 | -21.10 | 0/5 |
| 0.6 | coauthorcs | 62.54+/-7.41 | 92.40+/-0.65 | -29.86 | 0/5 |
| 0.6 | cora | 46.64+/-3.57 | 71.34+/-3.91 | -24.70 | 0/5 |

## CoraFull diagnostic

CoraFull was stopped by author decision after two completed uniform-0.4 seeds. These rows are diagnostic only and are excluded from every formal aggregate and win/loss count.

| seed | test (%) | teacher val (%) | active codes | wall time (min) |
|---:|---:|---:|---:|---:|
| 0 | 4.65 | 5.86 | 1 | 92.0 |
| 1 | 4.73 | 5.31 | 3 | 85.9 |

## Protocol audit

- Formal rows: 40/40.
- All formal rows have finite test metrics, positive active-code counts, ROCm teacher/student parameters, training-time graph use, graph-free student inference, and paired split/noise hashes.
- No CPU model training or inference fallback was used.
- CoraFull and Coauthor-CS are protocol extensions because they are not official VQGraph datasets.

## Interpretation and decision

The sampled port removes the global N-by-N reconstruction allocation and runs end to end on the AMD GPU. It does not make VQGraph a strong noisy-label baseline under this protocol. The four-dataset results are below the paired granular-ball teacher in every aggregate cell and show large seed variance, especially on Amazon-Photo. The two completed CoraFull runs use only one and three active codes and yield near-random accuracy.

The complete five-dataset row is rejected because CoraFull was stopped. The four-dataset results may be retained as a supplementary protocol-reimplementation diagnostic, but they must not be presented as an official reproduction or as a complete Table 4 row.
