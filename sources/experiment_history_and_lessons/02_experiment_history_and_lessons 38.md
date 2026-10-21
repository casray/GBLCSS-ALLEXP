# MLP-62: VQGraph neighbor-sampled ROCm protocol reimplementation

## Research hypothesis

The previous full-batch VQGraph port failed on CoraFull and Coauthor-CS because it reconstructed a dense adjacency matrix over the entire graph. Replacing that execution path with the official GraphSAGE-style neighbor sampling should remove the global memory bottleneck while preserving VQGraph's tokenizer, reconstruction losses, and student distillation objective.

## Role of granular balls

Granular balls are not part of VQGraph. The formal granular-ball consensus teacher is the paired reference method. Every comparison uses the same train/validation/test split and the same noisy-label realization.

## Method motivation

The official repository uses DGL `MultiLayerNeighborSampler([5,5])` for its large-graph SAGE path. Stable DGL GPU builds do not support the native Windows AMD ROCm environment used here. The experiment therefore ports the sampled execution semantics to PyTorch while retaining the vendored official `VectorQuantize` implementation.

## Code changes

- Added deterministic CPU inbound-neighbor sampling with target-first local-node ordering.
- Added a PyTorch equivalent of DGL `GraphConv(norm='both')`.
- Added local dense-adjacency reconstruction for each sampled subgraph.
- Added a separate `vqgraph_sampled_faithful` predictor without changing the existing full-batch predictor.
- Reused the official vector quantizer and the existing VQGraph soft-label and soft-token student training path.
- Added five dataset configurations and a summary/audit script.

## Configuration changes

- Noise: uniform 0.4 and 0.6, training labels only.
- Split: 20%/10%/70% train/validation/test.
- Seeds: 0-4.
- Sampling: fanouts [5,5], batch size 512.
- Devices: CPU for sampling/index assembly; ROCm `cuda:0` for all teacher/student trainable operations.
- Published VQGraph settings are used where available. CoraFull and Coauthor-CS use the pre-registered nearest-node-count mapping because they are not official VQGraph datasets.
- No post-result parameter tuning was performed.

## Expected mechanism

Neighbor sampling should reduce graph reconstruction from a global N-by-N tensor to one local tensor per mini-batch. This addresses memory feasibility but does not change the information or loss functions used by VQGraph.

## Pre-registered success and failure criteria

The smoke stage passed if Cora, CoraFull, and Coauthor-CS seed 0 completed with finite outputs, at least one active code, ROCm teacher/student parameters, matched split/noise hashes, and no global N-by-N reconstruction. Method ranking was explicitly excluded from the feasibility decision. Once feasibility passed, every retained run had to be reported without post-result tuning.

## Results

The smoke stage passed technically. Neighbor sampling removed the full-graph allocation and all trainable operations remained on the AMD GPU. CoraFull remained prohibitively slow and collapsed to one or three active codes. After two completed uniform-0.4 seeds (4.65% and 4.73% test accuracy), the author stopped further CoraFull runs. These two rows are diagnostic only.

The remaining four datasets completed 40/40 planned runs:

| Noise | Dataset | Sampled VQGraph | Ball teacher | Delta (pp) | Positive seeds |
|---|---|---:|---:|---:|---:|
| 0.4 | Cora | 60.17 +/- 4.31 | 81.35 +/- 1.05 | -21.18 | 0/5 |
| 0.4 | CiteSeer | 51.28 +/- 9.99 | 69.27 +/- 1.57 | -17.99 | 0/5 |
| 0.4 | Coauthor-CS | 74.22 +/- 4.50 | 93.83 +/- 0.21 | -19.61 | 0/5 |
| 0.4 | Amazon-Photo | 38.88 +/- 18.01 | 90.40 +/- 0.77 | -51.53 | 0/5 |
| 0.6 | Cora | 46.64 +/- 3.57 | 71.34 +/- 3.91 | -24.70 | 0/5 |
| 0.6 | CiteSeer | 37.84 +/- 11.34 | 58.93 +/- 4.62 | -21.10 | 0/5 |
| 0.6 | Coauthor-CS | 62.54 +/- 7.41 | 92.40 +/- 0.65 | -29.86 | 0/5 |
| 0.6 | Amazon-Photo | 37.43 +/- 21.42 | 86.81 +/- 1.94 | -49.38 | 0/5 |

All 40 paired seed differences are negative. Amazon-Photo has the largest instability and generally uses only one to three active codes. Coauthor-CS trains without complete codebook collapse but remains substantially below the ball teacher. The CoraFull failure repeats across two independent split/noise seeds.

## Interpretation

The experiment separates engineering feasibility from method effectiveness. The PyTorch/ROCm sampled path solves the global-memory problem, so the earlier inability to run CoraFull and Coauthor-CS was not fundamental. However, the frozen VQGraph SAGE configuration is unstable under the manuscript's noisy-label protocol. The failure is associated with low codebook utilization, but this experiment does not establish a causal mechanism.

The result is not an official DGL reproduction. CoraFull and Coauthor-CS are protocol extensions, and the official VQGraph README itself recommends GCN for easier reproduction while noting that GraphSAGE can require additional tuning.

## Final decision

- Do not add VQGraph as a complete five-dataset Table 4 row.
- Do not replace these results with tuned or selected seeds.
- The four-dataset results can be retained as a supplementary diagnostic only if the manuscript explicitly identifies them as a faithful protocol reimplementation rather than an official reproduction.
- The existing ball-teacher conclusion is not weakened: it is higher in all eight aggregate cells and all 40 paired seed comparisons.
- Stop this direction. Further tuning would answer a different, post-result optimization question and is not required for the manuscript.

## Traceable outputs

- `external/NoisyGL/runs/mlp62_vqgraph_neighbor_sampled_rocm_20260907/mlp62_report.md`
- `external/NoisyGL/runs/mlp62_vqgraph_neighbor_sampled_rocm_20260907/mlp62_formal_four_dataset_raw.csv`
- `external/NoisyGL/runs/mlp62_vqgraph_neighbor_sampled_rocm_20260907/mlp62_corafull_diagnostic_raw.csv`
- `external/NoisyGL/runs/mlp62_vqgraph_neighbor_sampled_rocm_20260907/mlp62_paired_to_ball_teacher.csv`
- `external/NoisyGL/runs/mlp62_vqgraph_neighbor_sampled_rocm_20260907/mlp62_protocol_audit.csv`
