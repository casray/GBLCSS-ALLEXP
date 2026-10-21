# MLP-62 VQGraph Neighbor-Sampled ROCm Design

## Objective

Complete the five-dataset VQGraph comparison without changing the method's
published tokenizer, reconstruction losses, vector quantizer, or student
distillation objective. The implementation follows the official GraphSAGE
execution path, but replaces DGL sampling and `GraphConv` with a local
PyTorch implementation that runs on the installed ROCm stack.

## Evidence status

This is a faithful protocol reimplementation pinned to VQGraph commit
`60aa1c99782d542c316a37f0394eba934bcf97b0`. It is not an official DGL
reproduction. CoraFull and Coauthor-CS are protocol extensions because the
official VQGraph evaluation does not include them.

## Fixed protocol

- Datasets: Cora, CiteSeer, CoraFull, Coauthor-CS, and Amazon-Photo.
- Noise: uniform 0.4 and 0.6, injected into training labels only.
- Seeds: 0--4 with the existing percent 0.2/0.1/0.7 split procedure.
- Validation: the existing clean-validation checkpoint rule.
- Teacher: two sampled local `GraphConv` layers, raw-feature-dimensional
  vector quantization, node reconstruction, and sampled-local adjacency
  reconstruction.
- Sampling: two inbound-neighbor layers with fanout 5 and 5. Sampling and
  index assembly run on CPU; all trainable teacher/student operations run on
  ROCm `cuda:0`.
- Student: the existing official-core VQGraph MLP soft-label and soft-token
  distillation path.
- Hyperparameters: reuse the frozen MLP-61 dataset configurations. No
  post-result tuning is permitted.

## Success and failure criteria

The implementation is feasible only if Cora seed 0 and the two previously
blocked large-dataset seed-0 runs complete on ROCm, produce finite outputs,
use more than zero active codes, and never allocate an `N x N` graph
reconstruction tensor. A run fails if any trainable operation falls back to
CPU, if split/noise hashes differ from the paired ball-teacher records, or if
the sampled local graph exceeds the configured safety bound.

If feasibility passes, execute all 50 rows and report every result. Method
ranking is not a feasibility criterion.

