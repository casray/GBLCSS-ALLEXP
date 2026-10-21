# MLP-62 VQGraph Neighbor-Sampled ROCm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate a faithful PyTorch/ROCm port of the official neighbor-sampled VQGraph path and, if feasible, run the complete five-dataset comparison.

**Architecture:** A CPU CSR sampler builds DGL-compatible nested two-hop local node orderings. A focused ROCm teacher module applies DGL-equivalent normalized graph convolution, the vendored official vector quantizer, and local reconstruction losses. The existing VQGraph student path is reused unchanged.

**Tech Stack:** Python 3.12, PyTorch 2.9.1+ROCm 7.2.1, NumPy, pytest, existing NoisyGL runner.

---

### Task 1: Lock the sampling and graph-convolution contract

**Files:**
- Create: `predictor/module/vqgraph_neighbor_sampling.py`
- Create: `tests/test_mlp62_vqgraph_neighbor_sampled_rocm.py`

- [ ] Write tests for nested target-first node ordering, deterministic fanout,
  bidirected self-loop local graphs, bounded dense adjacency, and DGL-equivalent
  degree normalization.
- [ ] Run the new test file and confirm it fails because the module is absent.
- [ ] Implement the minimal CSR sampler and graph convolution.
- [ ] Run the tests and confirm they pass.

### Task 2: Add a separate sampled VQGraph predictor

**Files:**
- Modify: `predictor/G2MFaithful_Predictor.py`
- Modify: `scripts/run_standard_downstream.py`
- Create: `config/vqgraph_sampled_faithful/vqgraph_sampled_faithful_{dataset}.yaml`
- Modify: `tests/test_mlp62_vqgraph_neighbor_sampled_rocm.py`

- [ ] Add failing registration, configuration, GPU-device, and X-only student tests.
- [ ] Implement `vqgraph_sampled_faithful_Predictor` while leaving the old
  full-batch predictor unchanged.
- [ ] Add five frozen configs by carrying over MLP-61 values and adding fanout,
  graph batch size, and sampling seed fields.
- [ ] Run targeted and regression tests.

### Task 3: Run staged ROCm feasibility checks

**Files:**
- Create: `runs/mlp62_vqgraph_neighbor_sampled_rocm_20260907/`

- [ ] Run Cora uniform-0.4 seed 0 on `cuda:0`.
- [ ] Audit trainable parameter devices, local graph sizes, active codes,
  finite losses, and split/noise hashes.
- [ ] Run CoraFull and Coauthor-CS uniform-0.4 seed 0 on `cuda:0`.
- [ ] Stop without parameter tuning if any pre-registered feasibility criterion fails.

### Task 4: Execute and audit the formal grid

**Files:**
- Create: `scripts/run_mlp62_vqgraph_neighbor_sampled_rocm.py`
- Create: `scripts/summarize_mlp62_vqgraph_neighbor_sampled_rocm.py`
- Create: `runs/mlp62_vqgraph_neighbor_sampled_rocm_20260907/mlp62_report.md`
- Create: `C:/Users/Admin/Desktop/实验8去噪5.19-/02_experiment_history_and_lessons 38.md`

- [ ] If all smoke checks pass, run 5 datasets x 2 noise rates x 5 seeds.
- [ ] Verify row completeness, finite metrics, ROCm devices, paired split/noise
  hashes, active-code counts, and attribute-only student inference.
- [ ] Aggregate mean, sample standard deviation, and paired differences against
  the existing granular-ball teacher.
- [ ] Record the full protocol, results, anomalies, and final keep/drop decision.

