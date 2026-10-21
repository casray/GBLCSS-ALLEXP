# Graph-MLP Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the defining Graph-MLP architecture and NContrast objective into the train-only noisy-label protocol.

**Architecture:** A dedicated `GraphMLPReimplemented_Predictor.py` contains the official GELU/LayerNorm network, sparse r-hop adjacency preprocessing, all-train-node batch construction, and dense in-batch NContrast. The old sampled-edge `graphmlp` remains an explicitly named proxy.

**Tech Stack:** Python, PyTorch, SciPy sparse matrices, ROCm.

---

### Task 1: Lock NContrast and batch semantics

**Files:**
- Create: `tests/test_graphmlp_reimplemented.py`
- Create: `predictor/GraphMLPReimplemented_Predictor.py`

- [ ] Test NContrast against the official `-log(sum(exp(tau*S)*A_r)/sum(exp(tau*S)))` formula.
- [ ] Test that every train node appears exactly once at the front of each batch and receives noisy-label CE.
- [ ] Test graph-free inference by comparing outputs with/without adjacency.
- [ ] Implement the minimal helpers and network after observing RED.

### Task 2: Register method and official/default configs

**Files:**
- Modify runner registries.
- Create: `config/graphmlp_reimplemented/*.yaml`

- [ ] Use official Cora settings (hidden 256, dropout 0.6, alpha 10, tau 2, order 2) and CiteSeer settings (hidden 256, dropout 0.6, alpha 1, tau 0.5, order 2).
- [ ] For CoraFull/Coauthor-CS/Amazon-Photo use repository defaults (hidden 256, dropout 0.6, alpha 2, tau 1, order 2), not result-tuned values.
- [ ] Batch size is `max(2000, number_of_train_nodes)` so all supervised nodes are included.

### Task 3: GPU smoke and scalability gate

- [ ] Run Cora/CiteSeer seed0.
- [ ] Run CoraFull seed0 separately; stop on OOM or >30 minutes.
- [ ] Only after both gates pass, run five core datasets × seeds 0--4.

---

Official code provenance: `yanghu819/Graph-MLP`, commit `5448add06b6f95a580fe984b5e6e199ab8aeff18`.
