# ProCon / TFR Same-Protocol Reference Experiment

## Material Passport

- Date: 2026-07-07
- Protocol: node classification with train-only uniform label noise.
- Split: percent split, train/validation/test = 0.2/0.1/0.7.
- Noise: `noise_type=uniform`, `noise_rate=0.4`, `noise_scope=train`.
- Datasets: Cora, CiteSeer.
- Seeds: 0, 1, 2, 3, 4.
- Device: `cuda:0` with ROCm Python environment.
- Methods: `procon`, `tfr_gcn`.
- Raw output: `procon_tfr_uniform0p4_seeds_0_1_2_3_4.csv`.
- Aggregates: `procon_tfr_uniform0p4_aggregate.csv`.
- Paired seed table: `procon_tfr_uniform0p4_paired.csv`.

## Research Hypothesis

The previous manuscript table listed TFR as pending and ProCon as a paper-reported reference. This experiment tests whether both 2025 graph-dependent methods can be evaluated under the paper's own protocol and included as same-protocol graph-dependent references.

## Role of Granular Balls

Granular balls are not part of ProCon or TFR. Their role in this experiment is indirect: the results position LCSS and LCSS-distill against newer graph-dependent noisy-label graph methods.

## Method Motivation

The paper should not rely only on CT as a 2025 reference. ProCon and TFR are both recent graph-native noisy-label methods, but their inference/training mechanisms require graph structure. Running them under the same local protocol strengthens the graph-dependent reference table while preserving the graph-free scope boundary.

## Code / Config Changes

- No predictor or config changes were made.
- Existing local implementations were used:
  - `predictor/ProCon_Predictor.py`
  - `predictor/TFR_Predictor.py`
  - `config/procon/*.yaml`
  - `config/tfr_gcn/*.yaml`
- Manuscript HTML was updated after the experiment:
  - `docs/paper_manuscript_v1_zh_20260706_alg32_revised.html`

## Commands

Smoke:

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" scripts/run_standard_downstream.py `
  --datasets cora,citeseer `
  --methods procon,tfr_gcn `
  --seeds 0 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --noise_scope train `
  --device cuda:0 `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --output runs/procon_tfr_same_protocol_20260707/smoke_seed0.csv `
  --stop_on_error
```

5-seed:

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" scripts/run_standard_downstream.py `
  --datasets cora,citeseer `
  --methods procon,tfr_gcn `
  --seeds 0,1,2,3,4 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --noise_scope train `
  --device cuda:0 `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --output runs/procon_tfr_same_protocol_20260707/procon_tfr_uniform0p4_seeds_0_1_2_3_4.csv `
  --stop_on_error
```

## Results

| Dataset | Method | Mean ± std |
|---|---|---:|
| Cora | ProCon | 76.93 ± 2.20 |
| Cora | TFR+GCN | 78.77 ± 0.87 |
| CiteSeer | ProCon | 61.44 ± 3.23 |
| CiteSeer | TFR+GCN | 68.71 ± 1.71 |

## Seed-Level Values

| Dataset | Seed | ProCon | TFR+GCN |
|---|---:|---:|---:|
| Cora | 0 | 76.20 | 78.89 |
| Cora | 1 | 74.78 | 78.94 |
| Cora | 2 | 75.25 | 77.31 |
| Cora | 3 | 78.52 | 79.05 |
| Cora | 4 | 79.89 | 79.63 |
| CiteSeer | 0 | 62.20 | 67.35 |
| CiteSeer | 1 | 59.71 | 68.08 |
| CiteSeer | 2 | 63.75 | 67.18 |
| CiteSeer | 3 | 64.78 | 69.76 |
| CiteSeer | 4 | 56.79 | 71.18 |

## Final Decision

Continue using ProCon and TFR+GCN as graph-dependent reference methods, not as X-only main-table baselines. TFR+GCN is a stronger 2025 graph-dependent reference than ProCon under this protocol. LCSS-distill remains competitive with these graph-dependent references while preserving X-only inference: Cora 80.60 vs TFR+GCN 78.77, CiteSeer 69.40 vs TFR+GCN 68.71. This supports the manuscript's scope claim that graph-free deployment can approach graph-dependent noisy-label GNN performance after distillation.

## Failure / Boundary Notes

- ProCon under the local 20/10/70 train-only noise protocol is substantially weaker than its original-paper Symmetric-0.4 numbers on CiteSeer. This is likely affected by protocol and implementation differences, so the manuscript should not claim an official reproduction.
- TFR+GCN should be reported as same-protocol local reimplementation/reference, not as an official reproduced number unless the implementation is further audited against the official repository.
- Both methods require graph structure and therefore belong in the graph-dependent reference table, not the X-only main table.
