# MLP-55: FF-G2M, KRD, and VQGraph for Table 4

## Research hypothesis

Adding representative GNN-to-MLP methods to the teacher-distillation table can
test whether the granular-ball consensus teacher remains competitive against
more elaborate graph-knowledge transfer mechanisms under noisy labels.

## Role of granular balls

Granular balls are used only in the consensus-teacher route. They aggregate
noisy training labels in the frozen representation space and provide the soft
targets used to train the graph teacher. The final student remains X-only.

## Motivation

The previous Table 4 compared GLNN-style response distillation, the
granular-ball consensus teacher, and an NRGNN teacher. FF-G2M, KRD, and VQGraph
cover frequency-aware distillation, reliable-knowledge selection, and
token-based structure transfer. Adding them makes the extension comparison
more representative without changing the formal GB-LCSS method.

## Code and configuration changes

- Added explicit resource identities for `ffg2m_reimplemented`,
  `krd_adapted`, and `vqgraph_adapted` in
  `scripts/run_standard_downstream.py`.
- Added `scripts/summarize_mlp55_g2m_table4_20260812.py` and its tests.
- Reused existing per-dataset configs. No hyperparameter was selected from the
  new results.
- FF-G2M uses the MLP-26 common-backbone paper reimplementation. KRD and
  VQGraph are mechanism adaptations, not official tuned reproductions.

## Protocol

- Datasets: Cora, CiteSeer, CoraFull.
- Noise: uniform 0.4 and 0.6, injected only into training labels.
- Split: 0.2/0.1/0.7 train/validation/test.
- Seeds: 0--4.
- Methods: FF-G2M, KRD, VQGraph.
- Each dataset-noise-seed-method run retrains its teacher and student.
- Training uses the graph and unlabeled nodes; inference uses only node
  features.

## Expected mechanism

- FF-G2M may preserve structure by separately transferring low- and
  high-frequency knowledge.
- KRD may reduce harmful transfer by selecting teacher knowledge stable to
  feature perturbations.
- VQGraph may transfer local structural information through soft tokens.
- The granular-ball teacher may retain an advantage when noisy-label local
  consensus is more important than general structure transfer.

## Success and failure criteria

The experiment is valid if all 90 records complete, validation/test labels are
unchanged, student inference is X-only, and every new row matches the existing
Table 4 split/noise hash. Methods enter Table 4 regardless of ranking. A method
failure is NaN/OOM, protocol leakage, hash mismatch, or incomplete seeds.

## Results

All 90/90 new records succeeded, and all 30 protocol groups passed the hash
audit.

| Method | Cora u0.4/u0.6 | CiteSeer u0.4/u0.6 | CoraFull u0.4/u0.6 |
|---|---:|---:|---:|
| FF-G2M | 77.65/63.63 | 65.80/52.32 | 56.58/45.78 |
| KRD | 71.77/61.28 | 64.96/54.17 | 47.12/41.61 |
| VQGraph | 78.89/65.75 | 65.64/53.34 | 58.52/52.01 |
| Granular-ball teacher | 80.60/71.27 | 69.40/62.28 | 58.22/51.82 |
| NRGNN teacher | 81.07/74.81 | 70.87/66.62 | 57.71/53.55 |

The granular-ball teacher exceeds all three added routes on Cora and CiteSeer
at both noise rates. Relative to the strongest added route in each cell, its
lead is 1.71/5.52 points on Cora and 3.60/8.11 points on CiteSeer. VQGraph is
higher on CoraFull by 0.29/0.19 points. NRGNN remains the strongest high-noise
teacher across all three datasets.

## Final decision

Continue with all three rows in Table 4. The result supports a scoped claim:
the granular-ball consensus teacher is competitive with the tested
GNN-to-MLP mechanisms and stronger on Cora/CiteSeer, while CoraFull and NRGNN
show that it is not universally best. The formal train-target GB-LCSS method is
unchanged.
