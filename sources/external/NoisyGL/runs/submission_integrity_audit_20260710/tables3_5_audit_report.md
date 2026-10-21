# Submission Integrity Audit - Phase 2: Tables 3-5

## Numerical and Protocol Verdict

- Status: PASS for the reported Table 3-5 numerical cells and train-only protocol.
- Table 3 rows: 225 (5 datasets x 9 methods x 5 seeds).
- Table 4 rows: 150 (3 datasets x 2 noise rates x 5 teacher arms x 5 seeds).
- Table 5 rows: 200 (5 datasets x 8 methods x 5 seeds).
- Every audited source row reports `noise_scope=train`, zero modified validation/test labels, and `modified_count == modified_train_count`.
- Split and noise fingerprints match across paired methods for every dataset/setting/seed.
- Manuscript cell mismatches: 0.

## Evidence Scope

- Table 3 values trace to 225 seed rows. CoraFull uses the MLP-21 completion; the other four datasets trace through MLP-19 to their formal raw CSVs.
- Table 4 values trace to the MLP-9 teacher ablation for Cora/CiteSeer and the MLP-22 completion for CoraFull.
- Table 5 values trace to the clean MLP-18 graph-method reruns, the later 2025-method completion, and MLP-22 for CoraFull.

## Important Boundaries Not Certified by This Pass

- This pass proves that the manuscript numbers can be recomputed from existing seed rows under the stated protocol; it does not prove that adapted/reimplemented methods are faithful official reproductions.
- Graph-MLP, GSDN, FF-G2M, KRD, VQGraph, TFR, ProCon, and GD2 remain local adapted/core implementations and require a separate method-fidelity audit.
- The CoraFull teacher ablation is a negative attribution result for the ball teacher: it is below GCN[X||H], kNN, and shuffled teachers. The manuscript currently discloses this boundary.
- CLNode is intentionally absent from Table 5; its current local implementation reads validation labels during curriculum construction and is not citable until fixed and rerun.
