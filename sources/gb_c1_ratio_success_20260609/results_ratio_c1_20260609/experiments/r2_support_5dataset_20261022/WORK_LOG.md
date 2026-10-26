# Work Log

## Question and hypothesis

Measure the absolute performance of R2 + SUPPORT + LCSS on five datasets under three noisy-label conditions. SUPPORT supplies local training-label counts; LCSS converts them to leave-one-out soft targets; R2 aligns an attribute MLP to a fixed GRACE representation before classifier training. This run has no comparison arm, so it is descriptive rather than a test of whether SUPPORT or LCSS improves performance.

## Protocol and mechanism

- Datasets: Cora, CiteSeer, CoraFull, Coauthor-CS, Amazon-Photo.
- Seeds: 0, 1, 2; training noise: uniform 0.4, uniform 0.6, pair 0.4.
- Existing ordered 20/10/70 split, fixed GRACE embeddings, clean-validation epoch selection.
- SUPPORT tree: normalized GRACE geometry, at most 256 leaves, minimum 32 members and 3 training members per child, support target k=32.
- LCSS: leave-one-out, beta=0.5, minimum 2 other votes; ties retain the empirical vote distribution.
- R2: 100 label-free alignment epochs followed by at most 100 classifier-only epochs; encoder and batch-normalization state frozen in the classifier stage.
- No clean-training, KNN, or K-means comparison cells were run. Test labels were used only for final scoring.

## Execution

The first three datasets had 27 audited R2 + SUPPORT cells already available and were reused after protocol, target, input, checkpoint, probability, and epoch-log checks. The two remaining datasets required 18 new student fits and six label-free R2 alignment checkpoints. All new fits used the verified AMD Radeon RX 7700 XT environment.

One runner import-path error occurred before data preparation or training; it was fixed locally by adding the existing R2 experiment directory to the import path. The subsequent preflight and run completed without failed cells.

## Result

All 45 expected cells passed the final audit. Across the 45 dataset-seed-noise cells, Accuracy is 70.52% and Macro-F1 is 62.39%. When averaging the 15 cells within each seed, the between-seed sample standard deviations are 1.18 percentage points for Accuracy and 1.26 for Macro-F1. Per-dataset and per-noise summaries are in `report.md`, `condition_summary.csv`, and `dataset_summary.csv`.

These are absolute results only. They do not establish an improvement over another method. The GRACE embedding is fixed within each dataset rather than independently pretrained for each seed; seed variation therefore reflects split/noise generation, R2 alignment initialization, and classifier training, not GRACE-pretraining variability.

## Decision

The requested characterization is complete. Keep this run as a five-dataset R2 + SUPPORT + LCSS result package. Do not infer SUPPORT/LCSS efficacy from these numbers; a separate matched comparison is needed for that claim.
