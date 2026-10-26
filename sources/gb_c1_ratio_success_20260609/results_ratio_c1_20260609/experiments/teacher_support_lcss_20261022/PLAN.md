# Matched graph-teacher SUPPORT-LCSS experiment

Date: 2026-10-22

## Question and hypothesis

Does replacing the historical 64-ball partition with the SUPPORT partition improve a graph GCN teacher when the label-noise process, split, seed, model, optimizer, training budget, and clean-validation checkpoint selection are held fixed?

The primary hypothesis is that SUPPORT-LCSS improves direct teacher test accuracy over historical-partition LCSS. CE is included to measure the value of target smoothing, not as the primary SUPPORT contrast.

## Experimental role and motivation

This is a teacher-side test. It does not train an MLP student and does not test whether a better teacher improves distillation. SUPPORT is a label-free partition; labels enter only when constructing leave-one-out targets. All three arms train on the same original graph and features.

## Frozen protocol

- Datasets: Cora, CiteSeer, CoraFull, Coauthor-CS, and Amazon-Photo.
- Protocol seeds: 0, 1, and 2; the existing 20/10/70 split for each seed.
- Training-label conditions: uniform noise at 0.4 and 0.6, and pair noise at 0.4. Validation and test labels remain clean.
- Arms: CE on observed one-hot labels; LCSS on the historical partition; LCSS on the SUPPORT partition.
- Architecture: two-layer GCN, hidden width 64, dropout 0.5.
- Optimization: Adam, learning rate 0.01, weight decay 0.0005, 200 epochs.
- Selection: first epoch attaining the highest clean validation accuracy. Test labels are not used for selection.
- Paired initialization: reset the same protocol seed before each arm, preserving the historical dummy-MLP-then-GCN initialization order.
- Target construction: beta 0.5, at least two remaining votes, leave-one-out, tie-safe empirical distribution. The same target builder is used for both partitions.
- GRACE representations are not used by the GCN teacher. They are used only by the already-frozen, label-free SUPPORT partition.

Expected grid: 5 datasets x 3 seeds x 3 noisy conditions x 3 arms = 135 fits.

## Metrics and decision rule

Primary metric: paired test-accuracy difference SUPPORT-LCSS minus historical LCSS, averaged equally over the 45 dataset/seed/noise cells. Macro-F1 is secondary. CE contrasts are descriptive controls.

Predeclared practical continuation gate: mean paired SUPPORT-LCSS gain >= 0.5 percentage points; positive seed-level noisy-grid mean in all 3 seeds; and positive dataset-level noisy-grid mean on at least 3 of 5 datasets. This is an exploratory threshold, not a significance test. If the gate fails, conclude only that this SUPPORT partition did not show the predeclared material teacher-side increment under this protocol; do not infer that all adaptive partitions are ineffective.

If the gate passes, the next experiment is a paired student-distillation test using the saved CE, historical-LCSS, and SUPPORT-LCSS teacher probabilities. That student test is not part of this experiment.

## Required records

The run stores the resolved configuration, source/input hashes, environment, per-epoch logs, best checkpoints, full-node probabilities, per-cell metrics, paired summaries, and a decision report. Results must be interpreted as five-dataset exploratory evidence from three protocol seeds.
