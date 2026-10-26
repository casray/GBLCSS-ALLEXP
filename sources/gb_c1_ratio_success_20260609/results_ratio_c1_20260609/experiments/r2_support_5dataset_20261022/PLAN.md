# R2 + SUPPORT + LCSS on Five Datasets

## Material Passport

- Type: transductive node-classification experiment.
- Status: planned.
- Goal: characterize R2 + SUPPORT + LCSS over five datasets and three noisy-label conditions.
- No KNN, K-means, or clean-training comparison is included.

## Hypothesis and method

SUPPORT partitions are used to aggregate the other training nodes' observed labels into leave-one-out soft targets. R2 first aligns an attribute MLP representation to a fixed GRACE embedding without labels, then freezes the aligned encoder and trains its classifier using the LCSS targets.

The experiment reports absolute test accuracy and macro-F1. Because this grid contains no comparison arm, it cannot establish that SUPPORT or LCSS improves over another method.

## Frozen protocol

- Datasets: Cora, CiteSeer, CoraFull, Coauthor-CS, Amazon-Photo.
- Protocol/student seeds: 0, 1, 2.
- Training-label noise: uniform 0.4, uniform 0.6, pair 0.4. Noise is applied to training labels only.
- Split: the existing ordered 20/10/70 split for each seed.
- Representation: reuse the existing fixed GRACE embedding for each dataset. No independent GRACE pretraining is added.
- SUPPORT: the 2026-10-22 support-only dynamic-programming cut; the candidate tree uses normalized GRACE representations, at most 256 leaves, minimum 32 total members and 3 training members per child, with k=32.
- LCSS: leave-one-out, beta=0.5, minimum 2 other votes, and the existing tie rule.
- R2: 100 label-free alignment epochs, then 100 classifier-only epochs with the encoder and batch-normalization state frozen. Adam, learning rate 0.01, weight decay 0.0005, hidden width 64, dropout 0.5.
- Validation: clean validation accuracy, first epoch attaining the maximum. Test labels are scored only after the full grid finishes.
- Expected final cells: 5 x 3 x 3 = 45. Reuse the 27 already audited R2+SUPPORT cells for the first three datasets; fit the 18 Coauthor-CS/Amazon-Photo cells. Train six new label-free R2 alignment checkpoints for those two datasets and three seeds.

## Evidence and decision limits

The previously completed three-dataset cells are reused only after checking their run identity, target hash, input hash, probability/checkpoint hashes, and epoch log. New fits are stored separately.

Completion means all 45 cells are available and protocol checks pass. This descriptive experiment has no comparative efficacy threshold. Report seed variability and any failed cells; do not infer method superiority from absolute scores.
