# DIST regional KL: independent protocol seeds

Frozen before training, 2026-10-07.

## Hypothesis and motivation
Seed-0 screening found almost no average regional gain with DIST, although
CoraFull uniform .6 improved. Test generalization of the fixed regional term
instead of adding mechanisms or selecting a favorable dataset.

## Granular-ball role and mechanism
Balls define size-weighted KL between teacher and student regional mean
probabilities. They do not supply noisy-label votes or hard positive pairs.
Improved regional matching is a hypothesized mechanism, not an accuracy theorem.

## Changes and configuration
New isolated runner, protocol tests, analysis, and records; old runs unchanged.
Five datasets, uniform .4/.6, seeds 1 and 2: 20 ERASE fits and 40 student fits.
Each seed changes split, training-label corruption, teacher initialization and
student initialization. Reuse the corresponding seed's hash-verified ball
partition. The original label-free GRACE representation remains fixed per
dataset, as in the main protocol. This is NOT GRACE-pretraining-seed replication.
Within each pair: identical teacher, partition, initialization, dropout seed,
200 epochs, two-layer attribute-only MLP (64, BN, dropout .5), Adam .01,
weight decay .0005. Official DIST core, temperature 1, coefficients 1/1.
Regional coefficient 1. ERASE presets and ROCm adapters unchanged; maximum
400 epochs, patience 150. First strict clean-validation maximum checkpoint.
No label is used in student losses; test labels only in final analysis.
The historical .4 split/noise/partition and .6 split/noise hashes must match.

## Success and failure criteria
Primary: 20 paired differences, macro average over datasets, noise and seeds.
Practical continuation gate: mean >= .20 percentage points, positive macro
gain for BOTH seeds, and positive two-seed mean in >= 7/10 dataset-noise cells.
Failure of any condition pauses this fixed regional-on-DIST direction.
Report all negative cells, two-seed sample standard deviations and fixed-epoch
200 sensitivity. Two seeds do not establish significance or stable superiority.
Seed 0 is development evidence, excluded from the confirmatory mean and gate.
No tuning, no new method comparisons, no further experiments auto-launched.

## Verification and deliverables
Protocol tests; seed-0 Cora DIST +/- GB engineering replay; frozen source/input
hashes; full raw epoch logs/configs/checkpoints/probabilities; paired initial
state and RNG equality; checkpoint reload equality; result CSV, report, and
final continue/modify/pause decision. Failures are recorded, not dropped.

Results and final decision will be written separately after completion.
