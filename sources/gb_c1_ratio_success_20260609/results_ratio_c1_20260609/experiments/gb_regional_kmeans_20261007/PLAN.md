# Matched-count k-means control (2026-10-07)

## Hypothesis and role
Test whether the previously observed gain from regional distillation requires
granular-ball membership or is reproduced by another label-free geometric partition.
Regions supply a soft constraint on mean teacher/student probabilities, not hard
positive pairs, replacement labels, or prototype attraction.

## Frozen comparison
- Five datasets: cora, citeseer, corafull, coauthorcs, amazonpho.
- Uniform label noise 0.4 and 0.6; fixed split/noise/ERASE teacher seed 0.
- Student seeds 0, 1, 2: 30 new KM fits. Reuse R0/R1 from the deterministic
  dense-reduction stability experiment, including its regenerated seed-0 references.
- Use the same GRACE embedding artifact and NumPy row normalization as the original
  granular-ball builder. Fit sklearn KMeans with the actual GB count, k-means++,
  n_init=10, random_state=0, max_iter=300, tol=1e-4, algorithm=lloyd.
- One partition per dataset, shared by both noise rates and all student seeds.
  Match region count, not region-size distribution or minimum group size.
- CPU clustering with four threads; all student training on the existing ROCm GPU.
- Same MLP, initialization, dropout RNG, optimizer, 200 epochs, temperature 1,
  all-node KD, regional coefficient 1, and first strict validation maximum.
- Test labels are used only for final reporting, never selection or configuration.

## Motivation and expected mechanism
The existing random/global/scale controls and student-seed study favor meaningful
regional structure. This test isolates grouping choice within that mechanism.
Both GB and KM use CE plus sum_j |B_j|/N KL(mean teacher || mean student).
No teacher retraining, feature changes, hyperparameter search, or new losses.

## Code and configuration changes
Only a new runner, analysis, tests and records in this directory. Import the proven
dense RegionalKL implementation; retain observations and fixed-epoch snapshots.
Before new KM fits, replay Cora uniform 0.4 seed 0 with GB and require identical
selected probabilities and validation trajectory to the numerical reference.
Freeze source, input, partition and reference hashes before starting the fit grid.

## Decision rules (exploratory, not significance tests)
- Generic-region support: KM-R0 mean > 0 in each student seed and positive
  three-seed cell means in at least 6 of 10 cells.
- Evidence to prioritize GB-specific follow-up: GB-KM overall mean >= 0.20 pp,
  positive in each student seed, and positive three-seed means in >= 6/10 cells.
  This pragmatic continuation threshold is not statistical significance.
- If KM matches or exceeds GB, do not claim GB-specific accuracy superiority.
  Continue, if justified, as general region-level distillation; practical GB
  advantages would require separate controlled construction/storage evidence.
- Mixed results are reported as inconclusive, not tuned away.
- If replay/protocol/finite checks fail, stop the grid and diagnose first.

## Reporting
Save config, raw per-epoch logs, checkpoints, probabilities, source hashes, all
paired differences, sample mean/std over student seeds, group-size diagnostics,
and a detailed Chinese interpretation. Seeds vary only student randomness:
they are not three independent splits/noise realizations/teachers. Results and
final continue/modify/defer decision are appended after the frozen experiment.
