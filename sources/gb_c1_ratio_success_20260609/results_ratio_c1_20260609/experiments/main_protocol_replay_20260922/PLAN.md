# Main-protocol replay

## Purpose and hypothesis
Re-run the representation and sensitivity experiments through the original main-experiment execution path. Test whether restoring GPU target arithmetic and the archived execution configuration reproduces the main default results. This is a reproducibility test, not a search for favorable scores.

## Fixed method and protocol
Granular balls remain label-free local regions for tie-safe leave-one-out soft supervision. Use the five main datasets, uniform 0.4 training-label noise, seeds 0--4, archived percentage splits, frozen GRACE tensors, and the original MLP, optimizer, epoch budget, and validation selection. Do not change method defaults. Both target construction and training use the ROCm GPU; sklearn partitioning remains on CPU.

## Sources and code changes
Read the literature map, experiment history, September 19/20 reports, and immutable July main-run manifest. Use the original NoisyGL driver and archived command parameters. The old Python path no longer exists; use the available ROCm 7.2.1/PyTorch 2.9.1 environment and record its exact versions. Add only an external replay orchestrator and reports. Do not edit the original project.

## Gate
First run archived uniform-0.4 softdist commands (default plus random grouping), all five datasets and seeds. Preserve archived environment flags. Compare ordered split/noise hashes, ball assignments, embedding hashes, configuration IDs, validation/test accuracies and aggregate means/sample standard deviations. A mismatch is recorded and investigated before the sensitivity matrix is launched. Do not overwrite values or select runs by test accuracy.

## Matrix after the gate
Use the same 18 configurations as the previous supplementary study: default; beta 0, 0.25, 0.75, 1; B_max 32, 128; s_min 5, 10; n_min 1, 16, 64; delta 0.01, 0.1; MLP+CE; random size-matched grouping; raw X; A^2X. Preserve all outcomes, including losses. Record per-seed outputs, resolved settings, targets/diagnostics, losses, logs and paired summaries.

## Success, failure and decision
Success requires protocol identity and traceable paired results, not improved accuracy. Exact main-default reproduction permits a common default row; otherwise report the remaining discrepancy and do not alter the manuscript to imply identity. Recalculate every affected sensitivity delta and target diagnostic before paper integration. No paper change is part of the initial gate.

## Results
Pending execution. Final report will separate verified findings, plausible explanations and remaining uncertainties.

## Execution correction
The July 27 immutable manifest contains the self-vote control, not the final leave-one-out method. Its Cora, CiteSeer and Coauthor-CS controls were reproduced exactly, and the remaining control runs were stopped. The authoritative uniform-0.4 final-method rows are in `runs/v31_tiesafe_affected_20260728/loo_uniform0p4.csv`, consumed by the MLP44 full-spectrum summary. The formal methods end in `_loo` and `_loo_shuffled`.

The July 27 one-thread flags do not reproduce the later leave-one-out partitions on every dataset. A partition-only diagnostic with the original feature preparation and six sklearn threads reproduces the archived assignments for Coauthor-CS seed 0, CoraFull seed 0 and Amazon-Photo seed 1. The September supplementary investigation also used six grouping threads. The matrix therefore uses four Torch host threads and six threads scoped to the original KMeans grouping call; every default run must match the archived assignment, split, noise and test score before acceptance. This reconstructs execution from archived outputs; it does not establish that a historical thread manifest exists for the July 28 runs.
