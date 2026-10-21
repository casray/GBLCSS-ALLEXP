# Nearest-work full-grid experiment amendment 03

## Status

- Experiment state: `PAUSED`
- Result: `PENDING—experiments paused by user`
- This amendment authorizes only code review, CPU unit tests, mocked execution,
  dry-run command generation, and static protocol checks. It does not authorize
  training, GPU smoke runs, or formal experiment execution.

## Research hypothesis

Under the same dataset split, train-only label-noise realization, seed set,
training protocol, and X-only inference constraint, the frozen GB-LCSS v2
result can be compared fairly with PosteL, GSDN, ALS, and SALS. Whether GB-LCSS
has a stable accuracy advantage is an unverified hypothesis until the complete
grid and paired audit exist.

## Role of granular balls

Granular balls remain frozen, label-independent local semantic relation units.
They provide a confidence-bearing weak-view relation teacher and a soft
structural prior. Same-ball membership is not treated as a hard positive label,
a hard negative filter, or a mandatory prototype-pulling constraint.

## Motivation

PosteL, GSDN, ALS, and SALS directly narrow the defensible novelty boundary.
Historical results produced under different methods, seeds, splits, or
evaluation protocols cannot answer the nearest-work question. The purpose of
this grid is therefore protocol closure and independent evidence construction,
not retroactive validation of a superiority claim.

## Code changes

- Complete the non-dry-run orchestration path in
  `paper_pipeline/scripts/run_nearest_work_full_grid.py`.
- Execute 25 dataset–noise command specifications only when explicitly
  authorized; exact-complete output pairs are resumed without rerunning.
- Preserve native command CSV bytes in `raw_immutable/`, write a SHA-256
  manifest, and keep the enriched working copies separate.
- Keep the read-only tie-safe v2 source in a separate lineage and verify its
  frozen hash before producing cross-directory evidence.
- Fail immediately on a non-zero return code, missing output, incomplete key
  coverage, protocol drift, or lineage mismatch.

## Configuration changes

- Datasets: Cora, CiteSeer, Coauthor-CS, CoraFull, and Amazon-Photo.
- Noise regimes: uniform 0.2, uniform 0.4, uniform 0.6, pair 0.4, and instance
  0.4.
- Seeds: 0, 1, 2, 3, and 4.
- Recent-work methods: `gsdn_faithful_core`, `postel_mlp_noipl_rm`,
  `als_mlp_rm`, and `sals_mlp_adapted`.
- GSDN identity: `equation-grounded common-protocol adaptation`. The legacy
  identity `equation_faithful_reimplementation` is rejected by the frozen
  capability-registry gate and must not be used in reports or manuscript
  claims.
- Noise scope: training labels only.
- Split: 20% train, 10% validation, and 70% test.
- Inference resource tier: X-only.
- The training interpreter must be supplied explicitly with `--python`; an
  ROCm virtual-environment path may be supplied, but this amendment does not
  authorize executing it while the pause remains active.

## Expected mechanism

The four comparators use graph relations during training but not inference.
GSDN is an equation-grounded adaptation under the common protocol, not an
equation-faithful reproduction. Their native 500-row grid is audited
separately. For the independent main-line evidence, 375 PosteL/ALS/SALS rows
are combined with 125 read-only GB-LCSS v2 rows under matching split/noise
hashes. GSDN remains in the four-comparator cross-vs-LCSS paired analysis under
that adaptation identity.

## Success criteria

1. Exactly 25 commands produce 20 native rows each, for 500 unique recent-work
   rows.
2. Every native output has an immutable byte-for-byte copy and a complete
   SHA-256 manifest entry.
3. The independent evidence has exactly 375 PosteL/ALS/SALS rows plus 125
   frozen GB-LCSS v2 rows.
4. Completeness is 500/500 and all 125 protocol groups pass.
5. The aggregate contains 100 dataset–regime–method rows.
6. Cross-vs-LCSS output contains 500 seed-level deltas and 100 paired tests,
   with 25-test Holm correction families per comparator.
7. A second exact resume starts zero training subprocesses and marks all 25
   manifest rows `skipped_exact_complete`.
8. The frozen ours-v2 input, initial input snapshots, and initial environment
   fingerprint are not overwritten.

## Failure criteria

The experiment fails and no performance conclusion is admitted if any command
returns non-zero, any output is missing or malformed, any expected key is
missing/duplicated/extra, any method changes validation or test labels, any
split/noise/config lineage is inconsistent, any frozen hash changes, or any
comparison mixes incompatible seeds or protocols.

## Experiment result

`PENDING—experiments paused by user`

No new accuracy, ranking, variance, significance, or superiority result is
reported in this amendment.

## Final decision

`PAUSED`
