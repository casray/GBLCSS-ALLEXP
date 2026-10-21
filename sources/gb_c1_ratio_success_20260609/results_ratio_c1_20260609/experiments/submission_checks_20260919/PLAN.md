# GB-LCSS submission checks (2026-09-19)

## Material Passport
- Mode: run; computational experiments on public graph datasets.
- Status: protocol frozen before inspecting new outcomes.
- Sources: current `D:/papercode_final_v4` formal implementation, archived frozen GRACE representations, and original NoisyGL public-data cache.
- Existing evidence read: project literature map and experiment history; MLP-44, MLP-50 and MLP-51 reports. The old self-inclusive sensitivity experiments are not used as current-method evidence.

## Research hypothesis and role of granular balls
The current local-consensus mechanism should be evaluated for sensitivity to its fixed parameters, without replacing it or selecting a new setting from test accuracy. Granular balls remain label-free shared regions, not true classes. More votes can improve averaging but larger regions can mix classes; stronger sharpening can either help or amplify an incorrect majority.

## Frozen matrix
- Datasets: Cora, CiteSeer, CoraFull.
- Noise: uniform 0.4, training labels only; clean validation; percent 0.2/0.1/0.7 split.
- Downstream seeds: 0, 1, 2, 3, 4; one archived frozen representation per dataset.
- Formal target: leave-one-out, `distribution_only` ties, normalized multiclass consensus weight.
- Reference: beta=0.5, B_max=64, n_min=32, delta=0.05, s_min=2.
- One factor at a time: beta={0,0.25,0.5,0.75,1}; B_max={32,64,128}; s_min={2,5,10}; n_min={1,16,32,64}; delta={0.01,0.05,0.1}.
- Controls: MLP+CE, random size-matched grouping, raw X, raw A^2 X (no self-loop or adjacency normalization).
- 18 unique arms x 3 datasets x 5 seeds = 270 logical evaluations. Bit-identical targets within a dataset/seed may reuse one deterministic training result, explicitly recorded as reuse, not independent replication.
- Training: unchanged 2-layer MLP with hidden width 64, batch normalization and dropout 0.5; Adam lr=0.01, weight decay=0.0005; 200 epochs; first highest clean-validation-accuracy checkpoint.

## Implementation and provenance
Only a new driver, tests and configuration are added in this directory. Existing source repositories and manuscripts are unchanged. The driver snapshots the imported implementation and configurations. It records environment, ordered split/noise/representation/target fingerprints, raw results, loss curves, partition statistics, and complete aggregates. Historical numbers are not substituted for fresh controls. GPU results reside in `gpu/`; the four CPU smoke evaluations remain separate.

## Quality gates and interpretation rules
1. Existing algorithm tests must pass before training.
2. Ordered split and noise fingerprints must match archived examples; noisy-label Brier must agree with archived Figure 4 within 2e-5. Historical default-target agreement is recorded separately under the provenance amendment below. All parameter comparisons use freshly rerun defaults, not historical accuracy.
3. Every arm uses the same split, noise, initialization seed, training budget and selection rule. Validation/test labels are unchanged by corruption. Clean labels enter diagnostics only, never grouping/target construction/model selection beyond the existing clean validation protocol.
4. Report accuracy mean and sample SD, paired seed differences, target Brier, corrected-corrupt versus perturbed-clean contributions, eligible support, actual ball count and label composition. Do not call differences smaller than run variability significant.
5. Descriptive sensitivity: report the full range and all boundary failures; no best-test-setting selection and no change to paper defaults. A loss of >2 percentage points versus the reference flags material sensitivity, not grounds for discarding a result.
6. Representation controls receive five downstream seeds. Equality of raw-X and A^2X targets is checked rather than assumed. This is not a comparison against all alternative graph encoders.
7. The new theory is conditional; measurements do not turn it into a guarantee of generalization or a proof that granular balls outperform kNN.

## Success, failure and decision
Success means the complete fixed matrix and controls are runnable, comparable and traceable; positive accuracy changes are not required. Protocol mismatch is a validity failure and halts the affected run. An unfavorable parameter setting is a scientific result and stays in the report. Final decision will distinguish support for the current defaults, sensitivity/boundary evidence, and unresolved environment differences. No new algorithm, OGB expansion or additional baseline training is included in this batch.

## GPU and provenance amendment (before any GPU training outcome)
- User requested the previous GPU environment. Verified `D:/codex_temp/noisygl-rocm721/Scripts/python.exe`, PyTorch 2.9.1+rocm7.2.1, AMD Radeon RX 7700 XT, forward/backward on cuda:0. Training must fail rather than fall back to CPU. sklearn 2-means remains a CPU operation.
- Cora seed-0 default target matches Figure 4. CoraFull seed-0 partition does not: current single-thread grouping gives target Brier 0.6326618525 versus archived 0.6334725022. Ordered split, corrupted labels and frozen representation match. Current submitted and original builders agree exactly, as do target tensors for a shared assignment.
- KMeans thread changes alter the CoraFull partition, confirming numerical sensitivity. The exact historical cause is not fully isolated. Do not select a thread count to obtain a favorable result or silently overwrite the archived row. Keep one CPU thread for all grouping runs and four Torch host threads for all GPU training runs.
- Therefore this batch is a fresh, internally paired supplementary study, not exact regeneration of the archived paper table. The old strict archival-target gate failed; relaxing it does not establish historical reproducibility. Report default-target discrepancies for all seeds. Reconcile provenance before inserting these outcomes into the final paper.
- Original random size-matched control uses seed+17; the new driver follows that rule. No CPU random-control result was produced under the initial seed-only draft.

### Resolution before the full GPU matrix
The diagnostic with six grouping threads reproduces the archived CoraFull seed-0 assignment exactly (`3ef5c501...`), with Brier 0.6334725002 (archive 0.6334725022). One, two, four and twelve threads do not. This is an archival-fingerprint check, not accuracy-based parameter selection. Restore the strict 2e-5 default-target gate for every dataset/seed and fix grouping threads to six. Use `gpu_reference_matched/` for all final runs; `gpu/` contains only the earlier four smoke evaluations. GPU smoke Cora default accuracy 56.939% and raw-X 42.691% reproduce the archived seed-0 figures. All 15 default target checks must pass before the full matrix.

## Final status
All 270 logical evaluations completed (214 GPU training runs, 56 exact-target reuses). The 15 default Brier checks passed, with maximum difference 6.92e-8 from archive. Reverse-order replay of two Cora runs reproduced logits exactly. The full matrix and conclusions are in `REPORT_zh.md` and `gpu_reference_matched/RESULTS.md`.

Some historical accuracies differ despite matching mathematical targets. Two targeted checks traced the largest discrepancies to CPU versus GPU target arithmetic (maximum element difference 5.96e-8); GPU-constructed original targets restored those two archived accuracies. These checks remain separate from the matrix, which consistently uses CPU target construction followed by GPU MLP training. Do not mix paths or replace paper defaults with the best test setting. Keep current method; add sensitivity and representation diagnostics only after author review.
