# GRACE-first GAGB Grouping Diagnostic

Rows: 220
Summary rows: 44

This script does not train a classifier. It rebuilds groupings and uses build_method_targets for target diagnostics.

## Purpose

This diagnostic tests the v2.5 GRACE-first topology-refined granular ball design before running a full classifier gate. The hypothesis was that selecting candidate members in frozen GRACE space and refining them with graph similarity would produce higher-quality graph-aware balls than current GAGB, ordinary GB, topology-only balls, and shuffled controls.

Clean labels are used only for diagnostics. Grouping, noisy teacher training, and target construction use embeddings, graph topology, noisy train labels, and train masks only.

## Granular Ball Role

The granular ball is treated as a local semantic-relation unit and reliability signal, not as a hard positive label or hard correction target. In v2.5 the ball first retrieves candidates from frozen GRACE embedding space, then applies topology support and ball quality checks before assigning members.

## Code Changes

- Added summary support for grace_gagb methods and gfgb diagnostics in src/noisy_label_gb_grace/evaluation/summary.py.
- Exposed grace_gagb methods in scripts/run_gate_experiment.py CLI choices and defaults.
- Added scripts/diagnose_grace_first_gagb_grouping.py for grouping-only diagnostics without classifier training.
- Added unit tests for GRACE-first grouping behavior, method routing, CLI exposure, summary paired deltas, and diagnostic-script target reuse.

## Expected Mechanism

The intended mechanism was:

1. GRACE embedding retrieves semantically plausible candidates.
2. Topology support filters or weakly validates candidates.
3. Ball quality diagnostics identify reliable local regions.
4. Existing soft-label and conflict-gate target construction uses the new grouping without changing target formulas.

## Success Criteria

- grace_gagb should improve clean_purity or noise_detection_auroc over gagb in at least 3/4 dataset-noise settings.
- clean_damage_rate should not exceed gagb by more than 0.01.
- grace_gagb should not be dominated by khop_only, embedding-only, topk matched, or shuffled controls.
- A mechanism-positive but not method-ready outcome is allowed if AUROC improves while purity/NCR remain weak.

## Failure Criteria

- topk_grace_gagb_train_matched consistently beats grace_gagb on the same diagnostic signal.
- grace_gagb clean_purity is consistently below gb/gagb.
- target repair fails to produce positive NCR.
- topology refinement does not clearly improve over grace_gagb_emb_only.

## Protocol

- Datasets: Cora, CiteSeer
- Split: ratio, train_ratio=0.2, val_ratio=0.1, split_seed=0
- Noise: symmetric, rates 0.2 and 0.4
- Seeds: 0, 1, 2, 3, 4
- Embedding: frozen GRACE embedding H.pt
- Methods: gb, gagb, khop_only, grace_gagb, grace_gagb_emb_only, grace_gagb_topo_only, kmeans_grace_gagb_matched, topk_grace_gagb_train_matched, grace_gagb_shuffled_alpha_matched, grace_gagb_qb_conflict_gate_alpha03, grace_gagb_qb_conflict_gate_alpha07
- Outputs: per_seed_grouping_diagnostics.csv, grouping_summary.csv, run_config.json

## Key Aggregated Results

| Dataset | Noise | Method | Clean purity | Noise AUROC | NCR | CDR |
|---|---:|---|---:|---:|---:|---:|
| Cora | 0.2 | gb | 0.893 | 0.682 | 0.000 | 0.000 |
| Cora | 0.2 | gagb | 0.893 | 0.650 | 0.000 | 0.000 |
| Cora | 0.2 | khop_only | 0.927 | 0.571 | 0.000 | 0.000 |
| Cora | 0.2 | grace_gagb | 0.699 | 0.890 | 0.000 | 0.000 |
| Cora | 0.2 | grace_gagb_emb_only | 0.681 | 0.882 | 0.000 | 0.000 |
| Cora | 0.2 | grace_gagb_topo_only | 0.992 | 0.514 | 0.000 | 0.000 |
| Cora | 0.2 | topk_grace_gagb_train_matched | 0.699 | 0.969 | 0.000 | 0.000 |
| Cora | 0.4 | gb | 0.893 | 0.645 | 0.000 | 0.000 |
| Cora | 0.4 | gagb | 0.893 | 0.619 | 0.000 | 0.000 |
| Cora | 0.4 | khop_only | 0.927 | 0.551 | 0.000 | 0.000 |
| Cora | 0.4 | grace_gagb | 0.699 | 0.850 | 0.000 | 0.000 |
| Cora | 0.4 | grace_gagb_emb_only | 0.681 | 0.844 | 0.000 | 0.000 |
| Cora | 0.4 | grace_gagb_topo_only | 0.992 | 0.508 | 0.000 | 0.000 |
| Cora | 0.4 | topk_grace_gagb_train_matched | 0.699 | 0.948 | 0.000 | 0.000 |
| CiteSeer | 0.2 | gb | 0.797 | 0.606 | 0.000 | 0.000 |
| CiteSeer | 0.2 | gagb | 0.876 | 0.567 | 0.000 | 0.000 |
| CiteSeer | 0.2 | khop_only | 0.908 | 0.535 | 0.000 | 0.000 |
| CiteSeer | 0.2 | grace_gagb | 0.635 | 0.811 | 0.000 | 0.000 |
| CiteSeer | 0.2 | grace_gagb_emb_only | 0.613 | 0.801 | 0.000 | 0.000 |
| CiteSeer | 0.2 | grace_gagb_topo_only | 0.941 | 0.508 | 0.000 | 0.000 |
| CiteSeer | 0.2 | topk_grace_gagb_train_matched | 0.635 | 0.891 | 0.000 | 0.000 |
| CiteSeer | 0.4 | gb | 0.797 | 0.582 | 0.000 | 0.000 |
| CiteSeer | 0.4 | gagb | 0.876 | 0.554 | 0.000 | 0.000 |
| CiteSeer | 0.4 | khop_only | 0.908 | 0.531 | 0.000 | 0.000 |
| CiteSeer | 0.4 | grace_gagb | 0.635 | 0.777 | 0.000 | 0.000 |
| CiteSeer | 0.4 | grace_gagb_emb_only | 0.613 | 0.769 | 0.000 | 0.000 |
| CiteSeer | 0.4 | grace_gagb_topo_only | 0.941 | 0.506 | 0.000 | 0.000 |
| CiteSeer | 0.4 | topk_grace_gagb_train_matched | 0.635 | 0.871 | 0.000 | 0.000 |

## Interpretation

The v2.5 grouping has a real diagnostic signal: grace_gagb improves noise_detection_auroc over current gagb and gb in all four dataset-noise settings. This supports the idea that frozen GRACE space contains useful information for identifying suspicious noisy labels.

However, the same grouping does not yet produce high-quality granular balls. Clean purity is much lower than GB/GAGB on all four settings. The topology-only variant has very high purity but near-random AUROC, while the GRACE-first and embedding-only variants are close to each other. This suggests the current topology refinement is too weak to improve semantic grouping quality beyond embedding retrieval.

The strongest warning is the matched topk control: topk_grace_gagb_train_matched is better than grace_gagb on AUROC in all four settings while using the same train-size matching idea. This weakens the claim that the current GRACE-first ball construction has independent value beyond local neighbor smoothing in embedding space.

The target-repair variants have NCR=0 in this grouping-only diagnostic, so the current q_B / conflict-gate target construction is not converting the AUROC signal into argmax-level correction.

## Decision

Do not claim v2.5 as a successful graph-aware granular-ball algorithm yet. It is mechanism-positive for noise detection, but not ball-quality-positive and not correction-positive.

The next useful step is not a long full classifier gate over all methods. The next algorithmic step should add ball-quality filtering or reliability Q(B) before target repair, especially enforcing a minimum clean-free proxy for semantic compactness and topology support. A classifier gate is still useful as a small sanity check, but the current grouping diagnostic already says the main GAGB-specific claim is not ready.

Final decision: modify before continuing. Do not promote v2.5 to a full paper method. Use these results to design a Q(B)-filtered or reliability-gated v2.6.
