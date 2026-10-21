# Protocol Audit Notes (2026-07-08)

This note records protocol and reporting risks found during a code audit after the Appendix G noisy-label visualization issue. It is intended as handoff context for later Codex/Claude windows. Do not treat every item as a bug in the core method; several are reporting or baseline-protocol risks.

## Quick fixes already applied

1. `scripts/summarize_mlp14_graphfree_family.py`
   - Updated the resource annotation for `frozengrace_gb_mlp_softdist_beta05`.
   - Rationale: the deployed/downstream MLP is X-only, but frozen GRACE-H preprocessing uses the graph, all edges, and unlabeled nodes. The row now marks graph/unlabeled/edge usage as training/preprocessing resources while preserving `inference_uses_graph = 0`.

2. `scripts/generate_mlp17_paper_assets.py`
   - Updated Appendix G t-SNE plotting for the noisy-label panel.
   - Non-train nodes are now grey in the noisy panel; train nodes are colored by their noisy labels; flipped train nodes are outlined. This prevents the previous misleading impression that clean labels and train-only noisy labels are identical across the whole graph.

## Remaining risks that need explicit follow-up

### 1. Graph-dependent reference table may mix protocols

File: `scripts/generate_mlp17_paper_assets.py`

Function: `generate_graph_dependent_reference`

Risk: the function scans multiple run folders and collects CSV rows by method-name matching, but it does not strictly filter `noise_type`, `noise_rate`, `split_type`, `train_percent`, `seed`, `status`, or smoke/formal run type. It can therefore mix incompatible rows or duplicates into a citable table.

Recommended fix: replace the broad directory scan with an explicit allowlist of source CSV files and mandatory filters for the paper protocol. If a row lacks required protocol metadata, either reject it or place it in a clearly marked "reference only / non-comparable" section.

### 2. CLNode baseline uses validation labels inside training logic

File: `predictor/CLNode_Predictor.py`

Risk: the predictor fills `input_label[self.val_mask] = self.noisy_label[self.val_mask]` before difficulty measurement. Under the train-only-noise protocol, validation labels remain clean, so this can leak validation label information into training-time selection.

Recommended fix: if CLNode is reported in current paper tables, rerun a no-val-label variant or explicitly footnote that the original implementation uses validation labels and should be treated as a reference baseline, not a strict same-resource comparison.

### 3. Appendix G embedding path may not match current experiment configs

File: `scripts/generate_mlp17_paper_assets.py`

Function: `resolve_embedding_path`

Risk: Cora/CiteSeer visualization currently resolves to older `embeddings/...ratio_train0p2.../H.pt` paths, while many current runs use `embeddings_v2/.../grace/H.pt`. If the paths differ, Appendix G may visualize a different frozen H than the main experiments.

Recommended fix: resolve `gb.embedding_path` from the actual method config used by the reported run instead of hardcoding dataset-to-path mappings.

### 4. Missing `train_percent` metadata can pass filtering

File: `scripts/summarize_mlp14_graphfree_family.py`

Function: `_matches_train_percent`

Risk: rows with missing `train_percent` currently pass the filter. This was useful for old files but can reintroduce non-20% label-rate or older-split rows into main-protocol summaries.

Recommended fix: for citable main tables, require `train_percent == 0.2` unless the row comes from an explicit source allowlist with separately verified protocol metadata.

## Non-finding: core target construction appears clean

Files inspected:

- `utils/frozen_grace_targets.py`
- `predictor/FrozenGraceGB_GCN_Predictor.py`

The target builders for ball/kNN local consensus use train noisy labels for target construction. `clean_label` is passed into diagnostics such as NCR/CDR/purity reporting, but no inspected training path used clean labels to construct targets or update model parameters. These diagnostic fields must remain diagnostic-only and should not become deployment-time triggers.

## Recommended short-term priority

1. Fix graph-dependent Table C source selection before regenerating paper assets.
2. Decide whether CLNode is still reported. If yes, run/disclose a no-val-label protocol.
3. Align Appendix G H paths with the exact `gb.embedding_path` used in current configs.
4. Tighten metadata filtering before final table generation.
