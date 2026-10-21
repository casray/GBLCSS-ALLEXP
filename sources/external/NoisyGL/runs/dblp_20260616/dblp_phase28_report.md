# Phase 28: DBLP Dataset-Breadth Gate

Date: 2026-06-16

## Purpose

DBLP was selected as the first dataset-breadth gate because it is a homophilous citation graph, appears in NRGNN's original evaluation, and should be the most favorable new dataset for SoftDist [X||H].

## Protocol

- Dataset: DBLP
- Split: `percent`, train/val/test = `0.2/0.1/0.7`
- Noise: `uniform`, noise rate = `0.4`
- Seeds for ball diagnostics: `0,1,2,3,4`
- Downstream smoke seed: `0`
- GPU Python: `C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`

## GRACE Pretraining

The execution plan suggested a PubMed fallback if no dataset-specific GRACE configuration was available. The upstream GRACE `config.yaml` already contains an explicit DBLP configuration, so Phase 28 used the official DBLP GRACE defaults instead of transferring PubMed hyperparameters.

- Command artifact: `gb_graph_aware_minival_6_1/logs/phase28_grace_dblp_pretrain_20260616.out.log`
- Summary: `gb_graph_aware_minival_6_1/checkpoints/embeddings_v2/phase28_grace_dblp_summary_20260616.csv`
- Embedding: `gb_graph_aware_minival_6_1/checkpoints/embeddings_v2/DBLP/grace/H.pt`
- Shape: `17716 x 256`
- Finite check: passed
- Row L2 norm: mean `1.0000`, min `0.9999998`, max `1.0000001`
- Final pretrain loss: `9.1554`

This verifies the basic H-file shape and node-count alignment with the NoisyGL DBLP graph. It does not prove semantic quality, which is measured by the ball diagnostics below.

## Ball Diagnostics

Source: `runs/dblp_20260616/dblp_ball_diagnostics.csv`

| Metric | Mean | Std |
|---|---:|---:|
| num_balls | 1.0000 | 0.0000 |
| mean_ball_size | 17716.0000 | 0.0000 |
| true_all_purity_weighted | 0.4471 | 0.0000 |
| soft_pseudo_ncr | 0.4446 | 0.0157 |
| soft_pseudo_cdr | 0.5545 | 0.0082 |
| mean_alpha_train | 0.1262 | 0.0089 |

The DBLP GRACE embedding collapses into a single geometric ball under the registered SoftDist ball construction (`max_balls=64`, `min_ball_size=32`, `split_improvement=0.05`). This triggers the pre-registered no-go condition `num_balls < 8`.

The high NCR is not useful here because it comes from applying one global dataset-level distribution to every train node. The CDR is also extremely high, meaning many originally clean labels would be pushed toward the global majority distribution.

## Downstream Seed0 Smoke

Source:

- `runs/dblp_20260616/dblp_gcn_seed0_download_smoke.csv`
- `runs/dblp_20260616/dblp_softdist_seed0_smoke.csv`
- `runs/dblp_20260616/dblp_nrgnn_seed0_smoke.csv`

| Method | Test acc |
|---|---:|
| GCN + CE | 0.8160 |
| GCN [X||H] | 0.8300 |
| SoftDist [X||H] | 0.8307 |
| SoftDist [X||H] shuffled | 0.8307 |
| NRGNN | 0.8136 |
| NRGNN [X||H] | 0.8340 |

Seed0 confirms the diagnostic: SoftDist and shuffled are identical because the real ball structure has collapsed to one global ball. The useful signal on DBLP is H as an input feature, not granular-ball structure.

## Decision

Phase 28 result: **no-go for DBLP as a third successful SoftDist dataset**.

Reasons:

- `num_balls = 1` across all five diagnostic seeds.
- Real SoftDist equals shuffled in seed0 downstream smoke.
- Ball target becomes a global distribution rather than local semantic correction.
- NRGNN [X||H] is stronger than SoftDist [X||H] in seed0.

Recommended next action:

- Do not expand DBLP to a full 5-seed downstream matrix.
- Do not use DBLP as evidence for dataset-breadth success.
- Use DBLP as another boundary case supporting the claim that SoftDist requires a GRACE embedding that is not only linearly useful but also locally clusterable.
- Phase 29 should be reduced in scope unless the goal is purely to document boundaries.
- Phase 30 noise-spectrum completion remains orthogonal and can be run separately on Cora/CiteSeer.

