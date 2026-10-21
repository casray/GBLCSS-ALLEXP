# Phase 37d Deployable Proxy Execution Log

## Objective

Test whether ball gains can be predicted by deployable proxies computed before test evaluation, using only frozen GRACE geometry `H` and noisy train labels. Clean labels and test results are used only as already-computed target deltas from Phase 37c.

## Commands

```powershell
$py='C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe'
& $py scripts\extract_deployable_proxies.py --datasets cora,citeseer,pubmed,amazonpho,dblp,coauthorcs,amazoncom,coauthorph,wikics --seeds 0,1,2 --device cuda:0 --output_dir runs\deployable_proxy_20260617
```

The first full run hit the 20-minute shell timeout, but it had already written the complete raw matrix: 9 datasets x 3 seeds x 2 proxy configs = 54 rows. Postprocessing was regenerated from the raw CSV:

```powershell
$py='C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe'
& $py -m py_compile scripts\extract_deployable_proxies.py
& $py scripts\extract_deployable_proxies.py --postprocess_only --datasets cora,citeseer,pubmed,amazonpho,dblp,coauthorcs,amazoncom,coauthorph,wikics --seeds 0,1,2 --output_dir runs\deployable_proxy_20260617
```

## Data Availability

- Coauthor-Physics is marked `missing_h`; no GRACE embedding was available, so H-dependent proxies were not fabricated.
- Cora/CiteSeer have no `gbselect` method config, so those rows are marked `missing_config`.
- All other requested H-backed/config-backed rows completed.

## Main Results

37d-pre did not show numerical wash-flat:

- noisy train purity min = 0.2259
- noisy train purity max = 0.5534
- range = 0.3275

However, noisy purity alone is not yet a reliable deployable trigger:

- SoftDist target, `noisy_train_purity_weighted`: Spearman = 0.3000, Pearson = 0.6358, n = 5
- GBSelect target, `noisy_train_purity_weighted`: Spearman = -0.0286, Pearson = 0.8205, n = 6

The strongest current deployable candidates are:

- GBSelect vs GCN-only: `mean_vote_margin_weighted`, Spearman = 0.8286, Pearson = 0.8525, n = 6
- GBSelect vs GCN-only: `mean_vote_entropy_weighted`, Spearman = -0.7714, Pearson = -0.7577, n = 6
- GBSelect vs GCN-only: `ball_gcn_disagree_confident_train`, Spearman = -1.0000, Pearson = -0.9511, n = 5
- SoftDist vs GCN[X||H]: `ball_gcn_original_label_disagree_train`, Spearman = -0.9000, Pearson = -0.9688, n = 5

## Decision

The maximum available correlation sample is n = 6, below the pre-registered n >= 7 threshold. Therefore Phase 37d is underpowered: it cannot legally pass or fail the adaptive-trigger criterion.

Do not implement the adaptive layer yet. The current evidence supports the weaker claim that deployable noisy-label/geometry proxies contain signal, especially vote margin and GCN-ball disagreement, but more target deltas are required before turning them into an algorithmic switch.

## Claim Boundary

Allowed wording:

- Evidence suggests ball-clusterability may mediate ball gains.
- Deployable noisy-label/geometry proxies show preliminary association with ball gains.
- Current sample size is too small to define a reliable adaptive trigger.

Not allowed:

- Ball purity is the determining causal variable.
- The adaptive trigger is validated.
- Noisy purity alone is sufficient as a deployment-time switch.

