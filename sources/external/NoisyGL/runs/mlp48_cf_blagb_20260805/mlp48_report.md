# MLP-48 CF-BLAGB Phase A report

## Protocol

- Cora, CiteSeer, CoraFull; uniform 0.4; seed 0; train-only noise.
- Candidate tree is label-free; final budget equals the formal K_geo.
- Five-fold cross-fitting excludes each held-out fold from guidance and votes.
- Clean guidance is diagnostic only and cannot pass the method by itself.

## Target quality

| Dataset | Method | Acc | Brier | NCR | CDR | Balls | Exact | JSD |
|---|---|---:|---:|---:|---:|---:|---|---:|
| citeseer | cf_blagb | 0.5865 | 0.5864 | 0.0816 | 0.0135 | 64.0 | True | 0.1074 |
| citeseer | cf_blagb_clean_oracle | 0.5865 | 0.5864 | 0.0816 | 0.0135 | 64.0 | True | 0.1483 |
| citeseer | cf_blagb_shuffled_guidance | 0.5850 | 0.5871 | 0.0782 | 0.0135 | 64.0 | True | 0.1054 |
| citeseer | crossfit_adaptive_knn | 0.5910 | 0.5240 | 0.1088 | 0.0270 |  |  |  |
| citeseer | crossfit_formal_gb | 0.5865 | 0.5864 | 0.0816 | 0.0135 | 64.0 |  |  |
| citeseer | fixed_budget_geometry_only | 0.5865 | 0.5864 | 0.0816 | 0.0135 | 64.0 | True | 0.0000 |
| citeseer | formal_loo_gb | 0.5805 | 0.5733 | 0.0612 | 0.0081 | 64 |  |  |
| cora | cf_blagb | 0.6470 | 0.4919 | 0.2304 | 0.0450 | 52.0 | True | 0.1509 |
| cora | cf_blagb_clean_oracle | 0.6470 | 0.4919 | 0.2304 | 0.0450 | 52.0 | True | 0.1740 |
| cora | cf_blagb_shuffled_guidance | 0.6470 | 0.4919 | 0.2304 | 0.0450 | 52.0 | True | 0.1318 |
| cora | crossfit_adaptive_knn | 0.6617 | 0.3852 | 0.2174 | 0.0096 |  |  |  |
| cora | crossfit_formal_gb | 0.6470 | 0.4919 | 0.2304 | 0.0450 | 52.0 |  |  |
| cora | fixed_budget_geometry_only | 0.6470 | 0.4919 | 0.2304 | 0.0450 | 52.0 | True | 0.0000 |
| cora | formal_loo_gb | 0.6396 | 0.4778 | 0.1957 | 0.0322 | 52 |  |  |
| corafull | cf_blagb | 0.6016 | 0.6852 | 0.0214 | 0.0080 | 64.0 | True | 0.4701 |
| corafull | cf_blagb_clean_oracle | 0.6018 | 0.6498 | 0.0201 | 0.0068 | 64.0 | True | 0.4713 |
| corafull | cf_blagb_shuffled_guidance | 0.6016 | 0.6852 | 0.0214 | 0.0080 | 64.0 | True | 0.4675 |
| corafull | crossfit_adaptive_knn | 0.6488 | 0.5017 | 0.2393 | 0.0757 |  |  |  |
| corafull | crossfit_formal_gb | 0.5980 | 0.6330 | 0.0025 | 0.0013 | 64.0 |  |  |
| corafull | fixed_budget_geometry_only | 0.6016 | 0.6595 | 0.0214 | 0.0080 | 64.0 | True | 0.0000 |
| corafull | formal_loo_gb | 0.5978 | 0.6335 | 0.0000 | 0.0000 | 64 |  |  |

## Preregistered decision

- Pass: **False**
- Reason: `oracle_no_headroom`
- Exact-budget datasets: 3/3
- Brier better than both controls: 0/3
- Clean-oracle accuracy headroom: 1/3
- Mean accuracy delta vs cross-fit formal: +0.0012

Phase B is allowed only when every preregistered gate passes. No score, prior, fold count, or budget may be changed after this report.

## Detailed conclusion

### Hypothesis and granular-ball role

CF-BLAGB tested whether cross-fitted noisy labels could improve where a fixed granular-ball budget is spent. Labels only re-ranked label-free geometric splits; they could neither veto a split nor change the final K_geo. All other LOO-GB-LCSS target and student settings remained frozen.

### What the diagnostics show

- **cora:** candidate capacity 52, K_geo 52. CF-BLAGB changed target accuracy by +0.0000 and Brier by +0.0000 versus geometry-only; its Brier delta versus shuffled was +0.0000. The clean oracle changed Brier by +0.0000, while cross-fit adaptive-k changed it by -0.1067.
- **citeseer:** candidate capacity 65, K_geo 64. CF-BLAGB changed target accuracy by +0.0000 and Brier by +0.0000 versus geometry-only; its Brier delta versus shuffled was -0.0007. The clean oracle changed Brier by +0.0000, while cross-fit adaptive-k changed it by -0.0624.
- **corafull:** candidate capacity 128, K_geo 64. CF-BLAGB changed target accuracy by +0.0000 and Brier by +0.0257 versus geometry-only; its Brier delta versus shuffled was +0.0000. The clean oracle changed Brier by -0.0097, while cross-fit adaptive-k changed it by -0.1577.

Cora offered no alternative cut because candidate capacity equalled K_geo, so every fixed-budget arm reproduced the same partition. CiteSeer offered only one additional leaf and noisy guidance selected the geometry-only cut. CoraFull offered substantial re-ranking capacity, but real noisy and shuffled guidance produced the same target accuracy and Brier; only clean guidance selected a measurably better cut. Thus fixed budgeting repaired CF-NAGB's collapse without creating a reliable noisy-label advantage.

### Supported conclusion

The preregistered Phase A gate failed: exact budgeting passed on 3/3 datasets, but CF-BLAGB improved Brier over both geometry-only and shuffled controls on 0/3 datasets, and clean-oracle accuracy headroom appeared on only 1/3. Phase B is not triggered. Directly using the available noisy labels to alter granular-ball construction is closed under this frozen formulation.

### Interpretation boundary

The experiment does not show that labels can never help partitioning. It shows that, with 20% noisy-label coverage, a bounded JSD re-ranking signal adds no stable information beyond the frozen geometry under the current fixed budget. Teacher-denoised guidance would change the information source and belongs to a separate method rather than a second CF-BLAGB tuning round.
