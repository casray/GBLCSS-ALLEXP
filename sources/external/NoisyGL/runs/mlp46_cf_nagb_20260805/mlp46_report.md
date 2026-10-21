# MLP-46 CF-NAGB Phase A report

## Protocol

- Cora, CiteSeer, CoraFull; uniform 0.4; seed 0; train-only noise.
- Candidate tree: 128 leaves, min ball size 32, geometry gain 0.01.
- Five-fold cross-fitting; held-out labels never prune or vote for their own local target.
- Clean labels are used only by diagnostic metrics and the preregistered oracle.

## Target quality

| Dataset | Method | Acc | Brier | NCR | CDR | Alpha coverage | Median balls | Geometry |
|---|---|---:|---:|---:|---:|---:|---:|---|
| cora | formal_loo_gb | 0.6396 | 0.4778 | 0.1957 | 0.0322 | 1.0000 | 52 | not_applicable |
| cora | crossfit_formal_gb | 0.6470 | 0.4919 | 0.2304 | 0.0450 | 0.9908 | 52.0 | not_applicable |
| cora | exact_knn20_loo | 0.6285 | 0.4682 | 0.1261 | 0.0000 | 1.0000 |  | not_applicable |
| cora | crossfit_knn20 | 0.6100 | 0.4788 | 0.0826 | 0.0000 | 1.0000 |  | not_applicable |
| cora | crossfit_adaptive_knn | 0.6617 | 0.3852 | 0.2174 | 0.0096 | 1.0000 |  | not_applicable |
| cora | fine_tree_crossfit | 0.6192 | 0.5053 | 0.1696 | 0.0482 | 0.9908 | 51.0 | not_applicable |
| cora | cf_nagb_shuffled_guidance | 0.5749 | 0.7677 | 0.0000 | 0.0000 | 1.0000 | 1.0 | not_applicable |
| cora | cf_nagb | 0.5749 | 0.5445 | 0.0000 | 0.0000 | 1.0000 | 7.0 | collapsed |
| cora | cf_nagb_clean_oracle | 0.5749 | 0.5170 | 0.0000 | 0.0000 | 1.0000 | 11.0 | not_applicable |
| citeseer | formal_loo_gb | 0.5805 | 0.5733 | 0.0612 | 0.0081 | 0.9940 | 64 | not_applicable |
| citeseer | crossfit_formal_gb | 0.5865 | 0.5864 | 0.0816 | 0.0135 | 0.9910 | 64.0 | not_applicable |
| citeseer | exact_knn20_loo | 0.5654 | 0.5750 | 0.0204 | 0.0027 | 1.0000 |  | not_applicable |
| citeseer | crossfit_knn20 | 0.5699 | 0.5869 | 0.0272 | 0.0000 | 1.0000 |  | not_applicable |
| citeseer | crossfit_adaptive_knn | 0.5910 | 0.5240 | 0.1088 | 0.0270 | 1.0000 |  | not_applicable |
| citeseer | fine_tree_crossfit | 0.5820 | 0.5873 | 0.0714 | 0.0135 | 0.9880 | 67.0 | not_applicable |
| citeseer | cf_nagb_shuffled_guidance | 0.5579 | 0.8490 | 0.0000 | 0.0000 | 1.0000 | 1.0 | not_applicable |
| citeseer | cf_nagb | 0.5579 | 0.6303 | 0.0000 | 0.0000 | 1.0000 | 5.0 | collapsed |
| citeseer | cf_nagb_clean_oracle | 0.5654 | 0.5935 | 0.0170 | 0.0000 | 1.0000 | 14.0 | not_applicable |
| corafull | formal_loo_gb | 0.5978 | 0.6335 | 0.0000 | 0.0000 | 1.0000 | 64 | not_applicable |
| corafull | crossfit_formal_gb | 0.5980 | 0.6330 | 0.0025 | 0.0013 | 1.0000 | 64.0 | not_applicable |
| corafull | exact_knn20_loo | 0.6198 | 0.5368 | 0.0609 | 0.0042 | 1.0000 |  | not_applicable |
| corafull | crossfit_knn20 | 0.6187 | 0.5448 | 0.0565 | 0.0030 | 1.0000 |  | not_applicable |
| corafull | crossfit_adaptive_knn | 0.6488 | 0.5017 | 0.2393 | 0.0757 | 1.0000 |  | not_applicable |
| corafull | fine_tree_crossfit | 0.6026 | 0.6031 | 0.0270 | 0.0101 | 1.0000 | 128.0 | not_applicable |
| corafull | cf_nagb_shuffled_guidance | 0.5978 | 0.7889 | 0.0000 | 0.0000 | 1.0000 | 1.0 | not_applicable |
| corafull | cf_nagb | 0.5978 | 0.7546 | 0.0000 | 0.0000 | 1.0000 | 3.0 | collapsed |
| corafull | cf_nagb_clean_oracle | 0.6011 | 0.6156 | 0.0119 | 0.0025 | 1.0000 | 72.0 | not_applicable |

## Preregistered decision

- Pass: **False**
- Reason: `geometry_failed`
- Noisy guidance vs formal: 0/3
- Noisy guidance vs shuffled: 3/3
- Clean oracle headroom: 1/3

Phase B may run only when all Phase A gates pass. No thresholds were changed after observing these diagnostics.

## Detailed conclusion

### Hypothesis and granular-ball role

The experiment tested whether the 20% noisy training labels could select a better level of a label-free GRACE geometry tree. Granular balls remained local semantic units, while cross-fitted noisy labels were allowed only to prune the candidate hierarchy. A held-out node never contributed its label to its own pruning or local vote.

### What happened

The geometry tree itself was healthy: it produced 51, 67, and 128 candidate leaves on Cora, CiteSeer, and CoraFull. The label-guided Bayes pruning was not healthy. Real noisy guidance retained median partitions of only 7, 5, and 3 balls. Shuffled guidance retained one ball on every fold, while clean guidance retained 11, 14, and 72 balls. Thus the method did react to label information, but the available noisy evidence was too weak to preserve a useful regional resolution.

Against the same-fold fixed geometric partition, CF-NAGB changed the mean target metrics as follows:

- target accuracy: -0.0336
- Brier score: +0.0727 (worse)
- noise correction rate: -0.1049
- clean damage rate: -0.0199

The lower clean damage rate is not a useful win here. CF-NAGB rarely changed the target argmax at all, so both NCR and CDR approached zero. Its targets became coarse smoothing distributions rather than effective corrections.

Real guidance beat shuffled guidance on Brier in all three datasets, confirming that the labels carried some non-random regional signal. This signal was still insufficient: real guidance passed neither the geometry gate nor the comparison with cross-fit formal GB. The clean-guidance oracle showed preregistered headroom on only one of three datasets, so label noise is not the sole cause of failure.

Cross-fit formal GB remained viable, which isolates the failure to label-guided pruning rather than cross-fitting. Cross-fit adaptive-k produced the strongest target accuracy/Brier on these diagnostics, especially on CoraFull, but also increased CoraFull clean-label damage. No downstream claim is made from target diagnostics alone.

### Final decision

Phase A failed. No classifier was trained and Phases B-D were not triggered. The thresholds and prior were not retuned after observing the result. The current CF-NAGB definition is closed: under 20% noisy-label coverage, Dirichlet-multinomial pruning over-merges the candidate tree and degrades target quality relative to the fixed geometric GB partition.
