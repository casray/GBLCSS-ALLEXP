# MLP-52 Cross-Partition Vote Pooling report

| Dataset | Noise | Method | Acc | Brier | NCR | CDR |
|---|---:|---|---:|---:|---:|---:|
| citeseer | 0.4 | mean_target_gb_3h | 0.5669 | 0.5606 | 0.0238 | 0.0027 |
| citeseer | 0.4 | pooled_gb_3h | 0.5639 | 0.5817 | 0.0170 | 0.0027 |
| citeseer | 0.4 | pooled_gb_3h_shuffled | 0.5579 | 0.8016 | 0.0000 | 0.0000 |
| citeseer | 0.4 | pooled_knn_3h | 0.5639 | 0.5717 | 0.0170 | 0.0027 |
| citeseer | 0.4 | single_h0_gb | 0.5774 | 0.5913 | 0.0714 | 0.0216 |
| citeseer | 0.6 | mean_target_gb_3h | 0.4135 | 0.8549 | 0.0000 | 0.0000 |
| citeseer | 0.6 | pooled_gb_3h | 0.4150 | 0.8970 | 0.0026 | 0.0000 |
| citeseer | 0.6 | pooled_gb_3h_shuffled | 0.4135 | 1.0581 | 0.0000 | 0.0000 |
| citeseer | 0.6 | pooled_knn_3h | 0.4150 | 0.8583 | 0.0051 | 0.0036 |
| citeseer | 0.6 | single_h0_gb | 0.4286 | 0.8868 | 0.0256 | 0.0000 |
| cora | 0.4 | mean_target_gb_3h | 0.6081 | 0.4665 | 0.0783 | 0.0000 |
| cora | 0.4 | pooled_gb_3h | 0.6007 | 0.4925 | 0.0609 | 0.0000 |
| cora | 0.4 | pooled_gb_3h_shuffled | 0.5749 | 0.7573 | 0.0000 | 0.0000 |
| cora | 0.4 | pooled_knn_3h | 0.6229 | 0.4449 | 0.1130 | 0.0000 |
| cora | 0.4 | single_h0_gb | 0.6137 | 0.4879 | 0.1391 | 0.0354 |
| cora | 0.6 | mean_target_gb_3h | 0.3752 | 0.8801 | 0.0059 | 0.0000 |
| cora | 0.6 | pooled_gb_3h | 0.3715 | 0.9249 | 0.0000 | 0.0000 |
| cora | 0.6 | pooled_gb_3h_shuffled | 0.3715 | 1.1292 | 0.0000 | 0.0000 |
| cora | 0.6 | pooled_knn_3h | 0.3789 | 0.9070 | 0.0118 | 0.0000 |
| cora | 0.6 | single_h0_gb | 0.3789 | 0.9028 | 0.0147 | 0.0050 |
| corafull | 0.4 | mean_target_gb_3h | 0.5978 | 0.6054 | 0.0000 | 0.0000 |
| corafull | 0.4 | pooled_gb_3h | 0.5978 | 0.6326 | 0.0000 | 0.0000 |
| corafull | 0.4 | pooled_gb_3h_shuffled | 0.5978 | 0.7810 | 0.0000 | 0.0000 |
| corafull | 0.4 | pooled_knn_3h | 0.5993 | 0.5860 | 0.0038 | 0.0000 |
| corafull | 0.4 | single_h0_gb | 0.5993 | 0.6266 | 0.0050 | 0.0008 |
| corafull | 0.6 | mean_target_gb_3h | 0.3982 | 0.9848 | 0.0000 | 0.0000 |
| corafull | 0.6 | pooled_gb_3h | 0.3982 | 1.0194 | 0.0000 | 0.0000 |
| corafull | 0.6 | pooled_gb_3h_shuffled | 0.3982 | 1.1735 | 0.0000 | 0.0000 |
| corafull | 0.6 | pooled_knn_3h | 0.3982 | 0.9640 | 0.0000 | 0.0000 |
| corafull | 0.6 | single_h0_gb | 0.3982 | 1.0059 | 0.0000 | 0.0000 |

## Preregistered gate

- PASS: `0`.
- Near-pass: `0`.
- Reason: `single_gb_brier_gate_failed`.
- Brier wins vs single-H GB: `1/6`.
- Pareto wins vs pooled kNN: `0/6`.
- Accuracy recovery vs mean-target averaging: `3/6`.
- Brier wins vs pooled shuffled GB: `6/6`.
- Accuracy regressions vs single-H GB: `['citeseer:0.4', 'citeseer:0.6', 'cora:0.4', 'cora:0.6']`.

Phase A failed; no downstream training is authorized.

## Detailed interpretation

Pooling leave-one-out votes before forming the target was intended to recover the corrections diluted by final-target averaging. It did not. Relative to the formal single-H GB target, pooled GB improves Brier score in only 1/6 cells and causes target-accuracy regressions larger than 0.5 percentage points in four cells. It never achieves a Pareto win over the same pooled-vote kNN construction.

The failure is mechanistically informative. Combining votes from three broad, mutually exclusive partitions produces an effective neighborhood substantially larger and more diffuse than the medium-scale consensus units that previously worked. The pooled distribution reduces class concentration and therefore weakens alpha; it does not selectively add independent clean evidence. The real assignment remains better than shuffled in 6/6 cells, so local geometry still matters, but the proposed pooling rule over-smooths that signal.

Cross-partition raw-vote pooling is therefore closed. A future variant must introduce genuinely new information or a predeclared reliability model; changing the pooling normalization, target beta, or ball size after this result would be post-hoc tuning of the same failed mechanism.
