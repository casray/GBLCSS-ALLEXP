# GSSC adapted protocol: partial results

Paused2026-10-11 at the user's request to release GPU after Coauthor-CS finished. Only this experiment's verified grid and training processes were stopped. No matching runner remained afterward.

Protocol unchanged: five main datasets, uniform0.4, paired seeds0..4, original20/10/70 splits, training-only label corruption, common two-FC MLP, first clean-validation maximum across200 supervised epochs. Five existing seed0 records were reused;16 new runs completed. No test-based tuning or seed exclusion.

| Dataset | Completed seeds | Accuracy (%) |
| --- | --- | --- |
| Cora | 0..4 | 65.00 +/- 3.37 |
| CiteSeer | 0..4 | 54.79 +/- 2.58 |
| CoraFull | 0..4 | 40.39 +/- 1.31 |
| Coauthor-CS | 0..4 | 84.37 +/- 1.87 |
| Amazon-Photo | 0 only | 82.35 (single seed) |

Mean and sample standard deviation (ddof1) were computed on CPU from21 completed result.json files. All21 stored audits report200 supervised epochs, first validation maximum and zero checkpoint-replay difference; no GPU replay was repeated after pausing.

Coauthor-CS completes the fourth five-seed group. Amazon-Photo seed1 was already training when the pause arrived; its configuration, protocol and partial epoch log remain intact. It has no completed result and is excluded from statistics. Seeds2..4 have not started. This is a user-requested interruption, not evidence of training failure.

The experiment is incomplete: Amazon-Photo has no five-seed mean/std, and the final25-cell report has not run. These partial results alone do not establish a ranking against other methods. Resume only on user authorization, preserving the interrupted seed1 artifacts in a separate attempt strategy; the existing runner intentionally refuses partial-run overwrite.
