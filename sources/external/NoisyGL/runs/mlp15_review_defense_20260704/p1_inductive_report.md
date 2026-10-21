# MLP-15 P1 Inductive Probe

Protocol: observed-subgraph GRACE retraining, inductive_test_ratio=0.2, noise=uniform0.4, train split=0.2.

## GRACE Ball Diagnostics

| dataset   |   seed |   true_train_purity_weighted |   true_all_purity_weighted |   noisy_train_purity_weighted |   num_balls |   observed_count |   inductive_count |   grace_epochs |   grace_loss_last |
|:----------|-------:|-----------------------------:|---------------------------:|------------------------------:|------------:|-----------------:|------------------:|---------------:|------------------:|
| citeseer  |      0 |                        0.218 |                      0.208 |                         0.202 |           1 |             2861 |               466 |            200 |           7.59022 |
| cora      |      0 |                        0.32  |                      0.304 |                         0.226 |           1 |             2329 |               379 |            200 |           6.47925 |

