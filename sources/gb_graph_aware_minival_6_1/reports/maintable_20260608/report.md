# Formal Main Table Phase 1

| dataset | noise | ce | coteaching | sel_topk | sel_dcgb | topk-coteach pp/t | topk-CE pp/t | dcgb-topk pp/t | Gate1 | Gate2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Cora | pair0.4 | 0.5920 | 0.5225 | 0.6082 | 0.5834 | 8.57/4.74 SIG | 1.63/2.19 ns | -2.48/-1.54 ns | PASS | PASS |
| Cora | pair0.6 | 0.2506 | 0.2448 | 0.2721 | 0.2592 | 2.73/1.42 ns | 2.15/1.30 ns | -1.29/-1.77 ns | PASS | PASS |
| Cora | symmetric0.4 | 0.7935 | 0.7064 | 0.7969 | 0.8002 | 9.05/8.13 SIG | 0.35/0.78 ns | 0.33/0.71 ns | PASS | PASS |
| CiteSeer | pair0.4 | 0.5956 | 0.6521 | 0.6293 | 0.6082 | -2.29/-4.30 SIG | 3.37/4.21 SIG | -2.10/-1.69 ns | FAIL | FAIL |
| CiteSeer | pair0.6 | 0.3114 | 0.3284 | 0.3088 | 0.3088 | -1.97/-0.61 ns | -0.27/-1.00 ns | 0.00/0.00 ns | FAIL | FAIL |
| CiteSeer | symmetric0.4 | 0.6979 | 0.6958 | 0.7082 | 0.7000 | 1.24/2.96 SIG | 1.03/2.00 ns | -0.82/-5.31 SIG | PASS | PASS |
