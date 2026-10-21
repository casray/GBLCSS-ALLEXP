# MLP-36 latency and memory benchmark

Protocol: uniform 0.4, train-only, percent 0.2/0.1/0.7, seed 0, one ROCm GPU.
GRACE pretraining is excluded because H is reused across all downstream noise settings.
Latency is full-batch transductive inference averaged over repeated forward passes.

| Dataset | Method | Downstream wall (s) | Ball build (s) | Inference (ms/full graph) | us/node | Peak train MB | Peak inference MB |
|---|---|---:|---:|---:|---:|---:|---:|
| cora | MLP+CE | 2.156 | 0.000 | 0.3883 | 0.1434 | 126.8 | 126.2 |
| cora | LCSS | 5.201 | 4.178 | 0.3895 | 0.1438 | 126.9 | 126.2 |
| cora | GCN | 2.395 | 0.000 | 2.1705 | 0.8015 | 160.5 | 160.4 |
| citeseer | MLP+CE | 1.318 | 0.000 | 1.1207 | 0.3368 | 193.7 | 193.4 |
| citeseer | LCSS | 4.991 | 3.618 | 1.1108 | 0.3339 | 193.8 | 193.5 |
| citeseer | GCN | 2.020 | 0.000 | 2.3641 | 0.7106 | 196.2 | 196.1 |
| corafull | MLP+CE | 8.829 | 0.000 | 15.1203 | 0.7639 | 838.5 | 821.7 |
| corafull | LCSS | 13.337 | 4.489 | 14.9865 | 0.7572 | 844.1 | 827.7 |
| corafull | GCN | 7.152 | 0.000 | 10.9959 | 0.5555 | 863.7 | 846.8 |
