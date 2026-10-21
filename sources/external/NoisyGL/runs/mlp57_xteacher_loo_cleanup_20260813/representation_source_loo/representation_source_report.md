# Formal LOO representation-source diagnostic

All target rows use distribution_only and leave_one_out=true.
Raw X and A2X are stopped at seed0 because both form one ball on all three datasets.

| dataset | GRACE H (5 seeds) | raw X seed0 | A2X seed0 | raw/A2X balls |
|---|---:|---:|---:|---:|
| cora | 57.49+/-0.76 | 42.69 | 42.69 | 1/1 |
| citeseer | 55.24+/-2.04 | 43.08 | 43.08 | 1/1 |
| corafull | 37.39+/-0.51 | 31.87 | 31.87 | 1/1 |
