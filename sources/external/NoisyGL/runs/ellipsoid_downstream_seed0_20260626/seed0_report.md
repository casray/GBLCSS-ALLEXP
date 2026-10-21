# Ellipsoid SoftDist downstream seed0 smoke

Protocol: datasets `coauthorcs, amazonpho, amazoncom`; seed `0`; uniform noise `0.4`; train noise scope; percent split `0.2/0.1/0.7`; device `cuda:0`.

## Test accuracy

| Dataset | GCN[X||H] CE | SoftDist | SoftDist shuffled | Ellipsoid | Ellipsoid shuffled | E-Soft | E-Eshuf | E-CE | Smoke |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| coauthorcs | 0.9189 | 0.9208 | 0.9199 | 0.9235 | 0.9203 | +0.0026 | +0.0032 | +0.0046 | PASS |
| amazonpho | 0.9109 | 0.8964 | 0.9001 | 0.8986 | 0.8949 | +0.0022 | +0.0037 | -0.0123 | FAIL |
| amazoncom | 0.8564 | 0.8558 | 0.8578 | 0.8558 | 0.8616 | +0.0000 | -0.0058 | -0.0006 | FAIL |

## Decision
- Smoke pass datasets: 1/3.
- Coauthor-CS passes: ellipsoid beats current SoftDist, ellipsoid shuffled, and GCN[X||H] CE on seed0.
- AmazonPhoto fails: ellipsoid beats current SoftDist and ellipsoid shuffled, but remains below GCN[X||H] CE and below ordinary SoftDist shuffled.
- AmazonCom fails: ellipsoid ties current SoftDist, loses to GCN[X||H] CE, ordinary shuffled, and ellipsoid shuffled.
- This seed0 smoke does not justify running strong baselines such as NRGNN yet. If continuing, the only reasonable next 5-seed candidate is Coauthor-CS ellipsoid vs current SoftDist/shuffled/CE.
