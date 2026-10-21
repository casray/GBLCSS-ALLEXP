# Posterior-SoftDist support diagnostic

- Decision: **STOP**
- Datasets: cora, citeseer, coauthorcs
- Seeds: 0, 1, 2, 3, 4
- Noise: uniform 0.4
- Split: percent, train/val/test=0.2/0.1/0.7
- Support comparison: 2-3 vs 8+; use 4-7 only when 8+ is absent.
- Gate: mean CDR gap >= 0.05 or mean clean-mass gap >= 0.02, with >= 4 positive paired seeds, on >= 2 datasets.

| Dataset | Comparator | Paired seeds | CDR gap | CDR +seeds | Mass gap | Mass +seeds | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| citeseer | 8+ | 4 | 0.1745 | 3 | -0.0112 | 1 | no |
| coauthorcs | 8+ | 4 | 0.2131 | 3 | 0.0123 | 3 | no |
| cora | 8+ | 3 | 0.0866 | 2 | 0.0026 | 1 | no |

结论：低票数球的统计可靠性缺口未达到预注册标准，停止主方法改动。
