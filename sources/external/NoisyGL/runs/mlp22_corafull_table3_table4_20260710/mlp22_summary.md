# MLP-22 CoraFull Table 3/4 Completion

## Table 3 Replacement Cells

| 教师 | CoraFull u0.4 | CoraFull u0.6 |
|---|---:|---:|
| GCN(X)+CE（GLNN 式） | 58.74±0.82 | 51.63±2.02 |
| GCN[X‖H]+CE | 59.45±0.63 | 52.75±0.88 |
| 随机分组共识教师 | 58.98±0.41 | 52.20±0.72 |
| kNN 共识教师 | 58.95±0.56 | 52.73±0.95 |
| 球式共识教师（本文） | 58.22±0.69 | 51.82±0.86 |

## Table 4 Replacement Cells

| 方法 | 推理需图 | CoraFull u0.4 |
|---|---|---:|
| GCN(X) | 是 | 61.28±0.52 |
| GCN[X‖H] | 是 | 61.42±0.44 |
| NRGNN | 是 | 62.70±0.63 |
| NRGNN[X‖H] | 是 | 62.06±0.52 |
| TFR-GCN | 是 | 42.12±1.14 |
| ProCon | 是 | 36.42±3.48 |
| GD2 | 是 | 59.88±0.79 |
| LCSS 蒸馏扩展（学生） | 否 | 58.22±0.69 |

## Table 3 Paired Deltas Against Ball Teacher

| method                           | base                    |   n |   mean_delta |   positive |   negative |
|:---------------------------------|:------------------------|----:|-------------:|-----------:|-----------:|
| mlp_distill_gcn                  | mlp_distill_softdist_xh |  10 |   0.00163118 |          7 |          3 |
| mlp_distill_gcn_xh               | mlp_distill_softdist_xh |  10 |   0.0107759  |         10 |          0 |
| mlp_distill_knn_xh               | mlp_distill_softdist_xh |  10 |   0.00814868 |         10 |          0 |
| mlp_distill_softdist_xh_shuffled | mlp_distill_softdist_xh |  10 |   0.0056947  |          8 |          2 |

## Protocol Audit

- Table 3 rows: 50; all status ok: True.
- Table 4 rows: 35; all status ok: True.
- Table 3 max modified val/test: 0/0.
- Table 4 max modified val/test: 0/0.
