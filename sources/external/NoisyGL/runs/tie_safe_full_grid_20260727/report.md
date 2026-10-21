# Tie-safe 完整网格运行报告

- 生成时间（UTC）：2026-07-27T12:16:52.551662+00:00
- 冻结范围：5 datasets × 5 regimes × 5 seeds × 4 methods = 500 rows。
- 命令拆分：25 个 SoftDist 双方法命令 + 25 个 baseline 双方法命令。
- 完成状态：{"duplicate": 0, "failed": 0, "missing": 0, "ok": 500}。
- 协议 PASS：125/125。
- 聚合行：100/100；配对检验行：75/75。
- Holm family：每个 comparator 独立校正 25 个 dataset×regime tests。

## Holm 结果

| comparator | significant / 25 |
|---|---:|
| frozengrace_gb_mlp_softdist_beta05_shuffled | 25 / 25 |
| mlp_ce | 24 / 25 |
| mlp_jocor | 8 / 25 |

## Claim 边界

- 只有 500/500 行完整且 125/125 协议组 PASS 时，聚合结果才可进入新主表。
- 配对 t 检验和 Holm 校正不替代效应量、sample std 或逐 seed 差值。
- 真实球与 shuffled 球的 assignment hash 允许不同；两者均必须非空。
- 任一 missing、failed、duplicate 或协议漂移均保留并拒绝主表闭合。
