# MLP-8 P2a Rescue Downstream Plan

日期：2026-07-03
定位：P2a 窄验证，只检查 `GRACE + 0.1 noisy CE` 表征在放松球构造参数后是否有下游收益；不重启 P2b loop，不改变主方法 anchor。

## 研究假设

`GRACE + 0.1 noisy CE` 表征在默认 `split_improvement=0.05` 下塌缩成单球，导致 MLP-8 P2a 被判负。P2 ball-parameter rescue 诊断显示该表征在 `max_balls=64, min_ball_size=16, split_improvement=0.0` 下恢复到 64 个球，且 `true_all_purity=0.8117`，高于 frozen-GRACE anchor 的 `0.7873`。因此需要一个 seed0 下游验证，确认几何恢复能否转化为分类收益。

## 粒球角色

粒球仍只作为 frozen representation 上的局部共识 soft-target builder。P2a 的唯一变化是构球表征从 frozen GRACE 换成 `GRACE + 0.1 noisy CE`，并使用诊断过门的 loose ball 参数。

## 协议

- Dataset：Cora
- Noise：train-only uniform 0.4
- Split：percent 0.2/0.1/0.7
- Seed：0
- Backbone：X-only 2-layer MLP，沿用 `frozengrace_gb_mlp_softdist_beta05`
- P2a embedding：`runs/mlp8_p2_advisor_smoke_20260702/embeddings/Cora/grace_ce_l0p1/H.pt`
- Loose ball params：`max_balls=64, min_ball_size=16, split_improvement=0.0`

## 实验臂

| Arm | Ball embedding | Ball params | Method | 目的 |
|---|---|---|---|---|
| A0 anchor_default | frozen GRACE | default 64/32/0.05 | beta05 | 原始 anchor |
| A1 anchor_loose | frozen GRACE | 64/16/0.0 | beta05 | 检查收益是否只是参数放松 |
| B0 p2a_default | GRACE+0.1 noisy CE | default 64/32/0.05 | beta05 | 复现单球判负 |
| B1 p2a_loose | GRACE+0.1 noisy CE | 64/16/0.0 | beta05 | 核心验证 |
| B2 p2a_loose_shuffled | GRACE+0.1 noisy CE | 64/16/0.0 | shuffled beta05 | 检查 P2a 局部结构是否真实有贡献 |

## 成功标准

P2a 只在 seed0 下游验证层面成立，不扩 5-seed，除非同时满足：

1. `B1 > A0`：P2a loose 至少超过 frozen-GRACE default anchor。
2. `B1 >= A1`：P2a 表征本身有增量，而不只是 loose 参数带来的收益。
3. `B1 > B2`：P2a 的真实局部结构优于 shuffled control。

若只满足 `B1 > B0`，结论只能写作“参数救回了塌缩”，不能写“P2a 有下游价值”。

## 输出

目录：`runs/mlp8_p2a_rescue_downstream_20260703/`

- `anchor_default.csv`
- `anchor_loose.csv`
- `p2a_default.csv`
- `p2a_loose.csv`
- `p2a_rescue_report.md`

## Claim 边界

这是 seed0 smoke，不产生论文主 claim。若失败，P2a 关闭；若成功，也只能作为 advisor-suggestion 的补充证据，后续是否扩 5-seed 需另立计划。

---

## 执行结果备案

运行时间：2026-07-03

### Seed0 下游结果

| Arm | Ball embedding | Params | Real/Shuffled | Balls | Noisy train purity | True train purity | Valid | Test |
|---|---|---|---|---:|---:|---:|---:|---:|
| A0 anchor_default | frozen GRACE | 64/32/0.05 | real | 52 | 0.4954 | 0.7893 | 57.04 | 55.46 |
| A1 anchor_loose | frozen GRACE | 64/16/0.0 | real | 64 | 0.5009 | 0.8022 | 57.41 | 53.61 |
| B0 p2a_default | GRACE+0.1 noisy CE | 64/32/0.05 | real | 1 | 0.2255 | 0.3198 | 46.30 | 42.96 |
| B0 p2a_default_shuffled | GRACE+0.1 noisy CE | 64/32/0.05 | shuffled | 1 | 0.2255 | 0.3198 | 46.30 | 42.96 |
| B1 p2a_loose | GRACE+0.1 noisy CE | 64/16/0.0 | real | 64 | 0.5028 | 0.7978 | 56.30 | 53.35 |
| B2 p2a_loose_shuffled | GRACE+0.1 noisy CE | 64/16/0.0 | shuffled | 64 | 0.3407 | 0.3870 | 45.93 | 41.58 |

### Gate 判断

- `B1 > B0`：通过。P2a loose 从默认单球的 `42.96` 提升到 `53.35`，说明放松构球参数确实救回了塌缩问题。
- `B1 > B2`：通过。P2a loose real 比 shuffled 高 `+11.77pp`，说明恢复后的局部结构不是随机分组。
- `B1 > A0`：不过。P2a loose `53.35` 低于 frozen-GRACE default anchor `55.46`，差 `-2.11pp`。
- `B1 >= A1`：不过。P2a loose `53.35` 也略低于 frozen-GRACE loose `53.61`，差 `-0.26pp`。

### 最终决策

P2a 不扩 5-seed。更准确的结论是：**P2a 的原始大失败主要来自单球塌缩，放松参数可以恢复球数和大幅修复下游；但 `GRACE+0.1 noisy CE` 表征没有超过 frozen-GRACE anchor，因此不能作为新主线。** 这条结果可以在导师沟通包里作为“有监督辅助预训练被认真补救过，但最终没有带来 anchor 之外的收益”的证据。

输出文件：

- `runs/mlp8_p2a_rescue_downstream_20260703/anchor_default.csv`
- `runs/mlp8_p2a_rescue_downstream_20260703/anchor_loose.csv`
- `runs/mlp8_p2a_rescue_downstream_20260703/p2a_default.csv`
- `runs/mlp8_p2a_rescue_downstream_20260703/p2a_loose.csv`
- `runs/mlp8_p2a_rescue_downstream_20260703/p2a_rescue_report.md`
