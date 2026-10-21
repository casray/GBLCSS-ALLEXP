# MLP-8 P2a CE-0.2 Rescue Check

日期：2026-07-03
定位：P2a 的最后一刀排除实验。只检查 `GRACE + 0.2 noisy CE` 是否比 `GRACE + 0.1 noisy CE` 更可能超过 frozen-GRACE anchor；不改主线，不扩 5-seed，除非 seed0 明确超过 anchor 且 real > shuffled。

## 假设

`GRACE + 0.1 noisy CE` 在放松球参数后能恢复球结构，但下游仍低于 frozen-GRACE anchor。可能原因之一是 CE 权重太小，没有充分利用 train label。将 noisy CE 权重从 0.1 提到 0.2，可能让类别边界更清晰；但也可能更强地拟合 noisy labels。

## 协议

- Dataset：Cora
- Seed：0
- Noise：train-only uniform 0.4
- Split：percent 0.2/0.1/0.7
- Pretrain：GRACE loss + `0.2 * CE(noisy train labels)`
- Backbone：X-only MLP，`frozengrace_gb_mlp_softdist_beta05`
- 对照：
  - frozen-GRACE default `64/32/0.05`
  - frozen-GRACE loose `64/16/0.0`
  - CE-0.2 default `64/32/0.05`
  - CE-0.2 default shuffled
  - CE-0.2 loose `64/16/0.0`
  - CE-0.2 loose shuffled

## Gate

P2a CE-0.2 才允许考虑扩 5-seed，必须同时满足：

1. CE-0.2 real 明确超过 frozen-GRACE default anchor。
2. CE-0.2 real 超过同参数 shuffled。
3. 若使用 loose 参数，CE-0.2 loose 不能低于 frozen-GRACE loose。

<0.5pp 的 seed0 差异只写 comparable，不视为成功。

## 执行结果

输出目录：`runs/mlp8_p2a_ce02_rescue_20260703/`

| Arm | 表征 | 参数 | balls | true train purity | valid | test |
|---|---|---|---:|---:|---:|---:|
| anchor default | frozen GRACE | 64/32/0.05 | 52 | 0.7893 | 57.04 | 55.46 |
| anchor loose | frozen GRACE | 64/16/0.0 | 64 | 0.8022 | 57.41 | 53.61 |
| CE-0.2 default | GRACE+0.2 noisy CE | 64/32/0.05 | 39 | 0.7967 | 57.78 | 55.46 |
| CE-0.2 default shuffled | GRACE+0.2 noisy CE | 64/32/0.05 | 39 | 0.3530 | 47.04 | 42.64 |
| CE-0.2 loose | GRACE+0.2 noisy CE | 64/16/0.0 | 64 | 0.8219 | 57.78 | 53.67 |
| CE-0.2 loose shuffled | GRACE+0.2 noisy CE | 64/16/0.0 | 64 | 0.3833 | 47.41 | 42.16 |

补充：生成脚本内置的 CE-0.2 default downstream 给出 `55.67`，比 anchor default 高 `+0.21pp`；独立加载保存的 H 后复跑为 `55.46`，与 anchor default 持平。该差异低于 seed0 噪声和预注册 `0.5pp` 成功门槛，按 comparable 处理。

## 结论

CE-0.2 比 CE-0.1 明显改善：默认构造不再塌缩，`num_balls=39`，`true_train_purity=0.7967`，下游从 CE-0.1 default 的 `42.96` 恢复到约 `55.46`。real 相比 shuffled 高 `+12.20pp`，说明 CE-0.2 表征中的局部结构不是随机结构。

但它仍没有形成可扩展的新主线：

- CE-0.2 default 与 frozen-GRACE default 只是持平或最多 `+0.21pp`，不足以视为超过。
- CE-0.2 loose 为 `53.67`，低于 frozen-GRACE default，也只与 frozen-GRACE loose `53.61` 基本持平。
- 提高 CE 权重没有带来 anchor 之外的稳定下游收益。

最终决策：**P2a CE-0.2 不扩 5-seed。P2a 关闭，但结论更精确：有监督 noisy CE 辅助预训练可以救回 0.1 的单球塌缩，且 real structure 很强；然而它最多追平 frozen GRACE，不能替代 label-free frozen GRACE anchor。**
