# Tie-safe 代表性对照实验报告

## 1. 最终状态

- 有效批次：`PASS_REPRESENTATIVE_STAGE`
- 时间：2026-07-27（Asia/Shanghai）
- 最终决策：`继续`
- 下一阶段：按训练前冻结的决定，进入 25 settings × 5 seeds × 4 methods 的同轮确定性完整网格。
- 主修复仍为训练前固定的 `distribution_only`；没有根据 test accuracy 改选模式。

## 2. 研究假设

1. `distribution_only` 与 `alpha_zero` 应消除多数类并列时的类别编号依赖。
2. 代表性设置用于估计性能变化，不作等价性或非劣效性声明。
3. 完整网格是否执行不由代表性 test 结果决定。

## 3. 粒球角色与方法

粒球仅提供局部训练标签分布 `p_j` 和混合强度 `α_j`。

- `min_index`：历史多数类 one-hot tie-break，对照。
- `alpha_zero`：并列球停止干预，目标退回 noisy-label one-hot。
- `distribution_only`：并列球令有效 `β=0`，保留 `p_j` 与 `α_j`。

所有非并列球保持历史目标逐元素不变。

## 4. 代码与配置改动

- 目标构造增加显式 `tie_mode`，默认 `min_index`。
- 两个 safe 模式通过类别置换等变性测试。
- 两个真实/随机粒球方法在五个主数据集配置中显式加入 `tie_mode: min_index`。
- raw CSV 新增 tie exposure 与 assignment hash。
- 训练子进程固定单线程数值环境，消除跨进程 KMeans partition 漂移。

没有改变 split、noise realization、embedding、构球参数、MLP、优化器、训练轮数、validation/test mask 或 checkpoint 规则。

## 5. 运行范围

- 数据集：Cora、CoraFull
- 噪声：uniform 0.6，仅污染训练标签
- seeds：0、1、2、3、4
- 模式：`min_index`、`alpha_zero`、`distribution_only`
- 计划行：30
- 完成且 `status=ok`：30/30

## 6. 协议完整性

对 10 个 dataset × seed 配对组逐一检查：

- 三个模式齐全：10/10；
- split hash 唯一：10/10；
- noise hash 唯一：10/10；
- modified train count 唯一：10/10；
- embedding path 一致：10/10；
- assignment hash 唯一：10/10。

确定性环境已写入 environment fingerprint：

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
OMP_DYNAMIC=FALSE
MKL_DYNAMIC=FALSE
PYTHONHASHSEED=0
```

## 7. 机制暴露

三个模式在干预前使用完全相同的 tie exposure 统计。

| Dataset | 平均 tie balls | 平均 active tie balls | 平均 affected train nodes | affected nodes 范围 |
|---|---:|---:|---:|---:|
| Cora | 15.6 | 15.6 | 138.0 | 83–174 |
| CoraFull | 12.0 | 12.0 | 397.4 | 214–567 |

Cora 每个 split 有 541 个训练节点，平均 affected 比例约 25.51%；CoraFull 有 3958 个训练节点，平均约 10.04%。这说明代表性设置确实执行了 tie-safe 分支，不是空消融。

## 8. 测试准确率结果

数值为 mean ± sample std，单位为比例。`Δ` 是与同 seed `min_index` 的配对差；括号为 df=4、未校正 nominal 95% t 区间。

| Dataset | Mode | Test accuracy | Paired Δ | 正/负/零 seeds | Detectable | Practical material |
|---|---|---:|---:|---:|---:|---:|
| Cora | min_index | 0.4104 ± 0.0237 | 0 | 0/0/5 | no | no |
| Cora | alpha_zero | 0.3902 ± 0.0266 | −0.0203 [−0.0410, +0.0004] | 0/5/0 | no | **yes** |
| Cora | distribution_only | 0.4021 ± 0.0294 | −0.0083 [−0.0234, +0.0067] | 1/4/0 | no | no |
| CoraFull | min_index | 0.2716 ± 0.0096 | 0 | 0/0/5 | no | no |
| CoraFull | alpha_zero | 0.2657 ± 0.0076 | −0.0059 [−0.0117, −0.0001] | 0/5/0 | **yes** | no |
| CoraFull | distribution_only | 0.2735 ± 0.0085 | +0.0020 [−0.0044, +0.0083] | 4/1/0 | no | no |

换算为百分点：

- `distribution_only`：Cora −0.834 pp；CoraFull +0.196 pp。
- `alpha_zero`：Cora −2.026 pp；CoraFull −0.588 pp。

## 9. 结果解释

### 已被实验支持

- 两个 safe 模式可在不改非并列球的情况下运行。
- 有效批次的三个模式使用相同 split、noise、embedding 和 partition。
- `distribution_only` 在两个高 tie-exposure 设置中均未触发预设的 detectability 或 practical materiality。
- `alpha_zero` 在 Cora 触发 practical materiality，在 CoraFull 触发 nominal detectability；完全丢弃并列球的软分布比保留 `p_j` 更不稳。

### 仅为合理解释

`distribution_only` 相对温和，可能是因为它只删除有编号偏置的 majority one-hot 分量，同时保留了球内分布和混合强度。该机制解释尚未由单独的因果实验确认。

### 尚未验证

- 25 个主设置下 `distribution_only` 的总体性能与胜场；
- 对 shuffled、MLP+CE、JoCoR 的同轮比较；
- 完整网格的 Holm 敏感性；
- 新确定性结果能否维持旧论文主表的结论。

### 已被负结果削弱

`alpha_zero` 作为默认修复的合理性被削弱：两个数据集均为 5/5 负向，其中 Cora 平均下降超过预设 1 pp practical threshold。

## 10. 历史复现边界

- Cora 历史 `min_index`：5/5 seed 精确复现，test 差为 0，split/noise hash 相同。
- CoraFull 历史 `min_index`：0/5 达到 `1e-6`；旧行缺少 split/noise hash，且本轮已证明旧多线程构球存在 partition lineage 不确定性。

因此：

- Cora 历史数值可作为一致性锚点；
- CoraFull 旧数值不能与新单线程结果直接拼接；
- 新主表必须来自完整网格的四方法同轮运行。

## 11. 失败尝试

Attempt A 虽完成 30/30 行，但 CoraFull seed 1 跨模式 assignment hash 不同，被汇总器拒绝。该批次已完整保存在：

`external/NoisyGL/runs/tie_safe_representative_20260727_invalid_multithread`

没有删除或覆盖。

## 12. 证据资产

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `raw_combined.csv` | 32,043 | `73C615053912FD6BDDA8FF56376806035A4985E47EE5FCD9A3E45D1A2D23E090` |
| `paired_deltas.csv` | 6,516 | `47CF19882D532F3D461D82E295CDA73CBE1ADBE726B3C6EF92AE1AD01A0CFD9F` |
| `summary.csv` | 1,557 | `18F09F0F8A4A25101C4413C383CA330EFC1DAB2C696E2485908ED2669E184B31` |
| `historical_reproduction.csv` | 1,008 | `BE2DA3E05DC442BA81B36C0B90B4C85B63F0A10D69233DD7F716E8A8E5DD0D7B` |
| `report.md` | 2,348 | `2C366B1E8FDC4ECECDA7BBC78E8E0CBF3FB1EB0DC3C3D3334C37B63E761DE1E6` |
| `environment_fingerprint.json` | 1,441 | `97078E02D6485519C0461104B5C11BD80E1A50E9CB892DF32AE1F8BCEA3351DE` |

## 13. 最终决策

`继续`

理由不是代表性结果“看起来不错”，而是：

1. 数学安全性测试通过；
2. 确定性协议和 30/30 行完整性通过；
3. 主修复没有触发预设的大幅性能变化信号；
4. 完整网格在看到这些结果前已经被无条件冻结。

不得在完整网格完成前把历史主表改写成 `distribution_only` 结果。

