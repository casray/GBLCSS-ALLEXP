# SUPPORT 支持量与监督强度：已有证据

日期：2026-10-23。状态：只读核查旧源码、配置与结果；本轮没有训练。

## 1. SUPPORT32 的实际定义

源码：

- `experiments/neighbor_preserving_ball_20261022/generator.py`：`build_tree`、`select_cut`。
- `experiments/neighbor_support_ablation_20261022/kernels.py`：`support_costs`。
- `experiments/r2_support_5dataset_20261022/config.json`：历史生成配置。

先在归一化 GRACE 表示上生成二分 K-means 候选树。几何与训练节点位置参与构树，标签取值不参与。
令候选球中训练节点数为 m，留一投票数为 s=m-1，目标支持量为 k。保留该球的代价为：

    c(B;k) = m * abs((m-1)-k) / max(m-1,k)

动态规划比较保留父球的代价和保留最优子树的总代价；平票保留父球。
因此，k=32 是目标留一票数。整个树截断在加权意义上尽量接近这个目标，实际球大小受候选树影响。

历史 SUPPORT 使用 max_leaves=256、min_nodes=32、min_train=3。
其中 min_nodes=32 是总节点数约束，reference_k=32 是留一训练票数目标；两个 32 的含义不同。
E1 为覆盖小支持量，换成 max_leaves=4096、min_nodes=8、min_train=3。
历史 SUPPORT32 与 E1 新树 S32 的分区需要分别保存，不能直接当成同一输入。

## 2. 做过哪些扫描

### 2.1 2026-10-22 三种子支持量消融

`experiments/neighbor_support_ablation_20261022/` 固定 k=32，比较 NP、SUPPORT、COUNT、KM、匹配 kNN。
这轮支持简单 SUPPORT 可以保留多数 NP 修复收益；没有比较其他 k，也没有得出 32 最优。

### 2.2 2026-10-23 E1 扫描

来源：`experiments/sc_support_channel_pilot_20261023/`。

- 离线目标：5 图、3 个协议 seed、4 条件、k={4,8,16,32,64,128}，共 360 个候选目标。
- 下游分类：仅 R0，Cora/CoraFull/Amazon-Photo，seed0，4 条件。
- 没有完成全 5 图、多种子、R2/GCN 的支持量下游扫描。

从 `E1_selector_results.csv` 提取 VAL 选择，完全平票按原规则取较大支持量：

| 数据集 | clean | U0.4 | U0.6 | Pair0.4 |
|---|---:|---:|---:|---:|
| Cora | 64 | 8 | 32 | 64 |
| CoraFull | 16 | 16 | 16 | 16 |
| Amazon-Photo | 8 | 32 | 4 | 128 |

单种子、每格验证集选择，存在选模波动。该表说明偏好随数据和条件变化，尚未确认新的默认参数。

从 `results.csv` 筛选 stage=E1、arm=S*、三种带噪条件，按 3 图 x 3 条件等权汇总：

| k | 平均验证准确率 % | 平均测试准确率 % |
|---|---:|---:|
| 4 | 53.65 | 52.17 |
| 8 | 54.61 | 53.34 |
| 16 | 54.65 | 53.11 |
| 32 | 55.26 | 53.18 |
| 64 | 54.89 | 52.68 |
| 128 | 51.70 | 50.25 |

这里 32 的平均验证准确率最高，8 的事后测试均值略高；0.16pp 的单种子差异仅作描述。
既有数据已经用于开发，本计划不会据这张测试表换默认值。

E1 SC 相对 VAL 的两档 uniform 验证均值为 -2.0585pp，未通过预设门槛。
SC 与真实投票准确率候选排名的平均 Spearman 为 U0.4 0.9377、U0.6 0.8857。
这支持投票质量与实际分类收益需要分开研究。强度、方向与学生拟合各自的因果份额仍待实验。

### 2.3 实际票数

来源：`diagnostics/e1/geometry_profiles.csv`，新树 S32、seed0。均值按训练节点加权。

| 数据集 | 球数 | 最少票 | 中位票数 | 平均票数 | 最多票 |
|---|---:|---:|---:|---:|---:|
| Cora | 20 | 9 | 30 | 29.21 | 40 |
| CiteSeer | 23 | 8 | 32 | 31.24 | 44 |
| CoraFull | 146 | 2 | 31 | 29.97 | 45 |
| Coauthor-CS | 133 | 3 | 30 | 30.91 | 85 |
| Amazon-Photo | 54 | 5 | 33 | 30.94 | 45 |

## 3. 强度控制已经有哪些先例

- `experiments/heterophily_mechanism_20261014/PLAN.md`、`实验结论.md`：固定 GB128 的 q，离线比较 half、ESS、跨尺度 JS、组合、平均强度匹配及打乱。可靠性组合相对对照未通过目标质量门槛。这轮的下游分类表没有完整训练全部收缩方法。
- `experiments/r2_mechanism_controls_20261021/PLAN.md`：MASS 已逐节点保留移出观测类别的概率，替代类均分。当前计划沿用这项控制思想。
- `experiments/gce_lcss_fusion_20261022/`：模型置信度加权 LCSS 的增量小，R2 clean 损害仍存在。

因此本轮不将“给 alpha 乘系数”写成首次提出。新增问题是新骨干上的实用性、真实转移量与方向的因果拆分，以及必要时直接以学生收益校准监督。

## 4. SimMLP 插件的可复用输入

- 编码器：`experiments/simmlp_tined_audit_seed0_20261009/runs/<dataset>/simmlp/checkpoint.pt`。
- 已核实的调用路径：`experiments/r2_body_comparison_20261023/external_v2.py`。
- 作者适配器：`experiments/simmlp_tined_audit_seed0_20261009/simmlp_adapter.py`。
- 标签/划分：`experiments/sc_support_channel_pilot_20261023/diagnostics/e1/<dataset>/seed<seed>/`。
- 历史 SUPPORT32 在上述 `geometry.pt` 的 `legacy32` 字段。
- KM：`experiments/r2_body_comparison_20261023/geometry/<dataset>/seed<seed>.pt`。
- R2 最新头训练：`experiments/r2_body_comparison_20261023/run.py`。

SimMLP 历史头使用 AdamW、200 轮、最后一次验证平票；R2 最新头使用 Adam、200 轮、首次严格最佳。
新实验各自骨干内统一规则。A 阶段改成首次严格最佳并重跑 CE，旧 SimMLP CE 仅作桥接检查。
历史 SimMLP 仅 seed0 编码器，新增头/协议 seed 不作为独立预训练 seed 计数。

## 5. 相关原始文献

- SimMLP，作者实现：<https://github.com/Zehong-Wang/SimMLP>。
- Learning to Reweight Examples for Robust Deep Learning，ICML 2018：<https://proceedings.mlr.press/v80/ren18a.html>。以干净元验证梯度指导样本权重。
- Meta Label Correction for Noisy Label Learning，AAAI 2021：<https://ojs.aaai.org/index.php/AAAI/article/view/17319>。以双层优化学习修正标签。

后两项支持“以学习收益指导监督”的设计依据；本计划提出的区域候选、改动量分解和受限共识门控尚无新实验结果。
