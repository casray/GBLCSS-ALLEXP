# MLP-37 GRACE 预训练种子分层方差与投稿收口计划

## 1. 研究问题

现有正式下游实验在每个数据集上固定一份冻结 GRACE 表征 `H`，五个下游种子只覆盖数据划分、标签噪声注入与学生训练随机性，不覆盖 GRACE 预训练随机性。本实验回答：LCSS 相对 CE 与随机分组的增量，是否在不同 GRACE 预训练种子下保持方向一致；预训练种子引入的波动是否与下游波动处于同一量级。

本实验只补完整管线不确定性，不改 LCSS、GRACE 配置、数据集、噪声协议或学生模型。

## 2. 固定协议

- 数据集：Cora、CiteSeer、CoraFull。
- GRACE seeds：`0,1,2`。
- 下游 seeds：`0,1,2`，与每个 H seed 完全交叉。
- Split：percent `0.2/0.1/0.7`。
- 噪声：uniform `0.4`，仅污染 train labels。
- 下游方法：LCSS、size-matched shuffled LCSS；MLP+CE 与 JoCoR 各数据集/下游 seed 只运行一次并复用于三个 H seed。
- LCSS：`max_balls=64`、`min_ball_size=32`、`split_improvement=0.05`、`beta=0.5`，全部冻结。
- GRACE：Cora/CiteSeer 使用各自官方配置；CoraFull 使用预注册的 DBLP 配置原样迁移。不得搜索 epoch、维度、dropout、tau 或学习率。

## 3. 预训练资产要求

每个数据集和 H seed 必须独立保存：

- encoder/projector `checkpoint.pt`；
- 归一化表征 `H.pt`；
- 完整 GRACE config；
- 逐 epoch loss；
- wall-clock 时间；
- GPU peak allocated/reserved memory；
- checkpoint/H 文件大小与 SHA256；
- 数据集节点数、特征维数、H 维数、代码入口和配置来源。

训练损失只读取 `X` 与 `edge_index`。标签、train/val/test mask 不得进入预训练损失。

## 4. 预注册判据

对每个数据集，先在每个 H seed 内对三个下游 seed 求均值。

### 4.1 主稳健性判据

以下条件必须同时满足：

1. 三个 H seed 上 `LCSS - shuffled >= 1.0pp`，方向全部为正。
2. 三个 H seed 上 `LCSS - MLP+CE > 0`。
3. 令 `s_H` 为三个 H-seed 均值之间的样本标准差，`s_down` 为三个 H seed 内下游-seed 样本标准差的均方根；要求

   `s_H <= max(0.25pp, 2 * s_down)`。

4. 9 个 LCSS 单元与 9 个 shuffled 单元全部成功，split/noise hash 成对一致，且 val/test 标签零污染。

三个数据集全部满足时，正文可写“LCSS 的核心增量在三个 GRACE 预训练种子下方向稳定，预训练种子波动未超过预注册阈值”。任一数据集不满足时，不调参、不删数据；摘要与 Highlights 必须显式保留 fixed-H conditional 限定，并报告失败数据集。

### 4.2 强对照次级判据

- `LCSS - JoCoR` 至少 `2/3` 个 H seed 为正；
- 任一 H seed 不得低于 `-0.5pp`。

该判据单独报告，不替代 4.1 的主稳健性判据。

### 4.3 描述量

球数、球数范围、target Brier/NCR/CDR、不同 H seed 分区的 adjusted Rand index 只作描述，不设置通过线，不参与模型选择。

## 5. 论文双口径

禁止把 CoraFull、Coauthor-CS、Amazon-Photo 写成 `held-out` 或“从未见过的数据集”。允许且应完整使用的表述是：

> 在 beta 与全部构造参数冻结后完成全噪声谱评测的三个数据集上，LCSS 在 15 个设置中的 14 个取得最高均值；合并开发阶段两个数据集后，在五个完成全谱评测的数据集上为 24/25。前者是参数冻结后的条件评测，后者是开发筛选后的描述性结果，均不外推为任意未筛选数据集上的无条件胜率。

## 6. 并行收口任务

1. 在 v15 中补 pair/instance 的精确生成公式、pair 类映射和随机种子协议。
2. 保留 LOO 不替换默认目标的理由：平均增量有限、Holm 后未显著、CDR 上升，故只作机制消融。
3. 用本轮预训练日志生成 GRACE 成本表。
4. 建立全新、无 git 历史的匿名 artifact；只复制投稿所需的 allowlist 文件，并扫描用户名、机器名、绝对路径与本机中文目录名。
5. 仅在 MLP-37 结果落地后生成 v15 与 Elsevier HTML。

## 7. 明确不做

- 不扩 10 seeds、真实噪声、新数据集、CPU/单节点 latency 或 inductive 实验。
- 不修改 beta、球粒度、GRACE 配置或主算法。
- 不把蒸馏扩展升级成主算法。
- 不用球数、纯度或 test accuracy 选择 H seed。
- 不因实验结果调整本文件的判据。
