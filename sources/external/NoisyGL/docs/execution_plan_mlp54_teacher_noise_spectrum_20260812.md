# MLP-54 图教师蒸馏完整噪声谱执行计划

## 研究问题

现有图教师蒸馏比较覆盖 Cora、CiteSeer 和 CoraFull 的 uniform 0.4 与
uniform 0.6。本实验补齐 uniform 0.2、pair 0.4 和 instance 0.4，检验教师
路线的相对排序是否依赖噪声强度或噪声结构。

## 方法与角色

固定比较五条已经审计的教师路线：

1. `mlp_distill_gcn`：GCN(X)+CE 教师，作为 GLNN 式基础蒸馏。
2. `mlp_distill_softdist_xh_shuffled`：随机等规模分组共识教师。
3. `mlp_distill_knn_xh`：排除查询节点自身的 kNN 共识教师。
4. `mlp_distill_softdist_xh`：粒球共识教师。
5. `mlp_distill_nrgnn`：NRGNN 鲁棒图教师。

所有教师均在每个 seed 内重新训练，并为全体节点生成概率目标。所有学生
均为相同的两层 X-only MLP。学生推理不访问邻接矩阵、冻结表征或教师。

## 固定协议

- 数据集：Cora、CiteSeer、CoraFull。
- 新增设置：uniform 0.2、pair 0.4、instance 0.4。
- 已有设置：uniform 0.4、uniform 0.6，仅在最终完整谱汇总时复用。
- Seeds：0--4。
- Split：percent 0.2/0.1/0.7。
- Noise scope：train。
- Validation：干净验证标签，与现有主协议一致。
- Device：cuda:0。
- Student：2-layer MLP，hidden=64，dropout=0.5。
- 不改变任何教师配置、蒸馏温度、学生结构或模型选择协议。
- kNN 教师必须记录 `knn_exclude_self=1`。

新增正式运行量：

\[
3\text{ datasets}\times3\text{ settings}\times
5\text{ teachers}\times5\text{ seeds}=225\text{ runs}.
\]

## Smoke 与完整性门

先运行全部三个新增设置的 seed 0，共 45 条。smoke 只检查工程与协议，不
根据精度筛选设置或教师。通过条件：

1. 45/45 行 `status=ok`，无 NaN、OOM 或单类预测坍缩。
2. 噪声只污染训练标签，验证集和测试集修改数为 0。
3. 同一 dataset--noise--seed 下五条路线的 split/noise hash 完全一致。
4. kNN 教师排除查询节点自身。
5. NRGNN 教师概率有限且逐行和误差不超过 1e-5。
6. 五条路线的学生推理身份均为 X-only。

smoke 通过后补 seeds 1--4。任何失败行必须修复后原设置重跑，不得删除。

## 必报结果

- 五个噪声设置的 mean±sample std。
- 每个 dataset--noise 下的逐 seed 配对差值。
- 球式教师相对 GCN、随机分组、kNN 和 NRGNN 教师的差值与正向 seed 数。
- 各教师在 15 个 dataset--noise 单元中的最高均值计数。
- 按噪声设置汇总的跨数据集平均差。
- 完整性、污染范围、split/noise hash 和推理资源审计。

## 解释规则

本实验无性能筛选门，所有新增设置均进入报告。

- 若球式教师仅在部分噪声下领先，写成噪声条件相关结果，不外推为普遍
  优于 kNN 或 NRGNN。
- 若 pair/instance 下球式教师受损，结合相关噪声可能在局部区域内共同
  出现这一事实讨论，但不把解释写成已验证因果。
- 若不同噪声下排序稳定，保留正文的 u0.4/u0.6 紧凑表，完整谱放附录。
- 若排序明显变化，正文报告变化规律，仍不按有利设置重排主表。

## 额外教师处置

本轮不新增 RTGNN 教师。当前仓库没有 CoraFull 的 RTGNN 配置，也没有
经过审计的 RTGNN-to-MLP 蒸馏封装。临时迁移配置会把教师实现忠实度与
噪声设置同时变成变量。NRGNN 作为成熟鲁棒图教师锚点已足以回答“鲁棒
图教师能否优于简单共识教师”的问题。

## 输出

目录：`runs/mlp54_teacher_noise_spectrum_20260812/`

- `smoke_uniform0p2_seed0.csv`
- `smoke_pair0p4_seed0.csv`
- `smoke_instance0p4_seed0.csv`
- `formal_uniform0p2_seeds1_4.csv`
- `formal_pair0p4_seeds1_4.csv`
- `formal_instance0p4_seeds1_4.csv`
- `mlp54_new_raw.csv`
- `mlp54_new_aggregate.csv`
- `mlp54_new_paired_deltas.csv`
- `mlp54_completeness.csv`
- `mlp54_protocol_audit.csv`
- `mlp54_report.md`

## 论文处置

不覆盖 v52。实验完成后创建新版本：

- 表 3 中方法名删除重复的“协议适配”，统一移到表注。
- 表 4 增加“学生推理输入”列，所有路线均写 X-only。
- 表 4 正文仍保留 u0.4/u0.6，避免过宽。
- 新增完整五设置教师谱附录表，并在第 6 节正文概括排序是否随噪声变化。
