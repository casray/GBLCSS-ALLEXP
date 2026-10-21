# MLP-55 GNN-to-MLP 方法补入表 4 执行计划

## 研究问题

在当前图教师蒸馏表中补入 FF-G2M、KRD 和 VQGraph，检验球式共识教师相对更丰富的 GNN-to-MLP 知识迁移机制是否仍具有竞争力。

## 方法身份与角色

1. `ffg2m_reimplemented`：保留 FF-G2M 的低频邻域 KL、高频边差异 KL 与含噪标签 CE，统一为本文的两层 GCN 教师和两层 X-only MLP 学生。该行称为 `FF-G2M (common-backbone reimplementation)`，不是官方调参复现。
2. `krd_adapted`：根据教师在特征扰动下的熵稳定性筛选可靠知识并蒸馏给 X-only MLP。该行称为 `KRD-inspired reliability adaptation`，用于机制比较。
3. `vqgraph_adapted`：使用图教师、软码本分配和 token 对齐训练 X-only MLP。该行称为 `VQGraph-inspired token adaptation`，不是 VQGraph 官方端到端复现。

三条路线训练时均使用图和无标签节点，学生推理只读取节点属性。每个数据集、噪声设置和 seed 均独立重训教师与学生。

## 固定协议

- 数据集：Cora、CiteSeer、CoraFull。
- 噪声设置：uniform 0.4、uniform 0.6。
- Seeds：0、1、2、3、4。
- Split：percent 0.2/0.1/0.7。
- Noise scope：train。
- Validation：干净验证标签，与当前论文主协议一致。
- Device：ROCm Python，`cuda:0`。
- 学生：两层 X-only MLP，hidden=64，dropout=0.5。
- 不修改三方法配置，不按本轮结果选择数据集、噪声率或超参数。

新增正式运行量：

\[
3\text{ datasets}\times2\text{ noise rates}\times
3\text{ methods}\times5\text{ seeds}=90\text{ runs}.
\]

## Smoke 与正式运行

先运行全部三数据集、两个噪声率、三个方法的 seed 0，共 18 条。通过条件只检查工程和协议：

1. 18/18 行 `status=ok`，test、teacher test 和损失诊断均有限。
2. `modified_val_count=modified_test_count=0`，且 `modified_count=modified_train_count`。
3. 同一 dataset--noise--seed 下三方法的 split/noise hash 一致，并与当前表 4 对照行一致。
4. 三方法均记录 `training_uses_graph=1`、`inference_uses_graph=0` 和 `g2m_uses_unlabeled_nodes=1`。
5. 无 OOM、NaN 或单类预测坍缩。

smoke 通过后补 seeds 1--4，共 72 条。失败行修复后原协议重跑，不得删除。

## 汇总与判定

本实验没有性能入场门。三方法无论结果是否有利，均进入表 4。

必须输出：

- 90 条不可混用旧结果的正式原始记录。
- mean±sample std。
- 相对球式共识教师、GLNN 式教师和 NRGNN 教师的逐 seed 配对差。
- 90/90 完整性审计、train-only 污染审计、split/noise hash 审计和推理资源审计。
- 表 4 可直接使用的合并 CSV 与 Markdown 表。

## 论文处置

- 不覆盖 v54，创建新的论文版本。
- 表 4 增加 FF-G2M、KRD-inspired 和 VQGraph-inspired 三行。
- 表下注释集中说明 common-backbone reimplementation 和 inspired adaptation 的身份差异。
- 粗体仅表示当前表内最高均值，不把机制适配写成官方方法排名。
- 正文先报告正式教师路线的结果，再用一句话概括两个 inspired adaptation 的机制参照结果。

## 输出目录

`runs/mlp55_g2m_table4_20260812/`

- `smoke_uniform0p4_seed0.csv`
- `smoke_uniform0p6_seed0.csv`
- `formal_uniform0p4_seeds1_4.csv`
- `formal_uniform0p6_seeds1_4.csv`
- `mlp55_raw.csv`
- `mlp55_aggregate.csv`
- `mlp55_paired_deltas.csv`
- `mlp55_completeness.csv`
- `mlp55_protocol_audit.csv`
- `mlp55_table4.csv`
- `mlp55_table4.md`
- `mlp55_report.md`

## 执行结果（2026-08-12）

- Seed 0 smoke：18/18 成功，资源身份、train-only 污染和既有表 4 哈希全部通过。
- 正式网格：90/90 成功，18 个方法--数据集--噪声组合均有 5 个种子。
- 合并表 4：210/210 条逐种子记录完整，30/30 个协议组的 split/noise hash 一致。
- FF-G2M 在 Cora、CiteSeer、CoraFull 的 u0.4/u0.6 结果分别为
  77.65/63.63、65.80/52.32、56.58/45.78。
- KRD 的对应结果为 71.77/61.28、64.96/54.17、47.12/41.61。
- VQGraph 的对应结果为 78.89/65.75、65.64/53.34、58.52/52.01。
- 球式共识教师在 Cora 和 CiteSeer 的两个噪声率下均高于三种新增路线；
  CoraFull 上 VQGraph 分别高出 0.29 和 0.19 个百分点。
- NRGNN 教师保持高噪声下的最强结果。

## 最终决策

三条路线全部进入论文表 4。论文将球式教师表述为在测试的
GNN-to-MLP 路线中具有竞争力，并明确保留 CoraFull 与 NRGNN 所揭示的边界。
基础 GB-LCSS 方法、主实验和 headline 计数均不改变。
