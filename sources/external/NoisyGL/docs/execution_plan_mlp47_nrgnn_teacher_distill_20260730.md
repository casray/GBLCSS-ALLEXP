# MLP-47 NRGNN 鲁棒教师蒸馏执行计划

## 研究问题

检验共识教师蒸馏的性能是否具有粒球特异性，或成熟图标签去噪教师
NRGNN 也能将鲁棒预测蒸馏给 X-only MLP。

## 方法角色

- NRGNN：训练期读取图结构和含噪训练标签的鲁棒图教师。
- NRGNN[X‖H]：在 NRGNN 基础上额外读取冻结 GRACE 表征，和现有球式
  教师的输入资源对齐。
- Student：与表 5 相同的两层 X-only MLP，测试时不读取图、H 或教师。

## 新增方法

1. `mlp_distill_nrgnn`
2. `mlp_distill_nrgnn_xh`

每个 seed 内先独立训练 NRGNN 教师，以相同的干净验证协议选择教师
checkpoint，再从最佳教师导出全部节点的温度 softmax。学生使用这些
全节点概率训练，蒸馏温度固定为 1.0，学生 checkpoint 仍由相同验证协议
选择。

## 固定协议

- 数据集：Cora、CiteSeer、CoraFull
- 噪声：uniform 0.4、uniform 0.6
- Seeds：0--4
- Split：percent 0.2/0.1/0.7
- Noise scope：train
- Device：cuda:0
- Student：2-layer MLP, hidden=64, dropout=0.5
- 不混合 hard-label student loss
- 不调整 NRGNN 官方项目配置
- 不根据结果调整蒸馏温度

## 对照

复用 split/noise hash 一致的现有五臂：

- GCN(X)+CE → MLP
- GCN[X‖H]+CE → MLP
- 随机分组共识教师 → MLP
- 排除自身的 kNN 共识教师 → MLP
- 球式共识教师 → MLP

另复用直接 NRGNN 与 NRGNN[X‖H] 教师精度，报告蒸馏压缩差值。

## Smoke

先运行：

- Cora、CiteSeer、CoraFull
- uniform 0.4
- 两个新方法
- seed 0

smoke 通过条件：

1. 六条记录全部 `status=ok`。
2. teacher/student 概率全有限，行和约为 1。
3. `student_input_dim == X.shape[1]`。
4. `teacher_use_xh` 与方法身份一致。
5. train-only 污染审计通过。
6. 学生测试精度不低于 MLP+CE 10 个百分点以上，避免明显实现错误。

smoke 只判断工程可行性，不筛选有利方法。

## 正式实验

smoke 通过后运行：

\[
3\text{ datasets}\times2\text{ noise rates}
\times2\text{ teachers}\times5\text{ seeds}=60\text{ runs}.
\]

seed 0 可复用 smoke，但必须通过 split/noise hash 校验。

## 必报结果

- mean±sample std
- 逐 seed paired delta
- Distill-NRGNN − GLNN student
- Distill-NRGNN[X‖H] − GCN[X‖H]+CE student
- Distill-NRGNN − direct NRGNN teacher
- Distill-NRGNN[X‖H] − direct NRGNN[X‖H] teacher
- 两个新学生分别相对球式和 kNN 共识教师 student 的差值
- 教师与学生训练时间、参数量和推理资源说明

## 解释规则

该实验没有“只有赢才报告”的性能门。

- 鲁棒教师学生整体更强：第 6 节收窄为一种简单共识教师扩展，明确
  NRGNN 蒸馏是更强替代。
- 球式教师整体更强：作为粒球教师的正面证据，但不外推到其他教师或
  数据集。
- 结果互有胜负：报告噪声率、数据集和资源条件下的差异，不宣称统一
  最优。

## 输出

目录：

`runs/mlp47_nrgnn_teacher_distill_20260730/`

文件：

- `smoke_seed0_uniform0p4.csv`
- `formal_uniform0p4.csv`
- `formal_uniform0p6.csv`
- `mlp47_combined_raw.csv`
- `mlp47_aggregate.csv`
- `mlp47_paired_deltas.csv`
- `mlp47_protocol_audit.csv`
- `mlp47_report.md`

## 论文处置

不覆盖 v38。实验完成后创建 v39，并只更新第 6 节、表 5、对应附录和
证据登记。基础 GB-LCSS 主表与 headline 不受该扩展实验影响。
