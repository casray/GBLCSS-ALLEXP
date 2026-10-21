# V39 NRGNN 鲁棒教师蒸馏完成报告

## 1. 研究问题

本实验检验：在相同的 X-only student 部署约束下，成熟图标签去噪教师
NRGNN 是否能够替代球式共识教师，以及球式教师在高噪声下的提升是否具有
粒球特异性。

## 2. 冻结协议

- 数据集：Cora、CiteSeer、CoraFull。
- 噪声：uniform 0.4 与 uniform 0.6，只污染训练标签。
- 划分：20%/10%/70%。
- 种子：0--4。
- 新方法：
  - `mlp_distill_nrgnn`
  - `mlp_distill_nrgnn_xh`
- 每个 seed 内独立训练教师，并由其最佳干净验证检查点生成全节点概率。
- student 为同一两层 X-only MLP，hidden=64，temperature=1.0。
- 不混合 student hard-label loss，不根据结果调整 NRGNN 或蒸馏参数。

预注册计划：

`external/NoisyGL/docs/execution_plan_mlp47_nrgnn_teacher_distill_20260730.md`

## 3. 工程与完整性

- 新增正式记录：60/60。
- 七臂合并记录：210。
- 完整性组：42/42。
- dataset-rate-seed 协议组：30/30 split/noise hash 一致。
- 所有新记录 `status=ok`。
- `modified_val_count=modified_test_count=0`。
- student 输入始终为原始 \(X\)，`student_inference_uses_graph=0`。
- teacher 概率行和最大误差低于 \(10^{-5}\)。
- 28 个相关 predictor/runner 回归测试与 20 个子测试通过。
- 6 个 v38/v39 稿件与证据审计测试通过。

## 4. 五种子结果

### Uniform 0.4

| Teacher | Cora | CiteSeer | CoraFull |
|---|---:|---:|---:|
| 球式共识 | 80.60±0.69 | 69.40±1.08 | 58.22±0.69 |
| kNN 共识 | 81.61±0.74 | 69.65±1.61 | 58.58±0.55 |
| NRGNN | 81.07±0.55 | 70.87±1.06 | 57.71±0.58 |
| NRGNN[\(X\Vert H\)] | 80.37±1.29 | 70.43±0.98 | 56.77±0.59 |

NRGNN student 相对球式 student 的配对均值差为
\(+0.46,+1.47,-0.52\) pp。该噪声率下不存在统一最优教师。

### Uniform 0.6

| Teacher | Cora | CiteSeer | CoraFull |
|---|---:|---:|---:|
| 球式共识 | 71.27±2.46 | 62.28±2.30 | 51.82±0.86 |
| kNN 共识 | 71.80±2.63 | 62.90±2.26 | 52.65±0.93 |
| NRGNN | 74.81±1.17 | 66.62±1.27 | 53.55±1.00 |
| NRGNN[\(X\Vert H\)] | 74.50±2.25 | 67.75±1.79 | 52.78±0.62 |

NRGNN student 相对球式 student 的配对均值差为
\(+3.54,+4.34,+1.73\) pp，分别为 5/5、5/5、5/5 seeds 正向。
相对排除自身的 kNN teacher student，差值为
\(+3.01,+3.72,+0.90\) pp。

## 5. 教师—学生压缩与资源

- Cora/CiteSeer 上，NRGNN student 与自身 teacher 的均值差位于
  \(-1.22\) 至 \(+0.52\) pp。
- CoraFull 上，差值为 \(-5.00\) 至 \(-5.61\) pp，显示明显的
  X-only 蒸馏压缩边界。
- 额外输入 \(H\) 未稳定改善 NRGNN student，仅在 CiteSeer u0.6
  上高于普通 NRGNN student。
- seed 0、u0.4 下，普通 NRGNN teacher 训练时间约为
  25.50/27.63/151.64 秒，student 约为 1.20/1.32/9.16 秒。
- 总墙钟包含 NRGNN 图构造与 runner 开销，因此高于两段训练时间之和，
  尤其在 CoraFull 上明显。

## 6. 结论与论文决策

1. 鲁棒 teacher 蒸馏值得进入论文。它直接回答了多轮冷审提出的
   “为什么不蒸馏成熟图标签去噪方法”。
2. 球式共识 teacher 不是统一最强 teacher。uniform 0.6 下，NRGNN
   teacher 在三个数据集上均更强。
3. 第 6 节应从“共识教师带来精度提升”收窄为“图教师蒸馏扩展”。
   球式共识是简单局部目标实现，NRGNN 是高噪声下更强的替代。
4. 该结论只影响二级蒸馏扩展，不改变基础 LOO GB-LCSS、主表或
   25-setting headline。
5. 新稿创建为
   `paper_final_zh_v39_nrgnn_teacher_20260730.md`，v38 保留不动。

## 7. 证据文件

- `external/NoisyGL/runs/mlp47_nrgnn_teacher_distill_20260730/mlp47_combined_raw.csv`
- `external/NoisyGL/runs/mlp47_nrgnn_teacher_distill_20260730/mlp47_aggregate.csv`
- `external/NoisyGL/runs/mlp47_nrgnn_teacher_distill_20260730/mlp47_paired_deltas.csv`
- `external/NoisyGL/runs/mlp47_nrgnn_teacher_distill_20260730/mlp47_protocol_audit.csv`
- `external/NoisyGL/runs/mlp47_nrgnn_teacher_distill_20260730/mlp47_completeness.csv`
- `external/NoisyGL/runs/mlp47_nrgnn_teacher_distill_20260730/mlp47_report.md`
