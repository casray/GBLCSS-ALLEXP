# MLP-42：表 4 共识教师蒸馏数据集广度补齐

## 研究假设

正文表 4 当前只给出 Cora、CiteSeer 与 CoraFull。Coauthor-CS 和 Amazon-Photo 已有 uniform 0.4 的原始五臂五种子结果，但缺少 uniform 0.6，因此无法判断共识 teacher 的归因是否能跨到 coauthor/co-purchase 图。

## 粒球角色

粒球仅用于在 frozen GRACE H 上形成 train-only 局部共识 soft target，训练 GCN[X||H] teacher；最终 student 始终是相同的 X-only 两层 MLP。粒球不进入推理。

## 固定协议

- 数据集：Coauthor-CS、Amazon-Photo。
- 噪声：train-only uniform 0.4/0.6。
- Split：percent 0.2/0.1/0.7。
- Seeds：0--4。
- 五臂：`mlp_distill_gcn`、`mlp_distill_gcn_xh`、`mlp_distill_softdist_xh_shuffled`、`mlp_distill_knn_xh`、`mlp_distill_softdist_xh`。
- kNN teacher 按每个 seed 的平均球内训练票数匹配 k，与正文表 4 一致。
- u0.4 复用 `runs/mlp16_distill_tableb_completion_20260707/distill_tableb_uniform0p4_seeds_0_1_2_3_4.csv` 的 50 行；不得重跑。
- u0.6 新增 50 行；不使用 MLP-29 的 `beta05/robust-teacher` 变体冒充原五臂。

## 执行顺序

1. u0.6 seed0 smoke：2 数据集 × 5 臂，共 10 runs。
2. 检查 `status=ok`、无 NaN/OOM、`modified_val_count=modified_test_count=0`、`modified_count=modified_train_count`，并确认 kNN 的 `matched_k` 为正整数。
3. smoke 通过后补 seeds 1--4，共 40 runs。
4. 与 u0.4 精确按 dataset/method/seed 汇总 mean±std、逐 seed paired delta 和三层教师增益。
5. 无论结果是否有利，均补入正文表 4 或附录 I，并据实更新边界叙事。

## 完整性标准

- u0.4/u0.6 合并矩阵必须为 `2 datasets × 2 rates × 5 methods × 5 seeds = 100` 个唯一单元。
- 全部行使用相同 percent split、train-only noise 和对应 seed；同 dataset/rate/seed 的五臂 split/noise hash 必须一致。
- validation/test 标签修改数必须为 0。
- 不以性能高低决定是否报告；若球式低于 kNN、CE teacher 或 shuffled，必须保留。

## 输出

- `runs/mlp42_table4_breadth_20260717/uniform0p6_seed0.csv`
- `runs/mlp42_table4_breadth_20260717/uniform0p6_seeds_1_2_3_4.csv`
- `runs/mlp42_table4_breadth_20260717/mlp42_combined.csv`
- `runs/mlp42_table4_breadth_20260717/mlp42_aggregate.csv`
- `runs/mlp42_table4_breadth_20260717/mlp42_paired_deltas.csv`
- `runs/mlp42_table4_breadth_20260717/mlp42_report.md`

## 最终决策规则

- 只要完整性标准通过，就把两数据集加入表 4，使正文覆盖 citation/coauthor/co-purchase 三类图。
- 结论按真实结果写，不设必须超过某基线的性能门。

## 2026-07-17 协议暂停记录

用户在 seeds1--4 执行期间指出超参数预算不对称：粒球使用跨数据集固定构造参数，而票数匹配 kNN 从每个 seed 的球结果获得数据依赖的 k。需要澄清，`matched-k` 没有通过 validation/test 性能搜索“最优 k”，其用途是控制有效投票数的机制诊断；但它不应被包装成与固定参数粒球具有相同调参预算的工程方法对照。

- GPU 批次已停止；`uniform0p6_seeds_1_2_3_4.csv` 当前仅 8 条部分行，禁止进入任何汇总。
- 后续若继续，正文方法比较应使用预先固定、跨数据集一致的 `k=5`；票数匹配 kNN 只作为控制票数的诊断行放附录。
- 最稳妥的披露方式是同时报告 fixed-k 与 vote-matched 两种口径，并明确二者回答不同问题，不以其中任一行单独声称粒球普遍优于 kNN。

## 2026-07-17 恢复执行与最终结果

用户明确要求恢复表 4 广度实验。为保持与冻结表 4 的可比性，本轮继续使用原始 vote-matched kNN 定义；未采用暂停记录中建议的 fixed `k=5` 正文替换。vote-matched kNN 仅控制有效共识票数，k 由同 seed 球诊断的平均训练票数确定，不通过 validation/test 精度选择。

- 复用 u0.4 的 50 条历史行、u0.6 seed0 的 10 条行以及中断前经审计合格的 8 条基础教师行。
- GPU 补跑 32 条缺失行；全部 `status=ok`，无 NaN/OOM，validation/test 标签修改数为 0。
- 最终矩阵为 100/100 唯一单元；每个 dataset/rate/seed 的五臂 split/noise hash 完全一致。
- u0.6 的 vote-matched k：Coauthor-CS 为 57，Amazon-Photo 为 24，所有 seeds 一致。

### 五种子均值±标准差（%）

| 教师 | Coauthor-CS u0.4 | Amazon-Photo u0.4 | Coauthor-CS u0.6 | Amazon-Photo u0.6 |
|---|---:|---:|---:|---:|
| GCN(X)+CE（GLNN 式） | 93.20±0.09 | 89.86±0.77 | 91.13±0.58 | 84.80±1.48 |
| GCN[X‖H]+CE | 93.35±0.55 | 90.51±1.08 | 91.16±1.32 | 85.10±3.27 |
| 随机分组共识教师 | 93.11±0.39 | 89.29±0.26 | 91.23±0.50 | 80.11±9.78 |
| vote-matched kNN 共识教师 | 93.69±0.43 | 90.15±0.93 | 91.66±0.90 | **86.82±0.61** |
| 球式共识教师 | **93.75±0.22** | 90.49±1.05 | **92.25±0.69** | 85.87±2.25 |

### 关键配对增量

- Coauthor-CS u0.4：ball−GCN(X) `+0.55pp`（5/5），ball−GCN[X‖H] `+0.40pp`（4/5），ball−shuffled `+0.64pp`（5/5），ball−kNN `+0.06pp`（3/5）。
- Coauthor-CS u0.6：ball−GCN(X) `+1.12pp`（5/5），ball−GCN[X‖H] `+1.09pp`（5/5），ball−shuffled `+1.02pp`（5/5），ball−kNN `+0.59pp`（4/5）。
- Amazon-Photo u0.4：ball−GCN(X) `+0.63pp`（4/5），ball−GCN[X‖H] `−0.02pp`（2/5），ball−shuffled `+1.20pp`（4/5），ball−kNN `+0.34pp`（4/5）。
- Amazon-Photo u0.6：ball−GCN(X) `+1.07pp`（2/5），ball−GCN[X‖H] `+0.77pp`（3/5），ball−shuffled `+5.76pp`（4/5），ball−kNN `−0.95pp`（1/5）。

### 最终决策

- Coauthor-CS 提供了可信的广度证据：球式在 u0.4/u0.6 均为五臂最高，尤其 u0.6 的四项配对归因均为正且方向稳定。
- Amazon-Photo 只提供部分支持：u0.4 球式与直接 XH teacher 近似持平，u0.6 则明显低于 kNN；其 shuffled 与 XH teacher 在 seed3/4 出现大幅退化，不能把均值差解释为稳定机制优势。
- 本轮结果先作为独立探索证据交由作者决定是否扩表；不自动修改冻结的 v15。若纳入论文，必须完整报告两数据集和两噪声率，不得只选 Coauthor-CS。
