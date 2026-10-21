# 执行计划：审稿防御批 + Linear 发现后续（Phase MLP-15）

日期：2026-07-04
执行者：Codex
定位：**发稿前最后一批防御性补齐，不是新机制。** 来源两处：(1) 全线证据架构审计发现的遗漏（inductive 协议缺失、β 选择审计、clean 列、图空间共识对照、显著性检验）；(2) linear 消融（`runs/linear_ablation_20260704`）引出的 PubMed 容量伪影假说需要一个公平对照关死。**硬规则：本计划之后，任何新实验必须满足"导师点名"或"防御一个措辞挡不住的具名审稿攻击"之一。**

## 0. 总红线

1. **主线不换**：主方法保持 SoftDist-MLP β=0.5、主表 A 不动。linear 学生只进消融/分析——基于 test 结果换主线学生 = MLP-13 拒绝过的 test 知情调参，同一纪律对自己执行。
2. 所有新臂 clean 只评测、噪声只加 train、协议显式 percent 20/10/70、ROCm cuda:0、paired 同 seed。
3. 每个子项的判读分支预注册（见各节），结果无论落哪支都如实进论文对应位置。
4. P0 三件零成本项立即做，不等导师；P1-P5 按优先级推进，导师若有 sign-off 意见可裁剪。

## P0 —— 零成本三件（纯重算，立即做）

### P0a β 选择的 val 审计
- **动机**：β=0.5 是 MLP-4 看 5-seed test 选的；MLP-13 我们用"test 知情"拒绝了换默认——同一标准必须对自己执行一次。
- **做法**：从 MLP-4 的 run CSV（`valid` 列已落盘）重算：若当时用 clean-val 精度选 β∈{0,0.5,1}，逐数据集×噪声会选谁。
- **分支**：val 也选 0.5（预期，val/test 相关）→ 论文 Setup 写一句 "β selected by validation accuracy"，审计表进附录；val 选了别的 → 如实记录差异幅度，论文写 "β=0.5 fixed a priori after initial exploration; validation-based selection would choose X with ≤Ypp difference"。
- **补充证据（写显式）**：CoauthorCS/AmazonPhoto/pair/instance 全部是 β 冻结后加入且确认——真实的 out-of-sample 泛化。

### P0b 显著性检验
- 对主表 A 全部 "Ours vs 每 baseline" 格、shuffled 对照、表 B 关键行补 paired t-test 与 Wilcoxon signed-rank（n=5 paired seeds），输出 p 值表。论文标注方式：主表加显著性标记（*p<0.05），完整表进附录。n=5 的检验功效有限，如实写 "5 paired seeds"。

### P0c linear 消融补全
- `linear_ablation_20260704` 缺的 30 行 = CoauthorCS/AmazonPhoto 的 SoftDist-MLP/MLP+CE/MLP-shuffled 参照（MLP-10/12 现成），summarizer 合并（照 MLP-14 教训：**按 train_percent==0.2 过滤 + 断言参照均值等于主表值**）。补 delta：softdist_linear_vs_softdist_mlp 在五数据集的完整配对。

## P1 —— Inductive/冷启动探针（最高优先，唯一可能补出新卖点）

- **动机**：论文动机句含"冷启动/孤立新节点"，但全部数字是 transductive（GRACE 见过 test 节点）。GLNN/GSDN/NOSMOG 全部单独报 inductive；GLNN 自报 tran→ind 掉 ~6pp（Cora）。现有 7c-ext 只是评测期删边，撑不起冷启动措辞。
- **协议**（GLNN 式 hold-out）：test 节点 20% 划为 inductive 集，训练前从图中移除（连边一起）；**GRACE 在 observed 子图上重新预训练**（每 seed 一次）；球/目标只建在 observed 节点上；student 训练完对 inductive 节点纯 X 前向。
- **矩阵**：Cora/CiteSeer/PubMed × u0.4 × {ours-train-only, mlp_ce, shuffled, ours-distill(teacher 只见 observed)} × seeds 0-4 = 60 run + 15 次 GRACE 重预训练。1-2 天。
- **诚实先验（双向）**：正向——train-only 的目标只用 train 票，对无标签节点依赖是全家族最弱，tran→ind 掉幅可能小于蒸馏系（若成立 = 新卖点"inductive 退化最小"）；反向——GRACE 在小子图上预训练 H 质量下降，球纯度可能不过入场诊断（先跑 seed0 诊断，purity ≥0.65 门）。
- **分支**：掉幅 ≤ CE 的掉幅 → 写 inductive 小节，冷启动措辞保留；掉幅明显大 → 冷启动措辞从动机删除，协议标注 transductive，掉幅如实进 limitation。

## P2 —— Clean（0% 噪声）列

- **动机**：LNL 论文标配"药不伤健康人"。β=0 时代测过安全（+0.4~+2.1pp）；β=0.5 更硬，球内少数派在干净标签下可能受伤，未测。
- **矩阵**：{ours, mlp_ce, shuffled} × 5 数据集 × clean × seeds 0-4 = 75 run，半天。
- **分支**：ours ≥ CE−0.5pp → 一句话进 Setup；ours 明显低于 CE → 如实进 limitation（"半硬共识目标在完全干净标签下有 Xpp 代价"），不改主表。

## P3 —— LP-target 臂（图空间共识对照，关死"要 GRACE 干嘛"）

- **动机**：监督侧消融缺"图空间"一角——X 空间共识（X-ball，塌缩）和 H 空间共识（主线）都测过，"直接在原图上标签传播当目标"没测。审稿人一句"噪声标签在图上传播两步不就是共识吗"目前只有 GSDN 的间接答案。
- **做法**：train 噪声标签在邻接上传播（2 步、行归一化，标准 LP），传播分布当 in-group 分布用同一 multiclass-α + β=0.5 公式构造目标，同一 MLP 训练。Cora/CiteSeer/PubMed × u0.4 × 5 seeds = 15 run + 目标构造器 + 单测（不见 clean）。
- **分支**：LP-target 明显弱于 H-consensus-target（预期：原图邻接含噪边、传播扩散噪声标签，与 GSDN 同病）→ "为什么需要 GRACE 表征空间"有了正面消融；LP-target 接近或更强 → 如实披露，H 的必要性 claim 收窄到"至少等效且免邻接"，并入 §6。

## P4 —— PubMed linear-backbone 探针（linear 发现的公平对照）

- **动机**：linear 消融显示 PubMed 上 Linear+CE=79.35 已超全部 MLP 系选择族（small-loss 77.30/JoCoR 77.84），SoftDist-Linear=80.03 数值超主表最强 baseline（DivideMix-lite 78.28）——**PubMed 边界可能部分是"弱 MLP 记忆噪声"的容量伪影**。但这是跨 backbone 比较，写进论文前必须给选择族同样的 linear backbone。
- **矩阵**：{linear_small_loss, linear_jocor, linear_dividemix_lite, linear_ct} × PubMed × 5 噪声设置 × seeds 0-4 = 100 run，半天（线性模型极快）。适配：只换 backbone 为线性层，策略逻辑不动，逐键单测。
- **分支**：选择族 linear 版仍胜 SoftDist-Linear → PubMed 边界成立且与容量无关，边界解释不变；SoftDist-Linear ≥ 选择族 linear 版 → PubMed 边界改写为"容量依赖的边界：弱 MLP 下选择族胜，线性学生下本方法追平/反超"——**边界收窄是重大改写，需同步骨架 §6 与主表脚注**（主表仍是 MLP backbone，linear 结果进消融+分析）。
- **无论哪支**：linear 系列进论文的位置 = 容量消融表（SoftDist-Linear vs Linear+CE vs linear-shuffled 五数据集，10/10 real>shuffled 的容量无关 claim）+ §6 容量分析段（"容量是噪声记忆旋钮；标尺规律在线性下游复现：Linear+CE 越强的数据集球增量越小"）。

## P5 —— 可选保险（导师定，默认不跑）

- seeds 5-9 复核 headline 设置（ours/CE/shuffled × 5 数据集 × u0.4 = 75 run）；
- 换 2 份 GRACE 预训练种子测 H 方差（部分 checkpoint 现成）。

## 不做 + 一句 limitation

- 真实噪声数据集：图 LNL 无标准真实噪声基准（DND-NET/NRGNN/RNCGLN 全合成），不跑；limitation 写 "instance-dependent noise 是最接近真实噪声的代理"。

## 运行量汇总

| 项 | 运行量 | 时长 |
|---|---:|---|
| P0a/b/c | 0（纯重算） | 数小时 |
| P1 inductive | 60 run + 15 GRACE 预训练 | 1-2 天 |
| P2 clean 列 | 75 run | 半天 |
| P3 LP-target | 15 run | 半天内 |
| P4 PubMed linear | 100 run | 半天 |
| 合计（P5 除外） | ~250 run + 15 预训练 | ~3 天 |

## 产出

- 输出目录：`runs/mlp15_review_defense_20260704/`（P0c 直接更新 linear_ablation 目录）
- 报告 `mlp15_report.md`：P0a 审计表 / P0b p 值表 / P1 tran-vs-ind 对照 + 掉幅表 / P2 clean 列 / P3 三空间共识对照（X/graph/H）/ P4 PubMed linear 全对照 + 边界判定；每项对号入座预注册分支。

## 完成后

- 骨架 v2 更新：Setup（β 审计句 + clean 列）、§6（容量分析段 + 三空间消融 + PubMed 边界措辞按 P4 分支）、动机句（按 P1 分支保留或删冷启动）、附录（敏感性 + 显著性 + inductive）。
- 更新记忆与 `99_session_handoff.md`。
- **此后执行硬规则：新实验必须"导师点名"或"防御具名攻击"。Method + Setup 正文与本计划并行动笔（不依赖任何 sign-off，品牌名留占位符）。**

---

## 6. 执行结果备案（Codex 跑完后按 MLP-8~14 格式追加于此）

### 2026-07-04 Codex 执行记录

输出目录：`runs/mlp15_review_defense_20260704/`。本轮已完成 P0/P2/P3/P4；P1 inductive 需要重新预训练 observed-subgraph GRACE，未在本轮执行；P5 默认不跑。

#### P0a β validation audit

β=0.5 不能写成"由 validation 统一选出"。审计发现 Cora/CiteSeer/PubMed 的若干 uniform 0.2/0.4 设置在 validation 或 test 上会选择 β=1，PubMed uniform0.2 仍偏向 β=0.5。论文措辞应改为：β=0.5 是初始探索后冻结的保守默认；CoauthorCS/AmazonPhoto/pair/instance 是冻结后 out-of-sample 验证，不能把 β=0.5 包装成全局 val-best。

#### P0b 显著性检验

已生成 `p0b_paired_significance.csv`。注意 n=5 时 Wilcoxon signed-rank 即使 5/5 seeds 同向也常为 0.0625，论文中只可写 "5 paired seeds; paired t-test/Wilcoxon reported"，不要把 n=5 的 Wilcoxon 当作否定稳定性的证据。

#### P0c linear capacity

完整表已生成 `p0c_linear_capacity_aggregate.csv` / `p0c_linear_capacity_paired_deltas.csv`。核心结果：

| dataset | MLP+CE | Linear+CE | SoftDist-MLP | SoftDist-Linear | MLP-shuffled | Linear-shuffled |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 44.24±2.62 | 47.86±1.30 | 55.59±1.50 | 54.06±2.26 | 43.63±1.62 | 47.84±1.59 |
| CiteSeer | 43.50±3.02 | 50.12±1.49 | 53.83±2.75 | 54.88±3.36 | 42.35±2.74 | 48.47±2.05 |
| PubMed | 63.77±4.40 | 79.35±0.45 | 71.90±0.88 | 80.03±0.57 | 62.72±4.49 | 78.15±0.45 |
| Coauthor-CS | 88.42±0.62 | 89.59±0.18 | 90.62±0.57 | 91.03±0.30 | 87.60±1.03 | 89.33±0.11 |
| Amazon-Photo | 73.90±1.80 | 82.76±1.11 | 81.64±0.93 | 86.46±0.37 | 72.06±2.97 | 81.61±0.81 |

结论：SoftDist-Linear 在 5/5 数据集上超过 Linear+CE 与 Linear-shuffled，说明局部共识监督不是 MLP 容量伪影；但 linear 学生在 PubMed/Amazon-Photo/Coauthor-CS/CiteSeer 上明显强于 MLP 学生，论文里应把 "student capacity" 作为重要分析变量，而不是只讲 MLP。

#### P2 clean-label safety

已跑 clean/noise_rate=0.0，输出 `p2_clean_aggregate.csv` / `p2_clean_paired_deltas.csv`：

| dataset | MLP+CE | Ours β=0.5 | Shuffled | Ours-CE |
|---|---:|---:|---:|---:|
| Cora | 66.49±1.19 | 66.93±1.39 | 63.12±0.94 | +0.44pp |
| CiteSeer | 64.55±0.75 | 66.88±0.43 | 63.68±0.77 | +2.33pp |
| PubMed | 84.63±0.22 | 84.65±0.41 | 84.87±0.29 | +0.02pp |
| Coauthor-CS | 92.96±0.27 | 92.08±0.38 | 91.47±0.69 | -0.88pp |
| Amazon-Photo | 88.82±0.77 | 87.62±0.62 | 86.67±0.93 | -1.20pp |

结论：clean safety 是混合结论。Cora/CiteSeer 正向，PubMed 近似持平；Coauthor-CS/Amazon-Photo 在完全干净标签下低于 CE 约 0.9-1.2pp。论文应写成 limitation：半硬局部共识在强特征可分、标签干净时可能带来轻微代价；不要写"完全无害"。

#### P1 inductive / cold-start probe

已实现独立 runner `scripts/run_mlp15_inductive_probe.py`，严格按 P1 协议做 observed-subgraph：从 test 节点中抽 20% inductive hold-out，训练前从 GRACE 图中移除；GRACE 只在 observed 子图上重新预训练；球/target 只在 observed H 上构造；inductive 节点不参与球、teacher 或 student 训练。

按预注册先跑 seed0 入场诊断。Cora 与 CiteSeer 已完成，输出 `p1_inductive_seed0_diagnostic_raw.csv` / `p1_inductive_diagnostics.csv` / `p1_inductive_report.md`：

| dataset | observed | inductive | GRACE epochs | num_balls | true_train_purity | true_all_purity | noisy_train_purity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 2329 | 379 | 200 | 1 | 0.320 | 0.304 | 0.226 |
| CiteSeer | 2861 | 466 | 200 | 1 | 0.218 | 0.208 | 0.202 |

预注册门槛是 seed0 球 purity ≥0.65 才进入 5-seed downstream；Cora/CiteSeer 均 `num_balls=1` 且 purity 远低于门槛，因此 P1 **gate fail**。PubMed 默认 GRACE 需要 1500 epoch，原诊断进程已进入 PubMed 后因前两图已明确不过门而中止，避免无效算力消耗。结论：当前严格 inductive 协议下，observed-subgraph GRACE 表示无法形成可用粒球；冷启动/inductive 卖点不能保留，应将主协议明确写为 transductive，并把 P1 作为 limitation 或 appendix boundary。

#### P3 LP-target graph-space consensus

已实现 `lp_mlp_softdist_beta05`：train noisy label 在行归一化邻接上传播 2 步，传播分布进入同一 multiclass-α + β=0.5 target 公式；加入 no-clean-label 单测。输出 `p3_lp_target_aggregate.csv` / `p3_lp_target_paired_deltas.csv`。

| dataset | MLP+CE | LP-target MLP | SoftDist-MLP | Shuffled |
|---|---:|---:|---:|---:|
| Cora | 44.24±2.62 | 47.34±2.00 | 55.59±1.50 | 43.63±1.62 |
| CiteSeer | 43.50±3.02 | 46.98±1.64 | 53.83±2.75 | 42.35±2.74 |
| PubMed | 63.77±4.40 | 64.98±0.83 | 71.90±0.88 | 62.72±4.49 |

Paired delta：SoftDist-MLP 比 LP-target 高 Cora +8.25pp、CiteSeer +6.86pp、PubMed +6.92pp，全部 5/5 seeds 同向。结论：简单图空间标签传播 target 不是 SoftDist 收益的替代解释；这支持"需要 GRACE 表征空间中的局部共识，而不是原图邻接上的噪声传播"。

#### P4 PubMed linear-backbone fairness

已新增 linear_small_loss / linear_jocor / linear_ct / linear_dividemix_lite，并跑 PubMed 的 uniform0.2/0.4/0.6、pair0.4、instance0.4。输出 `p4_pubmed_linear_aggregate.csv` / `p4_pubmed_linear_best_robust_summary.csv`。

| noise | Linear+CE | SoftDist-Linear | Shuffled | best linear robust | Δ vs best robust |
|---|---:|---:|---:|---:|---:|
| uniform0.2 | 81.42±0.99 | 80.70±1.02 | 80.30±1.62 | Small-loss 69.60 | +11.10pp |
| uniform0.4 | 79.35±0.45 | 80.03±0.57 | 78.15±0.45 | DivideMix-lite 66.32 | +13.71pp |
| uniform0.6 | 57.32±6.02 | 59.66±5.68 | 56.05±5.86 | DivideMix-lite 52.27 | +7.39pp |
| pair0.4 | 66.61±1.46 | 69.94±1.16 | 63.67±0.90 | Small-loss 55.84 | +14.10pp |
| instance0.4 | 79.09±1.26 | 79.90±1.36 | 78.60±1.19 | DivideMix-lite 65.64 | +14.26pp |

结论：PubMed 的 MLP 主表边界不是"robust strategy 本身更强"。当所有方法都换成线性学生时，传统选择族明显崩溃，SoftDist-Linear 反而稳定超过 best linear robust；但 uniform0.2 下 Linear+CE 仍略高于 SoftDist-Linear，说明低噪声强线性可分时，直接 CE 仍是很强的边界。

#### 测试

- `pytest tests/test_frozen_grace_targets.py tests/test_mlp_softdist.py -q`：29 passed。
- `pytest tests/test_mlp_noise_baselines.py -q`：12 passed（P4 linear robust 注册与配置）。
