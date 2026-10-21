# 执行计划：去噪 teacher 蒸馏探针（Phase MLP-9）

日期：2026-07-02
执行者：Codex
定位：**写作前最后一个可选探针（seed0 半天，过门才 5-seed）。MLP-8 已证明：蒸馏是 X-only MLP 通向 GCN 精度的唯一有效通道（GLNN student u0.4 达 0.74/0.65/0.84），但裸噪声 GCN teacher 的 GLNN 压过球 expansion。本探针测最后一个未判死的组合：用球去噪过的 teacher（SoftDist-GCN[X‖H]，Cora u0.4 0.8137 > 裸 GCN 0.7842）再蒸馏，去噪增益能否传导给 student、且功劳归得了球。跑完无论结果 → 论文骨架 v2 → 动笔，不再有下一个实验。**

## 0. 背景与诚实先验

- **假设链**：teacher 训练标签更干净 → teacher softmax 更干净 → student 更好。teacher 端球增量已 5-seed 确认（ball-on-H：Cora +1.10pp 4/5、CiteSeer +1.30pp 5/5 over gcn_xh_ce）。
- **诚实先验（偏负，写在前面防自欺）**：
  1. teacher 端球增量只有 ~1.1-1.3pp，经蒸馏传导后可能进一步衰减；
  2. P0 已证明球目标在 MLP train-target 角色被 kNN/kmeans 平替（第 6 次），同样的平替大概率在 teacher 层复现——所以 kNN 去噪 teacher 是本探针的命门对照，缺它整个探针无效；
  3. PubMed 上 SoftDist-XH teacher 本身弱于 GCN+CE（Phase 10），预期负；
  4. u0.6 上 GCN 线球增量崩（Cora real < shuffled，噪声钟形曲线），球 teacher 在 u0.6 可能反而更差——u0.6 只作信息位，不作过门依据。
- **为什么仍值得跑**：这是"MLP 精度差 GCN ~20pp"痛点唯一未判死的正面回应；且 MLP-8 已把 GLNN/anchors 跑好，本探针新增运行极少。四个结局都有明确出路（见 §4 决策树）。

## 0.1 红线

1. clean 只进评测；噪声只加 train；teacher（含所有变体）只见噪声 train 标签；球/kNN 目标构造不见 clean。
2. student 一律同一 X-only 2-layer 弱 MLP、同 config（与 MLP-8 `mlp_distill_gcn` 完全一致：全节点 teacher softmax τ=1，KL/CE 目标，蒸馏细节逐键相同，只换 teacher）。
3. teacher 变体之间唯一差异 = 训练目标的构造来源（CE / 球软目标 / kNN 软目标 / shuffled 球软目标），backbone 与超参逐键相同（均为 GCN[X‖H]，除 T1 复用的 GCN(X)）。
4. kNN teacher 的 k 匹配同 run 球诊断的 `mean_train_per_ball`（P0 同规），匹配值写进报告；kNN 软目标用与 SoftDist 相同的 multiclass-α 公式（α 由 vote margin 算，只换 assignment 来源），保证公平。
5. ROCm venv；`--split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7` 显式传参；seed0 先行，过门才 5-seed；paired 同 seed 同 split。
6. 报告必须同时记录 teacher 自身 test acc 与 student test acc（看去噪增益的传导率）。

## 0.2 复用与新增

- **复用（不重跑）**：
  - `runs/mlp8_replacement_expansion_20260702/` 中 `mlp_distill_gcn`（= T1 GLNN student）与 train-only β0.5 anchor 各 setting seed0 行；
  - teacher 端 5-seed 锚点：SoftDist-XH Cora 0.8137±0.0067 / CiteSeer 0.6891±0.0159；gcn_xh_ce Cora ~0.803 / CiteSeer ~0.676（Phase 5）；
  - 现成 predictor：`frozengrace_gb_gcn_softdist_xh`、`frozengrace_gb_gcn_softdist_xh_shuffled`、`gcn_xh_ce`；`build_knn_soft_targets`（P0 已有，需接 multiclass-α 路径）；MLP-8 的蒸馏 student 训练代码。
- **新增**：
  - 蒸馏 predictor 泛化：把 `mlp_distill_gcn` 的 teacher 指定改为可配置（`gb:` 键 `teacher_method` + teacher config 路径），派生四个方法名：
    - `mlp_distill_gcn_xh`（T2：teacher = gcn_xh_ce）
    - `mlp_distill_softdist_xh`（T3：teacher = SoftDist-GCN[X‖H]，本假设）
    - `mlp_distill_knn_xh`（T4：teacher = GCN[X‖H] + kNN-in-H multiclass-α 软目标，命门对照）
    - `mlp_distill_softdist_xh_shuffled`（T5：teacher = shuffled 球软目标，结构 sanity）
  - T4 需要一个 GCN[X‖H] + kNN 软目标的 teacher predictor（新，复用 P0 的 kNN 目标构造 + SoftDist-XH 的训练循环）。
  - 单测：(a) 各 teacher 变体除目标构造外配置逐键相同；(b) kNN 目标不见 clean（篡改 clean 不变）；(c) k 匹配逻辑；(d) student 蒸馏路径与 `mlp_distill_gcn` 逐键一致（回归：teacher 换回 GCN(X) 时结果复现 MLP-8）。

## 1. 矩阵（seed0 smoke）

```
Datasets: Cora, CiteSeer, PubMed（PubMed 预期负，为完整性保留）
Noise: uniform 0.4（主判定）; uniform 0.6（信息位，不进过门）
Seed: 0（过门后幸存 setting 扩 seeds 0-4）
Student 臂（全部同一 X-only MLP + 同蒸馏配置，只换 teacher）:
  T1  GLNN: teacher = GCN(X)+CE          （复用 MLP-8，不重跑）
  T2  teacher = GCN[X‖H]+CE              （新：剥离 H-input 效应）
  T3  teacher = SoftDist-GCN[X‖H]        （新：球去噪，本假设）
  T4  teacher = GCN[X‖H]+kNN-α软目标     （新：命门平替对照）
  T5  teacher = SoftDist-XH shuffled      （新：结构 sanity）
参照（不重跑）: train-only β0.5、MLP+CE（MLP-8 行）；GCN+CE(X) 历史精度
新增运行量: 3 数据集 × 2 噪声 × 4 新臂 = 24 个 teacher+student run，半天内
```

## 2. ★预注册判据（u0.4，seed0 定方向；5-seed 定结论 ≥4/5 & ≥2/3 数据集）

| 门 | 条件 | 含义 |
|---|---|---|
| **G1 胜 GLNN** | T3-student − T1-student ≥ +1pp | 去噪 teacher 路线整体成立 |
| **G2 归因去噪** | T3-student − T2-student > 0 | 增益来自去噪目标，不是 teacher 多吃 H |
| **G3 命门归因球** | T3-student − T4-student ≥ +1pp | 球去噪不被 kNN 去噪平替（先验：大概率被平替） |
| **G4 结构 sanity** | T3-student > T5-student | 真球结构而非任意分组 |

- 过门顺序：G1 不过 → 直接收（G2-G4 无意义）；G1 过才看 G2/G3/G4。
- u0.6 结果只记录、只解释，不作为任何门的依据（球 teacher 在 u0.6 预期退化）。

## 3. 判读警示（防两类假阳性）

1. **T3 略胜 T1 但 T2 也同样胜 T1** → 功劳是 [X‖H] teacher 输入，不是球。此时若 G2 不过，不得写"球去噪 teacher 有效"。
2. **T3 ≈ T4（差 <1pp 或方向不稳）** → teacher 层平替成立（第 7 次），与 P0 同构。此时即使 G1/G2 过，claim 只能写"**去噪 teacher 蒸馏**（局部共识软目标实现）优于裸噪声 teacher"，球降为实现方式之一，品牌措辞与导师商量，不藏。

## 4. 决策树（跑完即定骨架 v2，无下一个实验）

| 结局 | 动作 |
|---|---|
| G1+G2+G3 全过 → 5-seed 复确认 | 最好结局：论文升级"ball-denoised teacher 的 graph-free 蒸馏"，精度痛点正面解决；骨架 v2 加蒸馏主结果节 |
| G1+G2 过、G3 不过 | 精度故事可讲但功劳归"去噪 teacher"非球：作为 context/扩展节写入（诚实标注 kNN 平替），主线仍按现脊梁；与导师商量是否要这个不带球品牌的增量 |
| G1 不过（或 G2 不过） | 探针关闭，GLNN 数字进 positioning/context + Intro scope 声明；按现脊梁动笔，一天止损 |
| 任何结局 | 报告含 teacher acc 与 student acc 对照表（传导率）；u0.6 行为如实记录 |

## 5. 执行顺序

```
1. 泛化蒸馏 predictor（teacher 可配置）+ T4 kNN-teacher predictor + configs + 单测（§0.2）
2. seed0 跑 §1 矩阵新增 24 run（先 u0.4 后 u0.6；PubMed 最后）
3. 按 §2 判门 → G1 过且 G2 过 → 幸存 setting 扩 seeds 0-4（仅 u0.4）
4. 汇总报告：teacher/student 配对表 + 四门判定 + 传导率 + §4 分支结论
```

- 输出目录：`external/NoisyGL/runs/mlp9_denoised_teacher_20260702/`
- 报告：`mlp9_report.md`（teacher acc 表 + student 四门表 + k 匹配值 + 分支判定）

## 6. 完成后

- 更新记忆（MLP-9 四门结果 + 走哪个分支 + teacher 层平替是否第 7 次成立）。
- 无论分支 → 论文骨架 v2（按 MLP-8+MLP-9 双结果定稿 claim 措辞）→ 动笔。本探针是写作前最后一个实验。

---

## 7. Seed0 执行结果备案（2026-07-02）

输出目录：`runs/mlp9_denoised_teacher_20260702/`

核心文件：

- `mlp9_uniform0p4_seed0.csv`
- `mlp9_uniform0p6_seed0.csv`
- `mlp9_combined_raw.csv`
- `mlp9_gates.csv`
- `mlp9_report.md`

### 7.1 u0.4 主判定结果

| Gate | 判据 | seed0 结果 | 决策 |
|---|---|---:|---|
| G1 胜 GLNN | T3 - T1 >= +1pp | mean +3.73pp, 2/3 datasets pass | pass |
| G2 归因去噪 | T3 - T2 > 0 | mean +0.46pp, 2/3 datasets positive | pass |
| G3 命门归因球 | T3 - T4 >= +1pp | mean -0.02pp, 0/3 datasets pass | not-pass |
| G4 结构 sanity | T3 > T5 | mean +1.01pp, 2/3 datasets positive | pass |

逐数据集关键 student test acc（u0.4）：

| Dataset | T1 GCN(X) teacher | T2 GCN[X‖H] CE teacher | T3 SoftDist teacher | T4 kNN teacher | T5 shuffled teacher |
|---|---:|---:|---:|---:|---:|
| Cora | 0.7435 | 0.7942 | 0.8090 | 0.8047 | 0.7794 |
| CiteSeer | 0.6482 | 0.6946 | 0.6912 | 0.6993 | 0.6954 |
| PubMed | 0.8368 | 0.8381 | 0.8404 | 0.8369 | 0.8352 |

### 7.2 u0.6 信息位

u0.6 不参与过门。结果显示 T3 相对 T1/T2 在 3/3 数据集上有正增量，但 G3 仍不过：

| Gate | u0.6 mean delta | 备注 |
|---|---:|---|
| G1 T3 - T1 | +9.03pp | 高噪声下去噪 teacher 明显强于裸 GCN(X) teacher |
| G2 T3 - T2 | +1.51pp | 高噪声下 soft-target teacher 有传导 |
| G3 T3 - T4 | +0.65pp | 只有 1/3 数据集达到 +1pp，球仍未稳定超过 kNN |
| G4 T3 - T5 | +1.24pp | 真结构优于 shuffled |

### 7.3 最终决策

分支：`denoised_teacher_possible_but_ball_replaced_by_knn`

解释：

- 去噪 teacher 蒸馏路线有信号：T3 在 u0.4 下胜过 T1，且总体略胜 T2。
- 但球归因失败：T3 没有稳定超过 kNN teacher，G3 为 0/3 pass。这意味着 teacher 层也复现了 P0 的平替问题：球在 H 空间中的信息可以被更细粒度的 kNN 局部共识解释。
- Seed0 当时初判为不进入 5-seed 主线扩展；后续按用户要求，因 G1+G2 已过，仍执行了 §8 的完整 5-seed 复核。最终结论以 §8 为准。

### 7.4 实现与验证

代码改动：

- `predictor/FrozenGraceGB_GCN_Predictor.py`：解耦 teacher/student 输入；T2-T5 teacher 可吃 `[X‖H]`，student 保持 X-only；支持 `teacher_target=soft_target`。
- `config/mlp_distill_gcn_xh/`
- `config/mlp_distill_softdist_xh/`
- `config/mlp_distill_knn_xh/`
- `config/mlp_distill_softdist_xh_shuffled/`
- `scripts/run_mlp9_denoised_teacher_20260702.ps1`
- `scripts/summarize_mlp9_denoised_teacher.py`
- `tests/test_mlp9_denoised_teacher.py`

验证：

- `python -m py_compile predictor/FrozenGraceGB_GCN_Predictor.py scripts/summarize_mlp9_denoised_teacher.py`
- `python -m pytest tests/test_mlp9_denoised_teacher.py tests/test_mlp8_replacement_expansion.py -q`
- 结果：`13 passed, 39 warnings, 34 subtests passed`

---

## 8. 5-seed 扩展结果备案（2026-07-02 追加）

触发条件：seed0 中 G1+G2 已过。按用户要求执行完整 5-seed 扩展，而不是只扩 u0.4。

执行范围：

- 数据集：Cora / CiteSeer / PubMed
- 噪声：uniform 0.4 / uniform 0.6
- 方法臂：T1-T5
- seeds：0-4
- seed0 全部复用既有行，不重跑；新增 seeds 1-4 共 `5 methods x 3 datasets x 2 noise rates x 4 seeds = 120` run。
- T1（GLNN, teacher=GCN(X)）也补齐 seeds 1-4，用于 G1 paired 比较。
- 每个 seed 内 teacher 重训、球/kNN 目标重建；kNN 的 `matched_k` 按该 seed 球诊断的 `mean_train_per_ball` 匹配。

新增与汇总文件：

- `runs/mlp9_denoised_teacher_20260702/mlp9_uniform0p4_seeds_1_2_3_4.csv`
- `runs/mlp9_denoised_teacher_20260702/mlp9_uniform0p6_seeds_1_2_3_4.csv`
- `runs/mlp9_denoised_teacher_20260702/mlp9_combined_raw.csv`
- `runs/mlp9_denoised_teacher_20260702/mlp9_aggregate.csv`
- `runs/mlp9_denoised_teacher_20260702/mlp9_paired_deltas.csv`
- `runs/mlp9_denoised_teacher_20260702/mlp9_gates.csv`
- `runs/mlp9_denoised_teacher_20260702/mlp9_decomposition_deltas.csv`
- `runs/mlp9_denoised_teacher_20260702/mlp9_knn_matched_k.csv`
- `runs/mlp9_denoised_teacher_20260702/mlp9_report.md`

汇总脚本输出：

- `MLP9_ROWS=150`
- `MLP9_BRANCH=denoised_teacher_possible_but_ball_replaced_by_knn`

### 8.1 u0.4 主判定（5-seed）

| Gate | 判据 | Overall mean delta | 数据集通过数 | 5-seed 决策 | 逐数据集 |
|---|---|---:|---:|---|---|
| G1 胜 GLNN | T3 - T1 >= +1pp | +2.22pp | 2/3 | pass | Cora +2.65pp 4/5; CiteSeer +3.31pp 4/5; PubMed +0.69pp 2/5 |
| G2 归因去噪 | T3 - T2 > 0 | +0.39pp | 2/3 | pass | Cora +0.37pp 4/5; CiteSeer +0.59pp 3/5; PubMed +0.21pp 4/5 |
| G3 命门归因球 | T3 - T4 >= +1pp | -0.26pp | 0/3 | not-pass | Cora -0.39pp 0/5; CiteSeer -0.17pp 0/5; PubMed -0.22pp 0/5 |
| G4 结构 sanity | T3 - T5 > 0 | +0.84pp | 2/3 | pass | Cora +1.79pp 4/5; CiteSeer +0.80pp 4/5; PubMed -0.07pp 3/5 |

结论：u0.4 下，去噪 teacher 蒸馏路线成立（G1/G2 过），真球结构相对 shuffled 有总体信号（G4 过），但球特异性归因失败（G3 不过）。kNN 局部共识在 teacher 层仍能平替甚至略强于球目标。

### 8.2 u0.6 信息位（5-seed）

u0.6 不作为主判定，但作为扩展节候选证据单独记录。

| Gate | 判据 | Overall mean delta | 数据集通过数 | 5-seed 决策 | 逐数据集 |
|---|---|---:|---:|---|---|
| G1 胜 GLNN | T3 - T1 >= +1pp | +7.38pp | 3/3 | pass | Cora +5.77pp 5/5; CiteSeer +8.98pp 5/5; PubMed +7.39pp 5/5 |
| G2 归因去噪 | T3 - T2 > 0 | +1.39pp | 3/3 | pass | Cora +1.84pp 5/5; CiteSeer +1.24pp 4/5; PubMed +1.09pp 5/5 |
| G3 命门归因球 | T3 - T4 >= +1pp | +0.20pp | 0/3 | not-pass | Cora +0.20pp 1/5; CiteSeer +0.32pp 2/5; PubMed +0.08pp 1/5 |
| G4 结构 sanity | T3 - T5 > 0 | +1.46pp | 2/3 | pass | Cora +2.77pp 5/5; CiteSeer +1.21pp 4/5; PubMed +0.42pp 3/5 |

重点观察：u0.6 下 G2 很强，T3-T2 在 3/3 数据集为正，且 Cora/PubMed 为 5/5、CiteSeer 为 4/5。这说明高噪声时 denoised teacher 相对 H-input CE teacher 的蒸馏增益更稳定。但 G3 仍完全不过门，因此不能把增益归因到粒球本身。

### 8.3 三层分解表

| Noise | Dataset | D1: T2-T1 H-input 增量 | D2: T4-T2 kNN 去噪增量 |
|---:|---|---:|---:|
| 0.4 | Cora | +2.28pp, 5/5 | +0.76pp, 4/5 |
| 0.4 | CiteSeer | +2.71pp, 5/5 | +0.76pp, 4/5 |
| 0.4 | PubMed | +0.48pp, 4/5 | +0.42pp, 4/5 |
| 0.6 | Cora | +3.94pp, 4/5 | +1.64pp, 3/5 |
| 0.6 | CiteSeer | +7.74pp, 5/5 | +0.92pp, 4/5 |
| 0.6 | PubMed | +6.30pp, 5/5 | +1.02pp, 4/5 |

解释：第一层 `[X||H] teacher` 增量稳定且是最大来源；第二层局部共识去噪也有小幅增量，但 kNN 已经能获得该增量，因此它不是粒球特异贡献。

### 8.4 kNN matched-k 审计

`matched_k` 逐 seed 写入 `mlp9_knn_matched_k.csv`，摘要如下：

| Dataset | Noise | seeds | matched_k | matched_num_balls |
|---|---:|---|---|---|
| Cora | 0.4 | 0-4 | 10,10,10,10,10 | 52,53,52,53,52 |
| Cora | 0.6 | 0-4 | 10,10,10,10,10 | 52,53,52,53,52 |
| CiteSeer | 0.4 | 0-4 | 10,10,10,10,11 | 64,64,64,64,63 |
| CiteSeer | 0.6 | 0-4 | 10,10,10,10,11 | 64,64,64,64,63 |
| PubMed | 0.4 | 0-4 | 62,62,62,62,62 | 64,64,64,64,64 |
| PubMed | 0.6 | 0-4 | 62,62,62,62,62 | 64,64,64,64,64 |

### 8.5 5-seed 最终决策

分支维持：`denoised_teacher_possible_but_ball_replaced_by_knn`

可写结论：

- Denoised teacher distillation 是有效的：T3 在 u0.4/u0.6 均显著优于 T1，且相对 T2 有正增量。
- 但粒球品牌归因不成立：T3 没有超过 T4 kNN teacher，G3 在 u0.4/u0.6 均为 0/3 数据集通过。
- 因此 MLP-9 不应升级为主方法；最多作为扩展/边界实验：**local-consensus-denoised teacher can improve graph-free MLP distillation, but the improvement is not granular-ball-specific under the matched-k control.**

### 8.6 5-seed 验证

验证命令：

- `python scripts/summarize_mlp9_denoised_teacher.py --run_dir runs/mlp9_denoised_teacher_20260702`
- `python -m py_compile scripts/summarize_mlp9_denoised_teacher.py predictor/FrozenGraceGB_GCN_Predictor.py`
- `python -m pytest tests/test_mlp9_denoised_teacher.py tests/test_mlp8_replacement_expansion.py -q`

结果：

- `MLP9_ROWS=150`
- `13 passed, 39 warnings, 34 subtests passed`
