# 执行计划：kNN k-敏感性曲线——把 K-free 差异化从措辞变成测量（Phase MLP-11）

日期：2026-07-02
执行者：Codex
定位：**写作期的 claim 加固，不是机制翻案。** 测量两件事：(1) kNN-共识软标签的精度对 k 的敏感程度、最优 k 的跨数据集变异（先验：P0/MLP-9 的 matched_k 是 Cora/CiteSeer≈10、PubMed=62，差 6 倍——但那是从球诊断读出来的，kNN 部署时没有这个来源）；(2) 球式共识用同一套**全程冻结的默认构造参数**（max_balls=64、split_improvement=0.05，从未按数据集调过）能否贴住各数据集 kNN best-k 的精度。两种结局都进论文 §6："K-free 自适应粒度"要么获得测量支撑，要么如实降级。

## 0. 诚实先验（防自欺，写在最前）

1. **本实验不可能翻案"球优于 kNN"**——P0/MLP-9 双平替已 5-seed 盖章，第 118 篇文献审查也确认"固定表征上球赢 kNN"零先例。它只测"k 选择负担"的大小，这是骨架 v2 §6 已经口头 claim、但从未测量过的那句话。
2. **我们协议里 val 是干净的**，k 原则上可以用 val 调。所以即使 kNN 对 k 敏感，审稿人可以说"扫一遍 k 就行"。因此 R3（val 选 k）必须做：若 clean val 能可靠找到 best-k，K-free 优势收窄为"省一遍 k 扫描"（便利/效率 claim），照实写；若 val 选出的 k 次优，如实报差值。
3. 球的"免调"指 max_balls/split_improvement 是全项目冻结的全局默认，不是"零参数"——报告与论文措辞都要写成 "one fixed construction config across all datasets"，不写 "parameter-free"。

## 0.1 红线

1. kNN 目标构造与 P0 完全同管线（`build_knn_soft_targets` + multiclass-α + β=0.5 插值），唯一变量 = k；目标构造不见 clean。
2. **球臂不重跑**：复用 β=0.5 四数据集 u0.4 的 5-seed 现成数字（Cora/CiteSeer/PubMed 来自 MLP-5，CoauthorCS 来自 MLP-10）。
3. **回归校验先行**：Cora/CiteSeer u0.4 k=10 的新行必须逐 seed 复现 P0 的 matched-kNN 行（数字一致），不一致先修管线再跑其余 k。
4. 协议显式 `--split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7`；ROCm `cuda:0`；直接 5-seed（弱 MLP 便宜，P0 先例）；paired 同 seed 同 split。
5. C1/C2/C3 与结局分支预注册（§2），结果无论落哪支都写进论文 §6，不许因为结果不利就不引用。
6. 本计划不改主表、不改主线 claim；只更新骨架 v2 §6 的 K-free 句为测量版本。

## 0.2 复用与新增

- **复用**：`knn_mlp_softdist_beta05` predictor 与 configs（P0 已有，k 需改为可配置项——核对 `gb:` 键经 `_conf_value` 覆盖类属性的老坑，把实际生效的 k 落进 CSV 列）；球 β=0.5 与 shuffled 的四数据集 5-seed 行；P0 的 matched-kNN 行（回归校验参照）。
- **新增**：CoauthorCS 的 `knn_mlp_softdist_beta05_coauthorcs.yaml`（P0 只跑了三个老数据集）；汇总脚本 `summarize_mlp11_k_sensitivity.py`（k-曲线表、best-k 表、C1-C3 判定、R3 表）；单测：(a) k 配置真实生效（两个 k 的目标不同）；(b) k 覆盖 train 集内近邻不足时的退化处理（k > 球内 train 数的小数据集角落）；(c) 篡改 clean 不改变目标。

## 1. 矩阵（Stage A，直接 5-seed）

```
knn_mlp_softdist_beta05, k ∈ {3, 5, 10, 20, 50, 100}
  × {Cora, CiteSeer, PubMed, CoauthorCS} × uniform 0.4 × seeds 0-4
  = 120 run（弱 MLP，半天内）
复用参照（不跑）: 球 β=0.5 u0.4 四数据集 5-seed；shuffled；P0 matched-kNN 行
```

k 网格说明：log 间隔覆盖 matched_k 的已知范围（10~62）；k=100 为 PubMed/CoauthorCS 的上界探针。

## 2. ★预注册判读（u0.4，全 5-seed）

| 条件 | 判据 | 含义 |
|---|---|---|
| **C1 k-变异** | 各数据集 best-k 的 max/min ≥ 4 | 最优邻域尺寸跨数据集不可迁移 |
| **C2 k-敏感** | 用四数据集池化的单一全局 k* 时，至少 1 个数据集比自己 best-k 掉 ≥2pp | 单一 k 服务不了所有数据集 |
| **C3 球贴住** | 球默认配置 ≥ (该数据集 kNN best-k − 1pp) 或差距 ≤1σ，在 ≥3/4 数据集成立 | 球免调 k 达到 per-dataset 调优精度 |
| **R3 val 选 k**（无新运行，纯选择） | 逐数据集用 clean val acc 选 k，报所选 k 与其 test acc；与 best-k(test) 和球对照 | 测"扫 k"的实际可靠性与代价 |

### 结局分支（写死）

| 结局 | 论文动作 |
|---|---|
| C1+C2+C3 全成立 | §6 写测量版 K-free claim："球式共识用同一套固定构造参数达到各数据集 per-dataset best-k kNN 的精度（≤1pp/1σ），而 kNN 的最优 k 跨数据集差 N 倍、单一全局 k 最多损失 Xpp"；R3 的限定如实并列（"k 可由 clean val 扫描选出，K-free 的净优势是免扫描"） |
| C2 不成立（k 平坦） | K-free 卖点降级为便利性措辞，骨架 v2 §6 同步收窄；曲线照进论文（诚实边界） |
| C3 不成立（球输 best-k >1pp 于 ≥2 数据集） | 披露"球免调 k 但略输 per-dataset 调优 kNN"，K-free 框架带此 caveat 写 |

## 3. Stage C（可选附带臂：Codex 提议的 adaptive-β smoke，kill-level 公平对照）

先验强负（椭球/GMM 球内精细加权已被 β=0.5 吸收——MLP-5a；compactness/半径有 kNN 廉价等价物——文献审查确认），跑不跑由用户定；若跑，规则如下，半天：

```
Cora/CiteSeer × uniform 0.4 × seed0，5 臂：
  1. ball β0.5                （复用）
  2. ball adaptive-β          （新：β_B = f(compactness 或 radius)，映射规则先声明后跑）
  3. kNN β0.5 k=matched       （复用 P0）
  4. kNN adaptive-β           （新：同型规则、驱动量换第 k 近邻半径——公平命门臂，缺它无效）
  5. shuffled β0.5            （复用）
判据: 臂2 同时 > 臂1、臂3、臂4 各 ≥1pp，且 2/2 数据集 → 才谈 5-seed
任何其他结局 → 永久关闭（第 8 次平替检验），不调第二组规则，不进论文正文（最多 future work 一句）
```

## 4. 产出

- 输出目录：`runs/mlp11_k_sensitivity_20260702/`
- 报告 `mlp11_report.md` 必含：
  1. k-曲线总表（4 数据集 × 6 k，mean±std，5-seed）+ 球/shuffled 参照行；
  2. best-k 表 + C1/C2/C3 判定 + 全局 k* 的逐数据集损失；
  3. R3 表（val 选出的 k、其 test acc、与 best-k 及球的差）；
  4. 回归校验结果（k=10 vs P0 行逐 seed 对照）；
  5. （若跑 Stage C）五臂表 + 判定；
  6. 完整性 expected/observed/missing。

## 5. 完成后

- 骨架 v2 §6 "结构必要性 vs 形式可平替性"段的 K-free 句替换为测量版本（按 §2 结局分支措辞）。
- 更新记忆与 `99_session_handoff.md`。
- 本计划是写作前最后一个 claim 加固实验；此后只剩导师沟通包与正文写作。

---

## 6. 执行结果备案（Codex 跑完后按 MLP-8/9/10 格式追加于此）

### 2026-07-02 Codex Stage A 结果

执行范围：

- 已跑 Stage A：`knn_mlp_softdist_beta05`，`k ∈ {3,5,10,20,50,100}` × Cora/CiteSeer/PubMed/CoauthorCS × uniform0.4 × seeds 0-4。
- 未跑 Stage C：计划中标注为可选，用户未额外指定。
- 球 β=0.5 与 shuffled 按红线复用；未重跑。

代码与配置改动：

- 新增 `config/knn_mlp_softdist_beta05/knn_mlp_softdist_beta05_coauthorcs.yaml`。
- 给四个 `knn_mlp_softdist_beta05_*` config 补 `gb.matched_k: ~`，否则严格 `--config_overrides {"gb.matched_k": k}` 会被拦截。
- 新增 `scripts/run_mlp11_k_sensitivity_20260702.ps1`。
- 新增 `scripts/summarize_mlp11_k_sensitivity.py`。
- 新增 `tests/test_mlp11_k_sensitivity.py`。

输出目录：`runs/mlp11_k_sensitivity_20260702/`

主要输出：

- `mlp11_report.md`
- `mlp11_knn_curve.csv`
- `mlp11_best_k.csv`
- `mlp11_global_k_loss.csv`
- `mlp11_r3_val_selected.csv`
- `mlp11_c123_gates.csv`
- `mlp11_regression_check.csv`
- `mlp11_completeness.csv`

运行完整性：

- `MLP11_NEW_ROWS=120`
- `MLP11_REFERENCE_ROWS=40`
- expected groups = 24，missing groups = 0
- stderr 为空

回归校验：

- Cora k=10 vs P0 matched-kNN：5/5 seeds 精确一致。
- CiteSeer k=10 vs P0 matched-kNN：seeds 0-3 精确一致。
- CiteSeer seed4 的 P0 `matched_k=11`，而本实验固定 `k=10`，按执行前说明标记为 `skipped_p0_matched_k_11`，不视为管线错误。

#### k 曲线与判定

| Dataset | Best k | Best test | Ball β=0.5 | Gap(best−ball) |
|---|---:|---:|---:|---:|
| Cora | 20 | 57.20±2.26 | 55.59±1.50 | +1.61pp |
| CiteSeer | 20 | 56.41±2.31 | 53.83±2.75 | +2.58pp |
| PubMed | 20 | 72.48±0.95 | 71.90±0.88 | +0.57pp |
| CoauthorCS | 20 | 90.91±0.62 | 90.62±0.57 | +0.29pp |

预注册判定：

| 条件 | 结果 | 证据 |
|---|---:|---|
| C1 k-变异 | 0 | 四数据集 best-k 全为 20，max/min=1.00 |
| C2 k-敏感 | 0 | 全局 k*=20，四数据集相对各自 best-k 损失均为 0 |
| C3 球贴住 | 1 | 3/4 数据集成立；CiteSeer 差 +2.58pp，略超 best-k std 2.31pp |

R3 clean-val 选 k：

| Dataset | Val-selected k | Test | Test-best k | Loss vs best | Δ vs ball |
|---|---:|---:|---:|---:|---:|
| Cora | 20 | 57.20 | 20 | 0.00pp | +1.61pp |
| CiteSeer | 50 | 56.30 | 20 | 0.11pp | +2.47pp |
| PubMed | 100 | 72.27 | 20 | 0.20pp | +0.37pp |
| CoauthorCS | 20 | 90.91 | 20 | 0.00pp | +0.29pp |

#### 最终决策

- C1/C2 不成立：kNN 的 k 在本协议下不构成显著选择负担。强 K-free claim 关闭。
- C3 成立但只能支撑弱 claim：球式固定构造在 3/4 数据集贴住 best-kNN；CiteSeer 明确落后 best-kNN 约 2.58pp。
- R3 显示 clean val 扫 k 几乎能找到 test-best：因此论文中 K-free 只能写成"免扫描/固定配置的便利性"，不能写成 kNN 难以调优或跨数据集不可迁移。
- 骨架 v2 §6 已同步收窄：不再写"k 选择负担显著"，改写为"局部共识结构必要，但球/kNN/kmeans 形式可平替；球式实现的差异化只是固定构造、免扫描 profile"。
