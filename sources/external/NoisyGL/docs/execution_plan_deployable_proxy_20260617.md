# 执行计划：可部署 proxy 相关性验证（Phase 37d）

日期：2026-06-17
执行者：Codex
前置状态：Phase 37c 发现球增量与 **clean-label 诊断量**（true_ball_purity / orthogonal_AUC / ball_auc）强相关（Spearman 1.00, n=4-5）。**但这些量部署期拿不到**——true_purity 用 clean label，orthogonal_AUC/ball_auc 的 AUC 标签是"该 train label 是否真被污染"(= noisy≠clean)，也需 clean label。它们能**解释**球为何有用（机制发现，有效），但**不能做部署期触发器**。

**本计划补缺口**：球增量是否与**只依赖 H + noisy train label** 的 proxy 强相关？只有该相关成立，"想法2 自适应层"才有合法地基（否则用 clean-label 量做触发 = 破坏 label-free 设定）。**不改主方法机制。**

### claim 纪律（写进所有产出）

- 现阶段只能写 **"evidence suggests ball-clusterability mediates ball gains"**，**禁止**写 "we found the determining causal variable"。n=4-9 的 Spearman=1.00 是强信号不是定律，样本量限制必须主动声明。
- 区分两类变量，全程不得混淆：
  - **诊断量（clean-label 依赖,只能解释,不能部署）**：true_ball_purity、orthogonal_AUC、ball_auc、noise_harm、homophily
  - **可部署 proxy（只用 H + noisy train label）**：本计划要验证的

---

## 0. 环境与红线

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- 已有 H：`embeddings_v2/{Cora,CiteSeer,PubMed,AmazonPhoto,DBLP,CoauthorCS,amazoncom,CoauthorPhysics,WikiCS}/grace/H.pt`（9 数据集,Phase 37 已预训练）
- 复用已有诊断：`frozengrace_gb_gcn_diagnostics_v2.csv` 已记录 num_balls/mean_ball_size/eligible_balls/eligible_train_nodes/noisy_train_count 等
- percent split 0.2/0.1/0.7，uniform 0.4，seeds 0-4；clean label 只用于"被解释量(球增量)的事后核对"，不进 proxy 计算；研报告；测试全绿

---

## Phase 37d-pre：proxy 区分度前置检验（先做,防白跑,几分钟）

### 为什么先做这个

高噪声(40%)下有一个真实风险：**noisy purity 可能被噪声"洗平"**——每个球都混入约 40% 错标，所有球的 noisy purity 趋同，导致它**天生区分不开**成功/失败数据集，即使 true purity 能区分。若如此，整个 37d 注定失败，应提前止损。

### 检验

1. 对 9 数据集，算每个的 **noisy_train_purity_weighted**（球内 train noisy 标签多数占比，加权）
2. 看它和 **true_ball_purity**（已有）的相关 + 它本身在 9 数据集间的**方差/区分度**
3. 判定：
   - noisy_purity 在数据集间有明显差异（方差够大,和 true_purity 正相关）→ **有区分潜力,继续 37d**
   - noisy_purity 被洗平（9 数据集都挤在 ~0.4+1/C 附近,无区分度）→ **noisy purity 单独不可用**，37d 改为重点测其他 proxy（margin spread / disagreement），若全部无区分度则**自适应层关闭**，老实写边界

---

## Phase 37d-main：可部署 proxy 全量测量 + 相关性

### 抽取的 proxy（全部只用 H 几何 + noisy train label,部署期可得）

新脚本 `scripts/extract_deployable_proxies.py`（复用现有建球 + 球投票逻辑,seeds 0-2,9 数据集）：

| proxy | 定义 | 部署期可得? |
|---|---|---|
| noisy_train_purity | 球内 train noisy 多数占比(加权) | ✓ 只用 noisy |
| mean_vote_margin | 球内 (最高票−次高票)/总票 的均值 | ✓ |
| mean_vote_entropy | 球内 noisy label 分布熵的均值(低=纯) | ✓ |
| eligible_train_ratio | 有效球(train≥阈值)覆盖的 train 比例 | ✓ |
| num_balls / mean_train_per_ball | 已有 | ✓ |
| ball_gcn_disagree | warmup GCN([X‖H]) 预测 vs 球多数票的不一致率（**只用 noisy 训练的 warmup GCN + 球投票,不碰 clean**） | ✓ 关键候选 |

注意 ball_gcn_disagree：warmup GCN 用 noisy CE 训练(早停在记忆噪声前),它和球投票的分歧度是一个纯部署期信号——分歧高可能意味着球提供了 GCN 没有的信息。这是最有希望的 proxy。

### 相关性（被解释量用已有的真实球增量）

对每个 proxy,与四个真实球增量做 Spearman + Pearson：
- `softdist_xh − gcn_xh_ce`、`softdist_xh − shuffled`
- `gbselect − gcn_only`、`gbselect − shuffled`

样本 = 9 数据集（比 37c 的 4-5 扩了一倍）。

### 输出 `runs/deployable_proxy_20260617/`：proxy 全表 + 相关性表 + 报告

---

## Phase 37d-judge：自适应层可行性判定

| 37d 发现 | 自适应层(想法2)判定 |
|---|---|
| 某 deployable proxy 在 9 数据集上与球增量稳定强相关（Spearman≥0.7 且方向能分开成功/失败集） | **自适应层有合法地基** → 下一计划设计 λ=f(proxy)：高 proxy 开 SoftDist/GBSelect、低 proxy 退化 GCN[X‖H]、中间降权。**但需先扩到 12+ 数据集坐实相关,n=9 仍偏小** |
| 所有 deployable proxy 都弱相关/分不开 | **自适应层关闭**（触发信号不存在,强行做=cherry-picking）。论文写："ball-clusterability 中介球增量(由 clean-label 诊断量证据支持)，但缺乏可部署的 label-free 触发信号,自动适配留作 future work" |

### claim 输出（两种情况都要诚实）

- 即使 proxy 强相关,论文也只写 "a noisy-label-computable proxy [X] is predictive of ball gains, suggesting a path to self-adaptive deployment"，配 n=9 样本量声明 + 散点图
- 主机制 claim 始终是 "evidence suggests ball-clusterability mediates ball gains"（由 true_purity/orthogonal_AUC 的强相关支持,标注这些是诊断量）

---

## 执行顺序

```
37d-pre（noisy purity 区分度检验,防白跑,几分钟）
   ├─有区分潜力─► 37d-main（全 proxy 测量 + 相关性,9 数据集）
   │                └─► 37d-judge（自适应层可行性 + claim）
   └─洗平无区分──► 重点测 disagreement proxy;全失败则关闭自适应层
```

- 37d-pre 先跑,几分钟,决定值不值得做 main
- 37d-main 复用已有 H 和建球逻辑,主要是补 proxy 抽取 + 相关分析,不重训
- 全程 clean label 只用于"球增量"这个被解释量(它本来就是 test acc 差,合法),proxy 侧绝不碰 clean

## 完成后

- 论文 §8：新增"可部署 proxy vs 球增量"相关性分析（与 §37c 的 clean-label 诊断量相关性并列,明确区分两类）
- 明确给出自适应层结论：有地基→列为下一步/future work 并说明需扩样本；无地基→诚实写 future work
- 更新经验文档与记忆。claim 全部按"mediates / evidence suggests"级别,标注 n 限制
