# 执行计划：收口型补证据矩阵（Phase 38）

日期：2026-06-17
执行者：Codex
定位：**收口型补证据,不是发明新机制。** 论文主线已成立（见下），Phase 38 只把 §8 的"可部署 proxy / 适用域预测"做扎实,给论文加分,**不决定论文能不能写**。Phase 37d 因 n≤6 underpowered 无法判定 adaptive layer；本计划补齐证据矩阵到 n≥10 再判定,跑不出就把 adaptive trigger 诚实降级为 future work。**不改主方法机制。**

### 论文主线（已成立,不押在 Phase 38 上）

1. GRACE 表示双重利用：`[X‖H]` 分类 + 粒球 soft target
2. 粒球受控增量在 Cora/CiteSeer/Coauthor-CS 上成立（vs gcn_xh_ce 与 shuffled,5/5）
3. 失败边界由 **ball-clusterability** 解释（WikiCS/DBLP/PubMed/Amazon 系,由 clean-label 诊断量证据支持）
4. deployable proxy 目前是 **preliminary evidence / future adaptive trigger**,不是已确立的部署律

---

## 0. 环境与红线

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- 已有 H：`embeddings_v2/{Cora,CiteSeer,PubMed,AmazonPhoto,DBLP,CoauthorCS,amazoncom,WikiCS}/grace/H.pt`（8 个,Physics 缺）
- percent split 0.2/0.1/0.7，uniform 0.4，seeds 0-4；anchor 机制不动；clean 只进诊断/被解释量；proxy 侧只用 H+noisy；研报告；测试全绿

### ★预注册纪律（防 formula shopping / cherry-picking,执行中不得违反）

1. **冻结 3 个 proxy 候选,跑完只报这 3 个**——即使 37d 里别的 proxy 相关性更高也不换：
   - `mean_vote_margin_weighted`
   - `mean_vote_entropy_weighted`
   - `ball_gcn_disagree_confident_train`
2. **n≥10 才谈 adaptive layer**；n<10 只写机制证据,不实现自适应
3. **不为凑 n 硬加勉强数据集**——够则判定,不够则如实降级 proxy 为 future work
4. claim 上限：只写 "evidence suggests ball-clusterability mediates ball gains"；每处标注 n 限制；禁止 "found the determining/causal variable"

---

## Phase 38a：补齐当前证据矩阵的窟窿（核心,补洞不加机制）

### 要补的（按 ROI 排序）

1. **gbselect 在 Cora/CiteSeer**：37d 显示这两个 gbselect target delta 缺（missing_config）。加 config + 跑 `gbselect`/`gbselect_gcn_only`/`gbselect_shuffled` × 5 seeds × uniform 0.4 → 补上 gbselect 增量
2. **扫现有 8 数据集所有 NA target delta**：对照 37c/37d 表,哪些 (数据集 × {softdist 增量, gbselect 增量}) 仍是 NA,优先补齐。这些是已有 H、只差跑下游的,成本低
3. **Coauthor-Physics H（带硬时间上限）**：34.5k 大图,GRACE 预训练设**硬上限(如 30 分钟单次)**,OOM/超时 → 记 `missing_h`,**不被它拖死**,继续后续。跑出则补它的下游 + proxy

### 输出 `runs/evidence_matrix_20260617/`：补全后的 9（或 8）数据集 × {softdist, gbselect} 增量全表

---

## Phase 38b：3 个冻结 proxy 的相关性（只报这 3 个）

### 计算

对补全后的全部数据集,抽 3 个预注册 proxy（只用 H+noisy,复用 37d 的 `extract_deployable_proxies.py`）,与 4 个真实球增量做 Spearman+Pearson：
- `softdist_xh − gcn_xh_ce`、`softdist_xh − shuffled`
- `gbselect − gcn_only`、`gbselect − shuffled`

### 报告必须包含

- 每个 proxy × 每个 target 的 (Spearman, Pearson, **n**)
- **每个相关性显式标注 n**;n<7 的相关性标"underpowered,仅供参考"
- 散点图（proxy vs 球增量,标数据集名）——比相关系数更能看清是否真单调还是个别点拉动

### 输出 `runs/evidence_matrix_20260617/frozen_proxy_correlations.csv` + 报告

---

## Phase 38c：判定（按预注册门槛,不松动）

| 结果 | 判定 |
|---|---|
| n≥10 **且** 3 个 proxy 里至少 1 个对球增量稳定相关（Spearman≥0.7,方向能分开成功/失败集,散点非单点拉动） | adaptive layer 有合法地基 → 做 **38d 极简二值开关** |
| n≥10 但 3 个 proxy 都弱/不稳 | adaptive 关闭。论文写"tested deployable proxies, none robustly predictive at n=X → adaptive trigger left as future work" |
| n<10（如 Physics 拿不到 + 部分 delta 补不上） | **不判定 adaptive**,proxy 作为 preliminary evidence 写入 §8 + future work,标注样本量不足 |

---

## Phase 38d：极简二值 adaptive switch（仅当 38c 过门,n≥10）

### 严格约束（防止又变成新机制）

- **只做开关,不做连续加权**：proxy ≥ 阈值 → 启用 ball method（SoftDist 或 GBSelect）；proxy < 阈值 → 退化为 `GCN[X‖H]`
- 阈值**用 38b 的 proxy 分布定,不看 test 调**（如取成功/失败数据集 proxy 的中点,预注册）
- 验证：在全部数据集上跑这个 switch,看它是否在成功集开球、失败集退化,且整体不劣于"全程开球"和"全程不开"——证明它**安全**（该开开、该退退）,而非证明它涨点（Phase 32/34 已知失败集球≈GCN[X‖H],所以开关价值是安全性/自动化,不是性能）
- 输出 `runs/evidence_matrix_20260617/adaptive_switch/`

---

## 执行顺序

```
38a 补洞（gbselect Cora/CiteSeer → 扫 NA delta → Physics H 带时限）
   └─► 38b 3 冻结 proxy 相关性（标 n,出散点）
         └─► 38c 判定
               ├─ n≥10 且强相关 ─► 38d 极简二值开关
               └─ 否则 ───────────► proxy 降级 future work
```

- 38a 先补低成本的（gbselect config、NA delta）,Physics 带硬时限并行/最后,不阻塞
- 38b/38c 纯分析,数据齐后即出
- 全程 clean 只用于"球增量"被解释量,proxy 侧绝不碰 clean

## 完成后（无论 n 是否够）

- 论文 §8：加"可部署 proxy vs 球增量"相关性（标 n,与 §37c 的 clean-label 诊断量相关性并列,明确两类区分）
- adaptive：n≥10 且过门 → 写成"a simple proxy-gated switch safely enables/disables the ball module"（强调安全性不是涨点）+ n 声明；否则 → future work
- 主线四点（见顶部）保持不变,Phase 38 只作为 §8 的 applicability/predictability 补强
- 更新经验文档与记忆。此后实验收口,进入正文写作
