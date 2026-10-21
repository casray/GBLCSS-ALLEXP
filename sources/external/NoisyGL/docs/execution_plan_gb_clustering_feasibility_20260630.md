# 执行计划：免K粒球图节点聚类 可行性最小验证（Phase 41 · 新主线 go/no-go 闸门）

日期：2026-06-30
执行者：Codex
前置状态：SoftDist 分类线全部收口为负——粒球增量与下游强度互斥（GCN[X‖H] +1.1pp / MLP(X) +6~9pp 但绝对精度最低），"中等分类器+轻量"卖点两头不占（精度不如强GCN、轻量不如裸X-GCN，且GRACE+建球开销在前端省不掉）。检测换题（Phase 39）、跨视角（Phase 40）均判负。**结论：放弃在分类/检测上救粒球，转新主线——图节点聚类。**

**调研已确认的机会**：(1)"自监督图表示(GRACE H) + 粒球作为聚类引擎"是真空白（最接近的 AGGR 用粒球做图重连、真聚类仍是GNN；MGBCC把Cora当多视图特征、球仍用kmeans切需K）；(2) **"免K"（不预设簇数）是深度图聚类综述点名的真痛点**，该赛道现有占位者（RGC/GCMA）标杆只是"逼近参数化方法"而非超越——**这把粒球从"必须赢SOTA精度"的死局里解放出来**：标杆变成"免K还能逼近被喂了真K的方法"，正好对齐粒球"赢不了精度"的客观短板；(3) 聚类领域认"免K/效率"为一等卖点（分类领域不认，这正是SoftDist死因）。

**本计划是新主线的 go/no-go 闸门：用最低成本确定"GRACE H + 免K粒球聚类"能否立一篇，而非赌。纯离线，不训练任何模型。** 过→正式投入、写完整方法+实验计划；不过→图聚类也关闭，回去把SoftDist分类按现状诚实收口。

---

## 0. 唯一要回答的问题（判据已锁死）

> 在 GRACE H 上构造粒球、再把球凝聚成簇——**在不告诉它真实簇数K的前提下**——能否 (a) 自动定出接近真K的簇数，(b) 聚类精度(ACC/NMI/ARI)逼近"被喂了真K的k-means/谱聚类"，且 (c) 当K被误设时显著比k-means更鲁棒？

**第一判据是"免K下的精度逼近 + 误设K鲁棒性"，不是"精度超过深度SOTA"。** 因为赢深度SOTA做不到（GRACE表示为分类优化、且领域极卷），而"免K+抗K误设"是k-means/深度聚类给不了、且综述认可的差异化。

---

## 0.1 红线

1. **clean 标签（真实类别）只进评测**：算 ACC/NMI/ARI 与"簇数 vs 真K"。**绝不进入建球或球→簇凝聚的任何决策**（聚类是无监督的，用了标签就作弊）。
2. **不改 SoftDist anchor、不动已注册 predictor、不训练任何下游**。本计划只新增一个离线聚类诊断脚本。
3. **seed0 单数据集先 smoke**，通过再多数据集 × 多 seeds。
4. 涉及"标签是否泄漏进聚类决策"的逻辑配单测；测试全绿；research-record 报告。

## 0.2 环境与复用

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- 数据集：`cora,citeseer,pubmed`（聚类标准benchmark；先 cora smoke）。**注意：聚类用全图节点，不做 noise injection，不做 train/val/test split**——所有节点参与聚类，真实类别作评测。
- H：复用 `load_softdist_xh_gb_config(dataset_key).embedding_path` + `load_frozen_embeddings`（与SoftDist同一份冻结H）。pubmed若无H则跳过并记录。
- **新脚本 `scripts/diagnose_gb_clustering.py`**。可直接复用：
  - `_load_dataset`（给 `data.feats`/`data.labels`/`data.adj`/`data.n_classes`）——聚类只需 `labels`(评测)、`adj`(谱聚类baseline与可选球凝聚)、H
  - `load_frozen_embeddings` → H
  - **`build_ordinary_granular_balls(H, max_balls, min_ball_size, split_improvement, random_state)`**——返回 `assignment`(每节点所属球) + `summary["num_balls"]`。球数由 `split_improvement` 自适应停止，**这是免K的天然来源**
  - `KMeans`、`from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score`、`from scipy.optimize import linear_sum_assignment`(写hungarian-ACC)、`SpectralClustering`
- 固定超参不搜：建球用 SoftDist 同款 H 参数（cora/citeseer 64/32/0.05，pubmed 256/32/0.005）作起点；`split_improvement` 作为"免K粒度旋钮"扫 3 档（0.02/0.05/0.1）报敏感性，但**主结果用单一预注册档 0.05**，不后验挑。

---

## 1. 方法（本计划的核心新机制：球→簇凝聚）

粒球数（几十~几百）≫ 真类数K（7/6/3），所以**球不能直接当簇**，必须把球凝聚成 ~K_auto 个簇。这一步是方法主体，也是与 AGGR（球做重连）/SAGBC（球做加速）的切割点：**粒球在这里是聚类引擎本身**。

```
Step 1  在 H 上 build_ordinary_granular_balls → 得到 B 个球（B 自适应，非预设）
Step 2  以球为节点构造"球图"：
        - 球间相似度 = 球中心(H均值)的余弦相似 + 可选图边密度（B_i 与 B_j 间原图边数 / 规模）
Step 3  在球图上做无参凝聚得到簇（免K关键）：
        - 主方案 agglomerative：按球间相似度合并，停止准则用 modularity 极大 或 相似度gap（自动定簇数 K_auto）
        - 备选：连通分量（相似度阈值切断）
Step 4  把球的簇标签广播回节点 → 每节点一个簇标签
```

- **K_auto 完全由数据决定，全程不接触真实类数。** 这是免K卖点的落地。
- 凝聚停止准则（modularity / gap）预注册一个为主，另一个作敏感性。

## 2. Baseline（必须含"被喂真K"的强对照，证明免K没掉太多）

| baseline | 是否被喂真K | 角色 |
|---|---|---|
| `kmeans_H_trueK` | **是**（n_clusters=真K） | **主对照**：嵌入直接kmeans，深度聚类标准下游。免K球聚类要逼近它 |
| `spectral_H_trueK` | 是 | 谱聚类对照 |
| `kmeans_H_autoK` | 否（用球定出的K_auto喂kmeans） | 拆解：K_auto好 vs 凝聚机制好，哪个在起作用 |
| `gb_cluster`（主） | **否** | 本方法：免K球凝聚 |
| 深度SOTA参考值 | 是 | 从调研取 HSAN/CCGC 在 Cora/CiteSeer/Pubmed 的公开 ACC/NMI 填表（不重跑，仅作"我们离SOTA多远"的诚实参照） |

## 3. 评测指标（每方法 × dataset × seed）

- **ACC**（hungarian 匹配后）、**NMI**、**ARI**——聚类三标配
- **K_auto vs 真K**：本方法自动定出的簇数与真实类数的差距（免K卖点的核心证据）
- 球→簇前的球数 B、凝聚后 K_auto

## 4. ★第一判据：免K精度 + 误设K鲁棒性（决定新主线生死）

**4a 免K精度逼近**：`gb_cluster`(免K) 的 ACC/NMI 与 `kmeans_H_trueK`(喂真K) 的差距。

**4b 误设K鲁棒性（最值钱、k-means给不了的）**：
- 给 k-means 喂**错误的K**：真K±2、±50%（如 Cora 真K=7，喂 5/7/9/3/11）
- 本方法不需要喂K（K_auto固定）
- 比较：当K被误设时，k-means 的 ACC/NMI 崩多少，本方法（不受K误设影响）相对优势多大
- **这是论文主图**：x=喂给kmeans的K，y=ACC，画 kmeans 曲线（误设即崩）vs 本方法水平线

## 5. go/no-go 闸门（预注册，按优先级）

均按 seed 平均 + 多数据集计数。

| 闸门 | 判据 | 不过的后果 |
|---|---|---|
| **G1 免K精度逼近（决定性）** | `gb_cluster`(免K) 的 NMI ≥ `kmeans_H_trueK`(喂真K) 的 **90%**，且 ACC 差距 ≤5pp，在 **≥2/3 数据集**成立 | 免K掉太多，卖点立不住 → **图聚类关闭**，回SoftDist分类诚实收口 |
| **G2 误设K鲁棒性（最值钱）** | 在 K 被误设(±2 或±50%)时，本方法相对 k-means 的 ACC 优势 mean ≥5pp，≥2/3数据集 | "免K"无实际鲁棒收益，卖点空心 → 需重新定位或关闭 |
| G3 K_auto 合理 | K_auto 与真K的比值落在 [0.5, 2.0]，不退化（不是1个巨簇、不是几百个碎簇） | 凝聚机制坏，需修Step3停止准则 |
| G4 凝聚非平凡 | `gb_cluster` 明显优于 `kmeans_H_autoK`（同样K_auto），证明"球凝聚"本身有贡献，不只是K_auto好 | 凝聚无独立价值，方法退化为"自动定K+kmeans"，新意不足 |

**决策矩阵**：
- **G1+G2 过**（G3/G4 至少不崩）→ **图聚类成为新主线**。规划完整方法（球图凝聚细化）+ 主表（vs 喂真K的kmeans/谱/深度SOTA参考）+ 误设K鲁棒主图 + 效率对照；目标 IEEE TKDE / Information Sciences / Pattern Recognition / IJCAI / 中文核心；related work 与 AGGR/AW-GBGAE/MGBCC 清晰切割。
- **G1 不过** → 免K粒球聚类也立不住，**图聚类关闭**。回去把 SoftDist[X‖H] 分类按现状（受控增量+适用边界）诚实收口写完，不再开新方向。
- G1过G2不过 → 免K成立但鲁棒卖点空，降级为"效率/可解释"卖点，重新评估是否够一篇。

## 6. 执行顺序

```
1. 新建 scripts/diagnose_gb_clustering.py
   - 加载 H（SoftDist同款）+ data.labels(仅评测)/adj
   - 实现 §1 球→簇凝聚（modularity停止为主）
   - 实现 §2 全部baseline（kmeans/spectral喂真K、kmeans喂K_auto）
   - 实现 §3 指标（hungarian-ACC/NMI/ARI/K_auto）
   - 实现 §4b 误设K扫描
   单测：篡改 data.labels → 建球与凝聚结果不变（标签不进聚类决策）；
        K_auto 不等于真K时流程不崩；hungarian-ACC 在完美聚类时=1.0
2. seed0 smoke（cora）：不崩、K_auto 合理(非1非数百)、gb_cluster NMI 与 kmeans_trueK 同量级、labels不进决策
   └─ 异常 → 停，修
3. cora,citeseer,pubmed × seeds 0-4 → 按 §5 判 G1-G4
4. 写报告 + go/no-go 结论
```

- 输出目录：`runs/gb_clustering_feasibility_20260630/`
- 报告：`runs/gb_clustering_feasibility_20260630/gb_clustering_report.md`（每方法×dataset 的 ACC/NMI/ARI + K_auto vs 真K + §4b 误设K鲁棒表 + §5 闸门判定 + 明确 go/no-go）

## 7. 完成后

- 更新经验文档（新增 Phase 41）、记忆 `gb-direction-turning-point-2026-06.md`、新窗口总结。
- **若 go**：起草图聚类完整方法与实验计划（球图凝聚细化、效率对照、与AGGR/MGBCC切割的related work），我（Claude）据此推进新主线写作。
- **若 no-go**：回 SoftDist[X‖H] 分类诚实收口，按已有 5-seed/shuffled/同H控制/适用边界 完成正文。

## 给导师的话术（诊断跑完后按结果二选一）

> 我把粒球在分类上"被强表示吸收"的根因想清楚了：粒球增量与下游强度互斥，在分类上救不回来。但这个根因在**聚类**任务上不存在——聚类无下游分类器、无监督，且粒球聚类有一个k-means给不了的卖点：**免K（不预设簇数）+ 误设K鲁棒**。我用现成的 GRACE 表示做了个最小验证，关键看免K时精度能否逼近被喂了真K的k-means、以及K误设时是否更稳。[过→把主线正式转为免K粒球图聚类，这是我们组主场且填补"自监督表示+粒球聚类"空白；不过→回去把已收口的SoftDist分类论文按适用边界写完。]

---

## Phase 41 执行结果备案（2026-06-30 已回填）

结果文件：`runs/gb_clustering_feasibility_20260630/`

### 执行范围

- 脚本：`scripts/diagnose_gb_clustering.py`
- 测试：`tests/test_gb_clustering.py`
- 验证：`py -3 -m pytest tests/test_gb_clustering.py -q` -> `4 passed`
- 数据集：`cora,citeseer,pubmed`
- seeds：`0,1,2,3,4`
- split sensitivity：`0.02,0.05,0.1`
- 主判定 split：`0.05`
- 聚类使用全图节点；真实标签只用于 ACC/NMI/ARI 评测，不进入构球、球图或凝聚决策。

### Gate 结果

| Dataset | true K | K_auto | balls | GB ACC | kmeans trueK ACC | ACC gap | GB NMI | kmeans trueK NMI | NMI ratio | wrong-K ACC adv | G1 | G2 | G3 | G4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 7 | 3.00 | 52.40 | 0.4823 | 0.6317 | -0.1493 | 0.3404 | 0.5193 | 0.6555 | -0.1208 | 0 | 0 | 0 | 0 |
| CiteSeer | 6 | 4.40 | 63.80 | 0.5342 | 0.6395 | -0.1053 | 0.2942 | 0.3782 | 0.7779 | +0.0240 | 0 | 0 | 1 | 0 |
| PubMed | 3 | 3.40 | 64.00 | 0.5248 | 0.6766 | -0.1518 | 0.1918 | 0.3107 | 0.6172 | +0.0096 | 0 | 0 | 1 | 0 |

### 判定

- **G1 免K精度逼近失败**：三个数据集均未达到 `NMI >= 90% * kmeans_H_trueK` 且 `ACC gap >= -5pp`。ACC 差距约 `-10.5pp` 到 `-15.2pp`，NMI ratio 仅 `0.62~0.78`。
- **G2 误设K鲁棒失败**：Cora 本方法比误设K kmeans 还低 `-12.08pp`；CiteSeer 只有 `+2.40pp`；PubMed 只有 `+0.96pp`，均未达到 `+5pp`。
- **G3 部分通过但不够**：CiteSeer/PubMed 的 K_auto 比值合理；Cora 自动凝聚为 `3` 类，低于真 `7` 类的一半。
- **G4 全失败**：`gb_cluster` 没有明显优于 `kmeans_H_autoK`，说明当前球凝聚不是独立有效机制，方法退化甚至弱于“自动定K后再 kmeans”。

### 最终 go/no-go

**NO-GO。** 免K粒球图节点聚类不能作为新主线。当前 `GRACE H + ordinary balls + ball-graph modularity agglomeration` 既不能逼近被喂真K的 kmeans，也没有形成误设K鲁棒优势；因此“免K + K误设鲁棒”卖点不成立。

后续不建议继续在该方向追加调参式改造。若要写论文，应回到 SoftDist 分类线，以“受控粒球增量 + 适用性边界 + 多方向负结果”诚实收口。
