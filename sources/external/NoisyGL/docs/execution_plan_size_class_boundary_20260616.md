# 执行计划：规模×类数二维边界诊断（Phase 35）

日期：2026-06-16
执行者：Codex
前置状态：SoftDist [X||H] 球增量只在 Cora(2.7k,7类)/CiteSeer(3.3k,6类) 出现，在 PubMed(19.7k,3类)/Amazon-Photo(7.6k,8类)/DBLP(17.7k,4类) 失败（soft-target 被吸收 Phase32 + selection 被样本损失抵消 Phase34）。

**核心问题**：失败的真正变量是「图规模大」还是「类别少」？现有 5 个数据集**全挤在对角线**（大图恰好都类少、小图恰好都类多），size 与 class-count **完全混淆**，无法区分。Phase 34 归因发现三个失败数据集的共同点其实是**类少 + 噪声伤害本就轻**（PubMed 3类 GCN 单视角已够好；Amazon 同质强基线 0.905 噪声没伤到）。

**本计划用「规模 × 类数」二维网格补两个关键空白象限,判别真变量**：
- Q2 大规模(>15k) + 类多(≥10)：**最关键的决胜点**
- Q1 小规模(<5k) + 类少(≤4)：补另一角

### 现有数据落点 + 空白象限

```
              类少(≤4)            类多(≥6)
小(<5k)     [Q1 空白,需构造]     Cora(2.7k,7)✅ CiteSeer(3.3k,6)✅
中(5-10k)   —                    Amazon-Photo(7.6k,8)❌
大(>15k)    PubMed(19.7k,3)❌     [Q2 空白,需补]
            DBLP(17.7k,4)弱
```

**决胜逻辑**：Coauthor-CS(18.3k, 15类) 与 DBLP(17.7k, 4类) **规模几乎相同,只差类数**。
- 若 CS 球有增量、DBLP 没有 → **类数是真变量,规模不是**,大图大门重开
- 若 CS 也无增量 → 规模(或规模主导)是真变量,边界确证为"小图"
- 这是一个**判别性实验**,直接拆开混淆变量

---

## 0. 环境与红线（沿用）

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- GRACE 预训练在 `gb_graph_aware_minival_6_1/`，导出 `checkpoints/embeddings_v2/<Name>/grace/H.pt`,节点序对齐 NoisyGL dataloader
- NoisyGL 显式 percent split 0.2/0.1/0.7；红线：anchor 机制不动、clean 只进诊断、SSL label-free、超参官方默认不调、研报告、测试全绿
- loader 已接好 `coauthorcs`→`Coauthor('CS')`、`amazoncom`→`Amazon('Computers')`(确认无需改代码,只加 config + 预训练)

---

## Phase 35a：Q2 决胜点 Coauthor-CS（先做,大规模+多类）

### 为什么先做这个

它是判别 size vs class-count 的唯一决胜点（与 DBLP 同规模、只差类数），且 PyG 原生/loader 已接好/成本与 DBLP 同档（18k 节点）。

### 步骤

1. 触发下载 `coauthorcs`（PyG `Coauthor('CS')`,首次联网）
2. GRACE label-free 预训练 → `embeddings_v2/CoauthorCS/grace/H.pt`。**预注册：用 DBLP/PubMed 同档 GRACE 配置**（18k 节点同量级），执行前写进报告。注意 CS 有 6805 维特征,GRACE 输入层更宽,监控显存
3. H-图对齐 sanity check（同 Phase 28 逻辑）
4. **球诊断（先,廉价）**：用默认 + Phase32a 大图适配参数,报 num_balls / true_purity / mean_train_per_ball / **球在 GCN-clean50 的条件正交 AUC**（复用 Phase 33 脚本）
5. **下游 5 seeds × uniform 0.4**：两条线都跑——
   - SoftDist 线：`softdist_xh` / `softdist_xh_shuffled` / `gcn_xh_ce` / `nrgnn`（球 soft-target 增量）
   - 若球诊断正交 AUC 高,加 GBSelect 线：`gbselect` / `gbselect_gcn_only` / `gbselect_shuffled`（命门对照,沿用 Phase34 标准）
6. 输出 `runs/coauthorcs_20260616/`

### 判定（决胜）

| Coauthor-CS 结果 | 对边界假设的结论 |
|---|---|
| `softdist_xh − gcn_xh_ce` ≥0 且 real>shuffled（或 GBSelect 命门过） | **类数是真变量!** 大图+多类球有效,边界重定义为"类数充足",大图大门重开 → Phase 35c 扩展 |
| 球无增量（同 DBLP/PubMed） | 规模是真（或主导）变量,边界确证"小图"。CS 作为"大+多类仍失败"的关键证据,排除 class-count 解释 |

---

## Phase 35b：Q1 小规模+类少（构造 Cora-3c/4c）

### 目的

补另一空白角：若小图但类少,球还有没有增量?与 Q2 一起把二维网格填满,确认边界形状。

### 构造（方法论上合理,需文档化）

- NoisyGL loader 加一个分支：加载 Cora（Planetoid 路径,无需下载）→ 保留**最大的 3 个类**（或 4 个）→ 取诱导子图的**最大连通分量(LCC)** → relabel。命名 `cora3c`（~1.5-2k 节点, 3 类）
- 文档化为"top-k 类 + LCC"的标准 ablation 构造,诚实标注是派生数据集
- 为 cora3c 预训练 GRACE H（小图,成本极低）

### 运行

- 球诊断 + 下游 5 seeds × uniform 0.4：`softdist_xh` / `softdist_xh_shuffled` / `gcn_xh_ce` / `nrgnn`
- 输出 `runs/cora3c_20260616/`

### 判定（填网格）

| cora3c(小+类少) 结果 | 结论 |
|---|---|
| 球有增量 | 边界主要由**规模**决定（小图就行,不论类数）→ 与 Q2 结果交叉确认 |
| 球无增量 | 边界由**类数**决定（类少就不行,不论规模）→ 与 Q2 结果交叉确认 |

### 二维交叉判定表（35a + 35b 合起来读,这是本计划的最终产物）

| | 类少 | 类多 |
|---|---|---|
| **小图** | cora3c（Q1,35b）| Cora/CiteSeer ✅ |
| **大图** | DBLP/PubMed ❌ | Coauthor-CS（Q2,35a）|

- Q1✅ + Q2❌ → **规模主导**（小图行大图不行,与类数无关）
- Q1❌ + Q2✅ → **类数主导**（类多行类少不行,与规模无关）
- Q1❌ + Q2❌ → 只有"小+多类"那一角行 → **二维联合边界**（你的猜想正确,size 和 class 共同约束）
- Q1✅ + Q2✅ → 边界不是 size/class,是别的（需重新找变量）

---

## Phase 35c（仅当 35a Coauthor-CS 球有效）：扩展确认

- 加 Amazon-Computers(13.7k,10类) 作 Q2 第二点（loader 已接好）,确认"大+多类有效"不是 CS 单数据集偶然
- 若两个都有效 → 重写论文：边界是"类数充足"而非"小图",数据集广度解锁(Cora/CiteSeer/Coauthor-CS/Amazon-Computers),格局完全不同
- 噪声谱补充：在生效的大+多类数据集上补 pair/instance 0.4

---

## 执行顺序

```
35a Coauthor-CS（决胜点,先做）
   ├─球有效─► 类数是变量 → 35c 扩展(Amazon-Computers) → 可能解锁大图
   └─球无效─► 35b 验证规模 vs 类数
35b cora3c（小+类少,填 Q1,与 35a 交叉读二维表）
```

- **35a 是判别实验的核心,先做**——它单独就能区分"大图必败"还是"只是之前的大图类少"
- 35b 填另一角,两者交叉给出完整二维边界
- 每步先球诊断(廉价)再下游;球诊断若直接显示塌缩/低正交,可省下游

## 无论结果都有重大论文价值

- **类数主导(Q2✅)** → 边界从"小图"修正为"类数充足",可能解锁 2-4 个新数据集,论文格局升级,且是一个**有趣的科学发现**（粒球去噪需要足够类数让 GCN 单视角搞不定）
- **二维联合边界(都❌)** → 论文给出 SoftDist 适用域的**精确二维刻画**（小规模 × 多类别），这是教科书级的 applicability characterization,比"只在 2 个数据集有效"强得多——它解释了 why
- 任何结果都把当前"为什么只有 Cora/CiteSeer"从模糊观察变成可测的、有网格证据的边界定律

## 完成后

回填论文初稿 §8（泛化/边界）：新增"规模×类数"二维边界表 + 散点图（x=节点数, y=类数, 点色=球增量正负）。更新经验文档与记忆。

---

## Phase 35 执行结果备案（2026-06-16）

结果报告：`runs/size_class_boundary_20260616/phase35_size_class_boundary_report.md`。

### 35a Coauthor-CS：Q2 决胜点成功

- GRACE H 已预训练：`../../gb_graph_aware_minival_6_1/checkpoints/embeddings_v2/CoauthorCS/grace/H.pt`，shape = 18333 × 256。
- 使用 DBLP GRACE config fallback，metadata 已记录。
- 诊断强阳性：`ball_auc_in_gcn_clean50 = 0.8533`，true weighted purity = 0.9052。
- SoftDist [X||H]：0.9243 ± 0.0016。
- GCN [X||H]：0.9190 ± 0.0009。
- Shuffled：0.9188 ± 0.0012。
- SoftDist - GCN [X||H] = +0.53pp，5/5 seeds；SoftDist - shuffled = +0.55pp，5/5 seeds。
- GBSelect 也成功：GBSelect - GCN-only = +0.71pp，4/5 seeds；GBSelect - shuffled = +2.16pp，5/5 seeds；GBSelect - GCN [X||H] = +0.67pp，5/5 seeds。

判定：**“大图必败”被 Coauthor-CS 推翻。类数/类别复杂度是有效调节变量。**

### 35c Amazon-Computers：扩展确认给出边界

- GRACE H 已预训练：`../../gb_graph_aware_minival_6_1/checkpoints/embeddings_v2/amazoncom/grace/H.pt`，shape = 13752 × 256。
- 使用 PubMed GRACE config fallback，与 Amazon-Photo 保持一致。
- 诊断强阳性：`ball_auc_in_gcn_clean50 = 0.9219`，true weighted purity = 0.7850。
- SoftDist [X||H]：0.8625 ± 0.0056。
- GCN [X||H]：0.8695 ± 0.0082。
- Shuffled：0.8443 ± 0.0176。
- SoftDist - GCN [X||H] = -0.70pp，0/5 seeds；SoftDist - shuffled = +1.82pp，5/5 seeds。
- GBSelect - GCN-only = +2.55pp，5/5 seeds；GBSelect - shuffled = +4.82pp，5/5 seeds；但 GBSelect - GCN [X||H] = -15.17pp，0/5 seeds。

判定：Amazon-Computers 支持“真实球结构有信号”，但不支持“球方法超过全样本 GCN [X||H]”。类数不是唯一充分条件。

### 最终决策

- 当前论文成功数据集从 Cora/CiteSeer 扩展为 Cora/CiteSeer/Coauthor-CS。
- 不能 claim “大图+多类必然成功”；更准确 claim 是：**规模不是单独死因，类别复杂度/标签噪声可分性是关键调节变量，但图 regime 和强 XH 分类器仍会吸收球信号。**
- `cora3c` 暂缓：Q2 已成功拆开 size/class 混淆，cora3c 的边际价值下降；后续若需要完整二维附录再做。
- 若 Coauthor-CS 进入主表，需要补 NRGNN 等 baseline；若只进入边界表，当前对照已经足够支撑 boundary claim。
