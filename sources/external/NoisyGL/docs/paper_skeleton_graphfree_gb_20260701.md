# 论文骨架：Graph-Free 粒球软监督的标签噪声鲁棒节点分类

日期：2026-07-01
目标：普通 SCI（英文）。本文档 = 逐节 claim→证据 映射 + 诚实定位 + 表图清单 + related-work 切割。**不是终稿，是写作前的地图，防止跑偏/过度声称。**

---

## 0. 一句话贡献（诚实、已被实验坐实）

> 在 **graph-free 节点分类**设定下（训练期借对比表征吸收结构、**推理只用 X-only 轻量 MLP、不碰邻接**），用**冻结 GRACE 表征上的粒球软标签（β=0.5 半硬）**做标签噪声鲁棒训练；它在同一弱 MLP backbone 下稳定优于一批（含最新）抗噪策略、且依赖真实球结构（>shuffled）、在**低标签率**下增益最大。**首个把粒球标签噪声鲁棒带入 graph-free 节点分类的工作。**

**不 claim**：全数据集 SOTA / 赢强 GNN 绝对精度 / 同等 H-access 下赢经典策略 / 软标签一定最优。

---

## 1. 标题 & 卖点排序（按证据强度）

- 候选标题：*Granular-Ball Soft Supervision for Label-Noise-Robust Graph-Free Node Classification*
- 卖点排序（写作时按此权重分配篇幅）：
  1. **球必要性**（最硬）：β0.5 > shuffled 全线 5/5。
  2. **X-only 设定内全胜抗噪策略**（含最新 FINE/DivideMix-lite/CT adapted）。
  3. **低标签率优势**（5-seed 曲线，Cora/CiteSeer）。
  4. （弱 profile，不当 headline）graph-free 推理对结构扰动 0.00pp；消息传递 GNN 会掉（MLP-2 C2）；轻量。

---

## 2. Introduction —— 每段一句 claim

- **P1 问题**：图节点分类现实里同时面临 (a) 标签噪声（错标注）(b) 部署想要 graph-free 推理（冷启动/孤立新节点、效率、隐私不传拓扑）。
- **P2 gap**：GNN 抗噪方法（NRGNN/TFR/ProCon…）推理**依赖图**；graph-free MLP 方法（Graph-MLP/GLNN/LinkDist）**不处理标签噪声**。两者交集是空白。
- **P3 我们的做法**：GRACE 对比表征（训练期吸收结构）+ 粒球软标签（抗标签噪声）+ 轻量 X-only MLP（graph-free 推理）。粒球软标签**必要**因为纯 MLP 表达弱、需要额外结构化监督。
- **P4 贡献列点**：①首个 graph-free 设定下的粒球标签噪声鲁棒方法；②同 backbone 下稳定优于经典+最新抗噪策略（Cora/CiteSeer）、且证明依赖真实球结构；③低标签率优势；④**诚实刻画适用域**（依赖可聚类 H；PubMed 边界；H-input 强 baseline 的 scope 说明）。
- ⚠️ **诚实调门**：Intro 末尾一句限定 scope——"本文聚焦 graph-free 推理场景；当推理期可用图表征时，[X‖H]+强选择是有力替代，我们在 §X 诚实对照。"

---

## 3. Related Work —— 四块 + 每块切割句

| 块 | 代表 | 与本文差异（切割句） |
|---|---|---|
| 通用 LNL | small-loss/co-teaching/JoCoR/ELR/SCE/FINE/DivideMix/**CT** | 图无关；我们在图节点 + graph-free 设定，且用球结构化软监督 |
| 图 LNL | NRGNN/**TFR**(CIKM25)/**ProCon**(IJCAI25)/**ICGNN**(26)/GD2 | 推理**依赖邻接**（消息传递），非 graph-free；重/迭代 |
| Graph-free 节点分类 | Graph-MLP/GLNN/LinkDist/ES-MLP | **不做标签噪声**；我们补上标签噪声鲁棒 |
| 粒球计算 | MgCNL/MgBIF/CGBC/MOGB | 球多为 end-to-end 重塑表征或图像/表格域；我们=冻结对比表征上的离线球软标签，且落在 graph-free 图节点 |

- **诚实一句**：与 MgCNL/MgBIF 一样，我们的球也是 post-hoc（不参与表征梯度）；胜在"弱下游 + 对比表征 + 球服务弱下游"的组合，而非新球算法本身。

---

## 4. Method —— 组件 + 对应实现/证据

1. **对比表征（冻结）**：GRACE H（label-free，训练期吸收图结构）。为何 GRACE 见 §分析（其 alignment-uniformity 平衡使 H 可被 2-means 递归划分）。
2. **几何粒球**：在行-L2-归一化 H 上递归 2-means 建球（`ordinary_gb`/`build_frozen_grace_*`）。
3. **球内软标签 + β=0.5 半硬**：`target = (1−β)·in-ball 分布 + β·onehot(多数票)`，β=0.5 定型（MLP-4）。
4. **轻量 X-only MLP 分类器**：2-layer，**推理只用 X**（graph-free）。
5. **红线**：clean 只评测；噪声只加 train；球构造/软标签**不看 clean**。

- 证据锚：主算法定型 = MLP-4（β=0.5 唯一 6/6 稳≥软标签 β=0）；精细加权(ellipsoid/gmm)被半硬吸收 → 不加（MLP-5a，进消融）。

---

## 5. Experiments —— 表图清单（哪张表撑哪句 claim）

| # | 表/图 | 内容 | 来源 run | 撑的 claim |
|---|---|---|---|---|
| Setup | — | 数据(Cora/CiteSeer/PubMed[+Coauthor-CS])、噪声(uniform0.2/0.4/0.6,pair0.4,instance0.4)、percent split、同弱 MLP backbone、公平性声明 | MLP-1~5 | 公平性 |
| T1 | **主表** | β0.5 vs CE/small-loss/co-teaching/JoCoR/ELR/SCE/**FINE**/**DivideMix-lite**/**CT**(all adapted)+shuffled，3 数据集×5 噪声×5 seed | MLP-5 + 7a | 卖点2 + 球必要性 |
| T2 | **消融** | β{0,0.5,1}（硬度）; w/o-GRACE(X-balls); ellipsoid/gmm; shuffled | MLP-4/5a/2 | 球+GRACE+半硬各自必要 |
| F1 | **低标签率曲线** | Cora/CiteSeer × train%{0.05,0.10,0.20,0.35,0.50,0.65} × {β0.5,small-loss,JoCoR,FINE,shuffled}，5-seed mean±std | MLP-6 Part B | 卖点3 |
| T3 | **H-access 公平对照(诚实)** | β0.5 vs CE/small-loss/JoCoR 的 X-only 与 [X‖H] 两版 | MLP-6 Part A | scope 说明(不是碾压) |
| T4 | **结构/profile** | X-only 扰动 0.00pp vs 消息传递 GNN 掉点(MLP-2 C2)；效率(参数/墙钟) | MLP-2 C2 + 效率 | 弱 profile(不 headline) |
| (opt) | 图方法定位表 | TFR 等原生设置 acc（若跑） | 7b(可选) | related 位置 |

- **诚实注**：T1 里 FINE/DivideMix-lite/CT 全标 "MLP-adapted, not original"。PubMed 在 T1 里对 small-loss/JoCoR/CT/DivideMix-lite **输**——照实列，作边界。

---

## 6. Analysis / 适用域（诚实，是加分项不是短板）

- **甜区 = Cora/CiteSeer 型 citation**（少类、bag-of-words、中高噪声），球增量稳。
- **PubMed = 边界**：样本选择类 baseline 强，球软监督不足以超过 → 写清"当 H 可形成高质量球时最强"。
- **球必要性机制**：β0.5 > shuffled 全线（真球结构≠随机分组）；w/o-GRACE 崩（raw X 不可聚类，num_balls=1）。
- **H-access scope（把 MLP-6 结果讲成 scope）**：给 baseline 同样 [X‖H]，强选择可追平/反超 → 说明本方法价值**条件于 graph-free 推理**；若部署允许推理用图，[X‖H]+选择是有力替代。**主动说 = 诚实控制；被问出来 = 翻车。**
- **结构 profile（弱，诚实）**：X-only 推理对结构扰动/冷启动 0.00pp（7c/7c-ext）；但 GRACE-H 抗 edge-drop，故对 H-input MLP baseline 无强结构碾压——只写"graph-free 天然不依赖图，消息传递 GNN 会掉(C2)"。

---

## 7. Limitations & Conclusion（诚实收尾）

- 限制：①依赖可聚类对比 H（PubMed/部分数据集边界）；②同 H-access 下不碾压强选择；③球为 post-hoc、非新球算法；④弱 MLP 绝对精度低于强 GNN（设定使然）。
- 结论：在 graph-free 标签噪声鲁棒这一空白交集，粒球软监督是一个简单、非迭代、低方差、低标签率友好的有效方案。

---

## 8. 写作顺序（建议）

```
1. Method(§4) + Experiments setup —— 最锁定、最好写，先落
2. 主表 T1 + 消融 T2 + 低标签率 F1 —— 脊梁三条
3. Analysis(§6) —— 把 PubMed/H-access 讲成 scope
4. Related Work(§3) —— 四块切割
5. Intro(§2) —— 最后写，回扣全文 claim
6. Abstract + 结论 + 限制
```

- 每写一节，回本骨架核对：**这句 claim 有没有对应 run 撑？有没有越过 §0 的"不 claim"红线？**
- Claude 起草，逐节交用户核。先起草 §4 Method + §5 Setup？（下一步待确认）
