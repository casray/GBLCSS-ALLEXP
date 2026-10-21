# 执行计划：kNN/kmeans 平替对照 + expansion 翻案(蒸馏式全节点球监督)（Phase MLP-8）

日期：2026-07-02
执行者：Codex
定位：**写作前的方向级重审落地。两件事:(P0) 还"入场硬判据"的债——球软标签 vs kNN/kmeans 软标签平替对照,MLP 线从未跑过;(P1) 翻案 expansion——2026-06-10 判负记录证实其死于强 GCN 吸收(类A)而非机制缺陷,在 X-only 弱 MLP 上球扩展是结构信息到达学生的唯一通道,若成立则解决"MLP 绝对精度低 20pp"的最大痛点,论文升级为 noise-robust graph-free distillation。** P0+P1 跑完才动笔,论文骨架 v2 按分支定。

## 0. 背景(为什么是这两件)

- **P0 的债**:护城河规则"kmeans 能平替真球就不进场"从未在 MLP 主线执行。MLP-1~7 全部实验的结构对照只有 shuffled——shuffled 杀"随机分组",杀不了"kNN 邻域软标签≈球软标签"。历史上球在冻结 H 上被 kNN/kmeans 平替 5 次(S3选择/检测corr0.78-0.90/图聚类/open-intent 0除7/密度球),平替先验不低,必须补。
- **P1 的翻案证据**(`runs/ball_expansion_seed0_rocm_20260610/`):扩展伪标签精度 **Cora 0.83 / CiteSeer 0.86**(999 候选对 830),Cora 覆盖 46% 无标签节点——诊断信号极好;下游判负(-2.85pp)发生在 **GCN 线**,报告自写死因"SoftDist/GCN baselines are already strong"(消息传递本身就在向无标签节点传播监督,扩展冗余)。X-only MLP 无此通道 → 单调规则(GCN[X‖H]+1pp < GCN(X)+1~2pp < MLP+6~9pp)预测大幅转化。与 SoftDist 本体"GCN吸收→弱MLP复活"同构。
- **老师的精度抱怨由 P1 正面回应**:当前 MLP 只用 20% 节点监督(Cora u0.4 仅 0.556,比 GCN 低 ~20pp);扩展后监督量升到 ~66% 节点(83% 精度),绝对精度有望向 0.70+ 靠,进入 GLNN(graph-free 蒸馏)对话圈——**靠强化自己而非削弱对手**。

## 0.1 红线

1. clean 只进评测;噪声只加 train;球构造/软标签/扩展目标/teacher 训练一律不见 clean。
2. 同一 X-only 2-layer 弱 MLP backbone、同 config(延续 MLP-1~7);paired 5-seed + 逐 seed 方向;≥2/3 数据集一致才下结论。
3. **公平匹配**(P0 命门):kNN 的 k 匹配球的平均 train 节点数(从球诊断读 mean_train_per_ball);kmeans 的 K 匹配同一 run 的 num_balls。匹配值写进报告。
4. GLNN teacher 只在噪声 train 上训练(标准 GCN config);蒸馏目标 = teacher 对全体节点的 softmax,不碰 clean。
5. ROCm venv;percent split 0.2/0.1/0.7 显式传参。seed0 先行,过 gate 才扩 5-seed。

## 0.2 复用与新增

- **复用**:`frozengrace_gb_mlp_softdist_beta05`(+configs)、`mlp_ce`、MLP-5 主表数据(train-only β0.5 各 setting 5-seed 现成)、`build_ball_expansion_soft_targets`(utils 已有,expansion_min_train_per_ball=10 / margin_threshold=0.5 默认)、`use_ball_expansion` 分支(GCN 版 predictor 已有)、`build_ordinary_granular_balls`。
- **新增**:
  - P0:utils 加两个目标构造器——`build_knn_soft_targets`(行L2归一 H 上,每个 train 节点取 train 集内 k 近邻的标签分布,β 插值同 `apply_ball_soft_targets`)、`build_kmeans_soft_targets`(H 上 kmeans,K=matched num_balls,簇内 train 标签分布,β 插值);predictors `knn_mlp_softdist_beta05` / `kmeans_mlp_softdist_beta05` + configs(cora/citeseer/pubmed)。
  - P1:predictor `frozengrace_gb_mlp_softdist_beta05_expand`(backbone=mlp + use_soft_targets + use_ball_expansion + target_beta=0.5;**扩展目标同样做 β=0.5 硬化**,加单测:β=0 退化为原扩展分布);`mlp_distill_gcn`(GLNN式:先训噪声 GCN(X) teacher(标准config),取全节点 softmax(τ=1),同一 MLP student 以 KL/CE 学 teacher 目标,X-only 推理);shuffled-expansion 对照(随机球上的扩展)。
  - 单测:kNN/kmeans/expansion 目标不碰 clean(篡改 clean 不变);k/K 匹配逻辑;β 插值一致性。

## 1. P0 —— kNN/kmeans 平替对照(kill-level,先跑)

```
Datasets: Cora, CiteSeer, PubMed
Noise: uniform 0.4 (主;若便宜加 pair0.4)
Seeds: 0-4 (便宜,直接5seed)
Methods (全 X-only 弱MLP, β=0.5, 同config):
  球 β0.5            (复用 MLP-5)
  kNN-in-H β0.5      (新, k=matched)
  kmeans-H β0.5      (新, K=matched)
  shuffled β0.5      (复用, 参照)
```

### P0 ★判据(预注册)

| 观察(≥2/3 数据集,≥4/5 seeds) | 结论 |
|---|---|
| 球 − kNN ≥ +1pp 且 球 − kmeans ≥ +1pp | **球有独立价值**——最硬的 defense,写进论文(球的自适应粒度在"标签聚合"角色下不可平替) |
| 球 ≈ kNN/kmeans(差距 <1pp 或方向不稳) | **平替成立(第6次)**——诚实记录;"粒球"在 train-target 角色是装饰;论文机制需改名"局部共识软监督"或把球价值押到 P1 的 teacher 角色;**与用户/导师商量品牌问题,不藏** |
| 球 < kNN/kmeans | 更强的负结果,同上处理,且球降为消融 |

> 注:P0 判的是 **train-target 角色**;teacher/expansion 角色(P1)是不同角色,若 P0 平替,P1 加跑 kNN-expansion 对照臂(见 §2 条件臂)。

## 2. P1 —— expansion 翻案 + GLNN 蒸馏对照(天花板)

```
Datasets: Cora, CiteSeer, PubMed
Noise: uniform 0.4, uniform 0.6   (0.6 是钱设定:teacher 记忆噪声最伤 GLNN 的地方)
Seeds: 0 先行 → 幸存者 5-seed
Methods (全 X-only MLP 推理, 同config):
  β0.5 train-only          (复用 MLP-5, 基线)
  β0.5 + expansion         (新, λ_u ∈ {0.1, 0.3, 1.0} 扫描; 原GCN线只敢 0.05/0.10, 
                            监督饥饿的MLP值得扫宽)
  shuffled-expansion       (新, 真球扩展 vs 随机球扩展, λ_u 用最优值)
  GLNN蒸馏 (噪声GCN teacher)(新, 关键对手)
  MLP+CE                   (复用, 参照)
  [条件臂] kNN-expansion    (仅当 P0 平替时加跑: kNN 邻域做 teacher 扩展)
参照数(不重跑): GCN+CE(X) 同噪声精度(历史数据), 用于 G2 差距计算
```

### P1 ★判据(预注册,四道门)

| 门 | 条件 | 含义 |
|---|---|---|
| **G1 提升** | expansion − train-only ≥ +3pp,≥2/3 数据集 | 扩展在弱 MLP 上真转化(翻案成立) |
| **G2 精度** | 与 GCN+CE(X) 同噪声差距从 ~20pp 缩到 ≤8pp | "MLP 精度说不过去"的痛点实质缓解 |
| **G3 钱门** | 球 teacher ≥ GLNN(噪声GCN teacher):u0.4 ≥2/3 数据集,u0.6 明显更好 | **核心 claim**:"噪声下 GNN teacher 把记住的噪声蒸给学生,球几何 teacher 在噪声免疫空间聚合标签更稳"。若 GLNN 全面赢 → 诚实记录"蒸馏已足够,球作 teacher 冗余",不硬写 |
| **G4 球必要性** | expansion > shuffled-expansion | 扩展通道依赖真实球结构 |

### 结局分支(论文骨架 v2 按此定)

- **P0 球赢 + P1 过 G1-G4** → 最好结局:论文升级 **noise-robust graph-free distillation**(球=teacher 本体;精度进 GLNN 圈;双重鲁棒/低标签率卖点保留为次轴)。
- **P0 球赢 + P1 败** → 按现骨架(`paper_skeleton_graphfree_gb_20260701.md`)写,P0 作最硬 defense 补进消融。
- **P0 平替 + P1 过**(且 kNN-expansion < 球-expansion)→ 球价值在 teacher 角色;机制叙事围绕 teacher 重写;train-target 角色诚实写平替。
- **P0 平替 + P1 败** → MLP 线到顶;与导师讨论换任务(此时才轮到导师的方向建议)。

## 3. P2 —— 导师建议 smoke(仅导师坚持时跑,各半天,预注册 kill gate)

- **P2a 多任务 stage-1**:GRACE + λ·CE(noisy),λ=0.1,Cora seed0。Gate:球几何不塌(num_balls/purity vs GRACE-H)且下游不降。已有近邻判负:S1(有监督表征 epoch6 记噪声)、Phase9(可分性≠可聚性)、Phase20(辅助目标塌球)。预期负,跑完即收。
- **P2b loop T=2**:F0=现管线 → 用训好的 MLP 隐层作 Z1 → 重建球 → 软标签 → F1,Cora/CiteSeer seed0。Gate:2/2 数据集 loop 增益 >0 才考虑 T=3。已有近邻判负:dynamic-alpha 自增强。预期 ≈0/负。
- 两个 smoke 的价值 = 拿本管线上的直接证据回复导师,不是期望翻案。

## 4. 执行顺序

```
1. P0: utils 目标构造器 + predictors + configs + 单测(k/K匹配、clean不进、β一致) 
   → 跑 3数据集×u0.4×4方法×5seed → 按 §1 判
2. P1: expand-MLP predictor(β硬化扩展目标+单测) + mlp_distill_gcn + shuffled-expansion
   → seed0: 3数据集×{u0.4,u0.6}×全方法×λ_u扫描 → 过G1才5-seed(最优λ_u) 
   → 按 G1-G4 判 → (P0平替时加 kNN-expansion 臂)
3. 汇总: P0 平替判定表 + P1 四门判定 + 分支结论(论文骨架v2走哪支)
4. (P2 仅导师坚持时另行触发)
```

- 输出目录:`external/NoisyGL/runs/mlp8_replacement_expansion_20260702/`
- 报告:P0 配对表(球vs kNN vs kmeans,含k/K匹配值) + P1 四门(expansion提升/绝对精度差距/vs GLNN/vs shuffled-exp) + λ_u 扫描 + 分支判定。

## 5. 完成后

- 更新记忆(P0 平替与否 = 第6次平替检验结果;P1 翻案与否 + 四门;论文走哪个分支)。
- 写作前置检查:精读 DND-NET(KDD2024, simple model+noisy labels+graph,最近 competitor)做切割——Claude 负责,不阻塞 Codex。
- 按分支产出论文骨架 v2 → 进写作。

---

## Phase MLP-8 执行结果备案
结果文件:`external/NoisyGL/runs/mlp8_replacement_expansion_20260702/`

### P0: kNN/kmeans 平替对照

- 完整性: expected 60 / observed 60 / missing 0。
- true ball vs shuffled 仍然强成立:
  - Cora: +11.96pp, 5/5 seeds
  - CiteSeer: +11.48pp, 5/5 seeds
  - PubMed: +9.18pp, 5/5 seeds
- 但 true ball 没有超过 kNN/kmeans 平替:
  - Cora: ball-kNN +0.32pp (2/5), ball-kmeans +0.05pp (3/5)
  - CiteSeer: ball-kNN -1.76pp (1/5), ball-kmeans -1.49pp (1/5)
  - PubMed: ball-kNN -0.72pp (0/5), ball-kmeans -0.27pp (2/5)
- P0 判定: `p0_replacement_not_rejected`。

结论: 在 train-target 角色下,真粒球相对随机分组有价值,但没有证明比 H 空间 kNN/kmeans 局部共识更有独立价值。论文不能再把这部分写成"粒球不可平替";若保留,应诚实表述为 H-space local consensus / ball-style local label aggregation。

### P1: expansion 翻案 + GLNN seed0

- 完整性: P1 observed 78 rows = MLP-5 anchors 18 rows + 新跑 expansion/GLNN 60 rows。
- u0.4 下 expansion 翻案的 G1 成立:
  - λ=0.1: expansion - train-only mean +7.07pp, 3/3 datasets pass
  - λ=0.3: +7.90pp, 3/3 pass
  - λ=1.0: +8.76pp, 3/3 pass
- u0.6 下 G1 不成立:
  - λ=0.1: +2.93pp mean, 1/3 datasets 达 +3pp
  - λ=0.3: +2.69pp mean, 1/3 datasets 达 +3pp
  - λ=1.0: +3.60pp mean, 1/3 datasets 达 +3pp
- G4 vs shuffled-expansion 成立: 所有 λ、所有噪声率均 3/3 为正。说明扩展监督通道不是随机分组造成的。
- G4 vs kNN-expansion 不成立: 所有 λ、所有噪声率均 0/3 为正。kNN-expansion 全面高于 true ball expansion。
- G3 vs GLNN 不成立:
  - u0.4: true expansion - GLNN mean 为 -6.20 / -5.37 / -4.51pp
  - u0.6: true expansion - GLNN mean 为 -14.22 / -14.45 / -13.55pp

关键 seed0 数字:

| Setting | Cora | CiteSeer | PubMed |
|---|---:|---:|---:|
| train-only beta0.5, u0.4 | 0.5546 | 0.5533 | 0.7224 |
| true expansion best, u0.4 | 0.6586 | 0.6117 | 0.8263 |
| kNN-expansion best, u0.4 | 0.7383 | 0.6443 | 0.8358 |
| GLNN noisy-GCN teacher, u0.4 | 0.7435 | 0.6482 | 0.8368 |
| train-only beta0.5, u0.6 | 0.3894 | 0.3862 | 0.4975 |
| true expansion best, u0.6 | 0.4174 | 0.4613 | 0.5160 |
| kNN-expansion best, u0.6 | 0.4876 | 0.5228 | 0.5805 |
| GLNN noisy-GCN teacher, u0.6 | 0.6274 | 0.4768 | 0.6832 |

### 分支决策

MLP-8 没有支撑"粒球 teacher/expansion 是 GLNN 圈内更强 teacher"这一升级主线。更准确的结论是:

1. expansion 在弱 X-only MLP 上确实能把 train-only SoftDist 拉高,尤其 u0.4,说明"GCN 吸收导致原 expansion 判负"这个解释有部分成立。
2. 但 kNN-expansion 和 GLNN teacher 全面压过 true ball expansion,所以提升不能归因到粒球自适应粒度;更像 H 空间近邻/GCN teacher 的全节点蒸馏优势。
3. u0.6 再次确认 offline noisy-label majority/expansion 在极高噪声下退化,不能作为高噪声优势点。
4. 不建议把 P1 扩到 5 seeds 作为主线;最多作为负/边界分析。若继续写 MLP 论文,主线应回到 graph-free/结构鲁棒/低标签率/轻量 profile,并诚实加入 kNN/GLNN 对照。

### P2: 导师建议 smoke

结果文件:`external/NoisyGL/runs/mlp8_p2_advisor_smoke_20260702/`

P2a: GRACE + 0.1 noisy CE, Cora seed0。

| Variant | Num balls | True all purity | Downstream test |
|---|---:|---:|---:|
| anchor GRACE | 52 | 0.7873 | 0.5546 |
| GRACE + noisy CE λ=0.1 | 1 | 0.3021 | 0.4285 |

- P2a gate: fail。
- num_balls 从 52 塌到 1, true_all_purity 从 0.7873 掉到 0.3021, 下游从 0.5546 掉到 0.4285。
- 结论: noisy supervised signal 即使只以 λ=0.1 加到 GRACE stage-1,也会破坏当前方法真正依赖的 ball-clusterability。这个结果和历史判断一致:有监督/CE 方向可能提高分类可分性,但会破坏粒球可聚类性;不适合作为本方法的 stage-1。

P2b: F0 -> Z1 -> balls -> F1 loop, Cora/CiteSeer seed0。

| Dataset | F0 test | F1 loop test | Loop delta | F1 num balls | F1 true all purity |
|---|---:|---:|---:|---:|---:|
| Cora | 0.5546 | 0.4285 | -12.61pp | 1 | 0.3021 |
| CiteSeer | 0.5533 | 0.4343 | -11.90pp | 1 | 0.2107 |

- P2b gate: fail, positive datasets 0/2, mean loop delta -12.26pp。
- F0 的 MLP hidden Z1 在两个数据集上都让球构造塌成 1 个球,然后 F1 明显下降。
- 结论: loop 方向不是"迭代改进",而是 self-produced representation 破坏球几何后的自增强退化。没有理由扩 T=3 或 5 seeds。

P2 总决策: 导师两个建议均被当前管线的直接 smoke 否决。它们可以作为回复导师的证据:本方法的关键不是"更有监督的 encoder"或"循环自训练",而是 frozen label-free GRACE H 的局部可聚类性。一旦让 noisy CE 或训练后 MLP hidden 进入建球空间,粒球结构直接塌缩。
