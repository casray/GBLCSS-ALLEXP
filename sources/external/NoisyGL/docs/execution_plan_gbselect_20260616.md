# 执行计划：粒球-GCN 双视角 clean-sample selection 下游验证（Phase 34）

日期：2026-06-16
执行者：Codex
前置状态：Phase 33 判 go——大图（DBLP/PubMed/Amazon-Photo）上球的噪声判别信号**条件正交于** GCN（GCN-clean50 子集内 ball AUC 0.66-0.95，均过 0.6 门槛）。离线 co-teaching 交集把 train 噪声率从 ~40% 降到 5.7-6.9%，保留 52-54% 样本。**小图（Cora/CiteSeer）上同样强**（Cora 39%→2.25%）——说明这是一个**统一**用法,不是大图专用补丁。

**本计划用真实下游训练验证离线信号能否转化为 test acc 增益。** 核心机制转变：粒球从「soft-target 校正器」改为「label-free、与模型预测正交的噪声判别视角，用于 clean-sample selection」。这是 SoftDist 之外的**新方法**（命名 `GBSelect` 或 `BallCoTeach`），不是 SoftDist 的变体。

---

## 0. 已被两次教训确立的红线（必读，决定实验对不对）

### 教训：离线信号 ≠ 下游增益（Phase 9、Phase 32b 各骗过一次）

离线降噪 40%→6% 很强，但**必须下游真训练确认**。且这次有一个比以往更致命的对照盲点,见下。

### ★命门对照：必须证明"球+GCN 双视角" > "GCN 单视角"

co-teaching 的全部价值在于**两个正交视角**。若 GCN 单视角自己选样本就能达到同样 test acc，球就仍是冗余,整条线作废。**这是 Phase 34 的核心判定,不是可选项。** 33 的离线数据只证明了"球在 GCN-clean 子集有信号",但**没有**离线对比"双视角最终选择质量 vs GCN 单视角选择质量"——所以必须靠下游真实验证。

### 标签泄漏红线（比以往更要小心）

1. **warmup GCN 的 `s_gcn` 只能用 noisy 标签训练**，早停在记忆噪声前（epoch 5-10 窗口，沿用 Phase 33 设置）。clean 标签**只用于最终 test 评估和诊断**,绝不进 selection 决策。
2. **`s_ball` 只用 train noisy 标签**（球内多数票不一致度），与 SoftDist 同源,不碰 clean。
3. selection 后在干净子集上训练最终 GCN——这是合法的半监督操作（用了哪些节点是由 noisy 标签+几何决定的,没用 clean）。
4. **val/test 选择**：selection 只作用于 **train** 节点。val 仍用全部 val（noisy）做早停,test 全量评估。不要对 val/test 做 selection。

### 环境

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- 已有 H（不重训）：`embeddings_v2/{DBLP,PubMed,AmazonPhoto}/grace/H.pt` + Cora/CiteSeer 锚点 H
- 大图球构造用 Phase 32a 适配参数（DBLP/PubMed max_balls=256, split_improvement≈0.001-0.005；小图用默认 64/32/0.05）
- NoisyGL 显式 percent split 0.2/0.1/0.7
- 不改 SoftDist anchor；新方法新 predictor 类；research-record 报告；测试全绿

---

## Phase 34a：实现 GBSelect predictor

### 机制（清晰三步）

```
Step 1 — 双视角噪声判别（只用 noisy 标签）
  s_ball(i) = 1 - dist_B[y_noisy_i]              # 球内反对票占比
  warmup GCN([X||H], noisy CE, ~10 ep, 取 ep5-10 窗口均值预测)
  s_gcn(i)  = 1 - p_warmup(y_noisy_i | i)
Step 2 — 双判干净交集（keep_r_each）
  clean_ball = train 节点中 s_ball 最低的 r 比例
  clean_gcn  = train 节点中 s_gcn 最低的 r 比例
  selected   = clean_ball ∩ clean_gcn            # 双视角都认干净
Step 3 — 在 selected 上训练最终 GCN([X||H])
  标准 CE，只在 selected train 节点上算 loss
  val(noisy) 早停，test 全量评估
```

### 实现要点

- 新 predictor `GBSelect_Predictor`（注册 `gbselect`），复用 `concat_l2_normalized_embeddings` + `build_ordinary_granular_balls` + 球投票逻辑;warmup GCN 复用现有 GCN 模块
- config 字段：`keep_ratio`（默认 0.6）、`warmup_epochs`(10)、`warmup_window`(5,10)、球构造参数（按数据集）
- 单元测试：selection 只动 train（val/test 不变）;clean 标签不进 selection（篡改 clean 标签 selected 集不变）;keep_ratio=1.0 时退化为全样本 = gcn_xh_ce

### 三个命门对照变体（都要注册）

| 变体 | selection 用什么 | 回答 |
|---|---|---|
| `gbselect`（主） | s_ball ∩ s_gcn 双视角 | 完整方法 |
| `gbselect_gcn_only` | 只用 s_gcn 选（同样 keep 比例下取最干净） | **球第二视角是否必要**（命门）|
| `gbselect_shuffled` | shuffled 球的 s_ball ∩ s_gcn | **真球结构是否必要** |
| `gcn_xh_ce`（已有） | 不选,全样本 | 基线 |

---

## Phase 34b：下游实验

### 运行

- 数据集：DBLP、PubMed、Amazon-Photo（大图主目标）+ Cora、CiteSeer（验证统一框架）
- 方法：`gbselect`、`gbselect_gcn_only`、`gbselect_shuffled`、`gcn_xh_ce`、`nrgnn`（参考）× 5 seeds × uniform 0.4
- keep_ratio：主设置 0.6；附加 0.5 / 0.7 敏感性（至少在 DBLP 上做）
- 必报：test acc + selected 集真实噪声率（clean 仅诊断,验证离线 40%→6% 在真 run 里复现）+ selected 样本量
- 输出 `runs/gbselect_20260616/`

### ★判定（按命门优先级）

| 对照 | 判据 | 含义 |
|---|---|---|
| **gbselect vs gcn_only** | gbselect mean ≥ gcn_only 且 ≥3/5 seeds 正 | **球第二视角必要——这是核心,不过则全线作废** |
| **gbselect vs shuffled** | gbselect > shuffled,≥4/5 seeds | 真球结构必要 |
| gbselect vs gcn_xh_ce | gbselect > 全样本基线 | selection 本身有用 |
| gbselect vs nrgnn | 参考,不强求 | 与强 baseline 的位置 |

### 判定后果

| 结果 | 论文影响 |
|---|---|
| gbselect > gcn_only **且** > shuffled（大图上成立） | **数据集广度解锁 + 统一框架成立**：粒球作为正交噪声判别视角,小图大图通吃。论文从"2 数据集 focused"升级为"统一 label-free 粒球去噪框架"。这是重大升级 |
| gbselect ≈ gcn_only（球第二视角无下游增益） | 球冗余的第三次确认。诚实记录,数据集广度仍只 2 个,回到 focused 定位。33 的离线正交信号写入 §6 作为"离线可分但下游被吸收"的边界 |
| gbselect > gcn_only 但仅小图 | 统一框架不成立,但小图上多一个有效用法,丰富方法 |

---

## Phase 34c（仅当 34b 大图成立）：扩展验证

- 噪声谱：在解锁的大图上补 pair 0.4 / instance 0.4,确认 selection 在不同噪声类型下稳健
- 与 SoftDist 的关系：明确论文双机制定位——小图标签稀缺用 SoftDist soft-target,大图标签足噪声高用 GBSelect selection,统一在"label-free 粒球作为噪声感知单元"框架下
- 单独 5-seed 正式表

---

## 执行顺序

```
34a 实现 GBSelect + 3 对照变体 + 单测
  └─► 34b 下游 5-seed（先 DBLP 单数据集验命门 gbselect vs gcn_only,再展开）
        ├─大图成立─► 34c 扩展 + 统一框架叙事 + 论文升级
        └─不成立───► 收束,33 离线信号进 §6 边界分析
```

- **34b 先在 DBLP 上验命门对照（gbselect vs gcn_only）**,这一格决定整条线生死,过了再铺开其余数据集,不一次性 all-in
- 每步报 selected 集真实噪声率,确认离线信号在真 run 复现

## 完成后

回填论文初稿：若成立,新增 GBSelect 方法节 + 统一框架图 + 大图主表 + selection 降噪诊断;§6/§8 更新。更新经验文档与记忆。无论成败,Phase 33-34 的"球-GCN 正交性"分析都是论文的重要内容（成则核心贡献,败则精确边界）。

---

## Phase 34 执行结果备案（2026-06-16）

结果文件：`runs/gbselect_20260616/`，汇总报告：`runs/gbselect_20260616/gbselect_phase34_report.md`。

### 关键结论

GBSelect 没有解锁大图统一框架。离线 co-teaching 降噪信号在真实训练中只部分转化：

- DBLP：真实球结构明显优于 shuffled，但命门对照 `gbselect > gcn_only` 不稳定。默认 keep=0.6 为 -0.04pp、1/5 seeds 正；keep=0.7 为 +0.35pp、3/5 seeds 正，未达到预注册 4/5 门槛。
- PubMed：`gbselect > shuffled` 为 +1.63pp、5/5 seeds，但 `gbselect > gcn_only` 仅 +0.06pp、2/5 seeds，命门不过。
- Amazon-Photo：`gbselect > gcn_only` 为 +3.11pp、5/5 seeds，说明球第二视角有效；但 `gbselect` 大幅低于全 train 的 `gcn_xh_ce`（-12.32pp、0/5 seeds），selection 范式整体失败。

### 判定

Phase 34c 不触发。不得把 GBSelect 写成当前论文的第三成功数据集或大图主方法。它可以进入论文 §6/§8 的边界分析：球信号能离线识别噪声，但下游 test acc 会被样本量损失、GCN 单视角吸收和数据集 regime 差异抵消。

### 后续约束

- 不再沿 GBSelect 做小幅 keep_ratio 调参来追主表，除非先提出不依赖 clean label 的选择强度自适应规则。
- DBLP keep=0.7 只能写作弱正现象，不能作为稳定 claim。
- Amazon-Photo 的结论是“球视角有信息，但丢弃监督量代价过高”，不是“球无效”。
