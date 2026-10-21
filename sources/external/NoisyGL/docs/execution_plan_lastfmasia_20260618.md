# 执行计划：LastFM-Asia 第4成功数据集诊断（Phase 39）

日期：2026-06-18
执行者：Codex
前置状态：成功数据集 3 个（Cora/CiteSeer/Coauthor-CS）。已知成功边界条件（Phase 35-38 归纳）：**类数≥6 + 同质性约 0.65-0.82 + GRACE H 可聚类（球纯度/正交AUC高）**。审稿门槛是"≥5 数据集做实验"——目前已满足（7+ 数据集，3 成功 + 4 有边界解释）。本计划追加 **LastFM-Asia**（PyG 原生，7.6k 节点，**18 类**，社交友谊图）——它是 PyG 里少数精准命中"类数≥6"的未用数据集，目标成为第 4 个成功数据集。**诊断先行,不改主方法机制。**

### 为什么是 LastFM-Asia

- 类数 18（远超阈值 6，类多→GCN 单视角难，球有正交空间）
- 规模 7.6k（适中，预训练成本低）
- **同质性未知**（社交友谊图，检索称"中等同质"，需跑出来确认落不落在 0.65-0.82 甜区——这正是诊断要回答的）
- 其余社交图候选（DeezerEurope 2类、FacebookPagePage 4类、Twitch 2类）均因类太少被排除

### 预期管理（诚实）

- LastFM-Asia 可能成功（条件匹配），也可能像 Amazon 系被吸收（同质性太高）或像 WikiCS 球纯度不足（同质性太低）——**跑了才知道,不预设成功**
- 即使失败：4 边界数据集进适用域刻画，论文"成功有解释/失败有边界"的故事不受损
- **不强求把成功数硬凑到 5**：3 干净成功（含大图）+ 完整边界刻画已是强论文；LastFM 成则锦上添花

---

## 0. 环境与红线（沿用）

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- NoisyGL 显式 percent split 0.2/0.1/0.7，uniform 0.4，seeds 0-4
- 红线：anchor 机制不动；clean 只进诊断；SSL label-free（只用 X,A）；GRACE 超参用同规模数据集（~7.6k 用 Amazon-Photo 同档）预注册不调；研报告；测试全绿

---

## Phase 39a：接入 + 预训练 + 诊断先行（核心，廉价，不下游）

### 步骤

1. **loader 接入**（小改动）：`utils/dataloader.py` 加 `from torch_geometric.datasets import LastFMAsia` + dataset key `lastfmasia` → `LastFMAsia(root=...)`；single_exp.py choices 加 `lastfmasia`。注意：无 default split mask,必须走 percent split（与 Amazon/Coauthor 同款,已支持）
2. 触发下载（PyG 自动）
3. **先量同质性**：加载后算 edge homophily ratio（clean label 仅诊断）——若 <0.6 或 >0.85，提前预警可能失败，但仍继续诊断（数据会说话）
4. GRACE label-free 预训练 → `embeddings_v2/LastFMAsia/grace/H.pt`，**预注册**用 Amazon-Photo 同档配置（~7.6k 同规模），执行前写报告；H-图对齐 sanity check
5. **球诊断（决定是否下游）**：复用 Phase 33/37 脚本,报：
   - num_balls（是否塌缩）、true_ball_purity、noisy_train_purity
   - **正交 AUC（ball_auc_in_gcn_clean50）**——核心判别量
   - homophily 实测值
6. 输出 `runs/lastfmasia_20260618/`

### 进入 39b 的门槛（对照已知成功/失败基准）

- 参照基准：成功集(Cora/CiteSeer/Coauthor-CS)正交 AUC 0.83-0.85、true purity 0.70-0.91；失败集 WikiCS 正交 AUC 0.528、Amazon 系球被吸收
- **正交 AUC ≥ 0.7 且 num_balls 不塌缩(≥8) 且 true purity ≥ 0.65** → 有成功潜力,进 39b 下游
- 任一不达标 → 大概率失败,可仍跑下游确认(成本不高)但预期为边界点,归入适用域刻画

---

## Phase 39b：下游验证（诊断过门 / 或确认边界）

- 方法：`softdist_xh`、`softdist_xh_shuffled`、`gcn_xh_ce`、`gcn`、`nrgnn`、`nrgnn_xh` × 5 seeds × uniform 0.4
- 必报：SoftDist−gcn_xh_ce（球增量）、SoftDist−shuffled（结构对照）、球诊断
- 输出 `runs/lastfmasia_20260618/lastfmasia_downstream.csv`

### 判定

| 结果 | 论文影响 |
|---|---|
| SoftDist−gcn_xh_ce ≥0 且 real>shuffled（多数 seeds） | **第 4 个成功数据集！** 进主表,成功集=Cora/CiteSeer/Coauthor-CS/LastFM-Asia。且 18 类进一步印证"类多→球有用" |
| real>shuffled 但输全样本（类 Amazon-Computers） | 边界点,球信号真实但被吸收,进 §8 适用域 |
| 球塌缩/正交AUC低/real≈shuffled | 边界点,进适用域刻画,与同质性实测值一起解释 |

---

## Phase 39c：回填 + pair 噪声 future work

1. 若 LastFM 成功 → 主表(§3.1)/消融(§4.1)/公平性(§4.2)加 LastFM-Asia 列；适用域图(§8.1f)加一个数据点（也让 proxy 相关性样本 n 增加 1,虽仍 <10）
2. 若边界 → §8 适用域表加点 + 同质性/球纯度解释
3. **pair 噪声 future work（写入 §8 limitations / future work,不开实验）**：
   - 现象：pair 噪声下球增量最弱（Cora +0.52pp vs uniform +1.10pp）
   - 机理：pair 噪声系统性把 A→B 翻转，球内出现"伪多数"，集体投票被一致地带偏（uniform 的随机错误投票能压过，pair 的系统错误压不过）
   - 候选解法（future work）：用 **H 几何一致性校验投票**——H 是 label-free 的，对 pair 翻转免疫；被翻转节点在 H 空间仍靠近真类簇中心，可据此识别"伪多数"。这是新机制，需重走诊断→预注册流程，留作未来工作
4. 更新经验文档与记忆、探索轴总览 HTML

---

## 执行顺序

```
39a 接入+预训练+球诊断（廉价,先做）
   ├─正交AUC≥0.7 且不塌缩 ─► 39b 下游（争取第4成功数据集）
   └─不达标 ───────────────► 仍跑下游确认,预期边界点
39c 回填（成功→主表;边界→适用域）+ pair future work 写入
```

- 39a 诊断先行,几分钟出正交AUC,直接预判成败
- 不为凑"5 个成功"硬来：LastFM 成则 4 个,不成则维持 3 成功 + 完整边界,都是强论文
- pair 噪声只写 future work,本计划不开新机制实验

## 完成后

成功：主表 4 个成功数据集（含 18 类社交图,数据集多样性更强）。边界：适用域刻画再加一个有解释的点。两种情况都更新三份文档（经验/初稿/探索轴总览），pair 几何校验入 future work。此后实验收口,进入正文写作。
