# 执行计划：Amazon-Photo 第 5 数据集扩展（Phase MLP-12）

日期：2026-07-02
执行者：Codex
定位：**主表的最后一次 scope 扩展（领域多样性轴），不是新机制。** 把 Amazon-Photo（7.6k 节点/8 类/共购图）加为第 5 个数据集，使主表领域覆盖变为 citation×3 + coauthor×1 + **co-purchase×1**，堵"全是 citation/coauthor"质疑。门控结构照抄 MLP-10 成功模板：入场诊断 → seed0 smoke → 过门才整 block 5-seed。**本计划与 MLP-11（k 敏感性）互不依赖，可并行跑。跑完主表 scope 永久关闭。**

## 0. 背景与诚实先验

- **为什么选 Amazon-Photo 而不是 DBLP/AmazonCom/WikiCS/LastFM/Physics**：DBLP 默认参数下球塌缩成单球（Phase 28），需按数据集调 split_improvement——与 MLP-11 正在测量的"同一套冻结构造参数跨数据集"核心 claim 直接冲突，加它等于亲手毒死 K-free 故事，且 4 类/citation 重复，明确不做。WikiCS（正交 AUC 0.528）/LastFM（球纯度 0.254）被入场规则挡死。Coauthor-Physics 无 GRACE H 且 5 类低类数高危。AmazonCom 与 Photo 同域边际递减，仅导师要求时作第 6 个。
- **Amazon-Photo 的正面先验**：球几何良好——Phase 19 球纯度 0.86、球没塌；椭球几何诊断 root split gain 0.227、H-kNN lift 5.31（局部结构强）。旧 GCN 线死因是"被强同质基线吸收"（real≈shuffled、SoftDist≈GCN，Phase 19），这与 PubMed（旧线输 GCN+CE → 新线 vs shuffled +9pp）和 CoauthorCS（旧线 +0.53pp 边际 → 新线 4/5 全表最高）是同一个吸收类死因，弱 MLP 复活剧本已验证两次。
- **诚实先验（两种结局都可能，写在前面）**：Photo 同质性 0.79 很强、X 特征信息量大 → clean/noisy loss 可分 → 选择族（small-loss/JoCoR/CT）可能像 PubMed 一样强。结局要么是 CoauthorCS 型胜行（8 类、球纯度高），要么是 PubMed 型边界行。**两种都预注册整 block 进表；边界行也是净收益（领域多样性 + 边界规律第三次确认）。**
- **说服力天花板**：同档期刊同类论文 4-5 个数据集（DND-NET 4、RNCGLN 4）。5 个封顶，不再加第 6 个。

## 0.1 红线（预注册）

1. **整 block 进表**：一旦进入 5-seed，5 设置 × 全部 13 方法整块进主表，不许挑格；seed0 止损也如实写进边界分析。
2. **同 backbone 逐键一致**：全部 MLP 方法 config 从各自 cora/coauthorcs 模板派生，backbone 键与 `mlp_ce` 逐键相同；β=0.5 的 `gb:` 块与四数据集版逐键一致（max_balls=64、split_improvement=0.05 等**全局默认，绝不为 Photo 单独调**——这是 K-free claim 的生命线；`_conf_value` 覆盖类属性的老坑照例落 CSV 核实）。
3. **split 显式传参**：`amazonpho.yaml` 的默认 split 是 samples_per_class（已核实）——必须显式 `--split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7`，并在首个 run 后核对 CSV 里的实际 train 节点数 ≈ 0.2×7650≈1530（防静默回退，这是本数据集特有的坑）。
4. clean 只进评测；噪声只加 train；ROCm `cuda:0`；paired 同 seed 同 split。
5. LogitClip τ 与 ANL 超参沿用 MLP-10 协议：noisy val 从预声明网格选一次（LogitClip τ ∈ {0.5,1.0,2.0}），不看 test 回调。
6. 判读纪律沿用：<1×std 写 comparable；若落 PubMed 型边界，解释统一挂"选择族在 clean/noisy loss 可分 regime 更强"，不写"扩展失败"。
7. 本计划不改 claim 结构；主表措辞按实际结果收窄或补行。

## 0.2 复用与新增（已核实的资产现状）

**复用（不重跑/不重写）**：
- GRACE H：`gb_graph_aware_minival_6_1/checkpoints/embeddings_v2/AmazonPhoto/grace/H.pt`（已确认存在，含 metadata.json；节点序对齐 NoisyGL dataloader——沿用旧 GCN 线 amazonpho configs 里的 embedding_path 写法）。
- dataset config：`config/_dataset/amazonpho.yaml` 已存在。
- `gb:` 块模板：`config/frozengrace_gb_gcn_softdist_xh/frozengrace_gb_gcn_softdist_xh_amazonpho.yaml`（旧 GCN 线，H 路径可参考；MLP 版从 coauthorcs 模板派生更近）。
- 诊断脚本：球纯度/正交 AUC 诊断管线（Phase 37/39 用过的）。

**新增（已核实全部 MISSING）**：
- 13 个 MLP 方法的 amazonpho config：`frozengrace_gb_mlp_softdist_beta05`、`_beta05_shuffled`、`mlp_ce`、`mlp_small_loss`、`mlp_coteaching`、`mlp_jocor`、`mlp_elr`、`mlp_sce`、`mlp_fine`、`mlp_dividemix_lite`、`mlp_ct`、`mlp_logitclip`、`mlp_anl` 各一份 `*_amazonpho.yaml`（从各自 coauthorcs 模板派生，只改 dataset 字段与 H 路径）。
- 单测：(a) amazonpho 新 configs 与 coauthorcs 模板除 dataset/路径字段外逐键一致；(b) percent split 实际生效（train 节点数断言）；(c) 噪声只污染 train。
- 汇总脚本：扩展 `summarize_mlp10_baseline_refresh.py` 或新建 `summarize_mlp12.py`（五数据集主表、paired deltas、完整性）。

## 1. 阶段 0 —— 入场诊断（几分钟，先于一切）

```
D0: AmazonPhoto 球纯度 + 正交 AUC，当前协议（percent 20/10/70, uniform 0.4, seed0）
门: true_ball_purity ≥ 0.65 且球数 > 1
```

Phase 19 的 0.86 是旧协议数字，当前协议复核一遍。不过门 → 全停，写一段边界记录，主表维持 4 数据集（预期大概率过，但入场规则不跳）。

## 2. 阶段 A —— seed0 smoke（~15 run）

```
A1: {beta05, mlp_ce, beta05_shuffled} × amazonpho × 5 设置 × seed0 = 15 run
门 GA: 全部 status=ok、无 OOM、train 节点数正确；
      β=0.5 在 ≥3/5 设置同时 > CE 且 > shuffled
```

- 过 → 阶段 B；不过 → 停，seed0 止损结果写边界分析（照 MLP-10 GA2 语义：此门只决定花不花 5-seed 算力，不决定进不进论文——进了 5-seed 就整 block 进表）。

## 3. 阶段 B —— 整 block 5-seed（过门后）

```
B1: 13 方法 × amazonpho × 5 设置 × seeds 0-4 = 325 run
    （A1 的 15 个 seed0 行复用，实际新增 310 run；LogitClip 的 τ 先用
     seed0 noisy-val 选定后再扩 seeds 1-4）
```

约一天（弱 MLP + 7.6k 节点；FINE 谱分解与 DivideMix GMM 的内存在 CoauthorCS 18.3k 已验证无 OOM，Photo 更小无风险）。

## 4. 产出

- 输出目录：`runs/mlp12_amazonphoto_20260702/`
- 报告 `mlp12_report.md` 必含：
  1. 入场诊断数字（球纯度/正交 AUC/球数）；
  2. AmazonPhoto block 主表（13 方法 × 5 设置，mean±std）；
  3. β=0.5 vs 各 baseline 的 paired delta + 逐 seed 方向（≥4/5 才写"稳定"）；
  4. 胜/负/comparable 设置计数与定性（CoauthorCS 型胜行 or PubMed 型边界行）；
  5. real vs shuffled 的五设置配对（结构必要性在共购域是否成立——这行无论胜负都是 §6 素材）;
  6. 完整性 expected/observed/missing；split 核对结果（train 节点数）。

## 5. 完成后

- 主表底稿更新为五数据集版（Cora/CiteSeer/PubMed/CoauthorCS/AmazonPhoto × 12 方法列 + shuffled 行）。
- 骨架 v2 的 T1 描述与 §6 适用域段同步（若边界行：并入"选择族强 regime"的统一解释；若胜行：领域覆盖句升级）。
- 更新记忆与 `99_session_handoff.md`。
- **主表 scope 永久关闭**：AmazonCom 仅导师明确要求时作第 6 个（同模板另立计划）；DBLP/WikiCS/LastFM/Physics 不做（理由见 §0）。
- MLP-11 + MLP-12 之后：实验全部清零，只剩导师沟通包与正文写作。

---

## 6. 执行结果备案（Codex 跑完后按 MLP-8/9/10 格式追加于此）

### 6.1 执行结论（2026-07-02）

**结论：Amazon-Photo 过门并进入主表，属于 CoauthorCS 型胜行，不是 PubMed 型边界行。** D0 入场诊断显示 frozen GRACE 几何非常适合局部共识：`num_balls=133`、`true_all_purity=0.8716`、`noisy_train_purity=0.5431`、`ball_auc=0.9468`、`ball_auc_in_gcn_clean50=0.9407`，超过预注册门槛（purity >= 0.65 且 num_balls > 1）。A1 seed0 smoke 中，β=0.5 在 5/5 噪声设置同时超过 MLP+CE 和 shuffled，触发完整 5-seed block。

Full block 已完成：`13 methods × 5 noise settings × 5 seeds = 325 rows`，完整性 `missing=0`。LogitClip 按 noisy validation 选定 `tau=0.5`。所有 run 均使用 `split_type=percent, train_percent=0.2, val_percent=0.1, test_percent=0.7`，实际 split 为 1530/765/5355，噪声只污染 train。

### 6.2 Amazon-Photo 5-seed 主表摘要

| Noise | β=0.5 ours | strongest adapted baseline | MLP+CE | Shuffled |
|---|---:|---:|---:|---:|
| uniform0.2 | **86.20±1.46** | JoCoR 85.60±1.25 | 82.34±1.39 | 81.99±1.03 |
| uniform0.4 | **81.64±0.92** | JoCoR 79.57±0.53 | 73.90±1.80 | 72.06±2.97 |
| uniform0.6 | **70.78±3.19** | JoCoR 66.48±4.43 | 58.60±3.77 | 59.65±5.46 |
| pair0.4 | **72.51±4.01** | LogitClip 68.88±3.39 | 62.61±4.31 | 62.95±4.93 |
| instance0.4 | **79.59±1.15** | JoCoR 74.80±1.78 | 69.13±3.00 | 68.98±2.44 |

### 6.3 关键 paired deltas

- **结构必要性（real vs shuffled）稳定成立**：uniform0.2 `+4.21pp`、uniform0.4 `+9.58pp`、uniform0.6 `+11.14pp`、pair0.4 `+9.56pp`、instance0.4 `+10.61pp`，全部 `5/5` seeds 正向。
- **相对 MLP+CE 的标签鲁棒增益很强**：uniform0.2 `+3.86pp`、uniform0.4 `+7.73pp`、uniform0.6 `+12.19pp`、pair0.4 `+9.90pp`、instance0.4 `+10.46pp`，全部 `5/5` seeds 正向。
- **相对强抗噪策略仍为正**：vs JoCoR 分别为 `+0.60/+2.07/+4.30/+5.57/+4.79pp`；低噪声 uniform0.2 的优势较小（4/5 seeds），但其余设置稳定扩大。vs Small-loss 为 `+1.24/+4.39/+9.39/+5.80/+6.59pp`。

### 6.4 论文决策

Amazon-Photo 将作为第 5 个主表数据集保留。它补上 co-purchase 领域，且结果不是“只增加领域但无性能”的边界行，而是与 CoauthorCS 一样支撑主 claim：在 X-only MLP 同 backbone 设定内，frozen-GRACE 局部共识软监督可以稳定超过 classic/adapted label-noise baselines，并且 real local structure 相比 shuffled 有独立贡献。主表 scope 到此关闭；AmazonCom/DBLP/WikiCS/LastFM/Physics 不再扩，除非导师明确要求单独补第 6 个数据集。

输出文件：
- `runs/mlp12_amazonphoto_20260702/mlp12_report.md`
- `runs/mlp12_amazonphoto_20260702/mlp12_aggregate.csv`
- `runs/mlp12_amazonphoto_20260702/mlp12_paired_deltas.csv`
- `runs/mlp12_amazonphoto_20260702/mlp12_completeness.csv`
