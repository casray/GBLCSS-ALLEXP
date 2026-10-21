# 执行计划：数据集广度扩展 + 噪声谱补满（Phase 28-31）

日期：2026-06-16
执行者：Codex
前置状态：SoftDist [X||H] 论文实验闭环完整，claim 已校准为「受控粒球增量 + 简洁 + 分析」诚实框架。当前只有 Cora/CiteSeer 上球有正增量（PubMed 输 GCN+CE、Amazon-Photo 球贡献消失）。审稿门槛（调研确认）：2025 投稿需 ≥5 数据集、含 Cora/CiteSeer/PubMed、覆盖不同规模+同质性。本计划补数据集广度与噪声谱。**不改主方法机制**。

---

## 0. 环境与红线（沿用）

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- GRACE 预训练在 `gb_graph_aware_minival_6_1/`（产出 H），导出到 `checkpoints/embeddings_v2/<Name>/grace/H.pt`，格式与现有 H 一致（N×d，对齐 NoisyGL 节点序——**关键：节点序必须与 NoisyGL dataloader 一致,否则 H 与图错位**）
- NoisyGL 显式：`--split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7`
- 红线：anchor 机制不动；clean label 只进诊断；SSL 预训练 label-free（只用 X,A，不碰 y/train_mask）；超参用官方默认不调；research-record 报告；测试全绿
- **新数据集注意**：6 个未跑数据集首次运行需 PyG 联网自动下载到 `data/`；下载是一次性的，真正成本在 GRACE 预训练（DBLP/Computers 上万节点，较慢）

### 当前数据集状态

| 数据集 | 节点 | 类 | 同质性 | 状态 |
|---|---:|---:|---:|---|
| Cora / CiteSeer | 2.7k/3.3k | 7/6 | 0.81/0.74 | 已完成，主战场 |
| PubMed | 19.7k | 3 | 0.80 | 已完成，输 GCN+CE（边界）|
| Amazon-Photo | 7.6k | 8 | 0.83 | 已完成，球贡献消失（边界）|

---

## Phase 28：DBLP 探路（先做，单数据集完整跑通,决定后续投入）

### 为什么先做 DBLP

DBLP 同质性最高（0.83）、NoisyGL 第一梯队、NRGNN 原文用它——**最该赢的新数据集**。先把它完整跑通（下载→GRACE→方法+baseline），验证我们方法在新同质图上到底有没有正增量。这是 go/no-go 探针：DBLP 上球有正增量则扩展，没有则论文叙事需进一步收缩。

### 步骤

1. 触发下载：`single_exp.py --dataset dblp --method gcn ...` 跑一次（PyG 自动下载 `CitationFull(name='dblp')`）
2. GRACE label-free 预训练 DBLP，导出 `embeddings_v2/DBLP/grace/H.pt`；用官方/PubMed 同档 GRACE 配置（节点数接近 PubMed，复用其 epochs/lr/drop），**预注册：复用 PubMed GRACE 超参,执行前写进报告**
3. **H-图对齐 sanity check**（必做）：验证 H 的行序与 NoisyGL dataloader 的节点序一致——可用一个已知数据集（如 Cora，已有对齐 H）跑一遍对齐脚本确认逻辑，再应用到 DBLP
4. 跑 5 方法 × 5 seeds × uniform 0.4：`softdist_xh`、`softdist_xh_shuffled`、`gcn_xh_ce`、`gcn`、`nrgnn`
5. 必报球诊断：num_balls、mean_train_per_ball、NCR、CDR、weighted purity（检验球是否塌缩，参考 Amazon-Photo/CCA-SSG 塌缩教训）
6. 输出 `runs/dblp_20260616/`

### 判定（决定 Phase 29 是否展开）

| DBLP 结果 | 后续 |
|---|---|
| `softdist_xh − gcn_xh_ce` ≥ 0 且 real > shuffled | 球有增量 → 展开 Phase 29（补 Computers + 异质探针）|
| 球贡献消失（≈0 或负，类 Amazon-Photo）| 主战场仍只 Cora/CiteSeer → Phase 29 仅补「广度展示」不指望赢，论文叙事收缩到「球在小同质引文图最有效」|
| 球塌缩（num_balls < 8）| 记录 H 几何不适配，与 §6 可分性≠可聚类性呼应 |

---

## Phase 29：同质广度 + 异质边界（Phase 28 后，按其结果决定规模）

### 29a Amazon-Computers（同质，同源扩展）

- 与已跑的 Amazon-Photo 同源（PyG `Amazon('Computers')`），审稿人熟悉
- 下载 → GRACE 预训练（复用 Amazon-Photo GRACE 配置，预注册）→ `embeddings_v2/AmazonComputers/grace/H.pt`
- 5 方法 × 5 seeds × uniform 0.4 + 球诊断
- 预期：可能像 Amazon-Photo 一样球贡献弱（共购图特性），如实记录

### 29b Flickr（异质边界探针，**预期输,作为 limitation**）

- 选 Flickr 而非 roman-empire：规模小（7.6k）预训练快、NoisyGL 官方异质集、审稿人认；roman-empire 太极端（0.05）可能连有意义的 GRACE H 都出不来
- 下载（PyG `AttributedGraphDataset('Flickr')`）→ GRACE 预训练 → H
- 5 方法 × 5 seeds × uniform 0.4 + 球诊断
- **判定与叙事**：异质图上球塌缩/负增益是**预期内的诚实边界,不是失败**。它实证了 §6「GRACE 几何对球可分性是必要条件」——异质图 GRACE 同质假设失效→球塌缩。报告必须把这个结果定位为「方法适用边界的机制验证」,配球诊断（塌缩证据）+ 与同质图球质量对比

### 判定

- 29a/29b 结果无论正负都进论文：同质图进主表（广度），异质图进 limitation（边界 + 机制解释）
- 目标数据集总数达到 7（6 同质 + 1 异质），满足审稿门槛

---

## Phase 30：噪声谱补满（与 28/29 并行，仅在主数据集上）

### 原则

「噪声谱完整性」只需在主数据集（Cora/CiteSeer）展示，新数据集跑标准 uniform 0.4 即可（已含在 28/29），避免 run 数爆炸。

### 补的缺口

1. **pair 0.6 / instance 0.6**（当前 pair/instance 只到 0.4）：Cora/CiteSeer × {pair, instance} × 0.6 × 5 seeds，方法 = softdist_xh/shuffled/gcn_xh_ce/gcn/nrgnn
2. **random 噪声类型**（可选，NoisyGL 支持但我们没碰）：Cora/CiteSeer × random × {0.2,0.4} × 5 seeds——若时间紧可跳过,uniform/pair/instance 三类已是事实标配
3. 注意 pair 0.6 接近标签语义反转上限（>50% 多数类翻转），所有方法预期明显掉，如实记录;instance 0.6 同理核对 actual_noise_rate
4. 输出 `runs/noise_spectrum_fill_20260616/`，并入现有噪声谱表

### 判定

- 无硬门槛，robustness 广度补全。预期延续「中高噪声球增量更强,但极高噪声 offline 管线退化」的既有结论

---

## Phase 31（可选）：大图可扩展性 ogbn-arxiv

### 说明

调研指出 2025 论文越来越期待一个大图（ogbn-arxiv,16.9万节点）证明可扩展性。**仅当 28-30 完成且时间/算力富余才做**——ogbn-arxiv 规模远超现有数据集,GRACE 预训练和 NRGNN 可能很慢甚至 OOM。

- 若做：下载 → GRACE 预训练（大图,监控显存,必要时分批）→ H → softdist_xh/gcn_xh_ce/gcn/nrgnn × 3 seeds（大图 3 seeds 可接受）× uniform 0.4
- 跑不动（OOM/超时）→ 记录为「当前实现的规模边界」,论文 limitation 诚实写明,不强求
- 价值：即便只跑通 softdist_xh + gcn baseline，也能回应「可扩展性」质疑

---

## 执行顺序与依赖

```
Phase 28（DBLP 探路，go/no-go）──┬─球有增量─► Phase 29（Computers + Flickr 全展开）
                                 └─球消失───► Phase 29 缩为广度展示
Phase 30（噪声补满,主数据集）── 与 28/29 并行
Phase 31（ogbn-arxiv,可选）──── 全部完成且富余才做
```

- **Phase 28 是 go/no-go 闸门**，先单独跑通再决定 29 规模，不一次性 all-in
- 每个新数据集的球诊断都必报（塌缩检测）
- 任一数据集失败/边界都不阻塞,无论正负都有论文位置（主表广度 or limitation 边界）

## 论文叙事出口

| 整体结果 | 论文定位 |
|---|---|
| DBLP/Computers 球有正增量 | 「球在多个同质引文/共购图上有受控增量」——广度 claim 立住,主战场扩大 |
| 仅 Cora/CiteSeer 有增量,其余边界 | 「球软目标在小-中规模同质引文图最有效」+ 明确同质性适用边界（C3 延伸）——诚实且有机制解释 |
| 异质图（Flickr）球塌缩 | 实证 GRACE 几何对球可分性的必要性,是 C3 的最强证据,不是失败 |

## 完成后

回填 `SoftDist_XH_论文初稿雏形.html`：主表（§3.1）加同质数据集行；新增「数据集广度表」（全部 7+ 数据集 × uniform 0.4 横向对比 + 球诊断）；§6 加异质图球塌缩作为可分性≠可聚类性的实证；§8 泛化章节扩充；§10 缺失清单勾掉数据集广度。更新经验文档与记忆。

---

## 执行记录（2026-06-16）

### Phase 28：DBLP 探路

已完成 DBLP 下载、GRACE label-free 预训练、H 形状检查、5-seed 球诊断和 seed0 下游 smoke。

预训练说明：计划中写了“复用 PubMed GRACE 超参”的 fallback，但上游 GRACE `config.yaml` 已经提供 DBLP 官方配置，因此实际使用 DBLP 官方 GRACE 默认配置，不做调参。输出 H 为 `checkpoints/embeddings_v2/DBLP/grace/H.pt`，形状 `17716 x 256`，row L2 norm 约为 1，节点数与 NoisyGL DBLP 对齐。

球诊断触发 no-go：

| 指标 | 结果 |
|---|---:|
| num_balls | 1.0000 ± 0.0000 |
| mean_ball_size | 17716.0000 ± 0.0000 |
| true_all_purity_weighted | 0.4471 ± 0.0000 |
| soft_pseudo_ncr | 0.4446 ± 0.0157 |
| soft_pseudo_cdr | 0.5545 ± 0.0082 |

seed0 下游 smoke：

| 方法 | DBLP seed0 |
|---|---:|
| GCN + CE | 0.8160 |
| GCN [X||H] | 0.8300 |
| SoftDist [X||H] | 0.8307 |
| SoftDist shuffled | 0.8307 |
| NRGNN | 0.8136 |
| NRGNN [X||H] | 0.8340 |

结论：DBLP 不能作为第三个有效成功数据集。SoftDist 与 shuffled 完全相同，说明真实球结构没有独立贡献；DBLP 的收益来自 H 特征而不是粒球。

报告见 `runs/dblp_20260616/dblp_phase28_report.md`。

Phase 32 复查更新：默认参数下的 `num_balls=1` 不能再直接解释为“DBLP H 完全不可聚类”。大图球构造参数扫描显示，DBLP 在 `max_balls=256/min_ball_size=32/split_improvement=0.001` 下可得到 256 个球，`true_all_purity_weighted=0.8377`。但下游验证仍失败：scaled SoftDist / GCN [X||H] / shuffled seed0 分别为 0.8300 / 0.8300 / 0.8383。更准确的边界是：大图默认球构造参数不适配可以修复，但修复后的球标签校正仍没有独立下游增量。

Phase 32 报告见 `runs/bigball_downstream_20260616/bigball_phase32_report.md`。

Phase 33 复查更新：大图 soft-target 形态失败不等于球信号完全无用。条件正交诊断显示，DBLP/PubMed/Amazon-Photo 的 `ball_auc_in_gcn_clean50` 分别为 0.6955 / 0.6633 / 0.9483，均超过 0.6 门槛。离线 co-teaching 交集（keep 0.6）可把三者 train 噪声率从约 40% 降到 5.7% / 6.9% / 6.4%，同时保留 53.8% / 52.5% / 43.5% 样本。因此数据集广度线不应彻底关闭；只是原 SoftDist soft-target 路线不能救大图，后续若继续应走“大图 clean-sample selection”新算法路线。

Phase 33 报告见 `runs/biggraph_orthogonality_diag_20260616/biggraph_orthogonality_report.md`。

### 后续决策

- Phase 29 不建议按“争取成功数据集”的目标全展开；若执行，只应定位为边界展示。
- Phase 30 噪声谱补满与数据集广度正交，可单独在 Cora/CiteSeer 上继续。
- Phase 31 暂缓。
