# 执行计划：投稿前最终收尾（Phase 21-23，P1/P2）

日期：2026-06-15
执行者：Codex
前置状态：机制探索已穷尽收官（机制×7、gate×3、SSL方法×4、SSL目标×1、数据集×3、backbone×2、噪声类型×3、覆盖率×4，全部 resolved）。主方法 SoftDist [X||H] 冻结。论文初稿 `SoftDist_XH_论文初稿雏形.html` 主线完整。本计划只补三项纯收尾（不影响任何 claim），完成后论文可投。**不改机制**。

---

## 0. 环境与红线（沿用）

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- NoisyGL 必须显式：`--split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7`
- GRACE H 锚点：`gb_graph_aware_minival_6_1/checkpoints/embeddings/{Cora,CiteSeer}/ratio_train0p2_val0p1_split0_pretrain0/H.pt`
- 红线：anchor 机制不动；clean label 只进诊断；report 用 research-record 格式；测试全绿 `py -3 -m unittest discover -s tests`

---

## Phase 21：球数超参敏感性（P1，证明 64 是稳健默认而非精调）

### 目的

α 公式无超参，但 `max_balls=64` / `min_ball_size=32` 是默认值。审稿人会问是否 cherry-pick。证明在合理范围内性能稳健。

### 运行

- 变量：`max_balls ∈ {16, 32, 64, 128}`，`min_ball_size` 相应设为 `total_nodes/max_balls` 量级的近似（或固定 32，二选一但双数据集同规则，报告写明）
- 方法：`frozengrace_gb_gcn_softdist_xh` + `softdist_xh_shuffled`（对照）
- {cora, citeseer} × 4 个 max_balls × 5 seeds × uniform 0.4
- 每点必报：实际球数、mean_train_per_ball、NCR、CDR、test acc
- 输出 `runs/ball_count_sensitivity_20260615/`

### 报告与判定

- 主曲线：test acc vs max_balls，双数据集，叠加 shuffled
- 无硬门槛。预期：64 附近平台，极端值（16 太粗/128 太细票稀）略降但不崩
- 论文用法：一句话 + 一张敏感性曲线，证明 64 落在稳健平台区

---

## Phase 22：两张分析可视化图（P2，数据已全部就绪，无需新训练）

### 22a — CDR 空间分布图

- 数据源：`runs/nodewise_diag_20260612/distance_bins.csv`（Phase 12 已产出，按到球心距离分桶的 CDR/NCR）
- 产出：折线/柱状图，x=到球心距离分位，y=CDR（和 NCR 叠加），双数据集
- 论文论点：直接可视化「干净标签误伤集中在球边缘」，支撑 §7 软校正必要性的论证
- 脚本：`scripts/plot_cdr_distance.py`，读 CSV 出 PNG/PDF 到 `runs/figures_20260615/`

### 22b — 表示几何 t-SNE 对比

- 数据源：`checkpoints/embeddings_v2/Cora/{grace,ccassg}/H.pt`（已有）；如 grace 锚点 H 在旧路径，用旧路径的 Cora H
- 产出：并排 t-SNE，左 GRACE（应见清晰簇）、右 CCA-SSG（应见弥散/塌缩），点按真实类着色（仅可视化用标签，不进训练）
- 论文论点：§6「可分性≠可聚类性」最直观的图证
- 脚本：`scripts/plot_tsne_embeddings.py`，固定 perplexity/seed 保证可复现，出图到 `runs/figures_20260615/`

### 判定

- 无门槛，纯出图。若 t-SNE 视觉上 CCA-SSG 并非明显弥散（可能高维 t-SNE 投影掩盖塌缩），改用「每类在 H 空间的平均成对距离 / 球分配熵」的定量小表补充，不强行美化

---

## Phase 23：TSS CiteSeer 补跑（P2，补全主表唯一空格）

### 背景

Phase 14 中 TSS 在 CiteSeer 因 ROCm `HIPBLAS_STATUS_ALLOC_FAILED` 五个 seed 全失败，主表 Table 1 缺一格。

### 运行

- 先试 **CPU** 跑 TSS CiteSeer × 5 seeds（`--device cpu`），TSS 不是计时对象，慢可接受
- CPU 也失败 → 试减小相关 batch/采样参数（仅 TSS 自身配置，记录改动），仍失败则在主表标注「OOM/不适配 ROCm，CPU 亦未完成」并保留空格 + 脚注
- 输出 `runs/tss_citeseer_completion_20260615/`

### 判定

- 跑通 → 回填主表 Table 1 的 CiteSeer TSS 格
- 跑不通 → 主表保留诚实脚注（不是隐瞒，是如实记录环境限制）

---

## 执行顺序

```
Phase 22（出图，最快，数据已有） → 先做，立刻可用
Phase 23（TSS 补跑，CPU 慢但独立） → 并行起跑
Phase 21（球数敏感性，40 个 run） → 算力主体
```

## 完成后

1. 把 Phase 21 敏感性曲线、22 两张图、23 的 TSS 格回填进 `SoftDist_XH_论文初稿雏形.html`（§4 消融加敏感性、§6 加 t-SNE、§7 加 CDR 图、§3.1 主表补 TSS）
2. 更新 §10 缺失实验清单，勾掉 P1/P2 已补项
3. 此后论文实验部分视为完整，进入正文写作
