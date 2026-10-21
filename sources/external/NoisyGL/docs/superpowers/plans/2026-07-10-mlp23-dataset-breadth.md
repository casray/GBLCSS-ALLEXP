# MLP-23 Dataset Breadth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 `Cora_ML → Attributed-Wiki → Coauthor-Physics` 顺序筛选最多三个新数据集，并把通过预注册闸门的数据集补齐后加入论文。

**Architecture:** NoisyGL 与上游 GRACE 共享明确的数据集别名和节点顺序；每个候选先导出 label-free frozen H，再复用现有 LCSS/MLP baseline runner。汇总器按相同 split、noise scope、seed 和方法集合做配对判定，只有 pass 数据集进入全量噪声谱和论文正文。

**Tech Stack:** Python 3.12、PyTorch 2.9 ROCm、PyG、pytest、NoisyGL YAML configs、CSV/Markdown 汇总。

---

### Task 1: Cora_ML 数据接入

**Files:**
- Modify: `utils/dataloader.py`
- Modify: `single_exp.py`
- Modify: `total_exp.py`
- Modify: `../gb_graph_aware_minival_6_1/src/noisy_label_gb_grace/data.py`
- Modify: `../gb_graph_aware_minival_6_1/scripts/pretrain_ssl_candidates.py`
- Create: `tests/test_mlp23_dataset_breadth.py`

- [ ] 先写 loader/alias/fallback 失败测试并确认因 `coraml` 未注册而失败。
- [ ] 最小实现 `coraml -> CitationFull(name="cora_ml")`，上游固定使用 Cora GRACE 配置。
- [ ] 验证 NoisyGL 与上游 loader 都得到 2995 节点、7 类，且节点标签逐元素一致。

### Task 2: Cora_ML 配置与 H

**Files:**
- Create: `config/_dataset/coraml.yaml`
- Create: `config/frozengrace_gb_mlp_softdist_beta05/frozengrace_gb_mlp_softdist_beta05_coraml.yaml`
- Create: `config/frozengrace_gb_mlp_softdist_beta05_shuffled/frozengrace_gb_mlp_softdist_beta05_shuffled_coraml.yaml`
- Create: baseline configs under `config/mlp_*/` for Cora_ML

- [ ] 复制 Cora 主协议配置，仅替换数据集名与 H 路径，不改算法超参。
- [ ] 用 ROCm 跑 GRACE 1 epoch smoke，确认有限 loss。
- [ ] 完成固定 Cora 配置的 GRACE 预训练，验证 `H.shape == (2995, 128)`、全有限、行范数约 1。

### Task 3: Cora_ML 正式闸门

**Files:**
- Create: `runs/mlp23_dataset_breadth_20260710/`
- Create: `scripts/summarize_mlp23_dataset_breadth_20260710.py`

- [ ] 跑 LCSS/shuffled/MLP+CE/JoCoR seed0 排雷。
- [ ] 通过排雷后跑 13 方法×5 seeds 的 uniform 0.4。
- [ ] 汇总 mean±std、paired delta、完整性和泄漏审计，判定 pass/boundary/fail。

### Task 4: 逐候选继续与全量扩展

**Files:**
- Modify/Create: 对应 Attributed-Wiki 或 Coauthor-Physics 的 loader/config/test 文件
- Modify: `scripts/summarize_mlp23_dataset_breadth_20260710.py`

- [ ] Cora_ML 完成判定后，按设计顺序对下一个候选重复 Task 1-3。
- [ ] 对每个 pass 数据集补 4 个剩余噪声设置与 Table B。
- [ ] 对符合设计文档次级全谱资格的数据集，在查看其余噪声结果前冻结判据并补 4 个剩余设置；仅在满足 `spectrum-pass` 时纳入核心实验表。
- [ ] 最多纳入三个新数据集；失败候选写入适用域附录。

### Task 5: 论文与上下文同步

**Files:**
- Create: `docs/paper_final_zh_v5_mlp23_dataset_breadth_20260710.md`
- Modify: `docs/99_session_handoff.md`

- [ ] 从 v4 复制出 v5，只加入有完整 5-seed 证据的 pass 数据集。
- [ ] 更新摘要、数据集段、主表、完整噪声表和限制；boundary/fail 仅进适用域。
- [ ] 运行汇总完整性检查、pytest 和论文占位符搜索，保存最终报告。
