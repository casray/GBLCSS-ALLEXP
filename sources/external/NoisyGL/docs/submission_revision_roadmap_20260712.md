# 投稿前修订路线图（2026-07-12）

修订基线：`paper_final_zh_v6_table34_completed_20260710.md`

当前工作稿：`paper_final_zh_v12_residual_fixes_20260712.md`

## Phase 1：事实与证据边界修正（已完成）

- [x] 将 `H∈R^{N×256}` 改为 `H∈R^{N×d_h}`，披露 Cora 为 128 维、其余完整噪声谱数据集为 256 维。
- [x] 将单一“约九万参数”改为五数据集实际参数量 48,392–562,182。
- [x] 将“22–133 个球、上限多数未触发”改为正式结果 52–64 个球、19/25 次触及 `B_max=64`。
- [x] 将复杂度改为本文实现可支持的保守形式 `O(B_max·N·d_h + T_2means)`；kNN 二次复杂度限定为当前显式相似度矩阵实现。
- [x] 将“证明、唯一、因类别数导致”等过强措辞改为与消融证据相符的关联性表述。
- [x] 披露五个下游种子共用同一份 H，误差条不覆盖 GRACE 预训练随机性。
- [x] 将 PubMed、Amazon-Computers、Cora_ML、Attributed-Wiki、CiteSeerFull 与 Coauthor-Physics 全部放入正文广度与边界表。
- [x] Table 5 正文只描述实际列出且通过协议审计的方法；CLNode 在验证标签问题修复前不报告。
- [x] 将当前 D 级基线统一改为 `inspired proxy`，明确不参与已发表方法排名。

## Phase 2：基线忠实度决策与重跑（下一步）

### 主表候选

- [x] 忠实实现并完成 5-seed：ANL-CE（`NCE + NNCE + L1`），结果见 `runs/mlp24_faithful_anl_ct_20260712/mlp24_report.md`。
- [x] 忠实实现并完成 5-seed：GMM+CT（置信差轨迹、Mann–Kendall 趋势检验与低损失 GMM 并集）；在当前 full-batch MLP 协议下选择比例趋近 1，结果基本退化为 CE。
- [x] 忠实实现并完成 5-seed：FINE (MLP-adapted)，保留未中心化 SVD 主方向、归一化投影与逐类双 GMM；结果见 `runs/mlp25_fine_fidelity_20260712/mlp25_report.md`。
- [x] DivideMix-lite 不再作为 DivideMix 结果解释；当前结果仅命名为 `GMM loss reweighting (DivideMix-inspired)`。

### 无图推理扩展表候选

- [x] 完成 Graph-MLP protocol port 与 5-seed；保留官方网络、NContrast 和数据集参数，结果见 `runs/mlp27_graphmlp_fidelity_20260712/mlp27_report.md`。
- [x] 完成 FF-G2M common-backbone 忠实重实现与 5-seed；保留官方 LFD/HFD，结果见 `runs/mlp26_ffg2m_fidelity_20260712/mlp26_report.md`。KRD/VQGraph 暂不以现有 proxy 进入正式排名。
- [x] GLNN-style 与 GSDN-inspired 已补精确适配说明：前者为统一协议响应蒸馏，后者为双头/mixup/邻域标签传播/负约束的机制适配；均不声称官方 tuned 复现。

### 通过标准

- 正式方法名必须保留原论文定义方法身份的核心模块。
- 协议必须继续为 percent 0.2/0.1/0.7、train-only noise、5 paired seeds。
- 无法忠实接入的方法删除定量排名行，只在 Related Work 引用。

## Phase 3：引用与评测卫生修正

- [x] 补齐 `[25]`、`[28]`、`[30]–[32]`、`[35]–[36]` 的正式出处与 DOI。
- [x] 修正 SA-MLP、MgCNL、RKD-MLP、Granular Ball Sampling 的出版信息；RKD-MLP 明确标注为 ICLR 2023 OpenReview submission。
- [x] 补 TFR、ProCon、GD² 等 Table 5 方法引用为 `[40]–[42]`。
- [x] GNN-to-MLP teacher 训练循环不再逐 epoch 读取 test accuracy；最佳验证检查点加载后只评一次 test。
- [x] 为五个核心数据集的 H 提供 SHA256，见 `docs/h_embedding_manifest_20260712.md`；后续重新预训练时仍需保存 encoder checkpoint。

## Phase 4：重新生成结果表

- [x] 已对 ANL-CE、GMM+CT、FINE、FF-G2M 和 Graph-MLP 完成五种子矩阵；KRD/VQGraph 代理不进入正式排名。
- [x] 已重新计算每个设置的最强正式对照、paired delta 和方向计数：24/25 均值胜出，23/25 至少 4/5 seeds 同向，见 `runs/phase4_faithful_claims_20260712/phase4_report.md`。
- [x] 已更新摘要、Highlights、贡献和结论中的方法范围与胜出计数。
- [x] 代理方法已从“已发表方法”计数中排除；全谱正式口径为 7 个完整覆盖对照，uniform 0.4 另加 3 个忠实度重跑。

## Phase 5：最终投稿闸门

- [x] 已重新运行数值一致性、协议、H 来源与方法忠实度审计；Tables 3–5 共 575 条种子行对拍通过，单元格 mismatch=0。v12 复核结果见 `runs/submission_integrity_v12_20260712/`。
- [ ] P0 全部关闭后再同步 Elsevier HTML。
- [x] v11 正文表 2、5.3 与结论均保留 PubMed、Amazon-Computers 等负结果和边界解释。
- [x] v12 已统一 PubMed 容量分析的噪声率与基线身份口径，并将教师消融观察更新为当前表 4 的全部六个设置。
