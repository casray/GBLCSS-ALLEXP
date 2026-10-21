# Claim–Evidence Ledger

审计对象：`paper_final_zh_v31_tiesafe_nearestwork_20260728.md`。每条 claim 均收缩到现有证据可直接支持的最窄边界；所有数值比较只在对应数据集、噪声、划分、seed 与正式基线集合内成立。带“历史批次”标记的条目只保留证据谱系，不进入当前 tie-safe 主排名。

### C-001
- Section: §1 引言，“研究问题与方法定位”（v29 L27–43）
- Narrowest defensible claim: GB-LCSS 研究的是训练标签受噪、推理时只允许使用节点属性的节点分类，并用粒球局部共识构造软监督，而非把同球关系当作硬类别标签。
- Claim type: SCOPE
- Evidence files: `paper_final_zh_v29_human_style_20260727.md`; `experiments_results_conclusion_audit_20260727.md`
- Evidence locator: v29 L27–35 的问题定义与方法概述；L39–43 的贡献边界；audit 中“方法角色/claim boundary”条目。
- Protocol: N/A；这是方法问题定义，不是实验性能结论；实验边界另由 C-002、C-020 限定。
- Status: SUPPORTED
- Allowed interpretation: 粒球在本文中是局部关系与置信度先验，产生软目标。
- Forbidden expansion: 不得写成同球节点天然同类、粒球是真实标签或硬正负关系。
- Manuscript action: KEEP

### C-002
- Section: §4.1“数据集与评测范围”+ §4.2“标签噪声与数据划分”+ §4.4“统计报告与复现信息”（v29 L357–397）
- Narrowest defensible claim: 正式实验只改写训练标签，验证/测试标签不改写；同一 setting 的方法复用相同 split/noise realization，并以 seed 0–4 做配对比较。
- Claim type: SCOPE
- Evidence files: `latest_baselines_trainonly_20260622_continuation/formal_main_raw.csv`; `latest_baselines_trainonly_20260622_continuation/protocol_audit_report.md`; `../../external/NoisyGL/runs/mlp36_review_defense_20260715/primary_statistics_raw.csv`
- Evidence locator: `formal_main_raw.csv` 过滤 `noise_scope=train_only AND modified_val_count=0 AND modified_test_count=0`，按 `(dataset,noise_type,noise_rate,seed)` 核对 `split_hash,noise_hash`；`primary_statistics_raw.csv` 按同一键检查四个统计比较方法各有 seed `{0,1,2,3,4}`。
- Protocol: Cora/CiteSeer/Coauthor-CS 由 formal raw 直接审计，扩展主表由 source lineage 审计；train/valid/test=20%/10%/70%；train-only label noise；5 paired seeds；同 setting 同 baseline protocol。
- Status: SUPPORTED
- Allowed interpretation: 满足共同键的正式表内结果可以做 paired-seed 比较。
- Forbidden expansion: 不得混入历史 global-noise、不同划分、不同验证协议或不重合 seed 的结果。
- Manuscript action: KEEP

### C-003
- Section: §5.1“与标签噪声学习策略的比较”（v31 Table 1）
- Narrowest defensible claim: 在当前并列安全 GB-LCSS 与七个同骨干标签噪声基线的共同 25-setting 网格中，GB-LCSS 在 23/25 个 setting 取得最高均值，相对每个 setting 最强通用基线的平均差为 +2.8096 pp；两个例外均为 uniform 0.2，分别是 Coauthor-CS 相对 JoCoR −0.50 pp、CoraFull 相对 JoCoR −0.12 pp。三个在构造参数冻结后完成全噪声谱评测的数据集对应 13/15。
- Claim type: PERFORMANCE
- Evidence files: `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/raw_combined.csv`; `../../external/NoisyGL/runs/v31_baseline_refresh_20260728/baselines_combined_raw.csv`
- Evidence locator: 从 tie-safe v2 读取 GB-LCSS、CE、JoCoR 与随机分组，从 baseline refresh 读取 Small-loss、Co-teaching、SCE、ELR 与 LogitClip；按 `(dataset,noise_type,noise_rate,seed)` 核对共同键后，在七个正式同骨干基线中逐 setting 比较五种子均值。
- Protocol: 5 datasets；每个数据集的 5 settings 为 `{uniform 0.2, uniform 0.4, uniform 0.6, pair 0.4, instance 0.4}`；20/10/70 split；5 seeds；7 个正式同身份标签噪声基线。
- Status: SUPPORTED
- Allowed interpretation: 可报告当前共同网格中的 23/25 与冻结参数子集的 13/15 均值排名，并明确两个例外及其小幅差值。
- Forbidden expansion: 不得将 23/25 外推为未测数据集、未测噪声或全部现有方法上的 SOTA；均值第一不自动等于统计显著。
- Manuscript action: KEEP

### C-004
- Section: §5.1“当前 tie-safe 主比较”
- Narrowest defensible claim: 在 v2 的 25 个主 settings 上，GB-LCSS 相对 CE、JoCoR、shuffled-ball 的平均配对差分别为 +7.251731、+2.963838、+7.840969 pp；三个 25 项族内 Holm 校正后分别有 24、8、25 项拒绝零假设，把 75 项合并为单一族后分别保留 20、8、22 项，共 7 项失去显著性。
- Claim type: PERFORMANCE
- Evidence files: `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/raw_combined.csv`; `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/aggregate.csv`; `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/paired_deltas.csv`; `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/paired_tests.csv`; `paper_pipeline/04_experiments/tie_safe_full_grid_experiment_report.md`
- Evidence locator: `paired_deltas.csv` 按 `comparison` 聚合 seed-level `delta_pp`；`paired_tests.csv` 分别按 25 项 family 与 75 项 global Holm 列计数；`raw_combined.csv` 回查四方法、五数据集、五噪声设置、五 seed 的共同键。
- Protocol: 5 main datasets；25 settings；train-only label noise；固定 split；5 paired seeds；CE、JoCoR、shuffled-ball。
- Status: SUPPORTED
- Allowed interpretation: 可逐字报告三组平均配对差、family/global Holm 计数和方向计数；这些结论只针对 v2 的四方法共同协议。
- Forbidden expansion: 不得把 JoCoR 的 8/25 写成普遍显著，也不得把跨 setting 平均差解释成每个 setting 的固定增益。
- Manuscript action: KEEP

### C-005
- Section: §5.1“与标签噪声学习策略的比较”（v31 Table 2）
- Narrowest defensible claim: 扩展到 11 个数据集后收益明显依赖数据；Cora_ML、CiteSeerFull 与 Coauthor-Physics 相对最强正式基线的均值差分别为 +0.33、+0.08 与 +0.02 pp，而 PubMed、Amazon-Computers 与 Attributed-Wiki 分别为 −5.81、−0.69 与 −2.58 pp；六个数据集上真实分组均优于随机分组。
- Claim type: SCOPE
- Evidence files: `../../external/NoisyGL/runs/v31_tiesafe_affected_20260728/breadth_uniform0p4.csv`; `../../external/NoisyGL/runs/mlp_maintable_20260701/mlp_maintable_combined_raw.csv`; `../../external/NoisyGL/runs/mlp21_dataset_replacement_20260710/mlp21_raw.csv`; `../../external/NoisyGL/runs/mlp23_dataset_breadth_20260710/mlp23_raw.csv`
- Evidence locator: 本轮 `breadth_uniform0p4.csv` 提供六个数据集的并列安全 GB-LCSS 与随机分组结果；最强正式基线沿用同协议原始结果，并按 `(dataset,method,seed,noise_type,noise_rate)` 回查。
- Protocol: 11-dataset breadth；Table 2 对应 noise/rate；train-only；固定 split；5 seeds；每个数据集内七个正式同身份基线的最强者。
- Status: SUPPORTED
- Allowed interpretation: 11 数据集证据支持“真实局部结构有贡献，但相对强训练策略的增量不普适”；+0.33、+0.08、+0.02 pp 只能称数值接近。
- Forbidden expansion: 不得把五主数据集的 23/25 外推成 11 数据集普遍 SOTA，不得把微小均值差包装成稳定提升。
- Manuscript action: KEEP

### C-006
- Section: 附录 I“历史资源分层诊断”
- Narrowest defensible claim: 早期 graph-free family 表可用于展示不同训练资源如何形成 X-only 学生，但其最小索引并列规则、协议移植身份和不一致训练资源使其不适合作为当前最近工作排名表。
- Claim type: PERFORMANCE
- Evidence files: `../../external/NoisyGL/runs/submission_integrity_v12_20260712/table3_recomputed.csv`; `../../external/NoisyGL/runs/submission_integrity_v12_20260712/tables3_5_evidence_manifest.csv`; `../../external/NoisyGL/runs/mlp27_graphmlp_fidelity_20260712/mlp27_combined_raw.csv`; `../../external/NoisyGL/runs/mlp26_ffg2m_fidelity_20260712/mlp26_combined_raw.csv`
- Evidence locator: `table3_recomputed.csv` 按 `dataset` 比较正式 graph-free 方法 `mean`；GraphMLP 过滤 `mlp27_combined_raw.csv: method=graphmlp_reimplemented`，FFG2M 过滤 `mlp26_combined_raw.csv: method=ffg2m_reimplemented`，均核对 seed 0–4；lineage 由 manifest 的 Table 3 条目连接。
- Protocol: 5 datasets；uniform 0.4 train-only；20/10/70 split；5 seeds；正式 graph-free 方法组；KRD/VQ 风格代理不进入官方方法排名。
- Status: SUPPORTED
- Allowed interpretation: 只可把该表作为历史资源层级诊断，并逐行标明 official port、reimplementation 或 inspired proxy。
- Forbidden expansion: 不得把该表当作当前并列安全协议下的正式最近工作排名，不得把代理实现当作原方法官方结果。
- Manuscript action: MOVE_TO_APPENDIX

### C-007
- Section: §6“扩展：以共识教师蒸馏弥合精度差距”（v29 L543–563，Table 5）
- Narrowest defensible claim: 六个教师消融 settings 中，向 GCN 教师加入 GRACE 表征 H 与主要增益相关；局部共识目标在部分高噪声 setting 进一步提高均值。
- Claim type: MECHANISM
- Evidence files: `../../external/NoisyGL/runs/submission_integrity_v12_20260712/table4_recomputed.csv`; `../../external/NoisyGL/runs/submission_integrity_v12_20260712/tables3_5_evidence_manifest.csv`
- Evidence locator: `table4_recomputed.csv` 按 `(dataset,noise_rate)` 比较 `method in {mlp_distill_gcn,mlp_distill_gcn_xh,mlp_distill_knn_xh,mlp_distill_softdist_xh,mlp_distill_softdist_xh_shuffled}` 的 `mean,std,n`；该 external Table 4 由 manifest 映射到稿件 Table 5。
- Protocol: Cora/CiteSeer/CoraFull；uniform `{0.4,0.6}` train-only；固定 split；5 seeds；五个教师变体。
- Status: PARTIAL
- Allowed interpretation: H 输入与表内提升相关；共识教师在若干高噪声 setting 有附加增益。
- Forbidden expansion: 消融未识别唯一因果路径；不得声称 H 或粒球共识单独导致全部提升，也不得外推到未测数据集。
- Manuscript action: WEAKEN

### C-008
- Section: §6“扩展：以共识教师蒸馏弥合精度差距”（v29 L564–580，Table 6）
- Narrowest defensible claim: 相对每个数据集表内均值最高的图依赖方法，X-only 蒸馏学生在 Cora、CiteSeer、CoraFull 上分别低 1.16、1.63、4.48 pp，在 Coauthor-CS 上高 0.03 pp（数值接近），在 Amazon-Photo 上比 GCN[X||H] 低 1.38 pp。
- Claim type: PERFORMANCE
- Evidence files: `../../external/NoisyGL/runs/submission_integrity_v12_20260712/table5_recomputed.csv`; `../../external/NoisyGL/runs/submission_integrity_v12_20260712/tables3_5_evidence_manifest.csv`; `latest_baselines_trainonly_20260622_continuation/formal_main_raw.csv`
- Evidence locator: `table5_recomputed.csv` 按 `dataset` 取 `method=mlp_distill_softdist_xh`，并在其余图依赖方法中选择最高 `mean`：Cora/CiteSeer/CoraFull 为 `nrgnn`，Coauthor-CS 为 `nrgnn_xh`，Amazon-Photo 为 `gcn_xh_ce`；逐项计算学生减参照的均值差。external Table 5 由 manifest 映射到稿件 Table 6；Cora、CiteSeer、Coauthor-CS 用 formal raw 的 `(dataset,method,seed)` 交叉核对。
- Protocol: 5 datasets；uniform 0.4 train-only；固定 split；5 seeds；X-only student 与 graph-dependent NRGNN/NRGNN-XH 为跨部署资源参考。
- Status: SUPPORTED
- Allowed interpretation: 可报告相对每个数据集表内最强图依赖参照的明确差值，作为不同部署资源约束下的比较。
- Forbidden expansion: 这不是同资源公平排名；不得声称学生全面优于图方法。
- Manuscript action: KEEP

### C-009
- Section: §5.3.1“局部共识与标签平滑”（v29 L462–475）
- Narrowest defensible claim: LOO 在六数据集均优于随机分组和两种显式平滑；Cora 的 Brier 由 0.4876 降至 0.4503，NCR 提高同时 CDR 由 0.2151 升至 0.3020；LOO 相对默认目标的单数据集精度增量均未通过 Holm 校正。
- Claim type: MECHANISM
- Evidence files: `../../external/NoisyGL/runs/mlp36_review_defense_20260715/mlp36_raw.csv`; `../../external/NoisyGL/runs/mlp36_review_defense_20260715/mlp36_aggregate.csv`; `../../external/NoisyGL/runs/mlp36_review_defense_20260715/mlp36_paired_deltas.csv`
- Evidence locator: `mlp36_raw.csv` 按 `(dataset,method,seed)` 比较 `test,soft_target_brier,soft_pseudo_ncr,soft_pseudo_cdr`，其中 `method in {frozengrace_gb_mlp_softdist_beta05,frozengrace_gb_mlp_softdist_beta05_loo,frozengrace_gb_mlp_softdist_beta05_loo_shuffled,mlp_uniform_smoothing,mlp_prior_smoothing}`；`mlp36_paired_deltas.csv` 过滤 `comparison in {loo_vs_current,loo_vs_loo_shuffled,loo_vs_uniform_smoothing,loo_vs_prior_smoothing}`，并对 `loo_vs_current` 核对六数据集 `holm_p_within_family>=0.05`。
- Protocol: 6 datasets；uniform 0.4 train-only；固定 split；5 paired seeds；默认、LOO、LOO-shuffled、uniform smoothing、prior smoothing。
- Status: SUPPORTED
- Allowed interpretation: 球内其他成员含有非随机监督信息；LOO 暴露 NCR/CDR 的纠错—误伤权衡。
- Forbidden expansion: 不得声称 LOO 必然优于默认目标，不得把 Brier/NCR/CDR 的相关变化当作唯一因果机制。
- Manuscript action: KEEP

### C-010
- Section: §5.3.2“分组形式与有效票数”（v29 L477–481）
- Narrowest defensible claim: uniform 0.4、五 seed 的四数据集诊断中，k=20 的 kNN 均值均高于默认粒球；增大粒球最小规模后，CiteSeer 与 kNN 的汇总差由 2.58 缩至约 0.77 pp。
- Claim type: PERFORMANCE
- Evidence files: `../../external/NoisyGL/runs/mlp11_k_sensitivity_20260702/mlp11_combined_raw.csv`; `../../external/NoisyGL/runs/mlp11_k_sensitivity_20260702/mlp11_knn_curve.csv`; `../../external/NoisyGL/runs/mlp13_ballparam_sensitivity_20260703/mlp13_stageB_raw.csv`; `../../external/NoisyGL/runs/mlp13_ballparam_sensitivity_20260703/mlp13_stageB_summary.csv`
- Evidence locator: `mlp11_knn_curve.csv` 过滤 `k=20`，按 `dataset` 与默认 ball mean 比较；`mlp13_stageB_summary.csv` 过滤 CiteSeer 的 default 与 large-ball config，使用其 `test_mean` 与 kNN k=20 汇总值计算 2.58→0.77 pp；raw 按 `(dataset,config,seed)` 回查。
- Protocol: Cora/CiteSeer/PubMed/Coauthor-CS；uniform 0.4；20% train labels；固定 split；5 seeds；kNN k-grid 与默认/大粒球配置。
- Status: SUPPORTED
- Allowed interpretation: 在该诊断中，默认粒球不总优于调好的 kNN，增大有效邻域可缩小差距。
- Forbidden expansion: 不得声称粒球普遍等价或优于 kNN，不得把跨两个汇总表计算写成单次直接测量。
- Manuscript action: KEEP

### C-011
- Section: §5.3.3“共识空间”（v29 L483–486）
- Narrowest defensible claim: Cora、CiteSeer、PubMed 的 uniform-0.4 五 seed 对照中，GRACE 空间粒球比所测试的邻接传播空间高 8.25、6.86、6.92 pp，三组均为 5/5 seed 正差。
- Claim type: PERFORMANCE
- Evidence files: `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p3_lp_target_uniform0p4_seeds_0_1_2_3_4.csv`; `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p3_lp_target_aggregate.csv`; `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p3_lp_target_paired_deltas.csv`
- Evidence locator: raw 文件提供 `method=lp_mlp_softdist_beta05` 的 seed-level 行；aggregate 按 `dataset` 比较 `method in {frozengrace_gb_mlp_softdist_beta05,lp_mlp_softdist_beta05}` 的 `mean`；paired 文件过滤 `ours=frozengrace_gb_mlp_softdist_beta05 AND baseline=lp_mlp_softdist_beta05`，核对 `mean_delta_pp` 与 `positive_seeds=5`。
- Protocol: Cora/CiteSeer/PubMed；uniform 0.4 train-only；固定 split；5 paired seeds；GRACE-space ball vs 指定 adjacency-propagation control。
- Status: SUPPORTED
- Allowed interpretation: GRACE 空间优于这一具体传播空间对照。
- Forbidden expansion: 不得写成 GRACE 优于所有传播表征、所有自监督编码器或所有邻域定义。
- Manuscript action: KEEP

### C-012
- Section: §5.3.4“学生容量与 PubMed”（v29 L487–492）
- Narrowest defensible claim: PubMed 的五个噪声 settings 中，线性 GB-LCSS 相对每个 setting 最强的 tested linear sample-selector 高 7.4–14.3 pp；因此所测学生容量差异不足以单独解释 PubMed 结果。
- Claim type: PERFORMANCE
- Evidence files: `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p0c_linear_capacity_raw.csv`; `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p4_pubmed_linear_aggregate.csv`; `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p4_pubmed_linear_best_robust_summary.csv`; `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p4_pubmed_linear_paired_deltas.csv`
- Evidence locator: `p0c_linear_capacity_raw.csv` 用于回查 PubMed uniform 0.4 的线性容量对照；`p4_pubmed_linear_aggregate.csv` 按五个 `(noise_type,noise_rate,method)` 取均值；`p4_pubmed_linear_best_robust_summary.csv` 每个 setting 读取 `best_linear_robust_method,best_linear_robust,delta_vs_best_linear_robust_pp`；paired 文件按同一 setting 核对 `mean_delta_pp`，范围 7.389–14.264。
- Protocol: PubMed；五个稿件 noise/rate settings；train-only；固定 split；5 paired seeds；linear GB-LCSS vs 每 setting 最强 tested linear sample-selector。
- Status: SUPPORTED
- Allowed interpretation: 只能说在已测试线性候选集内，容量不是唯一解释。
- Forbidden expansion: 不得写成对固定单一基线均为 7.4–14.3 pp，也不得排除所有优化、维度或容量交互因素。
- Manuscript action: KEEP

### C-013
- Section: §5.3.5“H-access 参照”（v29 L493–507）
- Narrowest defensible claim: H-linear 在 Cora/CiteSeer/PubMed 为 78.61/68.87/83.87，X-only GB-LCSS 为 55.59/53.83/71.90；H-access 是更强资源上界，不是同资源部署比较。
- Claim type: PERFORMANCE
- Evidence files: `../../external/NoisyGL/runs/grace_h_ablation_20260704/grace_h_ablation_uniform0p4_seeds_0_1_2_3_4.csv`; `../../external/NoisyGL/runs/grace_h_ablation_20260704/grace_h_ablation_aggregate.csv`; `../../external/NoisyGL/runs/mlp6_fairness_labelrate_20260701/mlp6_aggregate.csv`
- Evidence locator: H-ablation raw 按 `(dataset,method,seed)` 过滤 `method in {grace_h_linear_ce,grace_h_mlp_ce,mlp_xh_ce}`；`grace_h_ablation_aggregate.csv` 按三数据集读取 `method in {grace_h_linear_ce,frozengrace_gb_mlp_softdist_beta05}` 的 `mean,std`；`mlp6_aggregate.csv` 过滤 `table=fairness,train_percent=0.2` 回查 `mlp_xh_ce` 等 H-access 参照。
- Protocol: Cora/CiteSeer/PubMed；uniform 0.4 train-only；固定 split；5 seeds；H-access classifiers vs X-only student。
- Status: SUPPORTED
- Allowed interpretation: H 含强可分类信息，X-only 约束付出可测性能代价。
- Forbidden expansion: H-access 在推理时使用额外资源，不得进入与 X-only 学生同资源的正式排名。
- Manuscript action: KEEP

### C-014
- Section: §5.3.6“标签率与输入扰动”（v29 L508–513，label-rate）
- Narrowest defensible claim: Cora、CiteSeer 的 5%–65% 标注率实验中，GB-LCSS 优势总体随标注率上升而缩小，但各离散点并非严格单调。
- Claim type: SCOPE
- Evidence files: `../../external/NoisyGL/runs/mlp6_fairness_labelrate_20260701/mlp6_combined_raw.csv`; `../../external/NoisyGL/runs/mlp6_fairness_labelrate_20260701/mlp6_aggregate.csv`; `../../external/NoisyGL/runs/mlp6_fairness_labelrate_20260701/mlp6_paired_deltas.csv`
- Evidence locator: raw/aggregate 按 `(dataset,train_percent,method)` 过滤 `table=labelrate AND train_percent in {0.05,0.10,0.20,0.35,0.50,0.65}`；每个 `(dataset,train_percent)` 在 `method in {mlp_ce,mlp_small_loss,mlp_jocor,mlp_fine}` 中选择最高 `mean`，再由 paired 文件的对应 `comparison` 读取 `mean_delta`。
- Protocol: Cora/CiteSeer；uniform 0.4 train-only；六 label rates；固定 split；5 paired seeds；X-only sample selectors。
- Status: SUPPORTED
- Allowed interpretation: 只可描述两数据集、六标注率上的总体趋势。
- Forbidden expansion: 不得声称所有数据集严格单调，或把趋势写成已证明的机制。
- Manuscript action: KEEP

### C-015
- Section: §5.3.6“标签率与输入扰动”（v29 L508–513，clean-label control）
- Narrowest defensible claim: clean-label 下，GB-LCSS 相对 CE 在 Cora/CiteSeer/PubMed 为 +0.44/+2.33/+0.02 pp，在 Coauthor-CS/Amazon-Photo 为 −0.88/−1.20 pp，因此不是普遍的 clean-label accuracy enhancer。
- Claim type: SCOPE
- Evidence files: `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p2_clean_uniform0p0_seeds_0_1_2_3_4.csv`; `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p2_clean_aggregate.csv`; `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p2_clean_paired_deltas.csv`
- Evidence locator: raw 按 `(dataset,method,seed)` 过滤 `noise_type=uniform,noise_rate=0.0,method in {GB-LCSS,CE}`；aggregate/paired 读取五数据集 `mean_delta`。
- Protocol: 5 datasets；clean labels (`uniform 0.0`)；固定 split；5 paired seeds；GB-LCSS vs CE。
- Status: SUPPORTED
- Allowed interpretation: 该证据把方法定位为标签噪声防御，而不是 clean-label 普遍增益器。
- Forbidden expansion: 不得只报告三个非负数据集并隐藏两个负结果。
- Manuscript action: KEEP

### C-016
- Section: §5.3.6“标签率与输入扰动”（v29 L508–513，feature-noise control）
- Narrowest defensible claim: 所测特征扰动曲线上，Cora 的 GB-LCSS 从无扰动到最强扰动下降约 34.0 pp，CE 下降约 22.8 pp；现有证据不支持特征噪声鲁棒性。
- Claim type: SCOPE
- Evidence files: `../../external/NoisyGL/runs/mlp16_cutvalue_featnoise_20260706/mlp16_feature_noise_raw.csv`; `../../external/NoisyGL/runs/mlp16_cutvalue_featnoise_20260706/mlp16_feature_noise_aggregate.csv`; `../../external/NoisyGL/runs/mlp16_cutvalue_featnoise_20260706/mlp16_feature_noise_deltas.csv`; `../../external/NoisyGL/runs/mlp16_cutvalue_featnoise_20260706/mlp16_feature_noise_slopes.csv`
- Evidence locator: aggregate 过滤 `dataset=cora,method in {frozengrace_gb_mlp_softdist_beta05,mlp_ce}`，以 `alpha=0` 与 `alpha=0.9` 的 `mean` 作差；`mlp16_feature_noise_slopes.csv` 读取同两方法的 `drop_alpha0_to_0p9`，分别约 `0.3379` 与 `0.2282`；raw/deltas 用于 seed-level 与逐 alpha 交叉核对。
- Protocol: Cora 数值 claim；特征扰动 alpha 曲线；训练标签协议保持对应正式设置；固定 split；5 seeds；GB-LCSS vs CE。
- Status: SUPPORTED
- Allowed interpretation: 方法依赖特征及由其形成的表示/局部共识质量。
- Forbidden expansion: 不得声称对结构噪声、特征噪声或联合噪声具有未经测试的鲁棒性。
- Manuscript action: KEEP

### C-017
- Section: §5.3.5“H-access 参照”（v29 L500–507，GRACE seed stability）
- Narrowest defensible claim: 更换 GRACE seed 时，Cora/CiteSeer/CoraFull 最终精度标准差约 0.53/0.07/0.44 pp，分区 ARI 约 0.25–0.37；已测 H-seed 上相对 shuffled/CE/JoCoR 的差值方向均为正。
- Claim type: MECHANISM
- Evidence files: `../../external/NoisyGL/runs/mlp37_grace_seed_variance_20260716/mlp37_raw.csv`; `../../external/NoisyGL/runs/mlp37_grace_seed_variance_20260716/mlp37_variance.csv`; `../../external/NoisyGL/runs/mlp37_grace_seed_variance_20260716/mlp37_assignment_agreement.csv`
- Evidence locator: raw 按 `(dataset,h_seed,seed,method)` 取 `test`；variance 读取三数据集 `h_seed_std,delta_shuffled_min,delta_ce_min,delta_jocor_min`；agreement 按 `(dataset,h_seed_pair)` 读取 `adjusted_rand_index` 的最小/最大值。
- Protocol: Cora/CiteSeer/CoraFull；uniform 0.4 train-only；固定 split/noise；3 H-seeds × 3 downstream seeds；shuffled、CE、JoCoR。
- Status: SUPPORTED
- Allowed interpretation: 在三个数据集和已测 encoder seeds 上，下游性能比具体离散分区更稳定。
- Forbidden expansion: 低 ARI 不证明所有局部统计稳定，也不得外推到任意预训练算法、超参数或数据集。
- Manuscript action: KEEP

### C-018
- Section: §3.6“复杂度与部署开销”+ §5.5“计算与部署开销”（v29 L331–351、L521–539，Table 4）
- Narrowest defensible claim: Table 4 是 seed 0、指定硬件的成本测量；当前 tie-safe GB-LCSS 在 Cora、CiteSeer 和 CoraFull 的下游墙钟分别为 3.58、5.11 和 17.43 秒，其中构球为 1.59、2.51 和 6.08 秒。X-only 学生参数量和前向不依赖边数；GRACE 成本从小图约 0.07–0.13 分钟到 CoraFull 74.48 分钟、峰值显存 12.73 GiB。
- Claim type: EFFICIENCY
- Evidence files: `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/raw_combined.csv`; `../../external/NoisyGL/runs/mlp36_review_defense_20260715/latency_memory_seed0.csv`; `../../external/NoisyGL/runs/mlp36_review_defense_20260715/latency_memory_report.md`; `../../external/NoisyGL/runs/mlp37_grace_seed_variance_20260716/mlp37_grace_costs.csv`
- Evidence locator: tie-safe raw 过滤 `method=frozengrace_gb_mlp_softdist_beta05, noise_type=uniform, noise_rate=0.4, seed=0`，读取 `wall_time_seconds,ball_build_time_seconds,parameter_count`；历史 latency CSV 提供同一 MLP 架构的 `inference_peak_allocated_mb,fullgraph_latency_ms` 及 MLP/GCN 参照；GRACE costs 按 `dataset in {cora,citeseer,corafull}` 读取 `wall_time_minutes,peak_memory_allocated_gib`。
- Protocol: Table 4 datasets；seed 0；报告指定硬件/软件环境；训练与推理计时；baseline 为 MLP-CE、GCN 及稿件所列方法；GRACE 预训练单独计费。
- Status: SUPPORTED
- Allowed interpretation: 可报告该环境中的观测成本，并称 X-only MLP 推理对边数不敏感。
- Forbidden expansion: 不得声称跨平台普遍加速；固定粒球预算不等于无条件的严格线性总训练复杂度。
- Manuscript action: KEEP

### C-019
- Section: §6“扩展：以共识教师蒸馏弥合精度差距”（v29 L543–580）
- Narrowest defensible claim: 蒸馏扩展配置使用接收 X||H 的图教师、`target_beta=0` 的 soft-target 教师目标，再把教师软分布蒸馏给只接收 X 的学生；部署时丢弃图、H、粒球与教师。
- Claim type: MECHANISM
- Evidence files: `paper_final_zh_v29_human_style_20260727.md`; `../../external/NoisyGL/config/mlp_distill_softdist_xh/mlp_distill_softdist_xh_cora.yaml`; `../../external/NoisyGL/runs/submission_integrity_v12_20260712/table4_recomputed.csv`; `../../external/NoisyGL/runs/submission_integrity_v12_20260712/table5_recomputed.csv`; `../../external/NoisyGL/runs/submission_integrity_v12_20260712/tables3_5_evidence_manifest.csv`
- Evidence locator: YAML 键 `gb.target_beta=0.0, teacher_use_xh=true, teacher_target=soft_target`，同目录六数据集配置一致；v29 L268–329 定义教师/学生资源流并限定 student input=X；重算表按 `method=GB-LCSS-distill` 定位结果。
- Protocol: 蒸馏表覆盖稿件对应 5 datasets/教师消融 3 datasets；uniform 0.4（教师消融另含 0.6）；train-only；固定 split；5 seeds；学生 X-only，教师可用 graph+H。
- Status: SUPPORTED
- Allowed interpretation: 可称“训练期图知识转移、推理期 X-only”。
- Forbidden expansion: 不得称整个训练过程 graph-free，也不得称教师完全无图或完整方法 label-free。
- Manuscript action: KEEP

### C-020
- Section: §3.1“问题设定与符号”+ §7“结论”（v29 L81–92、L584–590）
- Narrowest defensible claim: 实验仍是同一张图上的 transductive 节点分类；测试节点在训练期图中可见但测试标签不用于训练，尚未验证全新图或全新节点的 inductive 泛化。
- Claim type: SCOPE
- Evidence files: `paper_final_zh_v29_human_style_20260727.md`; `latest_baselines_trainonly_20260622_continuation/protocol_audit_report.md`
- Evidence locator: v29 L83–92 的资源表/任务边界与 L584–590 的结论限制；protocol audit 的固定 20/10/70 transductive split 条目。
- Protocol: transductive single-graph node classification；train/valid/test=20/10/70；train-only label noise；测试阶段输入 X-only，但训练期图可见。
- Status: SUPPORTED
- Allowed interpretation: “测试时不用图”只描述部署输入，不改变 transductive 数据协议。
- Forbidden expansion: 不得称为 inductive、跨图泛化或 unseen-node generalization。
- Manuscript action: KEEP

### C-021
- Section: §3.3“冻结表征空间中的标签无关粒球构造”+ §3.4“粒球局部共识软监督”（v29 L114–232）
- Narrowest defensible claim: GRACE 预训练和粒球成员关系不读取类别标签；noisy training labels 只在后续软目标中进入，clean labels 仅用于验证、诊断和最终评估。
- Claim type: MECHANISM
- Evidence files: `paper_final_zh_v29_human_style_20260727.md`; `../../external/NoisyGL/runs/mlp36_review_defense_20260715/mlp36_raw.csv`; `latest_baselines_trainonly_20260622_continuation/protocol_audit_report.md`
- Evidence locator: v29 L116–160 的 `H=f_phi(X,A)` 与几何划分、L164–216 的 noisy-label 软目标、L218–232 的 X-only 学生；raw 定位 `ball_mode,target_beta,noise_scope,modified_val_count,modified_test_count`；protocol audit 核对 clean-label 使用边界。
- Protocol: 方法定义适用于正式实验；GRACE/ball construction 阶段 label-free；soft-target training 使用 noisy train labels；clean validation/test 不作为训练目标。
- Status: SUPPORTED
- Allowed interpretation: 可称粒球构造 label-independent 或几何共识先验不读标签。
- Forbidden expansion: 不得把完整 GB-LCSS 训练称为 label-free。
- Manuscript action: KEEP

### C-022
- Section: §1“引言”（v29 L29，连同 §2.3 GSDN 限定句 L57–59）
- Narrowest defensible claim: “训练期利用图抵抗标签噪声、推理期只依赖节点属性”的问题尚未得到系统研究。
- Claim type: NOVELTY
- Evidence files: `paper_final_zh_v29_human_style_20260727.md`; `paper_pipeline/00_evidence/material_manifest.md`; `01_literature_and_research_map.md`
- Evidence locator: PosteL 直接采用图传播构造 soft targets 再训练 MLP，并讨论无图推理；GSDN 在预测层做图结构自蒸馏并报告标签噪声；ALS/SALS 直接研究图标签噪声下的软标签。逐篇机制与资源边界记录见 `nearest_work_verification_notes.md`。
- Protocol: 截止 2026-07-27 的定向最近工作核验；它足以否定宽泛排他性表述，但不是可支持“穷尽全部文献”的系统综述。
- Status: CONTRADICTED
- Allowed interpretation: 只能陈述本文在统一 split、train-only 噪声、仅属性 MLP 骨干和 tie-safe 局部共识审计下评估一种具体实现，并逐项说明与 PosteL、GSDN、ALS、SALS 的差异。
- Forbidden expansion: 不得使用“尚未得到系统研究”“首次”“尚无工作”“首次结合粒球与节点级 GCL”等排他性新颖性表述。
- Manuscript action: WITHDRAW

### C-023
- Section: §5.1“与标签噪声学习策略的比较”（v29 L401–429，pair/instance 对照）
- Narrowest defensible claim: 相对每个 setting 七个正式同身份基线中的最强者，GB-LCSS 在 uniform/pair/instance、rate 0.4 的增益明显数据依赖：CoraFull 为 1.63/2.96/1.91 pp，Coauthor-CS 为 1.23/4.93/1.30 pp，Amazon-Photo 为 2.07/3.63/4.79 pp，Cora 为 3.64/3.30/3.41 pp，CiteSeer 为 3.93/1.45/3.76 pp。
- Claim type: PERFORMANCE
- Evidence files: `../../external/NoisyGL/runs/phase4_faithful_claims_20260712/phase4_strongest_formal_baselines.csv`; `../../external/NoisyGL/runs/phase4_faithful_claims_20260712/phase4_selected_raw.csv`; `../../external/NoisyGL/runs/phase4_faithful_claims_20260712/phase4_paired_deltas.csv`
- Evidence locator: `phase4_strongest_formal_baselines.csv` 按 `(dataset,noise_type,noise_rate)` 过滤 `noise_type in {uniform,pair,instance} AND noise_rate=0.4`，读取 `mean_delta`；`phase4_selected_raw.csv` 按 `(dataset,noise_type,noise_rate,method,seed)` 回查；paired 文件用于 seed-level 方向核对。
- Protocol: 5 main datasets；uniform/pair/instance 0.4 train-only；固定 split；5 paired seeds；每个 setting 七个正式同身份基线的最强者。
- Status: SUPPORTED
- Allowed interpretation: 可作为历史候选批次报告不同噪声类型下的均值增益具有数据依赖性。
- Forbidden expansion: 不得把该历史七基线候选集当作当前 tie-safe 主排名；这些 mean deltas 不是显著性结论；不得声称 pair 或 instance 在所有数据集都带来更大相对收益。
- Manuscript action: RETIRE

### C-024
- Section: §5.3.2“分组形式与有效票数”（v29 L477–481）
- Narrowest defensible claim: Cora 上从默认配置 `mb64_min32_si0.05` 调到较大球配置 `mb128_min64_si0.01` 时，球数 52.4→22、平均有效票数 10.325→24.611、纯度 0.505→0.477，但精度 55.59%→56.56%；k=5 的 kNN 为 53.93%，这些结果与“票数不足会限制多数统计”相容。
- Claim type: MECHANISM
- Evidence files: `../../external/NoisyGL/runs/mlp13_ballparam_sensitivity_20260703/mlp13_stageB_summary.csv`; `../../external/NoisyGL/runs/mlp13_ballparam_sensitivity_20260703/mlp13_stageB_raw.csv`; `../../external/NoisyGL/runs/mlp13_ballparam_sensitivity_20260703/mlp13_stageB_paired_deltas.csv`; `../../external/NoisyGL/runs/mlp11_k_sensitivity_20260702/mlp11_knn_curve.csv`; `../../external/NoisyGL/runs/mlp11_k_sensitivity_20260702/mlp11_combined_raw.csv`
- Evidence locator: `mlp13_stageB_summary.csv` 过滤 `dataset=cora` 与 `config_id in {mb64_min32_si0.05,mb128_min64_si0.01}`，读取 `num_balls_mean,mean_train_per_ball_mean,noisy_train_purity_weighted_mean,test_mean`；paired 文件核对两配置；`mlp11_knn_curve.csv`/raw 过滤 `dataset=cora,k=5`，读取 `test_mean=0.539314,test_std=0.008807`。
- Protocol: Cora；uniform 0.4 train-only；20% train labels；固定 split；5 seeds；两个 ball configs 与 kNN k=5 诊断。
- Status: PARTIAL
- Allowed interpretation: 可称这些受限观测与“有效票数比纯度单调性更关键”相容。
- Forbidden expansion: 不得写成已证明票数是唯一因果因素；当前未定位 kmeans raw，不得保留“kmeans 与粒球相近”的实证 claim；CiteSeer 2.58→0.77 是跨汇总组合计算。
- Manuscript action: WEAKEN

### C-025
- Section: §5.4“冻结无监督对比表征的经验选择”（v29 L514–518）
- Narrowest defensible claim: Cora seed 0 诊断中，GRACE 形成 52 个球、纯度 0.7873；CCA-SSG 与 GraphMAE 均退化为 1 个球、纯度 0.3021，尽管其 clean/noisy linear-probe 精度高于 GRACE；这只支持“线性可分性不保证在当前构造参数下的局部可聚类性”。
- Claim type: MECHANISM
- Evidence files: `../../external/NoisyGL/runs/ssl_swap_seed0_20260612/ssl_ball_diagnostics_seed0.csv`; `../../external/NoisyGL/runs/ssl_swap_seed0_20260612/ssl_swap_seed0_report_20260612.md`; `../../external/NoisyGL/runs/ssl_probe_20260612/probe_aggregate.csv`
- Evidence locator: `ssl_ball_diagnostics_seed0.csv` 过滤 `dataset=cora,ssl_method in {grace,ccassg,graphmae}`，读取 `num_balls,true_all_purity_weighted`：GRACE `52/.7873`、CCA-SSG `1/.3021`、GraphMAE `1/.3021`；report L53–62；`probe_aggregate.csv` 过滤 `dataset=cora,method in {grace,ccassg,graphmae},probe_label in {clean,noisy}`，读取 `test_acc_mean`：GRACE `.8294/.7949`、CCA `.8566/.8237`、GraphMAE `.8396/.8035`。
- Protocol: Cora；encoder seed 0；相同 ball construction parameters；clean/noisy linear-probe diagnostics；GRACE vs CCA-SSG vs GraphMAE；不是主性能 protocol。
- Status: PARTIAL
- Allowed interpretation: 仅可作为 Cora seed 0 的表征几何诊断，支持线性探针与局部聚类性可以分离。
- Forbidden expansion: CiteSeer 的 GraphMAE 实为 64 球；当前未定位 noisy-CE GRACE 52→1、纯度 0.79→0.30、−12.6 pp 或 MLP-hidden→1 的 raw，不得按现稿完整保留这些数值/对象，也不得声称 GRACE 普遍最优。
- Manuscript action: WEAKEN

### C-026
- Section: §4.4“统计报告与复现信息”+ §5.1“当前 tie-safe 主比较”
- Narrowest defensible claim: v2 主网格包含 4 方法 × 5 数据集 × 5 噪声设置 × 5 seeds 共 500 个唯一实验键，四种方法各 125 行，失败数为 0；125 个协议组全部通过共同 split/noise/protocol 审计。
- Claim type: REPRODUCIBILITY
- Evidence files: `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/raw_combined.csv`; `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/completeness_audit.csv`; `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/protocol_audit.csv`; `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/immutable_manifest.json`
- Evidence locator: 对 `raw_combined.csv` 的 `(method,dataset,noise_type,noise_rate,seed)` 去重并分方法计数；`completeness_audit.csv` 核对期望键和失败；`protocol_audit.csv` 核对 125 组均为 PASS；immutable manifest 回查文件 SHA-256。
- Protocol: 5 datasets；5 settings；5 paired seeds；方法为 GB-LCSS、CE、JoCoR、shuffled-ball；train-only noise；相同 20/10/70 split 和评估流程。
- Status: SUPPORTED
- Allowed interpretation: 可称该四方法主网格在预定键空间内完整且协议审计通过。
- Forbidden expansion: 完整性不等于实现忠实性，也不自动覆盖尚在实现的 PosteL/GSDN/ALS/SALS 最近工作。
- Manuscript action: KEEP

### C-027
- Section: §5.3.7“tie-safe 暴露审计”
- Narrowest defensible claim: v2 的 125 个 GB-LCSS 行中有 123 行出现 active tie；累计 911 个 tie balls、其中 910 个 active，涉及 12,788/259,000 个节点（4.937%）；在正 `alpha` 行内，tie-safe 分支的加权目标质量占比为 2.288%。
- Claim type: MECHANISM
- Evidence files: `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/raw_combined.csv`; `paper_pipeline/04_experiments/tie_safe_full_grid_experiment_report.md`
- Evidence locator: 从 GB-LCSS 行的 tie diagnostics 汇总 `num_tie_balls`、active tie balls、tie nodes、total nodes、positive-alpha weighted mass；报告保留汇总公式与逐行来源。
- Protocol: 与 C-026 相同；暴露统计仅描述 tie-safe 分支在已运行网格中的实际触发范围。
- Status: SUPPORTED
- Allowed interpretation: tie-safe 不是只存在于代码中的未触发分支；它在绝大多数 setting 中被激活，但覆盖的节点与加权目标质量均为少数部分。
- Forbidden expansion: 不得把 4.937% 节点覆盖或 2.288% 加权质量写成独立因果贡献；没有专门消融时不得声称 tie-safe 单独提升精度。
- Manuscript action: KEEP

### C-028
- Section: §4.4“统计报告与复现信息”
- Narrowest defensible claim: v2 的原始数值证据、输入源哈希和归一化命令可复核，但旧版恢复逻辑在 resume 时覆盖了 27 个输入快照及根环境指纹；初始/恢复环境指纹已从会话记录和现存文件恢复，因而不能宣称整个运行目录的全部元数据从创建起不可变。
- Claim type: REPRODUCIBILITY
- Evidence files: `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/environment_fingerprint.initial_recovered.json`; `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/environment_fingerprint.resume_20260727_230258.json`; `../../external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/lineage_amendment_20260727.md`; `paper_pipeline/04_experiments/tie_safe_full_grid_experiment_report.md`
- Evidence locator: amendment 记录初始与恢复 UTC、git diff SHA-256、27 个快照覆盖事件和可/不可主张边界；两份 fingerprint 文件分别保存恢复的初始状态和实际 resume 状态。
- Protocol: 恢复运行谱系审计；数值行、输入源、命令和环境字段做独立比对；后续 runner 已新增“存在即验证、差异即中止”的恢复测试。
- Status: SUPPORTED
- Allowed interpretation: 可主张数值结果和确定性输入/命令的一致性；必须伴随目录元数据不可完全追溯的限制。
- Forbidden expansion: 不得称 v2 整个目录“从创建起完全不可变”，不得隐去恢复运行覆盖事件。
- Manuscript action: KEEP

### C-029
- Section: §2“相关工作”+ §5“最近工作对照”
- Narrowest defensible claim: PosteL 是直接的结构辅助后验软标签工作，GSDN 是预测层图自蒸馏工作，ALS/SALS 是图结构辅助软标签工作。历史 `gsdn` 行只是 mechanism adaptation；新 `gsdn_faithful_core` 按 Eq. (3)/(7)/(8) 独立实现，但因每批唯一无向关系归一化等共同协议适配，只能称 `equation-grounded common-protocol adaptation`，不能称官方或精确复现。
- Claim type: IDENTITY
- Evidence files: `paper_pipeline/00_evidence/nearest_work_verification_notes.md`; `paper_pipeline/04_experiments/nearest_work_pause_state_20260728.md`; `paper_pipeline/scripts/run_nearest_work_full_grid.py`; `paper_pipeline/tests/test_nearest_work_full_grid.py`; `../../external/NoisyGL/predictor/GSDNFaithful_Predictor.py`; `../../external/NoisyGL/tests/test_gsdn_faithful.py`; `../../external/NoisyGL/scripts/run_standard_downstream.py`; `../../external/NoisyGL/tests/test_standard_runner_baselines.py`
- Evidence locator: verification notes 逐项记录论文机制、官方代码、训练/推理资源和本地身份；GSDN tests 用独立 oracle 覆盖唯一无向关系、Eq. (7)/(8) 与两端标签传播；formal grid 的 capability registry、原始结果和最终报告继续使用上述身份，不把项目适配重标为官方复现。
- Protocol: 定向最近工作核验、实现身份审计、CPU 单测、共同 5×5×5 网格和运行时诊断；性能结论另由 C-030 约束。
- Status: SUPPORTED
- Allowed interpretation: 可在相关工作和实验设置中陈述机制邻近性、实现边界及共同协议适配身份。
- Forbidden expansion: 不得把历史 `GSDN-inspired` 或新 GSDN common-protocol adaptation 重标为 faithful/exact/official；不得用身份核验替代 C-030 的性能证据。
- Manuscript action: KEEP

### C-030
- Section: §5.2“与结构辅助 X-only 最近工作的共同协议比较”（Table 3）
- Narrowest defensible claim: 在 5 数据集、5 噪声设置和 5 配对种子的正式共同协议中，GB-LCSS 相对 PosteL、ALS 和 SALS 的 25-setting 平均配对增量分别为 +5.202、+7.215 和 +4.202 个百分点；相对 GSDN 为 -0.698 个百分点。GB-LCSS 在 10/25 个设置取得最高均值，GSDN 在 15/25 个设置取得最高均值。
- Claim type: PERFORMANCE
- Evidence files: `../../external/NoisyGL/runs/nearest_work_full_grid_20260728/five_method_raw_combined.csv`; `../../external/NoisyGL/runs/nearest_work_full_grid_20260728/five_method_aggregate.csv`; `../../external/NoisyGL/runs/nearest_work_full_grid_20260728/cross_vs_lcss_paired_deltas.csv`; `../../external/NoisyGL/runs/nearest_work_full_grid_20260728/cross_vs_lcss_paired_tests.csv`; `paper_pipeline/04_experiments/nearest_work_full_grid_final_report_20260729.md`
- Evidence locator: `five_method_aggregate.csv` 提供 125 个方法--数据集--噪声设置聚合；paired deltas/tests 提供 500 个配对差、100 个检验和四个独立的 25 项 Holm 族；最终报告第 10 节给出 uniform 0.4 表、完整 25-setting 汇总和数据集级 GSDN 差异。
- Protocol: 5 datasets；uniform 0.2/0.4/0.6、pair 0.4、instance-dependent 0.4；train-only noise；20/10/70 split；5 paired seeds；全部学生测试时只接收节点属性。
- Status: SUPPORTED
- Allowed interpretation: 可称 GB-LCSS 稳定优于 PosteL、ALS 和 SALS；与 GSDN 的优势具有数据集依赖性，GB-LCSS 在 CiteSeer 和 Coauthor-CS 上更强，GSDN 在 CoraFull 和 Amazon-Photo 上更强。
- Forbidden expansion: 不得称 GB-LCSS 优于全部同资源最近工作、整体最优或 SOTA；不得用 GSDN 较大的标准差抵消其 15/25 个设置均值更高的事实。
- Manuscript action: KEEP

### C-031
- Section: §4.4“统计报告与复现信息”+ §8“数据与代码可用性”
- Narrowest defensible claim: 正式最近工作网格包含 25/25 个不可变命令块、500/500 条原生最近工作结果和 625/625 条五方法联合证据；125/125 个 dataset--noise--seed 组通过 split、noise、污染计数和 train-only 协议审计，精确恢复运行未启动训练子进程。
- Claim type: REPRODUCIBILITY
- Evidence files: `../../external/NoisyGL/runs/nearest_work_full_grid_20260728/immutable_raw_manifest.csv`; `../../external/NoisyGL/runs/nearest_work_full_grid_20260728/command_manifest.csv`; `../../external/NoisyGL/runs/nearest_work_full_grid_20260728/recent_work_raw_combined.csv`; `../../external/NoisyGL/runs/nearest_work_full_grid_20260728/five_method_raw_combined.csv`; `../../external/NoisyGL/runs/nearest_work_full_grid_20260728/protocol_audit.csv`; `../../external/NoisyGL/runs/nearest_work_full_grid_20260728/completeness.csv`
- Evidence locator: immutable manifest 的 25 个 SHA-256 全部匹配；recent-work raw 为 500 个唯一键，five-method raw 为 625 个唯一键；protocol audit 的 125 组全部通过；第二次 exact-resume 的 command manifest 全部为 `skipped_exact_complete`。
- Protocol: 与 C-030 相同；恢复逻辑采用“存在即验证、差异即中止”，派生五方法汇总在正式网格闭合后由冻结原始文件生成。
- Status: SUPPORTED
- Allowed interpretation: 可称正式最近工作共同协议网格完整、可配对且断点恢复没有覆盖已完成 immutable block。
- Forbidden expansion: 完整性不等于四个适配实现均为官方精确复现，也不证明结果可在不同软件环境中逐位复现。
- Manuscript action: KEEP

## Ledger Summary

- SUPPORTED: 27
- PARTIAL: 3
- UNVERIFIED: 0
- CONTRADICTED: 1
- P0 blockers: 0
- P0 IDs: none

## Validation

- Claim IDs: `C-001`–`C-031`，连续、无重复、无缺号。
- Entry schema: 每条均且仅有一个 `Section`、`Narrowest defensible claim`、`Claim type`、`Evidence files`、`Evidence locator`、`Protocol`、`Status`、`Allowed interpretation`、`Forbidden expansion`、`Manuscript action`。
- Status/action uniqueness: 每条恰有一个 Status 和一个 Manuscript action。
- Checked CSV columns: 主结果 `dataset,method,seed,noise_type,noise_rate,mean,std,mean_delta`；协议 `noise_scope,split_hash,noise_hash,modified_train_count,modified_val_count,modified_test_count`；统计 `comparison,n,positive_seeds,paired_t_p,holm_p,holm_p_within_family`；LOO `leave_one_out,soft_target_brier,soft_pseudo_ncr,soft_pseudo_cdr`；粒球 `num_balls_mean,mean_train_per_ball_mean,noisy_train_purity_weighted_mean,test_mean`；效率 `downstream_wall_seconds,ball_build_seconds,parameter_count,inference_peak_allocated_mb,fullgraph_latency_ms`。
- Remaining concerns: `C-022` 的宽泛新颖性表述已撤回；`C-024` 没有 kmeans raw；`C-025` 的 noisy-CE GRACE 与 MLP-hidden 现稿数值缺直接 raw，均已降级并禁止扩写。正式最近工作网格已经闭合，但 GSDN 在 15/25 个设置取得更高均值，因此 GB-LCSS 不得被表述为同资源 X-only 最近工作中的整体最优方法。
## V32 current-evidence amendment (2026-07-29)

This amendment supersedes earlier headline counts wherever they conflict.
The full method-level registry is
`paper_pipeline/00_evidence/v32_evidence_registry.csv`.

### V32-C-001: real grouping versus shuffled grouping

- Claim: GB-LCSS has a positive mean delta over the equal-size shuffled
  grouping control in 25/25 dataset-noise settings.
- Mean paired delta: +7.840969 pp.
- Source:
  `external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/`.
- Method identity: tie-safe v2, `tie_mode=distribution_only`.
- Allowed use: headline mechanism evidence.
- Forbidden use: proof that granular balls are uniquely superior to kNN,
  k-means, or every possible grouping.

### V32-C-002: same-resource training strategies

- Claim: GB-LCSS has a positive mean delta over MLP+CE in 24/25 settings
  and over JoCoR in 23/25 settings.
- Mean paired deltas: +7.251731 pp and +2.963838 pp.
- Source:
  `external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/`.
- Allowed use: comparisons within the X-only, train-target resource class.
- Forbidden use: overall SOTA or an unconditional win rate over all
  graph-assisted X-only methods.

### V32-C-003: nearest structure-assisted X-only methods

- Claim: under the frozen common protocol, GB-LCSS has average paired
  deltas of +5.202260 pp, +7.215330 pp, and +4.201898 pp over PosteL,
  ALS, and SALS, respectively.
- Positive settings: 24/25, 25/25, and 24/25.
- Source:
  `external/NoisyGL/runs/nearest_work_full_grid_20260728/`.
- Allowed use: current comparison with the three verified adapted methods.
- Forbidden use: a general claim that GB-LCSS is the best graph-assisted
  X-only method.

### V32-C-004: GSDN status

- The old `gsdn` result is a provisional adapted implementation and may
  appear only as a labeled positioning reference.
- `gsdn_faithful_core` is excluded from v32 claims pending the user's new
  implementation.
- No v32 headline may state that GB-LCSS wins or loses to GSDN overall.

### V32-C-005: validation-label sensitivity

- Claim: when validation labels are independently corrupted at uniform
  0.4 while training-noise hashes and test labels remain fixed, the
  pre-registered grouping-mechanism gates pass on Cora, CiteSeer, and
  CoraFull.
- GB-LCSS minus shuffled: +12.4538, +11.4519, and +4.9614 pp, all 5/5
  seeds positive.
- GB-LCSS minus MLP+CE: +12.6438, +11.4089, and +5.0494 pp, all 5/5
  seeds positive.
- Clean/noisy-validation six-method ranking Kendall tau: 0.8667 on each
  dataset.
- Source:
  `external/NoisyGL/runs/validation_scope_sensitivity_20260729/`.
- Allowed use: the main grouping increment is robust to this validation
  corruption test.
- Forbidden use: the full method ranking is invariant to arbitrary
  validation noise.

### V32-C-006: PubMed scope

- All PubMed accuracy, mechanism, capacity, H-access, LOO, label-rate,
  and distillation evidence is appendix-only in v32.
- PubMed uses historical target rules and is excluded from current
  headline denominators.
- Allowed use: transparent boundary case.
- Forbidden use: mixing PubMed into the current 25-setting tie-safe
  denominator.

### V32-C-007: alpha diagnostic

- Claim: in a hash-verified reconstruction of the 25 uniform-0.4
  dataset-seed targets, target argmax accuracy is monotonic across alpha
  bins within 4/5 datasets, but the pooled curve is not strictly
  monotonic.
- Source:
  `paper_pipeline/04_experiments/v32_alpha_calibration.csv`.
- Identity check: split, noise, and ball-assignment hashes match for
  25/25 reconstructions.
- Allowed use: alpha is an interpretable heuristic consensus weight.
- Forbidden use: alpha is a calibrated confidence or correctness
  probability.

## V38 formal LOO amendment (2026-07-30)

This amendment supersedes the V32 headline method identity and all
self-inclusive headline counts. Historical self-inclusive rows remain valid
only as explicitly labelled ablations.

### V38-C-001: formal method identity and promotion

- Claim: the formal GB-LCSS target uses
  `tie_mode=distribution_only` and `leave_one_out=true`.
- Node-specific quantities are
  \(p_{j,-v},q_{j,-v},\alpha_{j,-v}\); historical v1 LOO rows are excluded.
- Full-spectrum confirmation: global LOO minus self-inclusive mean
  \(+0.4916\) pp; all five datasets have non-negative cross-setting means;
  minimum setting delta \(-0.4855\) pp.
- Source:
  `external/NoisyGL/runs/mlp44_loo_tiesafe_fullspectrum_20260730/`.
- Allowed use: formal method definition and preregistered promotion.
- Forbidden use: claiming a large universal accuracy gain from LOO itself.

### V38-C-002: primary noisy-label comparisons

- Claim: formal LOO GB-LCSS has positive mean deltas in 25/25 settings
  versus LOO shuffled grouping, 25/25 versus MLP+CE, and 24/25 versus
  JoCoR.
- Mean paired deltas: \(+8.6879\), \(+7.7433\), and \(+3.4554\) pp.
- Holm-significant settings: 25/25, 24/25, and 11/25.
- GB-LCSS is best or tied among the seven same-backbone baselines in
  24/25 settings. The exception is Coauthor-CS uniform 0.2, where it is
  0.2712 pp below JoCoR.
- Source:
  `paper_pipeline/00_evidence/v38_loo_summary/`.
- Allowed use: comparisons within the X-only train-target resource class.
- Forbidden use: overall SOTA or significance in every setting.

### V38-C-003: nearest structure-assisted methods

- Claim: formal LOO GB-LCSS has mean paired deltas of \(+5.6938\),
  \(+7.7069\), and \(+4.6935\) pp over PosteL, ALS, and SALS.
- Positive settings: 24/25, 25/25, and 24/25.
- The manuscript retains the older explicitly labelled GSDN positioning row
  at the user's request; no new headline claim is made from
  `gsdn_faithful_core`.
- Allowed use: common-protocol comparison with the three verified adapted
  soft-target methods.
- Forbidden use: claiming that GB-LCSS dominates all graph-assisted X-only
  methods.

### V38-C-004: grouping alternatives and cost

- Claim: at uniform 0.4, LOO GB-LCSS differs from the better of exact kNN
  and LOO k-means by \(+0.27,-0.94,-3.11,-0.29,-0.11\) pp on Cora,
  CiteSeer, CoraFull, Coauthor-CS, and Amazon-Photo.
- Exact-kNN comparison: GB has lower end-to-end construction time, measured
  peak RSS, and grouping storage on 3/5 datasets for each metric.
- No approximate-kNN package was available in the frozen environment.
- Source:
  `external/NoisyGL/runs/mlp45_loo_grouping_pareto_20260730/`.
- Allowed use: granular balls are a fixed-budget, mutually exclusive,
  auditable implementation of local consensus.
- Forbidden use: a general accuracy, runtime, memory, or approximate-kNN
  advantage.

### V38-C-005: clean-label control

- Claim: at \(\rho=0\), formal LOO GB-LCSS minus MLP+CE is
  \(-0.24,+2.00,-0.38,-0.87,-1.75\) pp on the five main datasets.
- Formal LOO minus LOO shuffled is positive on 4/5 datasets and
  \(-1.21\) pp on CoraFull.
- Source:
  `external/NoisyGL/runs/mlp46_loo_clean_control_20260730/`.
- Allowed use: evidence that the main benefit is noise-specific and may
  trade off clean-label accuracy.
- Forbidden use: a claim of universal regularization benefit.

### V38-C-006: affected historical diagnostics

- Validation-label sensitivity, GRACE-seed sensitivity,
  representation-source comparison, supplemental six-dataset rows,
  vote-support bins, and `s_min` sensitivity use the tie-safe
  self-inclusive precursor.
- They may remain only when explicitly labelled as precursor diagnostics or
  boundary evidence and must not be merged into formal LOO headline counts.
- Teacher distillation remains a separate extension. Its ball teacher uses
  the full distribution with \(\beta_T=0\), so it is not redefined as the
  formal leave-one-out train-target method.

### V38-C-007: frozen manuscript lineage

- Frozen predecessor:
  `paper_final_zh_v37_cold_review_ready_20260730.md`,
  SHA256
  `CFB70F736B459BD049760EE095355CF62DF67B408E1C1C803A0446EFB7D60BB6`.
- Formal LOO manuscript:
  `paper_final_zh_v38_loo_main_20260730.md`,
  SHA256
  `3CAF4A54F7651E35DDB80B23E7C30F85C3FBC0DE2F74CD462726D3792488FDD0`.
- Frozen v38 evidence registry:
  `paper_pipeline/04_experiments/v38_evidence_registry_20260730.csv`,
  SHA256
  `FC978EB213A5C5781687633224A6009595E31BDD076CA3068B5A363E6B9F1C9E`.
- The v37 hash confirms that the predecessor was not overwritten. The v38
  manuscript is the first submission draft whose formal train-target method
  is the tie-safe leave-one-out variant.

## V39 robust-teacher distillation amendment (2026-07-30)

This amendment changes only the stronger-resource teacher-distillation
extension. It does not alter the formal LOO GB-LCSS method, the five-dataset
main grid, or any V38 headline count.

### V39-C-001: NRGNN teacher comparison

- Claim: under uniform 0.4, the NRGNN-teacher X-only student differs from the
  granular-ball-teacher student by \(+0.46,+1.47,-0.52\) pp on Cora,
  CiteSeer, and CoraFull.
- Under uniform 0.6, the paired differences are
  \(+3.54,+4.34,+1.73\) pp, with at least 4/5 positive seeds on every
  dataset.
- Adding \(H\) to the NRGNN teacher does not consistently improve the
  X-only student. It is higher than the plain NRGNN teacher only on
  CiteSeer uniform 0.6.
- Source:
  `external/NoisyGL/runs/mlp47_nrgnn_teacher_distill_20260730/`.
- Source raw SHA256:
  `6950B3452B97929FDF50D4BF49E6DAC0B65DA5F88B0F612779CDD512F605D682`.
- Allowed use: a mature robust graph teacher is a stronger distillation
  alternative at high label-noise rates, while the best teacher is
  dataset- and rate-dependent.
- Forbidden use: claiming that the granular-ball teacher is the strongest
  teacher, or that NRGNN teachers uniformly dominate every setting.

### V39-C-002: teacher-to-student compression

- Claim: the NRGNN-student minus its own teacher is within approximately
  \(1.2\) pp on Cora and CiteSeer across the two tested rates, but is about
  \(-5\) pp on CoraFull.
- All students are two-layer X-only MLPs. Teachers are retrained inside every
  seed; 30/30 dataset-rate-seed protocol groups have matching split and noise
  hashes across all seven teacher arms.
- Allowed use: graph-teacher predictions can be transferred to an X-only
  student with little loss on the two small citation graphs, while CoraFull
  exposes a clear compression boundary.
- Forbidden use: claiming full teacher retention on all datasets.

### V39-C-003: interpretation boundary

- The ball teacher remains a simple local-consensus teacher construction.
  The NRGNN result shows that the teacher-distillation extension is not
  granular-ball-specific.
- The extension jointly changes teacher architecture, teacher supervision,
  graph propagation, and full-node target coverage. It cannot identify a
  single causal source of the gain.
- Registry:
  `paper_pipeline/04_experiments/v39_evidence_registry_20260730.csv`.

### V39-C-004: frozen manuscript lineage

- Frozen v38 predecessor SHA256:
  `3CAF4A54F7651E35DDB80B23E7C30F85C3FBC0DE2F74CD462726D3792488FDD0`.
- V39 manuscript SHA256:
  `F6CDA518B6402ADB963FE065CCEF14BD481958099FF33CFEF4BF63423D30AC03`.
- V39 evidence registry SHA256:
  `7B5C97305709636E6EC7BBE8B3E08685BE89834DDBC1DCC7760599527A074C75`.
- The v38 hash is unchanged. V39 modifies only the robust-teacher
  distillation extension and its interpretation.

## V55 GNN-to-MLP Table 4 amendment (2026-08-12)

### V55-C-001: additional GNN-to-MLP comparisons

- FF-G2M, KRD, and VQGraph were evaluated on Cora, CiteSeer, and CoraFull
  under uniform 0.4 and 0.6 with five paired seeds.
- All 90 new records completed successfully. All 30 dataset-rate-seed groups
  match the split and noise hashes of the existing Table 4 routes.
- The granular-ball consensus teacher exceeds all three added routes on Cora
  and CiteSeer at both noise rates. On CoraFull, VQGraph is higher by 0.29 and
  0.19 percentage points at uniform 0.4 and 0.6.
- Allowed use: the consensus teacher is competitive with the tested
  GNN-to-MLP routes and is stronger on the two small citation graphs,
  especially under high noise.
- Forbidden use: claiming official reproduction for KRD/VQGraph, universal
  superiority of the granular-ball teacher, or extending the conclusion
  beyond the tested datasets and noise rates.
- Source: `external/NoisyGL/runs/mlp55_g2m_table4_20260812/mlp55_raw.csv`.
- Source SHA256:
  `70103B2D0E719ECC98BA6254461ECE43596A45EE105684CD9B50694F793D7166`.

## V57 X-only ball-teacher and formal LOO amendment (2026-08-13)

### V57-C-001: X-only granular-ball teacher

- The granular-ball teacher consumes only `X` and `A`; frozen GRACE `H` is
  used only to construct the teacher target.
- On Cora and CiteSeer, the X-only ball teacher exceeds GCN(X)+CE by
  (+1.93,+1.89) pp at uniform 0.4 and (+3.40,+2.79) pp at uniform 0.6,
  with 5/5 positive paired seeds in all four cells.
- On CoraFull, the corresponding differences are (-0.56) and (+0.89)
  pp. Both noise-rate-level adoption gates pass on 2/3 datasets.
- Source:
  `external/NoisyGL/runs/mlp57_xteacher_loo_cleanup_20260813/xteacher_raw.csv`.
- Source SHA256:
  `59E0962940A7D448EB9FE7CCA1DC0A102BF9DB13EC562BA5399FC675E5D2E6E6`.
- Allowed use: the ball target has an independent effect when teacher input
  is limited to X; in the six Table-4 cells, this route has the highest mean
  in five.
- Forbidden use: claiming that removing H is accuracy-neutral. The X-only
  teacher is lower than the earlier XH ball route, especially on CiteSeer
  uniform 0.6.

### V57-C-002: formal LOO validation-protocol sensitivity

- Real and shuffled ball targets both use `distribution_only` and
  `leave_one_out=true` under clean-validation and noisy-validation protocols.
- Under noisy validation, GB-LCSS exceeds the paired shuffled control by
  (+13.69,+12.35,+4.87) pp on Cora, CiteSeer, and CoraFull, and exceeds
  MLP+CE by (+13.51,+11.93,+4.91) pp; every comparison is positive for
  5/5 seeds.
- Kendall tau between clean- and noisy-validation method rankings is 1.000,
  0.867, and 0.867 on the three datasets.
- Source:
  `external/NoisyGL/runs/mlp57_xteacher_loo_cleanup_20260813/validation_loo_summary/loo_validation_combined.csv`.
- Source SHA256:
  `24E9CB1F0CC8E0052568DB07DF13F2593C2FC842562BE733A637CCB5535CF010`.
- This source supersedes the non-LOO validation diagnostic for all manuscript
  claims about the formal method.

### V57-C-003: formal LOO GRACE-seed stability

- Across three archived H seeds, the between-H mean-accuracy standard
  deviations are 0.13, 0.45, and 0.60 pp on Cora, CiteSeer, and CoraFull.
- GB-LCSS minus MLP+CE remains positive for all 3/3 H seeds on every dataset;
  all pre-registered dispersion gates pass.
- Cross-H assignment ARI has mean 0.300 and range 0.247--0.365.
- Source:
  `external/NoisyGL/runs/mlp57_xteacher_loo_cleanup_20260813/grace_seed_loo/grace_seed_tiesafe_raw.csv`.
- Source SHA256:
  `683F5A38CD1B9CF0A0D6815EE99D5452D638D9CBD47FE0FCA609D977FDA6BFA8`.
- This source supersedes the non-LOO GRACE-seed diagnostic.

### V57-C-004: formal LOO representation-source diagnostic

- Raw X and A2X each form one ball at seed 0 on Cora, CiteSeer, and CoraFull;
  these cells are geometry-collapse diagnostics, not multi-seed performance
  estimates.
- Formal LOO GRACE-H results are 57.49+/-0.76, 55.24+/-2.04, and
  37.39+/-0.51 on the three datasets.
- Source:
  `external/NoisyGL/runs/mlp57_xteacher_loo_cleanup_20260813/representation_source_loo/representation_source_formal_seed0.csv`.
- Source SHA256:
  `997709733A101F624AF50F1ADD4757F6D18A2D06D3C3FBE57B77A9C9E79FF24C`.
- Allowed use: under the fixed partition rule, GRACE H produces usable local
  partitions where raw X and A2X collapse on the three tested datasets.
- Forbidden use: generalizing this result to all self-supervised or
  label-free representations.

### V57-C-005: manuscript lineage

- Frozen predecessor:
  `paper_final_zh_v56_pretranslation_consistency_revised_20260813.md`,
  SHA256
  `8A02262BFCCB413E916219AD35A29214B3D37342F0584ED045EBB4D0E93AAF57`.
- V57 manuscript:
  `paper_final_zh_v57_xteacher_loo_formalized_20260813.md`,
  SHA256
  `E8A02263B3456D75BEF1A451C8DA32E9E9F19609AE2B11DC40BB25AD78F729C7`.
- V56 was not overwritten. V57 replaces the main teacher route with the
  X-only ball teacher, replaces affected precursor diagnostics with formal
  LOO evidence, retains the self-inclusive target only as an ablation, and
  removes obsolete XH-teacher and non-LOO breadth tables from the submission
  manuscript.

## V58 formal GB-LCSS teacher amendment (2026-08-13)

### V58-C-001: formal target transfer to the graph teacher

- The granular-ball GCN teacher consumes only `X` and `A`; frozen GRACE `H`
  is used only to construct granular balls and the teacher target.
- The teacher is trained by the same formal target as the base method:
  `target_beta=0.5`, `leave_one_out=true`, and
  `tie_mode=distribution_only`.
- Against GCN(X)+CE, the formal teacher improves Cora and CiteSeer by
  (+3.40,+3.18) pp at uniform 0.4 and (+5.84,+5.64) pp at uniform 0.6,
  with 5/5 positive paired seeds in all four cells.
- On CoraFull, the corresponding deltas are (-1.46) and (+0.69) pp. Both
  pre-registered rate-level gates pass on 2/3 datasets.
- Source:
  `external/NoisyGL/runs/mlp58_formal_lcss_teacher_20260813/mlp58_raw.csv`.
- Source SHA256:
  `A75308EECD6E540A52288528BDD8D770C038CA989607848C127F7388747FFB3E`.
- Paired controls:
  `external/NoisyGL/runs/mlp58_formal_lcss_teacher_20260813/mlp58_controls.csv`,
  SHA256
  `A0EFD7BDD2D7B611E4A7067983532742F7FD98BF8B308A9C856CFB87EED263BA`.
- Control hash audit:
  `external/NoisyGL/runs/mlp58_formal_lcss_teacher_20260813/mlp58_control_hash_audit.csv`,
  SHA256
  `197094442BFD5EF1A08168D6CAECFF8D5FA39D43E6A02CD55CE63E3978CD5D28`.
- All 60 formal-versus-control dataset-rate-seed comparisons match split and
  noise hashes.

### V58-C-002: comparison with the previous beta-zero teacher

- Relative to the beta-zero X-only ball teacher, the formal target changes
  Cora, CiteSeer, and CoraFull by (+1.47,+1.29,-0.90) pp at uniform 0.4 and
  (+2.44,+2.84,-0.19) pp at uniform 0.6.
- No dataset has a mean regression greater than 1 pp, and the rate-average
  deltas are +0.62 and +1.70 pp. Both pre-registered preservation gates pass.
- Allowed use: the formal base-method target can also train the graph teacher,
  improving method coherence and raising accuracy on Cora and CiteSeer.
- Forbidden use: claiming that the formal target improves every dataset or
  that it eliminates teacher-target sensitivity.

### V58-C-003: real-versus-shuffled teacher attribution

- The formal real-ball teacher exceeds the matched shuffled teacher by
  (+5.02,+4.52,-1.23) pp at uniform 0.4 and
  (+7.36,+6.40,+0.13) pp at uniform 0.6 on Cora, CiteSeer, and CoraFull.
- The pre-registered gate passes at both rates because Cora and CiteSeer each
  exceed +1 pp with 5/5 positive paired seeds.
- CoraFull is a negative attribution cell at uniform 0.4 and an effectively
  tied cell at uniform 0.6; this exception must remain visible.
- Paired-summary source:
  `external/NoisyGL/runs/mlp58_formal_lcss_teacher_20260813/mlp58_paired_summary.csv`.
- Source SHA256:
  `4F49C721D342266E1DE9E65A3A56E4DF2A1B48496225D192613892F1FA55AD8E`.

### V58-C-004: manuscript lineage

- Frozen predecessor:
  `paper_final_zh_v57_xteacher_loo_formalized_20260813.md`, SHA256
  `E8A02263B3456D75BEF1A451C8DA32E9E9F19609AE2B11DC40BB25AD78F729C7`.
- V58 manuscript:
  `paper_final_zh_v58_formal_teacher_revised_20260813.md`, SHA256
  `0BF71312E13FA0BC6E3E41D48E7B50709547DF9BE2BAEDFA4374879278EF2231`.
- V57 was not overwritten. V58 replaces only the beta-zero granular-ball
  teacher route, its Table-4 values, and the corresponding robust-teacher
  comparison with the pre-registered formal-target results.

## V60 MLP-59 dataset and teacher extension (2026-08-14)

### V60-C-001: five-dataset formal teacher comparison

- The formal ball-consensus teacher was extended to Coauthor-CS and
  Amazon-Photo at train-only uniform 0.4 and 0.6, with seeds 0--4.
- On Coauthor-CS, formal-teacher accuracy is 93.83+/-0.21 and 92.40+/-0.65;
  on Amazon-Photo it is 90.40+/-0.77 and 86.81+/-1.94.
- Relative to GCN(X)+CE, the four new deltas are +0.63, +1.27, +0.54, and
  +2.01 pp. Relative to matched shuffled teachers, they are +1.44, +2.66,
  +2.53, and +6.58 pp, with 5/5 positive seeds in every cell.
- Across the resulting ten Table-4 cells, the ball-consensus teacher has the
  highest mean in eight and exceeds GCN(X)+CE in nine. The exceptions are
  CoraFull uniform 0.4 and Amazon-Photo uniform 0.4.
- FF-G2M, KRD, and VQGraph were completed on the same two datasets and two
  rates. VQGraph is the strongest of these three routes; the ball-consensus
  teacher exceeds it in three of four new cells and trails by 0.14 pp on
  Amazon-Photo uniform 0.4.
- Allowed use: the formal teacher consistently improves the base X-only
  GB-LCSS student on the five tested datasets, with its strongest relative
  position at uniform 0.6.
- Forbidden use: claiming that the ball-consensus teacher is uniformly best,
  or that the adapted KRD/VQGraph rows are official reproductions.

### V60-C-002: final-target boundary entry test

- Six historical breadth datasets were rerun using the formal leave-one-out,
  tie-safe target at uniform 0.4. Real-versus-shuffled deltas are positive
  with 5/5 positive seeds on all six datasets.
- Relative to each dataset's strongest eligible same-backbone baseline, the
  deltas are -5.24 (PubMed), -0.60 (Amazon-Computers), +0.70 (Cora_ML),
  -2.66 (Attributed-Wiki), +1.46 (CiteSeerFull), and +0.04 pp
  (Coauthor-Physics).
- Only CiteSeerFull meets the frozen entry rule of at least +0.5 pp with at
  least 4/5 positive seeds, together with the shuffled-control requirement.
- Allowed use: real local grouping contributes on all six datasets, while
  superiority to strong robust baselines remains dataset dependent.
- Forbidden use: promoting Cora_ML or Coauthor-Physics from a positive mean
  alone, or replacing a current main dataset after observing these results.

### V60-C-003: CiteSeerFull full-spectrum boundary

- CiteSeerFull deltas against the strongest eligible baseline are +3.24,
  +1.46, -4.20, +3.89, and +1.57 pp for uniform 0.2/0.4/0.6, pair 0.4, and
  instance 0.4.
- Four of five settings are non-inferior and the cross-setting mean is
  +1.19 pp, but uniform 0.6 violates the frozen worst-cell rule with a
  -4.20 pp delta and 0/5 positive seeds.
- Decision: CiteSeerFull is not added to the formal main matrix and remains a
  reported applicability boundary.

### V60-C-004: sources and completeness

- Raw source: `external/NoisyGL/runs/mlp59_final_extensions_20260814/mlp59_raw.csv`,
  SHA256 `4D8C07D5D7E9CF6885DE88883E2C09E165ECF78F8553A373C2ABEB4CAC2E0BFC`.
- Aggregate source: `external/NoisyGL/runs/mlp59_final_extensions_20260814/mlp59_aggregate.csv`,
  SHA256 `C07CF140855A88CB933C844E7D85A2AAB301C9BF2AA53D8CAF39FCF817C5C4E0`.
- Paired source: `external/NoisyGL/runs/mlp59_final_extensions_20260814/mlp59_paired_deltas.csv`,
  SHA256 `108117FF4722B6330E394589474A3C699D2DA96F313F4A534DB7ACA8ED476A26`.
- Gate source: `external/NoisyGL/runs/mlp59_final_extensions_20260814/mlp59_gate_decisions.csv`,
  SHA256 `ED51CE3DF4936197FF51ABC5AAAB359D8482B31D16D03B7C8B4A839EBC1E0E97`.
- Completeness source: `external/NoisyGL/runs/mlp59_final_extensions_20260814/mlp59_completeness.csv`,
  SHA256 `B9FF7CADD5C4A5275ED675816F34AFC8A2F5CE330BDCFBCC22CFEB6A2FE07B83`.
- All five audited phases pass: teacher 60 rows, GNN-to-MLP 60 rows,
  boundary entry 60 rows, CiteSeerFull formal targets 50 rows, and
  CiteSeerFull baseline controls 175 rows.

### V60-C-005: manuscript lineage

- Frozen predecessor: `paper_final_zh_v59_pretranslation_revised_20260814.md`,
  SHA256 `0A785668C3D02311C2C4BD4F46BC18242353F8799C5779E601E599472F23FAD8`.
- V60 manuscript: `paper_final_zh_v60_mlp59_extension_20260814.md`, SHA256
  `6520FB0089734075E862C2E6F1AACEE0E27CF1B53141C334B3FA32B54C732792`.
- V59 was not overwritten. V60 extends Table 4 to five datasets and adds the
  formal-target boundary diagnostic in Appendix I; it does not change the
  five-dataset main matrix.
