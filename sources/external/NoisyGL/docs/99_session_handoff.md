# Session Handoff

## 2026-07-03 MLP-14 Graph-Free Family

- 已执行 `docs/execution_plan_mlp14_graphfree_family_20260703.md`。
- 新增 adapted graph-free family baseline：`gsdn` 与 `graphmlp`。
- 输出目录：`runs/mlp14_graphfree_family_20260703/`。
- 完整性：250/250 graph-free rows，smoke pass，missing groups = 0。
- 报告：`runs/mlp14_graphfree_family_20260703/mlp14_report.md`。
- 重要口径：本轮为 adapted fixed-default 实现，未做 noisy-val 超参网格；论文中不得写“网格调参后结果”。
- 2026-07-03 修正：MLP-14 参照行读取 `mlp_maintable_20260701` 时必须过滤 `train_percent=0.2`；已修复并重生成报告。修正后 Cora/CiteSeer u0.4 的 Ours train-only 为 55.59/53.83。
- 表 B 结论：Graph-MLP 与 Ours train-only 是 profile 互有胜负；修正后 Ours train-only 整体均值略高，但 Graph-MLP 在 Cora/CiteSeer 多个设置稳定占优，不能写全面胜利；GSDN 整体弱于 Ours train-only；Ours-distill 在 Cora/CiteSeer/PubMed 的 u0.4/u0.6 上明显强于 Graph-MLP/GSDN。
- 主表 A 不变；MLP-14 只用于 graph-free family 定位表。

## 2026-07-06 MLP-16 Cut Value + Feature Noise

- 已执行 `docs/execution_plan_mlp16_cutvalue_featnoise_20260706.md`。
- 新增 runner：`scripts/run_mlp16_cutvalue_featnoise.py`；新增测试：`tests/test_mlp16_cutvalue_featnoise.py`；新增 Amazon-Photo kNN 配置：`config/knn_mlp_softdist_beta05/knn_mlp_softdist_beta05_amazonpho.yaml`。
- 输出目录：`runs/mlp16_cutvalue_featnoise_20260706/`；完整性 180/180 model groups；α=0 anchor failures=0；测试 `3 passed`。
- P0 Cut Value：主预期不成立。train-only Ours 的 CV 不稳定高于 CE/shuffled；Cora/PubMed/AmazonPhoto 均出现 Ours≤CE 或 Ours≤shuffled。论文不得写“train-only 共识软目标显著提高全局拓扑一致性”。但 Ours-distill 在 Cora/CiteSeer/PubMed 上 CV 全 5/5 高于 train-only，支持“蒸馏传递更强 topology-aligned bias”。
- P1 特征噪声：Ours 随测试期 feature noise 明显退化，符合 X-only 方法共同特征依赖；Cora/CiteSeer/AmazonPhoto 多数 α 下相对优势保持，PubMed 继续是边界，CoauthorCS 中等 α 下被 JoCoR 反超。写法应是“多数场景相对优势保持 + feature-noise 不是本方法专门防御对象”，不能升级为新卖点。

## 2026-07-08 Protocol Audit

- 新增审查记录：`docs/protocol_audit_20260708.md`。
- 已快速修复两项：`summarize_mlp14_graphfree_family.py` 中 Ours 的资源标注改为“GRACE-H 预处理使用图/全边/无标签节点，但推理 X-only”；`generate_mlp17_paper_assets.py` 中 Appendix G noisy-label t-SNE 改为灰色非 train 节点、彩色 train noisy label、黑圈标出被翻转 train 节点。
- 未完成但必须跟进：graph-dependent Table C 的自动抽取可能混协议；CLNode 原实现使用 validation labels 参与 difficulty measurement；Appendix G 的 H 路径可能与当前实验 config 不一致；主表过滤中缺失 `train_percent` 的历史行仍可能通过。
- 核心方法检查结论：`frozen_grace_targets.py` 的 ball/kNN target 构造只统计 train noisy labels，clean labels 仅用于诊断字段；暂未发现核心 SoftDist target 构造标签泄漏。

## 2026-07-10 MLP-23 Dataset Breadth

- 完成 Cora_ML、Attributed-Wiki、CiteSeerFull、Coauthor-Physics 的 uniform 0.4、13 方法×5 seeds 同协议筛选；全部真实 LCSS 相对 shuffled 为 5/5 seeds 正向。
- 没有候选通过“相对最强 baseline ≥1pp 且 4/5 seeds 正”的核心胜出门。CiteSeerFull 与 JoCoR 相差 −0.08pp；Coauthor-Physics 数值第一但只 +0.02pp，均按“相当”处理。
- Coauthor-Physics 原全矩阵 GRACE OOM；已实现数学等价的 anchor-chunked InfoNCE（batch=1024）并跑满 DBLP fallback 1000 epochs。H 为 34493×256、全有限、节点/标签/边顺序一致，球纯度约 0.9235。
- 选择性噪声率：CiteSeerFull u0.2 +2.28pp、u0.6 −4.71pp；Coauthor-Physics u0.2 −0.69pp、u0.6 −2.29pp；因此二者不进入核心全谱胜出统计。
- 新论文版本：`docs/paper_final_zh_v5_mlp23_breadth_20260710.md`。正文新增 uniform0.4 数据集广度表（CiteSeerFull、Coauthor-Physics）；Cora_ML/Attributed-Wiki 留附录筛选审计。
- 汇总：`runs/mlp23_dataset_breadth_20260710/mlp23_report.md`、`docs/mlp23_dataset_breadth_table_20260710.md`。

## 2026-07-12 MLP-24 Scale-Adaptive Balls

- Implemented `fixed|sqrt_nodes` with `B_eff=max(64,ceil(sqrt(N)))`; historical fixed configs remain unchanged.
- Phase A passed: 33/33 geometry rows; Physics 64→186 balls and mean size 538.95→185.45 without runtime or vote sparsity failure.
- Phase B completed 44/44 seed0 rows but failed the preregistered gate: Physics adaptive-fixed was +0.10pp, below +0.50pp. Seven affected datasets averaged about +0.33pp.
- Stopped before 5 seeds/full spectrum. Keep fixed 64 and do not update the paper method. Results: `runs/mlp24_scaleadaptive_balls_20260712/`; plan: `docs/execution_plan_mlp24_scaleadaptive_balls_20260712.md`.

## 2026-07-12 MLP-25 Coarse-Ball Holdout Smoke

- Completed 15/15 seed0 runs on CiteSeerFull, Cora_ML, and Attributed-Wiki with five arms: fixed, vote-target real/shuffled, and frozen MLP-13 candidate real/shuffled.
- Vote-target deltas vs fixed were +0.27/+1.24/−5.29pp; frozen-candidate deltas were +0.88/+0.76/−1.96pp. Both failed the preregistered gate.
- No five-seed expansion. Keep fixed `64/32/0.05`; close the granularity axis. Results: `runs/mlp25_coarse_holdout_smoke_20260712/`.

## 2026-07-12 MLP-28 Soft-Target Robust Optimization Smoke

- Added the next-paper agenda at `docs/next_paper_research_agenda_20260712.md` and preregistered plan at `docs/execution_plan_mlp28_softtarget_robust_smoke_20260712.md`.
- Added isolated LCSS+LogitClip and LCSS+soft-small-loss predictors with real/shuffled controls. Existing LCSS and robust-loss anchors were not modified.
- Protocol: Cora/CiteSeer/CiteSeerFull, uniform 0.6, train-only noise, percent 0.2/0.1/0.7, seed0, ROCm GPU. All 21 rows succeeded; split/noise hashes matched within each dataset and val/test modified counts were zero.
- LCSS+LogitClip failed the preregistered gate because CiteSeer regressed by 0.77pp relative to the stronger single method.
- LCSS+soft-small-loss passed strongly: synergy over `max(LCSS, Small-loss-only)` was +6.02/+3.18/+9.42pp; real-shuffled was +10.40/+9.88/+18.54pp.
- This remains single-seed evidence. Next task is paired seeds 1–4 for soft-small-loss only; do not update the frozen paper or implement soft-JoCoR before that confirmation.
- Results: `runs/mlp28_softtarget_robust_20260712/`; tests: `tests/test_mlp28_softtarget_robust.py`.

### MLP-28 follow-up

- Soft-JoCoR seed0 passed the deliberately strict complexity gate: it exceeded soft-small-loss by +0.53/+3.48/+1.89pp on Cora/CiteSeer/CiteSeerFull and exceeded its shuffled control by +12.40/+11.47/+14.99pp. It remains single-seed evidence.
- Soft-small-loss was expanded to five seeds. Synergy over `max(LCSS, Small-loss-only)` was +4.75/+3.63/+6.24pp, all 5/5 paired seeds positive. Real-shuffled was +11.39/+9.75/+14.24pp, all 5/5 positive.
- Decision: Soft-small-loss is a confirmed next-paper candidate. Soft-JoCoR is retained pending a separate five-seed comparison; current frozen paper is unchanged.
- Reports: `runs/mlp28_softtarget_robust_20260712/mlp28_soft_jocor_report.md` and `mlp28_soft_small_5seed_report.md`.

## 2026-07-12 MLP-29 Breadth and Robust-Teacher Distillation

- Added six-dataset uniform0.6 seed0 breadth: CoraFull, Coauthor-CS, Amazon-Photo, PubMed, Amazon-Computers, Coauthor-Physics.
- Direct LCSS+soft-small-loss passed strongly: synergy over the stronger single method was +5.10/+2.65/+5.38/+6.50/+2.72/+2.66pp; real-shuffled was >=+6.80pp on all six.
- Added fair beta0.5 distillation arms: bare GCN[X||H], ball SoftDist, kNN consensus, robust ball SoftDist, and robust shuffled. Old beta0.0 teacher rows were not reused for attribution.
- Robust teacher distillation failed globally. PubMed dropped 15.94pp versus plain beta0.5 teacher, overall mean delta was -1.02pp, and only 3/6 datasets had >=1pp real-shuffled gain. Do not expand or tune this direct teacher plugin.
- Current paper remains frozen. Direct soft-small-loss is the next-paper candidate; all six new breadth results remain seed0 only.
- Report: `runs/mlp29_breadth_distill_seed0_20260712/mlp29_report.md`.

## 2026-07-12 MLP-30 Noise-Rate Grid Seed0

- Completed the 9-dataset x 3-rate x 4-arm seed0 matrix for LCSS, Small-loss, LCSS+soft-small-loss, and shuffled combination; total 108/108 unique cells.
- Newly ran uniform 0.2 and 0.4 on ROCm GPU; uniform 0.6 was reused from MLP-28/29 after split/noise hash checks.
- Uniform 0.2 failed the preregistered gate: 5/9 positive synergy, mean +0.46pp, one regression over 1pp, while real-shuffled remained >=1pp on 8/9 datasets.
- Uniform 0.4 passed: 6/9 positive synergy, mean +2.57pp, no regression over 1pp, real-shuffled >=1pp on 9/9.
- Uniform 0.6 passed strongly: 9/9 positive synergy, mean +4.85pp, real-shuffled >=1pp on 9/9.
- Eight of nine datasets showed monotonically increasing synergy with the noise rate. Final interpretation: the combination is a medium/high-noise method, not a universal low-noise improvement.
- Current paper and LCSS anchor remain frozen. Results: `runs/mlp30_noise_rate_grid_seed0_20260712/`; plan: `docs/execution_plan_mlp30_noise_rate_grid_seed0_20260712.md`.

## 2026-07-12 MLP-31 Representative Five-Seed Confirmation

- Added 144 ROCm GPU runs and combined them with exact-hash reusable rows into a 240/240-cell matrix; no failed, duplicate, or leaked rows.
- Uniform 0.4 passed the preregistered rate gate: 4/6 datasets met the paired synergy direction rule, mean synergy was +2.84pp, minimum was +0.10pp, and all 6/6 datasets had stable real-vs-shuffled gains. Amazon-Photo and Coauthor-Physics were only 3/5 synergy-positive and remain medium-noise boundaries.
- Uniform 0.6 passed strongly: all 6/6 datasets were 5/5 synergy-positive, mean synergy was +4.68pp, minimum was +3.27pp, and all real-vs-shuffled comparisons were 5/5 positive.
- Final claim boundary: stable cross-scale synergy under high uniform noise; conditional benefit at medium noise; no universal low-noise claim. The known synthetic noise rate is used by `forget_rate=auto` and must be disclosed.
- Current paper remains frozen. Results: `runs/mlp31_noise_rate_5seed_20260712/`; plan: `docs/execution_plan_mlp31_noise_rate_5seed_20260712.md`.

## 2026-07-12 MLP-32 Robust-Student Distillation and NRGNN Reference

- Implemented standard GCN[X||H] teacher plus equal-weight KD and train-only LCSS soft-small-loss on an X-only student, with a shuffled control and separate teacher/ball embedding paths.
- Seed0 gate failed: robust student exceeded plain distillation on only 1/6 datasets, mean delta -0.50pp, and only 2/6 real-shuffled deltas reached +1pp. No five-seed expansion or weight tuning.
- Completed exact-protocol five-seed plain distillation and NRGNN/NRGNN[X||H] comparison on six datasets. Coauthor-Physics graph-native rows were newly run and cost about 10.5 minutes per seed/method.
- Plain distillation beat NRGNN on Amazon-Photo (+1.49pp) and PubMed (+4.11pp), but lost on Cora (-1.53pp), CiteSeer (-2.23pp), CoraFull (-3.25pp), and Coauthor-Physics (-0.40pp).
- Final claim: competitive graph-free accuracy/resource trade-off, not graph-native SOTA. Direct LCSS+soft-small-loss remains the high-noise line; plain distillation remains a separate accuracy extension.
- Results: `runs/mlp32_robust_student_distill_20260712/`; record: `docs/execution_plan_mlp32_robust_student_distill_20260712.md`.

## 2026-07-13 MLP-33 High-Noise Graph-Native, Misspecification, and Noise Types

- Uniform 0.6 graph-native seed0 completed on Cora/CiteSeer/CoraFull/Amazon-Photo/PubMed. Best frozen X-only minus best graph-native was `-6.33/-8.89/-5.83/-3.75/+0.78pp`; only 1/5 met the within-2pp/win gate. No graph-native five-seed or Coauthor-Physics expansion.
- Forget-rate sensitivity at true u0.6 completed for assumed rates 0.4/0.5/0.6 on Cora/CiteSeer/CoraFull. A 0.1 error was mostly within about 1pp but CiteSeer reached -1.03pp; underestimating by 0.2 cost -1.48/-0.52/-1.14pp.
- Pair 0.4 five-seed passed: mean synergy `+1.49pp`, 4/5 qualifying datasets, minimum `-0.10pp`, and 5/5 stable real-shuffled comparisons.
- Instance 0.4 five-seed passed strongly: mean synergy `+3.04pp`, 5/5 qualifying datasets, minimum `+0.90pp`, and 5/5 stable real-shuffled comparisons.
- Integrity: 200/200 pair/instance cells, all unique/ok, no val/test label modification, and paired split/noise hashes match.
- Results: `runs/mlp33_u06_graphnative_sensitivity_noise_20260713/mlp33_report.md`; record: `docs/execution_plan_mlp33_u06_graphnative_sensitivity_noise_20260713.md`.

## 2026-07-14 MLP-34 Strong Baseline, Noise-Rate Estimation, and Selection Traces

- PubMed combination minus JoCoR was +3.78pp for pair0.4 and +4.05pp for instance0.4, both 5/5 positive. Uniform0.6 was only +1.79pp and 3/5, so do not claim a three-noise universal win.
- Noisy-only leave-one-out ball disagreement overestimated the injected rate. Mean absolute errors on Cora/CiteSeer/CoraFull were 0.120/0.132/0.233; the estimator failed its gate and no feedback training was run.
- Selection traces revised the mechanism claim: LCSS rarely changes hard target argmax. It shifts clean-class probability mass and the soft-loss ranking, raising selected-target accuracy by +3.60/+6.60pp on pair/instance while reducing final mask turnover from 0.280/0.243 to 0.105/0.083.
- Trace diagnostics are seed0 and clean-dependent post-hoc evidence; keep them separate from five-seed performance claims.
- Results: `runs/mlp34_pubmed_rhohat_mechanism_20260714/`; record: `docs/execution_plan_mlp34_pubmed_rhohat_mechanism_20260714.md`.

## 2026-07-14 MLP-35 Clean-Val Selection and ROCm Repeatability

- Implemented a preregistered clean-validation grid over forget rates `{0,0.2,0.4,0.6}` with fixed 0.5 as a nonselectable comparator. Protocol covered Cora/CiteSeer/CoraFull/Amazon-Photo/PubMed, uniform0.2/0.4/0.6, seeds 0-4.
- All 375 candidate runs and 18 repeatability runs completed with matching split/noise hashes and zero val/test label modification.
- Oracle-substitution gate passed: all 15 setting means were within 1pp of the true-rate oracle; global selected-oracle was +0.11pp and the minimum setting mean was -0.46pp.
- Fixed-default value gate passed: 12/15 setting means were nonnegative and the global selected-fixed0.5 mean was +1.26pp.
- The exact injected rate was selected in only 45/75 seed-level cells. Treat clean-val as a training-strength selector, not a noise-rate estimator. It still requires a clean validation set and must not be called fully noise-rate-free.
- Three same-seed, three-repeat audits had a maximum within-method range of 0.10pp and all paired deltas were 3/3 direction-consistent. Do not use the MLP-34 2.76pp trace discrepancy as a global ROCm variance band.
- Decision: retain a separately disclosed clean-val tuned variant; do not silently replace the frozen original method. Results: `runs/mlp35_cleanval_repeatability_20260714/`; record: `docs/execution_plan_mlp35_cleanval_repeatability_20260714.md`.

## 2026-07-16 MLP-37 GRACE Pretraining-Seed Variance

- Completed the only new experiment required by the second internal review: Cora/CiteSeer/CoraFull, GRACE seeds 0/1/2, downstream seeds 0/1/2, uniform0.4 train-only noise, percent 0.2/0.1/0.7. All 72 downstream rows and 27 geometry cells are complete and protocol matched.
- All preregistered direction and dispersion gates passed. The minimum LCSS-minus-shuffled mean delta over the three H seeds was +11.43/+11.61/+4.73pp; the minimum LCSS-minus-CE delta was +11.70/+10.81/+4.82pp. LCSS also exceeded JoCoR in every dataset-by-H-seed cell.
- Between-H-seed SD was 0.53/0.07/0.44pp, versus downstream RMS SD 0.87/1.11/0.39pp. The paper may claim directional robustness across these three datasets and three pretraining seeds, but not complete resolution of pretraining uncertainty.
- Partition ARI across H seeds averaged 0.300 (range 0.247--0.365). Treat this as evidence that exact partitions vary while aggregate local-consensus gains remain stable, not as a universal mechanism proof.
- Saved encoder/projector checkpoints, H, configs, loss traces, timing, peak GPU memory, and SHA256 for all nine pretraining runs. Average Cora/CiteSeer/CoraFull pretraining costs were 0.07/0.13/74.48 minutes and 0.40/0.67/12.73 GiB peak allocated memory.
- Evidence: `runs/mlp37_grace_seed_variance_20260716/`; plan: `docs/execution_plan_mlp37_grace_seed_variance_20260716.md`; manuscript candidate: `docs/paper_final_zh_v15_submission_candidate_20260716.md`.
- Submission packaging is complete: anonymous artifact v3, embedded v15 HTML, and the final numeric/wording/path audit all passed; the final relevant test suite reports 63 passed. The next step is author read-through and submission-form preparation. Do not reopen beta, granularity, method, or dataset search.

## 2026-07-17 MLP-39 LOO-LCSS + Soft-Small-Loss

- Ran the preregistered six-run seed0 smoke on Cora/CiteSeer/CoraFull at train-only uniform0.4. All rows completed with exact MLP-30/36 split/noise hashes and zero validation/test label modification.
- LOO-combo minus default-combo was -0.79/+0.39/-0.82pp, averaging -0.41pp. Only 1/3 datasets improved and two exceeded the allowed 0.5pp regression.
- Real LOO groups still beat shuffled by +13.46/+16.24/+6.49pp, so the failure concerns the incremental LOO switch, not local-consensus structure.
- Decision: NO-GO, no five-seed expansion or tuning. Keep LOO as a first-paper mechanism ablation; keep the default target for the next-paper soft-small-loss candidate. Evidence: `runs/mlp39_loo_soft_small_loss_smoke_20260717/`.

## 2026-07-17 MLP-40 Missing-Spectrum Seed0 Gates

- Completed 45/45 GPU runs for Cora_ML uniform0.6/pair0.4/instance0.4 and Coauthor-Physics pair0.4/instance0.4. All nine-arm cells are protocol matched with zero validation/test label modification.
- Cora_ML LCSS-minus-best was -4.20/+1.57/-2.62pp (mean -1.75pp), so it failed decisively despite all real-shuffled deltas exceeding +5.5pp.
- Coauthor-Physics was +0.09/+0.69pp (mean +0.39pp), narrowly below the frozen +0.5pp expansion gate; real-shuffled was +5.78/+5.91pp.
- Decision: no seeds 1-4 for either dataset. Keep Cora_ML as an appendix boundary and Coauthor-Physics as an existing uniform-spectrum breadth/boundary dataset. Do not alter the core table. Evidence: `runs/mlp40_coraml_coauthorph_fullspectrum_20260717/`.

## 2026-07-17 MLP-41 Fixed k=5 kNN Teacher

- Historical MLP-11 k=5 rows were direct kNN-target MLP training and were not reusable for Table 4 teacher distillation. A new fixed-k teacher method was therefore implemented and run.
- Completed 30/30 ROCm GPU rows: Cora/CiteSeer/CoraFull × uniform0.4/0.6 × seeds0-4. All rows use `matched_k=5`, exact protocol hashes, and zero validation/test label modification.
- Fixed-k5 minus ball teacher was `+0.63/-0.58/+1.21pp` at u0.4 and `-0.14/+0.13/+1.42pp` at u0.6 for Cora/CiteSeer/CoraFull. Fixed k=5 was higher in 4/6 means; ball was higher on CiteSeer u0.4 and Cora u0.6.
- Final paper decision: Main Table 4 retains the original vote-matched kNN teacher because it controls effective vote count. Fixed `k=5` is a post-freeze exploratory result and is archived for future work; it is not reported in the v15 submission manuscript or Appendix I. Do not claim that reducing k makes balls generally superior to kNN.
- Evidence: `runs/mlp41_knn_k5_teacher_20260717/`; plan: `docs/execution_plan_mlp41_knn_k5_teacher_20260717.md`; root history: `02_experiment_history_and_lessons 31.md`.

## 2026-07-17 MLP-42 Table 4 Breadth Pause

- The user later explicitly resumed this experiment. The eight valid partial rows were audited and reused; 32 missing GPU runs were completed under the original vote-matched Table 4 definition.
- Final matrix: 100/100 unique cells across Coauthor-CS/Amazon-Photo, uniform0.4/0.6, five teacher arms and seeds0-4. All rows are successful, hash-paired and free of validation/test label modification.
- Coauthor-CS ball teacher is best at u0.4/u0.6 (93.75±0.22 / 92.25±0.69); ball−kNN is +0.06/+0.59pp. Amazon-Photo is mixed: ball is 90.49±1.05 / 85.87±2.25, while kNN is 90.15±0.93 / 86.82±0.61.
- Paper decision remains pending author review. Do not modify v15 automatically. If Table 4 is expanded, report both datasets and both rates, including Amazon-Photo's negative ball−kNN u0.6 result.

## 2026-07-23 SAMTD Repository Audit

- Audited the paper-linked anonymous SAMTD snapshot before attempting a Table 5 protocol port.
- The snapshot contains 130 files, three self-supervised teacher implementations and pretrained weights, but no license, dependency manifest or usable README.
- The released entry is not executable as written: `arg_parse` requires `ites` and `pseudon`, while the only call omits both. The corresponding `iteration` and `pseudo_num` values are absent from all configs and source files.
- SAMTD's label-enhancement stage assumes nodes outside train/validation/test masks. Under the paper's fixed 20/10/70 protocol, Amazon-Photo has no such nodes and Cora/CiteSeer have only two each because of integer rounding.
- The ROCm environment also lacks `termcolor`, `torch_scatter` and `torch_sparse`, but these dependency issues are secondary to the unrecoverable algorithm settings and split mismatch.
- Decision: `not_runnable_under_same_protocol`. No classification seed0 or five-seed result was fabricated; SAMTD remains a related-work comparison and a possible rebuttal baseline if a complete official release appears.
- Evidence: `runs/samtd_protocol_port_20260723/samtd_repository_audit.md`; plan and decision record: `docs/execution_plan_samtd_protocol_port_20260723.md`.

## 2026-08-05 MLP-48 CF-BLAGB Fixed-Budget Label-Assisted Balls

- Implemented a small Stage-2-only variant: five-fold noisy labels can only re-rank label-free geometric splits through `G * (1 + JSD/log(2))`; they cannot veto splitting or change the formal `K_geo` budget.
- Added a formal-prefix candidate tree because the generic MLP-46 tree produced 51 leaves on Cora while the formal partition had 52. The revised builder reproduces the formal assignment hash exactly before extending deeper candidates; it does not lower `min_ball_size` or read labels.
- Phase A completed 21/21 target-quality cells and 60/60 fold partitions on Cora/CiteSeer/CoraFull, uniform0.4 seed0. All protocol checks passed and val/test labels remained clean.
- The preregistered gate failed: exact budget 3/3, Brier better than both geometry-only and shuffled 0/3, clean-oracle accuracy headroom 1/3, mean accuracy delta versus cross-fit formal `+0.00118`.
- Cora had no alternative cut (capacity=K=52); CiteSeer had only one extra leaf; on CoraFull, real noisy and shuffled guidance produced identical accuracy/Brier despite 128→64 re-ranking capacity.
- Decision: do not run Phase B or tune a second score. Together with CF-NAGB, direct noisy-label intervention in ball construction is closed. A denoised-teacher guidance method would be a separate research direction.
- Evidence: `runs/mlp48_cf_blagb_20260805/`; record: `docs/execution_plan_mlp48_cf_blagb_20260805.md`.

## 2026-08-05 MLP-49 Robust-Teacher-Verified GB-LCSS

- Tested whether an independently trained NRGNN can verify formal LOO-GB corrections without changing GRACE H, ball construction, local target content, beta, or the X-only student.
- Completed 36/36 Phase A target rows on Cora/CiteSeer/CoraFull, uniform0.4/0.6, seed0. Six NRGNN teachers were retrained; split/noise/cache/probability audits all passed.
- Preregistered gate failed: UTV beat formal LOO and shuffled UTV in Brier on 0/6 cells; mean target-accuracy delta versus formal was -1.335pp; Cora u0.4 and CiteSeer u0.4 lost 6.470pp and 2.256pp.
- Real balls still beat shuffled balls in Brier on 6/6, but UTV was worse than formal LOO in Brier on 6/6. Clean-oracle verification also had 0/6 Brier headroom.
- Mechanism diagnosis: NRGNN posteriors were useful as direct targets, but entropy-based multiplicative verification reduced mean alpha to roughly 0.006--0.032 and suppressed both useful and harmful corrections. MC stability was not the bottleneck.
- Decision: no downstream Phase B and no second verifier formula. Confidence-only teacher attenuation is closed; teacher knowledge must enter through a different information path if revisited.
- Evidence: `runs/mlp49_utv_gblcss_20260805/`; plan: `docs/execution_plan_mlp49_utv_gblcss_20260805.md`.

## 2026-08-05 MLP-50 LOO Minimum-Vote Gate

- Re-tested the old MLP-43 minimum-support idea under the current formal lineage: `tie_mode=distribution_only`, `leave_one_out=true`; old MLP-43 was self-inclusive and is not interchangeable with this result.
- Frozen grid: Cora/CiteSeer/CoraFull, uniform0.4/0.6, seed0, real and size-matched shuffled targets, `s_min in {2,5,10}`. The only candidate was `s_min=5`; `s_min=10` was diagnostic only.
- Completed 36/36 target rows. No real ball had zero training votes in any of the six cells, so ignoring label-empty balls is already a no-op under the current 20% split and ball scale.
- `s_min=5` improved Brier on 0/6 cells and worsened mean Brier by `+0.003886`; mean target-accuracy delta was only `+0.137pp`. Coverage remained 96.67--99.90%, so the failure is not explained by excessive coverage loss.
- Real targets still beat shuffled targets in Brier on 6/6 cells. The local geometry remains useful, but hard support filtering removes useful probability information already handled conservatively by LOO alpha.
- Decision: `abandon` hard minimum-vote gating as a current-method improvement. Keep formal `s_min=2`; do not run downstream Phase B or tune another threshold.
- Evidence: `runs/mlp50_loo_min_vote_gate_20260805/`; plan: `docs/execution_plan_mlp50_loo_min_vote_gate_20260805.md`.
