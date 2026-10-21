# GB-GCL Experiment History and Lessons - Experiment 7

本文档只整理 `实验7noisy_label_gb_grace` 方向的证据。它是对既有 `02_experiment_history_and_lessons.md` 的独立补充，不覆盖旧的 6 个方向。

# 1. Current Overall Judgment

- Confirmed: 实验7是一个新方向：GB 不再修改 GRACE pretraining 的 view/loss，而是在 frozen GRACE 表示后的 noisy-label classifier 阶段做 local label smoothing。
- Confirmed: `GB-strict` 没有通过强门控。Planetoid 低标签设置覆盖率太低；medium-label 提高覆盖率后，仍然只在 Cora 部分设置正向，CiteSeer 不稳定。
- Likely: 低标签覆盖不足解释了 Planetoid 失败的一部分，但不能解释全部失败；coverage 提高后仍缺少稳定 GB-specific advantage。
- Inconclusive: Cora train 20%, 40% noise 的正向结果可以作为弱信号，但不足以支持主方法 claim。
- Risk: 只汇报 Cora 或只比较 global smoothing 会高估方法；必须同时比较 CE、global、kmeans、topk_matched，并报告 paired deltas。
- Next priority: 不继续扩展 strict ball-only smoothing 为主方法；若继续，应重定义为 coverage-aware 或 prior-corrected local smoothing。

# 2. Direction Index

| Direction ID | Direction Name | GB Role | Versions Included | Datasets | Best Trend | Main Failure | Decision | Paper Value |
|---|---|---|---|---|---|---|---|---|
| DIR-07 | Noisy-label GB-GRACE frozen local smoothing | 在 frozen GRACE embedding 上构造 GB，用 noisy train labels 生成局部 soft labels | Planetoid low-label frozen gate; medium-label ratio frozen gate; `gb_prior` diagnostic | Cora, CiteSeer | Cora train 20%, noise 0.4: GB-strict mean 0.7974，高于 CE/global/kmeans/topk_matched | CiteSeer 不稳定；Planetoid coverage 低；medium-label coverage 提高后仍不能稳定超过 CE/kmeans/topk | modify | ablation only / motivation |

# 3. Direction-Level Detailed Records

## DIR-07: Noisy-label GB-GRACE Frozen Local Smoothing

### 3.1 Direction Summary

- Direction ID: DIR-07
- Direction name: Noisy-label GB-GRACE frozen local smoothing
- Time period: 2026-05-06
- Target problem: 在训练标签带噪声的节点分类中，测试 frozen GRACE 表示空间里的 granular balls 是否能提供局部可信 label smoothing。
- Core hypothesis: GRACE 无标签预训练后的表示空间中，局部区域可能比单个 noisy train label 更可靠；GB 可作为 confidence-weighted local smoothing unit。
- Baseline: `ce`, `global`, `kmeans`, `topk_matched`; auxiliary diagnostic method `gb_prior`。
- Main related files: `实验7noisy_label_gb_grace/AGENT.MD`, `README.md`, `configs/gate.yaml`, `src/noisy_label_gb_grace/smoothing/targets.py`, `scripts/run_gate_experiment.py`, `scripts/summarize_gate_results.py`
- Result files: `实验7noisy_label_gb_grace/reports/formal_frozen_gate_report.md`, `reports/medium_label_frozen_gate_report.md`, `reports/gate_summary.csv`, `reports/gate_paired_deltas.csv`
- Raw logs: `实验7noisy_label_gb_grace/logs/gate/20260506_131858`, `logs/gate/20260506_133542`, `logs/gate/20260506_143637`, `logs/medium_label_gate_20260506_143631.out`
- Current decision: modify

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: 不是 clean GRACE pretraining 弱点，而是 noisy train-label classifier 阶段容易被错误标签带偏。
- Why GB might help: GB 在 frozen GRACE representation 上提供局部分组，用组内 noisy train-label distribution 平滑单个标签。
- Expected improvement: 在 symmetric label noise 20% 或 40% 下，比 CE/global/kmeans/topk_matched 更稳。
- Literature or previous experiment inspiration: 旧方向显示直接把 GB 强塞进 GRACE pretraining 不稳定；实验7改为后处理/分类阶段的弱监督鲁棒性测试。
- Expected paper claim: Granular balls can serve as a local, confidence-weighted label smoothing prior for noisy-label node classification after self-supervised graph contrastive pretraining.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| Planetoid low-label frozen gate | fixed Planetoid split, noise 0/0.2/0.4, seeds 0-4, alpha 0.3/0.5 | Cora, CiteSeer | Cora 20/40% 有弱正向；CiteSeer coverage 很低 | 低标签设置下 strict GB local smoothing 是否成立 | 不通过强门控；Cora coverage ~=0.197, CiteSeer coverage ~=0.075，不能支持 GB-specific claim |
| Medium-label ratio frozen gate | ratio split train 10%/20%, val 10%, noise 0/0.2/0.4, seeds 0-4 | Cora, CiteSeer | Cora train 20%, 40% noise 最强；CiteSeer 多数被 CE/global/kmeans 匹配或超过 | coverage 提高后是否能验证 strict GB | coverage 提高但跨数据集仍不稳定；覆盖率不是唯一瓶颈 |
| `gb_prior` diagnostic | 用 global prior 修正低样本 ball distribution | Cora, CiteSeer | 个别 Cora 设置有帮助，但不是 strict GB claim | 低 train-label density 是否需要 prior fallback | 只能作为诊断或新方向线索，不能救回 strict GB 主张 |

### 3.4 Role of Granular Ball

- GB role: classifier-stage local label smoothing unit。
- Where GB is generated: frozen GRACE encoder output `H`，L2-normalized 后构球。
- Representation space: GRACE encoder representation, not classifier logits。
- Affects loss: yes，soft-label classifier cross entropy。
- Affects augmentation: no。
- Affects graph structure: no。
- Affects sampling: indirectly，决定哪些 train nodes 用同一 local label distribution。
- Affects evaluation: no；validation/test labels remain clean。

### 3.5 Result Summary

- Best result: medium-label Cora train 20%, noise 0.4, alpha 0.3: CE 0.7935, global 0.7830, kmeans 0.7945, topk_matched 0.7919, GB-strict 0.7974。
- Weakest trend: CiteSeer medium-label train 20%, noise 0.2: GB-strict 0.7009，低于 CE 0.7064、global 0.7082、kmeans 0.7068，只略高于 topk_matched 0.7001。
- Average trend: Cora 有局部正向信号，尤其 train 20% / 40% noise；CiteSeer 不支持该 claim。
- Mean/std: reports 提供 5-seed means；`reports/gate_summary.csv` 含 std；`reports/gate_paired_deltas.csv` 含 paired delta std。
- Baseline mean/std: CE/global/kmeans/topk_matched 在同协议下均有记录。
- Delta: medium-label Cora train 20%, 40% noise paired deltas for GB-strict: vs CE +0.0039, vs global +0.0144, vs kmeans +0.0029, vs topk_matched +0.0055。CiteSeer train 20%, 20% noise: vs CE -0.0055, vs global -0.0073, vs kmeans -0.0059, vs topk_matched +0.0008。
- Std overlap: 多数小 delta 可能与 std overlap；没有 paired significance 前不能声称显著。
- Datasets improved: Cora selected medium-label noisy settings。
- Datasets worsened / unstable: CiteSeer。
- Runtime/memory cost: not summarized as a blocker；若写论文需要补充。
- Raw evidence: `formal_frozen_gate_report.md`, `medium_label_frozen_gate_report.md`, `gate_summary.csv`, `gate_paired_deltas.csv`。

### 3.6 What Is Directly Supported

实验7直接支持一个窄结论：strict GB local smoothing after frozen GRACE pretraining 可以在 Cora medium-label high-noise 设置下出现正向信号，但不能在 Cora 和 CiteSeer 上稳定超过 CE/global/kmeans/topk_matched。Planetoid 失败部分来自 effective smoothing coverage 太低；medium-label 结果说明 coverage alone 不足以建立稳定 GB-specific advantage。

### 3.7 Reasonable Interpretation

解释：GB local label distribution 需要足够多 train labels 落在 ball 内才有意义；即使 coverage 足够，GB 分组质量也依赖数据集表示几何。Cora train 20% / 40% noise 可能符合预期机制，但 CiteSeer 表明 GB partition 未必比 kmeans 或 topk neighbor smoothing 更可靠。

### 3.8 What Cannot Be Concluded

- 不能声称 GB-strict local smoothing 已经验证为主方法。
- 不能声称 GB 改善 GRACE pretraining；这里 GRACE 是 frozen 的，GB 只在 classifier stage 使用。
- 不能声称 GB 普遍优于 kmeans 或 topk_matched。
- 不能把 `gb_prior` 当作 strict GB 的证据。
- 不能基于该结果直接进入 PubMed、asymmetric noise、instance-dependent noise 或 fine-tuning。

### 3.9 Failure Analysis

- Main failure reason: cross-dataset instability；CiteSeer 即使 coverage 提高也不支持 strict GB local smoothing。
- Secondary failure reason: Planetoid low-label split 中每个 ball 内 train labels 太少，strict smoothing 被大量禁用或弱化。
- Evidence: `formal_frozen_gate_report.md` 报告 low coverage and weak signal；`medium_label_frozen_gate_report.md` 报告 coverage improved but CiteSeer unstable。
- Alternative explanation: classifier/hyperparameter protocol 可能不是 noisy-label classification 最优，但改变协议会构成新方向，必须保留所有 baseline。
- Minimum check needed: 若复活，应做 coverage-aware/prior-corrected smoothing，并预注册成功标准和 paired significance，而不是继续调 strict GB。

### 3.10 Lessons Learned

- Lesson: Low train-label density can make ball-level local label distribution unusable。Evidence: Planetoid coverage Cora ~=0.197 and CiteSeer ~=0.075。Future constraint: report effective smoothing coverage before interpreting GB label smoothing failures。
- Lesson: Coverage improvement is necessary but not sufficient。Evidence: medium-label CiteSeer coverage rises to ~=0.311/0.580 but GB-strict remains unstable。Future constraint: do not equate coverage with GB-specific value。
- Lesson: kmeans and no-ball local baselines are mandatory。Evidence: GB often ties or trails CE/kmeans/topk_matched outside the best Cora setting。Future constraint: no noisy-label GB claim without these controls。
- Lesson: `gb_prior` is a different hypothesis。Evidence: reports explicitly keep it auxiliary。Future constraint: do not present prior-corrected smoothing as validation of strict ball-only smoothing。

### 3.11 What Is Still Useful

- 可复用 noisy-label artifact、strict GB smoothing、global/kmeans/topk controls、paired delta summaries、coverage diagnostics。
- 可作为“GB post-processing 比直接修改 GRACE pretraining 更安全，但仍不自动有效”的研究动机。
- 可作为 coverage-aware 或 prior-corrected local smoothing 新方向的弱动机。

### 3.12 Decision

- Decision: modify。
- Reason: strict GB 有窄的 Cora-positive signal 和有用诊断，但没有通过跨数据集 gate。
- What should not be tried again: 不继续把 strict GB local smoothing 扩展为 primary method；不在重定义方法前加入 PubMed/asymmetric/fine-tuning。
- Variant still worth trying: coverage-aware 或 prior-corrected local smoothing with separate hypothesis and acceptance criteria。
- Evidence required to revive this direction: Cora 和 CiteSeer 上对 CE/global/kmeans/topk_matched 的稳定正 paired deltas，统计可靠性，以及证明 GB grouping 相对 kmeans/topk 有有用差异的诊断。

### Experimental Setup Summary

- Datasets: Cora, CiteSeer。
- Seeds: 0, 1, 2, 3, 4；`pretrain_seed=noise_seed=classifier_seed=seed`。
- Encoder: frozen GRACE encoder from upstream GRACE source。
- Base loss: classifier soft cross entropy on frozen embeddings。
- Added mechanism: local label smoothing from GB train-label distribution。
- Alpha: primary 0.3；0.5 sensitivity only。
- Ball generation method: recursive kmeans-style granular ball splitting in `src/noisy_label_gb_grace/balls/granular_ball.py`。
- Small-ball fallback: `min_train_per_ball=3`；strict GB disables smoothing below this threshold。
- Evaluation protocol: clean validation early stopping, clean test at best validation。
- Split protocols: Planetoid fixed split and separate ratio split；两者不能混作同一性能比较。

### Baseline Comparability Check

- Same dataset: yes within each table。
- Same split: yes within Planetoid or ratio protocol；Planetoid and ratio are separate protocols。
- Same seeds: yes, 5 seeds for formal gate summaries。
- Same encoder/projector: same frozen GRACE protocol within each run group。
- Same classifier protocol: yes, 500 epochs, patience 50 in config/report。
- Same noisy labels: yes by protocol；noise artifacts are saved and shared by methods for each seed/rate。
- Baseline rerun: yes, reports include CE/global/kmeans/topk_matched。
- Comparable conclusion: yes within each protocol；no direct comparison between Planetoid and ratio as performance claims。

### Label Leakage Check

- Labels used in GRACE pretraining: no evidence；stage 1 remains unsupervised。
- Labels used in GB generation: no class labels；balls are generated from normalized embeddings。
- Labels used in smoothing: noisy train labels only。
- Validation/test labels used outside evaluation: no evidence；reports specify clean validation and clean test labels for early stopping/evaluation。
- Supervised GB mechanism used in pretraining: no。
- Leakage risk: low，但该方向是 supervised/noisy-label classification after pretraining，不是 unsupervised GCL main-method evidence。

### Granular-Ball Diagnostics

- Effective coverage Planetoid: Cora ~=0.197, CiteSeer ~=0.075。
- Effective coverage medium-label: Cora train 10% ~=0.368, Cora train 20% ~=0.638, CiteSeer train 10% ~=0.311, CiteSeer train 20% ~=0.580。
- GB-kmeans assignment NMI and GB-topk overlap: available in `gate_summary.csv` fields, but reports do not interpret them as supporting a strong claim。
- Missing diagnostics for paper: ball radius distribution, singleton ratio by seed, coverage by class interpretation, paired statistical tests, runtime。

# 4. Cross-Direction Failed Pattern Ledger

## FP-7: Coverage Alone Does Not Validate GB Local Smoothing

### Status
confirmed

### Related Directions
DIR-07

### Observation
Planetoid low-label coverage blocks interpretation, but medium-label ratio splits improve coverage without producing stable CiteSeer gains。

### Possible Reason
GB grouping in frozen GRACE space may not align with noisy-label denoising needs on all datasets；generic local smoothing can be equally good。

### Evidence
`formal_frozen_gate_report.md`, `medium_label_frozen_gate_report.md`, `gate_summary.csv`, `gate_paired_deltas.csv`。

### Future Constraint
Every noisy-label GB direction must report effective smoothing coverage and must beat kmeans/topk controls, not only global smoothing。

### When to Revisit
Only with a new coverage-aware or prior-corrected hypothesis and predeclared paired-delta success criteria。

# 5. Cross-Direction Positive Signal Ledger

## PS-7: Post-pretraining GB Use Is Safer Than Pretraining Intervention, But Still Weak

### Status
possible

### Related Directions
DIR-07

### Observation
DIR-07 avoids modifying GRACE pretraining and shows a Cora-positive noisy-label signal in the medium-label high-noise setting。

### Why It May Matter
It provides a safer framing for GB as an auxiliary downstream prior rather than a hard contrastive relation。

### Required Verification
Cora + CiteSeer stable paired gains over CE/global/kmeans/topk_matched；diagnostics proving GB local distributions are better than generic local controls。

### Possible Paper Use
motivation / ablation, not main method yet。

# 6. Anti-Repetition Memory

## Do Not Retry Without New Evidence

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| Strict GB local smoothing as main noisy-label method | Cora-positive but CiteSeer unstable；medium-label coverage did not fix cross-dataset weakness | `medium_label_frozen_gate_report.md` | Stable paired gains over CE/global/kmeans/topk on both Cora and CiteSeer |
| Treating Planetoid low-label failure as GB mechanism failure | Coverage too low to estimate local label distributions | `formal_frozen_gate_report.md` | Medium-label or another protocol with coverage >=0.3 before mechanism conclusion |
| Using `gb_prior` to rescue strict GB claim | Prior-corrected smoothing is a different method | `AGENT.MD`, `medium_label_frozen_gate_report.md` | Separate hypothesis and baselines for prior-corrected smoothing |

## Failed Form, Not Necessarily Failed Idea

| Failed Form | Underlying Signal That May Remain Useful | Safer Future Form |
|---|---|---|
| Strict ball-only local smoothing | Local representation regions may still carry denoising signal | Coverage-aware or prior-corrected local smoothing |
| Planetoid low-label strict smoothing | GB may need enough labeled nodes per ball to estimate distributions | Medium-label or confidence-calibrated protocol |
| `gb_prior` as auxiliary diagnostic | Global prior may stabilize sparse balls | Treat as a new method, not as strict GB evidence |

## Minimum Bar Before a Direction Enters Paper Main Line

- Comparable baseline: same split protocol, same seeds, same frozen GRACE embeddings, same noisy-label artifacts。
- Multi-seed evidence: 5 seeds minimum for Cora and CiteSeer。
- Multi-dataset evidence: Cora and CiteSeer must both support the claim before PubMed/fine-tuning expansion。
- Ablation removing GB signal: global, kmeans, and topk_matched are mandatory。
- Mechanism diagnostic: coverage, GB-kmeans NMI, GB-topk overlap, coverage-by-class, ball size/radius。
- No label leakage: labels only in noisy train-label classifier stage and clean validation/test evaluation。

# 7. Direction-Level Lessons for Future Experiments

- Do not judge success only from Cora。
- Do not mix Planetoid split and ratio split as a single protocol。
- Do not use validation/test labels for smoothing or ball construction。
- Do not present downstream noisy-label smoothing as evidence that GB improves GRACE pretraining。
- Every noisy-label GB experiment must report effective smoothing coverage。
- Every noisy-label GB experiment must include CE/global/kmeans/topk_matched controls。
- If coverage improves but kmeans/topk still match or beat GB, downgrade the GB-specific claim。
- If `gb_prior` helps, create a new hypothesis instead of using it to rescue strict GB。
- If improvement is smaller than overlapping std, mark it inconclusive or weak。

# 8. Current Multi-Direction Summary

| Direction | Main Idea | Overall Result | Most Likely Failure Reason | Still Useful? | Decision |
|---|---|---|---|---|---|
| DIR-07 | Frozen GRACE embedding + GB strict local label smoothing under noisy train labels | Cora-positive in selected medium-label high-noise setting, but CiteSeer unstable | Low label coverage plus dataset-dependent GB grouping quality | Yes, as diagnostic and motivation for coverage-aware smoothing | modify |

# 9. Recommended Next Directions

## NEXT-DIR-7: Coverage-aware / Prior-corrected Noisy-label Local Smoothing

### Why This Direction Is Recommended
DIR-07 shows strict ball-only smoothing is too brittle, but also shows low coverage and local distribution reliability are central failure modes。

### Previous Failure It Addresses
Planetoid low coverage and CiteSeer instability under strict GB。

### Core Hypothesis
GB can be useful only when local noisy-label distributions are coverage-aware and calibrated against a global or confidence prior。

### GB Role
Downstream local smoothing confidence prior；no GRACE pretraining modification。

### What Must Be Different From Failed Directions
Do not call it strict GB；include CE/global/kmeans/topk/gb-strict/gb-prior baselines and paired deltas。

### Minimal Implementation
Reuse existing `gb_prior` path, add explicit calibration/coverage bins and success criteria before more datasets。

### Minimal Datasets and Seeds
Cora + CiteSeer, ratio train 10%/20%, 5 seeds, symmetric 20%/40% first。

### Success Criteria
Positive paired deltas vs all critical baselines on both datasets, with coverage-bin diagnostics showing why the method works。

### Failure Criteria
kmeans/topk or global smoothing matches it, or gains only appear on Cora。

### Paper Claim If Successful
Coverage-aware GB local smoothing can improve noisy-label classification after frozen SSL pretraining。

### Reason to Stop Early
First 5-seed Cora/CiteSeer ratio protocol repeats CiteSeer instability。

# 10. Paper Claim Implications

## Claim-7

### Claim
Granular balls can serve as local label smoothing units for noisy-label node classification after GRACE pretraining。

### Supported By Which Directions
DIR-07 partly, mainly Cora medium-label train 20% / 40% noise。

### Weakened By Which Directions
DIR-07 CiteSeer results and Planetoid low-coverage gate。

### Current Evidence
Weak and dataset-dependent。

### Missing Evidence
Stable cross-dataset paired deltas, stronger GB-vs-kmeans/topk diagnostics, statistical reliability, PubMed/asymmetric noise only after gate pass。

### Required Experiments
NEXT-DIR-7 if continuing。

### Risk
High if phrased as validated strict GB；moderate if phrased as a failed/partial gate motivating coverage-aware smoothing。

### Current Readiness
weak

# 11. TODO List

## Missing raw logs

- Map every DIR-07 summary row to exact `results/gate_runs/*` row and `logs/gate/*` run。

## Missing baseline checks

- No PubMed baseline should be run until Cora/CiteSeer gate passes。

## Missing seeds/std

- Std exists in CSV；still need paired significance or confidence intervals for GB vs CE/global/kmeans/topk。

## Missing datasets

- PubMed, asymmetric noise, instance-dependent noise are intentionally deferred。

## Missing ablations

- Need confidence-component ablation only if a redesigned method passes the basic gate。

## Missing GB diagnostics

- Need interpretation of `gb_kmeans_assignment_nmi`, `gb_topk_selected_neighbor_overlap`, coverage-by-class, ball size/radius fields。

## Missing visualizations

- Coverage bins vs paired delta。
- GB vs kmeans/topk local-label distribution difference。
- Cora vs CiteSeer ball train-label count distributions。

## Missing explanation for failed directions

- Need deeper explanation of why CiteSeer remains unstable after coverage improves。

## Missing paper-claim evidence

- Strict GB noisy-label smoothing is not ready as a main paper claim。
- Prior-corrected smoothing would require a separate report and should not overwrite the strict-gate conclusion。
