# Methodology and Reproducibility Review

审查对象：`paper_final_zh_v29_human_style_20260727.md`  
目标期刊：Elsevier *Information Sciences*  
审查范围：问题定义、方法闭合、实验协议、数据使用边界、因果解释、复杂度及复现资产。审查以 C-001/002/007/009/010/011/013/017/018/019/020/021/024/025 为入口，仅依据指定稿件、claim–evidence ledger、material manifest 及其中指向的本地配置/结果资产；未联网，未运行新实验。

## Overall recommendation

**Major revision（当前不具备可接收性，但大部分问题可通过证据链补齐和收缩表述解决）。**

核心算法的训练—部署资源流已经基本讲清，train-only label noise、transductive 协议和 X-only 部署边界也总体成立。然而，稿件仍有四类会触发方法学拒稿的问题：第一，\(\alpha\) 与最小索引平票处理在方法定义上不具类别置换等变性，但正式目标中的 tie 发生率和经验影响尚未审计；第二，25-setting paired comparison 的 hash 绑定没有在当前证据入口中闭合；第三，稿件把“标签无关构球”多次扩写成“不依赖干净标签的整个过程”，与 clean validation model selection 冲突；第四，若干机制结论超过消融能够识别的因果范围，且 §5.4 含有当前 raw locator 无法支持的数值与对象。

如果作者修复 MET-01、MET-03、MET-04、MET-06 和 MET-09，并按其余条目收缩 claim，我认为不需要全面重跑主实验。MET-01 的初始修复也只是从现有资产审计 tie 暴露率；只有审计确认 tie 非零且足以改变正式目标时，才条件性升级到受影响设置的对照实验。

## Confidence

**High（0.92）**。方法公式、伪代码、显式 Cora 蒸馏配置、协议审计、关键 seed-level/aggregate CSV 和资源表均已交叉核对。限制是：当前材料未提供可执行源码入口、全部数据集配置和部分 encoder checkpoint，因此无法从代码层验证每一处实现细节；本审查也未运行实验。

## Strengths

1. **任务边界总体清楚。** §3.1 明确研究同一图上的 transductive node classification，并明确不声称 unseen-node 或跨图 inductive 泛化；C-020 与协议审计一致。
2. **基础版与蒸馏版的部署输入边界明确。** 基础版用 \(X,A\) 生成冻结 \(H\) 和粒球目标，部署仅保留 \(f_\theta(X)\)；蒸馏版训练期增加 GCN 教师、\(H\) 和全节点教师预测，部署仍仅保留 X-only student。C-019、Cora YAML 和蒸馏 raw 中的 `teacher_use_xh=1`、`student_input_dim=d_X` 支持这一资源流。
3. **训练标签污染边界设计正确。** `formal_main_raw.csv` 中可核验的行均为 train-only noise，`modified_val_count=modified_test_count=0`；稿件也明确 clean validation 用于选择、clean test 仅用于评估。
4. **基础软目标定义具备公式闭合。** \(p_j,m_j,r_j,q_j,\alpha_j,t_v\) 的关系、\(\beta=0.5\)、\(s_{\min}=2\) 以及小球回退到 noisy one-hot 均已写明，算法 1 与正文基本一致。
5. **负结果和资源不对等边界披露较好。** 稿件承认 PubMed 等数据集失败、H-access 不是同资源比较、蒸馏学生与图模型是跨部署参照，并把 GRACE 预训练与下游成本分开报告。

## Boundary audit

| 审计项 | 当前判断 | 依据与边界 |
|---|---|---|
| Label-independent | **仅 GRACE 预训练和粒球成员划分成立** | \(H=g_\psi(X,A)\) 与几何划分不读标签；式 (7)–(17) 明确使用 noisy train labels，checkpoint selection 使用 clean validation labels。完整流程不是 label-free。 |
| 基础版 \(\beta\) | **0.5** | §3.4 式 (9)；`mlp36_raw.csv` 的基础/LOO 行为 `target_beta=0.5`。 |
| 蒸馏教师 `target_beta` | **0.0** | §3.5 式 (13)；Cora YAML 为 `target_beta: 0.0`、`teacher_target: soft_target`。这不是基础版目标的直接复用。 |
| 蒸馏学生监督范围 | **全部 \(N\) 个节点** | §3.5 式 (17) 使用全节点教师分布；属于 transductive all-node distillation。 |
| 测试节点的训练期可见性 | **特征与图结构可见，测试标签不可见** | GRACE、教师和学生蒸馏均可使用同一图中的 test-node covariates/edges；这在 transductive 协议下不是标签泄漏，但不能称 inductive。 |
| Pairing | **局部可核验，25-setting 全链未闭合** | continuation formal raw 可核验 3 数据集的 hash 与污染列；`primary_statistics_raw.csv` 只有 `dataset,noise_type,noise_rate,seed,method,test`，不能单独核验 hash 绑定。 |
| 端到端复现 | **部分可复现** | Cora/CiteSeer/CoraFull 有多 H-seed checkpoint/H manifest；Coauthor-CS 与 Amazon-Photo 仅归档 H，且当前 manifest 无源码、环境锁文件和完整数据集配置入口。 |

## Top rejection blockers

- **MET-01：** 最小索引 tie-break 不具类别置换等变性，且 \(\alpha_j\) 被表述为“置信度”但没有样本量或类先验校准；tie 是否实际进入正式目标及其影响量级尚未审计。
- **MET-03：** 25-setting paired claim 的 split/noise hash 与 seed-level test 行在当前审计入口中未形成可独立复核的一一映射。
- **MET-04：** 当前资产只能支持部分 downstream replay，不能支持论文所暗示的完整 end-to-end reconstruction。
- **MET-06：** 蒸馏扩展同时改变目标 \(\beta\)、教师输入、图传播、监督覆盖和学生损失，现有表 5 不能识别“全节点覆盖/局部共识导致增益”的唯一因果路径。
- **MET-09：** §5.4 的 noisy-CE GRACE、MLP-hidden 和跨数据集 SSL 塌缩陈述超出已定位 raw 的支持范围。

## Issues

### MET-01
- Severity: P1
- Manuscript location: §3.4.1–§3.4.2，v29 L176–216，尤其 L185、L196–214
- Problem: 已证事实是当前最小索引 tie-break 不具类别置换等变性：一旦有效球出现最高票并列，目标会随类别编号置换而改变；同时，\(\alpha_j\) 只是多数票占比的线性重标定，不是经过样本量或类先验校准的统计置信度。稿件设 \(s_{\min}=2\)，当两个训练节点标签不同且 \(C>2\) 时，\(r_j=0.5>1/C\)，因此 \(\alpha_j>0\)，`argmax` 又把 \(m_j\) 固定为索引最小类别。尚未验证的事实是：tie 在正式 dataset×setting×seed 中是否实际发生、覆盖多少训练节点、承载多少 \(\alpha\) 质量，以及是否达到影响主结果的量级。
- Evidence: 稿件 L185 明确并列时取最小索引，L192 固定 \(\beta=0.5\)，L199–210 定义 \(\alpha_j,t_v\)，L214 设 \(s_{\min}=2\)。例如 \(C=7\)、两票平分时，\(\alpha=(0.5-1/7)/(1-1/7)=5/12\)，但“多数类”完全由类别索引决定。C-001/C-021 只支持软目标的资源边界，不能消除这一算法偏置。
- Rejection impact: 类别置换非等变性涉及核心目标定义，因此可保留 P1；但在发生率审计前，不能把它写成已污染正式性能的经验结论。若不先量化暴露率，作者和审稿人都无法判断这是仅需披露的零/低频边界，还是需要训练对照的实质风险。
- Minimal fix: 第一步从现有正式软目标、球内标签计数或目标生成日志导出 dataset×setting×seed 级审计表，至少报告 tie 球数、受影响训练节点数及占比、这些节点对应的 \(\sum\alpha\) 与总 \(\sum\alpha\) 占比、被最小索引选中的类别分布，并用类别置换检查目标是否变化；这一步不需要新训练。若所有正式设置无 tie，或受影响目标质量可忽略，只需把平票规则改为 tie-safe 定义、把 \(\alpha_j\) 攵称“启发式共识权重”并披露审计。仅当 tie 非零且足以改变正式目标时，才在受影响的代表性设置上比较“\(\alpha=0\)”或“仅用 \(p_j\)”等 tie-safe 规则；只有代表设置显示结果敏感时，再扩展到完整五种子范围。若继续称“置信度”，还需解释其为何不随 \(|T_j|\) 与经验类先验变化。
- New experiment required: NO
- Claim decision: TEST

### MET-02
- Severity: P1
- Manuscript location: 摘要 L19；§1 L33–35、L42；§3.4.2 L216；§4.2–§4.3 L381、L385
- Problem: “整个过程不依赖干净标签”“不额外使用干净标签”与实际 clean validation model/checkpoint selection 冲突。严格成立的只是 GRACE 预训练和粒球成员划分 label-independent，以及训练损失不直接读取 clean labels；完整开发与训练协议仍使用 clean validation labels。
- Evidence: C-021 的允许边界是“GRACE/ball construction label-free，soft-target training 使用 noisy train labels”，并明确禁止把完整 GB-LCSS 称为 label-free。`protocol_audit_report.md` 写明 “Validation: clean labels, used for model/config selection”；稿件 L381、L385 也承认验证集用于模型选择。
- Rejection impact: 该矛盾会被视为监督资源隐瞒，并使“无需干净标签”的现实适用性主张失真；它不构成 test-label leakage，但必须准确披露。
- Minimal fix: 全文统一改为“构球阶段标签无关；训练目标不使用 clean labels；实验协议使用独立 clean validation labels 做 checkpoint/超参数选择”。若目标应用中不存在 clean validation，应明确这是当前实验假设，而非方法已验证的能力。
- New experiment required: NO
- Claim decision: WEAKEN

### MET-03
- Severity: P1
- Manuscript location: §4.2–§4.4，v29 L377–395；§5.1 的 25-setting paired claims；C-002
- Problem: 当前证据入口不足以独立核验全部 25 settings 的 paired split/noise realization。`formal_main_raw.csv` 可核验 train-only、val/test 零污染和 hash，但只覆盖 Cora/CiteSeer/Coauthor-CS 的 continuation 子集；C-002 指向的 `primary_statistics_raw.csv` 只有测试结果六列，没有 `split_hash`、`noise_hash` 或污染计数。因此，“按同一键检查四方法”只能确认 seed 标签重合，不能确认相同 realization。
- Evidence: 本地检查显示 `formal_main_raw.csv` 有 135 行、数据集仅 `cora,citeseer,coauthorcs`、seed 0–4，且最大 `modified_val_count/modified_test_count` 均为 0；`primary_statistics_raw.csv` 的列仅为 `dataset,noise_type,noise_rate,seed,method,test`。C-002 自身称扩展主表依赖 source lineage，但该 claim 的 evidence files 没有给出可将每个统计行绑定到 source hash 的审计表。
- Rejection impact: 配对检验的有效性取决于同一 split/noise realization。若 pairing 不能由发布资产直接验证，24/25、Holm 检验和方向计数的统计基础会受到质疑。
- Minimal fix: 增加一张 machine-readable protocol ledger，至少包含 `(dataset, noise_type, noise_rate, seed, method, split_hash, noise_hash, modified_train_count, modified_val_count, modified_test_count, source_file_sha256)`，并用唯一键连接 `primary_statistics_raw.csv`；发布生成该 ledger 与统计表的脚本。
- New experiment required: NO
- Claim decision: TEST

### MET-04
- Severity: P1
- Manuscript location: §4.4 L393–395；§8 L594–596；material manifest“扩展实验资产/已知边界”
- Problem: 稿件暗示补充材料可完整复现，但当前资产实际分为两级：已有 H 时可复跑 downstream；从原始图端到端重建 H 并复跑全部方法则不完整。当前 manifest 仅显式列出一份 Cora 蒸馏 YAML，没有可执行源码入口、环境锁文件、命令、全部数据集配置或配置 ID 到配置快照的映射；Coauthor-CS/Amazon-Photo 缺 encoder checkpoint。
- Evidence: §8 已承认 Coauthor-CS 与 Amazon-Photo “未包含精确重建原始表征所需的 encoder checkpoint”。material manifest 列有 Cora 蒸馏 YAML，但没有源码/统计脚本条目。`grace_pretrain_manifest.csv` 对 Cora/CiteSeer/CoraFull 提供 checkpoint/H/hash/config，而另外两主数据集只由归档 H 支持 downstream replay。
- Rejection impact: 方法论文若没有可执行实现与配置闭环，读者无法验证递归分裂顺序、随机对照、LOO、teacher/student checkpoint selection 等关键实现，结果复现会退化为“相信已导出的 H 和 CSV”。
- Minimal fix: 明确区分 (a) exact downstream replay、(b) encoder-to-result end-to-end reproduction；归档可执行源码、依赖版本/ROCm 与 PyTorch/PyG 版本、数据预处理版本、所有数据集 YAML、运行命令、seed 派生规则、配置 ID 映射和统计脚本。缺 checkpoint 的数据集应明确标为“不能 exact end-to-end reconstruct”，不要使用“完整复现”。
- New experiment required: NO
- Claim decision: WEAKEN

### MET-05
- Severity: P1
- Manuscript location: §4.1 L359；§4.4 L393–395；附录索引 L729
- Problem: “Cora/CiteSeer 开发，固定 \(\beta\) 与全部构造参数后迁移”的冻结边界没有操作化。稿件没有在正文或当前 manifest 中给出 \(\beta\) 候选集、选择指标、选择层级、最终选择发生在何时，以及 Cora/CiteSeer 两个开发集如何合并决策；同时 GRACE 维数和预训练配置随数据集变化，因此“全部参数冻结”容易被误解为 encoder 也冻结。
- Evidence: L359 只给出口头开发/冻结顺序；L395 明确 Cora 的 \(d_h=128\)，其余完整噪声谱数据集为 256。`grace_pretrain_manifest.csv` 还显示数据集使用不同 learning rate、augmentation、temperature、epoch，CoraFull 的 `config_source_dataset=DBLP`。附录索引声称存在 \(\beta\) validation selection audit，但 material manifest 未给出该资产入口。
- Rejection impact: 缺少开发—冻结账本时，跨数据集结果可能被怀疑含有隐性后验调参；这会直接削弱“参数冻结后三数据集 14/15”的证据等级。
- Minimal fix: 发布冻结表：参数类别、候选网格、开发数据、选择指标（clean validation）、选择规则、冻结时间/commit、哪些参数允许 dataset-specific；单独说明 encoder 配置与 downstream \((B_{\max},n_{\min},\delta,s_{\min},\beta)\) 的冻结边界。把已声称存在的 \(\beta\) selection ledger 加入 manifest。
- New experiment required: NO
- Claim decision: TEST

### MET-06
- Severity: P1
- Manuscript location: §3.5 L268–329；§6 L543–561，Table 5；C-007/C-019
- Problem: 基础版与蒸馏版不是只改变“覆盖全部节点”。基础版使用 \(\beta=0.5\) 的半硬目标直接训练 train-node X-only MLP；蒸馏版改为 `target_beta=0` 的分布目标，训练 GCN[X‖H] teacher，再以全节点 teacher distributions 单独训练 student。教师输入、图传播、目标形状、监督覆盖和学生目标同时改变。Table 5 可以说明 H 输入和局部目标与结果“相关”，但不能唯一识别 all-node coverage 或粒球机制的因果贡献。
- Evidence: §3.4 式 (9) 固定基础 \(\beta=0.5\)；§3.5 式 (13) 明确教师 \(\beta_T=0\)。Cora YAML 为 `target_beta=0.0, teacher_use_xh=true, teacher_target=soft_target, distill_temperature=1.0`；蒸馏 raw 进一步显示 ball teacher 与 CE teacher 有不同 `teacher_target`。C-007 状态为 PARTIAL，并禁止声称 H 或粒球共识单独导致全部提升。Table 5 没有 \(\beta_T=0.5\) teacher，也没有“保持其他项不变、仅改变 train/all-node coverage”的臂。
- Rejection impact: 当前“扩大监督覆盖”“H 提供主要增量”“高噪声下局部共识进一步改善”容易被读成因果分解，而现有 factorial controls 不闭合。该问题不否定性能数字，但会使论文的机制贡献被判定为过度解释。
- Minimal fix: 把所有机制句改为 arm-level association，明确列出基础到蒸馏同时发生的五项变化；只保留 Table 5 可直接比较的差值。若不新增实验，不要声称已分离 all-node coverage 的独立贡献，也不要把 \(\beta_T=0\) 结果外推到基础 \(\beta=0.5\)。
- New experiment required: NO
- Claim decision: WEAKEN

### MET-07
- Severity: P1
- Manuscript location: §5.3.1 L462–475；C-009
- Problem: LOO 证据支持“非自票、非随机的局部标签信号”，但不支持“其他节点提供独立监督信息”这一统计独立性或唯一因果机制表述。LOO 相对默认方法的六个单数据集精度差均未通过 Holm 校正；Brier/NCR/CDR 是利用 clean labels 计算的 post-hoc diagnostics，不是训练期可观察置信度。
- Evidence: `mlp36_paired_deltas.csv` 中 `loo_vs_current` 的六个 `holm_p_within_family` 均不小于 0.1465；Cora LOO 的 Brier 0.4876→0.4503，同时 CDR 0.2151→0.3020。LOO 对随机分组和显式平滑的差值显著，但这同时包含真实几何、组内标签相关性和目标形状差异。C-009 明确禁止把 Brier/NCR/CDR 的相关变化当作唯一机制。
- Rejection impact: “independent supervision”是比现有对照更强的因果/统计结论，会被审稿人要求额外识别；clean-label diagnostics 若不标注为 post-hoc，也会混淆训练资源边界。
- Minimal fix: 改为“移除自票后，真实分组仍含有非随机的其他成员标签信号；其相对默认目标的精度优势未达到校正后显著性”。在表头和正文注明 Brier/NCR/CDR 仅用 clean labels 做事后诊断，不参与目标、选择或部署。
- New experiment required: NO
- Claim decision: WEAKEN

### MET-08
- Severity: P1
- Manuscript location: §5.3.2 L477–481；§7 L588；C-010/C-024
- Problem: “kmeans 与粒球相近”没有可追溯 raw；“默认球平均约 10 个训练节点”和“有效票数比具体分组形式更关键”也被写得过于普遍。默认配置的 mean train-per-ball 在 Cora/CiteSeer 约为 10，但在 Coauthor-CS、PubMed 和 Amazon-Photo 约为 57.3、61.6 和 23.9。大球配置同时改变 `max_balls`、`min_ball_size` 和 `split_improvement`，不能把性能变化唯一归因于票数。
- Evidence: C-024 状态为 PARTIAL，明确写明“当前未定位 kmeans raw”且禁止把票数写成唯一因果因素。`mlp13_stageB_summary.csv` 显示 Cora 从 default 到 large-ball 时球数 52.4→22、票数 10.33→24.61、noisy purity 0.505→0.477、accuracy 55.59→56.56，但该配置从 `mb64_min32_si0.05` 同时变为 `mb128_min64_si0.01`。同表的默认票数具有明显数据集差异。
- Rejection impact: 该段是方法机制与“粒球而非 kNN/kmeans”的核心解释；不可追溯对照和单因果叙事会削弱方法动机，而不是普通措辞问题。
- Minimal fix: 删除正文和结论中的 kmeans 实证 claim，直到提供 raw locator；把“约 10 票”限定为 Cora/CiteSeer；把机制改为“受限观测与有效票数假设相容”，并逐项列明 large-ball 配置同时变化的参数。不要把跨两个汇总表算出的 2.58→0.77 写成单次直接测量。
- New experiment required: NO
- Claim decision: DELETE

### MET-09
- Severity: P1
- Manuscript location: §5.4 L514–518；C-017/C-025
- Problem: §5.4 把多个证据等级不同的诊断合并成普遍机制结论。当前可定位证据只支持 Cora seed 0 上 GRACE 为 52 球、CCA-SSG/GraphMAE 为 1 球，以及对应 probe 差异；不支持 noisy-CE GRACE 52→1、purity 0.79→0.30、下游 −12.6 pp，也不支持 trained-MLP hidden 单球。CiteSeer 的 GraphMAE 实际为 64 球，进一步否定跨数据集泛化写法。
- Evidence: `ssl_ball_diagnostics_seed0.csv` 中 Cora 为 GRACE `52/0.7873`、CCA-SSG `1/0.3021`、GraphMAE `1/0.3021`；同一文件中 CiteSeer GraphMAE 为 64 球。`probe_aggregate.csv` 支持 Cora 的 clean/noisy probes，但 C-025 明确指出 noisy-CE GRACE 和 MLP-hidden 数值缺 raw/aggregate locator。C-017 的 3 H-seed 结果只说明已测 GRACE seeds 的下游均值较稳定，不能补足这些对象。
- Rejection impact: 这是当前最直接的 evidence–claim 不一致。保留无 raw 的精确数值会损害整篇稿件的可信度，并使“GRACE 经验选择”看似经过比实际更广泛的验证。
- Minimal fix: 删除 noisy-CE GRACE 和 MLP-hidden 的精确塌缩/降幅陈述；把 CCA-SSG/GraphMAE 限定为“Cora seed-0、当前构球参数下的几何诊断”。同时将 L518 的解释改为“低 ARI 与相近下游均值并存”，不要推断未直接测量的“跨分区稳定共识统计”。
- New experiment required: NO
- Claim decision: DELETE

### MET-10
- Severity: P2
- Manuscript location: §3.1 符号表 L94–107；§3.5 L268–329；算法 1 L234–265
- Problem: 基础版符号大体闭合，但复现实质相关的符号和选择算子未统一：验证/测试集合未记号化，\(\beta_T,\tau,\varepsilon,n_{\min},B_{\max},\delta,F\) 不在符号表；教师与学生的“最佳 checkpoint”由什么指标、在何种 tie-breaking 下选择没有在方法/伪代码中出现；LOO 也没有写成节点特异的 \(p_{j,-v},m_{j,-v},r_{j,-v},\alpha_{j,-v}\)。
- Evidence: §3.1 只列出 \(H,\mathcal B,T_j,p_j,m_j,r_j,q_j,\alpha_j,t_v,\beta,s_{\min}\)；§3.5 L301 仅写“选定教师检查点”，raw 中存在 `teacher_best_epoch` 和 `distill_student_best_epoch`，但稿件未给选择规则。LOO raw 有 `leave_one_out=1`，正文只用一句话描述删除自票。
- Rejection impact: 不会改变现有结果，但会导致独立实现对 LOO、teacher/student early selection 和平票行为作出不同选择，降低算法 1 的规范性。
- Minimal fix: 扩充符号表和算法 1/算法 2；为 LOO 给出节点级公式；明确 teacher/student 以 clean validation accuracy 选 checkpoint、并列时的 epoch 规则、是否每 seed 独立选择。建议单列蒸馏伪代码，避免基础版 `return` 后只靠散文描述扩展。
- New experiment required: NO
- Claim decision: KEEP

### MET-11
- Severity: P2
- Manuscript location: §3.6 L331–351；§5.5 L521–539；C-018
- Problem: 式 (18) 不是可用于规模分析的闭合复杂度，因为 \(T_{\mathrm{2means}}\) 把所有未知成本整体吸收，且未显式包含 `n_init=10`、每次 k-means 的迭代上限、每轮最多尝试多少候选球及候选排序成本。“近线性构造成本”只在 \(B_{\max}\)、`n_init` 和迭代上限视为常数时成立。效率表又只覆盖 3 个数据集、seed 0；raw 中没有硬件/软件版本列值，FP32 MiB 与 throughput 还是派生量。
- Evidence: C-018 的允许边界仅是指定环境中的观测成本，并明确“固定粒球预算不等于无条件严格线性”。`latency_memory_seed0.csv` 支持下游时间、构球时间、参数量、峰值显存和 full-graph latency，但 `hardware/software` 信息为空；`latency_memory_report.md` 只写 “one ROCm GPU”。material manifest 也注明 FP32 MiB 和 throughput 尚未物化为单一最终表资产。
- Rejection impact: 当前复杂度不会使方法错误，但会被认为回避了最昂贵的二均值项；效率结论也不足以外推到大图或跨平台部署。
- Minimal fix: 给出包含候选尝试次数、`n_init`、最大迭代数的上界，并明确“在固定 \(B_{\max}\) 与 k-means 预算下对 \(N\) 近线性”；将 seed-0 表称为 profile 而非稳定 benchmark；物化最终资源表，记录 GPU、ROCm、PyTorch/PyG、dtype、warm-up、同步方式和重复次数。
- New experiment required: NO
- Claim decision: WEAKEN

## Final methodological decision

基础 GB-LCSS 的**问题设置、软目标主公式和 X-only 部署定义可以保留**；transductive 与 inductive 的边界也无需新增实验。当前最需要改变的不是再扩大数据集，而是先从现有资产量化平票规则的实际暴露率，再据此决定只修正规则/术语还是条件性补充 tie-safe 对照；同时让 pairing、配置冻结和复现资产形成机器可审计闭环。

机制结论必须降到现有设计真正能支持的层级：

- 已支持：真实几何分组优于 size-matched shuffled control；LOO 保留非自票局部信号；指定 GRACE 空间优于所测 adjacency-propagation control；蒸馏学生保持 X-only 部署。
- 仅相关性支持：H 输入与主要增益相关；高噪声下局部共识教师在部分 settings 增益更高；低 ARI 与较稳定下游均值并存。
- 尚未识别：有效票数是唯一机制、all-node coverage 的独立因果贡献、任意 kmeans 与粒球等效、GRACE 在所有 SSL 表征中普遍最适合构球。
- 当前应删除：无 raw locator 的 noisy-CE GRACE/MLP-hidden 精确塌缩数值，以及未定位 raw 的 kmeans 实证结论。
