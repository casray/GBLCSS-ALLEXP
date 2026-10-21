# Information Sciences 叙事与创新性独立审查

## Overall recommendation

**MAJOR**

稿件与 Information Sciences 的主题适配度良好：它把粒球计算、鲁棒学习、图表示与受限推理资源连接为一个清楚的智能信息处理问题。当前阻碍不是实验规模不足，而是新颖性边界和主叙事尚未完全服从已有证据。最严重的问题是 C-022：摘要和引言仍保留无系统检索支持的排他性研究空白。其次，稿件将“粒球”置于标题和方法身份中心，但自身消融显示调优 kNN 往往更强、有效票数比具体分组形式更关键；蒸馏扩展的主要增益又来自 H 输入和全节点覆盖，而非粒球共识本身。若不主动收窄这些主张，EIC 很可能将论文判断为“一个有效的局部标签聚合与蒸馏组合，但粒球特异创新不足”。

## Confidence

**HIGH（稿件内部叙事、claim–evidence 一致性和 Information Sciences 适配判断）；MEDIUM（全球文献新颖性）**。

本审查完整读取 v29、claim ledger、material manifest 和本地文献图谱，并抽查 C-007、C-010、C-024、C-025 直接指向的小型汇总证据。依任务约束未联网、未做系统文献检索，也未逐篇核对原论文，因此所有关于“全球是否已有同类工作”以及稿件对相关文献的具体归纳均标记为 **UNVERIFIED**。

## Strengths

1. 问题组合明确且具有 Information Sciences 相关性：训练期图可用、训练标签含噪、推理期仅属性的资源约束，把鲁棒监督构造和可部署推理连接起来。
2. 训练与推理资源边界定义清楚。稿件明确区分 X-only 学生、H-access 分类器、图教师和图依赖参照，避免把“推理无图”偷换成“训练全程无图”。
3. 方法角色总体克制：粒球成员关系不读取标签，同球关系没有被硬化为真实类别；球内共识以置信度软混合进入目标，符合软结构先验的合理定位。
4. 负结果披露提高可信度。PubMed 和六个扩展数据集的混合表现、clean-label 有限收益、特征噪声退化、transductive 边界、GRACE 依赖和 kNN 竞争结果均被公开报告。
5. 证据审计基础强。主结果区分开发数据集与参数冻结数据集，报告配对 seed、均值/标准差、Holm 校正，并在表格中标注正式方法、移植实现、机制代理和跨资源参照。
6. 相关工作已覆盖四条必要脉络：通用 noisy-label learning、noisy graph learning、graph-free/GNN-to-MLP、granular-ball learning；结构基础已经具备，问题在于交叉处的最近邻定位还不够可审计。

## Top rejection blockers

1. **C-022 排他性新颖性主张无检索审计支持。** “尚缺乏系统研究”“尚未得到系统研究”必须删除或改为非排他性的研究定位。
2. **粒球特异创新与自身证据冲突。** kNN 在所测四数据集上高于默认粒球，稿件结论又承认票数比分组形式更关键；现有证据支持“局部共识框架的粒球实例”，不支持“粒球本身带来独特优势”。
3. **蒸馏扩展的增益归因过强。** H 输入与全节点覆盖是主要增益来源，球式共识只在部分高噪声设置带来附加收益，不能由 Highlights 到结论统一归功于“共识教师”。
4. **主故事混合了三种证据层级。** 基础 X-only 方法、资源更强的图教师扩展、以及 official port/proxy/cross-resource 参照在正文中轮番成为主角，削弱了核心贡献和公平比较边界。

## Issues

### NAR-01
- Severity: P0
- Manuscript location: 摘要 L19；引言 L29；相关工作 §2.3 L67
- Problem: 摘要称该问题“尚缺乏系统研究”，引言称“尚未得到系统研究”。这是排他性文献主张，但本地材料没有数据库、检索式、检索日期、纳入/排除标准和逐篇定位记录；GSDN 已被稿件自己识别为含非对称标签噪声的直接相邻工作。
- Evidence: C-022 状态为 UNVERIFIED，明确禁止“尚未得到系统研究”“首次”“尚无工作”等措辞；material manifest L99–100 说明 literature map 只用于定位、正式引用需另行核对；01_literature_and_research_map 不是 noisy graph learning × graph-free inference 的系统检索。相关文献的具体覆盖范围在本审查中均为 UNVERIFIED。
- Rejection impact: 这是可被编辑或审稿人直接识别的 novelty overclaim。即使方法有效，一个不可审计的“尚无系统研究”也足以破坏整篇论文的新颖性可信度。
- Minimal fix: 删除摘要和引言中的两处排他性表述。摘要改为“现有 noisy graph learning 与 graph-free inference 通常分别处理鲁棒监督和推理资源约束，本文直接评估二者交叉的设置”；引言改为“本文关注这一相对较少被直接评估的资源约束组合”，并在同段保留 GSDN 的明确邻近位置。无系统检索前不得用“首次”“尚无”“尚未系统研究”替代。
- New experiment required: NO
- Claim decision: DELETE

### NAR-02
- Severity: P1
- Manuscript location: 标题 L1–3；摘要 L19；贡献 2 L42；§5.3.2 L477–481；结论 L588
- Problem: 标题和贡献列表把“粒球”塑造成核心特异创新，但稿件自己的证据表明，默认粒球不是最强局部共识实现：k=20 的 kNN 在四个被测数据集上均高于默认粒球，结论还承认有效训练票数比具体分组形式更关键。现有结果更支持“局部共识软监督框架，以粒球作固定预算实例”，而不是“粒球机制本身产生独特性能优势”。
- Evidence: C-010：四数据集 uniform 0.4 下 k=20 kNN 均值均高于默认粒球；`mlp11_knn_curve.csv` 中 Cora/CiteSeer/PubMed/Coauthor-CS 的 k=20 均值分别为 57.20/56.41/72.48/90.91%，而正文默认粒球对应约 55.59/53.83/71.90/90.62%。C-024 仅支持“票数不足”相容性解释，禁止把票数写成唯一因果因素；kmeans claim 缺 raw。
- Rejection impact: Information Sciences 审稿人会追问“为什么一定是粒球”。若最接近的 kNN 实例更强，而标题仍暗示粒球特异优势，方法容易被判断为对局部标签平均的增量包装。
- Minimal fix: 将题目、摘要、贡献 2 和结论统一改为“几何局部共识软监督，本文采用固定预算粒球划分实现”。贡献 2 必须区分继承的递归粒球划分与本文新增的置信度软目标/资源流；明确写出“不主张粒球优于 kNN”。删除未定位 raw 的 kmeans 比较。若保留粒球在标题中心，可采用“Local-Consensus Soft Supervision …: A Granular-Ball Instantiation”式从属定位。
- New experiment required: NO
- Claim decision: WEAKEN

### NAR-03
- Severity: P1
- Manuscript location: Highlight 5 L13；摘要 L19；引言 L35；§6 L543–580；结论 L588
- Problem: 稿件多处把 X-only 学生达到“具有竞争力的 GCN 水平”归因于“共识教师蒸馏”，但现有消融只支持更窄的结论：图教师使用 X||H 并覆盖全部节点是主要提升来源，粒球共识在部分高噪声设置提供附加收益，在 CoraFull 上反而低于 GCN[X||H]+CE 和 kNN 教师。
- Evidence: C-007 状态为 PARTIAL，禁止把全部提升唯一归因于 H 或粒球；`table4_recomputed.csv` 显示 CoraFull u0.4/u0.6 的球式学生为 58.22/51.82%，低于 GCN[X||H]+CE 的 59.45/52.75%，也低于 kNN 教师的 58.95/52.73%。正文 L559–561 自己承认 H 输入提供主要增量，kNN 与球式教师六个差值均小于 1 pp。
- Rejection impact: 这是核心结果的 credit-assignment 问题。Highlights、摘要和结论若继续把资源增益写成粒球共识增益，审稿人会认为机制叙事选择性引用消融。
- Minimal fix: 将蒸馏明确降为“资源更强的二级扩展”。统一改写为：“X||H 图教师与全节点蒸馏显著缩小 X-only 学生和图模型的差距；局部共识在部分高噪声设置进一步改善教师监督。”Highlight 5 不再使用无条件的“共识教师将学生提升至 GCN 水平”，结论同时保留 CoraFull 反例和 kNN 近似结果。
- New experiment required: NO
- Claim decision: WEAKEN

### NAR-04
- Severity: P1
- Manuscript location: Highlights L9–13；摘要 L19；引言 L35–43；结论 L586–590
- Problem: 前端叙事没有完整携带证据适用域。摘要的“绝大多数噪声设置”和 Highlights 的“多数噪声设置”未说明这是五个主数据集、25 个合成 train-only noise settings、七个同骨干基线内的排名；读者也要到方法和结论才知道协议是同图 transductive，而六个额外数据集上的相对最强基线表现明显混合。
- Evidence: C-003 只支持五主数据集内 24/25、参数冻结三数据集 14/15；C-005 显示六个扩展数据集中的 PubMed、Amazon-Computers、Attributed-Wiki、CiteSeerFull 未超过最强正式基线，Cora-ML 和 Coauthor-Physics 仅数值接近；C-020 限定为同图 transductive；C-015、C-016 分别否定普遍 clean-label 增益和 feature-noise robustness。
- Rejection impact: 标题—摘要—结论的范围不一致会把一篇证据克制的论文读成选择五个有利数据集做强 headline，削弱 11 数据集审计本来带来的可信度。
- Minimal fix: 用精确计数替换模糊数量词：摘要写明“在五个主数据集的 25 个合成训练标签噪声设置中，相对七个同骨干基线取得 24/25 个最高均值；六个额外数据集显示收益具有数据依赖性”。摘要或首个 Highlight 同时加入“同图 transductive、训练期可用图、测试期 X-only”。结论保留 14/15 与 24/25 的开发/冻结区分，不将其外推为 11 数据集普遍领先。
- New experiment required: NO
- Claim decision: WEAKEN

### NAR-05
- Severity: P1
- Manuscript location: 摘要首句 L19；引言 L27；§5.5 L521–539；结论 L590
- Problem: 问题重要性的表述把 graph-free inference 与“隐私约束”和“低时延需求”直接绑定，但本文训练期仍访问完整图，不提供隐私机制；低时延也不是跨数据集成立的经验结论，CoraFull 上稠密高维 X-only MLP 的 full-graph 前向反而慢于 GCN。真正被证据支持的动机是“推理端邻接不可用/不允许访问”和“避免在线消息传递依赖”。
- Evidence: C-018 只允许报告指定硬件上的观测成本，禁止跨平台普遍加速；Table 4 中 CoraFull 的 GB-LCSS 与 GCN 推理分别为 14.987 ms 和 10.996 ms。C-019、C-020 说明训练期使用图与 H，且协议是 transductive。稿件没有隐私定义、威胁模型或隐私评测；“隐私约束”动机在本审查中为 UNVERIFIED。
- Rejection impact: 过度部署叙事会引发不必要的隐私和效率质疑，并掩盖论文真正清楚、可防御的资源边界。
- Minimal fix: 从摘要删除未经支撑的“隐私约束”和普遍“低时延”暗示，改为“推理端关系不可用、访问受限或不希望执行在线消息传递”。引言把延迟写成潜在成本而非必然优势，并紧邻 Table 4 结论明确“X-only 消除邻接依赖，但不保证在高维稠密属性上更快”。若坚持隐私动机，必须另有经核验的引用并明确本文不提供隐私保证。
- New experiment required: NO
- Claim decision: WEAKEN

### NAR-06
- Severity: P1
- Manuscript location: §2.2–2.4 L53–75；引言 L29–33；贡献 L39–43
- Problem: 相关工作分别综述 noisy graph learning、graph-free/GNN-to-MLP 和 granular-ball learning，却没有在交叉处建立可审计的“最近工作差异表”。当前差异散落在段尾，读者难以判断新增的是任务组合、软目标、标签无关分组、X-only 资源流，还是粒球算法。由于原论文未在本审查中逐篇核验，这些文献描述和缺口判断均为 UNVERIFIED。
- Evidence: C-022 指出没有跨 noisy graph learning、GNN-to-MLP 和 graph-free inference 的系统检索记录。manifest L99 将 literature map 定位为线索而非正式证据。稿件现有最近邻包括 GSDN、NRGNN/UnionNET/ProCon、GLNN/Graph-MLP/FF-G2M、MgBIF/BNO-GB/GEAF、GBGC/监督粒球粗化，但四组差异没有统一比较轴。
- Rejection impact: 删除“首次/尚未系统研究”只能消除 P0，不能自动建立可防御的新颖性。Information Sciences 的新颖性判断尤其需要说明相对于既有粒球噪声学习，本文的新接口究竟是什么。
- Minimal fix: 在相关工作末尾加入一个紧凑矩阵，逐项列出最近工作是否使用 noisy labels、训练期 graph、推理期 graph/H、标签无关分组、组级 soft target、全节点蒸馏，以及粒球的角色（分类器/过滤器/粗化单元/软监督实例）。正文只保留经原论文核对的事实；未核对项标为 UNVERIFIED 并在投稿前验证。该矩阵结论应收束为“本文的可防御差异是资源流与软监督接口的组合”，而非全球首创断言。
- New experiment required: NO
- Claim decision: TEST

### NAR-07
- Severity: P1
- Manuscript location: §5.1–5.5 L399–539，尤其 Table 3；§6 L543–580，Table 5–6
- Problem: 表格叙述混合三类证据：同骨干正式比较、统一协议移植/机制代理、跨部署资源参照。Table 3 同时出现 protocol port、common-backbone reimplementation、mechanism adaptation 和 inspired proxy；Table 6 又把 X-only 学生与推理需图方法并列。尽管脚注披露了差异，正文在这些表之间切换“最高均值”“竞争力”容易让主线被排名叙述淹没。
- Evidence: C-006 只允许在正式 graph-free 组内逐数据集陈述排名，并禁止把 KRD/VQGraph 代理当官方结果；C-008 明确 Table 6 不是同资源公平排名。正文 L440–458、L563–580 仍需要读者自行重建哪些行属于正式方法、代理或跨资源上界。
- Rejection impact: 比较身份不清会让审稿人把本来谨慎的资源审计解读为“通过混合实现争取排名”，同时削弱基础 GB-LCSS 的独立贡献。
- Minimal fix: 正文结果按三个问题重排并显式命名：(1) 同骨干鲁棒监督是否有效（Table 1–2）；(2) 局部共识的必要条件是什么（random/LOO/kNN）；(3) 增加图教师和全节点监督后能缩小多少资源差距（Table 5–6）。Table 3 将正式 fidelity ports 与 inspired proxies 分块，代理行移至补充材料；Table 6 保留“距离”而不做胜负排名。每张表后的首句只陈述该表允许回答的问题。
- New experiment required: NO
- Claim decision: KEEP

### NAR-08
- Severity: P2
- Manuscript location: 引言贡献列表 L39–43；摘要 L19；结论 L586–590
- Problem: 三条贡献不正交：贡献 1 主要复述任务设定，贡献 2 把继承的粒球划分与新增软目标合写成“设计”，贡献 3 同时塞入 11 数据集验证、随机分组分析和一个资源更强的蒸馏扩展。结果是摘要和结论无法稳定回答“论文到底贡献一个方法还是两个方法”。
- Evidence: C-001 支持的核心是局部关系/置信度软目标；C-019 将蒸馏定义为使用图教师和 H 的独立扩展；C-007 又限制其机制归因。正文方法结构本身已经把基础 GB-LCSS（§3.2–3.4）和蒸馏扩展（§3.5、§6）分开。
- Rejection impact: 贡献边界模糊不会单独导致拒稿，但会放大 NAR-02 和 NAR-03 的创新性质疑，使编辑难以用一句话概括论文。
- Minimal fix: 将贡献改为三条互斥陈述：(1) 提出面向 noisy-label transductive X-only inference 的局部共识软监督接口；(2) 以标签无关粒球分区和置信度混合实例化该接口，并明确哪些构造继承自粒球文献；(3) 通过同骨干主评测、随机/LOO/kNN 机制审计界定有效域。蒸馏只作为“secondary extension”另列一句，不与基础方法共同充当核心创新。
- New experiment required: NO
- Claim decision: WEAKEN

### NAR-09
- Severity: P2
- Manuscript location: §5.3.2 L479–481；§5.4 L514–518
- Problem: 两处机制叙事超过 ledger 支持范围。其一，稿件称“kNN 和 kmeans 与粒球相近”，但 kmeans raw 未定位；并把票数写成“主要”解释，现有观测只支持相容性。其二，§5.4 将 noisy-CE GRACE、MLP hidden、CCA-SSG 和 GraphMAE 的退化串成一般性结论，但多项数值缺 raw，CCA-SSG/GraphMAE 单球只由 Cora seed 0 支持，CiteSeer GraphMAE 实为 64 球。
- Evidence: C-024 状态 PARTIAL，明确要求删除无 raw 的 kmeans claim并禁止唯一因果归因；C-025 状态 PARTIAL，禁止保留未定位的 noisy-CE GRACE 52→1、−12.6 pp 和 MLP-hidden claim。`ssl_ball_diagnostics_seed0.csv` 显示 Cora 上 CCA-SSG/GraphMAE 各 1 球，但 CiteSeer GraphMAE 为 64 球。
- Rejection impact: 这些不是核心性能结论，但会让审稿人质疑作者是否只在主表上遵守证据门槛、在机制解释中放宽标准。
- Minimal fix: §5.3.2 删除 kmeans，改“主要与票数有关”为“与票数不足的解释相容，尚不能识别唯一因果因素”。§5.4 仅保留 Cora seed 0、相同构球参数下经支持的 CCA-SSG/GraphMAE 诊断，并明确 CiteSeer 不复制同一模式；删除 noisy-CE GRACE、MLP hidden 和 −12.6 pp，直至 raw locator 补齐。
- New experiment required: NO
- Claim decision: WEAKEN

### NAR-10
- Severity: P2
- Manuscript location: §8 L594–596；附录索引 L722–734
- Problem: 资产与附录部分仍含内部制作状态和修订流水账，例如“链接将在投稿排版阶段补入”“当前 CoraFull 已补齐”“暂不进入正文表”。这些句子不回答科学问题，也让稿件看起来尚未冻结。
- Evidence: v29 L596、L734；material manifest 已承担资产状态追踪功能，正文无需复制内部状态。Elsevier 稿件叙事应保留最终可复现范围，而不是记录材料何时补齐。
- Rejection impact: 主要影响投稿完成度与编辑第一印象；在已有复杂证据体系下，这类内部措辞会进一步强化“实验日志而非定稿论文”的感觉。
- Minimal fix: §8 只保留投稿时实际可访问的数据、代码、配置和 checkpoint 范围；没有匿名链接时删除占位句。附录索引改成最终内容目录，删除“当前”“已补齐”“暂不进入”等过程状态，未纳入正文的方法只在 supplement 的 protocol audit 中客观列出。
- New experiment required: NO
- Claim decision: DELETE

## Severity count

- P0: 1
- P1: 6
- P2: 3
- Total: 10

