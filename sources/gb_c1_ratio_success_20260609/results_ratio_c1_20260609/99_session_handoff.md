# 最新执行：2026-10-21 边界诊断与粗化复盘已完成

入口`experiments/r2_boundary_diagnostic_20261021/综合结论与接续.md`及目录交接。用户批准先执行非训练前两项，实际新模型0，旧代码/模型/稿件未改，无待续训练。20656训练节点—条件行、12协议、99相关/394分箱，seed0事后诊断；237源哈希与11测试通过。

球心相对距离与GB−kNN目标Brier劣势同时正rho/端组差：clean2/3、带噪3/9，单调1/12，普遍边界径向解释未支持。跨球近邻比例clean3/3、带噪7/9，单调4/12，得到有限描述性支持，但相关弱、部分球内调整反向。最高跨球组clean严格错误目标Cora GB34/137 vs kNN11/137；这是目标不是测试预测。不得声称已证明固定分区必输kNN或使用半径门控就能修复。原GB-on-R2推进门仍false。

复算预算粗化1080旧JSON、232配对行、264可用主表臂与24明确未运行空白。GCN C−B+1.148pp、C−LCSS+.565(6/15正)，缓存MLP+.346/−1.218(4/15)。保留局部高噪声球级GCN训练组织线索，但R3−R0−.033、O2−R0+.281，16clean效率联合门0通过，重建冷C/A1.22–2.17。架构/部署不同，不与R2绝对精度混排。下一轮若授权，先有限强聚类粗化/抗噪GCN对照，不救R3、不自动GCE组合/c组/多seed。

初次统计脚本因Coauthor-CS旧表B等臂为空停止，边界输出已完成并保留；恢复finish_review.py明确optional臂/五PRIMARY/共享A预算，audit最终passed，失败过程ENGINEERING.md可追溯。旧源和原冻结diagnose.py未改，不重启该入口。

# 前次执行：2026-10-21 R2机制对照已完成

入口`experiments/r2_mechanism_controls_20261021/综合结论与接续.md`及目录交接、PLAN/results/decision/audit。无待续训练，投稿未改。新156分类拟合、三份H行打乱预训练、三随机编码器、九个短冒烟全部完成；不是前一轮192矩阵重启。

同三图四条件协议seed0，固定旧R2、原split/noise/H/GB；REAL九监督臂108格，随机冻结/打乱H各CE/GB共48格。带噪均值CE56.602、GB57.921、LS56.444、GCE59.710、BOOT56.466、MASS58.049、KNN59.134、KM59.277、随机组55.720%。GB−GCE−1.789pp(1/9正)、−MASS−.128(4/9)、−KNN−1.213(0/9)、−KM−1.355(1/9)、−随机组+2.202(8/9)。GCE仅按验证全局选择；MASS逐节点保持GB对观测类概率，只移除替代类方向。

真实R2 CE比随机21.660/打乱17.304分别+34.942/+39.298pp，均9/9正，表示门通过；但不独立证明拓扑因果。GB clean−CE三图−1.478/−1.246/−2.360pp；局部方向/通用正则/KM/KNN/clean门失败。停止当前GB-on-R2默认推进，不扩seed、不自动解冻/加深或拼模块。GCE/KNN也有CoraFull clean损害，不当通用新方法。

10测试、172源哈希、48独立目标、156重载、24上一轮精确锚、独立156精度与144差值、96近邻查询核验通过。正式头网格56.474秒，复用旧GRACE/R2成本不计入，非完整效率结果；只一个seed，std=null。soft BOOT当前预测停止梯度等价0.8CE梯度，不是EMA纠错。两份审计均passed。

# 前次执行：2026-10-21 表示迁移与鲁棒监督筛查

本轮用户授权“设计并开始实验”已完成。入口`experiments/representation_supervision_20261021/综合结论与接续.md`，目录内交接、PLAN、results/paired_deltas/decision与两份audit可追溯。无待续训练，不自动扩种子或改投稿稿件。

Cora/CiteSeer/CoraFull，clean/U.4/U.6/Pair.4，完整协议seed0，冻结历史H及分区。R0无对齐200轮；R1联合对齐200轮；R2无标签对齐100轮+冻结编码器/BN分类100轮。四种监督CE/LOO-LCSS/small-loss/组合，144主格+24旧目标桥接+24H参考；另4验证调参和3共享预训练，网格511.259秒。

带噪九格宏平均CE/LCSS/small-loss/组合：R0 35.515/43.420/39.451/47.356；R1 38.022/47.699/44.375/51.321；R2 56.602/57.921/50.658/53.896。R2 LCSS相对原LCSS+14.501pp，但相对同R2 CE仅+1.320pp（8/9正）；R2组合对最好单模块-4.062pp（3/9正）。clean下R2 LCSS对CE三图-1.478/-1.246/-2.360pp，完整LCSS推进门失败，不能事后放宽。

验证选择的R2/LCSS U.4=75.673/71.005/47.333，配对SimMLP seed0=75.620/67.139/40.166；Cora只差一个节点，非多种子SOTA。14测试、12协议、52冻结哈希、192检查点重载与独立指标重算通过。建议先验证随机冻结/H打乱控制再确认多seed；本轮未验证粒球相对KM/kNN的独立增量。原准备格式错误保留，正式训练无失败。

# 历史执行：2026-10-15 属性与结构互补实验

用户批准的本轮已完成，无待续队列。入口experiments/attribute_structure_20261015/综合结论与接续.md、实验结论.md、final_validation.json；新Amazon-ratings/Tolokers+旧Roman/CS，四图五协议seed×三条件×七臂420目标，基础五臂300新学生全部完成。38项单元测试通过，420目标/300预测/420配对比较/953文件哈希独立审计通过，最终只读复核通过。Tolokers用sym.2/.4及val ROC-AUC，其余Accuracy。每图仅一SSL seed，Minesweeper未运行。

Amazon BFS−OLD64 clean/U/Pair分别+1.107/+1.019/+1.293pp，均5/5正；相对CE带噪+1.146/+.715pp，但clean−1.209pp。保留局部正信号，不宣布通用异配成功。Tolokers无稳定AUC增益；Roman X Pair−CE+1.182pp、−OLD64仅+.268pp且3/5正，clean−CE .844pp。CS Pair X/BFS−OLD64分别−2.442/−5.955pp。

四图固定等权残差R均未完整通过预设目标gate，R/SHUFFLE学生按计划未训练，其分类增益未知，不称分类失败。保留部分弱目标信号，不放宽门槛、不扫残差权重。事后发现原alpha可能把类别先验当局部可靠性；二分类全局反例已核，但不等于实际区域全部少数类改错或AUC受损。下一步可靠性方向尚未实现。源、数据、H、目标、logits均冻结，原默认入口未改。

prepare/workers/学生/有限队列均已完成，不重启。两新H各1000轮，Amazon1417.51秒/Tolokers693.55秒；CPU重调度仅独立seed并行，见EXECUTION.md。final_validation状态passed，完整audit/report SHA为02c8e44a13656f51345b12a6fc05a698d19c21ef6ff0cb5d589fffafa5388cfe。OLD64当前环境按原参数重构、学生全新训，不要求历史逐字节复现。其他BFS/GREET/数量/Context分支保留。

# 历史执行：2026-10-15 BFS拓扑粒球实验

用户在BFS附件审查后明确授权“可以，设计并开始实验”。独立目录experiments/topology_bfs_lcss_20261015，PLAN/config/source_manifest冻结；原算法/目标/学生未改。新增GB-LCSS_code/src/gblcss/granular_ball_topology.py与15测试（含63小图参考），原目标指标16回归测试通过。

目标阶段已完成315行：三图五协议seed×三条件×七臂，15protocol独立审计通过；check_gate.py核CSV/270配对差与原metrics.json一致。BFS球数CoraFull3517/Roman380/CS2200，单节点2256/72/1349，原图孤立点均0，存在残余碎片化。构球约.20/.20/.26秒。BFS无标签、容量floor(sqrtN)，最终数量不强行凑sqrt。

CoraFull/CS目标gate通过，两图七臂五seed三条件共210新学生已全部完成；315目标、210保存logits、3分区/15协议独立审计通过，12项历史baseline准确率全部一致。本分支无待续训练，不重启队列。完整报告experiments/topology_bfs_lcss_20261015/实验结论.md，decision.json及final_validation.json状态分别completed/passed。

BFS−旧GB64学生百分点（clean/Uniform/Pair）：CoraFull+1.374/−.947/−1.533，CS−1.197/−2.076/−6.030；BFS−CE带噪CoraFull+4.146/+4.743，CS+.140/+4.300，CS Uniform仅3/5正增量，不能称稳定提升。两图均未通过预设简化判据。Roman目标相对大小/训练支持匹配随机Brier无增量，按冻结门槛未训练学生；原X几何目标有强信号，但不是已验证分类提升。最终暂缓纯BFS替代GRACE作为异配主线，保留低成本诊断实现；碎片化/软化贡献仅是待验证解释。

当前本轮不实现GREET/HALF、context或BFS属性分裂，不与其他对话方向混比。下方是其他研究分支历史与待讨论方案，均保留。

## 并行独立方向：2026-10-15 GREET自适应分球复验已完成

### 本GREET对话最新任务：六图训练证据剪枝已完成

用户随后授权“设计并开始，可以更改算法，由你决定”并回复“开始”。新目录experiments/evidence_pruning_20261015，正式run session91756退出0；324新目标/学生记录，286新模型38别名，216旧锚另计，540预测审计通过（54协议/1944组件/1045冻结输入源），14测试通过，六组后处理配对图完成，无待续训练。详细结论见综合结论与接续.md；接续以该目录HANDOFF最前最终状态为准。

最终defer当前准则：主PRUNE六图目标/分类信号皆失败；相对CF_FINE的36带噪Brier配对0改善。对B64带噪准确率平均Cora−7.938pp、CS−2.166pp、Roman−.980pp；Squirrel+2.229pp但clean−.730pp且低于分折一球/不超原一球，不当作新语义收益。无惩罚与全票未救整体性能。Cora97.22%查询估计选根，同支持最大可行互斥区域均值45.89而PRUNE1.06；CF_FINE相对原全票细树带噪−2.804pp。不可把CF_FINE与互斥DP候选族差别忽略，也不能把票数损失单独当因果。暂不调delta/折数、不扩大seed、不改默认；其他属性残差/BFS/Context分支保留。

### 前一轮：六图数量诊断已完成

用户追加“增加几个异配数据集，暂不做knn/kmeans对比，诊断不同球数，再设计粒球算法”，并回复继续。目录experiments/ball_count_diagnostic_20261015已完成，无待续训练。Actor/Chameleon-filtered/Squirrel-filtered官方pin数据与六SSL完成；旧三图H复用。新单树固定minleaf16/support16、1/8/16/32/64/128/256+sqrt快照，单线程保证前缀一致；1026目标/学生记录、819独立模型、207别名，35测试通过。最终36分区288快照54标签协议1026目标/结果和81历史锚审计通过；18组图及详细结论见综合结论与接续.md。

原run.py在全部训练后因CSV数字字符串兼容问题退出1；finish_diagnostic.py保留初版失败记录，只转换数值标量后完整审计通过，绘图格式另追溯修复；未改模型/目标或冻结源。实际图已检查，status completed。Cora同GREET64→终点U40.10±.31→42.44±.38%，CS Pair86.45±.98→84.64±1.06%；Squirrel带噪偏粗，Roman目标改善未转化分类收益。约17%–36%的GREET正SSE分裂使目标Brier变差（步骤比例）。决策modify，不统一sqrt/更大预算；算法重设计建议.md提供无标签树＋独立训练证据剪枝，未实现验证；验证集选预算108单元只是事后普通调参诊断。不得重启训练、把测试峰值变默认参数或宣布异配普遍成功。其他对话BFS/Context不接管。

最终更新10:56：新目录greet_adaptive_ball_20261015的297目标/297学生全部完成，run.py退出0，18分区27标签协议297logits独立审计通过，30历史基线准确率全一致；无待续队列。入口综合结论与接续.md、实验结论.md、audit.json、decision.json。GREET旧1球已修复为64与151/141/136根号球，最小训练支持≥16。

主要结果：CoraFull新GREET−原GRACE clean/U/Pair +3.72/+4.51/+3.08pp；同规则新GREET−新GRACE +1.28/+1.72/+1.62pp，带噪6/6正，但比同空间kNN低1.41/1.83/1.91pp。Roman目标改善很大，同规则分类仅+.188/+.053pp且clean−.847pp；CS相对原GRACE Pair−1.016pp。决策modify，保留Cora局部信号，不宣布通用异配算法成功、不统一部署sqrt、不事后选球数。每空间单SSLseed及预算不同限制保留。BFS/Context其他分支未改。

以下为本轮启动时记录，已被上述完成状态覆盖：

GREET/HF首轮已于10:01完成：54目标协议324目标243辅助模型审计通过，A/B均未过门槛，0学生；原规则三图GREET均1球。旧目录greet_half_pilot_20261015无待续队列，不能按下面旧运行记录重启。

用户随后授权“更改粒球构造规则，生成合适数量后再实验”。新入口experiments/greet_adaptive_ball_20261015/HANDOFF.md；10:26启动run.py --device cuda，exec session53827，具体进度看status.json/raw.log。两空间共用已测试SSE优先/fallback/支持16；64与sqrt预算、KM/kNN匹配；三图三协议seed三条件，297目标和297个200轮学生固定全跑，无HF。5新增+11原生成器测试通过。勿改source_manifest内冻结源；完成后看audit.json/实验结论.md再下结论。共享BFS/Context分支保留。

以下两段为首轮进行时历史：

GREET/HF对话用户也已批准“设计并开始”。本方向入口experiments/greet_half_pilot_20261015/HANDOFF.md、PLAN.md/config.json。prepare_greet.py三图GPU队列当前Roman完成、CoraFull训练、CS待续；接续run_pipeline.py隐藏队列PID190412等待预训练，之后A/B诊断和条件性学生/融合。以其pipeline_status/greet_raw/queue日志为准，勿重复启动或接管另一方向。当前共享GPU的耗时不用于效率排名。

Roman A三协议seed已完成：原规则GREET1球、GRACE64；clean Brier差−.044457±.000875，Uniform+.056514±.001720，Pair+.043923±.000477。GREET首次分裂半径改善3.857%未过5%，提示构球规则兼容性问题，不是GREET整体失败结论。12测试与合成GPU/辅助小图检查通过，完整三图审计尚未完成。GREET结构编码MC近似、单SSLseed及HF-proxy非HALF复现均要保留。

# 历史交接：2026-10-14 GREET与HALF分别验证的方案审查

> 本聊天Context-GB执行分支新增记录（2026-10-14）：24独立SSL/324目标组/684分类已完成，完整seed0/1，严格审计通过。入口`experiments/context_gb_two_seed_20261014/综合结论与接续.md`及同目录方向历史。当前defer：B64带噪对原GRACE，MLP−6.332pp、GCN−1.888pp；GCN对PiNGDA+.516pp，对固定mean仅+.154pp。用户本轮排除独立KM/KNN。无本轮待续队列，不重跑已完成训练。下方GREET/HALF、异配强方法和生成器任务按各自研究分支及协议接续。

用户新增附件C:/Users/Admin/Downloads/新建文件 (6).txt，要求读两篇并结合论文/代码判断“先分别实验、再尝试融合”。本轮完成核读和具体设计，没有新训练。当前入口research/greet_half_review_20261014/00_建议与执行顺序.md；01_两篇文献与代码核查.md保存原文/代码差异，02_独立验证与融合实验设计.md保存独立矩阵与条件性融合。此前SOTA审查和实验保留。

- 推荐GREET完整表示替换先行，原分球/q/学生固定，匹配实际球数KM和唯一票数kNN。边引导BFS后置；边AUC只限制边引导，不足以否决完整双通道表示。
- GREET正式9页全文方法/实验及作者源码已归档；原model.eval仍Gumbel采样，需一次权重导出与H一致保存。作者rank loss/forward与正文式8/10存在差别，移植依据须固定。当前DGL缺失；Roman/Amazon/CoraFull不在作者loader，不直接运行main改split。无标签预训练不能用干净训练标签线性probe选checkpoint。
- HALF仍仅原文摘要/公开方法片段，出版社直连403，无完整公式/可确认官方实现。独立机制候选HF-proxy不称HALF复现。原GRACE球/q冻结，交叉拟合高低频辅助分支，以可疑分数控制alpha，必要普通属性小损失/单频/固定融合/均值缩放/打乱对照。rho未经校准不称错误概率；不复用ESS×JS，原Corollary独立性一般需重做。
- 初步计划三图CoraFull/Roman/CS、三噪声clean/U.4/Pair.4、3配对协议seed，确认再扩5seed与独立SSLseed。Roman沿用官方，旧图沿用旧split。原support16仅后续正交敏感性。最终固定两分支做2×2，比较融合是否超过较强单分支。

以上是设计评估，尚未实现模型适配器/新目标或开始训练，不把“继续”自动解释为重启已完成旧实验。

# 历史交接：2026-10-14 先核查异配分类强方法

历史补充（2026-10-14评估时）：用户附“新建文件 (5).txt”，请求评价无标签BFS替换GRACE+几何分组的路线。评估见research/topology_ball_review_20261014/00_BFS拓扑粒球替代GRACE路线审查.md，来源哈希与三个微型CPU核验在evidence.json。结论为值得低成本目标机制筛查，不能预期全异配普遍改善；需原X几何、同粒度GRACE、随机大小/训练支持匹配及kNN对照。评估当时未实现或训练；2026-10-15获用户明确授权后的已完成实验见本文件顶部。Texas当前87训练节点，不用20%估计；初始BFS约13节点不能套支持16门槛。原稿CoraFull74.48分钟已核，但预训练会复用。无标签分区保留理论分析条件，仍不保证root正margin。

本节记录此前SOTA核查要求，与下方分球路线相比仍更晚；当前最新优先级以上方GREET/HALF审查为准。SOTA核查轮没有新训练或模型改动。

- 主报告：research/heterophily_sota_20261014/00_异配图SOTA与GBLCSS干净对比.md；逐篇机制/协议卡：01_强方法与机制核读.md。公开来源及SHA、PDF表格截图、12行本地clean汇总和protocol_audit.json均在该目录。
- 本地clean原GB/MLP：Roman65.79±0.51/65.70±0.44；Texas70.81±5.86/80.54±7.97；Wisconsin76.47±5.55/79.22±5.65，前5官方split。CoraFull46.42/47.25、CS91.69/93.09只seed0。新support16 Cora49.52、CS92.36也只seed0，不是正式多seed新成绩。
- Roman文献参照：TunedGCN91.27、Polynormer-r92.55、GraphTARIF93.23（WWW2026）、Dir-NT94.77（预印本有向版），无向NT91.71。25–29pp仅跨论文参考跨度，不承诺X-only追回。ES-MLP无边推理65.44接近本地GB，不据此证明胜过或已到上界。Amazon SAGE+TS56.72为本轮标准密集划分最高已核参考，仍非穷尽榜单。
- 新补HarmonyGNNs ICLR2026：无标签编码器+冻结线性探针Roman75.86±0.47；同表GRACE51.58。属性投影、多深度加权GCN、token融合、EMA师生掩码训练很贴合第一阶段，但不把线性探针增益当LCSS学生增益。原文PDF给出RXPHD/H3GNNs-ICLR-2026代码链接，尚未审计实现/masks。
- CoraFull CNA71.82当前作者加载器默认80/10/10，另分支40/30/30；本地20/10/70，历史对应commit未核，不能相减作为同协议差距。WebKB HiGCN/HPGNN60/20/20不同本地约48/32/20。ARGNN为F1、FSpecGNN为2.5%标签、GraphPFN为10/10/80+预训练，不混排行。
- 数字纠错：Polynormer-r Amazon正式Table2是54.81±0.49，草稿±0.11已通过原图更正。HiGCN正式表Texas92.15/Wisc94.89，与arXiv/转引有版本差异。基准README定义无向边单次存储；不能断言NT文件方向具有真实语义或本地无向化错误。

下一讨论优先确定两条部署路线的同协议强参照及第一阶段机制；不要直接把“调球数”当足以补齐全监督模型差距。用户已允许比较有图/无图，不能再把graph-free当硬约束。

# 历史交接：2026-10-14 自适应粒球两轮完成

本节记录上一轮授权与已完成工作，当前优先级以上方文献核查交接为准。上一轮先修改粒球生成，参考sqrt(N)，再依据机制推进。所有相关队列已完成，无后台训练待续。

## 新窗口先读

- 01_literature_and_research_map.md、02_experiment_history_and_lessons.md；后者为新补的工作目录入口，原始方向记录仍在../../experiment_history_and_lessons/。
- experiments/adaptive_support16_ball_20261014/综合结论与接续.md、PLAN.md、后续对照结论.md、audit/report.json。
- experiments/adaptive_sqrt_ball_20261014/PLAN.md、实验结论.md、decision.json、audit/report.json。
- GB-LCSS_code/src/gblcss/granular_ball_adaptive.py；新生成器已实测，旧granular_ball.py及正式默认入口保留。
- research/algorithm_redesign_20261013/reviews/core/granular_generation_sqrt_and_geometry.md；涉及旧失败仍需回查原实验历史，不能只读本交接。

## 当前结果

新规则：ceil(sqrt(N))预算、ceil(N/(4B))最小叶球、最大总SSE优先2-means、可行投影fallback、无几何收益停止。support只读取训练集合位置，不读类别。内部float64和分裂随机种子规则也有变化，不把整体规则差单归因于一个门槛。

第一轮支持4：五图×五协议seed×三条件=75目标单元/600行，75次seed0学生200轮；审计25协议+75学生通过。Texas14/Wisconsin16/Roman151/CoraFull141/Coauthor-CS136，五协议均达到。带噪Brier新−旧64配对均值：+.048255/−.000080/−.007964/−.039477/+.002129；Roman/CoraFull过探索目标门槛，小图学生明显退化。Cora预算和规则均有信号，Roman主要预算效应；强KM/KNN目标对照仍更好。

第二轮唯一固定新变量支持16，仅seed0，15单元/120行+75学生；审计5协议+75学生通过，30项历史CE/旧64准确率全一致。由独立人工噪声原频率Var≤1/(4s)和预设std .13得其他票≥15，每球≥16；不保证语义正确，不界定q/alpha非线性误差，不声称阈值最优。

实际球数3/5/151/141/136；小图低于预算上限5/7，叶球不足32训练节点无法再分。不强行凑sqrt或上限。第二轮新−旧64学生pp（clean/U/Pair）：Texas+2.70/−5.41/0；Wisconsin0/−7.84/0；Roman−.11/+.46/−.30；CoraFull+3.10/+2.66/+1.02；CS+.67/+.13/−.73。全部N=1学生pilot，只有第一轮目标有五协议seed，均只有一个SSLseed。

## 决策与禁止误读

保留新生成器研究版；CoraFull候选值得固定后配对确认。不能主张通用sqrt最优、统一改善异配图、粒球优于kNN/KM或新对比算法已成功。小图支持16部分修复支持4损害，仍未统一超过旧版；CS Pair受损。当前不继续搜8/12/20、不按test逐图挑4/16，也不立即叠OFA/新损失。下一层先诊断冻结分组的干净投票边际、跨球同类遗漏、错误修正与q可靠性；ESS×JS旧负结果仍有效。

34测试+20subtests通过；最终validation_20261014.json记录命令。两轮source/config/hash/目标/关系/原始日志/CSV/logits/逐球拒绝记录保留。环境仍D:/codex_temp/noisygl-rocm721/Scripts/python.exe，RX7700XT；单GPU队列退出0，不要重启已completed任务。

旧交接原件已归档experiments/adaptive_support16_ball_20261014/previous_handoff_before_generator_20261014.md；下方为历史路线，若冲突以本节与综合报告为准。

# 历史交接：OFA引用核查与原GB-LCSS异配基线

当前没有后台训练待续。用户希望先把原算法放到几个异配图测试，再依结果逐层修改；graph-free部署未锁死。新算法OFA尚未实现或测试。

## 下一窗口先读

- research/algorithm_redesign_20261013/06_OFA引用核查与先基线后改进.md，以及三篇新核读卡HeterGCL_OFA/HOG_GCN_Wang2022/CCA_SSG_OFA_theory。
- experiments/heterophily_original_baseline_20261014/PLAN.md、实验结论.md、decision.json、audit/report.json、student_results/summary/paired_deltas.csv。
- experiments/heterophily_mechanism_20261014/实验结论.md、pipeline_status.json、audit/report.json：上一阶段已完整完成，不要重启队列。

## 已完成

原机制队列60单元/840行、48次seed0学生pilot，20个协议运行审计通过。GRACE Roman1000轮、GraphACL可移植Roman扩展50轮均完成；后者维度/预处理/预算不同，非官方指标复现。

最新原算法基线：Texas/Wisconsin/Roman，五官方split、clean/Uniform0.4/Pair0.4，CE/原GB64/票数匹配KNN64，135次200轮学生运行完成；源清单、冻结目标、原GB64公式、保存logits准确率和完整组合覆盖核查通过。Texas/Wisconsin新增目标10协议运行审核，seed0深度重建；Roman目标复用上一阶段已审计记录。两适配器测试通过，另19项原机制测试此前通过。

## 结果与边界

GB64−CE五split平均增量pp：Texas clean−9.73/uniform+7.03/pair≈0；Wisconsin−2.75/+2.75/+7.45；Roman+0.09/+0.02/+1.09。Roman Pair5/5改善，其余条件需结合完整配对std/同向率。Texas/Wisconsin干净损害大，不支持无条件迁移共识；WebKB小测试集一节点对应2.70/1.96pp。每图仅一个SSL seed，不能称五次独立预训练，也不能与文献不同protocol直接排名。

实际小图球数Texas均4、Wisconsin5/5/5/4/5；Roman GRACE64。原min_child32/分裂准则是待诊断瓶颈。OFA引用链[42]HeterGCL、[64]HOG-GCN、[65]CCA-SSG；Mlp原属性是合理架构动机，不能称独立创新或普遍优于GNN。互信息解释有归一化/分布/增强假设，不证明带噪分类必改善。

ESS×JS旧机制未过门槛，暂缓训练；不要无新信号重复调温度。当前下一步先固定球/软标签/学生，比较GRACE、原X和标签无关OFA空间；然后定位分裂或修正问题，一次改一层。本轮未做GCN分类对照、未做OFA消融、未测试Amazon-ratings。

## 实现故障与来源

旧Roman加权一次申请82.5GiB已通过diagnostic_v2内存分块解决，旧script/结果保留，逐比特小图回归及旧关系hash复核通过。CPU/GPU kNN边界差异曾被严格指纹捕获，保留初始失败和portability记录；当前审计用原GPU。队列GBK日志已改UTF-8，失败历史保存，不需重试已完成阶段。

运行环境D:/codex_temp/noisygl-rocm721/Scripts/python.exe，torch2.9.1+ROCm7.2.1，RX7700XT。正式GB-LCSS代码和已投稿论文未改；新脚本、数据、训练、核读、记忆均在可写workspace。旧中途交接归档在新基线目录previous_handoff_midrun_20261014.md。
