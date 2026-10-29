## 2026-10-29：两路线初筛完成，交叉拟合局部有益、原底座仍更强

用户授权后完成两条初筛：CS/Photo、seed0/1。S1对U.4/U.6/Pair.4做12反事实头＋12原头复现与缓存诊断；S2仅Uniform0.4、20轮短续训，20SSL＋20头，固定H0读出。完整计划、代码、报告在experiments/graphecl_honest_pilot_20261029；运行根D:/codex_temp/graphecl_honest_pilot_20261029。

ContextECL反事实H0＋新目标：CS+0.149±0.061pp、Photo−0.333±0.233pp；Photo票源变化94.97%、与Brier变化rho0.048，单纯变动大小解释弱。路线1GB额外AUROC+0.007148，KM−0.000769，GB的Brier预测MSE略增；未过+.01及连续指标门，未启动新增集合KL编码器训练，KMeans仅缓存对照。

交叉拟合测试real−等预算折内ECL：CS+0.043±0.039pp、Photo+0.411±0.158pp；real−打乱+0.023±0.011/+0.252±0.224pp；real−原全票−0.062±0.011/−0.439±0.013pp。验证宏real−b20+0.251pp（两seed+0.371/+0.131），real−打乱+0.084pp，real−原全票−0.185pp。Brier对b20：CS−0.009907、Photo+0.002723。两个推进门均失败，暂停本规格扩噪声/轮数/额外消融，clean未参与否决。

机制：真实软目标加权正对同类率CS83.43%、Photo85.90%，超过同球均匀78.94%/81.00%；实际H有变化。拆折仍约32票，但CS区域总节点156→329、Photo160→313，真实类别票差均下降。Photo纯ECL20轮全票也低原底座0.588pp，需同时分析续训与拆折代价。打乱改变有效正对数（CS25.39→5.05、Photo22.07→6.60），未来归因应匹配权重结构。Uniform.4下原SUPPORT相对CE已近零，较高噪声尚待独立验证。

9机制测试通过；20SSL/44新头/400SSL更新/8800分类头更新/428产物哈希独立核查，指标差≤2.22e-16；12原头概率逐位复现、20SSL和最终28头GPU重载通过；4格原独立转移噪声缓存重建一致。正式分支训练及保存共14.48分钟，构球/检查/诊断另计。前台进程一次无异常原因退出，7SSL/5头后由隐藏启动器接续，已完成产物复用；当前launcher COMPLETE、review passed，无待续训练。

已支持：局部正对更纯、相对折内对照小幅改善、净收益未过门。合理解释：区域扩大与续训漂移抵消收益。待验证：保持几何尺度/稳定续训是否改善、普通软监督对比与粒球额外贡献。削弱：当前KL定义与两折SUPPORT32短续训直接替换原底座。下一窗口先读RESULT_INTERPRETATION_zh.md及results，勿重启已完成队列；原稿与旧训练源未改。

## 最新完成：2026-10-27 GCN新监督五图三seed及DIST

全部45格、六路教师和学生完成，COMPLETE_AUDITED。入口research/gcn_support_d1_c1_3seed_20261027/RESULT_INTERPRETATION_zh.md；实验同名HANDOFF、summary/paired/gap CSV与TeX，D同名runtime重产物。无活动任务待恢复，勿重启旧launcher。三seed、五图、clean/U.4/U.6，不含新Pair；旧9格只读复用，新36格。

带噪五图seed内等权宏平均：完整GCN76.676±.697%，完整DIST学生77.464±.449%、F170.862±1.401。学生对CE+2.559±.315pp、OLD64+.832±.222、历史SUPPORT32+.551±.458、固定32+D1+.174±.168；完整学生对自身教师+.788±.313。逐图学生77.590/66.123/58.573/93.908/91.127%；对CE+5.215/+4.646/+.294/+.634/+2.004pp。clean对CE−.047±.304、对OLD64+2.016±.260；CF clean对SUPPORT32+3.051±.209。C1附加收益小，CF学生低固定D1，Photo对SUPPORT波动较大，全部负格保留。

新增1620教师轨迹/197物理学生/173925成功更新，另63中断更新单列。29tests通过；初次CPU audit漏绑定relative_input导致NameError，独立audit_recovery_v1仅绑定冻结analysis原校验函数，旧源/结果/选择保持。16:21:59完整审计passed：540metrics/2745prefix/2520snapshots/2270targets/720stats/453oldrefs/90CPUreload，最大概率差1.90735e-6类别同，270支持强度诊断。complete及launcher=COMPLETE；失败和wrapper hash保留。不新增训练、稿件或上传。

## 历史恢复：2026-10-27 15:34 GCN队列接续

原GPU64372及队列60548均已退出，无异常栈、未发现对应系统重启，原因待定。完成25/45教师格，763已有轨迹核验；唯一中断为CF seed2 U.4 S128_L0.5，日志138轮/模型optimizerRNG断点75轮。独立recover_interruption.py完整归档现场和残留锁到D同名runtime/interruption_recovery/20261027T073427Z，仅对齐有效日志到75，原训练/分析源码与科学选择冻结不变，额外63中断更新单列overhead。

28/28 tests passed。15:34:41原launch.py恢复，venv46732/实际driver66220/实际GPU55948，单队列串行fit→analysis→audit。中断候选已到200轮，原138轮日志与恢复后前138轮逐字节一致（含重跑76–138），replay_verified.json passed。继续其余教师与学生，无需重复队列；最新PID以runtime/status、launcher_status、lock和现场进程为准。尚无正式最终精度结果。

## 历史启动：2026-10-27 GCN新监督扩为五图三seed

用户明确授权GCN教师使用SUPPORT+D1+C1，扩展原三图seed0到五图seed0/1/2，保留clean/U.4/U.6，共45格；旧9格严格核验复用，新36格。六路教师CE/OLD64/LEGACY_SUPPORT32/DEEP_S32/FIXED_D1/JOINT及原DIST学生，270逻辑学生端点（54旧/216新），精确同概率可别名。GCN自己验证选参，无R2初始化；本轮无Pair/KM/kNN/新SSL/稿件或上传。

入口`experiments/gcn_support_d1_c1_3seed_20261027/HANDOFF.md`、PLAN/config；重产物D:/codex_temp/gcn_support_d1_c1_3seed_20261027。6合同测试、45格CPU预检、5条两轮GPU smoke、断点model/optimizer/RNG一致性及旧GCN/DIST初始化/前两轮桥接均passed。14:34:55隐藏启动run.py fit，venv PID61784、实际GPU worker64372，14:35进入Cora seed1 clean；这是启动记录，后续以status/进程为准。运行锁run.lock，进度status/events，完整background_stdout/stderr。

独立CPU分析/审计适配器已交付，主会话最终25/25 tests通过，另旧Cora clean单格61prefix/56snapshot/全部教师目标/2模型CPU重载/410hash passed；postprocessing_sources单独冻结。旧两次后处理测试失败留痕修复，拟合源码未变。14:42:26一次性launch.py接续driver已启动，venv50480/实际60548，WAITING_GPU正确识别现拟合61784/64372；该driver的training_started=false指它自己尚未接管，GPU拟合已活跃。现worker退出后，driver幂等skip已完拟合并串行analysis→audit；勿重复启动。14:47:43已有24/45教师格（旧9/新15），当前CoraFull seed2 clean，无failure_latest。全部教师选择冻结后训练学生，全部学生选择冻结后统一新test；complete仅独立audit通过后生成。当前无新精度结论，旧已完成R2→DIST五seed表仍保留，勿混作GCN结果。

## 最新完成：2026-10-27 CS/Photo蒸馏补齐，五图n5表已生成

入口`research/r2_teacher_dist_csphoto_20261027/RESULT_INTERPRETATION_zh.md`与实验同名REPORT/HANDOFF/complete。两图clean/U.4/U.6×五seed×CE/FULL教师，60新学生12000更新完成，固定原配置，教师缓存与选参不变。原SSL退出后队列自动串行GPU，当前queue_status=COMPLETE、锁均释放，不重复启动。

CS/Photo完整R2→DIST带噪accuracy92.470±.197/84.842±.379%，比CE→DIST+.676±.100/+2.084±.800pp，F1+1.428/+1.656；cleanaccuracy差−.019/+.373pp。新五图完整学生73.692±.488%、F166.563±.655，对CE学生accuracy+2.071±.136/F1+2.024pp，对自身教师accuracy+1.170±.186。五图clean对CE学生accuracy+.124、F1+.116。与主线三噪声宏均值分开，当前DIST仅U.4/U.6。

5契约/60教师预检/4两轮smoke及续训passed；独立120新教师/学生CPU重载、300合并指标、288meanSDpassed，最大概率差1.43e-6类别一致；额外492配对汇总与60净纠错检查passed。最新表`experiments/r2_teacher_dist_csphoto_20261027/tables_five_graphs/`，D盘同名runtime权重/日志；旧三图表保持。无稿件改动、上传或自动更多实验。

## 历史：2026-10-27 CS/Photo蒸馏补跑已排队（已完成，见上）

用户要求补齐Coauthor-CS/Amazon-Photo蒸馏，沿用完整R2与R2+CE两教师、seed0..4、clean/U.4/U.6，共60新学生12000更新；完成后合并旧三图90学生，输出五图n5表。入口`experiments/r2_teacher_dist_csphoto_20261027/HANDOFF.md`、config/PLAN，重产物D:/codex_temp/r2_teacher_dist_csphoto_20261027。5契约测试及60教师CPU重载/输入hash/冻结选择预检通过，0新test结果。

现有SSL worker60708/launcher41452仍运行。已启动单次后台gpu_queue.py，launcher49716/实际queue45164；queue_status.json=WAITING_GPU/training_started=false，其他Python退出后自动run.py all（两图短smoke→60fits→freeze→analysis）再audit.py→COMPLETE/FAILED。不重复启动、不停止SSL、不创建新定时任务。排队只是标准库进程，不分配GPU；只读进程检查60秒间隔，原任务未改。queue.lock防重复；若FAILED先查background_stderr/queue_status/failure_latest，确认进程再修复。

五图结果未来在tables_five_graphs，旧三图文件不覆盖。当前无新增精度结论，未改稿件或上传。下面SSL记录保持原样，GPU任务完成状态以其formal/status.json为准。

## 正式启动确认（2026-10-27）

最终版 29 项测试和 12 项 GPU 冒烟任务全部通过。run.py all 已在隐藏后台启动；已确认首个正式 GRACE/Cora/seed0 进入第 650/1000 轮，损失有限，当前 failures 为空。启动器 PID 41452，实际训练 worker PID 60708（以 formal/status.json 为准）。后续自动按冻结矩阵执行下游、A/B阶段，拟合全部结束或明确失败后统一评分与 CPU 审计。当前尚无正式精度结论。

输出：D:/codex_temp/ssl_backbone_screen_20261026；进度 formal/status.json，各任务 progress.json；完整日志 background_stdout.log / background_stderr.log；最终 analysis/REPORT_zh.md。无需重复启动；先核查单实例锁和状态。
## 2026-10-27 SSL 空间对照启动

用户已授权立即执行冻结方案。代码位于 experiments/ssl_backbone_screen_20261026；产物位于 D:/codex_temp/ssl_backbone_screen_20261026。29 项适配测试通过，145 个源/输入哈希通过。当前正在最终版三图四方法 seed0 两步冒烟，完成后按同一 receipt 启动 24 项正式 SSL、28 R2、224 R2 头、70 线性头（含 GraphECL native 别名，共 294 唯一预测单元）。

原延时启动 automation ssl 已暂停。训练入口 run.py all，状态 formal/status.json，日志 background_stdout.log/background_stderr.log；完成/失败记录 formal/fits_locked.json，评分审计 analysis/。正式结果尚未生成。

方法与超参保持 research/ssl_backbone_selection_20261026/protocol.json。工程适配及失败排查见新实验 README.md。GPU 确定性核已启用；旧工程尝试保留为 engineering_attempt1_nondeterministic / engineering_attempt2。测试标签在计划拟合完成或失败锁定后统一读取。两 seed、无 kNN/KMeans 扩展、无新调参。
# 最新准备：2026-10-26 成熟SSL空间对照

用户要求先设计实验、拉取官方代码，下一次一句话接续。入口`research/ssl_backbone_selection_20261026/START_HERE.md`；完整方案EXPERIMENT_PLAN、protocol.json、IMPLEMENTATION_PLAN、PREPARATION_REPORT在同目录。AG-EC后续暂缓。官方GRACE/GCA/ProGCL/GraphECL/BGRL已浅克隆稀疏检出并锁定5提交/96文件hash；49输入/配置hash，合计145项核验通过，3准备tests通过，无正式训练队列。

首轮A三图两seed的GRACE/GCA-degree/ProGCL-weight-degree，共18SSL；B同三图两seedGraphECL6SSL。family1000轮、Pro400启用，另留Cora/Cite的GRACE200快照；GraphECL保留1024/2048/1024维、100/100/1000轮。新R2+CE/SUPPORT32固定协议，GraphECL另列原生MLP。无kNN/KMeans、D1/C1/DIST搜索。RELGCL官方实现未确认暂缓，BGRL只归档。

统一适配器和训练run.py尚未实现；用户说“按冻结方案开始”后按IMPLEMENTATION_PLAN完成适配/核验/smoke，再直接运行正式阶段，常规步骤无需重复确认。已实测ProGCL官方batched预热核与dense不一致（小矩阵maxdiff4.034447），GCA零列权重非有限；Cora有1零列、Cite有15零行。GraphECL依赖DGL缺失，按计划用稀疏数学参考适配并核验，禁止把缺失DGL对照称已通过。官方文件保持原样，旧结果不动。大产物建议D:/codex_temp/ssl_backbone_screen_20261026，按当前权限正常处理输出写入。准备工具`prepare.py verify`只做hash检查，training_ready=false属真实状态。

# 已完成：2026-10-26 R2完整主线与教师DIST五种子表

本轮全部完成、无运行队列。入口`research/r2_five_seed_tables_20261026/REPORT_zh.md`与HANDOFF/complete；主线五图clean/U.4/U.6/Pair.4 seeds0..4，旧0..2保留、新3/4；完整R2教师DIST三图clean/U.4/U.6五seed，旧18seed0+新72seed1..4。算法/参数/选择规则固定，无稿件/上传/额外实验。

主线完整accuracy71.111±1.022%，macroF163.835±.962；对CE/GCE/SUPPORT/验证较优单模块accuracy差+4.057±.290/+.530±.328/+.945±.439/+.234±.173pp，五seed宏差均正。Cite对较优单模块−.032，CS/Photo仍略低GCE，clean对CEaccuracy+.065、F1−.198pp。完整组合额外收益较小，负格全部保留。

完整R2→DISTaccuracy63.716±.737%，对CE→DIST+2.532±.272（5/5seed正），对自身教师+1.096±.087pp；Cora73.858→73.852、Cite67.006→67.595、CF46.997→49.701。差中差−.112±.566pp，教师本身优势基本传递。clean对CE学生accuracy+.088/F1+.105，Coraaccuracy−.644。主线三噪声与DIST两噪声宏均值分开，不直接比较71.111与63.716。

新主线1840候选轨迹、159925分类更新、10新100epochR2；旧三seeduniform因果回放与新两seed真实prefix续训区别明示。GPU串行，13主线/启动器+6分析+7KDtests及exact桥通过；parent1000端点/100CPUreload、3520prefix、2263快照、独立目标passed，3078统计passed；KD180教师/学生CPUreload+指标/seedstats passed。父进程识别启动失败与报告GBK打印错误留痕修复，未改科研参数或重跑已完成拟合。

主线表`experiments/r2_integrated_seeds34_20261026/tables_n5/`，KD表`experiments/r2_teacher_dist_five_seed_20261026/`；CSV与TeX齐全。重产物分别在D:/codex_temp同名实验目录，C薄表/代码；用户未授权上传。旧GCN/ERASEseed0参照不混作五seed排名。下一窗口只读报告/表，不重复启动。

# 本会话完成：2026-10-26 完整R2教师→DIST三图seed0

入口`experiments/r2_teacher_dist_pilot_20261026/REPORT_zh.md`和`research/r2_teacher_dist_pilot_20261026/PARENT_REVIEW.md`。复用r2_integrated的CE/ADAPTIVE_EXPANDED冻结教师，三图clean/U.4/U.6、18个随机初始化原MLP64/DIST200学生，3600正式更新。无新教师/SSL/search/seed，所有选择先冻结再测试；无待续GPU任务。本节与下方其他方向记录分别保留。

带噪完整教师61.508→学生62.722%；R2+CE教师59.017→学生60.185%。完整学生比CE学生+2.538pp、macro-F1+2.102；教师优势+2.491pp基本传递，差中差+.047pp。完整再蒸馏逐图Cora71.530→71.530、Cite66.194→66.817、CF46.799→49.820；CF普通CE再蒸馏也+3.237pp。完整学生比GCN完整DIST分别−5.013/+.816/−8.614pp，算法统一扩展可保留，Cora/CF更强结果仍属图教师路线。clean完整学生对CE学生平均accuracy+.020，但Cora/CF分别−.422/−.505pp；单seedSD=null。

C盘空间保护在0拟合时停止，保留原failure/sources记录；新runtime只在`D:/codex_temp/r2_teacher_dist_pilot_20261026`，receipt为项目实验目录`runtime_location.json`。代码和薄报告/结果副本在项目，实际训练hash用`runtime_sources.json`或D/sources.json。5tests、2条2epoch续训smoke及旧GCN2epoch桥接；18教师/学生GPU重载误差0，独立18+18CPU重载、指标、firstmax及冻结hash链passed。无稿件改动/上传/自动更多训练授权。

# 最新完成：2026-10-26 AG-EC自主表示替换三图两完整seed

优先读`research/granular_evidence_ssl_20261026/RESULT_INTERPRETATION_zh.md`、`PARENT_AUDIT_V2_zh.md`及实验同名目录README/analysis。18SSL、18独立R2、144头全部结束，22实现测试、3附加回归，独立1322检查通过；无待续拟合。用户暂缓kNN/KM对照。

新框架为互补属性SVD锚定+粒球候选受限注意力+证据预测/阈值集合对比。三图seed0/1、四条件，GNN200轮；GB预热50、s32/C128/R4/lambda.1；同规格新R2映射100轮、固定CE/SUPPORT32头200轮。未套用旧R2权重，下游对齐最新E1树4096/8与LOO/beta.5。

带噪SUPPORT32 GRACE54.519/Base52.249/GB53.244%；GB−Base+.995±.178pp、三图均值正、两个seed宏差+.869/+1.121；对GRACE−1.274±.490pp。Cora/Cite/CF三图GRACE→GB为70.176→57.142、64.046→58.126、29.334→44.465，CF只为共同200轮参照，历史主线1000轮。Cora clean GB−Base−2.216（两seed−.633/−3.799）触发门失败。停止当前规格扩种子，保留证据模块信号，优先复查基础自监督目标。去set消融/区域对照/学习权重标签投票均未运行。

机制代理改善小、与类别证据rho约.11–.16；Cite干净证据Brier反升。零训练空间分析显示Cora/Cite差距在R2前及投影前已有体现。所有结果按完整seed先聚合并报样本SD，不称显著。GPU正式训练无失败；v1审计54误报（floor全覆盖、Brier缺arm键）已在v2修复，1322项通过，原v1保留。H重载max6.41e-7、144概率max1.79e-6类别同。散冒烟/候选已无损ZIP归档，有SHA；正式H/模型/目标/概率仍在原处。

下一窗口先读结果报告再定新假设；如仅复核，运行`research/granular_evidence_ssl_20261026/parent_audit_v2.py`，避免重复启动原训练或使用已知有审计错误的v1。Python固定`D:\codex_temp\noisygl-rocm721\Scripts\python.exe`。以下保留前次交接原文。

# 前次完成：2026-10-26 最新监督GCN→DIST三图seed0首轮完成

入口`experiments/teacher_integrated_dist_pilot_20261026/REPORT_zh.md`与`research/teacher_integrated_dist_pilot_20261026/PARENT_REVIEW.md`。三图seed0clean/U.4/U.6，GCN自己42→14→4选支持/分组，原模型/初始化/协议，DIST固定；405教师候选（含早停）、48实际/54逻辑学生，43000更新、5smoke，全部完成，无任务待续。无更多seed/图、稿件或上传授权。

新学生带噪66.993%，对SUPPORT+.1764pp（教师+.060）、fixedD1+.106、OLD64+1.449、CE+4.239；三图对SUPPORT−.106/+.494/+.141。CFclean学生+3.291对SUPPORT（fixedD1也实现），但macroF1仍比CE低.967。新/ERASE教师/ERASE-DIST带噪均值Cora76.54/79.55/80.00，Cite66.00/65.14/65.76，CF58.43/58.63/58.83；CF普通CE教师DIST也约58.43。保留简单对照，不按测试逐图挑路线。

6tests/5smoke，405候选/504前缀状态及18GPU重载通过；parent9races/549prefix/18选择/108paired/384summary/108指标/18CPU重载通过，最大差1.37e-6类别相同；18历史教师桥误差0。训练后同名analyze导入冲突，新增绝对路径finalize_results.py恢复汇总，旧源码/失败日志不改，无重训。计时GCN/MLP原生整图forward+softmax：Cora2.137/.469ms，Cite2.356/1.120，CF6.332/15.346；CF学生更慢，X-only与普遍加速分开。单seedSD=null、固定SSL、开发数据。当前保留小增量与clean修复，不自动扩展。

# 前次完成：2026-10-26 R2+SUPPORT+D1+C1三seed整合全部完成

入口`experiments/r2_integrated_20261026/REPORT_zh.md`与`research/r2_integrated_20261026/PARENT_REVIEW.md`。五图四条件三seed60格，1500条新增候选轨迹（407完整200轮）、123475更新、4smoke。无KM/kNN新增、无新SSL/R0/teacher；历史全局轨迹因果前缀复用，新分组真实续训，平票lambda1是显式新约定。全部选参冻结后才读test。脚本独立完成，agent额度中断未影响训练；无剩余Python进程，不重复启动。

完整带噪accuracy71.5305%，较R2CE/GCE/OLD64/SUPPORT为+3.901/+.379/+1.591/+.916pp；较验证选较优单模块+.307±.165pp，macroF1+.474。三seed增量均正，五图均值为+.610/+.220/+.048/+.213/+.446。clean对CE平均accuracy+.135、macroF1+.107；CFcleanF1−.286仍保留。联合2450更新含CE，全局1200、固定D11600，搜索预算差公开。历史开发seed/固定SSL，本轮整合验证；不宣称超KM/kNN。

Worker11tests/2760目标/120race/30GPUreload通过；parent120race/5280prefix/300选择/660paired/1134summary/30CPUreload独立通过，最大差1.37e-6且预测一致。决定保留完整组合供论文整理，无自动更多实验、稿件更改或上传。

# 历史完成：2026-10-23 R2正文对照全部完成

目录`experiments/r2_body_comparison_20261023/`。480个R2头+40外部单元完成。后台最后analyze同名导入冲突，finalize_results.py绝对路径恢复汇总与独立重载；无重训。REPORT_zh、summary、paired、matched_seed0、decision、delivery_audit齐全。480模型重载误差0、104额外源哈希、五GSSC split/noise一致，审计passed。无待续任务。

三seed带噪五图宏均值 CE67.630/GCE71.151/OLD64 69.940/SUPPORT70.615/KM70.985%；SUPPORT−CE+2.985pp，−OLD64+.675，−GCE−.536，−KM−.370。KM逐图均高，验证选固定竞争者后SUPPORT五图均负，门失败；clean相对CE Cora−1.953/CF−1.030pp。相同seed0外部SimMLP 4/5图高于SUPPORT（CS例外），GSSC较弱。当前粒球监督增量未胜强对照，保留R2骨干扩展与LCSS相对CE增量，不自动调参或扩种子。β历史配置和固定SSL/外部单seed范围保留。

# 历史启动：2026-10-23 R2正文对照队列

目录`experiments/r2_body_comparison_20261023/`。读PLAN/HANDOFF/ENGINEERING；background_status.json与status.json为运行入口。全部480个R2分类头已完成（五图四条件三seed八监督），尚待外部对照全部完成后统一test评分。

外部SimMLP/GSSC五图四条件seed0共40单元，历史5个GSSC uniform.4复用，其余补跑。当前独立后台background.py串行执行recover.py，然后自动analyze和verify_and_interpret，最终生成REPORT_zh/summary/paired/matched_seed0/decision/delivery_audit。不要重复启动。后台实际PID见background_status.json（启动时111568）；run.lock是当前训练子进程。

原前台63382在GSSC历史config文件不存在处退出；原源/失败保留，新external_v2仅改读旧result的config字段。恢复前台51627已停止，切换持久后台；中断的CoraFull pair.4 GSSC日志在interrupted/corafull_pair04_GSSC_foreground保存，该单元完整重跑。已完成模型均复用。大图GSSC单元约12分钟，预算保持原辅助阶段+200主训练，故队列较长。

4测试通过，15个初始分类头共享组验证一致，Cora历史SUPPORT概率桥接误差2.384e-7。新实验全部预定拟合结束前不读测试评分；原稿和旧实验未改。下次先查后台状态与进程，完成后读取报告并补详细结论；failed时先读日志，保留部分单元再修复。

# 前次执行：2026-10-23 节点路由并列对照已完成

入口 `experiments/node_channel_routing_20261023/REPORT_zh.md`、`HANDOFF.md`、`final_audit.json`。用户另行授权直接比较节点路由与全局融合，原L1停止记录保持。五图四条件seed0缓存，新增神经网络训练0；8个学生自身特征、L2逻辑回归C1，外5内4嵌套验证，20规则全部落盘后评分test。

验证门通过：ROUTER−GLOBAL带噪Cora/Cite/CF/CS/Photo +.864/+.904/+.185/−.200/+.044pp，clean均满足下限。测试同差−.229/−1.017/−.137/−.231/−.093pp，五图均负；相对最好单通道−.035/+.773/+.060/+.036/+.349pp。Cora路由AUC .558，Photo .796但净增量很小；Cora/Photo macroF1较GLOBAL +.935/+.616pp，其他三图负。验证正信号尚未兑现为测试精度优势，暂缓加复杂模块；图/GB统计未跑，单学生融合蒸馏未跑。

5tests、620路由模型独立重拟合、20外折预测、140指标行、20旧全局权重、9源90输入哈希通过。首次末尾哈希审计Windows默认编码损坏中文路径，run.py只加UTF8读取；原failure/source保留，finalize.py还原原文摘要确认。完成标记存在，无待续进程。旧实验/投稿/远端未改。下一步先读结果，保持验证门通过和测试负结果两项事实。

# 前次执行：2026-10-23 通道互补/随机性对照完成，第0层通过，L1停止

入口 `experiments/channel_routing_variance_20261023/REPORT_zh.md`、`RESULT_TABLES.md`、`HANDOFF.md`。新增40个GPU学生（五图四条件A300/C_head）和2个短smoke，复用40旧模型；protocol0/split/noise/teacher/GRACE/R2固定，仅学生seed1，C_head只改变头初始化。无待续进程。

主门槛A0C0并集−逐格max(A0A1,C0C1)，带噪三条件均值Cora/CiteSeer/CF/CS/Photo为+4.080915/+2.720504/+2.381812/+1.109120/+1.929661pp，前三图全通过2pp门。四跨配对平均也支持额外互补，固定表示的头随机性与单协议范围写明。

L1全局权重由验证选，验证增量+.123457/+.903614/+.134748/+.272777/+.740741pp；前三主图仅CiteSeer过.5pp，需两图，所以停止L2–L5。测试GLOBAL比最好单通道+.1935/+1.7898/+.1973/+.2675/+.4420pp；CiteSeer直接平均65.178%/全局64.977%/C0 63.187%。局部正信号保留。节点级逻辑回归和图/GB统计均未运行。

9测试、两桥接exact、40配对、80新旧结果核验、40独立checkpoint重载、20融合重算、11源/405输入/5L1源哈希通过。完整mean/std为固定protocol下两个学生初始化。第0层显式测试筛查、L1验证选择，旧验证选checkpoint的复用记录保留。无科学运行失败；投稿/旧源/远端未改。

下一窗口：先读本轮报告。当前问题为全局融合收益有限，直接可观察路由仍待测；原阶梯在L1停止。新方案须另立计划，保留本轮门槛与停止记录。

# 前次执行：2026-10-23 SC选支持量、对齐预算及双通道蒸馏试跑已完成

入口 `experiments/sc_support_channel_pilot_20261023/REPORT_zh.md`、`RESULT_TABLES.md`、三个reports阶段报告、`final_audit.json`。按两小时后单次启动的授权完成E1/E2/E4；无待续GPU任务，E3/E5未启动，不重跑旧入口。投稿稿件、旧实验和远端未改。

E1离线五图三seed四条件360目标/60次选择，15树均六种不同截断；R0分类只三图seed0四条件144拟合。SC−VAL uniform验证−2.058519pp，SC−新树32仅CF+.050531、Cora−3.888889/Photo−.522876；clean验证两图低CE超过1pp，门失败。测试带噪SC52.9133/新树3253.1804/历史3253.4698/VAL53.9682%，不可用测试局部正值救门。SC−真投票候选排名平均rho U.4 .9377/U.6 .8857，估计投票质量不等于选出最佳分类目标。

E2审15旧日志，三条600轮对齐+72头拟合；三份历史100状态精确复现。300/600对100验证增量Cora−.694/−2.593、CS−1.923/−1.439、Photo−2.255/−.229pp，全留100。Photo600测试+1.071pp仅事后信号；固定100轮分类头及转导拟合限制仍在，不断言X不可预测H。

E4五图四条件seed0六臂120学生，复用15份noisy普通SUPPORT-LCSS GCN并补5clean教师。A200/A300随机KD200/300，B=R2 100+全模型KD200，D=R2+冻结头KD200，C_full/C_head=R2后全模型/头SUPPORT200。KD为全节点soft CE，不是DIST论文损失。B整体验证入选，但相对逐格验证选最强对照无一图达到+1pp，门失败。带噪测试B−A300五图宏差−.0143pp；CiteSeer D−A300+2.577pp同时clean−1.246pp，且D−C_head macro-F1−1.693pp，不事后逐图挑B/D。

分层新线索：Cora教师错节点D31.188% vs A300 11.884%，教师对节点D88.154% vs A300 95.919%，总精度下降；CoraFull正确侧损失更大，Photo错误侧无获益。支持纠错/保留正确预测存在权衡的描述；可观察的可靠选择规则尚未验证，不自动加模块。

总336正式分类拟合+3对齐+5clean教师，另4短smoke；分类全seed0/std=null。11测试、15源/115输入、336选模/预算/结果与模型概率哈希、20配对轨迹审计passed。两次后处理工程错误（arm重复、Path类型）已留痕修复，原脚本/日志/首次test_release/所有选择保留，训练未重跑。原fit_sources保留冻结版本，最终审计通过原分析副本及修复后摘要验证，不重启原fit入口。

下一窗口先读报告并与门槛核对；本规格停止自动扩展。全局选s不保证原留一独立性、pair不适用对称恒等式、beta历史开发选择仍需保留说明。

# 前次执行：2026-10-22 SUPPORT 教师替换蒸馏已完成

入口`experiments/teacher_support_distillation_20261022/REPORT_zh.md`及目录内PLAN/config、审计、CSV与ENGINEERING。复用旧LCSS与SUPPORT-LCSS教师端冻结预测，不重训GCN；5数据集×3协议seed×3噪声×3教师臂共135个属性MLP学生拟合，GPU完成，无待续任务。学生结构、训练预算、干净验证选模、split/noise固定；CE蒸馏作参照，主比较SUPPORT−旧LCSS。

学生准确率74.670/74.249%，配对差+.421±1.965pp（45格，描述性SD），25胜/0平/20负；macro-F1差−.567pp，教师准确率差+.405pp。分噪声学生准确率差U.4−.074、U.6+.328、Pair.4+1.009pp；macro-F1三者均负（−.911/−.556/−.234）。预设门为平均至少+.5pp、3/3 seed均值正、至少3/5数据集均值正；结果分别为+.421、3/3、3/5，整体失败。学生增益没有超过教师增益，且准确率信号集中于Pair噪声；三seed仅作描述性筛查，无clean条件，不能称显著、普遍稳定或无clean损害。

135检查点重载、选模和指标独立复算通过。审计器仅因CPU/GPU归约差异将离线指标复核容差从1e-8调到1e-6；训练及模型产物未变，详见目录ENGINEERING/analysis_amendment。保留结果，不自动扩种子、不据此组合新模块；若继续，需单独授权并先明确要验证的机制。

# 最新执行：2026-10-22 支持量/近邻身份三种子消融已完成

入口`experiments/neighbor_support_ablation_20261022/综合结论与接续.md`及目录交接、RESULT_TABLES、decision/audit。
576结果格216旧锚360新模型、24冒烟完成，无后台队列；原稿/旧源/远端未改。三图四条件、seed0/1/2、固定历史SSL/R2，
仅同树支持量与成员身份消融，k32固定无搜索。R0主、R2辅。

R0 OLD44.580/NP45.931/SUPPORT45.935/COUNT46.010/KMNP46.533/KNN46.851%；R2为59.306/60.525/60.509/60.426/61.143/60.843。
NP−SUPPORT−.004±.080/+.016±.127pp，NP−COUNT−.080/+.100，均未过.5pp；KMNP两路线均优于NP且3/3seed宏差正。
NP−旧版+1.351±.727/+1.220±.392，两个路线3/3seed宏差正；R0修复门通过，但Cora U.4三seed全负平均−.985；
R2clean Cora−1.917/CF−1.131且CF−旧版仅+.984，修复门失败。停止扩完整NP，保留简单R0返修修复，不自动拼模块。

18测试、288目标、9几何、576重载/选模/别名、288冻结R2、1333输入源哈希、32配对宏均值/样本std独立复算通过。
准备CE元数据兼容已修复；首次审计浮点归一化容差过紧，保留原冻结源，恢复入口analysis_recovery_v2.py只改数值检查。
最终audit.passed=true，无训练失败/重跑。不要重跑原analyze.py full或再次prepare/freeze。
COUNT/KMNP仍借NP球数，不称独立低成本选数；SUPPORT免近邻但端到端速度未测。不是32最优、正式等价或独立SSL确认。

# 前次执行：2026-10-22 邻域保真生成器seed0已完成

入口`experiments/neighbor_preserving_ball_20261022/综合结论与接续.md`及目录交接。三图四条件R0/R2，八臂192单元（144新拟合48旧锚）+16smoke，GPU全部完成；没有待续训练，旧源/稿件/远端未改。固定历史H与seed0编码器，构球只读H/train位置；256叶候选树、32近邻TV代价DP截断，实际20/22/142球，原LCSS与训练不动。

R0带噪OLD64 43.420→NP45.590，+2.169pp，8/9胜；相对TREE64/FINE +.996/+.782且9/9胜。clean对CE三图+2.533/+3.136/+1.754，修复门通过。R2 OLD64 57.921→NP59.461，+1.540pp，7/9胜，但CoraFull clean仍低CE1.068pp，修复门失败。

关键负结果：同球数KM为R0 46.347/R2 60.304，NP两路线都0/9胜，独立算法增量门均失败；KNNMATCH也高NP .734/.486pp。CoraFull clean目标Brier由.127173变.158005，R0 NP macro-F136.628仍低CE37.318。不能将分类涨分写成软目标或小类已修复。

9测试、96目标独立公式、16smoke/四精确历史桥、三图重建/96完整近邻排序、192重载与选模/96冻结R2、425哈希、32汇总168差值四门槛审计均passed。只是seed0，std=null；按PLAN保留R0确认候选而非宣称成功，不自动扩seed或拼多视图/GCE。下一轮若授权仅做同规格seed1/2、保留KM最强对照。工程两处审计假设修复见目录ENGINEERING.md，无训练失败。

# 历史执行：2026-10-22 GCE-LCSS融合已完成

入口`experiments/gce_lcss_fusion_20261022/综合结论与接续.md`及目录交接。504最终拟合、14冒烟、六份新R2无标签encoder完成，旧实验/投稿/远端未改，无运行中队列。三图四条件、协议和学生seed0/1/2，固定历史GRACE、每seed分区；不是独立SSL确认。

R0带噪LCSS44.5796、W_LCSS45.0674、GCE34.6563、W_MASS40.6584；W−LCSS+.4878±.4247pp，8/9均值正1平、3/3seed宏差正，低于1pp门。R0clean对CE三图+.5101/+1.7612/−.9142通过；CoraFull U.6+1.3016±.6003三seed正是局部返修储备。完整主门失败，不调参救援、不把+10.411对GCE当新增模块贡献。

R2 LCSS59.3056、GCE60.6031、W_LCSS59.5839、W_MASS60.9805；W对GCE−1.0191，对W_MASS−1.3966，CoraFull clean−CE4.179pp。SOFT_GCE R0低LCSS3.577pp，R2高LCSS1.804但Cora/CoraFull clean−CE3.237/6.135pp。不要根据70%错类目标argmax直接推断训练必败，也不因R2带噪增益忽略clean风险。

v1 GCE下溢NaN已用exp(q*log_p)等价稳定实现恢复，旧冻结源不改；97旧成功+1失败尝试保存在engineering_v1_runs，504是最终唯一网格。12测试、36协议、504重载/选模、48旧锚逐元素/轮次一致、252冻结encoder/BN、168summary/336配对/六门槛、161源和三恢复源哈希均passed。不要重跑原v1入口。

停止当前融合规格自动扩展，R0小幅正信号保存为返修储备；软归属/自适应生成/新聚类路线本轮未执行，不拼模块、不改投稿主表，下一轮需另行授权。

# 历史执行：2026-10-21 边界诊断与粗化复盘已完成

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

## 2026-10-27 SSL 空间对照结果（成功单元分析）

用户要求暂不处理失败seed，本轮无重训。22/24主任务完成，267/294预测单元完成评分及CPU重载审计；最大概率误差2.4438e-6，独立重算Accuracy/Macro-F1/Brier差0。缺格来自CiteSeer seed1 GRACE/ProGCL导出检查及GRACE200关联下游。

固定R2+SUPPORT32、完整seed0三图三噪声宏均值：GRACE1000 55.584%、预登记GRACE200/200/1000 56.339%、GCA56.363%、ProGCL54.242%、GraphECL62.885%。CoraFull两seed均值±样本SD：GRACE38.768±2.263、GCA41.423±2.975、ProGCL29.928±4.234、GraphECL53.557±.058%。GraphECL对GRACE两seed配对+16.348/+13.230pp；CiteSeer seed0 clean退化3.737pp，目标Brier及同类比例也变差。

决策：优先研究GraphECL图—属性分支，先同容量与CiteSeer诊断；ProGCL保留Cora机制正信号，固定权重统一底座暂缓；GCA保留增强对照。GraphECL Cora原生SUPPORT seed0/1为78.364/79.543%，高于R2；CiteSeer原生弱，CF accuracy接近而R2 Macro-F1更强。未加D1/C1、DIST、knn或KMeans。自动全配对门与成功单元解释分别保留。

报告：research/ssl_backbone_selection_20261026/RESULTS_successful_runs_zh.md；配对数据 successful_runs_summary.json；原始和审计 D:/codex_temp/ssl_backbone_screen_20261026/analysis/。原失败记录与自动决策保留，未新增训练。

## 2026-10-27 15:57：SSL空间两图补充已排队
用户授权先补Coauthor-CS/Amazon-Photo、seed0/1。独立目录research/ssl_backbone_csphoto_20261027与experiments/ssl_backbone_csphoto_20261027；运行根D:/codex_temp/ssl_backbone_csphoto_20261027。四方法16次SSL、172预测单元，当前32项CPU测试/132哈希/5仓库核验通过。队列PID58192（启动器65796）已确认WAITING_GPU，等待gcn_support_d1_c1_3seed_20261027进程树结束后自动8项冒烟→正式训练→审计；无需重复启动。当前正式训练尚未开始，后续以queue_status.json为准。未处理旧失败seed，未启动新算法、kNN/KMeans。入口research/ssl_backbone_csphoto_20261027/START_HERE.md，配置差异与假设详见PLAN_zh.md。

## SSL两图补充结果与判断修订（2026-10-27完成后复核）
16/16 SSL、172/172预测完整，失败0；独立重算Accuracy/Brier差0、Macro-F1差3.33e-16；保存的CPU重载最大差2.563e-6，预测与审计哈希一致。双seed R2+SUPPORT三噪声均值：CS GRACE90.588/GCA90.110/Pro90.288/GraphECL87.676；Photo81.923/84.690/83.862/85.801。GraphECL对GRACE CS−2.912pp、Photo+3.878pp，两图均值+.483pp，三候选均未过双图推进门。完整四图双seed GraphECL74.672、GCA71.613、GRACE70.120；去CF后的Cora/CS/Photo GE81.710与GCA81.676接近，优势集中度需报告。
新诊断：CS GE参与率26.87/26.98、第一主成分仅8.76/8.83%，局部同类对及Brier仍劣于GRACE，未出现Cite式集中。GE原生/R2带噪CS89.198/87.676、Photo89.153/85.801；CS clean CE原生91.171低于R2 92.936，而带噪CE原生88.206高于R2 84.805，提示迁移后噪声敏感性变化。Photo GE R2 CE低于GCA，SUPPORT增量4.603pp后反超。
决策：继续GraphECL机制研究，保留GRACE/GCA；优先32个交叉分类头（CS GRACE×GE、Photo GCA×GE；两个非对角组合×2seed×4条件）定位特征/目标作用。之后统一读出检查原生H、PCA64、R2特征，按结果决定粒球模块位置；Cite谱问题专项处理，暂缓统一防塌缩/边权模块。新实验仅设计，未启动，仍不补旧失败seed或新增kNN/KMeans。
报告：research/ssl_backbone_csphoto_20261027/RESULTS_AND_NEXT_STEPS_zh.md；复核脚本与数据同目录review_diagnostics/。

## 2026-10-27 22:08：32头特征×目标交叉实验已完成
用户授权A批，代码experiments/ssl_cross_sources_20261027；运行D:/codex_temp/ssl_cross_sources_20261027。32/32新SUPPORT头、4次工程对角复现、64只读旧对照，96预测完成审计。6测试通过，477来源哈希通过；对角模型哈希一致/预测差0；新模型CPU重载最大差1.878e-6；旧对照重算差3.33e-16。
CS学生/目标：GRACE/GRACE90.588，GRACE/GE90.171，GE/GRACE88.380，GE/GE87.676。GE换GRACE目标测试+.704pp，两seed+1.127/+.281，验证+.636且clean+.355，通过门。相同目标下GE学生仍低2.208–2.495pp，优先查特征路径。
Photo：GCA/GCA84.690，GCA/GE85.073，GE/GCA85.773，GE/GE85.801。GE在两软目标下均较强，自身目标与GCA目标带噪Accuracy差仅.028pp；自身目标仍有clean+1.242、Macro-F1+.602收益。GCA换GE目标两seed方向相反，未过门。削弱“专属GE粒球目标主导Photo优势”的解释，支持特征与软监督配合。
下一步建议原B批统一原生H/PCA64/R2特征的读出，固定同一目标；0次SSL、64新头。尚未启动B，也未启动新粒球模块或补旧失败seed。详细结论experiments/ssl_cross_sources_20261027/REPORT_zh.md，原始表/效应/门/audit位于运行根analysis/。

## 2026-10-27 22:47：B批统一读出完成，CS与Photo机制分化

本轮目标：固定GraphECL原生H及其SUPPORT32目标，对比原生、PCA64、冻结R2_unit64；查清前轮特征×目标交叉实验剩余的特征路径差距。读取文献地图、历史、A批报告、原probe/downstream/data与R2编码器。新增目录experiments/ssl_readout_control_20261027；运行根D:/codex_temp/ssl_readout_control_20261027。

研究假设/粒球角色：64维压缩、属性迁移与读出尺度可能分别影响下游；粒球作为固定LOO软监督来源。代码/配置：独立operations/run/analyze/launch及测试；复用原probe.fit、固定原始成员回调。2图×2seed×2新特征×4条件×CE/SUPPORT=64新头，原生与旧raw R2各32只读头；0次SSL/R2重训。PCA全节点无标签中心化eigh/no-whitening/64维后行L2；R2旧权重eval提取后行L2；同nn.Linear、Adam .01/wd .0005/200轮、首次最佳验证。

64/64新头、4/4原生复现、128评分成功，失败0。7测试通过；497来源哈希通过；原生复现模型/初始化/轮次一致、概率差0；64新头CPU重载最大差4.7684e-7；32组PCA/R2初始化一致。独立NumPy重算16组LOO目标最大差1.0353e-7；独立128评分最大差4.7522e-8；均值/样本SD差2.043e-14。命令：python -B experiments/ssl_readout_control_20261027/review/check_readout.py；完整入口START_HERE.md。

已支持结果，SUPPORT三噪声双seed均值±样本SD：CS native89.198±.501，PCA64 88.613±.511，R2_unit64 89.700±.527，旧R2 87.676±.134。R2读出改动对旧R2 +2.023pp，两seed+1.745/+2.301，验证+1.864；clean测试−.432、clean验证−.873pp。对原生测试+.501，但验证+.236且seed方向不同。PCA对原生−.586，主要Pair .4−2.065；削弱此前将CS差距主要归于不可逆迁移损失的解释。

Photo native89.153±.084，PCA64 89.872±.158，R2_unit64 86.078±1.184，旧R2 85.801±1.048。PCA对原生+.719（seed+.890/+.548，验证+.850且clean验证持平），对同64维R2 +3.794（验证+3.333），通过两项预登记门；CS两监督主门均未过，Photo CE主门通过但接近原生门未过。PCA保留中心化方差CS91.3–91.6%，Photo76.2–78.3%。Photo R2_unit的clean测试较旧R2下降3.436pp、带噪Macro-F1−.679，统一单位化方向受到负结果削弱。

新线索：Photo PCA对原生CE−1.102pp、SUPPORT+.719pp；SUPPORT相对CE增量从2.340变4.161pp，Pair .4提升1.541pp。所有路径目标完全相同/Brier差0，收益定位到表示与读出对同一监督的利用。合理解释为尺度影响梯度与有效正则、截断与软监督互补；单因素机制尚待验证。

最终决策：继续GraphECL原生属性分支与粒球软监督适配，修改CS不可逆迁移损失解释，暂缓全图统一单位化/盲目增维/复杂模块叠加。GraphECL embedding(x)来自属性MLP，可直接X-only推理。下一窗口优先讨论一次最小2×2读出确认（原始/单位化×旧/统一初始化，补两个缺格共64新头），之后进入单一弱粒球关系约束；此项尚未启动。原R2与R2_unit还改变初始化，PCA包含中心化/截断/行归一化，分别记录待验；其他SSL需匹配读出后再更新排名。

详细报告experiments/ssl_readout_control_20261027/REPORT_zh.md；results/保存分数、逐条件、配对差值、门、审计与来源收据；review/保存独立复核。研究问题、预期机制、成功/失败门及完成检查见PLAN.md；原始模型/概率/日志保存在D盘运行目录。后续新算法、旧失败seed、kNN/KMeans均未新增运行。

## 2026-10-27 23:18：C批尺度×初始化完成，单位化机制定位成立但完整门均未通过

本轮用户授权验证B批提出的2×2假设。读取文献地图、A/B历史、旧downstream/probe/AlignedMLP初值与训练实现。目标：固定R2特征与原GraphECL SUPPORT32目标，拆开单位化与分类头初始化。代码experiments/ssl_readout_factorial_20261027；运行D:/codex_temp/ssl_readout_factorial_20261027。

设计：两图CS/Photo、seed0/1、clean/U.4/U.6/Pair.4、CE/SUPPORT；补raw+unified、unit+legacy两缺格64新头，引用raw+legacy/unit+unified各32头。原始/单位化特征来自B批冻结R2；legacy按原AlignedMLP构造恢复末层，unified直接Linear(64,C)，初值文件/历史哈希全部核对。粒球保持原LOO/tie-safe/beta=.5/min_votes=2目标，0次SSL/R2训练。Adam .01/wd .0005/200轮/首次最佳验证；统一train_rows内核，针对旧全图矩阵舍入差预登记16次U.4对角复现门。

完成：64/64新头、16/16复现、128评分，失败0；7测试通过；791来源哈希、64组同初值尺度比较、目标和选模核验通过。复现200轮验证计数/最佳epoch全一致，概率最大差2.02656e-6；新头CPU重载最大差9.53674e-7。独立NumPy目标最大差1.03534e-7、128评分差4.75223e-8、均值SD差2.84217e-14、效应恒等式差1.42109e-14。复核命令python -B experiments/ssl_readout_factorial_20261027/review/check_factorial.py。

SUPPORT三噪声双seedmean±SD：CS raw旧87.676±.134、raw统一87.636±.125、unit旧89.782±.584、unit统一89.700±.527；Photo85.801±1.048、85.798±1.061、85.910±1.501、86.078±1.184。

固定初值单位化效应：CS SUPPORT旧+2.105pp、统一+2.064pp；验证+1.964/+1.800且两seed均正，clean测试−.514/−.584，clean验证−.846/−.900。只改初值：raw−.040、unit−.082；SUPPORT主效应分解单位化+2.084、初始化−.061、交互−.042，解释B批对角+2.023。CS CE单位化+3.664/+4.309，raw CE初始化−.948，初始化“小影响”描述限定到SUPPORT。

Photo固定初值单位化SUPPORT仅+.109/+.280，验证−.436/−.065；clean测试−3.427/−3.352，clean验证−3.595/−3.529。初始化主效应+.082远小于clean单位化损失。CS四噪声门通过，Photo四噪声门未过；8个含clean≥−.5pp约束的完整门全部未过。

已有日志诊断：16个unit clean头最佳轮次全部182–200，raw对应16个中9个≥180；unit末轮训练损失更高。已支持“单位化操作在固定预算下的条件性影响”；解释层面保留尺度/有效正则/收敛速度与径向类别信息两条可能，尚未拆分。若专门研究紧凑部署，可比较raw/unit/g×unit（g为无标签训练节点平均范数），该实验未执行。

最终决策：结束初始化方向扩展；原生GraphECL继续作底座，R2单位化作机制对照，暂缓全数据集默认替换。后续优先固定原生读出和目标，在属性分支测试单一弱粒球关系约束，再单独测试重新构球的目标效应。弱粒球公式与lambda待预登记，未开始新SSL。保留B批Photo PCA+SUPPORT正信号；clean门和CS/Photo分化必须进入后续设计。未补旧失败seed或新增kNN/KMeans。

报告experiments/ssl_readout_factorial_20261027/REPORT_zh.md；PLAN.md保留动机、机制、配置、成功/失败门与完成清单；results/保存完整表、逐条件、factorial_rows/factorial_summary、门、审计与收据；review/保存独立复核及clean_training_diagnostics.json；START_HERE.md为接续入口。

## 2026-10-28：用户调整目标为带噪优先，clean退出当前否决门

用户明确允许算法在不同条件下有优劣，近期全心关注含噪训练标签表现，方法成熟后再降低clean损失。后续执行：clean作为观察指标留档，暂不设clean退化门或专门修复网格；原split、噪声标签、训练预算和使用原干净验证标签的选模协议继续沿用。历史A/B/C冻结方案和判定保持原记录，追加新目标下的解释。

C批重新定位：CS带噪门4/4通过，单位化继续；Photo带噪门0/4，仍作分化对照。固定旧初值SUPPORT单位化测试效应：CS U.4+.651、U.6+1.991、Pair.4+3.674pp；Photo−1.083/+.411/+.999pp。较强噪声/Pair优先深入，三噪声全表保留。

建议一轮36新头：Cora/CiteSeer/CoraFull×seed0/1×U.4/U.6/Pair.4×CE/SUPPORT，仅单位化R2、固定旧初始化；原始头与SSL/R2只读复用。六套GraphECL R2权重及36旧带噪结果路径确认存在，来源D:/codex_temp/ssl_backbone_screen_20261026/downstream/。正式运行仍需哈希与复现检查。本轮未启动训练。

后续主线：GraphECL原生属性分支加单一弱粒球关系项，编码器/预算/目标/读出匹配；若新方法采用单位化，对照同步单位化，保留粒球增量归因。先固定目标测表示、再改构球测目标。暂缓仅为clean服务的延长训练/尺度扫描/融合模块。新筛选按带噪验证、两seed配对、跨图和逐噪声表现；+.5pp是试验参考值，clean不参与否决。

详细建议：research/noisy_priority_20261028/RECOMMENDATION_zh.md。下一窗口按这一明确用户偏好设计，避免再次以clean门阻断带噪正结果。


## 2026-10-28 10:53：带噪单位化三图扩展完成，CS/Photo复用

用户要求开始实验并节省token；CS/Photo此前已齐，实际只补Cora/CiteSeer/CoraFull 36个unit旧初值头。两seed、U.4/U.6/Pair.4、CE/SUPPORT；0次SSL/R2重训，无新clean，无kNN/KMeans。36/36新头、12/12原始头复现、84引用、120评分完整，失败0。3测试、1289来源哈希、目标/选模核验通过；CPU重载max3.2783e-7，独立评分差2.2204e-16、均值SD差2.0428e-14、配对差0。

SUPPORT双seed三噪声mean±SD：Cora raw71.653±.585→unit74.274±.734（+2.621，验证+1.111，过门）；CiteSeer63.724±.921→62.708±1.326（−1.017，验证−.904）；CoraFull53.557±.058→52.087±.255（−1.470，验证−2.299）；CS87.676±.134→89.782±.584（+2.105，过门）；Photo85.801±1.048→85.910±1.501（+.109，验证−.436，未过）。Cora/CS两seed正，Cite/CF两seed负，Photo方向不同。

五图SUPPORT测试宏均值72.483→72.952（+.470），验证73.629→73.517（−.113）；新增三图测试仅+.045、验证−.697。CoraFull CE准确率+1.115但Macro-F1−4.426；SUPPORT Macro-F1−6.781。先前高噪声/Pair更受益仅局部成立：Cite Pair−1.761，CF Pair准确率+.458而F1−4.637。

最终决策：尺度扩展结束，单位化保留强对照及Cora/CS正向证据，暂缓全图默认替换；回到原生GraphECL+单一弱粒球关系项，固定读出与目标再测表示/目标效应。clean继续无否决门，下一SSL未启动。报告experiments/ssl_noisy_norm_extension_20261028/REPORT_zh.md；config/PLAN/START_HERE同目录；results/为120项表和门；review/check_results.py及JSON；原始D:/codex_temp/ssl_noisy_norm_extension_20261028/。

## 2026-10-28：GraphECL附件核读与GB-ContextECL设计

用户要求吸收附件的三个粒计算方向与11.3正样本边可靠性，推进自研算法。本轮完成原文/锁定官方源码/历史实验核读和设计，尚无新代码或训练。正式设计research/graphecl_gb_context_20261028/DESIGN_zh.md，接续START_HERE.md。

推荐GraphECL原生属性分支+粒球条件上下文重加权：预算前50%原损失预热，一次无标签H构球（SSE二分、max256/minnodes32、索引传全部节点），固定成员；按节点归一化邻域区域直方图，同球其他节点的两个子集留一平均，与留一全局背景比较；两子集共同支持的边获连续加权，单条原边相对原权重限制[.75,1.25]，行总质量及自环保持。只改正向项，排斥项/原图保持；eta0精确回归。保留跨球有用上下文。度0/1不受本重加权干预，需报告覆盖。

五图seed0/1，U.4/U.6/Pair.4；ECL/GB/行内权重打乱三臂。先固定旧SUPPORT目标测表示，再新H重构目标；240逻辑头（旧60可审计复用，180新增），20新SSL干预分支，10底座逻辑轨迹条件复用。原生H读出，暂无R2/clean/KNN/KMeans对照/D1C1。带噪验证为主，clean无否决门；无性能承诺，具体门和诊断见设计。

核读发现原论文公式4与锁定作者实现的分母索引有差异，沿已有实验实现保留a_dst+b_src，先作eta0回归，Photo沿原双层projector/lam.1。旧GB-EC对Base带噪+.995两seed正但基础路线弱，支持把局部证据机制迁入成熟底座；旧路线按用户偏好继续暂缓。边噪声只是假设，现阶段干预的是带噪训练标签与原始图下的上下文价值。原论文理论条件另行处理。
## 2026-10-28：GB-ContextECL V2正式启动

用户授权综合Claude意见，以Codex判断为主设计并实验。接受Cos、离线预检、剂量与异配扩展；修正Cos范围为(1+cos)/2，eta=.25/1各配GB/Cos/Shuffle，加原ECL共7臂。相关/边偏移阈值仅作警示，结合真实参数梯度；真值诊断在拟合后。Roman无现成GraphECL官方配置，单列1024维/2层/100轮项目适配，官方split0/1与历史噪声函数。

假设：粒球条件邻域统计可改善原生属性表示及带噪局部共识；角色为留一双成员子集的区域连接偏好估计，仅重加权正向项。五图+Roman×seed0/1×7臂=84SSL端点（共享50%预热），CE/固定旧SUPPORT/重新构球SUPPORT共720新头；原五图底座和旧60头作逐轨迹复现。无R2/D1C1/clean新训练/KNN/KMeans对照。五主图为主要聚合，Roman单列；固定目标验证>=+.5pp、两seed正、>=3/5图正且优于同剂量Cos/Shuffle为推进信号，完整报告负结果。

实现：experiments/graphecl_gb_context_20261028/{context,runtime,run,analyze,launch,test_context}.py与config/PLAN/REVIEW_RESPONSE/START_HERE；旧源不改。9测试RED→GREEN，570来源哈希冻结，Cora原生/两剂量GPU冒烟和8短头、Roman输入与Photo配置检查通过。首次Adam共享step状态影响恢复测试，保留engineering_attempt1并用restore_snapshot深拷贝优化器状态修复；科学设计未改。

零更新预检用10个旧最终H：Spearman约.398–.633；eta.25偏移>10%的边约6.54–13.52%，eta1约38.24–53.33%；Cite有效节点58.55%，其余约82.09–96.24%。具体值以PREFLIGHT_REPORT_zh.md/terminal_summary.json为准。这些是权重诊断，无新分类性能结论。旧缓存没有50%checkpoint，真实中点由重放取得并另存formal/warmup/，正式记录梯度比例/方向。

运行根D:/codex_temp/graphecl_gb_context_20261028；启动器PID28744、实际worker23232（以status.json为准）；background_stdout/stderr、launch.json、OS锁完整。Cora seed0底座已与原模型哈希相同，H差0；6个原生头通过历史轨迹与预测核验，已继续干预臂。勿重复启动。全部预定拟合后才写formal/fits_locked.json并评分；最终analysis/REPORT_zh.md与formal/final.json。研究结果待完成后详细复盘、独立核读与更新上下文。

## 2026-10-28 15:00：GB-ContextECL运行中核验

已提交37/84端点及对应315/720分类头；Cora/CiteSeer两seed完成，CoraFull seed0完成、seed1继续。570冻结来源及已提交产物共2997次哈希检查通过，6项底座与36个旧分类头复现通过，当前队列失败0。本轮只追加只读review/check_progress.py及阶段报告，冻结训练代码/配置保持原版本。

真实中点新线索：eta1的GB附加梯度/原梯度，Cora2.433–2.714%、CiteSeer0.565–0.637%、CoraFull1.199–2.482%。CiteSeer Cos仅0.000766–0.000794%，GB梯度与原梯度余弦约−.741/−.701。同eta实际强度有差异；最终与保留同节点权重多重集的Shuffle、fixed/own目标及CE联合解释。当前无分类性能判断，维持冻结矩阵。详见experiments/graphecl_gb_context_20261028/review/PROGRESS_zh.md及progress_audit.json。


## 2026-10-28 15:43：用户要求降算力，切换预算版58端点/492头

用户强调减少计算且尽量保持精度与结论。实际采用减少扩展分支和底座缓存复用，所有保留任务维持原总轮数。五主图与两个seed/三噪声保持，核心ECL/GB100/Cos100/Shuffle100四臂覆盖五图；小剂量三臂仅保留前三图，CS/Photo小剂量与Roman延后。原84端点/720头修订58/492；39已完成端点复用，CoraFull seed1 GB100从525/1000恢复。

CS/Photo完整底座复用冻结旧权重/H/日志/RNG，单独重放前50%以生成共享中点。两图原新SSL更新15200轮改为7600轮，省50%；额外延后Roman800轮。按历史两seed耗时，两图训练部分约336→168分钟，开销/负载另计。头训练通常少于1秒，仍保留200轮。此次选择依据成本与无标签梯度，未读取本批分类成绩来决定分支。

实现仅新增experiments/graphecl_gb_context_20261028/budget/下调度、缓存、汇总、启动与测试文件，原冻结源不改；预算收据SHA256 87109947dd760c1d6ddf4400edc73b1f354b4f19207674ea52270529cc10419d。13测试通过，Cora中点模型/优化器/RNG/H/粒球与下一步更新精确一致；Photo缓存权重/H一致且前向重载通过。首次缓存冒烟的Cora配置存储格式与实际CS/Photo复用范围不同，改为Photo完成验证；失败证据留budget_revision/engineering_attempt1。

新启动器31260/worker29740，状态仍在运行根status.json，日志改为budget_revision/stdout.log与stderr.log；旧队列已停。恢复入口budget/launch_budget.py。已完成产物重新校验期间进度计数短暂从1递增。最终拟合锁按58/492，分析输出analysis_budget/；五图主门对GB100执行，小剂量前三图另列，异配证据延后。

当前只有工程核验与预算结论，分类效果待拟合完成。以后默认先报估计耗时、核心小矩阵优先，有信号后再补剂量/扩展，避免初筛铺满所有组合。详细预算与影响见budget/COST_zh.md，当前入口START_HERE.md。


## 2026-10-28 20:32：GB-ContextECL训练与最终核验完成，整体无增益

58/58 SSL端点、492/492分类头完成于19:22，训练失败0。原汇总在CPU重载argmax严格一致检查退出；CoraFull的两个最高概率差4.47e-8引起CPU排序变化，GPU逐位重载一致。另有Photo一项同类情况。保留冻结源和原预测，新增review/postfit_20261028独立修复：CPU误差阈值保持2e-5、排序变化满足实测舍入界、GPU492/492逐位一致；4项新测试通过，0次重训。最终CPU最大差2.622604e-6、独立目标差1.020274e-7；独立评分Accuracy/F1误差0，Brier/mean-SD/配对差误差至多2.220446e-16。status已FINISHED_BUDGET_REVISION，formal/final.json完整。

五图原生ECL+SUPPORT准确率74.018%；GB100固定旧目标73.994%（−.023pp），重新构球73.952%（−.065pp）。固定目标验证−.090pp，两seed宏−.166/−.015，正图2/5，对Cos−.108、对Shuffle+.072，主推进门未过。GB固定目标对Cos测试+.012、对Shuffle+.137pp。GB025前三图固定目标测试−.000887pp、验证+.045951pp，近零增量。

各图GB100 fixed/own对原ECL+SUPPORT测试增量：Cora+.035/−.141，CiteSeer−.079/−.072，CoraFull−.053/+.272，CS+.142/+.342，Photo−.162/−.728pp。CS own两seed+.366/+.317，验证+.327/+.491；目标Brier .472652→.463393，Macro-F1+.871pp。Photo own两seed−.996/−.461；Brier .482864→.491180，Macro-F1−1.353pp；同H重新构球额外−.566pp。CoraFull own两seed−.183/+.727，seed敏感。

离线边语义诊断：CS高低分边同类率约94.48%/50.29%，Photo96.33%/51.52%，两者均有语义排序信息；下游目标质量和性能仍分化。现有证据削弱单纯正向边权重即可普遍改善局部共识的假设。待验解释为几何组织与目标构造的适配差异。

最终决策：暂缓本版本扩展训练，不补更多seed/Roman/大图小剂量。保留CS正信号；后续优先用已有缓存定位CS与Photo目标变化，再决定单变量修改。五图强剂量与前三图小剂量分别报告。详细结果experiments/graphecl_gb_context_20261028/REPORT_zh.md，results/留完整表/审计/收据。新窗口先读报告，勿重新启动已经完成的队列。
