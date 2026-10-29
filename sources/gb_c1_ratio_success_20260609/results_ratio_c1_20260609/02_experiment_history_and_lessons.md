## 2026-10-29：两路线初筛完成，交叉拟合局部有益、原底座仍更强

用户授权后完成两条初筛：CS/Photo、seed0/1。S1对U.4/U.6/Pair.4做12反事实头＋12原头复现与缓存诊断；S2仅Uniform0.4、20轮短续训，20SSL＋20头，固定H0读出。完整计划、代码、报告在experiments/graphecl_honest_pilot_20261029；运行根D:/codex_temp/graphecl_honest_pilot_20261029。

ContextECL反事实H0＋新目标：CS+0.149±0.061pp、Photo−0.333±0.233pp；Photo票源变化94.97%、与Brier变化rho0.048，单纯变动大小解释弱。路线1GB额外AUROC+0.007148，KM−0.000769，GB的Brier预测MSE略增；未过+.01及连续指标门，未启动新增集合KL编码器训练，KMeans仅缓存对照。

交叉拟合测试real−等预算折内ECL：CS+0.043±0.039pp、Photo+0.411±0.158pp；real−打乱+0.023±0.011/+0.252±0.224pp；real−原全票−0.062±0.011/−0.439±0.013pp。验证宏real−b20+0.251pp（两seed+0.371/+0.131），real−打乱+0.084pp，real−原全票−0.185pp。Brier对b20：CS−0.009907、Photo+0.002723。两个推进门均失败，暂停本规格扩噪声/轮数/额外消融，clean未参与否决。

机制：真实软目标加权正对同类率CS83.43%、Photo85.90%，超过同球均匀78.94%/81.00%；实际H有变化。拆折仍约32票，但CS区域总节点156→329、Photo160→313，真实类别票差均下降。Photo纯ECL20轮全票也低原底座0.588pp，需同时分析续训与拆折代价。打乱改变有效正对数（CS25.39→5.05、Photo22.07→6.60），未来归因应匹配权重结构。Uniform.4下原SUPPORT相对CE已近零，较高噪声尚待独立验证。

9机制测试通过；20SSL/44新头/400SSL更新/8800分类头更新/428产物哈希独立核查，指标差≤2.22e-16；12原头概率逐位复现、20SSL和最终28头GPU重载通过；4格原独立转移噪声缓存重建一致。正式分支训练及保存共14.48分钟，构球/检查/诊断另计。前台进程一次无异常原因退出，7SSL/5头后由隐藏启动器接续，已完成产物复用；当前launcher COMPLETE、review passed，无待续训练。

已支持：局部正对更纯、相对折内对照小幅改善、净收益未过门。合理解释：区域扩大与续训漂移抵消收益。待验证：保持几何尺度/稳定续训是否改善、普通软监督对比与粒球额外贡献。削弱：当前KL定义与两折SUPPORT32短续训直接替换原底座。下一窗口先读RESULT_INTERPRETATION_zh.md及results，勿重启已完成队列；原稿与旧训练源未改。

## 2026-10-27 GCN+SUPPORT+D1+C1五图三seed完成

- 假设/角色/配置：局部LOO监督改进原GCN再用官方DIST迁移给X-only原MLP；五图seed0..2、clean/U.4/U.6、六教师/学生route共45格270逻辑学生，旧9格保留、新36格。GCN独立验证因果选支持和D1，无R2初始化。原配置、分区、噪声/划分和CE/OLD64/legacy32/deep32/fixedD1对照不变。
- 结果：带噪教师76.676±.697%、学生77.464±.449%、学生F170.862±1.401%；对CE教师+2.770±.370pp/学生+2.559±.315，对OLD64学生+.832±.222，对SUPPORT32+.551±.458，对固定D1+.174±.168。学生对自身教师+.788±.313。三seed对SUPPORT32宏差+.187/+.401/+1.065、对固定D1+.054/+.101/+.366。学生对SUPPORT32的30带噪格21胜1平8负，非逐格全优。
- 差异/clean：Cora/Cite/CF/CS/Photo学生77.590/66.123/58.573/93.908/91.127%，对CE+5.215/+4.646/+.294/+.634/+2.004pp。CF学生低固定D1 .108pp，CS对SUPPORT仅+.086，Photo对SUPPORT波动较大。clean全图对CE−.047±.304、对OLD64+2.016±.260；CF对SUPPORT clean+3.051±.209。整体支持监督迁移和clean修复，C1额外平均收益较小。
- 工程/审计：1620新教师轨迹、197新物理学生、173925成功更新；中断63额外更新另记录，恢复日志前138轮逐字节一致。初次audit辅助函数漏绑定，独立wrapper修复而保留冻结科学源码；29tests。16:21:59全audit passed：540metrics/2745prefix/2520snapshots/2270targets/720stats/453oldrefs/90CPUreload，最大概率差1.90735e-6、argmax同；270目标支持强度诊断。失败历史及hash完整。
- 决策：保留完整统一监督算法和GCN→DIST扩展，三seed表完成；不自动调参/扩种子/改稿/上传。强图方法对齐比较仍按同协议证据另行组织。入口research/gcn_support_d1_c1_3seed_20261027/RESULT_INTERPRETATION_zh.md；源experiments同名，重产物D同名runtime，complete=COMPLETE_AUDITED、无运行任务。以下中断与启动记录为历史。

## 2026-10-27 15:34 GCN扩展中断恢复

原训练/接续进程均无异常记录退出，25/45教师格已完成。恢复前核验763轨迹和全部冻结源码/前缀快照；唯一不一致是CF seed2 U.4 S128_L0.5日志到138但断点75。现场完整归档到D同名runtime/interruption_recovery/20261027T073427Z，新增独立恢复工具将活跃日志对齐75、归档残留锁，科学源码/配置/选择不变；63中断更新额外成本单列。28测试通过，15:34:41原launch恢复，driver66220/GPU55948；中断候选已完成200，原138轮与重跑日志逐字节一致，optimizer/RNG接续核验通过。当前继续授权矩阵，最终结果待完成，未读新test择参。

## 2026-10-27 GCN新监督五图三seed扩展已启动

- 假设/角色：检验SUPPORT+D1+C1对GCN教师及其原DIST学生的增量；粒球仅在冻结GRACE几何中提供LOO类别统计。扩展原三图seed0 pilot为五图seed0..2、clean/U.4/U.6共45格，旧9格只读复用、新36格。教师六路及270逻辑学生端点全部保留，不混用R2教师选择或初始化。
- 改动/配置：独立experiments/gcn_support_d1_c1_3seed_20261027，D盘同名runtime。沿用原GCN64两层/.5dropout/Adam.01/.0005/200轮与coalesced COO邻接，原随机属性MLP+官方全节点DIST；42候选25/75/200因果淘汰、S32anchor、CE回退和固定D1/旧64/legacy32控制不变。1620新教师轨迹包含早淘汰，最多216新物理学生（精确概率别名另计）。
- 预期/判读：对SUPPORT32与固定D1的配对增量、教师质量与学生迁移分别分析；三seed均值/样本SD、clean/F1/逐类与成本完整。验收为完整可追溯矩阵，不按测试追加增益门槛、调参或组合。旧pilot增量很小，负格应保留。
- 检查/执行：6合同、45输入预检、5条两轮GPU smoke及精确恢复/旧两轮桥接均通过。14:34:55隐藏启动，venv61784/实际GPU64372，当前Cora seed1 clean；其他Python库存为空时启动，未停止其他任务。源/输入hash冻结、旧9格manifest留存；独立CPU分析/audit后才写最终complete。
- 当前结果/决定：仅启动证据，正式三seed精度待完成；不推断稳定收益。继续已授权完整矩阵，无新增Pair/KM/kNN/SSL/稿件/上传。恢复入口HANDOFF/status/run.lock；先核查进程勿重复启动。14:47:43已有24/45教师选择（旧9/新15），当前CF seed2 clean，正式拟合无失败。独立分析/audit适配器交付，主会话25/25最终tests与旧Cora clean有界兼容性检查通过，源码单独hash冻结；两次中途后处理测试失败保留。单次接续driver60548等待拟合64372退出后幂等接analysis→audit，不重复开GPU任务。

## 2026-10-27 CS/Photo完整R2教师DIST完成，五图n5表补齐

- 假设/角色：补齐属性完整教师的预测迁移证据，GB仍仅供教师LOO监督。CS/Photo、clean/U.4/U.6、seed0..4、CE/FULL两路，60新原MLP64/BN/dropout.5学生，官方DIST T1/beta1/gamma1、Adam.01/.0005、200轮first最高干净验证准确率；所有教师/选参复用冻结缓存，12000更新。
- 结果：CS完整学生accuracy92.470±.197%，对CE学生+.676±.100pp，对完整教师+.728±.224；Photo84.842±.379%，对CE+2.084±.800，对教师+1.836±.663。带噪F1对CE分别+1.428±.508/+1.656±1.953pp；Photo波动保留。cleanaccuracy对CE CS−.019±.065、Photo+.373±.385。
- 五图合并：旧三图90学生+新60学生，共150学生/150教师端点，五seed表重算。完整学生带噪accuracy73.692±.488%、F166.563±.655%；对CE学生accuracy+2.071±.136/F1+2.024pp，对自身教师+1.170±.186/F1+1.403。原教师对CE优势2.266pp，学生优势2.071，差中差约−.195，主要保留教师质量收益。clean完整对CE学生accuracy+.124±.200、F1+.116，原Cora代价保留。
- 审计/工程：5契约tests/60教师CPU预检/4条2轮smoke与续训一致性passed。等待原SSL进程结束后串行GPU，未杀或修改其他任务；无正式训练失败。120新教师/学生CPU重载、300指标/288meanSD、额外492配对/60transfer检查passed，最大概率差1.43e-6且argmax同。当前COMPLETE_AUDITED、无运行/锁。源输入hash/日志/所有概率/模型/成本完整保留。
- 决策：保留统一完整R2→DIST扩展，CS平均增量小而Photo更大；机制解释待验证。同结构再蒸馏，教师学生均X-only；历史GRACE固定，五seed含split/noise/R2/分区/学生随机。新表只U.4/U.6和clean；无新Pair/KM/kNN/强图五seed。入口research/r2_teacher_dist_csphoto_20261027/RESULT_INTERPRETATION_zh.md，表在experiments同名tables_five_graphs，D重产物同名runtime。旧三图结果保留，未改稿件/上传/追加实验。

## 历史：2026-10-27 CS/Photo完整R2教师DIST预检与等待（已完成）

- 授权/假设：补齐Coauthor-CS/Amazon-Photo，检验完整教师监督优势及再蒸馏收益；GB仅作冻结教师的LOO类别证据。两图clean/U.4/U.6×seed0..4×CE/FULL教师，60新学生，各原MLP64/BN/dropout.5、随机seed=s、全参数Adam.01/.0005、官方DIST T1/beta1/gamma1、200轮first最高干净验证准确率。复用r2_integrated0..2与新main3/4教师，不重训教师或选参。
- 改动/产物：独立experiments/r2_teacher_dist_csphoto_20261027及D盘同名runtime；仅新适配/队列/分析/审计/测试。5契约测试通过，60教师CPU重载与输入/划分/选择/encoder/checkpoint/probabilityhash预检通过。科学范围固定、无后验收益门，保留clean/F1/负格和成本。
- 当前执行：其他SSL任务worker60708运行，单次后台队列45164（launcher49716）已启动，状态WAITING_GPU、训练尚未开始。标准库只读进程检查，保留其他任务，退出后串行smoke→60DISTfits→全freeze→测试→独立CPUaudit和五图合并表。无定时automation或循环实验。重复启动由queue.lock拦截。
- 结果/决定：目前仅预检证据，无新准确率或seed统计。完成后旧三图90学生与新60学生合并150端点、n5均值/样本SD/配对/每类/成本及TeX；旧结果保留，原算法不变，无稿件修改/上传。恢复入口HANDOFF和queue_status，禁止把WAITING写成已训练。

## 正式启动确认（2026-10-27）

最终版 29 项测试和 12 项 GPU 冒烟任务全部通过。run.py all 已在隐藏后台启动；已确认首个正式 GRACE/Cora/seed0 进入第 650/1000 轮，损失有限，当前 failures 为空。启动器 PID 41452，实际训练 worker PID 60708（以 formal/status.json 为准）。后续自动按冻结矩阵执行下游、A/B阶段，拟合全部结束或明确失败后统一评分与 CPU 审计。当前尚无正式精度结论。

输出：D:/codex_temp/ssl_backbone_screen_20261026；进度 formal/status.json，各任务 progress.json；完整日志 background_stdout.log / background_stderr.log；最终 analysis/REPORT_zh.md。无需重复启动；先核查单实例锁和状态。
## 2026-10-27 SSL 空间对照启动

用户已授权立即执行冻结方案。代码位于 experiments/ssl_backbone_screen_20261026；产物位于 D:/codex_temp/ssl_backbone_screen_20261026。29 项适配测试通过，145 个源/输入哈希通过。当前正在最终版三图四方法 seed0 两步冒烟，完成后按同一 receipt 启动 24 项正式 SSL、28 R2、224 R2 头、70 线性头（含 GraphECL native 别名，共 294 唯一预测单元）。

原延时启动 automation ssl 已暂停。训练入口 run.py all，状态 formal/status.json，日志 background_stdout.log/background_stderr.log；完成/失败记录 formal/fits_locked.json，评分审计 analysis/。正式结果尚未生成。

方法与超参保持 research/ssl_backbone_selection_20261026/protocol.json。工程适配及失败排查见新实验 README.md。GPU 确定性核已启用；旧工程尝试保留为 engineering_attempt1_nondeterministic / engineering_attempt2。测试标签在计划拟合完成或失败锁定后统一读取。两 seed、无 kNN/KMeans 扩展、无新调参。
# 当前工作目录实验历史入口

## 2026-10-26 完整R2主线与同结构教师DIST五种子表完成

- 假设/角色/动机：验证完整SUPPORT+D1+C1在R2上的跨seed增量及作为属性教师的知识传递；粒球仅在冻结教师的LOO类别证据中使用，DIST学生仅输入X与教师概率。
- 配置/改动：独立`experiments/r2_integrated_seeds34_20261026`补五图四条件seed3/4，固定历史GRACE、同100轮无标签R2对齐、原冻结头200轮、JOINT42→14→4/GLOBAL18→6→2、旧rank/anchor/CE回退及固定控制；40新格1840轨迹（含早淘汰）、159925分类更新、10新对齐。旧0..2来源r2_integrated只读。独立`r2_teacher_dist_five_seed_20261026`补三图clean/U.4/U.6 seed1..4的CE/FULL教师学生，72新拟合14400更新，保留18旧seed0；原MLP64/BN/dropout.5、全参数Adam.01/.0005、官方全节点DIST T1、200轮first最高干净验证准确率。
- 预期/判读：教师原优势传递与再蒸馏增量分别测量，五seed均值/样本SD，clean/F1/逐类与全部负格保留；本轮验收为协议与追溯完整，不按测试新增增益门槛或调参。
- 主线结果：三噪声五图seed内宏平均完整accuracy71.111±1.022%、F163.835±.962%。对CE+4.057±.290pp、GCE+.530±.328、SUPPORT+.945±.439、验证较优单模块+.234±.173；五seed宏差+.153/+.481/+.288/+.012/+.234，均正。逐图完整accuracy72.616/65.378/45.236/91.057/81.267%；Cite对较优单模块−.032，CS/Photo低GCE .152/.041pp。clean对CEaccuracy+.065、F1−.198pp。
- DIST结果：两噪声三图n5完整教师62.620±.822→学生63.716±.737%，CE教师59.977→学生61.184%。完整学生比CE学生+2.532±.272pp、F1+2.345±.486，五seed均正；对自身教师+1.096±.087，F1+1.096pp；差中差accuracy−.112±.566。逐图完整教师→学生Cora73.858→73.852，Cite67.006→67.595，CF46.997→49.701。普通CE教师再蒸馏也平均+1.208pp，完整监督的教师优势主要被保留。clean完整对CE学生accuracy+.088、F1+.105；Coraaccuracy−.644pp保留。
- 审计/工程：主线11契约+2launcher回归、6分析tests，KD7契约+续训/旧核桥；旧geometry与100epochR2两图精确复现。parent1000端点/100CPU重载、1840artifacts/2263prefix快照、3520因果分数/独立NumPy目标、3078统计均passed；KD90教师+90学生CPU重载/180指标/四套seedstats passed。概率最大误差主线1.43e-6/KD1.25e-6，类别相同。启动把同命令Windowsvenv父进程当other，在0拟合时失败，launch_v2仅修识别、旧失败/source保留；报告GBK打印失败后改ASCII摘要，原log保留，无重训。
- 归因/成本：旧uniform历史轨迹因果回放，新uniform真实prefix续训，选择规则相同；新共享prefix物理成本独立记录。原config upper1800少计每格一锚，实际1840，字段未限流且冻结保留。五seed随机覆盖split/noise/学生/R2/分区，SSL固定历史H；本轮无新KM/kNN或五seed强图比较。C盘保护，重产物全D同名新目录，旧文件无移动清理。
- 决策：保留完整主线与同结构蒸馏扩展；额外组合收益较小、部分图近乎持平，完整表支持有限稳健增量的叙述。新机制因果和粒球特异性仍待另行验证。当前全部完成，无自动更多训练/稿件/上传。报告在`research/r2_five_seed_tables_20261026/REPORT_zh.md`；主线CSV/compactTeX在`experiments/r2_integrated_seeds34_20261026/tables_n5`，DISTCSV/TeX在实验同名目录，D重产物由receipt定位。

## 2026-10-26 完整R2+SUPPORT+D1+C1教师再蒸馏（三图seed0完成）

- 假设/角色：完整属性模型能否作为统一教师保留局部监督优势并通过DIST再提升；粒球仅在原教师的LOO证据中使用，学生只接收全节点概率。
- 方法/配置：复用r2_integrated_20261026已冻结CE/ADAPTIVE_EXPANDED教师，Cora/CiteSeer/CoraFull×clean/U.4/U.6×seed0；新学生原MLP64/BN/dropout.5、seed0随机初始化、Adam.01/wd.0005、官方DIST T1/beta1/gamma1、200轮first严格最高验证准确率。18新拟合、3600更新，教师/R2/分区不重训，选参不变。
- 判读：预设关注完整学生比CE学生和比自身教师的配对增量，clean/F1/成本保留；混合负结果结束本轮，不自动扩展，未设测试后新门槛。
- 结果：带噪R2完整教师61.508→学生62.722%；CE教师59.017→学生60.185%。完整−CE教师+2.491pp、学生+2.538pp，6带噪格5胜1负，macro-F1学生+2.102pp。差中差accuracy+.047pp，原监督优势主要被保留。完整再蒸馏+1.214pp，macro-F1+1.459pp。
- 逐图：完整教师→学生Cora71.530→71.530、Cite66.194→66.817、CF46.799→49.820；CF普通CE再蒸馏也+3.237pp。对已有GCN完整DIST学生Cora−5.013、Cite+.816、CF−8.614pp。历史适配ERASE教师79.55/65.14/58.63对本轮71.53/66.82/49.82，部分数据接近的信号局限当前Cite探索格。
- clean：完整学生比CE学生accuracy三图−.422/+.988/−.505pp，平均+.020pp；macro-F1−.382/+1.879/−.786，平均+.237。相对自身教师Cora−.792、Cite+.086、CF+2.006pp。保留全部负格，单seedSD为空。
- 机制：CoraFull软预测再训练和学生全模型可训练可能改善原冻结头限制，普通CE教师同样受益，尚未分离容量/目标/选模。完整选参4格分组D1、5格统一力度，CE回退0；不声称每格必须选D1。
- 工程/审计：C盘预检不足保护储备，0拟合失败留痕；新产物D:/codex_temp/r2_teacher_dist_pilot_20261026，项目保留代码/receipt/薄报告，旧结果无移动删除。5tests、连续/断点smoke和旧GCN2轮精确桥接；18教师/学生GPU重载0差、独立CPU18+18重载及指标/选模/hash链passed。正式循环约57.633秒，历史预训练/search成本另列。
- 决策：保留统一完整教师扩展候选，现有GCN教师线分别报告；不按测试逐图挑路线，不自动更多seed/稿件/上传。详细PLAN/config/rawlog/checkpoint/probabilities/audit见experiments/r2_teacher_dist_pilot_20261026与research同名目录。

## 2026-10-26 AG-EC替换GRACE（三图两完整seed完成）

- 假设/角色/动机：用互补属性预测锚定局部关系，粒球作为跨球证据候选索引，受限注意力和集合对比训练适合标签估计的空间。上游读取X/A及train位置，无类别值。
- 改动/配置：独立`experiments/granular_evidence_ssl_20261026`；GRACE/Base-Attr/GB-EC三臂，Cora/CiteSeer/CoraFull，完整SSL/协议/学生seed0/1，clean/U.4/U.6/Pair.4。各200轮GNN、独立100轮R2、冻结encoder/BN、CE/SUPPORT32各200轮，共18SSL/18映射/144头。GB预热50、lambda.1、s32/C128/R4、EMA.99、刷新20、每轮512证据查询。上游树64/32，下游最新E1树4096/8；LOO/beta.5/minvotes2固定，无D1/C1搜索和新kNN/KM对照。
- 成功/失败门：对Base带噪>=+.5pp、至少2/3图正、clean每图>=−.5pp；退化则暂停本规格扩展。预期模块改善证据预测与终端噪声监督。
- 结果：带噪SUPPORT32 GRACE54.519±.024/Base52.249±.643/GB53.244±.465%；GB−Base+.995±.178pp（seed+.869/+1.121，三图均值正），GB−GRACE−1.274±.490。分图GRACE/Base/GB：Cora70.176/56.535/57.142，Cite64.046/56.443/58.126，CF29.334/43.770/44.465。CF为共同200轮，历史GRACE1000轮强参照留待后续。GB宏F145.249，比Base+.690、低GRACE1.008pp；Brier .708646低Base .712508、高GRACE .693201。
- clean/门：CoraGB−Base−2.216pp（两seed−.633/−3.799），Cite+.021、CF+.469，完整门失败。GB对GRACE U.6宏差−4.340pp。CE链路在Cora/Cite也明显弱于GRACE，主要差距已出现在新表示/R2组合的基础臂。
- 机制/代价：隐藏预测误差仅小幅改善，预测误差与干净证据Brier rho约.11–.16；Cite干净证据Brier .61847→.62918。零训练类间占比诊断显示Cora/Cite上游及投影前也偏弱。基础代理任务错配为合理解释，独立对比项/粒球特异性尚待消融。SSL秒数GRACE/Base/GB：6.63/2.39/15.95，10.58/4.04/21.85，206.58/32.11/116.85；GB显存均更高。
- 验证/工程：22实现测试、9短SSL/9短R2/72短头、3附加回归；64源/输入SHA固定。v2独立1322项全通过，18H max6.41e-7、144概率max1.79e-6且类别相同，原目标/选模/配对统计复算通过。v1的54误报来自floor划分全覆盖假设及缺arm的Brier缓存，旧审计保留，修复只在新v2；无正式训练失败/重训。冒烟和候选索引无损归档，有逐文件SHA。
- 决策：defer当前属性锚定规格，保留证据模块整体正信号，优先复查基础目标，不自动扩seed/叠D1C1。文献设计、完整解释、v1/v2审计及冻结几何在`research/granular_evidence_ssl_20261026`；正式配置/日志/概率/权重/均值样本SD在实验同名目录。全部完成，无后台拟合或待续队列。

## 2026-10-26 最新监督GCN→DIST（三图seed0首轮完成）

- 假设/角色/动机：将SUPPORT/D1/C1局部LOO监督用于同一GCN教师，再用不变DIST迁移到属性MLP，检验教师增益和学生保留量。区域只组织标签证据，R2未用作GCN/学生初始化。
- 改动/配置：新独立`experiments/teacher_integrated_dist_pilot_20261026`，复制已验证续训内核，GCN自己42→14→4选s/力度，25/75/200，deepS32强制锚、D1平票lambda1、CE回退；GCN/MLP64、Adam.01/wd.0005、200轮，原MLP再GCN的初始化顺序。三图seed0clean/U.4/U.6，固定历史GRACE，官方DIST T1全节点、无区域项/额外监督。
- 判读：按预定较宽松原则保留微小正信号及clean/F1代价，未设置新增硬成功阈值；不自动扩五图三seed。预期分组与支持量改善教师目标，迁移收益与教师收益分别评估。
- 结果：教师CE/OLD64/SUPPORT/fixedD1/joint均值61.568/64.380/66.065/66.108/66.125%；学生62.754/65.544/66.816/66.887/66.993。joint学生对SUPPORT+.1764pp、F1+.5921；对fixedD1+.1060/F1−.0358；对旧OLD64+1.4487，对CE+4.2387。逐图对SUPPORT−.1055/+.4940/+.1407。CFclean对SUPPORT+3.291（fixedD1也实现），对CE仅+.079且F1−.967；Cora clean对fixedD1−.739保留。
- 强参照：共同seed0两uniform，新学生/适配ERASE教师/ERASE-DIST为Cora76.54/79.55/80.00、Cite66.00/65.14/65.76、CF58.43/58.63/58.83。CF CE→DIST也约58.43，保留强简单对照。GCN原生整图与MLP实测2.137/.469、2.356/1.120、6.332/15.346ms，CF学生更慢；速度证据按图/原生实现给出。
- 成本/核查：405教师轨迹含早淘汰、48物理/54逻辑学生、33400+9600更新、5短smoke。joint2450含CE、fixedD11600、固定教师单200，全部搜索/控制成本公开。6契约+smoke通过，18历史教师概率桥0；worker504前缀状态和18GPU重载通过；parent9race/549prefix/18selector/108paired/384summary/108sklearn/18CPU重载通过，max1.37e-6、类别相同。训练后裸analyze被DIST依赖路径抢占，新增finalize_results.py显式本地绝对导入；原冻结源/失败保留，未重训/改阈值。
- 决策：保留为监督规则的GCN/DIST扩展和CFclean修复储备，新增组合的带噪额外收益小，先报告后决定确认。单seedSD=null、固定SSL、开发协议，不自动扩seed/图/模块，不改稿或上传。报告/配置/日志/概率/权重/前缀/哈希/成本/失败在对应experiments；独立复核与推理CSV在research同名目录。

## 2026-10-26 R2四模块集成（三seed完成）

- 假设/动机：表示迁移后的学生能进一步受益于支持量与同意/冲突监督力度的联合选择。粒球承担LOO共享标签证据区域。
- 代码/配置：独立`experiments/r2_integrated_20261026`，R2编码器/BN冻结，200轮Adam头；五图四条件seed0/1/2，6支持×7规则，42→14→4的25/75/200真实分组续训+历史统一强度因果前缀回放。平票力度固定1；固定/自适应D1一致。原全局18→6→2按原rank且精确桥接。beta.5历史开发值，test后读。
- 成功/失败判读：保留跨条件小幅正增量与成本，近零选择简单组件，负格与clean/F1完整报告；未设新增硬阈值。预期机制为支持量与分组监督分配互补。
- 结果：完整带噪71.5305%，对R2CE+3.901、GCE+.379、OLD64+1.591、legacySUPPORT+.916、固定D1+.488、C1+.579、验证选较优单模块+.307±.165pp；主增量三seed均正、五图均值皆正但CF仅+.048。macroF1对较优单模块+.474；clean对CEaccuracy+.135/F1+.107平均，CFcleanF1−.286±1.044保留。
- 代价与证据：1500新轨迹其中407完整200轮、123475新更新；4smoke。联合含CE2450更新，全局1200，固定D11600，预算影响尚未分离。worker审计通过；parent独立120race/5280prefix/300选择/660paired/1134summary/30CPUreload通过，maxerror1.37e-6、预测相同；旧OLD64重训60格与旧概率差0。agent两次额度中断但脚本正常完成，无训练失败。
- 决策：保留完整组合用于论文整理，停止本轮。无新KM/kNN/R0/SSL/teacher，历史KM强对照结论保留；固定历史SSL、已开发图/seed限定为整合验证。机制解释待预算匹配与新独立条件进一步核查，无自动扩展。报告与主侧复核在对应experiments/research目录，所有配置/日志/概率/权重/means/std/失败记录可追溯。

## 2026-10-23 R2正文对照（已完成）

完成更新：480个R2头和40外部结果齐全，汇总阶段同名analyze导入错误经绝对路径入口finalize_results.py修复，未重训。480checkpoint独立重载误差0、104额外源、五旧GSSC split/noise哈希验证通过。三seed五图带噪宏均值CE67.630/GCE71.151/OLD64 69.940/SUPPORT70.615/KM70.985%；SUPPORT−CE+2.985、−OLD64+.675、−GCE−.536、−KM−.370pp。KM逐图均更高。clean对CE Cora−1.953/CF−1.030pp，验证选最强固定竞争者后五图均负，门失败。相同seed0 SimMLP在4/5图高于SUPPORT，CS例外；GSSC本协议表现较弱。决策保留骨干扩展与带噪监督增量，暂停当前粒球变体扩展，未改稿件。详细表格见本目录REPORT_zh/summary/paired/matched_seed0/decision/delivery_audit。以下是启动时的历史记录。

用户授权将R2与图辅助MLP比较并补同骨干监督对照。新目录`experiments/r2_body_comparison_20261023/`，PLAN/config冻结。五图四条件三seed，CE/GCE/OLD64/SUPPORT/KM/POSTEL/SALS/ALS共480个R2头已完成；原R2/BN冻结，头200轮、Adam.01/wd.0005，首次严格最佳clean validation。粒球仅构造LOO标签目标，beta .5历史配置保留，KM同SUPPORT球数。

外部作者核心SimMLP/GSSC seed0四条件共40单元正在独立后台串行执行；SimMLP共享无标签encoder并统一seed0重训四个头，GSSC保留辅助阶段及200主训练。旧uniform.4 GSSC五格在核对协议后复用。预定拟合全部结束后才统一测试评分，结果和最终决策待REPORT_zh与decision.json生成。

4测试和15共享初始化组通过；Cora SUPPORT与旧模型概率最大差2.384e-7。历史GSSC配置复用路径故障留痕，external_v2只改为读取旧result.json内config字段。因大图单元约12分钟切换后台，当前未完CoraFull pair.4 GSSC部分轨迹保存interrupted后重跑；原结果/稿件/远端保持。后台入口background.py依次恢复训练、分析和480模型独立重载，状态见background_status.json。此条为运行记录，尚无新测试结论。

## 2026-10-23 节点路由与全局融合并列诊断

- 假设与动机：A300/C_head的自身概率可识别节点级优势，将并集空间转为净纠错。粒球为条件式局部统计候选，本阶段未使用。
- 代码/配置：独立`experiments/node_channel_routing_20261023/`，8特征、L2逻辑回归C1，外5内4嵌套验证，同折比较GLOBAL，协议和学生seed0五图四条件缓存；0神经拟合、620路由拟合槽位。全规则落盘后评分测试。
- 门槛：ROUTER−GLOBAL验证带噪前三图至少2图+.5pp，clean五图≥−1pp；失败停止，成功允许后续设计局部统计。
- 结果：验证+.864/+.904/+.185/−.200/+.044pp通过；测试−.229/−1.017/−.137/−.231/−.093pp，五图均负。相对最好单通道−.035/+.773/+.060/+.036/+.349pp。Cora救回51误伤53；Cite619/441；CF138/113；CS67/53；Photo129/73。以上各图顺序Cora/Cite/CF/CS/Photo。
- 机制记录：Cora带噪路由AUC .558，Photo .796，后者仍落后融合；排序能力、阈值和可救回基数需要分别分析。小验证分歧样本可能导致不稳定，但CF有391–440个拟合样本，仍只有微弱增量。候选解释保持为假设。
- 其他指标：Cora/Photo macroF1较GLOBAL +.935/+.616pp，其他三图负；clean测试差−.739/0/+.007/−.031/0pp。单seed标准差空，summary条件间SD另标。
- 决策：验证门保留通过，测试尚未体现优势；暂缓新增复杂度。局部图/GB特征和再蒸馏未运行。旧L1停止记录及稿件保持原样。
- 审计：5tests，620独立重拟合，20外折预测，140测试指标行，20历史全局权重，9源90输入哈希通过。唯一工程故障为末尾UTF8清单默认编码读取，已单行修复且还原摘要确认；failure.txt留痕，无预测重选。REPORT_zh/HANDOFF/final_audit/complete齐全。

## 2026-10-22 SUPPORT 教师替换蒸馏已完成

入口`experiments/teacher_support_distillation_20261022/REPORT_zh.md`。研究问题：用新SUPPORT分球算法替换旧LCSS教师后，教师输出能否让同一属性MLP蒸馏学生受益。粒球/分球只改变教师目标来源；本轮复用已训练教师概率，不重训GCN；不比较kNN、k-means或clean。训练split、噪声、学生初始化、MLP、优化器、训练轮数和干净验证选模保持配对一致，CE教师蒸馏为参照。

5数据集×3协议seed×Uniform .4/Uniform .6/Pair .4×CE/旧LCSS/SUPPORT-LCSS，共135个学生拟合，GPU全部完成。SUPPORT−旧LCSS准确率74.670%−74.249%=+.421±1.965 pp（45配对格，描述性SD），胜/平/负25/0/20；macro-F1差−.567 pp；教师准确率差+.405 pp。按噪声条件，学生准确率差为U.4 −.074、U.6 +.328、Pair.4 +1.009 pp；macro-F1分别−.911/−.556/−.234 pp。增益没有被蒸馏放大，且准确率增益集中于Pair噪声；Coauthor-CS/Amazon-Photo数据集均值为负。学生减教师的平均准确率差由CE −.294、旧LCSS −.278至SUPPORT −.262 pp，只有轻微缩小。

预设实用门槛：总体准确率至少+.5 pp、3/3 seed宏观配对均值为正、至少3/5数据集均值为正。实际+.421 pp、3/3、3/5，故完整门失败；没有clean臂，不能评估干净标签损害。135检查点重载、首次最佳验证轮次及测试指标独立复算通过。离线复核容差因CPU/GPU ULP级归约差异由1e-8调至1e-6，未改训练代码、配置或任何模型/预测产物，审计器版本及哈希修订在ENGINEERING.md和analysis_amendment.json留痕。

决策：保留本轮结果作为SUPPORT教师蒸馏证据，但不宣称通用或显著提升；不自动扩种子、不与其他模块拼接。后续若要训练新教师或扩大确认，需另行确定假设与门槛并授权。

## 2026-10-22 支持量与近邻身份三种子消融已完成（前次）

入口`experiments/neighbor_support_ablation_20261022/综合结论与接续.md`。用户授权后完整三图、
三协议/学生seed0/1/2、四条件、R0/R2、八臂576格（216旧锚360新拟合）及24短GPU冒烟完成，
无待续训练；固定历史GRACE及每seed既有R2，旧源码/稿件/远端未改。k32固定，不扫参数不拼损失。

同候选树拆TV为票数差异+成员差异；SUPPORT只取前项DP，COUNT按SSE截至NP球数，另有KMNP/KMOLD与匹配NP票数KNN。
R0带噪OLD44.580/NP45.931/SUPPORT45.935/COUNT46.010/KMNP46.533/KMOLD45.664/KNN46.851%；
R2对应59.306/60.525/60.509/60.426/61.143/60.366/60.843%。NP−SUPPORT为−.004±.080/+.016±.127pp，
NP−COUNT−.080/+.100，全部未达.5pp增量；不能声称正式等价或旧增益完全归因支持量。
KMNP−KMOLD+.869/+.778，均3/3seed宏差正、8/9条件均值正；NP对KMNP两骨干三seed宏差均负。

NP−旧版+1.351±.727/+1.220±.392，三seed宏差均正。R0修复门通过，但Cora U.4三seed全负、均值−.985pp；
R2clean对CE Cora−1.917/CF−1.131，CF对旧版仅+.984，完整修复门失败。CFclean R0准确率/F1：
CE47.275/37.253，旧版46.563/33.161，NP48.445/36.234，SUPPORT48.683/36.970，不能说小类已修复。

决策modify/defer：保留简单R0生成修复供返修，停止扩完整NP近邻保真代价；COUNT/KMNP球数仍来自NP，
不能当成已验证的独立廉价选球数算法。SUPPORT无需近邻但本轮未单独测端到端效率。32最优、独立SSL/新数据未证。
18测试、288目标、9几何、576重载/首次最佳选模/别名、288冻结R2、1333源哈希及32配对宏统计独立复算通过。
准备CE元数据兼容和审计float32归一化容差两次工程停止均保留；训练无失败，无重训。最终入口analysis_recovery_v2.py，
只改归一化数值检查，原冻结analyze.py未改，audit.json.passed=true。

## 2026-10-23 通道互补与同通道随机性对照

- 研究假设/动机：A300与C_head的互补是否超过同方法随机初始化；通过后检查简单融合及节点路由。粒球沿用固定SUPPORT32，局部统计是条件式候选。
- 代码/配置：新目录`experiments/channel_routing_variance_20261023/`，PLAN/config冻结；复用旧fit运行时重绑定新输出目录和固定protocol0；学生seed1，五图四条件两臂40新拟合+2smoke，40旧锚。A300 300轮、C_head固定R2/BN头200轮，旧目标/teacher/数据不动。
- 预期机制：学习路径差异带来独特正确集合，可观察规则实现净纠错。成功门为前三图至少2图跨并集高max同通道并集2pp；L1验证融合至少2图高单模型.5pp，clean每图不损超1pp；未达则停止后续层。
- 结果：第0层差Cora/CiteSeer/CF/CS/Photo为4.080915/2.720504/2.381812/1.109120/1.929661pp，前三图全过；四跨配对平均差4.133685/2.770619/2.364971/1.023403/1.643324pp。支持固定协议及编码器下的额外互补。
- L1：验证增量.123457/.903614/.134748/.272777/.740741pp，仅CiteSeer过主门，流程停止。测试GLOBAL比最好单通道+.1935/+1.7898/+.1973/+.2675/+.4420pp，CiteSeer直接平均65.178高C0 63.187；其余图收益小，教师置信阈值同样增量有限。
- 最终决策：保留互补和CiteSeer融合证据，停止当前阶梯；L2–L5/融合蒸馏未运行。节点路由的可实现性尚待直接检验；无需将全局融合失败推广到未测机制。解释：整体模型强弱及错误分布可能决定全局混合效果。
- 证据与检查：REPORT_zh/RESULT_TABLES/model_summary/各决策/原日志/权重概率；9测试、2exact桥接、40配对、80结果哈希与选模、40checkpoint独立重载、20融合重算、11源405输入5L1源哈希通过。无训练/分析失败。一个protocol、两个学生初始化，C_head只测固定encoder的头随机性；第0层显式test探索门、L1仅validation选择，原验证checkpoint复用写明。

## 2026-10-22 邻域保真粒球生成已完成（前次seed0）

入口`experiments/neighbor_preserving_ball_20261022/综合结论与接续.md`、目录交接、PLAN/gates/audit。假设：固定几何树上保留查询的训练近邻投票权重，可优于固定粗细粒度。球仅共享类别统计；新独立generator/run/analyze/tests，原算法/目标/稿件不动。H与train位置构树，无标签值；SSE增益优先、min总节点32/min训练3、最多256叶，32近邻TV代价，DP选父或子，最终20/22/142球。beta.5/LOO/平票、学生训练不变。

三图四条件，protocol/student seed0，固定历史GRACE和R2编码器；R0主、R2辅，8臂192格（144新、48旧锚），16 GPU冒烟，全部完成无任务运行。R0带噪CE35.515/OLD64 43.420/TREE64 44.593/FINE44.807/NP45.590/KM46.347/KNN32 45.960/KNNMATCH46.323%；R2对应56.602/57.921/58.267/58.628/59.461/60.304/59.883/59.947%。NP−旧版+2.169pp(8/9胜)/+1.540(7/9胜)，对KM两路线0/9胜，均值−.758/−.843pp。

R0clean对CE三图+2.533/+3.136/+1.754，CoraFull对旧版+2.642，完整修复门通过；对TREE64/FINE +.996/+.782且9/9胜，但对KM增量门失败。R2clean对CE−.264/−.473/−1.068，完整修复/增量均失败，不将1pp门事后放宽。CoraFull clean目标Brier .127173→.158005反而变差，R0 macro-F1 NP36.628仍低CE37.318；不是软目标/小类全面修复。

决策：保留R0返修确认候选，不自动扩seed，不拼GCE/多视图/软归属，不改投稿。新颖性未得到同球数KM支持；kNN两对照完整保留。单seed标准差为空，固定历史SSL，不称稳定确认。9测试、96目标、16smoke四精确桥、三图重建与96近邻排序、192重载/选模/96冻结R2、425哈希、32汇总168差值四门槛独立审计通过。两处测试/审计假设修复记录见ENGINEERING；无新训练失败。后续只在另行授权后确认同规格seed1/2，并保留KM。

## 2026-10-22 GCE与LCSS融合已完成（前次）

入口`experiments/gce_lcss_fusion_20261022/综合结论与接续.md`和目录交接。三图四条件、协议/学生seed0/1/2、R0主/R2辅、七监督，504最终有效拟合；14短GPU冒烟，R2六份新无标签encoder、三份旧seed0复用。历史GRACE与每seed历史分区固定，不是三个独立SSL；只新增独立目录，旧稿/实验源/结果/远端未改，无后台任务。

假设为预测一致性降权能否超出LCSS、GCE和等幅MASS；粒球只组织固定LOO目标。新增权重stop_gradient((t*p).sum^q)，q=.7、不归一化；W_MASS同规则使用自身MASS目标，非逐节点同权重。R0监督200轮，R2对齐100+冻结encoder/BN头100，20/10/70、仅train噪声、首次严格最大clean val，网格完再统一test。

R0带噪CE/GCE/LCSS/SOFT_GCE/W_LCSS/W_MASS/MASS=35.6995/34.6563/44.5796/41.0025/45.0674/40.6584/39.8595%。W_LCSS−LCSS+.4878±.4247pp，8/9均值正1平、3/3seed宏差正；相对GCE+10.4111、W_MASS+4.4091，但新增模块相对LCSS未达1pp门，完整失败。clean对CE三图+.5101/+1.7612/−.9142，有限安全门通过。CoraFull U.6+1.3016±.6003pp、3/3正保留为返修储备，不降门槛宣布通用成功。

R2对应57.7611/60.6031/59.3056/61.1091/59.5839/60.9805/59.9042%。W_LCSS−LCSS+.2783，−GCE−1.0191，−W_MASS−1.3966；clean对CE−1.6007/−.1002/−4.1790，完整失败。SOFT_GCE R0对LCSS−3.5771（九均值均负），R2+1.8035但Cora/CoraFull clean−3.2366/−6.1350。不按“错标目标argmax>70%”判定训练必败：R2有带噪反例，理想t^3.33是无参数约束的目标最优解，不等于共享学生泛化。真实标签仅用于离线诊断，不挑训练臂。

数值工程：准备阶段CPU/GPU舍入差回到精确旧CPU目标构造。v1 GCE的exp(log_p).pow(q)下溢梯度NaN；冻结旧源不改，新增稳定exp(q*log_p)及恢复入口，97旧成功+1失败GCE尝试完整保留并退出主表，最终GCE全部稳定实现；不混选版本。12测试、36输入协议、504重载/选模、48旧锚概率/轮次精确、252冻结encoder/BN、168summary/336配对/六门槛独立复算和161源+三恢复源SHA均passed。

最终：停止当前融合规格自动扩展，不扫q/权重归一化、不拼软归属。小幅R0增量可以如实保留返修储备，但不改投稿方法。软归属与自适应生成未运行，需要独立计划/授权；不把它们写成失败。

更新：2026-10-21。此前本工作目录没有同名入口；原始方向历史实际位于`../../experiment_history_and_lessons/`，其中有基础文件及编号2–44的记录（编号并不保证连续）。这些原始文件保留，不能用本索引替代逐方向的配置、seed、日志与负结果。

## 2026-10-21 边界诊断与已有预算粗化复盘（最新，0新训练）

入口`experiments/r2_boundary_diagnostic_20261021/综合结论与接续.md`。研究假设为固定分区遗漏近邻证据是否解释GB目标劣势；粒球仍仅使用上轮固定几何分区，不新增算法。新增诊断/测试/恢复脚本，原模型/结果/稿件不动；只执行用户批准的非训练前两项，不默认开其余训练。

三图四条件seed0，共20656训练节点—条件行；GB票数>=2为主要分层，CS661/665其余全体。球心相对距离、跨球训练近邻比例及标签依赖voteTV；误差为GB−kNN目标Brier，正值有利kNN。按all/原干净/原污染分层；几何四分箱跨噪声冻结，不拆同值；软化、严格错类、平票分开；无独立节点p值。预写描述性方向检查不改变上一轮失败门。

相对球心距离同时正相关/端组差clean2/3、带噪3/9，四箱单调1/12；不支持普适径向边界解释。跨球比例clean3/3、带噪7/9、单调4/12，仅描述性支持，弱相关和球内反向保留。CiteSeer U.4 GB目标Brier更好.006612，学生却低kNN .2148pp，目标指标不直接决定分类。当前GB-on-R2仍停止，不据边界相关救援。

独立重算预算粗化1080保存指标、232配对均值/std、264可用臂及24缺省臂，数值复现。GCN球训练C−匹配目标B+1.148pp，−同骨干LCSS+.565但仅6/15均值正；缓存MLP+.346/−1.218。局部高噪声GCN信号保留，当前R3评分−.033pp、干净切面O2+.281，效率联合门16clean条件0通过，重建冷成本C/A1.22–2.17。不是新独立重训/全预测审计，不改旧报告。

决策：停止径向边界万能解释、固定R2粒球目标默认推进、R3救援及当前完整效率主张；保留条件性球级训练组织线索。下一步必须另行授权，再做固定预算强分区与抗噪GCN控制，不自动组合/调参/多seed。11测试、237冻结源、99相关/394分箱复算passed；初次旧表缺省臂后处理错误及恢复记录保留，当前无待续任务。

## 2026-10-21 R2机制对照已完成

研究假设：强表示收益是否来自真实逐节点对应，LCSS类别方向及粒球分组是否仍有独立作用。粒球仅作固定局部目标组织；不改旧学生、分区、split/noise、评价或投稿稿件。新增`experiments/r2_mechanism_controls_20261021/`的PLAN/config/kernels/run/tests/analyze/audit与完整结论。

三图四条件seed0，REAL CE/GB/LS/GCE/BOOT/MASS/KNN/KM/RANDOM共108格；随机编码器、打乱H各CE/GB共48格，总156。REAL R2沿用旧100轮对齐，所有头冻结编码器/BN100轮；另三份100轮打乱表示对齐，随机无预训练。KNN逐节点匹配其他训练票数，KM匹配实际52/64/64区域，随机精确匹配总大小与训练支持；MASS保留GB逐节点观测类概率、其余类均分。

预设成功：相对通用正则/MASS/KM/KNN至少+.5pp、至少6/9正（详PLAN），随机至少+.5，clean任图不损害>1pp；失败保存全部结果，不自动扩seed。实际带噪宏均值CE56.602/GB57.921/GCE59.710/MASS58.049/KNN59.134/KM59.277%。GB−CE+1.320(8/9)，−GCE−1.789(1/9)，−MASS−.128(4/9)，−KNN−1.213(0/9)，−KM−1.355(1/9)，−随机+2.202(8/9)；clean−CE三图−1.478/−1.246/−2.360。GB完整门失败。

真实CE对随机/打乱+34.942/+39.298pp，均9/9正，表示门通过；无纯X自监督对照，不唯一归因于拓扑。决策停止当前固定GB目标在R2上的默认推进，保留R2骨干；不自动做c组/组合。GCE在CoraFull clean损害3.760pp，不能包装为通用替代方案。BOOT当前detach预测只缩放CE梯度，非强自纠正。

10测试、九短冒烟、172冻结源、48独立目标、156checkpoint重载、24旧R2精确锚、独立156精度/144配对差/96近邻查询均核验。正式网格56.474秒，不含旧SSL/R2，只有一个协议/学生seed，std为空；完整原始日志/哈希/配置/结果/决策见目录。当前无待续训练。

## 2026-10-21 表示迁移与鲁棒监督已完成

入口`experiments/representation_supervision_20261021/综合结论与接续.md`。三主图、四条件、协议seed0，144主矩阵+24旧目标+24H探针全新完成；4验证权重开发、3共享无标签预训练另计。原H/64预算历史分区固定，未启用自适应球数。14行为/原函数测试、12输入协议、52冻结输入哈希、192检查点重载及独立192指标复核通过，无剩余训练。

研究假设：表示迁移是否保留直方图监督未传递的信息，LCSS和small-loss在更强属性学生上是否仍互补。粒球仅组织固定LOO软目标。两阶段100轮对齐+100轮冻结编码器/BN分类优于联合200轮；带噪LCSS宏均值R0 43.420、R1 47.699、R2 57.921%，R2对自身CE56.602仅+1.320pp、8/9正。组合对逐格最佳单模块R0/R1/R2分别+3.936/+3.217/-4.062pp。R2 clean三图损害1.478/1.246/2.360pp，因此表示门通过，R1/R2完整LCSS增量门因clean cap失败。seed std为空，不称稳定/显著。

决策：继续研究两阶段表示训练，固定LCSS标为条件性信号，R2不默认叠small-loss。主要收益不能算成粒球贡献；冻结随机编码器/H行打乱、多seed、KM/kNN目标替换尚未验证。已投稿论文未改；新实验不得用本轮单seed替代确认。

## 阅读顺序

1. `01_literature_and_research_map.md`与`99_session_handoff.md`，理解当前问题及最新授权。
2. `research/algorithm_redesign_20261013/01_项目与旧实验复盘.md`建立投稿稿件和历史证据定位；具体旧方向回查上述原始历史目录。
3. 当前方向的PLAN、结果CSV、audit和完整结论；本文件只保存入口及关键教训。

## 已完成的本阶段实验

| 方向 | 范围 | 实际记录 | 决策 |
|---|---|---|---|
| 训练内独立证据剪枝 | 六图同GREET树，三协议seed三条件，六新臂；固定惩罚/无惩罚/全票与分折粗中细对照 | experiments/evidence_pruning_20261015已完成324新目标/结果、286新模型38别名，另216历史锚；14测试，54协议1944组件540预测1045输入源哈希审计通过 | defer当前准则；六图目标/分类门槛皆失败，Cora带噪−7.938pp、CS−2.166pp；Squirrel相对细球+2.229pp但不超旧一球且clean−.730pp；Cora过粗与分折损失并存，不扫delta、不默认替换，完整解释见综合结论与接续.md |
| 属性与结构残差互补 | 新Amazon-ratings/Tolokers+Roman/CS，五协议seed、三条件、七目标臂；二分类AUC | experiments/attribute_structure_20261015已完成420目标/300新学生；38测试，420目标/300预测/420配对/953文件哈希独立审计通过 | modify/defer；Amazon BFS比OLD64三条件+1.107/+1.019/+1.293pp均5/5正，但clean低于CE；Tol无稳定AUC增益，CS Pair X/BFS低于OLD64 2.442/5.955pp。四图R目标gate均失败，R学生未启动、不称分类失败；不扫权重，详见综合结论与接续.md |
| 六图嵌套粒球数量诊断 | 新增Actor/Chameleon-filtered/Squirrel-filtered；固定树与support16，8预算，暂不做独立KM/kNN | experiments/ball_count_diagnostic_20261015已完成；1026目标/学生记录、819独立模型、207别名；36分区54协议及81历史锚审计通过，35测试通过 | modify；Cora U同树64→终点40.10→42.44%，CS Pair86.45→84.64%；Squirrel带噪一球较好；拒绝统一sqrt/扩预算，提出未验证的训练内证据剪枝；详见综合结论与接续.md |
| 无标签BFS拓扑区域与原LCSS | 三图五协议seed三条件七臂；目标门槛后训练CoraFull/CS | experiments/topology_bfs_lcss_20261015；315目标、210新学生，3分区/15协议/全部logits审计，12历史分数核查一致 | 修改/暂缓纯BFS异配主线；Cora带噪优于CE但低于旧GB64，CS Pair相对旧版−6.03pp，Roman目标未超过匹配随机；两图简化判据均失败 |
| GREET自适应分球复验 | 同冻结两空间、三图三协议seed三条件；固定64与sqrt支持16，KM/kNN匹配 | experiments/greet_adaptive_ball_20261015已完成；16测试、297目标/学生，18分区27标签协议297logits审计通过，30历史分数一致 | modify；Cora相同新规则GREET带噪+1.67pp且6/6正，但比GREET kNN低1.87pp；Roman增量不足，CS Pair比原版低1.02pp；不统一替换默认 |
| GREET与HF-proxy独立首轮 | 三图、三协议seed、三条件；按门槛推进学生与融合 | experiments/greet_half_pilot_20261015已完成；54协议324目标243辅助模型，审计通过 | A/B门槛均失败，0学生；三图GREET旧规则均1球，带噪目标退化；不等于GREET整体或完整HALF无效 |
| 异配机制诊断 | 原两图、Roman两种表示及可靠性对照 | experiments/heterophily_mechanism_20261014；60单元840行，48seed0学生，20协议审计 | ESS×JS未过推进门槛；不重复无新信号的调温度 |
| 原GB64异配基线 | Texas/Wisconsin/Roman，五split，三条件，CE/GB64/KNN | experiments/heterophily_original_baseline_20261014；135学生，审计通过 | 带噪收益与干净损害并存；小图实际仅4–5球 |
| 自适应sqrt支持4 | 五图五协议目标及seed0分类 | experiments/adaptive_sqrt_ball_20261014；75目标单元600行、75学生，25协议+75学生审计 | Cora/Roman目标过探索筛查；小图学生退化，不能统一采用 |
| 固定支持16后续 | 五图仅seed0；只改最少训练支持 | experiments/adaptive_support16_ball_20261014；15目标单元120行、75学生，5协议+75学生审计 | 小图部分修复但未统一胜旧；Cora正信号，CS Pair负；停止阈值搜索 |
| PiNGDA与上下文构球空间 | Cora/CiteSeer/CoraFull，完整seed0/1，三条件，两分类器，三预算 | experiments/context_gb_two_seed_20261014；24SSL、324目标、684学生，严格审计通过 | defer；相对原GRACE带噪MLP−6.332pp、GCN−1.888pp；GCN对弱PiNGDA+.516pp仅探索信号 |

## 最新生成器方向的必备记录

- **研究假设**：旧固定最小子球和停止门槛限制细分；规模预算可能改善局部证据，但过细分球会减少抗噪投票支持。
- **粒球角色**：冻结表示上的局部软监督证据区域；不作为硬正样本或分类标签。
- **方法动机**：先隔离区域生成，不同时修改GRACE、LCSS与MLP。
- **代码改动**：新增`GB-LCSS_code/src/gblcss/granular_ball_adaptive.py`与两轮独立引擎/测试/审计；旧生成器保留。
- **配置改动**：请求ceil(sqrt(N))、最小球ceil(N/(4B))、总SSE优先及投影fallback；支持4首轮，支持16唯一后续固定值。只读训练集合位置，不读类别。
- **预期机制**：局部性改善与投票支持减少存在权衡，不能以球数达标代替可靠监督。
- **成功标准**：第一轮预设带噪配对Brier≤−.005且至少4/5改善，仅作探索信号；必须报告学生和匹配KM/KNN。第二轮仅seed0，五seed门槛不评定。
- **失败标准**：目标/分类退化、干净错误修正、无匹配对照优势均须保存；不从测试集逐图选择阈值。
- **实验结果**：见`experiments/adaptive_support16_ball_20261014/综合结论与接续.md`及两轮原始CSV。支持4数量14/16/151/141/136；支持16实际3/5/151/141/136。分类结论只单seed，不伪造std。
- **最终决策**：保留生成器研究版；继续确认CoraFull及证据质量；暂缓统一规则部署。不扫8/12/20、不叠已失败ESS×JS，不把sqrt经验写成最优理论。
- **验证与追溯**：source manifests、data/H/split/noise/targets hashes、逐球/分裂统计、raw logs、config、logits、CSV和audit已保存；第二轮30项历史准确率全部一致；34测试+20subtests通过。

## 必须继承的边界

同球身份不等于同类语义；更紧凑不保证更适合分类。增加票数可以降低人工翻转的条件随机波动，但无法消除干净几何中错误类别多数。目标Brier改善未必转化为MLP准确率。五协议目标seed不等于五个学生seed，更不等于五次独立SSL训练。普通kNN/k-means目前仍是必要强对照。

任何后续“继续、修改、暂缓、放弃”必须对照冻结计划及全部条件，不能只引用本文件摘要中的正信号。

## Context-GB 独立方向，2026-10-14已完成

详见`experiments/context_gb_two_seed_20261014/综合结论与接续.md`、同目录方向历史与`runs/summary.csv`、`runs/audit/report.json`。当前defer，不扩大seed或自动调参。用户本轮排除独立KM/KNN，不评定粒球必要性。每个seed有独立SSL，统一200轮，不混比旧CoraFull1000轮或其他分支单seed成绩。

已支持：新空间多数关系/Brier退化，C低秩；MLP相对原GRACE12/12带噪配对下降；GCN总体下降但CoraFull局部改善接近CE。合理解释：增强→最终原图空间有偏移，因果未证。尚未验证：原GRACE增强只加context及真实子图。被削弱：新增强或VICReg自动改善粒球语义的假设。本轮无联合CE，不以分类/对比冲突解释此次失败。

## 2026-10-23：自一致率选支持量、R2对齐预算与表示/预测组合

- **研究假设**：训练标签SC可以选择有用的局部支持；延长无标签对齐可能补足R2；R2初始化后蒸馏可能超过等蒸馏轮数及等总更新对照。
- **粒球角色/动机**：共享局部标签统计；E1在同一几何树中比较粒度，E2/E4固定历史SUPPORT32以隔离表示与预测通道。
- **代码/配置**：新目录 `experiments/sc_support_channel_pilot_20261023/`，冻结PLAN/config；新增训练/分析/测试。E1新树min_nodes8/min_train3/max4096，s4/8/16/32/64/128；R0旧64隐藏MLP200轮；E2一条600轨迹取100/300/600并头100；E4六臂、R2原100轮，普通GCN教师及全节点soft CE KD。旧源码/稿件未改。
- **预期机制**：SC的固定候选期望仿射关系支持投票质量估计；表示对齐及教师预测可能保留不同正确集合。后者需要收益超过知识重复与错误传递。
- **成功标准**：E1 uniform验证SC−VAL至少−.5pp、三图至少2图高固定32、clean每图不低CE1pp；E2验证平均提升至少1pp且两头平均均不损；E4整体选B/D后至少3/5图带噪验证高最强配对对照1pp且clean不损超1pp。失败停止扩展，不以测试重选。
- **实验结果**：E1离线360格/60选择，R0三图seed0 144拟合；SC−VAL验证−2.0585pp、仅CF对32+.0505pp、clean两图过损，门失败。投票排名rho U.4 .9377/U.6 .8857较高，但测试SC52.9133低固定3253.1804/历史3253.4698；目标估计不保证下游选择。
- **E2结果**：15历史日志，三条600轨迹+72头，三份100模型精确复现。三图300/600验证均不达门；Photo600测试+1.0714pp是事后信号。保留100，不推断不可约预测误差；分类头100轮固定可能限制后续拟合。
- **E4结果**：五图四条件seed0六臂120学生，另5clean教师；整体选B但无图验证增量达1pp。测试B−A300宏差−.0143pp。CiteSeer D+2.577pp但clean−1.246pp，D对C_head macro-F1−1.693pp，不能事后据此宣布组合成功。
- **可复用线索**：Cora D在教师错节点31.188% vs A300 11.884%，教师对节点88.154% vs95.919%，净下降；不同通道纠错与正确预测保留存在权衡。是否有不用真值的选择规则未验证，Photo错误侧也无增益，不能泛化为所有图均互补。
- **最终决策**：停止当前三种规格自动推进；无seed1/2分类扩展，E3/E5未启动。保留固定支持量历史收益及本次诊断，不自动调参/门控组合。分类只有seed0/std=null；离线三seed不充当分类确认。
- **验证/失败记录**：336分类+3对齐+5教师，另4冒烟；11测试、15源/115输入、336首次验证选模与预算/模型概率哈希、20初始化及A200/A300轨迹审计passed。两处后处理错误保留修复，训练未重跑。完整数据/均值/空std/日志/配置/概率/权重/哈希及阶段REPORT均在目录；入口 `REPORT_zh.md`、`final_audit.json`。
- **claim边界**：全局SC选s重新依赖自身标签，不能继承固定目标的留一独立性；pair不适用对称噪声标量系数；固定历史GRACE和beta开发选择不作独立确认。已支持、解释、待验假设、负方向在报告中分别保留。

## 2026-10-27 SSL 空间对照结果（成功单元分析）

用户要求暂不处理失败seed，本轮无重训。22/24主任务完成，267/294预测单元完成评分及CPU重载审计；最大概率误差2.4438e-6，独立重算Accuracy/Macro-F1/Brier差0。缺格来自CiteSeer seed1 GRACE/ProGCL导出检查及GRACE200关联下游。

固定R2+SUPPORT32、完整seed0三图三噪声宏均值：GRACE1000 55.584%、预登记GRACE200/200/1000 56.339%、GCA56.363%、ProGCL54.242%、GraphECL62.885%。CoraFull两seed均值±样本SD：GRACE38.768±2.263、GCA41.423±2.975、ProGCL29.928±4.234、GraphECL53.557±.058%。GraphECL对GRACE两seed配对+16.348/+13.230pp；CiteSeer seed0 clean退化3.737pp，目标Brier及同类比例也变差。

决策：优先研究GraphECL图—属性分支，先同容量与CiteSeer诊断；ProGCL保留Cora机制正信号，固定权重统一底座暂缓；GCA保留增强对照。GraphECL Cora原生SUPPORT seed0/1为78.364/79.543%，高于R2；CiteSeer原生弱，CF accuracy接近而R2 Macro-F1更强。未加D1/C1、DIST、knn或KMeans。自动全配对门与成功单元解释分别保留。

报告：research/ssl_backbone_selection_20261026/RESULTS_successful_runs_zh.md；配对数据 successful_runs_summary.json；原始和审计 D:/codex_temp/ssl_backbone_screen_20261026/analysis/。原失败记录与自动决策保留，未新增训练。

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
