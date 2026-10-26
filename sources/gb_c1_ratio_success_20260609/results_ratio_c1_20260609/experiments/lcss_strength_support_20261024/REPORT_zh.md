# GB-LCSS 支持量与监督强度首轮实验报告

## 先读结论

科学训练于北京时间2026-10-24 05:39:34结束，结果与本地补充审计于05:42:05完成。1500个完整科学拟合、20个CPU小预测器、5个两轮工程smoke均已完成；另保留一次122更新的受控中断轨迹。当前没有待续训练。

本轮最值得保留的是D1的简单分组力度，首选在R0上进行下一轮独立配对seed确认。C1的短训筛选可作为次优先级的选参工具继续确认；C2跨图预测和D2元门控暂缓。当前仅提出后续建议，未自动扩展。

下表为预定三种带噪条件的等权均值，单位pp。A、C1-R0、C2为五图15格；其余为机制三图9格。所有方法均按验证选择。

|阶段|关键结果|本轮决策|
|---|---|---|
|A|SimMLP上SUPPORT对CE +1.270；对GCE -0.754，对KM -0.186|保留监督增量，暂缓强骨干上粒球优于强控制的主张|
|B1|六个数据集/骨干组合的BALANCED均选lambda=1|暂缓统一全局收缩，保留完整剂量曲线与补充约束表|
|B2|N128对N32：R0 -3.087、R2 -2.137；方向与转移量两种128控制都退化|保留机制分解，暂缓普遍加大支持量；8的作用依赖骨干|
|C1|PROBE对固定新S32：R0 +0.564，R2 -0.037；对FULL_VAL：R0 +0.080，R2 -0.022|继续小规模确认短训筛选，R0收益较明确；R2暂无最终准确率增量|
|C2|四种预测器对PROBE均落后，测试差-3.642至-4.378|暂缓跨图小网络选参，不加层或扫超参|
|D1|分组对逐格最佳全局力度：R0 +1.123、R2 +0.475；对精确均量全局 +2.793/+0.667，对均量打乱 +2.215/+0.896|优先确认R0的分组节点选择，R2作为较弱信号保留|
|D2|门控对均量全局/打乱仅+0.020/+0.027；对同清洁meta预算CE/GCE为-6.453/-8.343|暂缓当前10参数元门控，保留完整负结果|

D1的验证增量对逐格最佳全局力度分别为R0 +1.437、R2 +0.551pp；对应带噪测试为8胜1负、7胜2负。对精确均量全局和打乱控制，R0均9胜0负。其macro-F1对逐格最佳全局分别提高1.248和0.330pp。

分组规则呈现清晰的条件性：六个clean格全部选(agree=1, conflict=.25)；18个带噪格全部选(agree=0或.25, conflict=1)。选法只读本格验证分数，噪声类型/真实噪声率未作为输入。该结果支持分别控制同意与冲突节点的研究方向。

D1的clean测试对CE，R0在Cora/CoraFull/Photo分别+2.586/+2.952/+.205pp；R2为0/+.051/+.149pp。相对逐格最佳全局力度，R0 clean分别+.158/-.375/+1.046pp，保留CoraFull的轻微代价。

节点分层提供了进一步线索：在九个带噪格等权平均后，R0同意组平均lambda=.083、冲突组为1；原干净节点平均实际转移量=.096，原错标节点=.295。原错标节点的真实类别质量平均增加.1845、Brier变化-.7721；原干净节点真实类质量减少.0960、Brier变化+.0424。R2出现相近分层。后处理真值只用于解释这些差异。

小类收益仍有代价：D1 R0对最佳全局的小类测试准确率平均+1.908pp，单格范围[-3.290,+12.861]；R2平均-.041pp。Photo R0的U.4与Pair.4 macro-F1分别比最佳全局低1.307和1.697pp。后续确认应完整保留这些负条件。

PROBE与FULL_VAL的测试绝对差在0.5/1/2pp内比例：R0为80%/95%/100%，R2为91.67%/100%/100%；最差测试损失分别.560和.528pp。每格候选预算1000次更新，完整网格为3600次；本轮实际执行了全部审计补训，实际成本另见costs.csv和c1_real_resources.csv。

审计通过：1500条完整曲线/检查点/概率，13个共享初始化组，34个独立重载样本概率误差0；72组独立连续/续训桥概率误差0；32个因果前缀筛选；24格D1均量及4800个D2逐epoch均量核验。11项最终单元/回归测试通过。458204条节点明细、2719条分层汇总、12000条力度轨迹已保存。

报告覆盖缺口：CS R0的四个历史SUPPORT32对照未单列在1500预算中，保持缺失。CiteSeer历史分区与新S32严格相同，已记录四个精确等价别名。全部1500个预注册完整拟合均已完成。seed0标准差留空，本轮结果用于确定下一步确认优先级。

## 执行与协议

完成时间：2026-10-23T21:41:05.563528+00:00。全部预注册阶段A/B1/B2/C1/C2/D1/D2完成；分类唯一拟合1500，CPU预测器20，工程冒烟5。
协议/学生seed均为0；GRACE、R2与SimMLP使用固定历史预训练。seed标准差留空。所有选择先冻结，再统一读取测试真值。
五图为Cora、CiteSeer、CoraFull、Coauthor-CS、Amazon-Photo；机制三图为Cora、CoraFull、Amazon-Photo。四条件全部保留。
R0：原始64隐藏MLP、dropout/BN正常训练；R2：历史100轮编码器及BN固定、线性头Adam；SimMLP：作者固定编码器/线性头AdamW。分类200轮、首次严格最佳验证检查点。
粒球用于共享留一类别统计，beta=.5、最少2票、并列不强化；历史SUPPORT32与深树S32分别存储。

## 假设与改动

A检查固定SimMLP上的监督增量。B1固定历史分区扫描全局强度与GCE，BALANCED为四条件共同参数、.2pp近优取较小值；旧约束为补充表。
B2在深树8/32/128的共同可识别集合上分解方向、实际转移量与交互。C1用18→6→2真实前缀筛选支持量/强度，CE可回退；C2测试跨整图泛化。
D1比较同意/冲突分组力度与精确sum(m*lambda)全局/打乱控制。D2使用validation内部meta-fit/selection，9输入线性sigmoid门控及精确可微一步虚拟SGD，真实头Adam、目标stopgrad。
源码只新增本实验目录；训练入口以明确白名单排除测试/干净训练真值。D2所有对照重跑同子协议，干净meta标签预算通过直接CE/GCE对照评估。

## 成功与失败判读

首轮预设探索参考：带噪验证约+.5pp或至少两预列单元+.5pp且代价有限；clean或macro-F1约1pp改善也可作为条件性信号。约±.3pp列为待判断。
对同协议简单控制多数退化，或收益主要依赖额外清洁标签且低于直接监督控制，则暂缓。clean下降>2pp、macro-F1下降>1pp见单独警示表。
本轮未扩种子、图、模块组合，也未修改稿件或远端。

## A 结果

| 骨干 | 方法 | 参照 | 带噪测试Δpp | 验证Δpp | F1Δpp | 胜/负/格 |
|---|---|---|---:|---:|---:|---|
|SIM|SUPPORT|CE|+1.270|+1.267|+0.984|12/3/15|
|SIM|SUPPORT|GCE|-0.754|-0.918|-1.482|4/11/15|
|SIM|SUPPORT|KM|-0.186|+0.158|-0.178|5/10/15|

## B1 结果

| 骨干 | 方法 | 参照 | 带噪测试Δpp | 验证Δpp | F1Δpp | 胜/负/格 |
|---|---|---|---:|---:|---:|---|
|R0|SUPPORT_BALANCED|CE|+10.155|+10.623|+8.991|9/0/9|
|R2|SUPPORT_BALANCED|CE|+3.729|+3.128|+2.972|9/0/9|
|R0|SUPPORT_BALANCED|SUPPORT_L1|+0.000|+0.000|+0.000|0/0/9|
|R2|SUPPORT_BALANCED|SUPPORT_L1|+0.000|+0.000|+0.000|0/0/9|
|R0|SUPPORT_BALANCED|KM_BALANCED|-0.365|+0.018|-0.638|4/5/9|
|R2|SUPPORT_BALANCED|KM_BALANCED|-0.350|-0.694|-0.644|1/8/9|
|R0|SUPPORT_BALANCED|MASS_BALANCED|+5.917|+5.568|+5.212|9/0/9|
|R2|SUPPORT_BALANCED|MASS_BALANCED|+1.254|+0.700|+1.051|7/2/9|
|R0|SUPPORT_BALANCED|GCE_BALANCED|+9.361|+9.719|+9.047|9/0/9|
|R2|SUPPORT_BALANCED|GCE_BALANCED|-0.389|-0.955|-1.179|3/6/9|

## B2 结果

| 骨干 | 方法 | 参照 | 带噪测试Δpp | 验证Δpp | F1Δpp | 胜/负/格 |
|---|---|---|---:|---:|---:|---|
|R0|N8|N32|+0.071|-0.734|-0.473|6/3/9|
|R2|N8|N32|-1.181|-0.510|-1.350|3/6/9|
|R0|N128|N32|-3.087|-3.721|-4.175|1/8/9|
|R2|N128|N32|-2.137|-1.694|-2.714|1/8/9|
|R0|D8|N32|-0.217|-0.612|-0.355|3/6/9|
|R2|D8|N32|-0.264|-0.196|+0.113|4/5/9|
|R0|M8|N32|+0.123|-0.015|-0.968|5/4/9|
|R2|M8|N32|-0.581|-0.383|-0.757|4/5/9|
|R0|DM8|N32|-0.644|-0.894|-1.136|5/4/9|
|R2|DM8|N32|-0.937|-0.466|-0.689|3/6/9|
|R0|D128|N32|-1.154|-1.857|-2.476|2/7/9|
|R2|D128|N32|-1.069|-0.943|-1.762|1/8/9|
|R0|M128|N32|-2.055|-2.143|-2.813|0/9/9|
|R2|M128|N32|-1.352|-1.079|-1.532|0/9/9|
|R0|DM128|N32|-3.076|-3.821|-3.853|1/8/9|
|R2|DM128|N32|-2.130|-1.698|-2.679|0/8/9|

## C1 结果

| 骨干 | 方法 | 参照 | 带噪测试Δpp | 验证Δpp | F1Δpp | 胜/负/格 |
|---|---|---|---:|---:|---:|---|
|R0|PROBE|FULL_VAL|+0.080|-0.040|+0.027|1/0/15|
|R2|PROBE|FULL_VAL|-0.022|-0.112|-0.086|0/1/9|
|R0|PROBE|CE|+9.817|+10.939|+9.921|15/0/15|
|R2|PROBE|CE|+3.639|+3.633|+2.916|9/0/9|
|R0|PROBE|FIXED_S32|+0.564|+0.717|+0.518|8/3/15|
|R2|PROBE|FIXED_S32|-0.037|+0.535|-0.002|2/2/9|
|R0|PROBE|LEGACY_SUPPORT32|+0.509|+0.630|+1.146|7/2/9|
|R2|PROBE|LEGACY_SUPPORT32|-0.090|+0.505|-0.056|3/3/9|
|R0|PROBE|SC|+0.803|+1.477|+0.703|9/2/15|
|R2|PROBE|SC|+0.847|+0.678|+0.679|5/1/9|
|R0|PROBE|GCE_VAL|+9.438|+9.649|+10.115|15/0/15|
|R2|PROBE|GCE_VAL|-0.887|-1.123|-0.654|2/7/9|

## C2 结果

| 骨干 | 方法 | 参照 | 带噪测试Δpp | 验证Δpp | F1Δpp | 胜/负/格 |
|---|---|---|---:|---:|---:|---|
|R0|STATS_RIDGE|PROBE|-4.378|-4.853|-4.186|0/12/15|
|R0|STATS_MLP|PROBE|-3.837|-4.247|-3.909|0/12/15|
|R0|ENHANCED_RIDGE|PROBE|-3.642|-3.914|-3.751|1/12/15|
|R0|ENHANCED_MLP|PROBE|-3.844|-4.015|-3.911|1/11/15|

## D1 结果

| 骨干 | 方法 | 参照 | 带噪测试Δpp | 验证Δpp | F1Δpp | 胜/负/格 |
|---|---|---|---:|---:|---:|---|
|R0|GROUP_SELECTED|CE|+11.278|+12.060|+10.239|9/0/9|
|R2|GROUP_SELECTED|CE|+4.017|+3.679|+3.068|9/0/9|
|R0|GROUP_SELECTED|CELL_GLOBAL_VAL|+1.123|+1.437|+1.248|8/1/9|
|R2|GROUP_SELECTED|CELL_GLOBAL_VAL|+0.475|+0.551|+0.330|7/2/9|
|R0|GROUP_SELECTED|EXACT_MASS_GLOBAL|+2.793|+2.779|+2.543|9/0/9|
|R2|GROUP_SELECTED|EXACT_MASS_GLOBAL|+0.667|+0.675|+0.372|7/2/9|
|R0|GROUP_SELECTED|MASS_MATCHED_SHUFFLE|+2.215|+2.599|+2.014|9/0/9|
|R2|GROUP_SELECTED|MASS_MATCHED_SHUFFLE|+0.896|+0.844|+0.759|8/1/9|

## D2 结果

| 骨干 | 方法 | 参照 | 带噪测试Δpp | 验证Δpp | F1Δpp | 胜/负/格 |
|---|---|---|---:|---:|---:|---|
|R2|GATE_LCSS|TRAIN_CE|+2.249|+2.209|+1.455|9/0/9|
|R2|GATE_LCSS|CLEAN_META_CE|-6.453|-7.154|-9.026|1/8/9|
|R2|GATE_LCSS|CLEAN_META_GCE|-8.343|-8.323|-11.060|0/9/9|
|R2|GATE_LCSS|GLOBAL_LCSS|+0.049|-0.063|-0.087|4/5/9|
|R2|GATE_LCSS|GATE_MASS|+0.296|+0.369|-0.495|7/2/9|
|R2|GATE_LCSS|GATE_KM|-0.479|-0.590|-1.270|4/5/9|
|R2|GATE_LCSS|MEAN_LCSS|+0.020|-0.052|-0.031|4/4/9|
|R2|GATE_LCSS|SHUFFLE_LCSS|+0.027|-0.052|-0.026|5/3/9|

## C1 成本与最差单格

每格PROBE为1000次候选更新加CE200次；完整网格为3600次候选更新。此次真实执行了完整网格，节省预算仅描述冻结策略的候选更新，实际成本包含审计补训。
B2完整200轮结果未用于C1筛选或候选复用。C2 STATS使用可观测统计；ENHANCED额外使用5/10/25轮验证信息，需18×25=450次候选更新/格。
历史SUPPORT32主表覆盖机制三图R0/R2；CiteSeer经精确分区/目标核对可复用新S32并列于补充表；CS R0未单列历史对照，保持缺失。

R0 PROBE−FULL_VAL：平均测试+0.097pp，验证-0.136pp；测试损失≤0.5/1/2pp比例分别为95.0%/100.0%/100.0%；最差amazonpho/clean -0.560pp。

R2 PROBE−FULL_VAL：平均测试-0.111pp，验证-0.196pp；测试损失≤0.5/1/2pp比例分别为91.7%/100.0%/100.0%；最差cora/clean -0.528pp。

## 负结果与代价

警示表共231条比较，详见clean_macro_f1_warnings.csv。下列为按clean准确率损害排序的前12条：

- B2/cora/R2 N128 对 N32：accuracy -4.433pp，F1 -6.933pp。
- B2/amazonpho/R0 N128 对 N32：accuracy -3.417pp，F1 -7.542pp。
- B1/amazonpho/R0 SUPPORT_BALANCED 对 GCE_BALANCED：accuracy -2.988pp，F1 -4.251pp。
- D2/cora/R2 GATE_LCSS 对 GATE_MASS：accuracy -2.902pp，F1 -3.702pp。
- C2/cora/R0 STATS_RIDGE 对 PROBE：accuracy -2.744pp，F1 -5.105pp。
- B2/cora/R2 DM128 对 N32：accuracy -2.744pp，F1 -5.451pp。
- C2/corafull/R0 STATS_RIDGE 对 PROBE：accuracy -2.627pp，F1 -1.679pp。
- C2/corafull/R0 STATS_MLP 对 PROBE：accuracy -2.627pp，F1 -1.679pp。
- C2/corafull/R0 ENHANCED_RIDGE 对 PROBE：accuracy -2.627pp，F1 -1.679pp。
- D2/cora/R2 GATE_LCSS 对 TRAIN_CE：accuracy -2.480pp，F1 -2.812pp。
- B2/cora/R0 N128 对 N32：accuracy -2.322pp，F1 -5.215pp。
- B1/amazonpho/R0 SUPPORT_BALANCED 对 CE：accuracy -2.316pp，F1 -3.539pp。

## 证据分类与最终决策

已被本次实验支持的结论：上表为完整seed0配对结果；选择、概率和检查点均可追溯。各方法的收益/代价应结合分图四条件表，优先用同协议控制判断。
合理解释：支持量、类别方向、目标幅度与学生表示可能共同影响泛化；B2交互及D1/D2精确均量控制为进一步区分提供线索。
尚未验证：独立seed重复、独立SSL、新图迁移及C/D组合。五图留一只提供五个迁移任务。
被负结果削弱的具体对比见各阶段负Δpp与警示表，继续保留完整结果。最终决策：本轮结束；有条件性信号的最简规格可建议小规模配对seed确认，当前不自动续跑。

## 追溯与工程记录

分类拟合累计墙钟（逐fit求和）2654.29秒；D2虚拟更新累计19.49秒；初始化/分区输入读取及后处理另计，环境事件时间见events.jsonl。
sources.json/inputs.json保存源与输入SHA256；每臂target、原始日志、优化器/RNG恢复状态、checkpoint和全节点probabilities均保留。选择冻结时间/哈希与完整网格时序见selections及selections_frozen.json。
测试最初缺少实现为预期红灯；一处测试float32/float64断言类型修复保留在testing记录。任何运行故障见failures.jsonl；历史数值桥见evidence/*/*/bridge.json和historical_probability_bridges.csv。
独立本地audit见audit.json；主会话另做只读抽样复核。


## 分阶段解释与行动决策

### A

- SIM SUPPORT 对 CE：带噪accuracy +1.270pp，验证 +1.267pp，macro-F1 +0.984pp；12胜/3负/15格。
- SIM SUPPORT 对 GCE：带噪accuracy -0.754pp，验证 -0.918pp，macro-F1 -1.482pp；4胜/11负/15格。
- SIM SUPPORT 对 KM：带噪accuracy -0.186pp，验证 +0.158pp，macro-F1 -0.178pp；5胜/10负/15格。

本轮具体决策（SIM）：修改：保留对CE的局部监督增量，暂缓对强控制的优势主张。

判读：固定同一个SimMLP表示之后，SUPPORT对CE衡量局部监督增量，对GCE衡量通用抗噪替代，对KM衡量分区形式的额外价值。三者分别判断，强表示本身的收益沿用固定编码器来源。
决策：保留本轮完整证据；若SUPPORT对GCE/KM缺少优势，暂缓“强骨干上粒球独立增量”的主张。

### B1

- R0 SUPPORT_BALANCED 对 CE：带噪accuracy +10.155pp，验证 +10.623pp，macro-F1 +8.991pp；9胜/0负/9格。
- R2 SUPPORT_BALANCED 对 CE：带噪accuracy +3.729pp，验证 +3.128pp，macro-F1 +2.972pp；9胜/0负/9格。
- R0 SUPPORT_BALANCED 对 SUPPORT_L1：带噪accuracy +0.000pp，验证 +0.000pp，macro-F1 +0.000pp；0胜/0负/9格。
- R2 SUPPORT_BALANCED 对 SUPPORT_L1：带噪accuracy +0.000pp，验证 +0.000pp，macro-F1 +0.000pp；0胜/0负/9格。
- R0 SUPPORT_BALANCED 对 KM_BALANCED：带噪accuracy -0.365pp，验证 +0.018pp，macro-F1 -0.638pp；4胜/5负/9格。
- R2 SUPPORT_BALANCED 对 KM_BALANCED：带噪accuracy -0.350pp，验证 -0.694pp，macro-F1 -0.644pp；1胜/8负/9格。
- R0 SUPPORT_BALANCED 对 MASS_BALANCED：带噪accuracy +5.917pp，验证 +5.568pp，macro-F1 +5.212pp；9胜/0负/9格。
- R2 SUPPORT_BALANCED 对 MASS_BALANCED：带噪accuracy +1.254pp，验证 +0.700pp，macro-F1 +1.051pp；7胜/2负/9格。
- R0 SUPPORT_BALANCED 对 GCE_BALANCED：带噪accuracy +9.361pp，验证 +9.719pp，macro-F1 +9.047pp；9胜/0负/9格。
- R2 SUPPORT_BALANCED 对 GCE_BALANCED：带噪accuracy -0.389pp，验证 -0.955pp，macro-F1 -1.179pp；3胜/6负/9格。

本轮具体决策（R0）：暂缓统一收缩：保留剂量曲线及条件性代价，原lambda=1继续作为简单锚。

本轮具体决策（R2）：暂缓统一收缩：保留剂量曲线及条件性代价，原lambda=1继续作为简单锚。
cora/R0：BALANCED lambda=1.0，CONSTRAINED lambda=1.0。
cora/R2：BALANCED lambda=1.0，CONSTRAINED lambda=1.0。
corafull/R0：BALANCED lambda=1.0，CONSTRAINED lambda=1.0。
corafull/R2：BALANCED lambda=1.0，CONSTRAINED lambda=1.0。
amazonpho/R0：BALANCED lambda=1.0，CONSTRAINED lambda=0.5。
amazonpho/R2：BALANCED lambda=1.0，CONSTRAINED lambda=0.5。

判读：BALANCED使用四条件共同参数，代价完整保留。收缩若仅改善clean而显著损失带噪收益，应作为使用条件权衡；KM/MASS/GCE同候选预算结果用于识别通用收缩收益。
决策：优先保留与原强度接近且简单的规格；只对有清晰验证增量并超出同协议控制的局部规格建议配对seed重复。

### B2

- R0 N8 对 N32：带噪accuracy +0.071pp，验证 -0.734pp，macro-F1 -0.473pp；6胜/3负/9格。
- R2 N8 对 N32：带噪accuracy -1.181pp，验证 -0.510pp，macro-F1 -1.350pp；3胜/6负/9格。
- R0 N128 对 N32：带噪accuracy -3.087pp，验证 -3.721pp，macro-F1 -4.175pp；1胜/8负/9格。
- R2 N128 对 N32：带噪accuracy -2.137pp，验证 -1.694pp，macro-F1 -2.714pp；1胜/8负/9格。
- R0 D8 对 N32：带噪accuracy -0.217pp，验证 -0.612pp，macro-F1 -0.355pp；3胜/6负/9格。
- R2 D8 对 N32：带噪accuracy -0.264pp，验证 -0.196pp，macro-F1 +0.113pp；4胜/5负/9格。
- R0 M8 对 N32：带噪accuracy +0.123pp，验证 -0.015pp，macro-F1 -0.968pp；5胜/4负/9格。
- R2 M8 对 N32：带噪accuracy -0.581pp，验证 -0.383pp，macro-F1 -0.757pp；4胜/5负/9格。
- R0 D128 对 N32：带噪accuracy -1.154pp，验证 -1.857pp，macro-F1 -2.476pp；2胜/7负/9格。
- R2 D128 对 N32：带噪accuracy -1.069pp，验证 -0.943pp，macro-F1 -1.762pp；1胜/8负/9格。
- R0 M128 对 N32：带噪accuracy -2.055pp，验证 -2.143pp，macro-F1 -2.813pp；0胜/9负/9格。
- R2 M128 对 N32：带噪accuracy -1.352pp，验证 -1.079pp，macro-F1 -1.532pp；0胜/9负/9格。

本轮具体决策（R0）：继续机制记录：方向控制平均验证变化-1.234pp，转移量控制-1.079pp；当前不扩展尺度搜索。

本轮具体决策（R2）：继续机制记录：方向控制平均验证变化-0.569pp，转移量控制-0.731pp；当前不扩展尺度搜索。

判读：D8/D128隔离类别方向变化，M8/M128隔离实际转移量变化；DM与原生N8/N128分别列出，共同集合外沿用N32。交互表为描述性差分，不能以节点数替代独立seed。
决策：继续保留分解作为机制证据；后续候选优先依据验证方向一致性和原生实际收益，当前不扩大尺度网格。

### C1

- R0 PROBE 对 FULL_VAL：带噪accuracy +0.080pp，验证 -0.040pp，macro-F1 +0.027pp；1胜/0负/15格。
- R2 PROBE 对 FULL_VAL：带噪accuracy -0.022pp，验证 -0.112pp，macro-F1 -0.086pp；0胜/1负/9格。
- R0 PROBE 对 CE：带噪accuracy +9.817pp，验证 +10.939pp，macro-F1 +9.921pp；15胜/0负/15格。
- R2 PROBE 对 CE：带噪accuracy +3.639pp，验证 +3.633pp，macro-F1 +2.916pp；9胜/0负/9格。
- R0 PROBE 对 FIXED_S32：带噪accuracy +0.564pp，验证 +0.717pp，macro-F1 +0.518pp；8胜/3负/15格。
- R2 PROBE 对 FIXED_S32：带噪accuracy -0.037pp，验证 +0.535pp，macro-F1 -0.002pp；2胜/2负/9格。
- R0 PROBE 对 LEGACY_SUPPORT32：带噪accuracy +0.509pp，验证 +0.630pp，macro-F1 +1.146pp；7胜/2负/9格。
- R2 PROBE 对 LEGACY_SUPPORT32：带噪accuracy -0.090pp，验证 +0.505pp，macro-F1 -0.056pp；3胜/3负/9格。
- R0 PROBE 对 SC：带噪accuracy +0.803pp，验证 +1.477pp，macro-F1 +0.703pp；9胜/2负/15格。
- R2 PROBE 对 SC：带噪accuracy +0.847pp，验证 +0.678pp，macro-F1 +0.679pp；5胜/1负/9格。
- R0 PROBE 对 GCE_VAL：带噪accuracy +9.438pp，验证 +9.649pp，macro-F1 +10.115pp；15胜/0负/15格。
- R2 PROBE 对 GCE_VAL：带噪accuracy -0.887pp，验证 -1.123pp，macro-F1 -0.654pp；2胜/7负/9格。

本轮具体决策（R0）：继续：PROBE对固定S32的验证增量值得小范围重复，同时保留最差单格损失。

本轮具体决策（R2）：继续：PROBE对固定S32的验证增量值得小范围重复，同时保留最差单格损失。
R0 距FULL_VAL测试绝对差≤0.5/1/2pp比例：80.00%/95.00%/100.00%；单格测试差范围[-0.560,+1.203]pp。
R2 距FULL_VAL测试绝对差≤0.5/1/2pp比例：91.67%/100.00%/100.00%；单格测试差范围[-0.528,+0.000]pp。

判读：前缀策略先冻结、全网格后补训，适合衡量有限预算筛选损失。FULL_VAL自身按验证选择，测试中PROBE偶尔更高也完整保留。
决策：若最差单格损失较大，保留FULL_VAL作为研究参照，暂缓把PROBE当作统一替代；候选更新预算与完整审计实际成本分别披露。

### C2

- R0 STATS_RIDGE 对 PROBE：带噪accuracy -4.378pp，验证 -4.853pp，macro-F1 -4.186pp；0胜/12负/15格。
- R0 STATS_MLP 对 PROBE：带噪accuracy -3.837pp，验证 -4.247pp，macro-F1 -3.909pp；0胜/12负/15格。
- R0 ENHANCED_RIDGE 对 PROBE：带噪accuracy -3.642pp，验证 -3.914pp，macro-F1 -3.751pp；1胜/12负/15格。
- R0 ENHANCED_MLP 对 PROBE：带噪accuracy -3.844pp，验证 -4.015pp，macro-F1 -3.911pp；1胜/11负/15格。

本轮具体决策（R0）：暂缓加深MLP：当前保留Ridge、MLP和PROBE的完整比较，优先简单方法。
STATS_RIDGE：20格CE回退8，候选排名Spearman平均0.448。
STATS_MLP：20格CE回退8，候选排名Spearman平均0.663。
ENHANCED_RIDGE：20格CE回退7，候选排名Spearman平均0.425。
ENHANCED_MLP：20格CE回退5，候选排名Spearman平均0.383。

判读：整个图留出，scaler与回归目标统计均由四个拟合图估计；ENHANCED额外消耗短训练信息。五折只有五个图级迁移任务。
决策：MLP若未超过Ridge/PROBE，保留较简单方法，暂缓网络扩容与新增超参搜索。

### D1

- R0 GROUP_SELECTED 对 CE：带噪accuracy +11.278pp，验证 +12.060pp，macro-F1 +10.239pp；9胜/0负/9格。
- R2 GROUP_SELECTED 对 CE：带噪accuracy +4.017pp，验证 +3.679pp，macro-F1 +3.068pp；9胜/0负/9格。
- R0 GROUP_SELECTED 对 CELL_GLOBAL_VAL：带噪accuracy +1.123pp，验证 +1.437pp，macro-F1 +1.248pp；8胜/1负/9格。
- R2 GROUP_SELECTED 对 CELL_GLOBAL_VAL：带噪accuracy +0.475pp，验证 +0.551pp，macro-F1 +0.330pp；7胜/2负/9格。
- R0 GROUP_SELECTED 对 EXACT_MASS_GLOBAL：带噪accuracy +2.793pp，验证 +2.779pp，macro-F1 +2.543pp；9胜/0负/9格。
- R2 GROUP_SELECTED 对 EXACT_MASS_GLOBAL：带噪accuracy +0.667pp，验证 +0.675pp，macro-F1 +0.372pp；7胜/2负/9格。
- R0 GROUP_SELECTED 对 MASS_MATCHED_SHUFFLE：带噪accuracy +2.215pp，验证 +2.599pp，macro-F1 +2.014pp；9胜/0负/9格。
- R2 GROUP_SELECTED 对 MASS_MATCHED_SHUFFLE：带噪accuracy +0.896pp，验证 +0.844pp，macro-F1 +0.759pp；8胜/1负/9格。

本轮具体决策（R0）：继续：分组分配相对最佳全局及两种均量控制的验证信号值得小规模确认。

本轮具体决策（R2）：继续：分组分配相对最佳全局及两种均量控制的验证信号值得小规模确认。

24格所选规则分布：{"GROUP_A1.0_C0.25": 6, "GROUP_A0.0_C1.0": 13, "GROUP_A0.25_C1.0": 5}。
判读：对最佳全局力度、精确均量全局和精确均量打乱的三种比较，分别回答分组是否实用、是否只是总改动量变化、是否需要正确节点分配。
决策：只在分组规则同时超出精确均量/打乱控制的条件保留节点选择解释；单独优于CE时继续保留为监督增量证据。

### D2

- R2 GATE_LCSS 对 TRAIN_CE：带噪accuracy +2.249pp，验证 +2.209pp，macro-F1 +1.455pp；9胜/0负/9格。
- R2 GATE_LCSS 对 CLEAN_META_CE：带噪accuracy -6.453pp，验证 -7.154pp，macro-F1 -9.026pp；1胜/8负/9格。
- R2 GATE_LCSS 对 CLEAN_META_GCE：带噪accuracy -8.343pp，验证 -8.323pp，macro-F1 -11.060pp；0胜/9负/9格。
- R2 GATE_LCSS 对 GLOBAL_LCSS：带噪accuracy +0.049pp，验证 -0.063pp，macro-F1 -0.087pp；4胜/5负/9格。
- R2 GATE_LCSS 对 GATE_MASS：带噪accuracy +0.296pp，验证 +0.369pp，macro-F1 -0.495pp；7胜/2负/9格。
- R2 GATE_LCSS 对 GATE_KM：带噪accuracy -0.479pp，验证 -0.590pp，macro-F1 -1.270pp；4胜/5负/9格。
- R2 GATE_LCSS 对 MEAN_LCSS：带噪accuracy +0.020pp，验证 -0.052pp，macro-F1 -0.031pp；4胜/4负/9格。
- R2 GATE_LCSS 对 SHUFFLE_LCSS：带噪accuracy +0.027pp，验证 -0.052pp，macro-F1 -0.026pp；5胜/3负/9格。

本轮具体决策（R2）：暂缓复杂门控主线：同清洁标签预算、KM和逐轮均量控制仍有竞争力，当前不融合C/D。

判读：TRAIN_CE/GLOBAL_LCSS使用相同selection子集；CLEAN_META_CE/GCE直接使用同量干净meta标签；MEAN/SHUFFLE逐epoch回放相同实际转移量。元梯度与目标stopgrad均有测试和运行日志。
决策：GATE_LCSS若低于直接干净标签监督、KM门控或等量回放，暂缓复杂门控主线；仅有局部单格收益时保留原规格做小范围重复建议，当前不拼接C/D。

## D1/D2 节点层面结果

以下将每格训练节点先分层求均值，再对带噪格等权平均。original_clean/corrupted使用后处理真值；group_0/1/2分别为唯一领先同意、冲突、并列。
class_size使用真实训练类别频数的三分位分组，限后处理；节点明细含基线观测类概率/损失、目标Brier变化、真实类别质量变化及所选力度。

|阶段|骨干|臂|分层|平均力度|原m|实际转移量|目标Brier变化|真实类质量变化|学生clean accuracy|
|---|---|---|---|---:|---:|---:|---:|---:|---:|
|D1|R0|EXACT_MASS_GLOBAL|group_0|0.864|0.095|0.082|-0.0033|-0.0760|92.50%|
|D1|R0|EXACT_MASS_GLOBAL|group_1|0.864|0.308|0.262|-0.5046|+0.0578|43.82%|
|D1|R0|EXACT_MASS_GLOBAL|group_2|0.864|0.135|0.116|-0.1240|-0.0435|59.90%|
|D1|R0|EXACT_MASS_GLOBAL|original_clean|0.864|0.149|0.129|+0.0370|-0.1287|97.18%|
|D1|R0|EXACT_MASS_GLOBAL|original_corrupted|0.864|0.299|0.255|-0.6898|+0.1566|19.28%|
|D1|R0|EXACT_MASS_GLOBAL|true_class_size_0|0.864|0.209|0.179|-0.2298|-0.0298|55.37%|
|D1|R0|GROUP_SELECTED|group_0|0.083|0.095|0.008|-0.0012|-0.0074|93.29%|
|D1|R0|GROUP_SELECTED|group_1|1.000|0.308|0.308|-0.5656|+0.0707|48.28%|
|D1|R0|GROUP_SELECTED|group_2|1.000|0.135|0.135|-0.1407|-0.0503|60.67%|
|D1|R0|GROUP_SELECTED|original_clean|0.446|0.149|0.096|+0.0424|-0.0960|97.12%|
|D1|R0|GROUP_SELECTED|original_corrupted|0.957|0.299|0.295|-0.7721|+0.1845|25.91%|
|D1|R0|GROUP_SELECTED|true_class_size_0|0.767|0.209|0.187|-0.2535|-0.0138|57.10%|
|D1|R0|MASS_MATCHED_SHUFFLE|group_0|0.863|0.095|0.082|-0.0027|-0.0760|92.40%|
|D1|R0|MASS_MATCHED_SHUFFLE|group_1|0.864|0.308|0.263|-0.4971|+0.0575|46.48%|
|D1|R0|MASS_MATCHED_SHUFFLE|group_2|0.860|0.135|0.115|-0.1245|-0.0425|58.41%|
|D1|R0|MASS_MATCHED_SHUFFLE|original_clean|0.862|0.149|0.128|+0.0386|-0.1285|96.53%|
|D1|R0|MASS_MATCHED_SHUFFLE|original_corrupted|0.865|0.299|0.255|-0.6823|+0.1561|23.20%|
|D1|R0|MASS_MATCHED_SHUFFLE|true_class_size_0|0.861|0.209|0.179|-0.2229|-0.0311|55.74%|
|D1|R2|EXACT_MASS_GLOBAL|group_0|0.856|0.095|0.081|-0.0034|-0.0752|86.07%|
|D1|R2|EXACT_MASS_GLOBAL|group_1|0.856|0.308|0.259|-0.5018|+0.0574|58.93%|
|D1|R2|EXACT_MASS_GLOBAL|group_2|0.856|0.135|0.115|-0.1247|-0.0422|57.68%|
|D1|R2|EXACT_MASS_GLOBAL|original_clean|0.856|0.149|0.127|+0.0357|-0.1271|69.85%|
|D1|R2|EXACT_MASS_GLOBAL|original_corrupted|0.856|0.299|0.252|-0.6835|+0.1544|63.45%|
|D1|R2|EXACT_MASS_GLOBAL|true_class_size_0|0.856|0.209|0.177|-0.2295|-0.0289|43.95%|
|D1|R2|GROUP_SELECTED|group_0|0.056|0.095|0.004|-0.0008|-0.0040|86.98%|
|D1|R2|GROUP_SELECTED|group_1|1.000|0.308|0.308|-0.5656|+0.0707|59.66%|
|D1|R2|GROUP_SELECTED|group_2|0.944|0.135|0.126|-0.1376|-0.0450|58.03%|
|D1|R2|GROUP_SELECTED|original_clean|0.437|0.149|0.094|+0.0419|-0.0940|70.29%|
|D1|R2|GROUP_SELECTED|original_corrupted|0.950|0.299|0.294|-0.7695|+0.1841|64.52%|
|D1|R2|GROUP_SELECTED|true_class_size_0|0.751|0.209|0.184|-0.2529|-0.0118|42.98%|
|D1|R2|MASS_MATCHED_SHUFFLE|group_0|0.856|0.095|0.081|-0.0028|-0.0752|85.99%|
|D1|R2|MASS_MATCHED_SHUFFLE|group_1|0.856|0.308|0.259|-0.4940|+0.0573|59.01%|
|D1|R2|MASS_MATCHED_SHUFFLE|group_2|0.852|0.135|0.114|-0.1254|-0.0409|57.20%|
|D1|R2|MASS_MATCHED_SHUFFLE|original_clean|0.855|0.149|0.127|+0.0373|-0.1268|69.70%|
|D1|R2|MASS_MATCHED_SHUFFLE|original_corrupted|0.857|0.299|0.252|-0.6754|+0.1540|63.63%|
|D1|R2|MASS_MATCHED_SHUFFLE|true_class_size_0|0.853|0.209|0.177|-0.2233|-0.0298|43.05%|
|D2|R2|CLEAN_META_CE|group_0|0.000|0.095|0.000|+0.0000|+0.0000|85.14%|
|D2|R2|CLEAN_META_CE|group_1|0.000|0.308|0.000|+0.0000|+0.0000|64.26%|
|D2|R2|CLEAN_META_CE|group_2|0.000|0.135|0.000|+0.0000|+0.0000|64.30%|
|D2|R2|CLEAN_META_CE|original_clean|0.000|0.149|0.000|+0.0000|+0.0000|71.36%|
|D2|R2|CLEAN_META_CE|original_corrupted|0.000|0.299|0.000|+0.0000|+0.0000|68.97%|
|D2|R2|CLEAN_META_CE|true_class_size_0|0.000|0.209|0.000|+0.0000|+0.0000|51.57%|
|D2|R2|CLEAN_META_GCE|group_0|0.000|0.095|0.000|+0.0000|+0.0000|86.17%|
|D2|R2|CLEAN_META_GCE|group_1|0.000|0.308|0.000|+0.0000|+0.0000|65.21%|
|D2|R2|CLEAN_META_GCE|group_2|0.000|0.135|0.000|+0.0000|+0.0000|65.44%|
|D2|R2|CLEAN_META_GCE|original_clean|0.000|0.149|0.000|+0.0000|+0.0000|71.78%|
|D2|R2|CLEAN_META_GCE|original_corrupted|0.000|0.299|0.000|+0.0000|+0.0000|70.64%|
|D2|R2|CLEAN_META_GCE|true_class_size_0|0.000|0.209|0.000|+0.0000|+0.0000|53.58%|
|D2|R2|GATE_LCSS|group_0|0.503|0.095|0.048|-0.0040|-0.0445|83.25%|
|D2|R2|GATE_LCSS|group_1|0.517|0.308|0.161|-0.3471|+0.0394|58.31%|
|D2|R2|GATE_LCSS|group_2|0.489|0.135|0.066|-0.0762|-0.0247|55.11%|
|D2|R2|GATE_LCSS|original_clean|0.504|0.149|0.076|+0.0131|-0.0755|67.96%|
|D2|R2|GATE_LCSS|original_corrupted|0.517|0.299|0.157|-0.4580|+0.0987|62.33%|
|D2|R2|GATE_LCSS|true_class_size_0|0.511|0.209|0.108|-0.1581|-0.0159|41.09%|
|D2|R2|MEAN_LCSS|group_0|0.517|0.095|0.049|-0.0041|-0.0456|83.19%|
|D2|R2|MEAN_LCSS|group_1|0.517|0.308|0.159|-0.3425|+0.0376|57.99%|
|D2|R2|MEAN_LCSS|group_2|0.517|0.135|0.070|-0.0805|-0.0257|55.07%|
|D2|R2|MEAN_LCSS|original_clean|0.517|0.149|0.077|+0.0132|-0.0768|67.86%|
|D2|R2|MEAN_LCSS|original_corrupted|0.517|0.299|0.155|-0.4532|+0.0969|62.00%|
|D2|R2|MEAN_LCSS|true_class_size_0|0.517|0.209|0.108|-0.1575|-0.0166|41.17%|
|D2|R2|SHUFFLE_LCSS|group_0|0.517|0.095|0.049|-0.0042|-0.0456|83.14%|
|D2|R2|SHUFFLE_LCSS|group_1|0.517|0.308|0.159|-0.3424|+0.0376|57.97%|
|D2|R2|SHUFFLE_LCSS|group_2|0.517|0.135|0.070|-0.0805|-0.0256|55.05%|
|D2|R2|SHUFFLE_LCSS|original_clean|0.517|0.149|0.077|+0.0132|-0.0768|67.81%|
|D2|R2|SHUFFLE_LCSS|original_corrupted|0.517|0.299|0.155|-0.4531|+0.0969|61.98%|
|D2|R2|SHUFFLE_LCSS|true_class_size_0|0.517|0.209|0.108|-0.1573|-0.0166|41.06%|

完整分图、条件和所有臂在node_control_strata.csv；低频类别测试保护在small_class_test_strata.csv；逐epoch门控与实际转移量在gate_trajectories.csv。

目标Brier与最终分类分别衡量标签目标和共享学生泛化，二者的同向或反向都在明细保留。

## 运行版本与补充审计

训练源码与sources.json自正式启动后保持冻结。postprocess/supplement.py只读已冻结选择、日志、概率、目标与真值，不调用学生fit或选择器重设计。
补充源/输入哈希见supplement_manifest.json；testing.json原7项计数为文档滞后，原契约测试9项加静态特征及Windows路径修复各1项，共11项，见testing.final.json。
D2第9维观测类别频率最初遗漏静态缩放，在D2首次训练前由独立版本入口修复。原训练文件和sources.json保持原样，生效组件见repair_sources.json及每个门控result.json的static_feature_version。
为修复而受控中断本队列时，B1有一个尚无检查点的122更新部分轨迹，已原样归档；该格从同初始化重跑。1500指有效完整科学拟合，另披露该中断尝试及5个两轮smoke。

Windows路径分隔符与alias展示在独立后处理版本中归一化；原fit_registry.json和冻结选择保留。fit_registry.canonical.json及aliases.csv提供清晰新拟合/复用记录，40个新CE/GCE锚完整进入results.csv。
D2均量控制逐epoch匹配；各臂按自身selection最佳时点选模，节点表按各自选定epoch报告实际力度。
