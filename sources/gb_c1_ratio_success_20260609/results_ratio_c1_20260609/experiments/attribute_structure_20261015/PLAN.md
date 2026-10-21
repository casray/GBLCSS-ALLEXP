# 属性与结构差异互补：实施与实验计划

> 使用writing-plans、test-driven-development与subagent-driven-development执行；用户2026-10-15明确批准“可以，开始实验”。无需重复确认。实施文件与产物在本独立目录，原代码和旧实验保持冻结。当前工作数据和源码大部分未纳入上层Git，故采用独立实验目录与SHA清单隔离，不复制大型数据到新worktree，不提交无关改动。

**Goal:** 验证增加异配数据后原属性区域的抗噪能力，以及固定结构差异通道能否提供超过纯属性和打乱结构的监督增量。
**Architecture:** 固定X/A无标签构球→原LCSS→原X-only MLP；新图原GRACE作为参照独立1000轮无标签训练；二分类按ROC-AUC选epoch。先冻结全部目标与gate，再训练固定基础臂及通过gate的结构候选。
**Tech Stack:** 现有Python3.12/PyTorch ROCm/PyG/NumPy/SciPy/scikit-learn；GPU队列串行，CPU单线程KMeans与四线程准备。

## Material Passport与证据边界

状态planned→running→audited。输入为公开图、既有冻结表示与原训练实现。真实训练类别只用于生成观测带噪标签和冻结后诊断；验证/测试类别不进入构球或gate。首轮每图一个SSLseed，五协议seed不是五个独立预训练。本计划不主张新SOTA或已经成功。

## 研究假设、粒球角色、动机

H1：纯原属性可能比BFS连通区域更好地组织语义投票。H2：R=[rowL2(X),rowL2(PX−X)]/sqrt2中的对应关系可能改善原属性区域；打乱残差为必要对照。粒球只组织局部留一投票，不提供硬对比标签或强prototype损失。

旧BFS三图结果不能外推新图；同球数不保证同支持分布。结构差异是固定候选，不称异配理论保证。若目标改善而分类不改善，说明目标质量不足以说明X-only可学习性，暂不自动改有图学生。

## 数据、协议和配置

Amazon-ratings/Tolokers官方pin提交见config，下载本地NPZ并保存SHA、字段、节点顺序、五split完整不交叠核查。Roman/CS复用旧图和split来源。四图均五协议seed0–4；Amazon/Roman/CS clean/U.4/Pair.4；Tolokers clean/sym.2/sym.4，二分类不重复Uniform/Pair。验证/测试干净。逐条件保存split/noise指纹。

新两图GRACE只读取X/A，使用既有prepare_representations.py（原GB-LCSS配置，1000轮，chunk512，seed0，固定末轮，无标签probe）。旧Roman/CS H复用。CPU残差使用去环、双向去重邻接P=D^-1A，孤立行P=0；零范数行保留0。SHUFFLE只固定种子置换V的行，U不变；输出保存浮点32，以生成器既有内部精度处理。

基础5臂CE/OLD64/H_TREE/X_TREE/BFS。OLD64原max64/minchild32/5%门槛/协议seed；TREE沿用固定支持16轨迹，最大球数ceil√N，minleaf16，协议seed，单线程KMeans；BFS沿用确定性无标签模块，容量floor√N。R_TREE/SHUFFLE_TREE与X/H同TREE规则，不调权重或支持阈值。粒度和实际支持同时报告，不把整体空间差归因于单一球数。

420目标记录=4图×5seed×3条件×7臂。全部基础学生300次；通过gate图的R与SHUFFLE每图另30次，最多420。Minesweeper留第二阶段，不在本固定矩阵中。已实现Actor/filtered两图的独立分支不重训。

学生hidden64两层BN/dropout.5/Adam .01/wd.0005/200轮，配对初始化seed。Tolokers两logitsoftmax软CE，验证正类概率ROC-AUC选epoch，辅accuracy/macroF1/AP；其余与原accuracy验证选epoch相同，辅macroF1。只按严格大于更新最佳epoch，并列保留最早。所有保存logits，主指标必须独立重算。单类别验证折为协议错误，不用accuracy静默回退。

## 冻结门槛、成功与失败标准

所有目标先保存后评估。R相对X和SHUFFLE在该图两种噪声下的平均Brier差均≤−.005、各至少4/5改善才推进两个候选学生；全条件原始诊断保留。gate不读取test，X/H/BFS/OLD64/CE基础学生固定执行。

分类探索成功：候选在两噪声下分别比X主指标至少+.5百分点、各≥4/5正；clean相对X不低于−1pp；还须完整报告对OLD64/H_TREE及SHUFFLE，若仅单图成功不称普遍适配。1pp不是统计非劣检验；n5的std不是CI；不因test调整参数。

gate失败则停止该结构候选分类，不自动搜残差权重或二跳。若基础学生损害干净精度或不及旧方法，记录负结果。若真实结构与SHUFFLE没有差异，不称结构贡献。

## 文件、任务和验证

- [ ] T1 common.py/prepare.py：下载固定官方数据，核manifest、masks、旧协议、训练两新GRACE；已有trainer只做可追踪适配。最快检查`python -m py_compile ...`与小数据mask/类标验证。
- [ ] T2 features.py/test_features.py：先写手算链图、重复边/自环、孤立点、零属性、置换边际保持、无标签接口测试，观察import缺失红，再实现最小CPUCSR块乘，验证绿；保存红绿日志。
- [ ] T3 metric_training.py/test_metric_training.py：先验证AUC与accuracy排序相反时选点、softmax概率AUC、并列早停、保存logits指标一致；合成图CPU两轮训练；与旧多类训练相同seed/目标核logits。原trainer不改。
- [ ] T4 run.py：逐图准备无标签空间/逐seed分区，冻结目标和关系后调用旧LCSS公式，再输出target CSV和gate；完整raw日志/状态和resume source/hash守卫。中断后汇总始终从原JSON重建。
- [ ] T5 run_students.py：读取已审计目标及gate，串行训练300+条件候选，逐run config/result/logits；不复用历史学生分数。
- [ ] T6 audit.py：独立重建P/残差与置换、分区支持、原LCSS目标、split/noise、gate及完整键集合，重算保存logits的全部主辅指标；source/config/H/data SHA核查。
- [ ] T7 实验结论.md/decision.json/方向历史/根交接：逐图mean/std和配对差、clean损害、机制诊断、已支持/解释/待验证/削弱分层；不隐瞒gate失败或分类负结果。

命令以D:/codex_temp/noisygl-rocm721/Scripts/python.exe -X utf8执行：

1. `-m unittest discover -s experiments/attribute_structure_20261015 -p test_*.py -v`：红/绿与最快协议检查。
2. `experiments/attribute_structure_20261015/prepare.py --device cuda`：下载、核对数据和两新GRACE，写preparation_manifest/状态。
3. `experiments/attribute_structure_20261015/run.py`：420目标和gate。
4. `experiments/attribute_structure_20261015/audit.py --targets-only`：目标独立审计。
5. `experiments/attribute_structure_20261015/run_students.py --device cuda`：固定基础和通过gate学生。
6. `experiments/attribute_structure_20261015/audit.py`：最终完整审计，再生成报告。

## 代码与配置改动边界

仅新增本目录脚本/tests/config/产物，复用原granular_ball、trajectory、topology、targets、models；原source路径纳入SHA清单。不会覆盖旧目录H、split、目标或结果。数据下载是用户明确实验所需的公开数据访问，不添加产品遥测。若资源不足只做保持数学等价的内存分块，修改必须冻结在开跑前并验证。
