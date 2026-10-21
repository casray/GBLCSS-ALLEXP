# 无标签BFS→LCSS实验：冻结设计与执行计划

日期2026-10-15。用户已在前轮审查后授权“可以，设计并开始实验”。沿用research/topology_ball_review_20261014/00_BFS拓扑粒球替代GRACE路线审查.md。该授权足以执行，无需重复确认。该独立方向不修改其他对话的GREET/HALF与Context-GB实验。

**Goal:** 验证省去GRACE的拓扑区域是否提供可靠的抗标签噪声软监督，并量化简化价值。
**Architecture:** 无标签图→固定BFS分区→原留一LCSS目标→原属性MLP。对照使用既有GRACE/GB64、相同球数H/X几何、随机支持匹配和票数匹配kNN。
**Tech Stack:** 既有Python/NumPy/SciPy/scikit-learn/PyTorch环境；单GPU串行学生训练，CPU四线程诊断。

## 假设、角色与方法

研究假设：拓扑区域可能提供低成本局部监督；在异配结构下错误的干净多数可能主导失败。粒球只组织候选投票，不作为硬正样本。预期机制由支持量和root真类margin共同决定，不能用区域连通性或球数替代监督质量。

BFS接口仅接收edge_index和num_nodes、可选容量；不接收X、Y、训练集合或类别数。去自环、对称化、去重，邻居按节点ID升序。每次中心在未分配诱导图中按最大剩余度、同度最小ID选择；BFS节点出队时归属，最多floor(sqrt(N))；入队并不提前标为已分配。队列耗尽就结束当前球，保留小组件/碎片/孤立点。相同距离节点以FIFO/邻居升序打破并列。堆维护度变化，避免每球全图重算。所有节点完整互斥覆盖，每球连通，实际球数可超过sqrt(N)。该确定性实现是GBGCL启发版本，不是其完整复现；不做标签质量分裂。

随机对照分别随机打乱训练节点和非训练节点，在每个球中保留原大小和训练节点数；只读训练位置，排除标签。分组随机种子=协议seed。

## 数据与配置

数据CoraFull、Roman-empire、Coauthor-CS；复用现有loader、节点顺序、X、A、split和污染函数。seed0–4；CoraFull/CS既有20/10/70整数划分，Roman官方前5mask。clean/Uniform .4/Pair .4；Pair为既有类索引循环翻转。验证/测试标签不污染，真值仅用于冻结后诊断与评估。

LCSS min_other_votes=2、beta=.5、原alpha公式与并列处理固定。学生hidden64、2层含BN、dropout.5、Adam lr.01/wd.0005、200轮、干净验证选点、student_seed=协议seed；与历史同协议比较。每图一个既有GRACE表示，不能将五协议种子当五次SSL。

## 对照

CE：直接观测标签。OLD64：历史GRACE GB64原分组与目标逐字节核对。
BFS：上述拓扑分组。
H_GEOM：归一化GRACE H上的BisectingKMeans，目标cluster数为BFS实际球数。
X_GEOM：归一化原X上的同一个BisectingKMeans。
RANDOM：BFS大小及训练支持分布匹配随机分组。
H_KNN：每个训练节点其他票数逐节点匹配BFS；既有确定性稳定排序实现。

H/X几何统一seed0、n_init1、max_iter50、tol1e-4、biggest_inertia、k-means++、L2归一化，不用标签与训练mask；原X高维采用CSR以避免稠密重复。实际cluster数、大小/支持差异和任何退化警告均记录；同球数不声称同支持。该控制不同于旧递归门槛，不把收益归于一个变化。BFS/H_GEOM/X_GEOM先对每图固定一次，再跨协议seed复用；OLD64使用对应历史协议分组。

## 分阶段门槛

先45个dataset/seed/noise单元，每单元7臂，共315目标记录。完整冻结所有目标/关系/hash后才读干净标签评估。报告Brier、q准确率/真类概率、支持覆盖、root正margin、正确标签伤害、错标恢复、alpha>=.75时q错误率、区域大小/训练支持/组件数量和构球耗时。

进入学生阶段的探索门槛按数据集确定：BFS在Uniform .4和Pair .4均比CE平均Brier至少低.005，并各至少4/5配对改善；相对RANDOM同样要求平均低.005且至少4/5改善。H/X/KNN强对照全部报告，门槛通过不等于胜过它们。clean损害如实报告，不要求低于干净CE的零误差。若三图均未过门，停止学生训练，保留失败结论；不根据测试准确率修改规则。

门槛通过图再训练全部七臂×五seed×三条件，最多315学生；结果n为实际训练数。历史CE/OLD64可复用已有result/logits，但须独立核验来源config/目标逐字节/划分/污染/模型训练源码hash与准确率；若不能验证则同协议重跑，不能用缓存分数冒充新训练。配对学生同一个seed初始化。

低成本简化的探索判据：BFS三条件均值不比OLD64低超过1pp、带噪优于CE、至少4/5配对支持主要带噪收益；1pp是预设探索容忍度，不是正式统计非劣证明。若只同配图有效，限制结论。新论文机制还需胜过强邻域和跨图证据。

## 效率与科学边界

测构球、目标、学生的实际时间与内存，报告缓存复用。旧GRACE预训练仅引用原文/已有计时来源，未重跑，不能把74.48分钟计入每个下游。不使用验证/测试真值选球数、X/H聚类参数或目标权重；训练真值诊断用于研究门槛，不构成可部署的自动选择器。

## 文件与任务

- [ ] 新增GB-LCSS_code/src/gblcss/granular_ball_topology.py，仅BFS和随机对照。
- [ ] 新增本目录test_topology.py，先红后绿：覆盖/连通/容量、残余度中心、顺序和重复边不变、孤立点、非法输入、无标签API、随机支持匹配。
- [ ] 新增run_screen.py，复用既有loader/污染/CSR投票，冻结源清单、分组、关系、目标、协议及metrics。
- [ ] 新增run_students.py，按冻结gate串行执行并留raw logs/config/logits。
- [ ] 新增audit.py与analyze.py，独立原公式/边连通/随机支持/协议/hash/logits/汇总复核；保存decision与详细结论。
- [ ] 同步本方向历史与root实验入口、交接记忆；不覆盖其他方向文档。

最快检查命令：D:/codex_temp/noisygl-rocm721/Scripts/python.exe -X utf8 -m unittest discover -s experiments/topology_bfs_lcss_20261015 -p test_topology.py -v。
诊断命令：同解释器 -X utf8 experiments/topology_bfs_lcss_20261015/run_screen.py。
后续audit.py、analyze.py及run_students.py的命令在执行前说明。

成功/失败报告包含全部七类必需追溯：config、raw log、逐seed表、mean/std、配对seed、baseline和最终继续/修改/暂缓/放弃决策。结果尚未产生，完成后单独填入报告，不回改冻结规则。
