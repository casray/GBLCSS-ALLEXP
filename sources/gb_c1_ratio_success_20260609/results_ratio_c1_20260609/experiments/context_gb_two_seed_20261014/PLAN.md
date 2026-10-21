# Context-GB 两个完整seed探索实验

登记日期：2026-10-14。用户明确批准04方案开始实验，两个seed，暂不比较kNN/k-means。本文是本轮执行范围；共享工作目录中的其他异配/自适应生成器任务不自动改变本轮方案。

## 研究假设与动机

学习增强能够改善GRACE节点空间；独立上下文辅助目标可进一步改善局部标签证据估计。粒球作为冻结表示上的局部标签聚合器，不作硬语义标签或输入增强控制器。重点检验新空间→软目标质量→最终分类的链路。

## 范围及配置

- 数据：Cora、CiteSeer、CoraFull，使用现有原图文件。
- 完整流程seed：0、1；每个seed独立预训练、原20/10/70划分、噪声及分类训练。方法内配对相同seed。
- 条件：clean、uniform0.4、pair0.4；仅训练标签污染，干净验证选模。
- 主预训练arms：grace、pingda、pingda_context、pingda_hreg。原GRACE隐藏宽度、投影宽度、学习率、温度及特征预处理沿用已有配置；统一200轮。CoraFull原论文1000轮，因此本轮为预算明确的探索，不与旧论文数字作直接量化胜负。
- PiNGDA为作者生成器在统一GRACE骨干上的增强适配，不声称复现原论文全部配置或benchmark。拓扑一视图原图、另一视图作者soft Gumbel权重；属性依作者代码双采样。实际API、代码差异/来源hash另记provenance。
- context-only辅助：邻居至多16个，不含自环；共同有效root mask；φ输出128；λ固定0.1，VICReg/25；不按test搜索λ。Hreg对照等维度φ、同目标同λ，输入为H而非邻域均值。
- 上下文输出最终用原图全部邻居均值。组合空间context_mix=0.5，vote_mix=0.5分别配置。有效性和回退规则参见04。
- 表示spaces：GRACE-H；PiNGDA-H；PiNGDA固定mean-R；context训练后的H；context训练后的R；Hreg-H。
- 球预算：64、128、256，原生成器min_child32、radius改善0.05、二分10初始化；GB内部2-means保留，独立KM/KNN对照本轮排除。
- 学生/分类器：原X-only MLP及固定GCN，200轮，宽64，dropout0.5，lr0.01，wd0.0005；相同seed初始化与噪声，按干净验证accuracy保留首次最优。每图/seed/噪声各增加CE基线。首轮不同时研发新蒸馏。
- LS-GCL/CL-GCL是后续完整基座比较，本轮先执行最小主方案与关键消融；无已验证官方适配入口时不伪造复现结果。

## 代码改动与机制

仅新增本目录：core.py、pingda.py、test_core.py、test_pingda.py、run_experiment.py、audit.py及记录。复用GB-LCSS_code的data/targets/granular_ball/training/models，不修改正式默认入口。对比损失采用分块重计算与等价梯度验证以控制CoraFull显存，不采样/删除负例。

对上下文损失阻断生成器的直接梯度路径；生成器仍由原节点对比目标更新。节点θ仍共享，不能称完全无冲突。每轮日志记录节点/上下文loss、有效root数、耗时与GPU峰值。最终记录实际球数、支持覆盖、target Brier、clean vote margin和分类准确率。

## 成功/失败标准（运行前固定）

本轮仅两个seed，用于探索而非显著性宣称。context-R相对同预算PiNGDA-H在两个带噪条件的六数据集条件平均提升≥0.5pp、各数据集带噪平均不退化超过0.5pp，且相对固定mean-R/Hreg存在同向机制证据，才建议进入更大seed确认；clean副作用单独报告，不以平均遮盖。所有预算都完整报告，64为主预注册预算，128/256为容量敏感性，不能按test逐数据集选最好budget后包装单一算法。

任意非有限loss、无上下文有效root、表示退化、只改善球几何但下游不涨，需记录失败；某个arm失败不删除其他有效结果。若仅PiNGDA有效，保留表示升级、暂缓上下文；若context不优于mean/Hreg，拒绝上下文特有解释。因用户暂不做KM/KNN，本轮不能判断粒球不可替代性。

## 验证、结果与决策

先执行test_core.py与test_pingda.py，再小图短训练检查；满足数值、梯度及标签访问边界后执行正式队列。数据/split/noise/source/config哈希留存，断点续跑须相同源hash。原始日志/逐seed CSV/mean及样本std/配对差值/logits保存。

实验结果：待运行。最终决策：待按预注册规则填写。运行命令与完成状态以raw.log、status.json和最终实验结论.md为准。

数据身份审计：NoisyGL下Cora/CiteSeer的`data_undirected.pt`实际是CitationFull版本（19793/4230节点），不能按目录名使用；本轮使用`data.pt`的2708/3327节点版本，和已有GRACE数据副本X/Y逐值一致。保留既有索引生成中的整数取整剩余节点，不偷偷修改split；审计报告覆盖率及未划分节点数。

本轮解释性读图/标签统计只在表示训练完毕及目标冻结后进行，不用于预训练或构球。
