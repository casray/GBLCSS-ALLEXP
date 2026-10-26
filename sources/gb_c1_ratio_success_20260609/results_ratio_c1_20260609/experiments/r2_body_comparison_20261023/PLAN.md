# R2 正文对照实验

## 假设、动机和粒球角色
R2 从固定 GRACE 表示迁移结构知识；在同一 R2 编码器和分类头下，检验
LCSS 相对 CE、GCE 和图标签平滑仍有多少增量。粒球仅作为训练目标的共享计数单元。
第二部分与 SimMLP/GSSC 比较整体方法，记录表示预训练及辅助阶段预算。

## 冻结矩阵
R2: cora/citeseer/corafull/coauthorcs/amazonpho，clean/uniform0.4/uniform0.6/pair0.4，
seed0/1/2；CE/GCE/OLD64/SUPPORT/KM/POSTEL/SALS/ALS 共480个分类头。
每个数据集每seed共享已有无标签R2编码器，隐藏64，dropout .5，encoder与BN冻结。
分类头200轮，Adam .01、wd .0005，首次严格最大clean验证准确率；无早停和新参数搜索。
所有臂共享初始分类头。GCE q=.7，使用稳定exp(q*logp)。
SUPPORT为历史SUPPORT32；KM与SUPPORT实际组数匹配，KMeans++ n_init10 seed相同，H相同。
OLD64为历史分区，beta=.5、LOO、min_votes2、并列保留分布。
POSTEL为已审计no-IPL目标构造；SALS/ALS为原文机制适配，参数沿用投稿实现默认值，
ALS保留可学习类相关矩阵、传播和pacing，不将其简化为静态目标。
适配说明写入结果表；完整PosteL迭代伪标签另行实现后才可作为该版本结果。

SimMLP/GSSC: 同五图四条件seed0，既有uniform.4结果经协议核对后复用。
SimMLP复用作者核心和200轮无标签预训练checkpoint，四条件均新增200轮线性探针，
每次在初始化探针前固定seed0，故四条件共享同一初始头。
旧实现未保存预训练末尾的RNG状态，原uniform.4探针仅作历史参考，表中使用新统一初始化结果。
GSSC复用作者核心移植及原辅助阶段预算，其他三条件重训，保留原last-tie选模规则。
外部两法保留其算法结构和独立预算，单列部署参数量，不声称完全相同的网络。
本轮先补这些直接相关且本地有作者代码的对照；GraphECL后续优先级低。

## 数据纪律和质量验证
20/10/70固定历史split，仅train噪声，使用已有观测标签和验证标签。
模型输入中的非训练节点标签置-1；验证向量单独用于选模。
测试标签只在全部拟合选模结束后用于统一分析。预处理、loss、超参选择均不访问测试标签。
历史beta=.5作为开发配置，本轮不以测试指标重选beta。
保存config、源码和输入哈希、日志、checkpoint、probabilities、targets、初始化/冻结编码器hash。
先跑稳定GCE、标签隔离、目标归一化、冻结encoder测试和短GPU冒烟；主拟合GPU串行。
失败留痕，重复启动跳过完成单元，部分失败需明确恢复说明。

## 预期机制与门槛
R2相同骨干的SUPPORT−CE以及SUPPORT−max(GCE,POSTEL,SALS,ALS,KM)分别报告。
继续改进监督模块的实用门：带噪宏平均对最强对照至少+.5pp，至少3/5图均值正，
3个seed宏差均正，clean每图对CE损害不超1pp。失败保留适用范围，不追调参数。
外部基线是整体方法参考，无测试筛选后才决定运行的门槛；全部预定单元执行。
主表5种子协议尚需独立确认，本轮为3seed固定SSL的机制研究；外部seed0 std留空。

## 结果和最终决策
由REPORT_zh.md、summary.csv、paired.csv、audit.json记录；准确率与macro-F1并列。
所有结果均保存，正负单元完整报告。正文修改待结果确认后单独进行。

## 命令
D:/codex_temp/noisygl-rocm721/Scripts/python.exe -m unittest discover -s experiments/r2_body_comparison_20261023 -p test_*.py -v
D:/codex_temp/noisygl-rocm721/Scripts/python.exe -u experiments/r2_body_comparison_20261023/run.py
