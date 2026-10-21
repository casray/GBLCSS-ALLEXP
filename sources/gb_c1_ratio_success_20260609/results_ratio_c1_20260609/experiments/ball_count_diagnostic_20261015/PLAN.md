# 异配数据扩展与嵌套粒球数量诊断

2026-10-15。用户要求增加异配分类数据集，暂不做独立kNN/k-means对照，诊断不同球数如何影响实验，再据此重新设计构球。已有GREET结果完成后开启独立目录，不改冻结旧实验。

## Material Passport

任务类型code experiment；证据为本地配置、原始日志、冻结H、分裂树、目标及学生logits；状态planned→running→audited；仅描述实验支持的结论，不伪造统计显著性或最优球数。

## 问题、假设与粒球角色

RQ1：新异配图上，GREET或GRACE的原LCSS分类是否仍受分球限制？RQ2：固定表示、分裂轨迹、最小子球与训练支持时，增加区域数量如何改变监督质量和最终分类？

假设：粗球可能混合类别，细球可能耗尽抗噪证据；合适粒度依赖空间和数据，不是统一sqrt(N)。粒球仍作为局部软监督区域，不当硬正样本。优先检验数量/局部性/支持权衡，不叠新损失或HF门控。

## 数据及协议

新增Actor、Chameleon-filtered、Squirrel-filtered（ICLR2023官方基准仓库NPZ，固定提交与SHA）。保留其随附split0/1/2，不修改掩码；无向边双向化、去重复及自环按既有加载协议。保留Roman官方split、CoraFull/CS旧百分比split。完整保存节点数、类别数、类别分布、原/校正同配指标、split及noise指纹。新增图先读取数据结构再冻结数据manifest；不以测试表现决定纳入。

新图分别训练原预算GRACE1000epoch和GREET portable MC400epoch，seed0、固定最后epoch，无标签probe选点。旧三图复用已有对应H。比较是冻结空间内粒度诊断，不声称两SSL同计算预算排名。GREET内部原有特征20近邻保留；本轮不实现独立kNN/k-means邻域对照。

## 数量干预与必要控制

每图每表示每协议seed只有一棵树，固定min_leaf_size=16和min_train_per_ball=16。最大总SSE叶优先、2-means n_init10，seed+leaf_id；受约束时投影fallback；SSE正收益才分。1/8/16/32/64/128/256及ceil(sqrtN)处保存快照。请求列表增加不得改变已保存的前缀分区。可行性不足时保存终止分区，明确requested/actual/stop_reason，不强凑预算。

之前min_leaf_size=ceil(N/(4B))随B变化；本轮把它固定，不能与旧sqrt分组逐元素等同。旧生成器64球与共同CE保留作锚，旧历史精确复核；新曲线内的变化才用于数量归因。同一目标若因可达上限而相同，允许复用学生，必须保存alias和实际训练次数，不重复计入独立seed。

## 方法、指标与成功/失败标准

三条件clean/U0.4/Pair0.4，三个协议seed。LCSS原留一、q增强、alpha、X-only MLP200epoch与验证选epoch不变。每个预算全跑，不根据中间test缩减矩阵。

工程成功：嵌套性/前缀不变/完整覆盖、正SSE、支持约束、原公式/历史目标与split/noise、所有目标和logits审计通过。研究成功：获得可重复的数量—支持—干净边际—Brier—分类响应，而非必须升分。下降、平台、请求达不到、不同噪声反向都是有效诊断。若分类最佳点只单seed出现、与目标质量无对应或有clean损害，不称稳定最优。

记录实际球数、球大小/训练支持分位数、几何SSE/半径、相邻快照变化、少票覆盖；冻结后再用真值诊断干净投票边际/投票正确率、错标恢复/正确标签受损及Brier；保存accuracy与macroF1的三seed均值/样本std和配对差。不得用真实标签确定分裂或自适应停止。

曲线横轴以实际数量为主，标出sqrt参考；各seed点/细线加均值，标准差不装作置信区间；accuracy/Brier/support分面，不用双y轴。仅诊断性test峰值须明确标为事后观察，不能变成部署阈值。

## 文件、执行与核验

- 新增数据/表示适配：data_tools.py、prepare_data.py、prepare_embeddings.py、test_data.py；下载原官方数据并记SHA，原trainer调用只输入X/A。
- 新增轨迹模块：GB-LCSS_code/src/gblcss/granular_ball_trajectory.py、test_trajectory.py；不修改旧adaptive源码。
- 新增run.py、audit_report.py、plot_curves.py；config/本计划固定，执行阶段分层source manifest及快照。
- 保存raw.log、dataset_manifest、partitions/trace、targets、students/logits、aliases、CSV、mean/std、audit、实验结论与算法重设计建议。

命令（cwd为本workspace）：

1. `& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' experiments/ball_count_diagnostic_20261015/prepare_data.py` 下载并核验三个新图。
2. `& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' experiments/ball_count_diagnostic_20261015/prepare_embeddings.py --device cuda`，新图六个无标签SSL，固定原预算。
3. `& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -m unittest discover -s experiments/ball_count_diagnostic_20261015 -p 'test_*.py' -v`，最快协议与轨迹检查。
4. `& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' experiments/ball_count_diagnostic_20261015/run.py --device cuda`，全量曲线目标和学生，再独立审计及绘图。

## 结果与决策规则

最终实验结论记录已支持/合理解释/待验证/被负结果削弱四类信息；重新设计建议必须对应诊断，明确新算法的输入、局部停止条件和伪代码。不以test最佳B写成自适应算法，不自动增加数据集/阈值搜索。若机制仍无规律，决策为暂缓或保留可行生成器，不编造万能粒度。
