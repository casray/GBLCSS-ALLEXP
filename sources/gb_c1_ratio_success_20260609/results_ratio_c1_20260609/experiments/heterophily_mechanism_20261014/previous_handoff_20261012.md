# 交接：2026-10-12区域预算蒸馏诊断

本workspace：`gb_c1_ratio_success_20260609/results_ratio_c1_20260609`。
已投稿论文未改。无后台训练待续。

## 本轮目标与输入

用户批准测试教师知识压缩及区域复杂度预算蒸馏。先完成20261007未运行的知识诊断，再做GPU DIST损失/分类头梯度诊断。五数据集、uniform0.4/0.6、旧协议seed1/2，共20组。

输入：`experiments/gb_dist_protocol_seeds_20261007/`的ERASE概率、原GB成员、GRACE路径、epoch200学生；BKM来自`experiments/gb_signal_increment_diagnostic_20261012/partitions/`，区域数及表示散列匹配。

## 新结论与停止决定

知识压缩G1/G3通过，G2失败：GB/Random MSE=.197466，GB/KM=1.263040，0/10单元胜KM；25%预算自适应/规模均值估计MSE=.713249。

GPU预算DIST的D1-D4全部失败。25%预算分类头梯度相对RMSE：Uniform .40141、GB规模 .38643、GB自适应 .41876、KM .40666、BKM .40891。GB自适应比Uniform高4.32%，比GB规模高8.37%；epoch200平均梯度相对偏差.1433、方向余弦.8501未过门槛。

停止当前教师概率方差分配的预算DIST实现，没有启动学生训练。不得说全部采样方法失败；不得以均值估计改善支持蒸馏精度、速度或GB独有优势。

## 产物与核验

- `docs/budget_dist_diagnostic_20261012.md`：完整方向复盘、边界和下步约束。
- `experiments/gb_knowledge_diagnostic_20261007/实验结论.md`、summary.json和原始五类CSV。
- `experiments/gb_budget_dist_diagnostic_20261012/PLAN.md`、实验结论.md、summary.json、state_metrics.csv、replicates.csv、audit.json。
- 新增加权DIST、诊断、分析及测试脚本；未改冻结旧代码、教师或学生。
- 11测试通过；125及160文件散列未变；14400诊断记录、360预算配置通过独立核验；末层梯度与自动微分一致。
- CPU统计35.71秒，RX7700XT诊断44.57秒；不计作学生训练效率收益。

运行环境：`D:/codex_temp/noisygl-rocm721/Scripts/python.exe`，PyTorch2.9.1+rocm7.2.1。

## 下一窗口第一步

先读上述复盘及两个PLAN/summary。当前没有获批的新训练或参数搜索。下一方向须先解释改变了什么研究问题、为什么不会重复本轮负结果；不能自动追加梯度方差采样或更复杂粒球模块。

本轮交接写在可写workspace内；workspace外的全局项目总结、旧历史及已投稿材料未修改。
