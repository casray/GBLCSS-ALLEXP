# 2026-10-12 区域统计额外信息诊断交接

用户授权设计并执行新实验，已投稿论文不整理。实验已完成，无运行中的任务。

## 本轮范围

复用五数据集 × uniform .4/.6 × 协议seeds1/2 的20组 ERASE/GRACE/GB缓存。
GPU精确余弦检索；CPU聚类、50个整图留出逻辑回归拟合；200行指标。
响应仅为训练标签是否被污染；真实训练污染标记用于探针拟合/评估，因此不是可部署算法。
五个数据集此前已用过，本轮是探索性筛查，不是外部确认。

## 实验决策

Base+kNN AUROC .9558294095，加GB .9548947295，差 -.0009346800，8胜12负，2/5图正。
GB相对KM/BKM差 +.0005118226 / +.0004284385。
Precision@20%相对主对照下降 .0022479383，干净误标率上升 .0006629225。
两个种子、两档噪声的平均差均为负。G1/G2/G3都失败。
**停止当前教师标签区域可信度分支，不启动教师/学生训练，不加新特征追分。**

## 可追溯入口

- PLAN.md：结果前固定目标、对照、特征、评估与门槛。
- run_diagnostic.py / test_diagnostic.py / audit_results.py：运行、六项测试、独立核验。
- 实验结论.md：完整中文方法、数值、负结果与解释边界。
- input_manifest.json / environment.json / raw_log.txt：配置、原输入和环境。
- features/ / partitions/ / evaluation/：固定特征、对照成员、整图留出预测和拟合系数。
- per_cell_metrics.csv / dataset_metrics.csv / macro_metrics.csv / paired_deltas.csv / macro_deltas.csv。
- decision.json / execution.json / audit.json / output_manifest.json。

## 核验

运行 exit0，37.42秒；六测试通过，pytest缓存权限警告非测试失败。
20组协议指纹核验通过，133输入/脚本SHA-256未变，200行重算最大差0。
生成文件散列一致；留出图不参与相应探针拟合。未读取测试标签作为响应。

## 下一窗口

优先读实验结论.md，不把这次失败改写成“再调一下”。若提出新方向，先说明目标和新增证据。
原GB无树缓存，本轮没有多层级特征；ProCon控制是固定H空间诊断适配，不是完整ProCon。
本轮不整理论文，不修改原实验缓存；项目上下文与历史42仅新增此实验入口。
