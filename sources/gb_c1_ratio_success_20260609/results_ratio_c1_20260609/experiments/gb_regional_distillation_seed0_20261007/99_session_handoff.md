# 区域分布蒸馏单种子实验交接

日期：2026-10-07。本轮用户授权运行预先讨论的五组对照，已完成；没有活跃训练进程。

## 本轮目标与协议

检验固定 ERASE 教师、同一属性 MLP 和同一训练协议下，真实 GRACE 粒球区域分布匹配能否改善普通蒸馏。
五数据集 Cora/CiteSeer/CoraFull/Coauthor-CS/Amazon-Photo，uniform 0.4/0.6，seed 0。五组共 50 次、每次 200 epochs。
R0 原 CE，R1 CE+真实区域 KL，R2 CE+随机大小匹配区域 KL，R3 CE+全局 KL，R4 2*CE。区域系数固定 1，随机分组 seed 137。
只按干净验证精度选首个最优 epoch；测试标签不用于训练目标或本轮调参。原图教师没有重训。

## 关键来源

- 本目录 `PLAN.md`：运行前假设、机制、参数与五项门槛。
- `../gb_distillation_research_review_20261006/改进建议.md`、`cached_diagnostics.json`：前轮诊断与新假设。
- `../robust_graph_seed0_20261006/run_screen.py` 和其教师/普通 KD 缓存：复用训练协议和教师输出。
- `../gb_guided_distillation_seed0_20261006`：旧软目标混合候选，仅对照历史，不修改。
- `../main_protocol_replay_20260922/matrix`：真实粒球 assignment 来源。
- 项目文献图谱及上级 `experiment_history_and_lessons/02_experiment_history_and_lessons 33.md`：已阅读的失败模式。

## 文件与命令

新增 `run_regional.py`、`regional_loss.py`、`test_regional_loss.py`、`analyze.py`；没有修改原方法源码、旧实验或论文。
运行环境 `D:/codex_temp/noisygl-rocm721/Scripts/python.exe`，PyTorch 2.9.1+rocm7.2.1，RX 7700 XT。所有学生训练使用 GPU。
命令见 `COMMANDS.md`。5 项单元测试通过；`--prepare`、`--grid`、`analyze.py` 已完成。
50 个模型恢复预测差为 0；10 个 R0 与旧普通 KD 缓存逐元素完全一致；50 组选模轮次和指标正确；每格五组初始化指纹一致。
`cache/` 为冻结输入，`runs/` 包含配置、200 轮日志、模型、概率和结果。汇总见 `paired_results.csv`、`diagnostics.csv`、`per_class_results.csv`、`summary.json`、`output_audit.json`、`实验结论.md`。

## 数值检查修订

Amazon-Photo 0.4 R3 在训练前因 float32 GPU 聚合差异约 3.7e-7 被拦住。双精度 CPU 对照核验全部 30 组教师均值后，仅将检查容差从 rtol=1e-6/atol=1e-7 调为 rtol=1e-5/atol=1e-6。
未改损失、教师、分组、参数、选模或成功门槛。原始 `frozen_inputs.json` 保留；窄范围源码修订有原快照和新旧 SHA-256，见 `guard_amendment.json`、`roundoff_diagnosis.json`、`run_regional_v1.py`、`analyze_v1.py`。失败 traceback 单独保留。43 组完成结果没有覆盖；补跑剩余七组。

## 新结论与负结果

R1 相对 R0：平均 +0.322982 点，8 胜 2 负，最差 -0.686016。相对 R2/R3/R4 分别 +0.392104/+0.314252/+0.289189 点，均为 9 胜 1 负。
五项预设门槛通过四项；主收益门槛未通过，只因平均不足 +0.5 点。按原计划不直接扩大到多种子。
R1 真实区域 KL、区域共同残差、干净训练标签 Brier 十格全下降，但 Cora/CiteSeer 0.4 测试精度下降。更像教师不等于分类更准确。
R1 相对教师平均 +0.894147 点，但 R0 本身已有 +0.571164；新增区域项只能归因额外 +0.322982。CoraFull 两格仍低于教师，不支持全面超越图方法。

## 证据边界与下一窗口

已支持：这一 seed 和冻结协议下存在温和准确率增益及真实成员作用信号。
合理猜测：区域项改变普通 KD 对共同概率偏差的优化优先级。
未验证：跨种子稳定性、教师通用性、相对其他几何分组的价值、退化原因。
削弱：固定系数 1 达到 +0.5 点平均增益的预期。
本轮是新的区域蒸馏扩展，没有使用原 GB-LCSS 噪声票/alpha 软目标，不能把结果写成原机制已验证。
下一窗口首先读 `实验结论.md` 和冻结计划。除非用户授权新的候选实验，不继续调参或跑种子；先分析 Cora 0.4 的退化和与尺度对照的小差值，不按测试结果倒推新配置。
