# 固定教师区域蒸馏：本轮交接

日期：2026-10-07。用户批准增加学生 seed 1/2 的 40 次训练；已全部完成并核验。没有运行 k-means、调参或教师重训。

## 研究问题与实验身份

固定原 seed 0 数据划分、训练标签噪声、GRACE 粒球和 ERASE 教师概率；仅改变学生初始化和 dropout。
五数据集，uniform 0.4/0.6，学生 seed 1/2，R0 原 CE 与 R1 CE+粒球区域 KL（lambda=1）。
两层原始属性 MLP、hidden64、BN、dropout0.5、Adam lr0.01、weight_decay0.0005、200轮，全节点蒸馏，首个验证最优选模。
结论范围是学生优化随机性，不是三个完整数据/教师种子。

## 数值复现修订

首次 GPU index_add 严格回放失败。diagnose_replay.py 证实旧训练器同 seed 重复也有概率和验证轨迹差异。
修订前代码、计划、冻结清单保存在 revision_before_deterministic/；失败日志仍在 replay/。原始旧实验未改。
新 run_stability.RegionalKL 用固定成员均值矩阵 W@P/W@p 计算相同区域 KL，CPU 双精度损失/梯度等价测试通过。
本轮开启/关闭记录的完整 Cora 0.4 R1 训练逐元素一致，验证轨迹与最佳轮次完全一致，见 observation_replay.json。
因数值实现调整，额外重算了全部 20 次 seed 0 参考以及一次无记录控制训练。正式新增学生种子仍为 40 次。
活动冻结清单是 frozen_v2_inputs.json；原 frozen_inputs.json 只对应首次数值实现，不能当作现运行器的活动清单。

## 来源与文件

- 上一轮 ../gb_regional_distillation_seed0_20261007/：教师、区域缓存、旧结果、来源指纹、原 regional_loss。
- 更早 ../gb_distillation_research_review_20261006/：方向设计与旧软目标混合失败的诊断。
- PLAN.md：本轮冻结前假设、角色、参数和继续条件；numerical_amendment.md：数值修订。
- run_stability.py、analyze.py、test_observation.py；运行命令见 COMMANDS.md。
- runs/：40 次新学生训练；reference_seed0/：20 次统一实现参考；engineering/unobserved_R1/：记录开关控制。
- paired_results.csv：30 对；cell_summary.csv：10 格 mean/std 及历史 seed 0 差值。
- selected_diagnostics.csv、fixed_epoch_results.csv、trajectory.csv：选中模型、160 个固定轮次、840 行轨迹。
- summary.json、output_audit.json、实验结论.md：汇总、200 检查点核验、完整中文解释。

## 新结论

新增 seed 1 平均 +0.363958 点，9 胜 1 负；seed 2 平均 +0.246946 点，8 胜 2 负。
新增20对平均 +0.305452，17胜3负，最差 -0.214777；十格的新两种子均值9正1平。
预设三条件全部通过：两新增种子各自均值为正，至少6格均值为正。可继续机制验证，不等于显著性或完整协议稳定性。
统一数值实现后三学生种子平均 +0.296753 点。CoraFull、Coauthor-CS四格的配对方向较一致。
Cora0.6新两seed平均持平；CiteSeer0.6/Amazon0.4平均很小且有负值。Cora/CiteSeer0.4的原seed0退化未在新seed重复。

## 机制与边界

固定20/50/100/200轮的平均精度差为 -0.019335/+0.193102/+0.259252/+0.085163 点，固定轮次只做事后诊断。
同轮区域KL在20/50/200轮全部20对下降，100轮16对下降；最佳检查点区域KL仅15对下降，残差能量均值略升。
不能把不同选模轮次的残差差异直接视为因果证据，不能按测试结果改为选第100轮。
第50轮19/20对的节点/区域梯度余弦为负；区域梯度范数中位数约节点项0.740倍。这说明目标之间存在优化折中，不直接证明冲突有害。
这仍是新的区域蒸馏机制，未使用原 GB-LCSS 标签投票软目标。

## 重要数值差异

新数值实现的 seed0均值 +0.279356，旧为 +0.322982。R0十格全部逐元素回放一致。
R1三格测试精度变化：CoraFull0.4 +0.086611 点，Amazon0.4 -0.392157 点，Amazon0.6 -0.130719 点；其余七格不变。
原结果保留；不把新旧数值实现的种子混合，差异不计为算法改进。

## 下一窗口

先读实验结论和计划。本轮结果支持进入相同 GRACE 空间、匹配区域数的 k-means 对照，共三学生seed；需保持同一矩阵聚合数值实现。
本轮没有自动启动该阶段。强蒸馏对照、不同教师、完整独立协议种子仍未验证。
