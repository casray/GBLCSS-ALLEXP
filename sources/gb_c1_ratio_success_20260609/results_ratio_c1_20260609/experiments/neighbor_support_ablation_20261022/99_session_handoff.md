# 本轮接续

用户已授权并要求继续完成本轮实验。入口为 PLAN.md、实验设计说明.md 和 WORK_LOG.md。
仅新增本目录，旧实验与投稿文件保持原样。

当前已完成：576结果单元，216复用360新拟合，另24短GPU冒烟；无运行中队列。
18测试、288目标、9几何、576检查点重载/选模/别名、288冻结R2、1333源输入哈希通过。
另从results.csv独立复算32行配对宏均值/种子标准差一致。

环境 D:/codex_temp/noisygl-rocm721/Scripts/python.exe，RX7700XT。
不要再次 prepare/freeze，frozen_inputs.json 已绑定 smoke_audit.json。
审计最终入口 analysis_recovery_v2.py full，audit.json.passed=true；原analyze.py
归一化容差过紧的失败保留，不重跑原入口。读取综合结论与接续.md、RESULT_TABLES.md、
paired_macro.csv和decision.json即可复盘，不需要再次训练。

R0带噪OLD44.580/NP45.931/SUPPORT45.935/COUNT46.010/KMNP46.533/KNN46.851；
R2对应59.306/60.525/60.509/60.426/61.143/60.843%。
NP−SUPPORT为−.004±.080/+.016±.127pp；NP−COUNT −.080/+.100pp，全部未过.5pp门。
NP−旧版+1.351±.727/+1.220±.392，两个骨干均3/3seed宏差正；
R0修复门通过，但Cora U.4三seed均负，均值−.985pp；R2 clean Cora−1.917、CF−1.131失败。
NP对KMNP两路线三seed宏差均负。保留R0返修修复，停止扩展完整NP，不宣称32最优或因果分摊。

本轮固定 k32，不扩参数，不叠 small-loss，不自动修改投稿稿件。
第一次准备因旧 CE target_hash 字段语义停止，已在训练前修正；原失败记录保留。
正式训练无失败；首轮归一化审计因float32约3.58e-7误差停止，恢复仅改数值检查，
不改变预测、选模或科学门槛。原冻结源和两份工程失败记录均保留。
