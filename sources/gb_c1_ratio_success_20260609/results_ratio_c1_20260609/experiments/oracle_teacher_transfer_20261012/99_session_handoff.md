# Oracle 分解完成交接，2026-10-12

用户已授权的实验、分析和独立核验全部完成，无运行中的任务。已投稿论文没有整理或修改。

## 固定实验

三数据集Cora/CiteSeer/CoraFull，uniform .4/.6，协议/教师/学生seed1。
A含噪全训练集；B完整纠正原训练标签；C仅原本未污染的训练监督；
D原训练集内随机干净子集，与C逐真实类别数量相同。
所有节点/边保留。ERASE训练预处理、原型和分类探针均使用干预后的监督mask。
训练对象只保留监督标签和干净验证标签，test y为-1；所有测试指标等全部选模后统一算。
B两档噪声实际输入相同，只训练一次并复用。21个不同教师和21个学生，24逻辑单元。
DIST原始X两层64MLP，200轮，T1/beta1/gamma1，没有粒球项和噪声CE。

## 工程修订与已完成检查

最初计划复用6个历史A；Cora教师回放的概率最大差.00997847，argmax全部相同、bestepoch32相同。
旧函数相同seed两次3epoch训练也产生2.205e-6概率差，图、mask、预处理完全一致；
具体不确定性GPU内核未定位。故在读取任何新测试成绩前改为六个当轮A，主对照不复用旧概率。
原失败日志和冻结manifest保留，engineering_revision.json记录改动；科学干预和门槛未变。
DIST回放与历史概率逐元素相同，200轮验证轨迹相同，replay_check.json通过。
6个单测通过，只有上游弃用警告。preflight类别均覆盖，C与D类别计数一致。

## 历史运行命令

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/oracle_teacher_transfer_20261012/run_oracle.py --replay --grid
```

该命令的session11548已exit0，正式训练1029.57秒。禁止重复启动覆盖已有结果。
raw_log.txt记录START/DONE，runs中console/epochs含验证轨迹，不含测试成绩。
原先session93389因回放检查退出，日志保留；不是正式科学失败结果。
预算90分钟，已在预算内结束；没有自动扩种子或调参。

## 结果、核验与下一窗口

analyze.py与audit_results.py均已exit0；优先读实验结论.md、decision.json和完整CSV。
A教师/学生均值67.8359/68.4121；B72.0346/72.5388；C70.4561/70.4857；D70.2030/70.3177。
B-A教师+4.1987pp、学生+4.1267pp，均6/6正；C-A+2.6201/+2.0736pp。
B-C教师+1.5786pp、学生+2.0530pp；CoraFull .6学生+4.6048pp。
C-D仅+.2530/+.1681pp，方向混合。机会门槛通过，精度转移警告未触发。
Bstudent精度均值比teacher高.5041pp，但NLL/Brier仍弱，错误集合不同，不能称完整知识转移。

24逻辑单元重载teachermaxdiff1.788e-7、argmax全同，student逐元素相同；监督/初始化/选模检查通过。
独立测试精度重算差0、C/D类别计数匹配、旧输入及训练源码散列未变。
当轮vs旧A最大教师/学生精度差.2577/.7388pp，数值参考保留在historical_baseline_comparison.csv。
中文报告、历史43与项目上下文已更新。下一步优先A/B/C多种子确认，再检验无需真值的监督恢复信号。
没有自动启动确认或新方法，粒球不是新方向的必要前提。
单seed筛查，不声称稳定/显著，B跨噪声复用不是独立重复，干净教师不是理论天花板。
