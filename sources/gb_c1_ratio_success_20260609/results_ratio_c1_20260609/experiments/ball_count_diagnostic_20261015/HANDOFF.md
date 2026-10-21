# 六图嵌套粒球数量诊断接续

## 最终状态：2026-10-15，已完成

**没有待续训练，不要重启run.py。** 1026目标/学生记录、819独立学生、207等价别名；35项完整测试通过。最终audit.json passed：36套分区、288快照、54标签协议、1026目标/结果、81历史锚一致。status completed；18组图完成，人工全尺寸检查U/Pair accuracy与clean support，并以contact sheet检查18组整体布局。最终验证见final_validation.json。

完整解释见综合结论与接续.md，逐预算数表见粒球数量响应表.md。Cora U同GREET64→终点40.10±.31→42.44±.38%，CS Pair86.45±.98→84.64±1.06%；Actor无稳定抗噪收益，Chameleon波动大，Squirrel带噪一球更好，Roman目标改善未转化明显分类增益。不支持统一sqrt或扩大B。GREET正SSE但损害Brier的分裂步骤比例约17%–36%。

split_diagnostic通过108单元/15537分裂条件记录/216冻结端点检查；补充局部归一化相关，oracle只属当前树与原投票族的目标参考。validation_budget_diagnostic已完成108单元，测试分数不进选择；仍为事后普通调参而非新方法结果。算法重设计建议.md为完整候选：无标签候选树＋训练内独立证据剪枝，**尚未实现或训练**。

原run退出1只因全部训练后的CSV数值字符串审计兼容；初版状态已保存initial_postprocess。独立finish_diagnostic完整数值审计通过，图版式/配对均值修复后plots-only退出0。保留冻结源和全部模型/目标，修复有单独manifest及回归测试。final_validation保存源SHA重核。根02/99只更新本方向，其他方向保留。

以下为运行期间历史，均被以上最终状态覆盖。

## 2026-10-15 14:45接续状态

run.py session11290仍运行，917/1026行，CS seed1 clean；前五图含Cora全部结束，勿重跑。新六SSL session67676已退出0。正式全量源冻结，任何新增分析脚本不修改冻结run/config/PLAN/旧audit/旧plot。

新增后处理/诊断已完成：finish_diagnostic.py（数值字符串兼容）＋test_postprocess绿；plot_curves_reviewed.py（仅版式、规整刻度、support log轴）替代输出，保留原plot源；32项独立测试已分别验证（组合30绿，另val选择2绿），4快照集成绿。合成布局检查在plot_smoke_synthetic，明确不是实验数据，原版拥挤已修，parent已看新版；实际图仍待完整结果后检查。

split_diagnostic.py已运行通过：108树×噪声条件、15537分裂条目、216冻结端点检查通过。目录split_diagnostic里有per_split_utility/tree_summary/summary/manifest。真值仅事后诊断和oracle，不训练学生。GREET有害SSE分裂均值约17%–36%；相对SSE收益与Brier收益相关弱。oracle是当前树+全训练投票原LCSS的Brier参考下限，不是新方法或accuracy上界。

validation_budget_diagnostic.py及test_budget_selection已写、测试2绿，**尚未运行**（检查必须全1026行）。只按val_accuracy选预算，ties实际count小优先再requested小；test不进选择。是事后诊断、普通超参选择而非算法创新。全矩阵后运行保存108选择单元，不训练模型。

算法重设计建议.md已给具体候选：固定无标签树、查询Q/选择S/投票V三角色，S再分A/D评分，完整子树DP剪枝，独立V投票及两角色交换平均，delta .1/V支持16固定候选。**未实现或训练**，理论/样本减少/Pair边界都已说明。最终需根据完整CS曲线补综合结论，不自动额外训练。

队列最终会在旧CSV审计格式处失败（已提前复现），训练不受影响。待run退出且student CSV1026行，用finish_diagnostic.py保存initial_postprocess原失败记录再完整审计/绘图；若有其他实质错误必须定位，不能放宽容差。然后跑val诊断、看真正figures图（不能用synthetic图）、写六图详细结论/数量表/decision/final_validation、更新根02/99，仅本GREET段。

用户2026-10-15授权增加异配数据、暂不做独立kNN/kmeans对照、诊断不同球数再设计生成器；随后“继续”保持本任务。PLAN/config已冻结，不重复征询确认、不重启旧已完成GREET/HF/BFS等方向。

新增官方Actor/Chameleon-filtered/Squirrel-filtered，pin yandex-research/heterophilous-graphs a431395582e929d88271309716bea4fe24ce6318，原split0/1/2保持；新六SSL已完成，GRACE1000/GREET400，seed0固定末轮、无标签probe。旧Roman/CoraFull/CS H复用。各图一个SSLseed，不是三个独立预训练。dataset_manifest及embedding_source_manifest可核。

新granular_ball_trajectory.py固定minleaf16/support16，单线程2means（多线程对称点不确定已记录）、最大SSE优先、投影fallback；同树保存1/8/16/32/64/128/256+ceil√N快照。超过可行数不强凑。旧64与CE锚保留，不做独立kNN/KM；GREET内部原20近邻不改。19臂×6图×3seed×3条件=1026目标/学生记录。相同float32训练目标在同case复用canonical模型，alias不能当独立重复。

run.py --device cuda已启动，exec session11290；实际状态看status.json/raw.log，不因本文运行字样重复启动。先36套分区/54标签协议/1026目标，再全部学生，最后审计和绘图。27项单元检查通过；4快照独立集成核验通过。临时smoke命令初版dict重复num_balls出错已用dict union修正，不是生产代码错误，无产物影响。

## 已知审计兼容性修复

源码冻结后额外检查发现audit_report.compare_metric对keep_default_na=False产生的数字字符串执行np.isfinite会报TypeError。已用真实CSV复现；与构球/训练无关。不能改冻结源码或source_manifest。新增finish_diagnostic.py只在原训练队列退出后使用数字标量转换包装器重新运行完整独立审计，test_postprocess.py回归通过；保留原status/audit/报告到initial_postprocess并写postprocess_adapter_manifest，不改模型或目标。

原run.py预计在所有训练完成后的首轮审计报上述格式错误，不能把failed误认为缺少学生或直接重训。先核student_results.csv是否1026行，再运行：

`& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' experiments/ball_count_diagnostic_20261015/finish_diagnostic.py`

该后处理若有其他实质审计错误，必须查原因，不能静默降低容差。plot_curves.py输出accuracy/macroF1/Brier/support/margin/SSE的三条件共18组PNG/SVG/PDF及灰度预览，需parent视觉复核。可达数量重复点合并，最多3seed，std非CI。

## 完成后必须继续

1. 完整审计、核历史81锚、核canonical与alias计数；保存最终验证和命令。
2. 看曲线并详细解释六图/两噪声/seed波动、目标与分类分离、过细支持与cap平台；不能把test峰值当部署最优球数。
3. 写综合结论与接续、算法重设计建议（输入/局部停止/伪代码/理论边界），区别已证/解释/未证；用户要求的“从而重新设计”不能只交CSV。
4. 同步根02历史和99的本GREET段，保留共享其他方向。无用户新授权不扫阈值、不自动新预训练或叠HF。
