# 证据剪枝实验接续

## 最终状态：2026-10-15 已完成，无待续训练

正式run session91756退出0，status completed（18:38:05）。324新目标/结果，286实际新训练、38别名，另216历史锚，共540预测记录。最终audit passed：54协议/1944组件/540预测/216历史锚/1045输入源SHA；14测试通过。后处理session22361退出0，六组PNG/SVG/PDF及324区域诊断/1134机制差值/324切分容量计数完成。入口综合结论与接续.md及final_validation.json。

**决策defer，暂停当前具体剪枝准则，不替换默认。** 主PRUNE六图目标/分类信号全部失败；36带噪目标配对相对CF_FINE无一改善。Cora相对B64带噪平均−7.938pp、CS−2.166pp、Roman−.980pp；Squirrel+2.229pp但clean−.730pp且不优于已有一球锚。无惩罚恢复部分Cora性能但仍落后，全票不能挽回。Cora97.22%查询估计选根，同支持条件可行互斥区域均值45.89，PRUNE仅1.06，不能把退化全归样本不足。

CF_FINE有按查询祖先回退，可能重叠；DP为互斥切分，候选族不同。posthoc只计最大可行切分，未额外训练，因此评分损失与候选族限制尚未完全分离。两种折交换不独立，std为协议seed，不是CI。不要把Squirrel相对细球提升称为新局部语义贡献。

初版导入错误在任何正式目标前已修；主运行正常结束。后处理初版自动轴外刻度误报，显式轴域内刻度后通过；只改未冻结后处理脚本，正式冻结源未动。最终图三张全尺寸+六图概览人工检查，其他QA范围见posthoc/visual_qa.json。根02/99只增本方向，其他独立方向保留。

以下为启动时记录，已被上述最终状态覆盖。

用户2026-10-15明确授权设计并开始、可以更改算法；本轮“开始”接续。不得再询问相同授权。当前正式命令：`D:\codex_temp\noisygl-rocm721\Scripts\python.exe experiments/evidence_pruning_20261015/run.py --device cuda`，exec session91756，PID43640。实际状态看status.json/raw.log；不要重复启动。

六图Actor/Chameleon-filtered/Squirrel-filtered/Roman/CoraFull/CS，三协议seed0/1/2、clean/U.4/Pair.4；复用上一轮GREET单SSL空间及B256树。六新臂CF_ROOT/CF_64/CF_FINE/PRUNE/PRUNE_ZERO/PRUNE_ALL，324目标与学生记录，216历史锚另计；相同float32目标显式复用。全部目标正确性gate后完整训练，不按干净Brier选择哪些方法才训练。

14测试通过（含DP穷举2187成本组合、自己标签排除、参考独立投票、执行入口导入）；真实CoraFull Uniform集成审计36组件通过。初次启动在任何freeze/目标之前因set_seed从错误模块导入失败；已定位为gblcss.data.set_seed并完成入口测试red/green，非算法改动。preflight_validation保存诊断。不可修改当前source_manifest中冻结代码、PLAN/config及旧实验输入。

18:22:54全部54case/324新目标完成并独立审计通过；target_gate.json是correctness only，随后学生已启动。预检查PRUNE可能过于保守剪为根，PRUNE_ZERO约23–33区（仅Cora seed0 U例子，不是全量结论），不能据此改delta。

源代码：GB-LCSS_code/src/gblcss/granular_ball_pruning.py；本目录run/common/pruning_audit/final_audit/report.py。生产输出组件/decisions先冻结，再读取真值诊断；旧锚hash/logits只读。final_audit自动重开324目标/决策、540预测记录及所有源哈希，并生成实验结论.md/decision_signals.json。需要人工综合结论解释逐图负结果、clean/Pair和宏F1；预设信号在PLAN。

完整训练结束后：核audit/status与真实独立模型数；查看全部指标并写综合结论与接续/decision/final_validation；必要的后处理新增脚本单独manifest，不改冻结源。更新根02/99只增加本方向，保留其他属性残差/BFS/Context方向。不做独立KM/kNN、不重训SSL、不扫阈值；每空间1SSLseed，三协议seed不是独立预训练。

理论边界：每查询Q不参与自己的选区/投票；S与V分离但交换估计有关联，不能把两估计当独立重复。PRUNE_ALL选区与投票相关，仅是数据利用率经验消融。自适应树与评分风险不保证软目标或分类收益。若失败如实保存，不自动放宽门槛。
