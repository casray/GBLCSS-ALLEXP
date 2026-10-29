# 当前状态：已完成，勿重启训练

2026-10-28：58个SSL端点与492个分类头训练完成，汇总舍入检查修复后通过完整GPU精确重载和独立评分审计。首先阅读REPORT_zh.md，results/保存完整结果。

五图GB100固定目标相对ECL+SUPPORT测试−.023pp，重新构球−.065pp；预登记推进门未过。CS重新构球+.342pp、Photo−.728pp。当前决策为暂缓新增训练，优先离线诊断CS/Photo分化。

原训练源码与原预测保留。修复代码review/postfit_20261028/analyze_repaired.py；独立核查review/check_final_scores.py。status.json=FINISHED_BUDGET_REVISION，formal/final.json已保存。下文为历史运行接续说明，当前无需执行启动命令。

---

# GB-ContextECL 当前接续：算力精简版

用户2026-10-28要求减少算力。当前使用budget/plan.json、budget/PLAN_zh.md和budget/COST_zh.md；原84/720方案已调整为58/492，原文与代码保留用于追溯。避免启动旧launch.py或run.py all。

五主图、seed0/1、三噪声保持。五图ECL/GB100/Cos100/Shuffle100为核心对照；前三图另保留小剂量三个分支。CS/Photo小剂量与Roman延后。CS/Photo完整底座复用冻结缓存，另重放前半程生成共享中点；三个续训分支与分类头保持原轮数。

状态：D:/codex_temp/graphecl_gb_context_20261028/status.json。预算启动器PID31260，worker29740，以实际状态为准；已有进程时勿重复启动。日志budget_revision/stdout.log、stderr.log；暂停前39端点记录budget_revision/prior_completed.json。重新扫描已完成产物时completed计数会暂时从1递增，产物仍在原处。

只读状态命令（项目根）：

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -B experiments/graphecl_gb_context_20261028/run.py status
```

确认进程退出并处理错误后，使用预算入口恢复：

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -B experiments/graphecl_gb_context_20261028/budget/launch_budget.py
```

原570项来源和预算源码分别冻结；数学与工程13测试、真实中点/下一步更新和Photo缓存检查通过。所有保留拟合完成后才统一评分：formal/fits_locked.json、formal/final.json、analysis_budget/REPORT_zh.md及表格。五图主门仅用于GB100；剂量证据限前三图，Roman延后。clean无否决门，无R2/KNN/KMeans/D1C1新增实验。

完成后更新完整结论、独立核读与项目历史。历史工程失败分别保留根engineering_attempt1和budget_revision/engineering_attempt1。
