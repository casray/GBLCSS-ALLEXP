# GraphECL honest pilot：完成并核验

研究结果：两个预登记推进门均未通过；当前规格停止扩展。

- 详细解释：RESULT_INTERPRETATION_zh.md
- ContextECL反事实及门：REPORT_zh.md
- 冻结方案：PLAN.md、config.json（保留执行前原件，执行情况以本文件为准）
- 接续记忆：SESSION_HANDOFF.md
- 结果与独立核查：results/、results/review/audit.json
- 原始权重、预测、日志、分区与逐节点明细：D:/codex_temp/graphecl_honest_pilot_20261029

已完成：机制测试；S1反事实/复现/缓存筛查；S2标签隔离与GPU冒烟；20短训端点和20新头；独立分析；报告与项目历史同步。无待续GPU任务。

仅重算报告：在本目录执行既有Python环境的 `python -B analyze.py`、`python -B review_audit.py`。归档脚本生成基础报告，人工研究判断位于详细解释末尾及SESSION_HANDOFF.md；重新生成时需保留该段。
