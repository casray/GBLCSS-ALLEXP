# 属性与结构互补：已完成接续

2026-10-15完成。无待续训练，勿重新启动prepare/run_workers/run_students/finish_queue。入口为综合结论与接续.md和实验结论.md。

## 完成证据

- 四图Amazon-ratings/Tolokers/Roman/CS，五协议seed、三条件、七臂420目标；基础五臂300新学生完成。
- 四图R目标gate均未通过，R/SHUFFLE学生未启动，不能写为分类失败。38项单元测试及420目标/300预测/420配对比较/953文件哈希独立审计通过。
- 两新GRACE各1000轮完成；每图仅一SSL seed。Tolokers二分类主指标ROC-AUC、clean/sym.2/sym.4；其他图Accuracy、clean/U.4/Pair.4。
- Amazon BFS相对OLD64三条件+1.107/+1.019/+1.293pp，均5/5正；但clean相对CE−1.209pp。Roman X Pair相对CE+1.182pp、相对OLD64仅+.268pp且3/5正。Tol无稳定AUC收益；CS Pair X/BFS相对OLD64−2.442/−5.955pp。
- 决策：保留Amazon局部BFS信号；暂缓统一BFS和固定残差候选，不事后扫权重，不统一sqrt默认。先验可靠性仅下一步假设，尚未实现。Minesweeper本轮未运行。

## 指纹与边界

final_validation.json状态passed，target_rows420/student_rows300。

- audit/report.json SHA256：02c8e44a13656f51345b12a6fc05a698d19c21ef6ff0cb5d589fffafa5388cfe
- 实验结论.md：a6992da6340bfa97c18402b393b4d72cc2e5dade7137c81acc1381dfb8938c7e
- decision.json：5340d0d4615efe3bfc5ed3117c86857a2d49177f3285968b46e05cdbd76c8ebe
- finish_report.py：dea151dc4f01ba76d7fff18ad2abdd2046ad7230bfaf3a36f278166604bc50ae

原算法默认入口与原稿未改；OLD64当前环境按原规则重构、学生全部新训，不要求与历史成绩逐字节一致。所有冻结source/data/H/targets/预测保留。EXECUTION.md与cpu_reschedule.json解释已完成的CPU独立seed重调度；不按历史运行状态重启串行队列。根02/99其他聊天分支保留，不接管GREET/球数/Context。
