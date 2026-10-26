# 本轮已完成：节点路由直接对照全局融合

入口 REPORT_zh.md、PLAN.md、decision.json、final_audit.json。用户 2026-10-23
授权直接检验节点路由，独立于上一轮 L1 门槛停止。5 图四条件 seed0，缓存 A300/C_head；
神经训练 0，620 路由拟合槽位，审计另外重拟合620槽位，均无常量回退。

外层5折/内层4折验证，GLOBAL 与 ROUTER 同外折；最终全验证5折OOF选路由阈值后重拟合。
8个学生自身信号，L2 Logistic C1，无参数搜索。旧验证也参与学生checkpoint选择，属探索。
20组最终规则全部保存后才读取test真值评分。

预注册验证门通过：ROUTER−GLOBAL 带噪 Cora/Cite/CF/CS/Photo 为
+.864/+.904/+.185/-.200/+.044pp，clean均满足−1pp下限。
测试同差为−.229/−1.017/−.137/−.231/−.093pp，五图均负；相对最好单通道
−.035/+.773/+.060/+.036/+.349pp，原描述性+1.5pp目标前三图全未满足。
故保留通过的验证门，另记测试精度未体现优势。暂缓复杂化，图/GB特征未跑，未训练单学生蒸馏。

概率平均在CiteSeer65.178%，GLOBAL64.977%，ROUTER63.960%；Cora概率路由AUC仅.558。
Photo AUC.796，但净收益仅+.349pp，体现排序能力和阈值净纠错需分别评估。
Cora/Photo路由macroF1较GLOBAL +.935/+.616pp，其余三图负。

审计5tests、620modelreplay、20foldpredictions、140metricrows、20旧GLOBAL权重、9源90输入通过。
首次运行末尾sources.json默认编码读取中文路径失败；run.py只改UTF8读取，未重训选择模型。
原sources.json/failure.txt保留，finalize.py还原一行核对原源码摘要并完成独立审计。
勿重跑run.py；complete.json已存在。报告末尾人工结论不要由旧report函数覆盖。
无需启动E1/E2/E3/E5、无需稿件修改或上传。
