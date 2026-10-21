# GREET自适应粒球接续

## 最终状态，2026-10-15 10:56

已全部完成，run.py退出0；297目标/297学生，18分区/27标签协议/297logits独立审计通过，30项历史基线准确率全一致，无待续队列。入口综合结论与接续.md、实验结论.md、audit.json、decision.json。

新GREET三图均从1球恢复到64及151/141/136根号球，最小训练支持≥16。CoraFull新GREET−原GRACE（clean/U/Pair）+3.72/+4.51/+3.08pp；相同新规则GREET−GRACE +1.28/+1.72/+1.62pp，但落后匹配kNN 1.41/1.83/1.91pp。Roman同规则U/Pair仅+.188/+.053pp而clean−.847pp；CS相对原GRACE Pair−1.016pp。决策modify：保留Cora正信号，不宣称通用异配成功或粒球必要性，不按测试集继续扫阈值/球数。

下方启动说明保留历史，不能据此重复启动。每空间一个SSLseed；portable MC/不等预训练预算限制继续保留。

用户2026-10-15授权修改粒球规则、生成合适数量后重新实验。独立目录，不能重启旧greet_half_pilot已完成队列或改其冻结源。

本轮固定同规则比较GRACE/GREET：最大SSE优先、2means/投影fallback、每球16训练支持；ADAPT64与ceil(sqrtN)双预算，保留旧规则、同数量KM、逐节点同票数kNN及CE。三图三协议seed三条件共297目标/学生。全部学生训练，不套旧首轮目标门槛。无新预训练、无HF融合。

正式启动2026-10-15 10:26:29，exec session53827。状态以status.json/raw.log与实际result.json为准，不以本文运行字样重启。命令run.py --device cuda；先全部分区/目标，再297个200轮学生，最后audit_report.py独立审计与报告。已通过5新增协议测试+11已有生成器测试；preflight_validation.json记录。

执行source_manifest覆盖代码/PLAN/config/数据/H/历史协议。冻结后不修改这些文件；审计与报表若出错，应保留失败记录再写独立修复器，不能掩盖算法修改。

当前规则只是预定候选，不证明数量最优或GREET表示优。最终实验结论.md、decision.json、audit.json完成后更新本接续状态及根索引。每图单SSLseed，三协议seed不等于多SSL；GREET portable MC与旧GRACE预算不同，不能做公平预训练排名。
