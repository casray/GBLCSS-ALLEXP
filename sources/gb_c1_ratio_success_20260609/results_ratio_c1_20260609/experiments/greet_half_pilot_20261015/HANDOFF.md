# GREET/HF-proxy执行接续，2026-10-15

## 最终状态更新，2026-10-15 10:01

本首轮已全部结束：54目标协议、324目标、243交叉拟合模型审计通过，A/B门槛均未通过，实际0次学生训练，无待续队列。三图旧规则GREET均1球；带噪Brier相对GRACE旧64分别Roman+.050218、CoraFull+.143025、CS+.382379。HF-proxy相对原GB也退化，不等于HALF完整方法被否定。完整证据见audit.json、decision.json与实验结论.md。下方“当前运行”为当时历史，不得据此重启。

用户随后明确授权修改分球重实验，独立入口../greet_adaptive_ball_20261015/HANDOFF.md。复用本轮冻结H，固定64与sqrt/支持16同规则比较两空间；不修改本目录冻结源或配置，不叠HF。

用户已明确批准设计并开始。独立目录本文件所在处；PLAN/config冻结，旧GB-LCSS默认源码未改。与共享目录的BFS/Context-GB其他对话分开，不接管或停止其队列。

## 当前运行

- prepare_greet.py在本轮exec session37895运行三图顺序GPU预训练。Roman400epoch完成；CoraFull正在训练；CS后续。实际状态以greet_raw.log、representations/*/progress.json及summary.json为准。
- 隐藏后台run_pipeline.py --stage all --wait-pretrained已启动，PID190412，详见queue_launch.json。它等三个预训练summary再执行A→B→门槛→条件性学生/融合→审计报告。正常等待为pipeline_status.json中的waiting_for_pretrained。
- 不重复启动prepare或run_pipeline。先检查已知PID190412/日志；部分完成用SHA和config匹配的artifact续跑。执行源码已复制source_snapshot并冻结，修复若需改源码须明确新版本，不绕过完整性检查。
- 共享项目另一BFS方向也记录了正在训练学生；不停止其任务。当前耗时不用于算法效率优劣比较。只维持本方向串行GPU队列。

## 实现和限定

GREET-portable-MC总256维，400epoch固定终点；16步结构编码用1024轨迹Monte Carlo返回率，是显式近似。低/高通跨通道对比按公开forward，导出两路拼接；原rank权重及负集合分母与正文差别记录。2026-10-15纠正此前把导出连接误写成训练连接的笔记。

HF-proxy是简化机制，不是HALF复现；原属性、SX、X−SX三路，每训练集合三折，30epoch固定终点（留10/20诊断轨迹）。只有折外观测标签进入拟合。原q固定，比较小损失/单频/固定融合/局部自适应/同平均alpha常数/打乱预算。真实污染标记只冻结后诊断，clean AUC为null。

三图×三协议seed×三噪声=27单元；A/B共54目标协议、预计324目标文件、243辅助模型。学生和融合由PLAN工程门槛决定，不按test选。一个SSLseed0不称独立预训练重复；原GRACE预算不同不做同计算预算SOTA主张。

## 已经取得的阶段结果

12单元检查通过，包括稀疏/稠密值与梯度、训练通道、导出、随机游走、留一、平票、门控和预算；GPU两轮合成小图和9个CPU辅助模型通过。preflight_validation.json不代替完整审计。

Roman A已单独在CPU完成seed0/1/2。旧分球规则下GREET仅1球，GRACE64；匹配KM/kNN因此都变成所有其他训练节点投票。目标Brier新−旧：clean−.044457±.000875、Uniform+.056514±.001720、Pair+.043923±.000477，均是目标指标，不是分类accuracy。

首次2-means平均半径改善GREET3.857%未过原5%门槛，GRACE约30.469%。这支持空间/分裂准则不兼容，不能写成表示坍缩或GREET整体无效。Roman阶段性机制结果.md与roman_interim.json存详细记录。继续固定剩余队列，不临时降低split阈值。

## 完成后

先检查pipeline_status、audit.json、decision.json、实验结论.md与原CSV；若无完整审计，不声称整个队列完成。向用户详细报告各图/各噪声/配对mean/std、强对照、机制与分类差别、负结果和继续/修改/暂缓决定。最终同步本文件和根历史入口。
