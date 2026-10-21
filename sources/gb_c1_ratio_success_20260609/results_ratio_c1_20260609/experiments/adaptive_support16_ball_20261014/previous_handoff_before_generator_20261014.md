# 交接：2026-10-14 OFA引用核查与原GB-LCSS异配基线

当前没有后台训练待续。用户希望先把原算法放到几个异配图测试，再依结果逐层修改；graph-free部署未锁死。新算法OFA尚未实现或测试。

## 下一窗口先读

- research/algorithm_redesign_20261013/06_OFA引用核查与先基线后改进.md，以及三篇新核读卡HeterGCL_OFA/HOG_GCN_Wang2022/CCA_SSG_OFA_theory。
- experiments/heterophily_original_baseline_20261014/PLAN.md、实验结论.md、decision.json、audit/report.json、student_results/summary/paired_deltas.csv。
- experiments/heterophily_mechanism_20261014/实验结论.md、pipeline_status.json、audit/report.json：上一阶段已完整完成，不要重启队列。

## 已完成

原机制队列60单元/840行、48次seed0学生pilot，20个协议运行审计通过。GRACE Roman1000轮、GraphACL可移植Roman扩展50轮均完成；后者维度/预处理/预算不同，非官方指标复现。

最新原算法基线：Texas/Wisconsin/Roman，五官方split、clean/Uniform0.4/Pair0.4，CE/原GB64/票数匹配KNN64，135次200轮学生运行完成；源清单、冻结目标、原GB64公式、保存logits准确率和完整组合覆盖核查通过。Texas/Wisconsin新增目标10协议运行审核，seed0深度重建；Roman目标复用上一阶段已审计记录。两适配器测试通过，另19项原机制测试此前通过。

## 结果与边界

GB64−CE五split平均增量pp：Texas clean−9.73/uniform+7.03/pair≈0；Wisconsin−2.75/+2.75/+7.45；Roman+0.09/+0.02/+1.09。Roman Pair5/5改善，其余条件需结合完整配对std/同向率。Texas/Wisconsin干净损害大，不支持无条件迁移共识；WebKB小测试集一节点对应2.70/1.96pp。每图仅一个SSL seed，不能称五次独立预训练，也不能与文献不同protocol直接排名。

实际小图球数Texas均4、Wisconsin5/5/5/4/5；Roman GRACE64。原min_child32/分裂准则是待诊断瓶颈。OFA引用链[42]HeterGCL、[64]HOG-GCN、[65]CCA-SSG；Mlp原属性是合理架构动机，不能称独立创新或普遍优于GNN。互信息解释有归一化/分布/增强假设，不证明带噪分类必改善。

ESS×JS旧机制未过门槛，暂缓训练；不要无新信号重复调温度。当前下一步先固定球/软标签/学生，比较GRACE、原X和标签无关OFA空间；然后定位分裂或修正问题，一次改一层。本轮未做GCN分类对照、未做OFA消融、未测试Amazon-ratings。

## 实现故障与来源

旧Roman加权一次申请82.5GiB已通过diagnostic_v2内存分块解决，旧script/结果保留，逐比特小图回归及旧关系hash复核通过。CPU/GPU kNN边界差异曾被严格指纹捕获，保留初始失败和portability记录；当前审计用原GPU。队列GBK日志已改UTF-8，失败历史保存，不需重试已完成阶段。

运行环境D:/codex_temp/noisygl-rocm721/Scripts/python.exe，torch2.9.1+ROCm7.2.1，RX7700XT。正式GB-LCSS代码和已投稿论文未改；新脚本、数据、训练、核读、记忆均在可写workspace。旧中途交接归档在新基线目录previous_handoff_midrun_20261014.md。
