# 强方法插件与固定投影剂量实施计划

> For agentic workers: use superpowers:subagent-driven-development. User authorized execution on 2026-10-20 after the 180-run audit.

**Goal:** 在保留TFR原训练流程的前提下检验投影插件，并以一个固定剂量检验抗噪与信息损失折中。

**Architecture:** 原图选模/测试，训练分类器逐层采用M_rho=(1-rho)P+rho Pi P Pi。TFR解码器、伪标签、反馈权重和图增强不改。新源码和运行目录独立冻结；GPU串行、每臂独立进程。

**Tech stack:** 本地已验证Python/Torch ROCm/PyG；原TFR类与ROCm适配器；旧v2只读数据/树/R0缓存。

## 研究与冻结协议

- 假设：S能给完整TFR提供互补收益；固定中间rho=.5可能减少硬投影的信息损失。均待验证。
- 粒球角色：无标签几何投影与dropout共享区域，不额外生成B目标，不修改粒度。
- 方法动机：资格赛中CiteSeer高噪声有算子增量，CoraFull大部分CE损害却已存在于B目标，必须保留自身标签风险。
- 范围：Cora u0.6、CiteSeer u0.4/u0.6、PubMed u0.6、Photo u0.4、CoraFull u0.4，加CiteSeer/CoraFull clean，共8条件；seed0/1/2，原20/10/70划分和噪声映射；实际球数round(.1N)。
- TFR三臂：P+原生dropout，P+共享dropout，S+共享dropout。共72次。原模型DBLP预设：hidden16、dropout .6886554578、Adam lr .1/wd .001、500轮、每轮原图验证、不早停；原解码器/增强/伪标签比例及损失系数全部保留，原本未参与前向的encoder仍保留以匹配参数与RNG。统一raw属性和数据协议属于本项目适配，不称作者原协议最优复现。
- CE机制五臂：P+独立，P+共享，M_.5+共享，S+共享，KM分区S+共享；全部用自身带噪标签CE。共120次。hidden256/dropout .5/Adam .01/wd .0005，最多300轮，每5轮原图验证，patience10次检查。
- 合计192次新正式训练，不混入历史均值。先一个强方法，不自动扩第二方法。
- KM与GB同坐标U/PU/P²U、同实际M；n_init10/max_iter300/lloyd/seed20261019；分区不读标签。
- 逐轮诊断：每5轮与首末轮，分别在训练视图和原图P记录错标节点错误标签拟合、真实标签恢复、未翻转节点准确率及分母。仅诊断读取训练节点干净标签；不反传、不选模、不影响RNG。无错标时比例null。

## 成功/失败与最终决策

- 下一阶段插件筛查：四正信号单元明确为Cora u0.6、CiteSeer u0.4、CiteSeer u0.6、PubMed u0.6；其中至少2个TFR_S_shared−TFR_P_ind mean>=1pp且各至少2/3种子为正；同时TFR_S_shared−TFR_P_shared报告投影增量，不能将仅来自共享dropout的收益归给S。
- 六带噪条件和两个clean锚全部单列，不用总均值掩盖负结果。clean下降>2pp标为不能统一替代。
- 中间剂量仅rho=.5，不扫参数。修复标准：CoraFull clean/u0.4相对硬S改善且相对匹配P_shared损害不超过1pp；同时四正条件至少2个相对匹配P_shared提升>=1pp且2/3同向。只作筛查，不据此直接设计粒度。
- 强插件未过有限筛查，暂停当前规格；若适配核验失败，结论是适配未成立，不能判方法互不相容。不得反复调rho、门控或损失救援。
- 上述2/4为新增阶段门槛，不替换前一PLAN的最终五图三噪声8/15门槛；尚未检验pair噪声和未开发确认数据。
- 结果/结论：192/192完成、最终审计通过。TFR插件0/4，固定剂量4/4且CoraFull修复通过。决定暂停当前强插件/粒度扩展，早停收尾尚未执行。完整结果与解释边界见analysis/final_20261020/REPORT.md；上述冻结门槛未改。

## 文件与执行步骤

- [x] tfr_core.py/test_tfr_core.py：原GCN关闭投影的前向/loss/梯度/三步Adam；偏置、剂量、原生及共享mask；原TFR损失/解码器梯度。
- [x] run_job.py/test_worker.py：两族训练、原图选模、逐轮诊断、预测/checkpoint/trace和分阶段成本。
- [x] experiment.py/test_experiment.py：192清单、配对、哈希和剂量控制。
- [x] prepare_data.py：旧输入哈希、原始边与x/y一致性、P重建、一致分区控制。
- [x] run_queue.py：复用已核验队列，并校验trace工件；暂停/恢复/进程锁。
- [x] CPU/GPU核验、独立短轮次冒烟、最重CoraFull TFR显存与流程检查。21项测试、16次冒烟及完整旧TFR函数短程GPU对照通过。
- [x] 冻结源码/配置/依赖，后台启动。启动时间2026-10-20 12:14:39北京时间；Newidea与启动证据见同目录记录。
- [x] 完成192次后独立重算、全部均值/std与配对、按门槛裁决及方向历史。192次工件、64组汇总及72行配对通过；结果见analysis/final_20261020。

最小核验命令：`D:/codex_temp/noisygl-rocm721/Scripts/python.exe -m unittest discover -s experiments/operator_robust_extension_20261020 -p test_*.py -v`。正式启动绝对路径、配置/源码SHA和进程写入LAUNCH.json。raw log、config、seed、split/noise、mean/std、预测与原始模型保留。
