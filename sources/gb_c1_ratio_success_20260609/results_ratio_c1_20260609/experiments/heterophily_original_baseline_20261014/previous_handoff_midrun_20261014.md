# 交接：2026-10-14异配机制诊断与训练

用户已明确授权：设计实验，先做机制诊断，然后开始实验。先前“没有获批新训练”的状态已经失效；旧交接保存在 experiments/heterophily_mechanism_20261014/previous_handoff_20261012.md。

## 第一读取顺序

1. experiments/heterophily_mechanism_20261014/pipeline_status.json 与 pipeline_raw.log：当前是否在运行、失败或完成，不能重复启动。
2. 同目录 PLAN.md（冻结）、阶段分析与后续决策.md（首批旧图解释）、实验结论.md（最近一次自动汇总）。
3. audit/report.json 的覆盖时间与范围、student_results.csv、target_metrics.csv，以及 representation_sources.md。

## 已完成且核查过

- CoraFull/Coauthor-CS，冻结原GRACE，每图协议seed0–4，clean/Uniform0.4/Pair0.4：30单元、420行目标诊断。
- 原始10个协议运行通过来源/划分/噪声/冻结目标审计；seed0两图额外候选重建及原公式对照。
- 两图seed0、三条件、CE/GB64/GB128/匹配KNN，24次200轮MLP训练完成；只是pilot，没有五seed准确率稳定性证据。
- Roman-empire GraphACL可移植扩展50轮完成，H及模型保存。不能称官方Roman复现；与GRACE的维度/预处理/预算不同。
- 首批18测试通过，新增内存分块回归后19测试通过。

## 首批机制决策

CoraFull GB128比GB64平均Brier改善0.02559；Coauthor-CS退化0.00718。两图GB128均弱于匹配KNN和KM。ESS×JS相对平均alpha匹配收缩也未改善，暂缓作为新方法，不训练该权重。球内几何加权有弱改善但仍落后KNN。MLP准确率不能由目标Brier直接推出：Coauthor-CS Pair下GB128为86.38%、KNN85.58%，目标Brier排序却相反。

## 当前顺序任务

run_remaining.py依次完成Roman GraphACL诊断（五协议seed）和学生pilot，然后GRACE1000轮及相同诊断/学生，再审计和汇总。GPU串行。执行状态以pipeline_status.json为准，不能用本交接猜测是否已完成；已结束聊天不等于进程完成。

当前恢复执行的exec session为25040；记录时Roman GraphACL seed0已完成目标诊断，seed1进行中。seed0两个预算都只生成13球，最大训练球6577/11331（58.04%），说明实际分裂先停止。此为中途观察，不据此改本轮参数。跟进时先读取状态/PID和日志，不要直接重启。

首次队列因控制台GBK编码中断，恢复后明确发现大球加权一次申请82.5GiB。已修复UTF-8日志；新增diagnostic_v2.py分块计算相同权重，保留原diagnostic.py及旧源清单。小图分块数据与原版逐比特相同。恢复命令为同环境 `python -X utf8 experiments/heterophily_mechanism_20261014/run_remaining.py --resume`，只可在状态failed且确认无存活进程时运行；状态running时不能运行。失败尝试嵌套保存在prior_attempt，完整日志追加。

CPU重建kNN曾与GPU浮点边界排序不同；原GPU严格复现。保留audit/report_initial_cpu.json和cpu_cuda_portability.json；审计默认原CUDA设备，不能用放宽hash容差掩盖差异。

运行环境：D:/codex_temp/noisygl-rocm721/Scripts/python.exe，PyTorch2.9.1+ROCm7.2.1，RX7700XT。新工作均在可写workspace，未改已投稿论文和正式算法代码。

## 接续工作

若队列完成：检查所有阶段退出码、最终审计、Roman两个表示的机制及学生结果，更新阶段分析。若失败：先定位实际异常并保留失败记录，再最小修复。禁止基于单seed测试分数追加参数搜索；禁止把五协议seed说成五次独立SSL。Graph-free并非用户强制约束，本轮MLP是为了隔离监督机制。
