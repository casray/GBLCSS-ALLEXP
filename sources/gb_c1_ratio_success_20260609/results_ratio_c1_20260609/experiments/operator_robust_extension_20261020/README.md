# TFR插件与固定投影剂量扩展

用户批准后，于2026-10-20 12:14:39北京时间启动。正式队列共192次，结果未齐；本文件不是方法成功报告。

## 研究问题

在保留原图选模与推理的前提下，几何投影训练视图是否能给完整TFR带来互补收益？固定中间剂量是否能减轻硬投影的信息损失？不更改粒度、不添加B平均标签目标、不扫描rho。

协议、正负条件与停止标准见`PLAN.md`和`config.json`。

- TFR三臂：P独立dropout、P共享dropout、S共享dropout。原DBLP配置及完整解码器/重构/伪标签/动态系数保留；8条件×3seed×3臂=72次。
- CE五臂：P独立、P共享、rho=.5共享、S共享、k-means分区S共享。全部用自身含噪标签；8条件×3seed×5臂=120次。
- 每5轮及首末轮记录训练视图与原图P诊断，分别报告错标拟合、真标签恢复和未翻转节点表现。原图验证决定主检查点；诊断不干预优化器、模式或随机流。

## 实现与验证

`tfr_core.py`包装作者GCN、保留原TFR目标；`run_job.py`实现逐层剂量混合、两族训练和诊断；`prepare_data.py`校验原始边、属性、标签、P和匹配分区；`experiment.py/run_queue.py`负责192清单、配对、独立进程、冻结与安全恢复；`audit_results.py`核查预测/监督和轨迹工件。`analysis/check_native_tfr.py`以旧完整函数作短程GPU核验。

启动前证据：21项CPU/GPU测试通过，16/16独立冒烟及预测/监督/轨迹审计通过。CoraFull三TFR臂短程峰值显存约2.84–3.49GiB。CiteSeer u0.6两轮完整原TFR与新原生worker对照：loss最大差0，概率最大差1.1921e-7、参数差2.8312e-7，均选中第2轮；验证率的1.7594e-8差来自统计精度。它是短程实现核验，不是500轮独立复现或正式效能证据。

第一次冒烟在训练前失败：复制元数据使用系统默认编码，中文路径失真。失败目录保留，改为显式UTF-8且检查元数据相等后，新`smoke_v2`通过。冻结训练源码未因该问题改变。

## 路径与命令

独立运行根：`D:/codex_temp/Newidea_operator_robust_20261020`。训练只使用其`source`冻结包，包含`robust/legacy/support/prior`依赖；原始数据和旧缓存只读。

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' 'D:\codex_temp\Newidea_operator_robust_20261020\source\run_queue.py' --config 'D:\codex_temp\Newidea_operator_robust_20261020\source\config.json' --run-root 'D:\codex_temp\Newidea_operator_robust_20261020\formal'
```

`formal/status.json`和`formal/PROGRESS.md`为实时进度；`results`保存逐次JSON与预测；`traces`保存逐轮汇总诊断；`checkpoints`保存主检查点及TFR解码器状态；`logs`为原始日志。轨迹哈希、分母和比例恒等式可审计，但未保存每轮全部节点预测，不能将轨迹统计称为独立重训复现。

启动源码SHA：`f8a32d33c1c53c92fddab77e3bf300599af061a7c06a6ba2caa6b28516c651c4`；配置SHA：`a697df96f70feaaade5c4f207398fb9b60b367bcec02b904225274e287cc8592`。

TFR是本项目raw属性和共同split/noise上的协议适配，不冒称作者原协议最优成绩。时间记录区分进程、诊断与历史预处理，不以效率为主卖点。
