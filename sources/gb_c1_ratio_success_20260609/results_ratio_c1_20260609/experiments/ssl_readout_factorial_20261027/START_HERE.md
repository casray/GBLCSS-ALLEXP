# C批：R2尺度×初始化实验入口

状态：2026-10-27 23:18:52完成。64/64新头、16/16对角复现、128评分完整；失败0，独立复核通过。

结论：CS单位化噪声收益在两套初值下重复，Photo clean损失也重复。CS噪声门4/4，Photo0/4；包含clean约束的完整门0/8。详见REPORT_zh.md。

- 设计：PLAN.md与config.json。
- 完整表与效应：results/summary.json、factorial_summary.json、scores.csv、gates.json。
- 独立验证：review/check_factorial.py及JSON；clean训练日志诊断：review/clean_training_diagnostics.json。
- 冻结特征、初始权重、模型、目标和原始训练日志：D:/codex_temp/ssl_readout_factorial_20261027/。
- 后续粒球SSL方案尚未启动，本批自动流程已结束。

从项目根目录复核：

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -B experiments/ssl_readout_factorial_20261027/review/check_factorial.py
Get-Content -LiteralPath 'D:\codex_temp\ssl_readout_factorial_20261027\queue_status.json'
```

在本代码目录运行单元测试：

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -B -m pytest test_contract.py -q -p no:cacheprovider --rootdir=. --confcutdir=.
```

原始启动入口（完成标记已存在时直接退出）：

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -B experiments/ssl_readout_factorial_20261027/launch.py
```

新实验应另建目录，保留791项来源哈希、初始化锁、拟合锁和本批预登记门。后续优先在GraphECL原生属性分支固定读出后测试一个弱粒球关系项，R2尺度仅作机制对照。
