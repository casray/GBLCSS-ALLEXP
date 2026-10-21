# 修复后的E0机制分解与精简E2

先读[执行前计划](PLAN_V2.md)。本目录是新实现；旧e0_cost/是保留问题的历史冒烟。首次运行必须先通过CPU测试与独立GPU连通检查。

CPU检查（本目录中执行）：

```powershell
python -m unittest discover -s . -p 'test_*.py' -v
```

正式队列（配置冻结后，不修改源码或该配置；已有结果仅在哈希一致时续跑）：

```powershell
python run_suite.py --config config_v2.json --run-root D:/codex_temp/Newidea_budget_gb_v2_20261019/formal
```

队列包含1080个独立训练作业：1032个机制/成本/预算敏感性作业，48个新增E2作业；E2的R0直接复用主实验C。同一时刻只有一个GPU训练子进程。每臂1800秒超时；出错即停止，保留日志与状态，修复后若源码有变化须新建运行目录。

观察：运行目录的status.json、PROGRESS.md、runs.csv、summary.csv、paired_deltas.csv。原始结果与主/次预测在results/，日志在logs/。source/config/data/split/noise/assignment/prediction均可追溯。不把配置、源或工件不匹配的结果当成完成。

配置的data_root指向原机器公共图存储；Git仓库不包含数据。准备过程只使用属性/边构造候选树，监督目标和切面评分仅使用指定训练mask。O2仅选择切面时读取干净训练标签，下游目标仍含噪。

成本主读数是必需准备阶段与完整worker墙钟的重建单次总成本，实测准备/worker墙钟分别保留；摊销固定K=1/3/10/30，标签依赖选切面不跨噪声/划分摊销。三seed是筛查，不提供统计显著或等价结论。

确认图WikiCS已锁定，本轮不运行。正式结论要等任务完成并检查所有种子、失败记录和对照条件。
