# 执行调度补充

2026-10-15。算法、PLAN、config、source_manifest及随机种子全部不变。CS高维空间单seed三类属性构球耗时约20分钟；串行进程在seed0分区/全部目标落盘并核manifest后停止，仅未完成seed重做。cpu_reschedule.json保存精确进程、命令、保存边界及原因。

新增run_workers.py只调度互不依赖的seed1–4，以两个CPU进程调用冻结run.partitions/targets，单线程KMeans保持不变。每个进程独占seed目录，父进程汇总；不同时运行原CS串行队列。固定view目录已准备并核hash，避免两个进程重复构造或写同一个特征文件。worker_source_manifest另存调度器来源。

新图目标脚本仅在各自H已完成后运行，与CS CPU调度并行；不同图只写各自目录。最终在所有进程完成后无条件从原JSON重建完整CSV/gate，独立审计420精确键，防止中间并发汇总暂时不完整。GPU表示/学生队列仍串行。

精度核查history_precision_check.json复现了GPU/CPU float32 mean最多1ULP差。本轮仅history accuracy与最终CPU均值比较使用1.2e-7绝对容差；best epoch首次最大值仍精确，AUC和最终CPU指标比较保持1e-12。配对差与科研判据均从原学生JSON重建，并独立核派生CSV。
