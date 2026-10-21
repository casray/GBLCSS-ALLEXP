# 粒球投影训练视图：机制资格赛

协议与解释门槛见 [PLAN.md](PLAN.md)。这轮只检验训练传播投影、dropout相关性、GCE互补和分区依赖，不恢复R3，不启动LLP，不调粒度。

范围是CiteSeer的u0.6/clean和CoraFull的u0.4/clean，15臂、3个seed，共180次正式运行。所有主模型在原图P上验证和测试。短轮次冒烟另存，不能进入正式均值。

## 文件

- `operators.py`：可微分球内平均与四种逐层传播算子、独立/共享dropout、CE/GCE/标签平滑。
- `experiment.py`：变体、作业清单、源码指纹、同seed配对统计。
- `prepare_data.py`：校验旧缓存，只读复用R0粒球，重建相同坐标做球数匹配k-means。
- `run_job.py`：独立GPU进程；原图选模；保存checkpoint、预测、合法训练目标和指标。
- `run_queue.py`：串行队列、配置/源码冻结、工件校验和安全续跑。
- `test_*.py`、`VALIDATION_TESTS.txt`：27项核验，包括真实旧GCN、多步Adam、非行随机P和GPU小图。
- `analysis/audit_predictions.py`：从原始标签和保存预测重算指标，校验split/noise/targets/初始参数配对。它不参与训练。

## 运行位置与复现

独立冻结源码和运行根目录：`D:/codex_temp/Newidea_operator_view_20261020`。`source/legacy`与`source/support`包含只读依赖快照，源清单在`SNAPSHOT_MANIFEST.json`。

正式命令：

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' 'D:\codex_temp\Newidea_operator_view_20261020\source\run_queue.py' --config 'D:\codex_temp\Newidea_operator_view_20261020\source\config.json' --run-root 'D:\codex_temp\Newidea_operator_view_20261020\formal'
```

进度查看`formal/status.json`、`formal/PROGRESS.md`；逐次日志在`formal/logs`，原始结果和预测在`formal/results`。不能在源文件已改变时直接续跑旧目录。

成本口径区分实际进程墙钟、历史方法所需预处理的重建成本、集中准备/完整性校验成本；本轮不以效率为主张。Mixup是传播后特征混合适配，不是官方图Mixup复现。k-means对照同时改变分区及由该分区生成的B目标，不能只归因于某一种语义性质。

本文件记录执行规范，不代表180次运行已完成或假设已被支持。实际启动、冒烟审计及运行进度另存启动记录。
