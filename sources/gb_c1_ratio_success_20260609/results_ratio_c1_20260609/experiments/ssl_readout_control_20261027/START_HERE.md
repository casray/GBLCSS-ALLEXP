# GraphECL统一读出实验入口

状态：2026-10-27 22:47完成；64/64新头、4/4原生复现、128评分完整，失败0。详细解释见REPORT_zh.md。

本批保持GraphECL/R2权重、原始球成员与目标不变。测试seed0/1，CS与Photo，clean/U.4/U.6/Pair.4，CE/SUPPORT。新增PCA64与归一化R2读出。

- 代码与配置：当前目录。
- 汇总与原始分数：results/。
- 独立复核：review/check_readout.py与check_readout.json。
- 模型、特征、逐轮日志、来源绑定：D:/codex_temp/ssl_readout_control_20261027/。
- 下一步候选：相同初始化下的原始/单位化R2读出2×2确认；本批完成后没有自动追加训练。

## 只读查看状态

从项目根目录执行：

```powershell
Get-Content -LiteralPath 'D:\codex_temp\ssl_readout_control_20261027\queue_status.json'
Get-Content -LiteralPath 'D:\codex_temp\ssl_readout_control_20261027\formal\final.json'
```

## 复核结果

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -B experiments/ssl_readout_control_20261027/review/check_readout.py
```

## 数学与任务矩阵检查

工作目录设为本实验代码目录：

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -B -m pytest test_operations.py -q -p no:cacheprovider --rootdir=. --confcutdir=.
```

## 原始启动入口

`launch.py`持有单队列锁；未完成运行依次执行smoke和all，完成标记存在时直接退出。旧产物或代码哈希不一致会停止，保留现场。

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -B experiments/ssl_readout_control_20261027/launch.py
```

运行目录的receipt.json冻结497项来源；features_locked.json冻结特征；formal/fits_locked.json先于测试评分保存。后续新方案应创建独立目录，保留本批配置与结果。
