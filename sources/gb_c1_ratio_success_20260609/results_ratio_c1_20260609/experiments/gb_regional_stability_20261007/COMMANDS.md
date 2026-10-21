# 运行命令

工作目录：results_ratio_c1_20260609。使用既有 ROCm Python 和 RX 7700 XT。

~~~powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m unittest discover -s 'experiments/gb_regional_stability_20261007' -p 'test_observation.py' -v
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u 'experiments/gb_regional_stability_20261007/run_stability.py' prepare
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u 'experiments/gb_regional_stability_20261007/run_stability.py' replay
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u 'experiments/gb_regional_stability_20261007/run_stability.py' grid
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u 'experiments/gb_regional_stability_20261007/analyze.py'
~~~

prepare 在执行前冻结 v2 计划、源码、环境和原缓存引用。已冻结时拒绝覆盖。
replay 比较新数值实现下记录开关的完整训练；R0 与原结果严格回放。
grid 运行正式 40 次新学生 seed 1/2 配对实验，并补齐 20 次统一数值实现的 seed 0 参考。
analyze 恢复正式 40 次训练的 200 个检查点，并核验指标、选模轮次、配对初始化、随机状态和全部原输入。
已完成 fit 跳过，不覆盖。教师、划分、噪声、粒球、损失数学定义及 lambda 不变。

## 回放失败的诊断

首次 index_add 版本在 R1 严格回放时失败，记录保留于 replay/。以下命令证明旧训练器自身也存在逐轮数值差异：

~~~powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u 'experiments/gb_regional_stability_20261007/diagnose_replay.py'
~~~

诊断后切换固定权重矩阵乘法，细节见 numerical_amendment.md。该修订在正式 seed 1/2 训练开始前完成。
全部工程尝试保留，未选择性覆盖测试结果。运行耗时包含诊断和模型存盘，不用于论文效率比较。
