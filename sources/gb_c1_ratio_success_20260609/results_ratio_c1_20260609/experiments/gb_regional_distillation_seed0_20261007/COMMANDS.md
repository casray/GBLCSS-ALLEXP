# 运行命令

工作目录为 `results_ratio_c1_20260609`。使用原 GPU Python，不安装依赖。

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m unittest discover -s 'experiments/gb_regional_distillation_seed0_20261007' -p 'test_regional_loss.py' -v
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u 'experiments/gb_regional_distillation_seed0_20261007/run_regional.py' --prepare
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u 'experiments/gb_regional_distillation_seed0_20261007/run_regional.py' --fit --dataset cora --rate 0.4 --arm R0
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u 'experiments/gb_regional_distillation_seed0_20261007/run_regional.py' --grid
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u 'experiments/gb_regional_distillation_seed0_20261007/analyze.py'
```

prepare 冻结十格教师、分组与配置；fit 回放普通蒸馏；grid 顺序完成剩余 49 组；analyze 恢复检查点核验并生成报告。
已有成功 run 不覆盖；prepare 已存在冻结清单时拒绝重写。训练时间包含 200 轮和最终最优检查点概率计算，预处理及分析另计。
本轮显存只记录学生运行峰值；原教师训练的显存不能与该列直接比较。
