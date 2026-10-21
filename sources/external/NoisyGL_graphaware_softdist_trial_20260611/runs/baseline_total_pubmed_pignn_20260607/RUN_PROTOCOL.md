# PubMed PIGNN Full Baseline Run Protocol

生成日期：2026-06-07

## 研究假设

此前 PubMed 上 `PIGNN` 只完成了 `uniform 0.4 seed 3000` smoke test，test accuracy 为 `0.6600`。本次补跑完整 5-seed 四组合，用于判断 `PIGNN` 是否能进入 PubMed noisy-label baseline 主表。

## 粒球角色

无。本实验是 NoisyGL 外部鲁棒 baseline 补跑，不包含 GB/GAGB 模块，也不改变当前粒球算法。

## 方法动机

PubMed 上 `RNCGLN` 因 dense 相似度矩阵 OOM 不适合作为当前硬件下的完整 baseline；`RTGNN` 和 `CRGNN` 已补跑，但不是全条件稳定最强。`PIGNN` 单 seed smoke 较强，因此需要完整 5-seed 验证。

## 代码改动

无代码改动。使用 `external/NoisyGL/total_exp.py` 原始入口。

## 配置改动

无配置改动。使用 `config/pignn/pignn_pubmed.yaml`：

- hidden: 16
- layers: 2
- dropout: 0.5
- lr: 0.01
- weight decay: 0.0005
- epochs: 400
- start_epoch: 200
- patience: none

## 实验配置

- dataset: `pubmed`
- method: `pignn`
- noise type: `uniform`, `pair`
- noise rate: `0.2`, `0.4`
- runs: `5`
- seeds: `3000, 3001, 3002, 3003, 3004`
- device: `cuda:0`
- Python: `C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`
- framework: PyTorch `2.9.1+rocm7.2.1`, PyG `2.7.0`

## 启动命令

```powershell
cd C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL
$p='C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe'
& $p total_exp.py --runs 5 --methods pignn --datasets pubmed --noise_type uniform pair --noise_rate 0.2 0.4 --device cuda:0 --seed 3000
```

## 运行状态

- 启动时间：2026-06-07 00:25
- 启动方式：后台 `Start-Process`
- 主进程 PID：`13192`
- Python 子进程 PID：`27044`
- stdout log: `total_exp_pubmed_pignn_uniform_pair_0.2_0.4_runs5_cuda0.stdout.log`
- stderr log: `total_exp_pubmed_pignn_uniform_pair_0.2_0.4_runs5_cuda0.stderr.log`
- 备注：Windows venv Python 会显示一个 base Python 子进程；此前直接环境检查确认该 venv 中 `torch.cuda.is_available() == True`。

## 预期机制

如果 `PIGNN` 的图标签噪声建模在 PubMed 上确实有效，它应在至少部分噪声组合中稳定超过 `GCN`，并与已补跑的 `RTGNN` / `CRGNN` 形成可比较的强 baseline 候选。

## 成功标准

- 至少一个高噪声 setting 明显超过 PubMed `GCN`，且不是单 seed 偶然值。
- 若 `uniform 0.4` 5-seed 均值接近或超过此前 smoke 的强趋势，可进入 PubMed 强 baseline 候选。
- 若全条件平均或关键 setting 优于 `RTGNN`/`CRGNN`，可优先作为 PubMed 主鲁棒 baseline。

## 失败标准

- 5-seed 均值回落到 `GCN` 附近或低于 `GCN`。
- 方差过大，提升无法脱离 std 波动。
- 运行失败、显存不足或耗时不可接受。

## 结果

实验已完成。NoisyGL 原始输出已复制到本目录：

- `noisygl_2026-06-07_00-25-25.txt`
- `noisygl_2026-06-07_00-25-25.tex`
- `noisygl_2026-06-07_00-25-25.xlsx`

| Dataset | Method | Noise Type | Noise Rate | Test Acc |
|---|---|---|---:|---:|
| PubMed | PIGNN | uniform | 0.2 | 0.7228 +/- 0.0246 |
| PubMed | PIGNN | uniform | 0.4 | 0.6168 +/- 0.0665 |
| PubMed | PIGNN | pair | 0.2 | 0.7258 +/- 0.0269 |
| PubMed | PIGNN | pair | 0.4 | 0.6206 +/- 0.0276 |

## 最终决策

`PIGNN` 可进入 PubMed strong baseline candidate：

- `uniform 0.4` 明显高于 PubMed `GCN` 的 `0.5684 +/- 0.0489`，与此前 `RTGNN` 的 `0.6214 +/- 0.0514` 接近。
- `pair 0.4` 达到 `0.6206 +/- 0.0276`，高于此前 `CRGNN` 的 `0.5920 +/- 0.0594` 和 `GCN` 的 `0.5772 +/- 0.0757`。
- `uniform 0.2` / `pair 0.2` 与已有 PubMed baseline 接近，没有稳定压倒 GCN/Forward/Backward。

因此 PubMed 后续 baseline 表应保留 `GCN` 作为基础参照，并把 `PIGNN` 作为高噪声强候选，尤其用于 `pair 0.4`；`RTGNN` 仍是 `uniform 0.4` 的强候选。
