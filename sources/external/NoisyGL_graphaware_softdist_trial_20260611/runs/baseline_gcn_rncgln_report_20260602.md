# Baseline Experiment Report: NoisyGL GCN vs RNCGLN

## 研究假设

先不继续 GAGB/GB-GCL 改法，优先建立一个图标签噪声任务中流程完整、可复现、结果较强的基础 baseline。若 RNCGLN 在 NoisyGL 统一协议下相对 GCN 在中高噪声条件稳定占优，则后续可把 RNCGLN 作为主要鲁棒 baseline，并把 GAGB 类方法只作为后续可插拔模块探索。

## 粒球扮演的角色

本轮实验不使用粒球、GAGB 或任何 GB-GCL 模块。粒球角色为暂缓引入，避免在 baseline 尚未站稳时混入新的机制变量。

## 方法动机

- GCN 作为 NoisyGL 内置普通监督基线，用于判断噪声鲁棒方法是否真正超过普通训练。
- RNCGLN 作为 NoisyGL 内置图标签噪声鲁棒方法，用于建立第一版强 baseline。
- RTGNN、UnionNet、NRGNN、TSS、CRGNN 做过 Cora uniform 0.2 单 seed 筛选；RNCGLN 是单 seed 中最有希望的鲁棒候选。

## 代码改动

- `single_exp.py`: predictor 改为懒加载，避免未运行的方法因缺少 `torch_sparse`/`torch_cluster` 阻塞。
- `total_exp.py`: predictor 改为懒加载；`np.in1d` 替换为 `np.isin` 以兼容当前 NumPy。
- `utils/labelnoise.py`: 修复 independent label noise 中数组标签写入标量位置的问题，并删除一个无参数调用 instance-noise 的残留语句。
- `utils/tools.py`: `np.in1d` 替换为 `np.isin`。

上述改动用于让 NoisyGL 在当前 GPU 环境下可运行，不改变 GCN/RNCGLN 模型结构、训练协议或数据划分。

## 配置改动

未修改 `config/gcn/*.yaml` 或 `config/rncgln/*.yaml`。使用 NoisyGL 默认 Planetoid split：

- Cora: train 140, val 500, test 1000
- CiteSeer: train 120, val 500, test 1000
- seeds: 3000, 3001, 3002, 3003, 3004
- device: `cuda:0`
- environment: `torch 2.9.1+rocm7.2.1`, `AMD Radeon RX 7700 XT`
- NoisyGL commit: `b6ac057fe8c45a1475a32dba823971372186e445`

## 预期机制

RNCGLN 应在较高噪声率下通过鲁棒标签/图结构处理减轻错误监督的影响；低噪声条件下可能因额外机制带来方差或收益不明显。

## 成功标准

- baseline 能在 GPU 上完整跑通。
- 每个条件至少 5 seeds，保留 raw log 和 mean/std。
- RNCGLN 在多数噪声条件下优于 GCN，尤其 40% 噪声。

## 失败标准

- 方法无法在 GPU 环境跑通。
- 与 GCN 相比没有稳定优势。
- 结果方差过大，无法作为强 baseline 主张。

## 实验结果

Accuracy 为 test accuracy mean/std，5 runs。

| Dataset | Noise | Rate | GCN | RNCGLN | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Cora | uniform | 0.2 | 0.7620 ± 0.0082 | 0.7600 ± 0.0161 | RNCGLN 不占优 |
| Cora | uniform | 0.4 | 0.6920 ± 0.0083 | 0.7532 ± 0.0185 | RNCGLN 明显占优 |
| Cora | pair | 0.2 | 0.7318 ± 0.0176 | 0.7366 ± 0.0348 | 小幅占优但方差较大 |
| Cora | pair | 0.4 | 0.5610 ± 0.0339 | 0.6062 ± 0.0729 | 占优但方差较大 |
| CiteSeer | uniform | 0.2 | 0.6262 ± 0.0342 | 0.6734 ± 0.0377 | RNCGLN 占优 |
| CiteSeer | uniform | 0.4 | 0.5024 ± 0.0630 | 0.6020 ± 0.1179 | 占优但方差很大 |
| CiteSeer | pair | 0.2 | 0.6088 ± 0.0249 | 0.6636 ± 0.0418 | RNCGLN 占优 |
| CiteSeer | pair | 0.4 | 0.4216 ± 0.0284 | 0.4174 ± 0.0506 | RNCGLN 不占优 |

## 结果文件

- Combined summary: `runs/baseline_gcn_rncgln_cora_citeseer_summary_20260602.csv`
- Cora uniform raw logs: `runs/baseline_5seed_cora_uniform0.2_20260602/`, `runs/baseline_5seed_cora_uniform0.4_20260602/`
- CiteSeer uniform total log: `runs/baseline_total_citeseer_uniform_20260602/`
- Pair total log: `runs/baseline_total_pair_20260602/`
- NoisyGL generated summaries: `log/2026-06-02_17-12-11.*`, `log/2026-06-02_17-18-50.*`

## 最终决策

继续使用 NoisyGL 作为基础实验框架。RNCGLN 可作为当前主要鲁棒 baseline，但必须同时报告 GCN，因为 RNCGLN 并非所有条件都强于 GCN。后续若要做自己的方法，优先在 Cora/CiteSeer 的 40% uniform、Cora 40% pair、CiteSeer 20% uniform/pair 上验证；不要用 Cora uniform 0.2 或 CiteSeer pair 0.4 单独包装优势。
