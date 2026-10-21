# PubMed Baseline Extension Report

## Research Hypothesis

先把图标签噪声任务里的基础 baseline 跑稳定，再开始设计自己的 GB-GCL / GAGB-free 方法。PubMed 作为大图补充数据集，用来检查 Cora/CiteSeer 上较强的 RNCGLN 是否还能作为跨数据集强 baseline。

## Granular Ball Role

本轮实验不引入粒球、GAGB 或 GB-GCL 模块。粒球角色暂缓，避免在 baseline 尚未稳定前混入新机制变量。

## Motivation

- Cora/CiteSeer 上 RNCGLN 在多数组合中强于 GCN，但不是全条件占优。
- PubMed 节点数明显更大，需要验证 RNCGLN 是否仍可在当前 12GB ROCm GPU 上完整运行。
- 若 RNCGLN 不可运行或不稳定，需要找一个 NoisyGL 内部可复现、流程完整、结果不弱的 PubMed baseline。

## Code Changes

沿用前一阶段 NoisyGL 兼容性修复：

- `single_exp.py`: predictor lazy import，避免未运行方法因缺少可选依赖阻塞。
- `total_exp.py`: predictor lazy import，`np.in1d` 改为 `np.isin`。
- `utils/labelnoise.py`: 修复 independent label noise 的 scalar 写入 bug，并移除残留无参调用。
- `utils/tools.py`: `np.in1d` 改为 `np.isin`。

本轮 PubMed 扩展没有修改模型结构、训练协议、split 或 YAML 配置。

## Config

- Framework: NoisyGL commit `b6ac057fe8c45a1475a32dba823971372186e445`
- Environment: `torch 2.9.1+rocm7.2.1`
- GPU: `AMD Radeon RX 7700 XT`
- Device: `cuda:0`
- Dataset: PubMed, default NoisyGL Planetoid split
- Split: train 60, validation 500, test 1000
- Seeds: `3000,3001,3002,3003,3004`
- Noise: `uniform` and `pair`
- Noise rates: `0.2` and `0.4`

## Commands

```powershell
$p='C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe'
& $p total_exp.py --runs 5 --methods gcn --datasets pubmed --noise_type uniform pair --noise_rate 0.2 0.4 --device cuda:0 --seed 3000
& $p total_exp.py --runs 5 --methods gcn rncgln --datasets pubmed --noise_type uniform pair --noise_rate 0.2 0.4 --device cuda:0 --seed 3000
& $p total_exp.py --runs 5 --methods unionnet --datasets pubmed --noise_type uniform pair --noise_rate 0.2 0.4 --device cuda:0 --seed 3000
& $p total_exp.py --runs 5 --methods forward backward --datasets pubmed --noise_type uniform pair --noise_rate 0.2 0.4 --device cuda:0 --seed 3000
```

Candidate single-seed screening used:

```powershell
& $p single_exp.py --method <method> --dataset pubmed --noise_type uniform --noise_rate 0.4 --device cuda:0 --seed 3000
```

Screened methods: `crgnn`, `nrgnn`, `unionnet`, `coteaching`, `jocor`, `sce`, `forward`, `backward`, `cp`, `r2lp`, `clnode`.

## Expected Mechanism

- RNCGLN: 通过鲁棒标签/图结构处理降低错误监督影响。
- UnionNet / Forward / Backward: 作为 NoisyGL 内部可复现的噪声鲁棒训练 baseline，检查是否能在 PubMed 上比普通 GCN 更稳。

## Success Criteria

- 每个保留 baseline 至少 5 seeds。
- raw log、mean/std、seed、dataset split 和 config 可追溯。
- 候选方法应在多个 PubMed 噪声设置下明显超过 GCN，且提升不能只落在 std 波动内。

## Failure Criteria

- 方法在当前 GPU 上不能完整运行。
- 方法只在单种子或单条件上超过 GCN。
- 提升小于或接近标准差，不能作为强 baseline 主张。
- 训练出现 NaN 或运行时错误。

## Results

Accuracy is test accuracy mean/std over 5 seeds.

| Dataset | Noise | Rate | GCN | UnionNet | Forward | Backward | RNCGLN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PubMed | uniform | 0.2 | 0.7036 +/- 0.0435 | 0.7058 +/- 0.0492 | 0.7042 +/- 0.0523 | 0.7050 +/- 0.0533 | OOM |
| PubMed | uniform | 0.4 | 0.5684 +/- 0.0489 | 0.5664 +/- 0.0430 | 0.5734 +/- 0.0449 | 0.5728 +/- 0.0449 | not run after OOM |
| PubMed | pair | 0.2 | 0.7266 +/- 0.0224 | 0.7350 +/- 0.0217 | 0.7290 +/- 0.0333 | 0.7298 +/- 0.0310 | not run after OOM |
| PubMed | pair | 0.4 | 0.5772 +/- 0.0757 | 0.5706 +/- 0.0585 | 0.5770 +/- 0.0555 | 0.5762 +/- 0.0573 | not run after OOM |

Single-seed PubMed uniform 0.4 screening:

| Method | Seed | Test Acc | Status |
| --- | ---: | ---: | --- |
| unionnet | 3000 | 0.6240 | ok, but 5-seed mean did not stay above GCN |
| forward | 3000 | 0.5820 | ok, small 5-seed gain within std |
| backward | 3000 | 0.5820 | ok, small 5-seed gain within std |
| coteaching | 3000 | 0.5260 | train loss reached NaN |
| nrgnn | 3000 | 0.5020 | train loss NaN throughout late training |
| clnode | 3000 | 0.4970 | weak |
| crgnn | 3000 | 0.4940 | weak |
| sce | 3000 | 0.4910 | weak |
| jocor | 3000 | 0.4520 | weak |
| cp | 3000 | n/a | failed: Node2Vec requires `pyg-lib` or `torch-cluster` |
| r2lp | 3000 | n/a | failed: storage size calculation overflow |

## RNCGLN PubMed OOM Review

RNCGLN failed on PubMed before completing the first robust run. The log reports:

```text
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 1.45 GiB
```

The direct cause is the dense PubMed node-pair path in `predictor/module/RNCGLN.py`, where `x @ x.T` materializes an `N x N` matrix. For PubMed, `N=19717`, so one dense matrix is already about 1.45GB in float32; training needs multiple tensors and gradients. Current 12GB GPU cannot support this default implementation. I did not downscale hidden dimensions, alter sparse flags, or change the method silently, because that would make the baseline protocol no longer comparable to the default NoisyGL setting.

## Raw Logs

- GCN PubMed 5-seed: `runs/baseline_total_pubmed_gcn_20260602/total_exp_pubmed_gcn_uniform_pair_0.2_0.4_runs5_cuda0.log`
- RNCGLN PubMed OOM: `runs/baseline_total_pubmed_20260602/total_exp_pubmed_gcn_rncgln_uniform_pair_0.2_0.4_runs5_cuda0.log`
- UnionNet PubMed 5-seed: `runs/baseline_total_pubmed_unionnet_20260602/total_exp_pubmed_unionnet_uniform_pair_0.2_0.4_runs5_cuda0.log`
- Forward/Backward PubMed 5-seed: `runs/baseline_total_pubmed_forward_backward_20260602/total_exp_pubmed_forward_backward_uniform_pair_0.2_0.4_runs5_cuda0.log`
- Candidate single-seed logs: `runs/baseline_pubmed_candidate_single_20260602/`
- Combined table: `runs/baseline_all_three_datasets_summary_20260602.csv`

## Final Decision

继续使用 NoisyGL 作为基础实验框架。Cora/CiteSeer 上保留 GCN 和 RNCGLN 两个核心 baseline；PubMed 上保留 GCN 作为主 baseline，并把 UnionNet / Forward / Backward 作为“已筛查但不稳定强于 GCN”的辅助记录，不把它们包装成强 baseline。

下一步可以开始设计自己的方法，但第一版方法不要以 PubMed 上击败 RNCGLN 为目标，因为 RNCGLN 默认实现无法在当前 GPU 上跑通 PubMed。更合理的门槛是：

- Cora/CiteSeer: 同协议下同时对比 GCN 和 RNCGLN。
- PubMed: 同协议下至少对比 GCN，并报告 UnionNet / Forward / Backward 的已筛查边界。
- 任何提升必须看 5-seed mean/std，不能拿单种子或低于 std 的差距做论文结论。
