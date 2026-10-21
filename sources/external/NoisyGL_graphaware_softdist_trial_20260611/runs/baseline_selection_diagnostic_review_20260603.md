# 图标签噪声 Baseline 选择与跑通审查文档

生成日期：2026-06-03

本文档只整理图标签噪声任务的 baseline 选择、跑通情况和结果边界。不包含 GAGB、不包含粒球模块，也不把粒球机制写入方法说明。

## 1. Context And Objective

当前目标是：在开始设计自己的图标签噪声鲁棒方法之前，先找到一个实验流程完整、代码可运行、结果相对较强的基础 baseline，并确认现有实验代码是否已经达到这个要求。

本轮使用 NoisyGL 作为基础实验框架。选择原因是它包含多种图标签噪声鲁棒训练方法，已有 `single_exp.py` 和 `total_exp.py` 两类入口，能够统一处理数据集、噪声注入、训练、评估、seed 和 mean/std 汇总。

当前结论：

- Cora/CiteSeer 上，`GCN + RNCGLN` 已经能作为第一版完整 baseline 组合。
- PubMed 上，`RNCGLN` 默认实现因 dense `N x N` 相似度矩阵导致 GPU OOM，不能作为当前硬件下可运行 baseline。
- PubMed 上，`GCN` 是基础参照 baseline；`UnionNet`、`Forward`、`Backward` 已筛查并 5-seed 跑通但没有稳定强于 GCN。2026-06-03 又补跑了 `RTGNN` 和 `CRGNN`：`RTGNN` 在 uniform 0.4 上明显强于 GCN，`CRGNN` 在 pair 0.4 上较强，但二者都不是全条件稳定最强。`PIGNN` 已完成 PubMed uniform 0.4 seed 3000 smoke test，5-seed 全组合暂缓。

## 2. Prior Experiment Chain

实验链如下：

1. 停止继续推进 GAGB / GB-GCL 方向，先建立图标签噪声任务的稳定 baseline。
2. 从 GitHub 拉取并修复 NoisyGL，使其能在当前 ROCm GPU 环境运行。
3. 先在 Cora uniform 0.2 单 seed 上筛查多个候选方法：`GCN`、`RTGNN`、`UnionNet`、`NRGNN`、`TSS`、`CRGNN`、`RNCGLN`。
4. 单 seed 筛查中，`RNCGLN` 是最有希望的鲁棒候选，因此对 Cora/CiteSeer 的 `uniform/pair`、`0.2/0.4` 做 5-seed 汇总。
5. PubMed 上尝试复用 `GCN + RNCGLN`，但 `RNCGLN` OOM。
6. PubMed 上补跑 `GCN` 5-seed，并筛查 `UnionNet`、`Forward`、`Backward` 等可运行候选。
7. 根据外部 benchmark 提示，2026-06-03 补跑 PubMed `RTGNN`、`CRGNN` 四个噪声组合 5-seed；`PIGNN` 先做 PubMed uniform 0.4 seed 3000 smoke test，确认能跑通但全量 5-seed 暂缓。
8. PubMed 当前结论从“只有 GCN 主 baseline”更新为：GCN 是基础参照；RTGNN 可作为 uniform 0.4 的强候选；CRGNN 可作为 pair 0.4 的补充候选；PIGNN 待全量补跑后再定。

## 3. Method / Algorithm

本轮没有设计新算法，只建立 baseline。

### GCN

普通监督 GCN baseline。输入为带噪训练标签、图结构和节点特征；训练目标为 NoisyGL 默认配置中的监督分类损失；输出 test accuracy。

作用：

- 判断标签噪声下普通监督训练的基准性能。
- 作为所有鲁棒方法是否真正有效的最低参照。

### RNCGLN

NoisyGL 内置图标签噪声鲁棒方法。此处不重新定义其论文算法，只按仓库默认实现运行。它用于检查鲁棒图标签噪声方法是否能在同一 NoisyGL 协议下稳定超过 GCN。

注意：RNCGLN 在 PubMed 上默认实现会构造 dense `x @ x.T` 级别的节点对矩阵，PubMed `N=19717` 时单个 dense float32 矩阵约 1.45GB，反向传播和中间张量叠加后当前 12GB GPU 无法承载。

### UnionNet / Forward / Backward

这些方法用于 PubMed 候选筛查：

- `UnionNet`: NoisyGL 内置鲁棒方法，单 seed PubMed uniform 0.4 曾达到 0.6240，但 5-seed 后没有稳定超过 GCN。
- `Forward` / `Backward`: 损失修正类方法，PubMed 上可以完整运行，但相对 GCN 只有 std 范围内的小幅波动。

### RTGNN / CRGNN / PIGNN

这些方法是根据 NoisyGL 官方 benchmark 经验在 PubMed 上补充的候选：

- `RTGNN`: PubMed 5-seed 已补完。在 uniform 0.4 上达到 `0.6214 +/- 0.0514`，明显高于 GCN `0.5684 +/- 0.0489`；但 uniform 0.2、pair 0.2、pair 0.4 不占优或波动很大。
- `CRGNN`: PubMed 5-seed 已补完。在 pair 0.4 上达到 `0.5920 +/- 0.0594`，高于 GCN `0.5772 +/- 0.0757`，但 std 重叠；其他设置不稳定或不占优。
- `PIGNN`: PubMed uniform 0.4 seed 3000 smoke test 跑通，test acc `0.6600`，但单 run 耗时长；全量 uniform/pair、0.2/0.4、5-seed 暂缓，不能先写成稳定结论。

## 4. Clean-Label And Data-Leakage Boundary

本轮实验不使用 clean labels 参与训练目标构造。

NoisyGL 流程中：

- train labels 按指定噪声类型和噪声率注入噪声。
- validation/test split 使用默认 Planetoid 划分。
- test labels 只用于最终评估 test accuracy。
- 没有额外使用测试集标签、clean label purity、oracle correction 或人工筛选样本参与训练。

代码修复没有改变数据 split、噪声协议、模型结构或 YAML 默认训练超参。

## 5. Experimental Protocol

代码仓库：

- NoisyGL commit: `b6ac057fe8c45a1475a32dba823971372186e445`
- 本地路径：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL`

运行环境：

- Python: `C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`
- PyTorch: `2.9.1+rocm7.2.1`
- GPU: `AMD Radeon RX 7700 XT`
- Device: `cuda:0`

数据集：

- Cora
- CiteSeer
- PubMed

噪声设置：

- noise type: `uniform`, `pair`
- noise rate: `0.2`, `0.4`

随机种子：

- 5 seeds: `3000, 3001, 3002, 3003, 3004`

PubMed split：

- train 60
- validation 500
- test 1000

主要命令模板：

```powershell
$p='C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe'
& $p total_exp.py --runs 5 --methods gcn rncgln --datasets cora citeseer --noise_type uniform pair --noise_rate 0.2 0.4 --device cuda:0 --seed 3000
& $p total_exp.py --runs 5 --methods gcn --datasets pubmed --noise_type uniform pair --noise_rate 0.2 0.4 --device cuda:0 --seed 3000
& $p total_exp.py --runs 5 --methods unionnet --datasets pubmed --noise_type uniform pair --noise_rate 0.2 0.4 --device cuda:0 --seed 3000
& $p total_exp.py --runs 5 --methods forward backward --datasets pubmed --noise_type uniform pair --noise_rate 0.2 0.4 --device cuda:0 --seed 3000
& $p total_exp.py --runs 5 --methods rtgnn crgnn --datasets pubmed --noise_type uniform pair --noise_rate 0.2 0.4 --device cuda:0 --seed 3000
```

## 6. Baselines And Fair Controls

### 保留 baseline

- `GCN`: 所有数据集都保留，是最基本的监督 baseline。
- `RNCGLN`: Cora/CiteSeer 保留，是当前较强鲁棒 baseline。
- `PubMed GCN`: PubMed 基础参照 baseline，因为 RNCGLN 默认实现不可运行。
- `PubMed RTGNN`: PubMed uniform 0.4 的当前强候选。
- `PubMed CRGNN`: PubMed pair 0.4 的补充候选，但 std 重叠较大。

### Cora/CiteSeer baseline screening

Cora/CiteSeer 的筛查分两步。第一步只在 Cora uniform 0.2 seed 3000 上做候选预筛，目的是快速排除明显弱、训练不稳定或依赖不适配的方法；第二步只把预筛中最有希望、且作为最低参照必须保留的方法扩展到 Cora/CiteSeer 的全噪声组合 5-seed。

第一步：Cora uniform 0.2 seed 3000 单 seed 预筛。

| Method | Dataset | Noise | Rate | Seed | Test Acc | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| rncgln | Cora | uniform | 0.2 | 3000 | 0.7620 | 最强候选，进入 5-seed 主 baseline |
| gcn | Cora | uniform | 0.2 | 3000 | 0.7520 | 基础参照，必须保留 |
| nrgnn | Cora | uniform | 0.2 | 3000 | 0.7470 | 接近 GCN，但 train loss 为 NaN，不作为强 baseline |
| crgnn | Cora | uniform | 0.2 | 3000 | 0.7320 | 弱于 GCN/RNCGLN |
| unionnet | Cora | uniform | 0.2 | 3000 | 0.7260 | 弱于 GCN/RNCGLN |
| rtgnn | Cora | uniform | 0.2 | 3000 | 0.6660 | 明显偏弱 |
| tss | Cora | uniform | 0.2 | 3000 | 0.6630 | 明显偏弱 |

第二步：把 `GCN` 和 `RNCGLN` 扩展到 Cora/CiteSeer 的 uniform/pair、0.2/0.4、5 seeds 汇总。CiteSeer 没有单独做一轮所有候选方法的单 seed 筛查，因为 Cora 预筛已经给出了最有希望的鲁棒候选；为了控制实验成本和避免过度筛选，只对 `GCN` 和 `RNCGLN` 做完整 5-seed 对比。

| Dataset | Noise | Rate | GCN | RNCGLN | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Cora | uniform | 0.2 | 0.7620 +/- 0.0082 | 0.7600 +/- 0.0161 | RNCGLN 不占优 |
| Cora | uniform | 0.4 | 0.6920 +/- 0.0083 | 0.7532 +/- 0.0185 | RNCGLN 明显占优 |
| Cora | pair | 0.2 | 0.7318 +/- 0.0176 | 0.7366 +/- 0.0348 | 小幅占优但 std 较大 |
| Cora | pair | 0.4 | 0.5610 +/- 0.0339 | 0.6062 +/- 0.0729 | 占优但 std 较大 |
| CiteSeer | uniform | 0.2 | 0.6262 +/- 0.0342 | 0.6734 +/- 0.0377 | RNCGLN 占优 |
| CiteSeer | uniform | 0.4 | 0.5024 +/- 0.0630 | 0.6020 +/- 0.1179 | 占优但 std 很大 |
| CiteSeer | pair | 0.2 | 0.6088 +/- 0.0249 | 0.6636 +/- 0.0418 | RNCGLN 占优 |
| CiteSeer | pair | 0.4 | 0.4216 +/- 0.0284 | 0.4174 +/- 0.0506 | RNCGLN 不占优 |

### 已筛查但不作为强 baseline 的方法

下面这个表不是全局 baseline 筛查表，而是 PubMed fallback candidate screening。它的触发条件是：Cora/CiteSeer 上已保留 `GCN + RNCGLN`，但 PubMed 上 `RNCGLN` 默认实现 OOM，所以需要额外筛查是否存在一个可运行且能替代 PubMed GCN 的鲁棒候选。

因此，`GCN` 和 `RNCGLN` 没放进这个候选表：

- `GCN` 是 PubMed 的参照 baseline，不是待筛查的替代候选；它已在主结果表中完整报告 5 seeds。
- `RNCGLN` 是原计划的强鲁棒 baseline，但 PubMed 默认实现 OOM；它的失败原因在 OOM 诊断部分单独记录。

PubMed uniform 0.4 seed 3000 的参照状态：

| Method | Seed | Test Acc | Status |
| --- | ---: | ---: | --- |
| gcn | 3000-3004 | 0.5684 +/- 0.0489 | PubMed 基础参照 baseline，5-seed 已跑通 |
| rncgln | 3000 | n/a | PubMed 默认实现 OOM，未进入候选筛查 |

PubMed uniform 0.4 seed 3000 的替代候选筛查：

| Method | Seed | Test Acc | Status |
| --- | ---: | ---: | --- |
| unionnet | 3000 | 0.6240 | 单 seed 好，但 5-seed 后没有稳定超过 GCN |
| pignn | 3000 | 0.6600 | smoke test 跑通，单 run 耗时长；全量 5-seed 暂缓 |
| rtgnn | 3000 | 0.6980 | smoke test 很强，已补全四组合 5-seed |
| forward | 3000 | 0.5820 | 可运行，5-seed 只在 std 内小幅波动 |
| backward | 3000 | 0.5820 | 可运行，5-seed 只在 std 内小幅波动 |
| coteaching | 3000 | 0.5260 | 训练 loss 出现 NaN |
| nrgnn | 3000 | 0.5020 | 后期 train loss NaN，不可靠 |
| clnode | 3000 | 0.4970 | 弱于 GCN |
| crgnn | 3000 | 0.4940 | 弱于 GCN |
| sce | 3000 | 0.4910 | 弱于 GCN |
| jocor | 3000 | 0.4520 | 弱于 GCN |
| cp | 3000 | n/a | 失败，缺 `pyg-lib` 或 `torch-cluster` |
| r2lp | 3000 | n/a | 失败，storage size calculation overflow |

Cora/CiteSeer 没有放进这张表，是因为这张表只回答 PubMed 上 RNCGLN OOM 后的替代方案问题。Cora/CiteSeer 的筛查路径是另一条：先用 Cora uniform 0.2 单 seed 做候选预筛，再对 `GCN` 和 `RNCGLN` 做 Cora/CiteSeer 全噪声组合 5-seed 汇总。也就是说，Cora/CiteSeer 已经进入主 baseline 对比表；PubMed 由于 RNCGLN OOM 才需要额外 fallback 筛查。

## 7. Metrics And Definitions

主要指标：

- `test accuracy`: NoisyGL predictor 在 test split 上计算的分类准确率。
- `mean +/- std`: 5 个 seeds 的 test accuracy 均值和标准差。

判定规则：

- 单 seed 结果只用于候选筛查，不作为论文结论。
- 如果提升落在 std 波动范围内，只能说“弱信号”或“不稳定”，不能说显著优于 baseline。
- 方法如果训练出现 NaN、依赖缺失、OOM 或运行时错误，不作为当前可运行强 baseline。

## 8. Main Results

### Cora/CiteSeer: GCN vs RNCGLN

| Dataset | Noise | Rate | GCN | RNCGLN | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Cora | uniform | 0.2 | 0.7620 +/- 0.0082 | 0.7600 +/- 0.0161 | RNCGLN 不占优 |
| Cora | uniform | 0.4 | 0.6920 +/- 0.0083 | 0.7532 +/- 0.0185 | RNCGLN 明显占优 |
| Cora | pair | 0.2 | 0.7318 +/- 0.0176 | 0.7366 +/- 0.0348 | 小幅占优但 std 较大 |
| Cora | pair | 0.4 | 0.5610 +/- 0.0339 | 0.6062 +/- 0.0729 | 占优但 std 较大 |
| CiteSeer | uniform | 0.2 | 0.6262 +/- 0.0342 | 0.6734 +/- 0.0377 | RNCGLN 占优 |
| CiteSeer | uniform | 0.4 | 0.5024 +/- 0.0630 | 0.6020 +/- 0.1179 | 占优但 std 很大 |
| CiteSeer | pair | 0.2 | 0.6088 +/- 0.0249 | 0.6636 +/- 0.0418 | RNCGLN 占优 |
| CiteSeer | pair | 0.4 | 0.4216 +/- 0.0284 | 0.4174 +/- 0.0506 | RNCGLN 不占优 |

结论：RNCGLN 在 Cora/CiteSeer 多数组合中优于 GCN，尤其是高噪声或 CiteSeer uniform/pair 0.2，但不是全条件胜出。

### PubMed: GCN and Runnable Candidates

| Dataset | Noise | Rate | GCN | UnionNet | Forward | Backward | RTGNN | CRGNN | RNCGLN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PubMed | uniform | 0.2 | 0.7036 +/- 0.0435 | 0.7058 +/- 0.0492 | 0.7042 +/- 0.0523 | 0.7050 +/- 0.0533 | 0.6890 +/- 0.0469 | 0.6970 +/- 0.0315 | OOM |
| PubMed | uniform | 0.4 | 0.5684 +/- 0.0489 | 0.5664 +/- 0.0430 | 0.5734 +/- 0.0449 | 0.5728 +/- 0.0449 | 0.6214 +/- 0.0514 | 0.5648 +/- 0.0711 | not run after OOM |
| PubMed | pair | 0.2 | 0.7266 +/- 0.0224 | 0.7350 +/- 0.0217 | 0.7290 +/- 0.0333 | 0.7298 +/- 0.0310 | 0.7038 +/- 0.0619 | 0.7264 +/- 0.0305 | not run after OOM |
| PubMed | pair | 0.4 | 0.5772 +/- 0.0757 | 0.5706 +/- 0.0585 | 0.5770 +/- 0.0555 | 0.5762 +/- 0.0573 | 0.5317 +/- 0.1176 | 0.5920 +/- 0.0594 | not run after OOM |

结论：PubMed 上 `UnionNet`、`Forward`、`Backward` 没有稳定强于 GCN。`RTGNN` 在 uniform 0.4 上形成当前最强本地候选，但 pair 噪声下不稳定；`CRGNN` 在 pair 0.4 上略强于 GCN，但 std 重叠较大。`PIGNN` 的 full 5-seed 仍是待补项。

## 9. Diagnostic Results

### 代码兼容性修复

为让 NoisyGL 在当前环境跑通，做了小范围兼容性修复：

- `single_exp.py`: predictor 改为 lazy import，避免未运行方法因缺少可选依赖阻塞。
- `total_exp.py`: predictor 改为 lazy import，避免 unused predictor import 失败；`np.in1d` 改为 `np.isin`。
- `utils/labelnoise.py`: 修复 independent label noise 中 noisy label scalar 写入问题；移除一个残留无参函数调用。
- `utils/tools.py`: `np.in1d` 改为 `np.isin`。

这些修复没有改变模型设计、训练协议或数据划分。

### PubMed RNCGLN OOM

PubMed RNCGLN 日志显示：

```text
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 1.45 GiB.
```

诊断：

- PubMed 有 19717 个节点。
- RNCGLN 默认实现中存在 `x @ x.T` 级别的 dense 节点对矩阵。
- 单个 dense float32 矩阵约 1.45GB，训练时还需要梯度和多个中间张量。
- 当前 GPU 总显存约 11.98GB，默认实现不可运行。

决策：

- 不静默缩小 hidden dim、删模块或改变 sparse flag。
- 不把改过配置的 RNCGLN 当默认 baseline。
- PubMed 不以击败 RNCGLN 为第一版方法门槛。

### PubMed RTGNN/CRGNN raw log

RTGNN/CRGNN 的 PubMed 四组合 5-seed 结果来自：

```text
runs/baseline_total_pubmed_rtgnn_crgnn_20260603/total_exp_pubmed_rtgnn_crgnn_uniform_pair_0.2_0.4_runs5_cuda0.log
```

PIGNN 当前只完成 smoke test，日志为：

```text
runs/baseline_pubmed_pignn_rtgnn_crgnn_smoke_20260603/pignn_pubmed_uniform0.4_seed3000_cuda0.log
```

## 10. Supported Conclusions

现有实验和代码已经达到以下要求：

1. 已跑通一个图标签噪声任务的完整 baseline 流程：数据加载、噪声注入、GPU 训练、评估、5-seed mean/std 汇总、raw log 保留。
2. Cora/CiteSeer 上已有较强 baseline：`RNCGLN` 在 6/8 个设置中相对 GCN 有优势，但有两个设置不占优。
3. PubMed 上已有可运行 baseline：`GCN`、`UnionNet`、`Forward`、`Backward`、`RTGNN`、`CRGNN` 均完成四个噪声设置 5-seed。
4. PubMed 上 `RTGNN` 在 uniform 0.4 上明显强于 GCN；`CRGNN` 在 pair 0.4 上略强于 GCN，但 std 重叠较大。
5. `PIGNN` 已完成 PubMed uniform 0.4 seed 3000 smoke test，结果为 0.6600；由于全量 5-seed 暂缓，不能作为稳定结论。
6. 当前 baseline 表足够支撑开始设计第一版自己的方法，但论文表述必须按数据集和噪声类型区分 baseline 强度。

## 11. Unsupported Or Still-Hypothetical Claims

以下说法目前不被支持：

- “RNCGLN 是所有数据集、所有噪声设置下的最强 baseline。”
- “PubMed 上已有一个方法在所有噪声设置下稳定明显优于 GCN。”
- “UnionNet 单 seed PubMed uniform 0.4 的 0.6240 可以代表稳定优势。”
- “后续方法只要超过 PubMed GCN 一点点就说明显著有效。”尤其在 PubMed uniform 0.4 上还应参考 RTGNN。
- “当前 baseline 结果可以直接作为 GAGB 或粒球机制有效性的证据。”

## 12. Missing Information / Residual Risk

残余风险：

- NoisyGL 默认配置是否与原论文最佳超参完全一致尚未逐篇核对。
- PubMed RNCGLN 如果做稀疏化或低显存改造，可能能跑，但那已经不是当前默认实现 baseline。
- PubMed PIGNN 只完成 smoke test，尚缺四组合 5-seed，因此不能进入最终 baseline 主表。
- 当前只有 Cora/CiteSeer/PubMed 三个 citation network；若论文目标更广，后续可能还需补其他数据集。
- Cora/CiteSeer 的部分 RNCGLN 提升 std 较大，后续方法比较时必须谨慎。
- 代码兼容性修复虽然不改变训练协议，但仍应在最终论文实验前固定 commit、环境和配置。

## 13. Questions For GPT

希望外部 GPT 基于本文档帮助判断：

1. 当前 baseline 选择是否足以作为新方法设计前的起点？
2. 是否应该把 Cora/CiteSeer 的 RNCGLN 和 PubMed 的 GCN 分开作为主 baseline？
3. PubMed 是否还需要继续寻找更强可运行 baseline，还是可以先开始方法设计？
4. 后续自己的方法应优先在哪些噪声设置上验证，才能避免单点过拟合？
5. 论文中如何表述 RNCGLN 的优势和 PubMed OOM 边界，才不会过 claim？

## 14. Full Paste-Ready GPT Prompt

### 可直接复制给 GPT 的提示词

```text
你看不到我的本地实验文件夹、代码、CSV 和日志，所以我把关键信息完整整理在下面。请只基于这份 self-contained 信息进行科研判断，不要假设你能访问本地路径。

我正在做图标签噪声鲁棒学习任务。在设计自己的方法前，我想先确认是否已经找到并跑通了一个实验流程完整、结果相对较强的 baseline。注意：这里没有 GAGB，没有粒球模块，也不要把任何粒球机制当作本实验内容。

实验框架：NoisyGL，commit b6ac057fe8c45a1475a32dba823971372186e445。
运行环境：torch 2.9.1+rocm7.2.1，AMD Radeon RX 7700 XT，device cuda:0。
数据集：Cora、CiteSeer、PubMed。
噪声类型：uniform、pair。
噪声率：0.2、0.4。
seeds：3000、3001、3002、3003、3004。
指标：test accuracy mean/std。

代码只做了兼容性修复：single_exp.py 和 total_exp.py 改为 predictor lazy import；np.in1d 改为 np.isin；utils/labelnoise.py 修复 independent label noise 的 scalar 写入 bug，并移除残留无参调用。没有改变模型结构、训练协议、split 或 YAML 默认配置。

Cora/CiteSeer 结果：

| Dataset | Noise | Rate | GCN | RNCGLN | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Cora | uniform | 0.2 | 0.7620 +/- 0.0082 | 0.7600 +/- 0.0161 | RNCGLN 不占优 |
| Cora | uniform | 0.4 | 0.6920 +/- 0.0083 | 0.7532 +/- 0.0185 | RNCGLN 明显占优 |
| Cora | pair | 0.2 | 0.7318 +/- 0.0176 | 0.7366 +/- 0.0348 | 小幅占优但 std 较大 |
| Cora | pair | 0.4 | 0.5610 +/- 0.0339 | 0.6062 +/- 0.0729 | 占优但 std 较大 |
| CiteSeer | uniform | 0.2 | 0.6262 +/- 0.0342 | 0.6734 +/- 0.0377 | RNCGLN 占优 |
| CiteSeer | uniform | 0.4 | 0.5024 +/- 0.0630 | 0.6020 +/- 0.1179 | 占优但 std 很大 |
| CiteSeer | pair | 0.2 | 0.6088 +/- 0.0249 | 0.6636 +/- 0.0418 | RNCGLN 占优 |
| CiteSeer | pair | 0.4 | 0.4216 +/- 0.0284 | 0.4174 +/- 0.0506 | RNCGLN 不占优 |

PubMed 结果：

| Dataset | Noise | Rate | GCN | UnionNet | Forward | Backward | RTGNN | CRGNN | RNCGLN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PubMed | uniform | 0.2 | 0.7036 +/- 0.0435 | 0.7058 +/- 0.0492 | 0.7042 +/- 0.0523 | 0.7050 +/- 0.0533 | 0.6890 +/- 0.0469 | 0.6970 +/- 0.0315 | OOM |
| PubMed | uniform | 0.4 | 0.5684 +/- 0.0489 | 0.5664 +/- 0.0430 | 0.5734 +/- 0.0449 | 0.5728 +/- 0.0449 | 0.6214 +/- 0.0514 | 0.5648 +/- 0.0711 | not run after OOM |
| PubMed | pair | 0.2 | 0.7266 +/- 0.0224 | 0.7350 +/- 0.0217 | 0.7290 +/- 0.0333 | 0.7298 +/- 0.0310 | 0.7038 +/- 0.0619 | 0.7264 +/- 0.0305 | not run after OOM |
| PubMed | pair | 0.4 | 0.5772 +/- 0.0757 | 0.5706 +/- 0.0585 | 0.5770 +/- 0.0555 | 0.5762 +/- 0.0573 | 0.5317 +/- 0.1176 | 0.5920 +/- 0.0594 | not run after OOM |

PubMed RNCGLN OOM 原因：RNCGLN 默认实现中存在 dense x @ x.T 节点对矩阵。PubMed N=19717，单个 dense float32 矩阵约 1.45GB，训练时加上梯度和中间张量，当前 12GB GPU 无法运行。日志包含：torch.OutOfMemoryError: HIP out of memory. Tried to allocate 1.45 GiB.

PubMed single-seed 候选筛查：
unionnet 0.6240，但 5-seed 后没有稳定超过 GCN；
pignn 0.6600，smoke test 跑通，但全量 5-seed 暂缓；
rtgnn 0.6980，smoke test 很强，并已完成四组合 5-seed；
forward/backward 0.5820，5-seed 只在 std 内小幅波动；
coteaching 0.5260 但训练 loss 出现 NaN；
nrgnn 0.5020 且后期 train loss NaN；
clnode 0.4970、crgnn 0.4940、sce 0.4910、jocor 0.4520 较弱；
cp 因缺 pyg-lib/torch-cluster 失败；
r2lp 因 storage size calculation overflow 失败。

我的初步判断：
1. Cora/CiteSeer 上可以用 GCN + RNCGLN 作为第一版强 baseline 组合。
2. PubMed 上不能把 RNCGLN 当当前可运行 baseline；GCN 是基础参照，RTGNN 是 uniform 0.4 的当前强候选，CRGNN 是 pair 0.4 的补充候选。
3. PIGNN 是重要待补项：single-seed smoke test 已跑通且 0.6600，但 full 5-seed 暂缓，不能先写成稳定结论。
4. 现有实验足以支撑开始设计自己的方法，但论文表述必须区分“已支持结论”“合理猜测”“负结果”和“缺失信息”。

请你帮我判断：
1. 当前 baseline 选择是否足以作为新方法设计前的起点？
2. 是否应该把 Cora/CiteSeer 的 RNCGLN 和 PubMed 的 GCN 分开作为主 baseline？
3. PubMed 是否还需要继续寻找更强可运行 baseline，还是可以先开始方法设计？
4. 后续自己的方法应优先在哪些噪声设置上验证，才能避免单点过拟合？
5. 论文中如何表述 RNCGLN 的优势和 PubMed OOM 边界，才不会过 claim？

请不要做代码审查；只做实验设计、baseline 充分性、结论边界和下一步实验建议的科研判断。
```
