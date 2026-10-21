# Phase 34 GBSelect 下游验证报告

日期：2026-06-16  
执行者：Codex  
协议：NoisyGL percent split 0.2/0.1/0.7，uniform noise 0.4，seeds 0-4，ROCm GPU。  
核心问题：Phase 33 的离线强降噪信号能否转化为真实下游 test accuracy，并且是否证明“球 + GCN 双视角”优于“GCN 单视角”。

## 1. 实现摘要

新增 `gbselect` 系列 predictor：

- `gbselect`：球嫌疑分 `s_ball` 与 warmup GCN 嫌疑分 `s_gcn` 的 clean-sample 交集。
- `gbselect_gcn_only`：只用 `s_gcn` 选干净样本，是命门对照。
- `gbselect_shuffled`：随机同尺寸球 + `s_gcn`，检验真实球结构是否必要。
- 最终分类器统一使用 `GCN([X||H])`，selection 只作用于 train 节点，val/test 不参与 selection。

标签泄漏控制：

- `s_ball` 只读取 train noisy label。
- warmup GCN 只用 noisy label。
- clean label 只用于最终 test 与诊断噪声率。
- 单测覆盖 clean label 篡改不影响 selection、train-only selection、`keep_ratio=1.0` 退化为全 train。

## 2. DBLP：keep_ratio 敏感性

| 设置 | GBSelect | GCN-only | Shuffled | GCN [X\|\|H] | GBSelect - GCN-only | GBSelect - Shuffled | GBSelect - GCN [X\|\|H] |
|---|---:|---:|---:|---:|---:|---:|---:|
| keep=0.5 | 0.8416 ± 0.0024 | 0.8403 ± 0.0051 | - | - | +0.13pp, 2/5 | - | - |
| keep=0.6 | 0.8403 ± 0.0036 | 0.8407 ± 0.0026 | 0.7935 ± 0.0198 | 0.8348 ± 0.0076 | -0.04pp, 1/5 | +4.68pp, 5/5 | +0.55pp, 4/5 |
| keep=0.7 | 0.8394 ± 0.0046 | 0.8358 ± 0.0092 | 0.8087 ± 0.0055 | 0.8348 ± 0.0076 | +0.35pp, 3/5 | +3.06pp, 5/5 | +0.46pp, 5/5 |

诊断噪声率：

- keep=0.5：GBSelect selected noise 0.0258，GCN-only selected noise 0.0362。
- keep=0.6：GBSelect selected noise 0.0558，GCN-only selected noise 0.0802。
- keep=0.7：GBSelect selected noise 0.1070，GCN-only selected noise 0.1571。

判定：

- 真实球明显优于 shuffled，说明球结构不是随机效果。
- 但命门对照 `GBSelect > GCN-only` 不稳定：keep=0.6 失败，keep=0.7 仅 3/5 seeds 为正，未达到预注册的 4/5 稳定门槛。
- DBLP 只能算“弱正信号”，不能作为大图成功 claim。

## 3. PubMed：命门对照不过

| 方法 | PubMed |
|---|---:|
| GBSelect keep=0.7 | 0.8326 ± 0.0066 |
| GCN-only keep=0.7 | 0.8319 ± 0.0047 |
| Shuffled keep=0.7 | 0.8163 ± 0.0055 |
| GCN [X\|\|H] | 0.8296 ± 0.0046 |

配对差：

- GBSelect - GCN-only：+0.06pp，2/5 seeds。
- GBSelect - Shuffled：+1.63pp，5/5 seeds。
- GBSelect - GCN [X||H]：+0.30pp，3/5 seeds。

诊断噪声率：

- GBSelect selected noise 0.1259。
- GCN-only selected noise 0.1779。
- Shuffled selected noise 0.1659。

判定：

- 球确实选择出更干净的 train 子集，并且优于 shuffled。
- 下游 test acc 没有稳定超过 GCN-only，说明“更干净的子集”被样本量损失和模型自身筛选能力抵消。
- PubMed 不能支持 GBSelect 主线。

## 4. Amazon-Photo：双视角胜单视角，但 selection 形态整体失败

| 方法 | Amazon-Photo |
|---|---:|
| GBSelect keep=0.7 | 0.7818 ± 0.0590 |
| GCN-only keep=0.7 | 0.7506 ± 0.0660 |
| Shuffled keep=0.7 | 0.7334 ± 0.0846 |
| GCN [X\|\|H] | 0.9049 ± 0.0130 |

配对差：

- GBSelect - GCN-only：+3.11pp，5/5 seeds。
- GBSelect - Shuffled：+4.83pp，5/5 seeds。
- GBSelect - GCN [X||H]：-12.32pp，0/5 seeds。

诊断噪声率：

- GBSelect selected noise 0.1521。
- GCN-only selected noise 0.3231。
- Shuffled selected noise 0.2785。

判定：

- 球作为第二视角很强，但 selection 丢弃了大量有用监督，导致最终模型远低于全 train 的 `GCN [X||H]`。
- 这说明 Amazon-Photo 的问题不是“球无信号”，而是“样本选择范式不适合该 regime”。该数据集 plain / XH GCN 已很强，降噪收益无法覆盖监督量损失。

## 5. 总体判定

GBSelect 没有解锁“大图统一框架”：

- DBLP：真实球结构有效，但 `GBSelect > GCN-only` 不够稳定。
- PubMed：真实球结构有效，但命门对照不过。
- Amazon-Photo：命门对照过，但大幅低于全 train `GCN [X||H]`。

因此 Phase 34c 不触发。论文不应新增“GBSelect 大图成功主方法”或“第三成功数据集” claim。

可以保留的论文价值：

- Phase 33-34 构成一个完整边界分析：离线噪声判别信号很强，但下游训练会被样本量损失、GCN 单视角吸收和数据集 regime 差异抵消。
- 这进一步支持主线收束：本文主方法仍是小图 Cora/CiteSeer 上的 SoftDist [X||H]，大图样本选择作为边界/未来工作，而非当前贡献。

## 6. 文件与验证

主要实现文件：

- `utils/gbselect.py`
- `predictor/GBSelect_Predictor.py`
- `tests/test_gbselect.py`
- `scripts/run_ssl_swap_downstream.py`
- `single_exp.py`
- `total_exp.py`
- `config/gbselect*/gbselect*_dblp.yaml`
- `config/gbselect*/gbselect*_pubmed.yaml`
- `config/gbselect*/gbselect*_amazonpho.yaml`

原始结果：

- `runs/gbselect_20260616/dblp_seed0_smoke.csv`
- `runs/gbselect_20260616/dblp_5seed.csv`
- `runs/gbselect_20260616/dblp_keep05_5seed.csv`
- `runs/gbselect_20260616/dblp_keep07_5seed.csv`
- `runs/gbselect_20260616/dblp_keep07_shuffled_5seed.csv`
- `runs/gbselect_20260616/pubmed_keep07_5seed.csv`
- `runs/gbselect_20260616/amazonpho_keep07_5seed.csv`
- `runs/gbselect_diagnostics_20260616.csv`

