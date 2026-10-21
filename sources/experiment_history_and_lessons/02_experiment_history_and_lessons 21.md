# MLP-24 尺度自适应球数（2026-07-12）

## 研究假设

固定 `max_balls=64` 可能使大图上的粒球过粗。预注册唯一候选为

\[
B_{\mathrm{eff}}=\max(64,\lceil\sqrt{|V|}\rceil).
\]

只改变球数上限；frozen GRACE H、`min_ball_size=32`、`split_improvement=0.05`、`beta=0.5`、两层 X-only MLP、uniform 0.4 train-only 噪声和 20/10/70 split 均保持不变。不得根据结果调整第二个公式。

## 粒球角色与方法动机

粒球仍是在 frozen GRACE H 中无标签构造的局部共识单元，球内只统计 train noisy labels 并形成半硬 soft target。实验检验的是更细粒度能否给现有 LCSS 增加新的监督信息，而不是重新调主方法。

## 代码与配置改动

- `external/NoisyGL/utils/ordinary_gb.py`：增加 `fixed|sqrt_nodes` 球数策略解析。
- `external/NoisyGL/predictor/FrozenGraceGB_GCN_Predictor.py`：记录 configured/effective cap、是否触顶和构球时间。
- `external/NoisyGL/scripts/run_ssl_swap_downstream.py`：增加 CLI override 和完整 split 字段。
- `external/NoisyGL/scripts/diagnose_mlp24_scaleadaptive_balls_20260712.py`：11 数据集 fixed/adaptive/shuffled 几何审计。
- `external/NoisyGL/scripts/summarize_mlp24_scaleadaptive_balls_20260712.py`：协议、hash、完整性和预注册 gate。
- `external/NoisyGL/tests/test_mlp24_scaleadaptive_balls.py`：策略、等价性、泄漏和 gate 回归测试。

旧 YAML 与历史 fixed-64 anchor 未被覆盖。

## Phase A：几何结果

几何闸门通过，33/33 行完整。Coauthor-Physics 从 64 球增至 186 球，平均球规模从 538.95 降至 185.45（下降 65.6%），每球平均 train votes 仍为 37.09；adaptive 构球约 24.44 秒，低于 10 倍和 10 分钟上限。Cora/CiteSeer/Cora_ML/Attributed-Wiki 的 assignment、soft target 和 alpha 在新策略下逐元素不变。

这证明公式确实改变了大图几何，并非只提高上限但未触发分裂。

## Phase B：uniform 0.4 seed0

| 数据集 | fixed LCSS | adaptive LCSS | adaptive-fixed | adaptive-shuffled |
|---|---:|---:|---:|---:|
| CiteSeerFull | 59.81 | 59.81 | +0.00pp | +14.96pp |
| CoraFull | 37.39 | 38.32 | +0.93pp | +5.98pp |
| Coauthor-CS | 90.35 | 90.56 | +0.20pp | +2.97pp |
| Amazon-Photo | 81.44 | 81.48 | +0.04pp | +9.37pp |
| PubMed | 71.60 | 71.97 | +0.37pp | +9.30pp |
| Amazon-Computers | 75.33 | 76.03 | +0.71pp | +8.05pp |
| Coauthor-Physics | 93.05 | 93.15 | +0.10pp | +5.53pp |

七个受影响数据集的平均 adaptive-fixed 为约 `+0.33pp`。CoraFull 与 Amazon-Computers 有局部 seed0 正信号，但关键 Coauthor-Physics 只有 `+0.10pp`，未达到预注册 `+0.50pp` 门。Phase B 因此正式失败。

## 最终决策

- **停止 Phase C 5-seed 和 Phase D 全噪声谱。** 不为局部有利数据集放宽门槛。
- 主方法继续使用固定 `max_balls=64`；论文方法公式、主表和复杂度均不修改。
- 尺度自适应方向记录为负结果：更细分组显著改变几何并保持真实结构信号，但没有给下游分类带来足够新增信息。
- CoraFull/Amazon-Computers 的 seed0 提升只属于合理线索，不是稳定结论，不单独扩 5 seeds。

## 证据分级

### 已被实验支持

- `sqrt(N)` 上限能在大图上显著增加球数并降低平均球规模，且不造成票数稀疏或构球成本失控。
- adaptive real 在七个受影响数据集均明显优于其 shuffled 控制，真实局部结构仍有效。
- Coauthor-Physics 的分类增量仅 +0.10pp，未通过预注册 smoke 门；固定 64 不应被替换。

### 合理解释

- fixed-64 soft target 已吸收大部分可用局部共识；继续细分主要改变几何分辨率，却没有增加足以改变 MLP 决策边界的信息。

### 尚未验证

- CoraFull/Amazon-Computers 的 seed0 小提升是否跨 seeds 稳定。由于总闸门失败，按预注册纪律不继续验证。

### 已被负结果削弱

- 大图表现受限主要是固定 64 球过粗。
- 简单按节点数扩展球数即可把 Coauthor-Physics 从“与 JoCoR 相当”提升为明确领先。

## 关键证据

- 计划：`external/NoisyGL/docs/execution_plan_mlp24_scaleadaptive_balls_20260712.md`
- 结果：`external/NoisyGL/runs/mlp24_scaleadaptive_balls_20260712/`
- gate：`mlp24_gate_decisions.csv`
- 几何：`mlp24_geometry.csv`
- seed0：`mlp24_smoke_seed0.csv`
- 报告：`mlp24_report.md`
- 验证：50 tests passed，Phase A 33/33、Phase B 44/44，完整性失败 0。
