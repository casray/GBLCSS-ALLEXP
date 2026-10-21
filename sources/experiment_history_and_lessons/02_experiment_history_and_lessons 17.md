# Phase44: GEAF-style 粒椭球噪声过滤桥接实验（2026-06-29）

## 研究假设

前一轮椭球 SoftDist 的主要矛盾是：离线 target diagnostic 多次显示 clean-label mass 或 purity 有改善，但 downstream accuracy 大多没有收益，甚至出现负效果。因此本轮检验一个桥接假设：

> 当前离线 SoftDist 信号不能转化，可能不是因为粒椭球局部结构完全无效，而是因为缺少“粒椭球生成 -> 显式噪声过滤/修正 -> 分类”的中间层。

本轮只把 GEAF-style hard correction 当作诊断桥，不把它直接升级为论文主方法。

## 粒球扮演的角色

- 粒球不再只提供 soft target 分布，而是提供 train mask 内的 hard label correction。
- 构球仍保持 label-free：只在 frozen GRACE `H` 空间构球，不读取 noisy label、clean label、val/test label。
- clean label 只用于离线诊断中的 NCR/CDR/hard-net/gate，不参与构球、过滤规则或训练。
- shuffled control 保持球大小分布一致，用于排除“改标签数量本身”带来的假阳性。

## 方法动机

孙旭毕业论文第五章的可借鉴点不是把 clean-label 纯度规则直接搬到 noisy-label 任务，而是把流程拆成：

1. 粒椭球生成；
2. 显式噪声过滤或修正；
3. 再进入分类。

本项目此前 SoftDist 是“构球 -> soft target -> 分类”的单层替换，没有显式过滤阶段。GEAF bridge 的目的就是验证这个缺失层是否解释 offline-to-downstream 断裂。

## 代码改动

主要代码位于 `external/NoisyGL`：

- 新增 size-adaptive ellipsoid 构球：`external/NoisyGL/utils/ordinary_gb.py` 中 `build_size_adaptive_ellipsoid_balls`。
- 扩展 frozen GRACE target 构造：`external/NoisyGL/utils/frozen_grace_targets.py` 支持 `ball_mode="size_adaptive_ellipsoid"`，并增加 GEAF correction 与分区诊断。
- 新增 downstream 方法：
  - `frozengrace_geaf_gcn_xh`
  - `frozengrace_geaf_gcn_xh_shuffled`
- 注册位置：
  - `external/NoisyGL/predictor/FrozenGraceGB_GCN_Predictor.py`
  - `external/NoisyGL/scripts/run_standard_downstream.py`
  - `external/NoisyGL/single_exp.py`
- 新增诊断与执行脚本：
  - `external/NoisyGL/scripts/diagnose_geaf_bridge.py`
  - `external/NoisyGL/scripts/run_geaf_bridge_downstream.py`
- 新增测试：`external/NoisyGL/tests/test_geaf_bridge.py`。

## 配置改动

新增 GEAF downstream 配置：

- `external/NoisyGL/config/frozengrace_geaf_gcn_xh/{dblp,pubmed,amazonpho,amazoncom}.yaml`
- `external/NoisyGL/config/frozengrace_geaf_gcn_xh_shuffled/{dblp,pubmed,amazonpho,amazoncom}.yaml`

实验协议保持与主项目一致：

- noise type: `uniform`
- noise rate: `0.4`
- noise scope: `train`
- split: percent `0.2/0.1/0.7`
- seeds: offline `0..4`，downstream 先 seed `0`
- device: 用户要求后改为 `cuda:0`

注意：构球内部的 sklearn KMeans 仍是 CPU 实现；`--device cuda:0` 主要覆盖数据/tensor 与 downstream 训练路径。不要把本轮描述成“全流程 GPU 构球”。

## 预期机制

如果粒椭球局部多数类确实能修复 train noisy labels，则 hard correction 应同时满足：

- NCR 足够高；
- CDR 足够低；
- hard-net 为正；
- real balls 明显优于 shuffled size-matched control；
- corrected labels 进入 GCN[X‖H] 后，至少在 seed0 上超过 `gcn_xh_ce`、当前 SoftDist 与 GEAF shuffled。

如果离线过门但下游失败，说明“标签修正质量”仍可能被 `[X‖H]` baseline 吸收，或边界样本损伤抵消收益。

## 成功标准

离线 gate：

- `hard_net_mean >= 0.02`
- `ncr_mean >= 0.05`
- `cdr_mean <= 0.03`
- real 的 hard-net 高于 shuffled，positive seeds `>= 4/5`
- `zero_train_ball_ratio <= 0.25`

下游 seed0：

- `frozengrace_geaf_gcn_xh` 应高于 `gcn_xh_ce`
- 高于 `frozengrace_geaf_gcn_xh_shuffled`
- 最好同时不弱于当前 `frozengrace_gb_gcn_softdist_xh`

## 失败标准

- 离线 gate 全失败：停止 downstream。
- 离线 gate 通过但 downstream 不优于 shuffled 或 `gcn_xh_ce`：不进入 seed5。
- seed0 只有极小提升且低于强 control：只记为边缘或负证据。

## 实验结果

### 测试与静态检查

通过测试：

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" -m pytest tests\test_frozen_grace_targets.py tests\test_geaf_bridge.py tests\test_ellipsoid_param_grid.py tests\test_xh_features.py -q
```

结果：`38 passed, 1 warning`。

脚本编译检查通过：

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" -m py_compile scripts\diagnose_geaf_bridge.py scripts\run_geaf_bridge_downstream.py
```

### 主四数据集 GEAF 离线诊断

产物目录：`external/NoisyGL/runs/geaf_bridge_20260629_gpu/`

关键文件：

- `geaf_bridge_report.md`
- `geaf_bridge_gate.csv`
- `geaf_bridge_raw.csv`
- `geaf_bridge_aggregate.csv`
- `geaf_bridge_paired_deltas.csv`

规模：

- raw rows: `120`
- gate passed settings: `6`

Gate 结果：

| Dataset | Confidence | Gate | hard_net | NCR | CDR | positive seeds | 结论 |
|---|---:|---|---:|---:|---:|---:|---|
| AmazonCom | 0.0 | FAIL | 0.3682 | 0.5763 | 0.2081 | 5 | 改太多，伤 clean label |
| AmazonCom | 0.6 | FAIL | 0.1604 | 0.1906 | 0.0302 | 5 | CDR 略超阈值 |
| AmazonCom | 0.7 | PASS | 0.0574 | 0.0646 | 0.0072 | 5 | 通过 |
| AmazonPhoto | 0.0 | FAIL | 0.4264 | 0.5952 | 0.1687 | 5 | CDR 高 |
| AmazonPhoto | 0.6 | PASS | 0.1963 | 0.2220 | 0.0257 | 5 | 通过 |
| AmazonPhoto | 0.7 | PASS | 0.0739 | 0.0809 | 0.0070 | 5 | 通过 |
| DBLP | 0.0 | FAIL | 0.5054 | 0.6836 | 0.1782 | 5 | CDR 高 |
| DBLP | 0.6 | PASS | 0.1955 | 0.2209 | 0.0254 | 5 | 通过 |
| DBLP | 0.7 | PASS | 0.0685 | 0.0761 | 0.0076 | 5 | 通过 |
| PubMed | 0.0 | FAIL | 0.4429 | 0.6489 | 0.2060 | 5 | CDR 高 |
| PubMed | 0.6 | FAIL | 0.1886 | 0.2246 | 0.0360 | 5 | CDR 高 |
| PubMed | 0.7 | PASS | 0.0642 | 0.0736 | 0.0094 | 5 | 通过 |

离线结论：

- confidence threshold 是必要变量。`0.0` 虽然 NCR 高，但 CDR 也高，属于“纠噪多但误伤 clean 也多”。
- `0.6/0.7` 可以把 CDR 压低，说明 GEAF-style 显式过滤确实比无阈值硬改更可控。
- 离线 gate 通过并不保证 downstream 收益，仍需下游验证。

### 主四数据集 downstream seed0

产物目录：`external/NoisyGL/runs/geaf_bridge_downstream_seed0_20260629_gpu/`

关键文件：

- `geaf_bridge_downstream_seed0_report.md`
- `geaf_bridge_downstream_seed0_summary.csv`

规模：

- gate settings consumed: `6`
- downstream rows: `30`
- all rows status: `ok`

核心结果：

| Dataset | Confidence | GCN[X‖H] | SoftDist | SoftDist shuffled | GEAF | GEAF shuffled | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| AmazonCom | 0.7 | 0.8564 | 0.8558 | 0.8578 | **0.8642** | 0.8564 | seed0 明确正 |
| AmazonPhoto | 0.6 | 0.9109 | 0.8964 | 0.8995 | **0.9115** | 0.9109 | 只比 GCN 高约 0.06pp，太小 |
| AmazonPhoto | 0.7 | 0.9109 | 0.8964 | 0.8995 | 0.9046 | 0.9109 | 负于 GCN/shuffled |
| DBLP | 0.6 | 0.8429 | 0.8426 | 0.8426 | 0.8349 | 0.8429 | 负 |
| DBLP | 0.7 | 0.8429 | 0.8426 | 0.8426 | 0.8429 | 0.8429 | 持平 |
| PubMed | 0.7 | 0.8249 | **0.8252** | 0.8241 | 0.8243 | 0.8249 | 负于 GCN/SoftDist |

downstream 结论：

- AmazonCom `confidence=0.7` 是本轮唯一明显 seed0 正结果：GEAF 比 GCN[X‖H] 与 GEAF shuffled 高约 `+0.78pp`，比当前 SoftDist 高约 `+0.84pp`，比 SoftDist shuffled 高约 `+0.64pp`。
- AmazonPhoto `confidence=0.6` 只比 GCN[X‖H] 高约 `+0.06pp`，低于可以推进的强证据标准。
- DBLP/PubMed 的离线 gate 没有转化为 downstream，继续支持“offline-positive 不等于 downstream-positive”的结论。

### Cora/CiteSeer/Coauthor-CS 补充诊断

产物目录：

- `external/NoisyGL/runs/geaf_bridge_softdist_success_20260629_gpu/`
- `external/NoisyGL/runs/geaf_bridge_softdist_success_downstream_seed0_20260629_gpu/`

该补充覆盖 Cora、CiteSeer、Coauthor-CS，confidence `0.6/0.7/0.8`。离线 raw rows 为 `90`，gate passed settings 为 `2`：

| Dataset | Confidence | Gate | hard_net | NCR | CDR | 结论 |
|---|---:|---|---:|---:|---:|---|
| Coauthor-CS | 0.6 | PASS | 0.2263 | 0.2405 | 0.0142 | 通过 |
| Coauthor-CS | 0.7 | PASS | 0.0890 | 0.0935 | 0.0045 | 通过 |
| Coauthor-CS | 0.8 | FAIL | 0.0338 | 0.0353 | 0.0015 | NCR 低 |
| Cora | 0.6 | FAIL | 0.1286 | 0.1645 | 0.0359 | CDR 高 |
| Cora | 0.7 | FAIL | 0.0393 | 0.0453 | 0.0061 | NCR 低 |
| CiteSeer | 0.6 | FAIL | 0.0521 | 0.1008 | 0.0487 | CDR 高 |

Coauthor-CS downstream seed0：

| Confidence | GCN[X‖H] | SoftDist | SoftDist shuffled | GEAF | GEAF shuffled | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 0.6 | 0.9189 | **0.9208** | 0.9199 | 0.9185 | 0.9189 | GEAF 负于 SoftDist 与 shuffled |
| 0.7 | 0.9189 | **0.9208** | 0.9199 | 0.9189 | 0.9189 | GEAF 持平 GCN，负于 SoftDist |

补充结论：在原主成功数据集 Coauthor-CS 上，GEAF hard correction 不能替代 SoftDist。即使离线 gate 通过，下游仍被当前 SoftDist 压制。

## 最终决策

### 已经被实验支持

- GEAF-style 显式 hard correction 确实能在离线层面产生可控的噪声修正信号，特别是 `confidence=0.6/0.7` 能显著降低 CDR。
- 无阈值 hard correction 不可用：`confidence=0.0` 在四个主诊断数据集上均因 CDR 过高失败。
- 主四数据集中，只有 AmazonCom `confidence=0.7` 在 downstream seed0 上形成明确正信号。
- DBLP、PubMed、AmazonPhoto 的 GEAF 离线 gate 通过不等于下游收益。
- Coauthor-CS 补充实验显示，GEAF hard correction 即使离线通过，也不能替代当前 SoftDist 主方法。

### 合理猜测

- AmazonCom 的局部 H-space 粒椭球多数类可能更贴近真实可纠正噪声，因此 hard correction 的条件增量没有完全被 GCN[X‖H] 吸收。
- DBLP/PubMed 失败可能来自 corrected labels 与 `[X‖H]` baseline 决策边界信息冗余，或边界样本误改造成训练损伤。
- AmazonPhoto `confidence=0.6` 的 `+0.06pp` 更可能是 seed0 波动或极弱边际，而不是稳定方法收益。

### 尚未验证

- AmazonCom `confidence=0.7` 是否能在 5 seeds 下稳定超过 GCN[X‖H]、GEAF shuffled 与当前 SoftDist。
- 是否存在 soft/hybrid GEAF，比 hard correction 更适合 DBLP/PubMed/AmazonPhoto。
- 是否能设计不依赖 clean label 的 deployable confidence selection policy。
- `gradient_weighted_target_gain` 是否能解释为什么部分离线 gate 通过设置不转化为下游准确率。

### 已被负结果削弱

- “离线 hard correction gate 通过即可进入 downstream 扩展”的判断被削弱。
- “GEAF-style hard correction 可以普适修复 SoftDist offline-to-downstream 断裂”被削弱。
- “DBLP/PubMed 的问题主要靠显式标签修正即可解决”被削弱。
- “Coauthor-CS 上 GEAF 可以替代当前 SoftDist”被削弱。

## 下一步建议

只建议对 AmazonCom `confidence=0.7` 做一个严格的 5-seed sanity：

- 若 5-seed 仍稳定高于 `gcn_xh_ce`、`frozengrace_geaf_gcn_xh_shuffled` 和当前 SoftDist，可把 GEAF bridge 写成 AmazonCom-specific positive boundary case。
- 若 5-seed 不稳，GEAF bridge 就写成“离线过滤信号大多不能稳定转化，下游唯一 seed0 正例也未能稳定”的负结果。

不建议把 DBLP、PubMed、AmazonPhoto、Coauthor-CS 的 GEAF 结果推进到 seed5 或强 baseline。当前最有价值的论文贡献仍是 SoftDist [X‖H] 主线与适用域/边界刻画。
