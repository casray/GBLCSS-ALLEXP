# V32 Submission Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 v29 的完整披露和自然语言为写作底稿，合并 tie-safe v2 与最近工作正式网格，补充验证标签噪声敏感性证据，生成数据、协议、claim 和附录完全一致的 v32 中文投稿候选稿。

**Architecture:** 工作分为五条顺序依赖的证据线：先冻结证据谱系，再实现并运行 `train_val` 协议敏感性实验，然后生成三层资源表和 v32，最后完成 P1 诊断、复现资产与冷审。基础 GB-LCSS 不再改动；所有新增实验均使用当前 tie-safe `distribution_only` 规则。协议敏感性暂用历史 `gsdn` adapted 实现，新 GSDN 的重实现和正式比较留给后续独立版本，不阻塞 v32。

**Tech Stack:** Python 3.12、PyTorch 2.9.1 ROCm、NoisyGL、本地 CSV/Markdown 审计脚本、pytest/unittest、PowerShell。

---

## 0. 用户已确认的范围

- [x] **D1：v32 谱系。** 采用“v29 语言与完整披露 + v31 tie-safe/最近工作事实”，废弃 condensed 作为投稿底稿。
- [x] **D2：三层正文表。** 表 A 为训练节点目标构造族；表 B 为训练期图辅助、推理期 X-only 方法；表 C 为更高资源蒸馏扩展。旧 `gsdn` 只能以 adapted 临时实现身份进入表 B，不视为最终忠实复现。
- [x] **D3：唯一 P0 新实验。** 执行验证标签敏感性网格，新增 90 个 `train_val` 运行；clean-validation 行复用现有正式证据。
- [x] **D4：新 GSDN 暂缓。** 本轮不修改或使用 `gsdn_faithful_core`，不执行蒸馏对新 GSDN 的正式比较。用户后续另行提供更贴合协议的新版本。
- [x] **D5：P1 外部有效性。** 默认不新跑异质图；WikiCS/Flickr 只进入 rebuttal 预案。P1 先完成理论、alpha 诊断、macro-F1/校准和成本表。
- [x] **D6：PubMed 处置。** 删除正文中的 PubMed 专门分析，PubMed 的所有结果、边界讨论和诊断统一移入附录。

### 本轮完成状态（2026-07-29）

- [x] v32 中文稿、完整附录、三层资源表和自动审计已完成。
- [x] 验证标签敏感性实验完成 90/90 个新增运行，预注册机制门槛通过。
- [x] tie-safe v2 与 PosteL/ALS/SALS 正式证据已登记，旧 GSDN 仅作临时 adapted 参照。
- [x] \(\alpha\) 事后校准诊断和对称噪声理论附录已完成。
- [x] PubMed 全部移至附录 E，不进入正文分析或当前核心结论计数。
- [ ] macro-F1/ECE 需要最多 150 个新运行，留待用户另行确认。
- [ ] 新 GSDN 正式实现等待用户后续提供，不在 v32 中形成强弱结论。
- [ ] 匿名复现仓库、无上下文冷审和英文稿不属于本轮 v32 中文稿交付范围。

## 1. 文件结构

### 新建

- `paper_pipeline/00_evidence/v32_evidence_registry.csv`  
  登记每个正式表格、结论、源 CSV、算法版本、tie mode、协议和可用范围。
- `paper_pipeline/scripts/build_v32_evidence_registry.py`  
  从不可变结果生成证据登记表，拒绝旧 tie 规则进入正式主结论。
- `paper_pipeline/tests/test_v32_evidence_registry.py`  
  测试旧结果隔离、正式结果完整性和 headline 数字。
- `paper_pipeline/04_experiments/validation_scope_sensitivity_plan_20260729.md`  
  运行前协议、假设、判据和时间线。
- `paper_pipeline/scripts/run_validation_scope_sensitivity.py`  
  生成、运行、续跑和审计 90 个 noisy-validation 结果。
- `paper_pipeline/tests/test_validation_scope_sensitivity.py`  
  测试命令、噪声隔离、完整性和统计输出。
- `paper_pipeline/04_experiments/validation_scope_sensitivity_report.md`  
  运行后正式报告。
- `paper_pipeline/scripts/generate_v32_tables.py`  
  从正式 CSV 生成三层正文表、边界表和摘要数字。
- `paper_pipeline/tests/test_generate_v32_tables.py`  
  防止新旧结果混用和资源层错位。
- `paper_pipeline/03_tables/v32_table_a_target_only.md`
- `paper_pipeline/03_tables/v32_table_b_graph_assisted_xonly.md`
- `paper_pipeline/03_tables/v32_table_c_distillation.md`
- `paper_pipeline/03_tables/v32_boundary_datasets.md`
- `paper_pipeline/03_tables/v32_headline_metrics.json`
- `paper_final_zh_v32_submission_candidate_202607xx.md`
- `paper_pipeline/scripts/audit_v32_manuscript.py`
- `paper_pipeline/tests/test_audit_v32_manuscript.py`
- `paper_pipeline/01_reviews/v32_cold_review_pack.md`

### 修改

- `C:/Users/Admin/Desktop/实验8去噪5.19-/external/NoisyGL/scripts/run_standard_downstream.py`  
  增加 `noise_scope=train_val`，记录训练和验证噪声子哈希。
- `C:/Users/Admin/Desktop/实验8去噪5.19-/external/NoisyGL/tests/test_standard_runner_baselines.py`
- `C:/Users/Admin/Desktop/实验8去噪5.19-/external/NoisyGL/tests/test_mlp14_graphfree_family.py`
- `paper_pipeline/00_evidence/claim_evidence_ledger.md`
- `paper_pipeline/00_evidence/nearest_work_verification_notes.md`

### 只读正式证据

- `C:/Users/Admin/Desktop/实验8去噪5.19-/external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/`
- `C:/Users/Admin/Desktop/实验8去噪5.19-/external/NoisyGL/runs/nearest_work_full_grid_20260728/`
- `paper_pipeline/04_experiments/tie_safe_full_grid_experiment_report.md`
- `paper_pipeline/04_experiments/nearest_work_full_grid_final_report_20260729.md`
- `paper_final_zh_v29_human_style_20260727.md`
- `paper_final_zh_v31_tiesafe_nearestwork_20260728.md`

## 2. 正式 claim 冻结

### 允许的 headline

- 真实粒球分组相对随机等规模分组在 25/25 设置上取得正增量。
- GB-LCSS 相对 MLP+CE 在 24/25 设置上均值更高。
- GB-LCSS 相对 JoCoR 在 23/25 设置上均值更高。
- GB-LCSS 在统一协议下整体优于 PosteL-noIPL、ALS 和 SALS。
- 旧 `gsdn` adapted 结果只作为临时定位参照，不用于形成对 GSDN 的最终强弱 claim。
- 粒球是固定预算的局部共识实现，不是已被证明优于 kNN/k-means 的唯一构造。

### 禁止的 headline

- 旧规则下的 `14/15`、`24/25 overall best` 和旧 Holm 计数。
- GB-LCSS 在全部 X-only 最近工作中总体最优。
- 基础方法或蒸馏扩展全面超过 GSDN。
- 沿用 NoisyGL 完整评测协议。
- 不需要任何干净标签。
- 粒球优于 kNN 或 k-means。

## 3. 协议敏感性实验冻结设计

### 研究问题

当前结果在 clean validation 下得到。实验只检验：保持训练噪声实例、划分、超参数和测试集不变时，验证标签也受同率噪声污染是否改变主要相对结论。

### 数据和方法

- 数据集：`cora,citeseer,corafull`
- 噪声：`uniform 0.4`
- seeds：`0,1,2,3,4`
- 方法：
  - `frozengrace_gb_mlp_softdist_beta05`
  - `frozengrace_gb_mlp_softdist_beta05_shuffled`
  - `mlp_ce`
  - `mlp_jocor`
  - `postel_mlp_noipl_rm`
  - `gsdn`
- split：percent `0.2/0.1/0.7`
- tie mode：GB 两臂固定 `distribution_only`
- clean-validation 行：GB/CE/JoCoR 从 tie-safe v2 复用，PosteL 从最近工作网格复用，旧 `gsdn` 从 MLP-14/MLP-21 同协议记录复用
- noisy-validation 行：新增 `3×6×5=90` runs

### 噪声隔离

不能直接对 `train|val` 合并索引调用一次随机污染，否则训练节点的噪声实例可能改变。实现必须：

1. 使用当前 seed 和当前函数生成 train-only noisy labels。
2. 使用独立派生 seed `seed + 1000003` 只污染 validation labels。
3. 将第二步的 validation 部分合并到第一步的 noisy labels。
4. 记录：
   - `train_noise_hash`
   - `val_noise_hash`
   - `noise_hash`
   - `modified_train_count`
   - `modified_val_count`
   - `modified_test_count`
5. clean/noisy-validation 配对必须满足 `train_noise_hash` 完全一致。

### 预注册判据

主要机制结论保持：

- `GB-LCSS − shuffled > 0` 在 3/3 数据集上均值为正，且每个数据集至少 4/5 seeds 为正。
- `GB-LCSS − MLP+CE > 0` 至少 2/3 数据集均值为正且至少 4/5 seeds 为正。
- 任一数据集 `GB-LCSS − shuffled` 不得低于 `−0.5 pp`。

比较性结果：

- 分别报告 GB-LCSS 相对 JoCoR、PosteL 和 GSDN 的 clean-val/noisy-val 配对差。
- Kendall \(\tau\) 和前二名变化仅作描述，不作为唯一通过判据。
- 报告交互量
  \[
  I_m=
  [Acc_{\mathrm{GB}}-Acc_m]_{\mathrm{noisy\ val}}
  -
  [Acc_{\mathrm{GB}}-Acc_m]_{\mathrm{clean\ val}}.
  \]

结果处置：

- 若主要机制判据通过：正文写“核心分组增量对验证标签污染保持稳健”，完整表放附录。
- 若判据未通过：摘要、实验设置和结论醒目标注 clean-validation conditional，不调整算法或重搜超参数。

## 4. Task 1：建立 v32 证据登记表

**Files:**

- Create: `paper_pipeline/scripts/build_v32_evidence_registry.py`
- Create: `paper_pipeline/tests/test_v32_evidence_registry.py`
- Create: `paper_pipeline/00_evidence/v32_evidence_registry.csv`

- [x] **Step 1：编写失败测试，拒绝旧 tie 规则进入 current claim**

测试构造一条 `tie_mode=min_index` 的主表记录，期望 registry builder 抛出：

```text
legacy tie_mode cannot enter current evidence registry
```

- [x] **Step 2：运行失败测试**

```powershell
py -3 -m pytest paper_pipeline/tests/test_v32_evidence_registry.py -q
```

预期：FAIL，因为 builder 尚不存在。

- [x] **Step 3：实现 registry builder**

登记以下固定字段：

```text
evidence_id, manuscript_role, source_path, source_sha256,
method, dataset_scope, noise_scope, tie_mode, protocol,
headline_allowed, limitations
```

正式源仅允许：

```text
tie_safe_full_grid_20260727_v2_immutable
nearest_work_full_grid_20260728
```

历史 LOO、kNN、H-access 和蒸馏结果必须标记 `headline_allowed=0`。

- [x] **Step 4：运行测试并生成 registry**

```powershell
py -3 -m pytest paper_pipeline/tests/test_v32_evidence_registry.py -q
py -3 paper_pipeline/scripts/build_v32_evidence_registry.py
```

预期：

- 测试 PASS；
- registry 包含 tie-safe 500 行和最近工作 625 行的聚合证据；
- 不含旧 min-index 主表 claim。

## 5. Task 2：增加 `train_val` 噪声范围

**Files:**

- Modify: `C:/Users/Admin/Desktop/实验8去噪5.19-/external/NoisyGL/scripts/run_standard_downstream.py`
- Modify: `C:/Users/Admin/Desktop/实验8去噪5.19-/external/NoisyGL/tests/test_standard_runner_baselines.py`

- [x] **Step 1：编写失败测试**

测试必须验证：

```python
assert train_hash_train_only == train_hash_train_val
assert modified_val_count_train_only == 0
assert modified_val_count_train_val > 0
assert modified_test_count_train_val == 0
```

- [x] **Step 2：运行失败测试**

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" `
  -m pytest tests/test_standard_runner_baselines.py -q
```

工作目录：

```text
C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL
```

预期：FAIL，parser 尚不接受 `train_val`。

- [x] **Step 3：实现独立两阶段污染**

`--noise_scope` choices 改为：

```python
choices=["train", "train_val", "all"]
```

`train_val` 必须先生成 train-only 版本，再使用派生 seed 单独生成 validation 噪声并合并，不得改变 train 节点 noisy labels。

- [x] **Step 4：增加 CSV 字段**

```text
train_noise_hash,val_noise_hash,modified_train_count,
modified_val_count,modified_test_count
```

- [x] **Step 5：运行测试**

预期所有 standard runner 测试 PASS，且原 `noise_scope=train` 行为逐元素不变。

## 6. Task 3：审计旧 GSDN 的验证标签来源

**Files:**

- Read-only: `C:/Users/Admin/Desktop/实验8去噪5.19-/external/NoisyGL/predictor/GraphFreeFamily_Predictor.py`
- Modify: `C:/Users/Admin/Desktop/实验8去噪5.19-/external/NoisyGL/tests/test_mlp14_graphfree_family.py`

- [x] **Step 1：编写失败测试**

锁定旧 `gsdn` 的训练循环使用 `self.noisy_label` 计算 validation 指标。测试不得把 `gsdn_faithful_core` 混入本轮身份。

- [x] **Step 2：运行失败测试**

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" `
  -m pytest tests/test_mlp14_graphfree_family.py -q
```

- [x] **Step 3：最小处置**

当前实现已经调用 `self.evaluate(self.noisy_label, self.val_mask)`，因此不修改 predictor。若测试发现行为回退，才做最小修复。

- [x] **Step 4：运行全部相关测试**

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" `
  -m pytest tests/test_mlp14_graphfree_family.py tests/test_standard_runner_baselines.py -q
```

预期：PASS。

## 7. Task 4：实现协议敏感性编排器

**Files:**

- Create: `paper_pipeline/scripts/run_validation_scope_sensitivity.py`
- Create: `paper_pipeline/tests/test_validation_scope_sensitivity.py`
- Create: `paper_pipeline/04_experiments/validation_scope_sensitivity_plan_20260729.md`

- [x] **Step 1：写计划文档**

写入本计划第 3 节的假设、方法、90 个新增 run、判据和两个结果分支。文档落盘后计算 SHA256，训练启动后不得修改原文件。

- [x] **Step 2：编写编排器测试**

测试 dry-run 必须生成：

```text
3 datasets × 6 methods × 5 seeds = 90 rows
```

并验证：

- GB 两臂带 `gb.tie_mode=distribution_only`；
- 所有命令使用 `noise_scope=train_val`；
- 所有命令使用 uniform 0.4；
- exact resume 不重复启动已完成命令；
- 输出含 immutable raw、manifest、protocol audit。

- [x] **Step 3：运行失败测试**

```powershell
py -3 -m pytest paper_pipeline/tests/test_validation_scope_sensitivity.py -q
```

- [x] **Step 4：实现编排器**

复用：

- `run_tie_safe_full_grid.py` 的 immutable raw、hash、resume 和 paired tests；
- `run_nearest_work_full_grid.py` 的最近工作身份审计；
- clean-validation 正式行只读合并，不复制或重写。

- [x] **Step 5：dry-run**

```powershell
py -3 paper_pipeline/scripts/run_validation_scope_sensitivity.py `
  --python "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" `
  --noisygl-root "C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL" `
  --output-dir "C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\runs\validation_scope_sensitivity_20260729" `
  --dry-run
```

预期：

- 90 个 planned rows；
- 0 个训练进程；
- 所有命令和预期字段通过审计。

## 8. Task 5：运行协议敏感性实验

**Files:**

- Output: `C:/Users/Admin/Desktop/实验8去噪5.19-/external/NoisyGL/runs/validation_scope_sensitivity_20260729/`
- Create: `paper_pipeline/04_experiments/validation_scope_sensitivity_report.md`

- [x] **Step 1：运行 seed0 smoke**

先临时限制 seeds 为 `0`，执行 18 个 run。

通过条件：

- 18/18 `status=ok`；
- `modified_test_count=0`；
- train-only 与 train-val 的 `train_noise_hash` 一致；
- validation 污染率接近 0.4；
- 所有方法验证阶段读取 noisy validation labels；
- 无 NaN/OOM。

- [x] **Step 2：补 seeds 1–4**

```powershell
py -3 paper_pipeline/scripts/run_validation_scope_sensitivity.py `
  --python "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" `
  --noisygl-root "C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL" `
  --output-dir "C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\runs\validation_scope_sensitivity_20260729"
```

- [x] **Step 3：exact resume**

原命令再次执行，预期启动 0 个训练子进程。

- [x] **Step 4：生成报告**

输出：

```text
raw_combined.csv
aggregate.csv
paired_deltas.csv
paired_tests.csv
protocol_interactions.csv
protocol_audit.csv
completeness.csv
command_manifest.csv
immutable_raw_manifest.csv
validation_scope_sensitivity_report.md
```

- [x] **Step 5：按预注册判据判定**

不得根据结果调整阈值、方法或数据集。

## 9. Task 6：生成三层资源表

**Files:**

- Create: `paper_pipeline/scripts/generate_v32_tables.py`
- Create: `paper_pipeline/tests/test_generate_v32_tables.py`
- Create: `paper_pipeline/03_tables/v32_*.md`

- [x] **Step 1：编写资源层测试**

固定约束：

```text
Table A 不含 GSDN/PosteL/ALS/SALS
Table B 必含旧 GSDN adapted/PosteL/ALS/SALS/GB-LCSS
Table C 明确标记 teacher、全节点蒸馏和推理输入
```

测试同时拒绝：

- 将 adapted 方法写成 official reproduction；
- 将旧 min-index 数字放进 Table A/B；
- 将历史蒸馏结果写成对 GSDN 的正式胜利。

- [x] **Step 2：实现表生成**

表 A 使用 tie-safe v2。

表 B 中 PosteL/ALS/SALS 使用正式最近工作网格；旧 `gsdn` 使用 MLP-14/MLP-21 同协议记录，并在方法名和脚注中标记为 provisional adapted implementation。不得引用 `gsdn_faithful_core` 的 10/25 或均值差作为本轮 current claim。

表 C 保留历史蒸馏表，但表注明：

```text
historical extension evidence; not a current formal comparison against GSDN
```

- [x] **Step 3：生成 headline JSON**

必须精确包含：

```json
{
  "vs_shuffled_positive_settings": "25/25",
  "vs_mlp_ce_positive_settings": "24/25",
  "vs_jocor_positive_settings": "23/25",
  "vs_postel_mean_pp": 5.202,
  "vs_als_mean_pp": 7.215,
  "vs_sals_mean_pp": 4.202
}
```

- [x] **Step 4：运行测试**

```powershell
py -3 -m pytest paper_pipeline/tests/test_generate_v32_tables.py -q
py -3 paper_pipeline/scripts/generate_v32_tables.py
```

## 10. Task 7：编写 v32 中文稿

**Files:**

- Create: `paper_final_zh_v32_submission_candidate_202607xx.md`
- Modify: `paper_pipeline/00_evidence/claim_evidence_ledger.md`

- [x] **Step 1：复制 v29 作为语言底稿**

不得从 condensed 或 v31 整篇复制。v31 只作为事实源。

- [x] **Step 2：更新方法节**

写入 tie-safe 规则：

- 非平票：经验分布与唯一多数 one-hot 按 \(\beta=0.5\) 混合；
- 平票：仅使用经验分布；
- `alpha` 称为 normalized consensus weight / 归一化共识权重，不称校准置信度。

- [x] **Step 3：更新实验协议**

明确：

- 噪声生成形式参考 NoisyGL；
- 本文使用自定义 percent 20/10/70；
- 默认仅污染训练标签；
- clean validation 用于 model selection；
- transductive；
- test labels 仅用于最终评估。

- [x] **Step 4：插入三层表**

表 A 回答训练节点目标构造。

表 B 回答全部结构辅助 X-only 最近工作。

表 C 回答更高资源蒸馏扩展。

- [x] **Step 5：恢复完整审计性证据**

保留并压缩：

- LOO；
- kNN/k-means；
- H-access；
- smoothing；
- shuffled；
- 失败数据集；
- 计算与部署开销。

PubMed 不在正文分析中展开。所有 PubMed 表格、诊断数字、失败解释和筛选时间线统一移入附录“PubMed 边界案例”。正文最多在数据范围或局限中用一句话指向该附录，不出现 PubMed 专门小节。

- [x] **Step 6：写入协议敏感性结果**

严格按 Task 5 的两个结果分支书写。

- [x] **Step 7：更新摘要**

推荐结构：

1. 任务和 transductive/clean-validation 范围；
2. GB-LCSS 方法；
3. 25/25 shuffled 机制结果；
4. 优于 PosteL/ALS/SALS；
5. 局限一句。

摘要不写 overall SOTA 或 14/15。

- [x] **Step 8：更新结论和局限**

至少披露：

- synthetic noise；
- clean validation；
- transductive；
- fixed frozen representation；
- 旧 GSDN 仅为 adapted 临时参照，最终忠实实现尚待独立更新；
- 粒球可由其他分组近似替代。

## 11. Task 8：理论附录

**Files:**

- Modify: `paper_final_zh_v32_submission_candidate_202607xx.md`

- [x] **Step 1：加入排序保持命题**

对 \(C\) 类对称噪声：

\[
\tilde{\pi}_c
=
(1-\rho)\pi_c+
\frac{\rho}{C-1}(1-\pi_c).
\]

因此：

\[
\tilde{\pi}_a-\tilde{\pi}_b
=
\left(1-\frac{C\rho}{C-1}\right)(\pi_a-\pi_b).
\]

当 \(\rho<(C-1)/C\) 时，期望排序保持。

- [x] **Step 2：写清假设**

- 球内真实分布具有正 margin；
- 标签噪声独立且对称；
- ball assignment 不由标签生成；
- 结论不覆盖 pair/instance noise；
- 命题不证明 GRACE 空间中的球一定语义纯净。

- [x] **Step 3：加入有限样本说明**

使用 ball 内训练票数和观测 margin 给出 Hoeffding 型错误概率上界。推导只解释票数、margin 和噪声率的关系，不包装为完整泛化理论。

## 12. Task 9：P1 alpha 诊断

**Files:**

- Create: `paper_pipeline/scripts/diagnose_v32_alpha_calibration.py`
- Create: `paper_pipeline/tests/test_v32_alpha_calibration.py`
- Create: `paper_pipeline/04_experiments/v32_alpha_calibration_report.md`

- [x] **Step 1：只做目标后处理**

复用冻结 H、split、noise 和 ball assignment，不训练分类器。

- [x] **Step 2：输出**

```text
alpha_bin,count,mean_alpha,target_argmax_accuracy,
clean_label_mass,brier,mean_train_votes,tie_rate
```

- [x] **Step 3：判读**

若 alpha 与 target correctness 基本单调，正文称其为可解释启发式权重。

若不单调，只在局限中披露；本轮不得据此更换默认公式。

## 13. Task 10：P1 分类与校准指标

默认范围：

- 五个核心数据集；
- uniform 0.4；
- GB-LCSS、shuffled、MLP+CE、JoCoR、GSDN、PosteL；
- seeds 0–4。

先检查是否有可复用 logits/checkpoint。没有时需要 150 个重跑，不得声称是零成本后处理。

输出：

- accuracy；
- macro-F1；
- balanced accuracy；
- ECE；
- Brier score。

该任务在 P0 完成后单独请求用户确认。

## 14. Task 11：P1 构造成本

同一机器、同一 H、单线程 CPU 和 GPU 分开报告：

- granular ball；
- kNN；
- k-means。

指标：

- wall-clock；
- peak RAM/VRAM；
- 构造输出大小；
- 是否需要逐节点近邻查询；
- 超参数/候选规模。

不得直接引用 2019 年 GB 分类器复杂度替代本实现测量。

## 15. Task 12：新 GSDN 正式比较（本轮暂停）

本轮不执行。旧 `gsdn` 只服务于协议敏感性和临时资源表定位。待用户提供新的、更贴合实验协议的 GSDN 版本后，另建执行计划，重新冻结方法身份、资源使用、配置和比较范围。v32 不得据此声称基础方法或蒸馏扩展超过 GSDN。

## 16. Task 13：复现资产

- [ ] 生成匿名新仓库，不保留原 git 历史。
- [ ] 清理绝对路径、用户名、机器名和中文实验目录。
- [ ] 提供环境、命令、配置、raw CSV、统计脚本和 H manifest。
- [ ] 为所有正式文件计算 SHA256。
- [ ] 在本地空目录按 README 复现一个 seed0 smoke 和一次 summarize-only。
- [ ] 论文中的仓库 URL、Data Availability、许可证和 checkpoint 边界改为真实值。

## 17. Task 14：v32 自动审计

**Files:**

- Create: `paper_pipeline/scripts/audit_v32_manuscript.py`
- Create: `paper_pipeline/tests/test_audit_v32_manuscript.py`

审计必须拒绝：

- `14/15`
- 旧 `+7.16/+7.83/+2.87`
- “沿用 NoisyGL 协议”
- “无需干净标签”
- “全面优于 GSDN”
- “粒球优于 kNN”
- `表 0`
- placeholder URL
- 本地绝对路径

审计必须验证：

- 三层表资源标注；
- 25/25、24/25、23/25；
- PosteL/ALS/SALS 正式差值；
- `distribution_only`；
- PubMed 仅出现在附录边界案例及正文的附录指引中；
- adapted/reimplementation 身份；
- 附录与正文表号一致。

运行：

```powershell
py -3 -m pytest paper_pipeline/tests/test_audit_v32_manuscript.py -q
py -3 paper_pipeline/scripts/audit_v32_manuscript.py `
  --manuscript paper_final_zh_v32_submission_candidate_202607xx.md
```

预期：0 errors。

## 18. Task 15：无上下文冷审

只向审稿窗口提供：

- v32 完整中文稿；
- 完整附录；
- 不提供项目历史、失败路线或本次争论记录。

四个审稿角色：

1. 方法正确性与统计；
2. 最近工作、新颖性和资源公平；
3. 实验协议与可复现性；
4. 仅检查数字、表号和 claim 一致性。

冷审结果只允许触发：

- 事实修正；
- 表述收紧；
- 已有证据重排；
- 真正阻塞投稿的复现问题。

不得因为审稿人泛泛提出“再加数据集/再加模型”而自动开启新实验。

## 19. 最终停止标准

满足以下全部条件后停止修改并进入英文稿：

- v32 只使用 tie-safe v2 和正式最近工作网格作为 current evidence；
- noisy-validation 敏感性结果已披露；
- NoisyGL、clean validation、transductive 和 synthetic noise 表述准确；
- 旧 GSDN 位于正文结构辅助 X-only 表并标记为 provisional adapted，未被包装为忠实复现；
- 粒球可替代性、LOO、H-access、失败数据集和开销均保留；
- 理论命题假设明确；
- 自动审计 0 errors；
- 匿名仓库 smoke 可运行；
- 冷审无新的事实性 Major。

---

## 执行顺序与预计新增运行量

| 阶段 | 内容 | 新增训练运行 | 是否阻塞 v32 |
|---|---|---:|:---:|
| P0-A | 证据 registry、表生成、稿件谱系 | 0 | 是 |
| P0-B | noisy-validation 协议敏感性 | 90 | 是 |
| P0-C | v32、自动审计、复现资产 | 0 | 是 |
| P1-A | alpha 诊断、理论附录 | 0 个训练 | 建议 |
| P1-B | macro-F1/ECE | 最多 150 | 用户另行确认 |
| P1-C | 构造成本 | 仅构造 | 建议 |
| P1-D | 新 GSDN 正式重实现与比较 | 未冻结 | 本轮暂停 |
| Rebuttal | NRGNN 蒸馏、异质图、10 seeds、inductive | 未冻结 | 不做 |

## 自检

- 计划没有将旧 min-index 数字混入 current evidence。
- 旧 GSDN 被放入正文结构辅助表并明确标为临时 adapted 实现，未强行塞进训练节点目标构造表。
- noisy-validation 实验保持 train noise 实例不变，避免把两个变化混为一谈。
- 新 GSDN 的重实现与正式比较已从本轮移除，不形成任何强弱 claim。
- 理论只解释对称独立噪声下的局部投票，不扩张到全部噪声类型。
- PubMed 全部实验迁入附录透明呈现，不生成 24/30 混合分母。
