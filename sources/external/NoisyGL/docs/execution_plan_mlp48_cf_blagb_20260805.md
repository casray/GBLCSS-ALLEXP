# MLP-48 CF-BLAGB 执行备案

## 1. 研究假设

MLP-46 证明，允许 20% 含噪训练标签决定粒球分裂是否保留，会把候选树过度剪枝并造成分区塌缩。MLP-48 检验一个更弱的假设：保持正式粒球数量不变时，跨折含噪标签能否帮助排序几何候选分裂，使固定预算优先分配给标签分布差异更明显的区域。

## 2. 粒球角色

粒球仍是冻结 GRACE 表征空间中的互斥局部共识单元。标签不决定是否停止构球，只对标签无关候选树中的分裂顺序施加有界修正。

## 3. 方法与冻结项

候选分裂得分为：

\[
S(a)=G(a)\left(1+\frac{\operatorname{JSD}(\bar p_l,\bar p_r)}{\log 2}\right),
\qquad G(a)\le S(a)\le 2G(a).
\]

- 使用五折 cross-fitting，held-out 折标签不参与该折节点的分区引导或局部投票。
- 最终球数固定为相同 seed 下正式几何方法的 K_geo。
- 候选树先精确复现正式几何树前缀，再以零增益阈值向下扩展到最多 128 个叶节点。
- GRACE H、`min_ball_size=32`、正式 `split_improvement=0.05`、LOO、`distribution_only`、beta=0.5 和 alpha 公式不变。
- clean label 只用于目标质量指标和 clean-guidance oracle。

## 4. 实现变更

- `utils/noise_aware_gb.py`
  - `BudgetedPartition`
  - `label_assisted_split_score()`
  - `build_prefixed_geometry_candidate_tree()`
  - `cut_candidate_tree_fixed_budget()`
  - `build_crossfit_budgeted_targets()`
- `scripts/run_mlp48_cf_blagb.py`
  - 七臂 Phase A 目标质量诊断
  - 协议、完整性和固定预算审计
  - 预注册 gate 与 Markdown 报告
- `tests/test_cf_blagb.py`
- `tests/test_mlp48_cf_blagb.py`

正式 predictor、YAML 和历史 MLP-46 结果均未修改。

## 5. 实验协议

- 数据集：Cora、CiteSeer、CoraFull。
- 噪声：uniform 0.4，仅污染 train。
- Split：0.2/0.1/0.7。
- Seed：0。
- 表征：已有 frozen GRACE H。
- 本阶段只构造目标，不训练分类器。

七个实验臂：

1. `formal_loo_gb`
2. `crossfit_formal_gb`
3. `fixed_budget_geometry_only`
4. `cf_blagb_shuffled_guidance`
5. `cf_blagb`
6. `cf_blagb_clean_oracle`
7. `crossfit_adaptive_knn`

## 6. 预注册成功与失败标准

进入 Phase B 必须同时满足：

1. 三数据集的每一折都恰好得到 K_geo 个球。
2. CF-BLAGB 在至少 2/3 数据集上同时低于 geometry-only 和 shuffled 的 Brier。
3. 相对 cross-fit formal GB 的平均 target accuracy 增量不为负。
4. 任一数据集 target accuracy 回退不超过 1pp。
5. clean oracle 在至少 2/3 数据集存在正 accuracy headroom。
6. 无泄漏、NaN、空目标或塌缩。

任一条件失败即停止，不调整折数、先验强度、JSD 乘子、球预算或第二版公式。

## 7. 实际执行

测试命令：

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" -m pytest `
  tests/test_cf_blagb.py tests/test_noise_aware_gb.py `
  tests/test_mlp48_cf_blagb.py tests/test_mlp46_cf_nagb.py -q
```

结果：`22 passed`，仅有一个 torch-geometric 上游弃用警告。

实验命令：

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" `
  scripts/run_mlp48_cf_blagb.py `
  --phase diagnostics `
  --datasets cora,citeseer,corafull `
  --seed 0 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --device cuda:0 `
  --output_dir runs/mlp48_cf_blagb_20260805
```

## 8. 实验结果

### 8.1 构造与协议

| Dataset | K_geo | Candidate leaves | Exact budget | Modified train | Modified val/test |
|---|---:|---:|---|---:|---:|
| Cora | 52 | 52 | 5/5 folds | 230 | 0/0 |
| CiteSeer | 64 | 65 | 5/5 folds | 294 | 0/0 |
| CoraFull | 64 | 128 | 5/5 folds | 1592 | 0/0 |

七臂在三个数据集上全部完整，共 21 行目标质量记录；四个固定预算臂共 60 行 fold-level 分区记录。

### 8.2 关键目标质量

| Dataset | Cross-fit formal Acc/Brier | Geometry-only Acc/Brier | CF-BLAGB Acc/Brier | Shuffled Acc/Brier | Clean oracle Acc/Brier | Adaptive-k Acc/Brier |
|---|---|---|---|---|---|---|
| Cora | .6470/.4919 | .6470/.4919 | .6470/.4919 | .6470/.4919 | .6470/.4919 | .6617/.3852 |
| CiteSeer | .5865/.5864 | .5865/.5864 | .5865/.5864 | .5850/.5871 | .5865/.5864 | .5910/.5240 |
| CoraFull | .5980/.6330 | .6016/.6595 | .6016/.6852 | .6016/.6852 | .6018/.6498 | .6488/.5017 |

Gate 结果：

- 固定预算：3/3 通过。
- CF-BLAGB 同时改善 geometry-only 与 shuffled Brier：0/3。
- 相对 cross-fit formal 的平均 target accuracy：+0.00118。
- 超过 1pp 的 accuracy 回退：0/3。
- clean oracle accuracy headroom：1/3。
- 最终判定：`pass=false`，报告主原因 `oracle_no_headroom`；Brier 控制门也独立失败。

## 9. 详细解释

### 已被实验支持的结论

1. 固定预算设计确实解决了 CF-NAGB 的分区塌缩。三个数据集所有 fold 均严格保持 K_geo。
2. 在当前候选树和 20% 含噪标签条件下，JSD 标签辅助没有改善 geometry-only 目标质量。Brier 双控制通过数为 0/3。
3. Cora 没有可重排空间：候选容量与 K_geo 同为 52，因此所有固定预算切法必然使用整棵树。
4. CiteSeer 只有一个额外候选叶，真实标签辅助最终选回 geometry-only 分区。
5. CoraFull 有充分重排空间，但 noisy 与 shuffled 得到相同 accuracy/Brier；只有 clean oracle 改善了 Brier，说明可利用信号未能从当前含噪票中稳定恢复。
6. Cross-fit adaptive-k 在三个数据集上均给出明显更低的 Brier，说明失败不是 GRACE H 中完全没有局部信息，而是固定层级候选树的标签重排没有提取出额外价值。

### 合理但尚未完全验证的解释

有界 JSD 乘子修复了“标签硬否决几何”的问题，但没有改变标签证据稀疏和受噪声污染的根本限制。Cora/CiteSeer 的候选层级本身也缺少足够替代切法，进一步压缩了潜在收益。

### 被负结果削弱的方向

“直接使用当前含噪训练标签改善粒球生成”已经连续被 CF-NAGB 的硬剪枝失败和 CF-BLAGB 的固定预算重排失败削弱。继续调整 JSD 权重、Dirichlet 强度、折数或候选阈值将构成事后调参，不再继续。

### 尚未验证的独立假设

去噪教师预测可能比原始含噪标签更适合作为分区引导，但这改变了信息源和算法规模，属于下一篇或独立方法，不是 CF-BLAGB 的第二版。

## 10. 最终决策

**放弃并关闭 CF-BLAGB。** 不进入 Phase B，不训练下游 MLP，不修改第一篇论文的正式 LOO-GB-LCSS。保留代码、测试和负结果，用于证明该方向已经在“硬剪枝”和“固定预算软重排”两个层面得到检验。

完整结果位于：`runs/mlp48_cf_blagb_20260805/`。
