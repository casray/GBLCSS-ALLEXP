# GB-LCSS 正式目标 tie 安全暴露审计

审计日期：2026-07-27  
审计对象：`paper_final_zh_v29_human_style_20260727.md` §3.4、MET-01、PKG-P1-09  
机器可读结果：`tie_safety_audit.csv`  
只读重建脚本：`../scripts/audit_tie_safety.py`

## 1. 审计范围与结论边界

本审计覆盖论文五个主结果数据集的完整正式目标网格：

- 数据集：Cora、CiteSeer、CoraFull、Coauthor-CS、Amazon-Photo；
- 设置：uniform 0.2、uniform 0.4、uniform 0.6、pair 0.4、instance 0.4；
- seeds：0–4；
- 共 `5 × 5 × 5 = 125` 个 `dataset × setting × seed` 目标实例。

本轮没有启动训练，没有改动模型实现、原始结果、论文正文、`material_manifest.md`、claim ledger 或 traceability。审计只重建正式 split、train-only noisy labels、固定 H 上的粒球 assignment 和训练软目标。

本报告只回答“当前最小索引 tie-break 在正式目标中暴露到什么程度”。它不能回答不同 tie-safe 规则会使最终 test accuracy 改变多少；后者需要条件性对照训练。

## 2. 公式与真实实现定位

v29 §3.4 定义

\[
p_{jc}=\frac{1}{|T_j|}\sum_{v\in T_j}\mathbb I(\tilde y_v=c),\quad
m_j=\arg\max_c p_{jc},\quad
q_j=(1-\beta)p_j+\beta e_{m_j},
\]

\[
\alpha_j=\max\left(0,\frac{r_j-1/C}{1-1/C}\right),\quad
t_v=(1-\alpha_j)e_{\tilde y_v}+\alpha_jq_j,
\]

其中正式基础方法为 `β=0.5`、`s_min=2`。v29 明确并列时使用实现中的 `argmax`，取索引最小类别。

真实实现位于：

- `external/NoisyGL/utils/frozen_grace_targets.py`：`torch.argmax(counts)`、multiclass α、distribution/majority 混合及节点软目标；
- `external/NoisyGL/utils/ordinary_gb.py`：固定 `64/32/0.05` 的几何粒球构造；
- `external/NoisyGL/config/frozengrace_gb_mlp_softdist_beta05/`：五数据集正式配置。

正式结果 lineage 为：

- Cora、CiteSeer、Coauthor-CS：`runs/mlp10_baseline_refresh_20260702/mlp10_combined_raw.csv`；
- Amazon-Photo：`runs/mlp12_amazonphoto_20260702/mlp12_combined_raw.csv`；
- CoraFull：`runs/mlp21_dataset_replacement_20260710/mlp21_raw.csv`；
- 最终选择入口：`runs/phase4_faithful_claims_20260712/phase4_selected_raw.csv`。

上述源码、配置、H 和正式结果文件的 SHA-256 已逐行写入审计 CSV。

## 3. 审计指标与 material 判据（先于结果冻结）

以下定义在查看 tie 结果前已冻结，脚本常量保留了门槛。

### 3.1 审计指标

对每个 `dataset × setting × seed`：

1. **eligible ball**：球内训练节点数 `|T_j| ≥ 2`。
2. **tie ball**：eligible ball 中，最高票数由至少两个类别共同取得。
3. **target-active tie ball**：tie ball 且 `α_j > 0`、`β > 0`，因此最小索引选择真正进入 `t_v`。
4. **affected train nodes**：位于 target-active tie ball 中的训练节点；占比分母为该行全部训练节点。
5. **affected Σα**：受影响训练节点的 α 之和；占比分母为该行全部训练节点的 α 之和。
6. **最小索引选中类别分布**：按实现的 1-based 类别编号，分别统计被选中的 tie 球数和受影响节点数。
7. **类别置换目标检查**：将类别编号反转 `π(c)=C+1-c`，重算目标，再逆置换回原类别顺序。该置换会把任意最高票 tie 中的原最小类别变成最大索引、原最大类别变成最小索引，因此是覆盖全部 active tie 的确定性见证。
8. **置换目标变化量**：训练节点目标的 L1 差异。对本方法有

\[
\sum_{v\in \text{affected}}\|t_v-\pi^{-1}t_v^\pi\|_1
=2\beta\sum_{v\in \text{affected}}\alpha_v.
\]

正式 `β=0.5` 时，L1 总量应等于 affected `Σα`。

### 3.2 material 判据

单个 `dataset × setting × seed` 行满足以下任一条件即记为 `material_flag=1`：

- affected train-node ratio `≥ 1%`；或
- affected `Σα / total Σα ≥ 1%`。

理由：前者防止大量低 α 节点被忽略，后者防止少量但高权重目标被节点占比稀释。低于门槛的非零 tie 仍记录为真实算法非等变暴露，只是不单独触发该行的 material 标记。

只要正式网格中存在 material 暴露，PKG-P1-09 的“仅定义/披露、不训练”分支就不再成立，应进入代表性 tie-safe 对照；是否扩展完整五种子网格取决于代表设置的性能敏感性。

## 4. 重建与复核方法

每行按原 runner 的顺序执行：

1. `setup_seed(seed)`；
2. 加载 dataset 并生成 percent `0.2/0.1/0.7` split；
3. 使用同一 `label_process` 生成 train-only noise；
4. 加载归档 frozen GRACE H；
5. 按正式配置构造几何粒球；
6. 用正式函数生成 soft target；
7. 用审计脚本中的独立公式再次生成 target 和 α；
8. 执行反转类别置换检查。

这只是确定性目标重放，没有进入 predictor 的训练循环。

## 5. 结果

### 5.1 总体暴露

- 125/125 行成功重建，`GAP=0`。
- 124/125 行出现 target-active tie；唯一经验证的真 0 为 `Coauthor-CS × uniform 0.2 × seed 3`。
- 共发现 914 个 tie balls，其中 913 个为 target-active；仅 1 个 tie ball 因 `α=0` 不改变目标。
- 类别反转后，124/125 行的目标发生变化，变化节点与 affected train nodes 逐行完全一致。
- 按正式目标实例计数，受影响训练目标为 `12,336 / 259,000 = 4.763%`。这里的分母会跨 setting/seed 重复同一数据集节点，表示“正式目标实例”，不是唯一物理节点数。
- affected `Σα = 2,212.598966`，占 total `Σα = 98,836.933487` 的 `2.239%`。
- 108/125 行达到预冻结 material 门槛。
- 23/25 个 `dataset × setting` 至少有一个 material seed；20/25 个 setting 的 5 个 seeds 全部 material。
- 仅 `Coauthor-CS × uniform 0.2` 与 `Coauthor-CS × uniform 0.4` 没有 material seed，但其中仍分别有 4/5 和 5/5 个 seeds 出现非等变 tie。

最强单行暴露是 `Cora × uniform 0.6 × seed 2`：174/541 个训练节点受影响（32.163%），承载该行 18.044% 的 α 质量。该量级不能称为“罕见边界条件”。

### 5.2 dataset × setting 五种子汇总

下表把五个 seed 的目标实例合并；“material seeds”仍按第 3.2 节逐行判定。

| Dataset | Setting | Tie balls | Active tie balls | Affected nodes | Node % | Affected Σα | Σα % | Material seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Amazon-Photo | instance 0.4 | 22 | 22 | 213/7650 | 2.784 | 43.857145 | 1.237 | 5/5 |
| Amazon-Photo | pair 0.4 | 44 | 44 | 671/7650 | 8.771 | 212.714287 | 5.705 | 5/5 |
| Amazon-Photo | uniform 0.2 | 18 | 18 | 215/7650 | 2.810 | 55.000002 | 1.110 | 5/5 |
| Amazon-Photo | uniform 0.4 | 28 | 28 | 268/7650 | 3.503 | 53.142859 | 1.513 | 5/5 |
| Amazon-Photo | uniform 0.6 | 57 | 57 | 564/7650 | 7.373 | 90.857143 | 4.220 | 5/5 |
| CiteSeer | instance 0.4 | 54 | 54 | 399/3325 | 12.000 | 77.399999 | 6.167 | 5/5 |
| CiteSeer | pair 0.4 | 56 | 56 | 502/3325 | 15.098 | 129.999998 | 9.300 | 5/5 |
| CiteSeer | uniform 0.2 | 33 | 33 | 226/3325 | 6.797 | 48.400001 | 2.906 | 5/5 |
| CiteSeer | uniform 0.4 | 58 | 58 | 475/3325 | 14.286 | 89.799999 | 7.440 | 5/5 |
| CiteSeer | uniform 0.6 | 91 | 90 | 799/3325 | 24.030 | 115.000002 | 13.167 | 5/5 |
| Coauthor-CS | instance 0.4 | 9 | 9 | 106/18330 | 0.578 | 20.285713 | 0.220 | 1/5 |
| Coauthor-CS | pair 0.4 | 21 | 21 | 453/18330 | 2.471 | 172.285710 | 1.813 | 3/5 |
| Coauthor-CS | uniform 0.2 | 6 | 6 | 52/18330 | 0.284 | 14.499999 | 0.114 | 0/5 |
| Coauthor-CS | uniform 0.4 | 8 | 8 | 82/18330 | 0.447 | 17.714286 | 0.191 | 0/5 |
| Coauthor-CS | uniform 0.6 | 17 | 17 | 193/18330 | 1.053 | 31.214287 | 0.536 | 4/5 |
| Cora | instance 0.4 | 33 | 33 | 251/2705 | 9.279 | 51.500000 | 4.325 | 5/5 |
| Cora | pair 0.4 | 36 | 36 | 370/2705 | 13.678 | 113.333331 | 8.835 | 5/5 |
| Cora | uniform 0.2 | 14 | 14 | 106/2705 | 3.919 | 27.833333 | 1.774 | 5/5 |
| Cora | uniform 0.4 | 32 | 32 | 217/2705 | 8.022 | 46.666665 | 4.083 | 5/5 |
| Cora | uniform 0.6 | 78 | 78 | 690/2705 | 25.508 | 105.500002 | 14.295 | 5/5 |
| CoraFull | instance 0.4 | 37 | 37 | 810/19790 | 4.093 | 101.884060 | 2.319 | 5/5 |
| CoraFull | pair 0.4 | 42 | 42 | 1180/19790 | 5.963 | 182.753624 | 3.998 | 5/5 |
| CoraFull | uniform 0.2 | 25 | 25 | 695/19790 | 3.512 | 116.739130 | 1.988 | 5/5 |
| CoraFull | uniform 0.4 | 35 | 35 | 771/19790 | 3.896 | 112.594203 | 2.552 | 5/5 |
| CoraFull | uniform 0.6 | 60 | 60 | 2028/19790 | 10.248 | 181.623188 | 6.337 | 5/5 |

### 5.3 最小索引选中类别分布

以下为所有 active tie balls 中，被当前 `argmax` 选中的 1-based 类别及球数。逐 `dataset × setting × seed` 的球数分布和对应受影响节点分布保存在 CSV。

- Amazon-Photo：`{1:32, 2:44, 3:18, 4:28, 5:15, 6:19, 7:13}`。
- CiteSeer：`{1:101, 2:83, 3:57, 4:28, 5:22}`。
- Coauthor-CS：`{1:7, 2:21, 3:8, 4:2, 5:3, 6:1, 7:1, 8:2, 10:3, 11:1, 13:11, 14:1}`。
- Cora：`{1:59, 2:40, 3:47, 4:31, 5:14, 6:2}`。
- CoraFull：`{1:10, 2:1, 3:5, 4:14, 5:4, 6:7, 7:1, 8:9, 9:24, 10:4, 11:2, 12:8, 13:2, 14:12, 15:4, 17:3, 19:3, 20:11, 21:16, 23:6, 25:3, 27:2, 31:2, 32:10, 34:1, 35:5, 36:1, 37:1, 39:3, 40:2, 41:8, 43:1, 44:3, 46:1, 48:2, 49:2, 50:1, 52:1, 55:1, 58:2, 65:1}`。

这不是类先验校准后的统计量，不能仅凭低索引类别出现较多就估计 accuracy bias；但最高编号类别在多类 tie 中不可能被“最小索引”选中，且类别反转直接证明目标不具类别置换等变性。

### 5.4 类别置换目标检查

- 124 个含 active tie 的正式目标行全部被类别反转改变。
- 改变节点数逐行与 affected train nodes 完全相等，总计 12,336。
- 置换前后 α 逐元素相同，最大绝对误差为 0。
- 观测 L1 与 `2β × affected Σα` 的最大绝对误差为 `7.63×10^-6`，属于 float32 求和误差。
- 唯一等变行 `Coauthor-CS × uniform 0.2 × seed 3` 同时也是唯一 `tie_balls=0` 的行；这是真实 0，不是缺资产被填成 0。

## 6. 独立复核与证据强度

复核结果：

- 正式 split/noise hash：100/100 个有归档 hash 的目标行全部匹配；
- modified-train count：50/50 个有该字段的正式行全部匹配；
- actual train-noise rate：100/100 个有该字段的正式行匹配，最大绝对误差 `2.78×10^-8`；
- uniform 0.4 的归档目标锚点：25/25 个 `dataset × seed` 的球数全部匹配；
- uniform 0.4 mean α：25/25 匹配，最大绝对误差 `5.96×10^-8`；
- 独立目标公式与正式 builder：全 125 行通过，最大元素误差 `1.19×10^-7`；
- GAP：0 行。

证据强度分层：

- 100 行：`reconstructed_hash_verified`；
- CoraFull uniform 0.4 的 5 行：`reconstructed_uniform_anchor_verified`；
- CoraFull 其余 20 行：`reconstructed_modified_count_verified_no_archived_hash`。

CoraFull 后 20 行的正式 raw 没有 split/noise hash，这是 lineage 弱点，但不是无法审计的 gap：split/noise/构球代码、dataset cache、H、正式配置与 seed 均存在，修改计数与正式 raw 一致，且同一数据集 uniform 0.4 的五 seed 目标锚点完全对齐。报告不把这 20 行提升为“原始 target tensor 逐字节归档”，只称为确定性重建。

## 7. 决策：是否需要新训练

**需要，但只需按 PKG-P1-09 启动代表性 tie-safe 对照，不应立即重跑全部主实验。**

理由不是“数学上可能有 tie”，而是正式目标中已经出现广泛且 material 的经验暴露：108/125 行过门，23/25 个 setting 至少一个 seed 过门，最高单行影响 32.163% 的训练目标和 18.044% 的 α 质量。因此“tie=0 或可忽略，只改定义并披露”的分支已被否定。

最小下一步应保持 split、noise realization、H、球 assignment、模型和评估全部不变，只替换 tie 处理：

1. 当前最小索引规则；
2. tie ball 令 `α=0`；
3. tie ball 保留 `p_j`、去掉 majority one-hot，即该球取 `β=0`。

先在高暴露代表设置做五 seed 配对，例如 `Cora uniform 0.6`，并加入类别数高、规模更大的 `CoraFull uniform 0.6` 作为异质性检查。只有 test 结果对 tie-safe 规则敏感，才扩展完整 25-setting 网格。

无论后续性能是否敏感，方法定义都应改为 tie-safe，并将 `α` 从未经样本量/类先验校准的“置信度”改称“启发式共识权重”。本审计支持的是目标暴露与非等变性，不支持声称现有 test accuracy 已被某个确定方向污染。

## 8. 复现命令

```powershell
& 'C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe' `
  '.\paper_pipeline\scripts\audit_tie_safety.py' `
  --output '.\paper_pipeline\00_evidence\tie_safety_audit.csv'
```

该命令只重建目标并写审计 CSV，不执行训练。
