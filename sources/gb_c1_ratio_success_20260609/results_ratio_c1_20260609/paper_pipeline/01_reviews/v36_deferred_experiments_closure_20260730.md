# v36 延期实验闭环

日期：2026-07-30

目标稿件：`paper_final_zh_v36_experiments_completed_20260730.md`

## 1. 本轮范围

本轮只完成 v35 明确延期的两项工作：

1. 补齐 representation-source comparison，并按预注册的几何塌缩规则汇总。
2. 使用 `exclude_self=true` 重跑 Cora、CiteSeer 和 CoraFull 上的 kNN 教师，共 30 条正式记录。

主实验网格、主方法、GSDN、数据集范围和其他教师臂均未改动。

## 2. 表征来源比较

### 协议

- 数据集：Cora、CiteSeer、CoraFull、Coauthor-CS、Amazon-Photo。
- 噪声：train-only uniform 0.4。
- 划分：20%/10%/70%。
- 粒球参数、并列安全目标和 X-only MLP 全部固定。
- 只替换构球空间：原始 \(X\)、两步传播 \(\hat A^2X\) 和 GRACE \(H\)。
- seed 0 形成单球时按预注册规则停止，不为获得重复分类分数而扩种子。

### 结果

- 原始 \(X\) 在五个数据集上均形成单球。
- \(\hat A^2X\) 在 Cora、CiteSeer 和 CoraFull 上同样形成单球。
- \(\hat A^2X\) 在 Coauthor-CS 和 Amazon-Photo 上能够形成 57.2 和 64.0 个球，但五种子精度分别为 89.25±0.58 和 80.41±1.12。
- 对应 GRACE 结果为 90.62±0.47 和 81.33±0.70，\(\hat A^2X\) 分别低 1.37 和 0.92 个百分点，正向 seed 数为 0/5 和 1/5。
- 含噪标签训练的 GCN 隐层只作 seed 0 诊断，不与标签无关来源作正式排名。

### 结论

在本文固定构球准则下，GRACE 在所比较的标签无关来源中提供了更稳定的非塌缩局部几何。该结果支持“表征可聚类性是粒球有效的前提”，但不证明 GRACE 在所有自监督表征中唯一最优。

## 3. 排除自身的 kNN 教师

### 协议与完整性

- 数据集：Cora、CiteSeer、CoraFull。
- 噪声：train-only uniform 0.4 和 0.6。
- seeds：0--4。
- 新增记录：30/30，全部 `status=ok`。
- 所有行满足 `knn_exclude_self=1`、`target_beta=0`、验证/测试标签修改数为 0。
- 其余四个教师臂只在逐 seed 的 split/noise hash 与 corrected kNN 行完全一致后复用。
- 合并后的五臂网格为 150/150 条。

### corrected kNN 学生精度

| 数据集 | u0.4 | u0.6 |
|---|---:|---:|
| Cora | 81.61±0.74 | 71.80±2.63 |
| CiteSeer | 69.65±1.61 | 62.90±2.26 |
| CoraFull | 58.58±0.55 | 52.65±0.93 |

### 球式教师相对 corrected kNN

| 数据集 | u0.4 | u0.6 |
|---|---:|---:|
| Cora | −1.01pp，0/5 正向 | −0.53pp，2/5 正向 |
| CiteSeer | −0.25pp，2/5 正向 | −0.62pp，1/5 正向 |
| CoraFull | −0.36pp，1/5 正向 | −0.83pp，0/5 正向 |

### 结论

排除查询节点后，kNN 教师仍在 6/6 个聚合设置上数值高于球式教师，优势为 0.25--1.01 个百分点。因此，历史 kNN 优势不能归因于自包含错误。教师蒸馏实验支持“局部共识目标可以提高 X-only 学生精度”，但不支持“球式教师比逐点 kNN 教师更准确”。粒球在本文中的可辩护定位仍是固定预算、一次性分区的局部共识接口。

## 4. 稿件与证据更新

- 新稿：`paper_final_zh_v36_experiments_completed_20260730.md`。
- 新证据登记表：`paper_pipeline/04_experiments/v36_evidence_registry_20260730.csv`。
- 表 5 恢复 corrected kNN 教师行，并将标题改为五臂比较。
- 第 5.3.5 节与附录 D.3 加入表征来源比较。
- 附录 G 更新为 20 条表征诊断和 30/30 条 corrected kNN 教师记录。
- v35 保留未覆盖。

## 5. 验证

- MLP-43 实验与汇总测试：11 passed。
- v36 稿件与证据登记测试：4 passed。
- v36 稿件和登记表审计：0 errors。
- Elsevier 写作 lint：未发现配置中的 AI 写作模式。
- lint 单元测试：2 passed。
- Markdown 表格：22 个表格块，0 个列数不一致。
- 引用：正文引用均有条目，参考文献编号连续为 1--62。

## 6. 文件指纹

- v36 稿件：
  `94DFED1B14FE609933BFCE041C2C231409CE8594EE0108B9CFA8CFC03C900E75`
- v36 证据登记表：
  `C092E42876DC6C17AD896FC5F7E9242CE334599F120BAAFEB1273AA03D1081CA`
- representation-source raw：
  `A3104A8203F07BD4150B09AE2118925C57631B13CAC00BEB696DA73711639C8E`
- corrected five-arm teacher raw：
  `685378089198DC2FCC0DB1DD715BDDFFE6638E05A52337CF6F3CDF04070E7E7B`

## 7. 最终决策

两项延期实验均已关闭。v36 使用完成且哈希核验通过的证据，不再引用不完整表征结果或包含查询节点自身的历史 kNN 教师结果。
