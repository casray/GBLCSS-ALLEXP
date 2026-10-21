# GB-LCSS 投稿代码与论文一致性审计

日期：2026-09-16。本报告仅供作者内部处理投稿材料，不放入匿名仓库。

## 1. 结论与审计范围

**当前找到的投稿包不宜原样上传。** 核心粒球和软目标计算能够对应原始实验实现，但外部比较算法存在结果与实现身份错配、参数说明错误、训练入口缺失，以及影响固定种子结果的执行细节差异。

本轮没有修改论文、原始实验工程或投稿代码，没有重跑完整实验网格。新增文件仅为本报告及临时审计脚本、检查结果。

| 对象 | 版本与位置 |
|---|---|
| 投稿代码 | D:/papercode_final_v3 |
| 最新 Word | C:/Users/Admin/Downloads/GB-LCSS_final_submission_revised_20260915.docx |
| 最新 LaTeX | 工作区 GB-LCSS_Elsevier_LaTeX_submission_revised_20260915/main.tex 及 tables/ |
| 原始工程 | C:/Users/Admin/Desktop/实验8去噪5.19-/external/NoisyGL |
| GRACE 原始 checkpoint | C:/Users/Admin/Desktop/实验8去噪5.19-/gb_graph_aware_minival_6_1/checkpoints/ |
| 临时检查记录 | 工作区 .codex_tmp/repository_audit_20260916/ |

工作区为 C:/Users/Admin/Desktop/实验8去噪5.19-/gb_c1_ratio_success_20260609/results_ratio_c1_20260609。下文“论文路径”相对于上述 LaTeX 目录；“原始工程路径”相对于 NoisyGL。

投稿目录有 110 个参与快照比较的文件，不含 Python 字节码缓存。审计前后 SHA-256 完全一致。本轮未找到可确认与此目录对应的代码 ZIP，不能认证 ZIP 与目录一致。论文 LaTeX ZIP 不是代码 ZIP。

## 2. 必须处理的问题

### 2.1 Graph-MLP：当前数字来自旧代理，文字却写成多跳 NContrast

当前 Table 3 写的是 Multi-hop neighborhood-contrastive objective，Appendix F.1 写的是 Multi-hop NContrast objective。但是全部 25 条对应逐种子精度精确匹配原始旧 `graphmlp`，不匹配后来完成的 `graphmlp_reimplemented`。

旧实现位于原始 `predictor/GraphFreeFamily_Predictor.py:277`：抽取图边作为正样本，随机节点作为负样本，对一正一负两个相似度做交叉熵。这不是多跳邻接 NContrast。

后来完成的 `predictor/GraphMLPReimplemented_Predictor.py` 保留专用网络和多跳 NContrast，其 uniform 0.4、五数据集、五种子结果已经存在。

| 数据集 | 当前稿件：旧边采样代理 | 已有 NContrast 协议移植 |
|---|---:|---:|
| Cora | 63.23 ± 3.07 | 64.91 ± 2.04 |
| CiteSeer | 57.23 ± 1.85 | 59.87 ± 1.73 |
| CoraFull | 37.05 ± 0.91 | 44.51 ± 0.55 |
| Coauthor-CS | 88.95 ± 0.43 | 89.97 ± 0.39 |
| Amazon-Photo | 77.05 ± 1.24 | 80.44 ± 0.65 |

数值为精度百分数的五种子均值 ± 样本标准差。

证据：论文 `tables/table_3.tex:16`、`tables/table_F_1.tex:21`、`tables/table_F_2.tex:26`；原始 `runs/mlp27_graphmlp_fidelity_20260712/mlp27_report.md` 和 `mlp27_combined_raw.csv`；审计 `baseline_result_provenance.json`。旧实现 25/25 条相符，新实现 0/25 条相符。

**建议：采用已有 NContrast 协议移植的代码和 25 条结果，同步更新配置与结果分析。** 此时 GB-LCSS 高于 Graph-MLP 的是 Coauthor-CS 和 Amazon-Photo，不再包括 CoraFull。当前 `main.tex:460` 需相应修改。

移植版本有自己的网络和超参数，不能只换数字而仍写 common MLP，或保留旧代理的 512/.5/.5 配置。仍称协议移植，不称官方调参复现。

另一种处理是保留旧数字，但明确命名为边采样对比代理并删除 NContrast 身份说明。这只能修正文实不符，不能替代真正 Graph-MLP 比较。

**该问题属于 Table 3，不直接改变 Table 2 七种抗噪 MLP 比较中的 24/25 计数。**

### 2.2 FF-G2M：0.4 的含义写错了

论文 `tables/table_F_2.tex:30` 和投稿 `configs/main/protocol_adaptations.yaml` 把低频、高频蒸馏权重分别写为 0.4。

生成结果的原始 `predictor/G2MBaselines_Predictor.py:310` 至 `:337` 实际使用：

`L = 0.4 * L_label + 0.6 * (L_low + L_high)`

**0.4 是标签交叉熵权重，两种频率项各带 0.6 的外部系数。** `tau=1.5` 传入高频损失，也不能泛指所有损失使用同一温度。原始配置为 `ffg2m_lambda=0.4`、`ffg2m_tau=1.5`。

FF-G2M 的 50 条仓库逐种子精度全部匹配原始记录。

**建议：数字不动，修正论文和配置的参数含义，并接入实际产生结果的完整实现。** 不要按错误说明重新写一套损失。

### 2.3 Cora GRACE 学习率：论文错，代码配置正确

Word 与 LaTeX Appendix F.3 均写 `1e-4`。投稿 `configs/representations/grace.yaml` 与原始 Cora GRACE checkpoint 均为 `0.0005`，即 `5e-4`。

还核验了原始 Cora H 和投稿 `representations/cora/grace/H.pt`，两者逐元素完全相同。

证据：论文 `tables/table_F_3.tex:18`；原始 `checkpoints/grace/Cora/ratio_train0p2_val0p1_split0_pretrain0/checkpoint.pt` 及对应 embeddings 目录。

**建议：将论文学习率改成 `5e-4`，不用重训，也不要把正确代码改成错误表格值。**

### 2.4 九条外部比较路线没有可运行训练入口

当前入口 `scripts/run_main_experiments.py:20` 至 `:50` 只支持正式 GB-LCSS、七种同骨干基线、随机等规模分组、GB-LCSS 目标教师蒸馏。

缺少完整可运行入口的九条路线是：**PosteL、ALS、SALS、Graph-MLP、GSDN、GCN(X)+CE 响应蒸馏、FF-G2M、KRD、VQGraph。**

`configs/main/protocol_adaptations.yaml` 没有接入这些方法的训练。README 第 9 行也承认外部训练流程未提供。

论文 Table 5 的 kNN/k-means、Table 4 的显式平滑，以及表示源、验证标签污染等实验，也没有完整的当前 CLI。存在辅助函数或 YAML，不等于存在从输入到结果的可运行实验。

**建议：从原始工程迁入实际使用的实现和必要依赖，补上配置加载、训练、验证选模与保存入口。** 只收录当前论文所需路线，不打包实验探索史。最低验收不需要全量重跑，但每种方法都应能完成真实数据短程训练和结果输出。

### 2.5 checkpoint 平票规则改变，已实测影响测试精度

原始 `utils/logger.py:298` 在 metric 选模时使用 `validation > best_validation`，投稿包使用 `>=`。后者会用最后一次平票模型覆盖第一次最高验证精度的模型。

涉及投稿文件：`src/gblcss/baselines/noisy_label.py:130`、`training.py:69`、`distillation.py:78` 的学生选模。

控制检查：Cora、uniform 0.4、seed 0、200 epochs、PyTorch 2.12 CPU，同一环境下只切换比较符。

| 方法 | 当前包 >= | 原规则 > | 差值 |
|---|---:|---:|---:|
| Co-teaching | 47.86% | 48.50% | +0.63 pp |
| JoCoR | 49.18% | 51.35% | +2.16 pp |

二者各自的最高验证精度相同。JoCoR 选择的 epoch 分别为 191/130，Co-teaching 为 104/84。完整结果见 `checkpoint_probe.json`。

**这有明确代码原因，不能统称硬件波动。** 恢复原规则后仍不保证与历史 GPU 数值逐位相同，应先消除行为差异，再评估环境差异。

不要全局把 `>=` 换成 `>`：原始 GCN 教师本来就使用 `>=`。应分别保留基础模型、教师、学生的实际选模规则。

### 2.6 教师和学生初始化顺序改变

原始 `predictor/FrozenGraceGB_GCN_Predictor.py:111` 先初始化学生 MLP，后续才创建和训练 GCN 教师。投稿 `src/gblcss/distillation.py:37` 先创建教师，`:61` 在教师训练完成后才创建学生。

初始化和 dropout 都消耗随机数。固定 seed 0 的探针确认该调整导致教师初始参数不同，因此函数开头调用相同 set_seed 并不足以复现原流程。

**建议：恢复原实验的模型创建和随机数消费顺序，不事后发明新的 seed 规则。** 教师输入 X、图邻接以及正式软目标的方法定义本身没有发现不符。

## 3. 主要比较方法逐项结论

“来源一致”只表示逐种子精度匹配找到的原始记录，不等于已完整重跑，也不自动认证官方实现忠实度。

| 方法 | 投稿包可训练 | 已核对内容与处理 |
|---|---|---|
| MLP+CE | 是 | 模型结构、真实 Cora 运行；恢复选模规则 |
| Small-loss | 是 | 保留率日程、样本筛选；恢复选模规则 |
| Co-teaching | 是 | 交叉筛选、双网训练、单网测试；平票差异已实测 |
| JoCoR | 是 | 联合损失、保留率、单网测试；平票差异已实测 |
| SCE | 是 | 原始/投稿损失在测试输入上差值为 0；恢复选模规则 |
| ELR | 是 | EMA 和正则实现与配置对应；不能由一次运行认证全部复现 |
| LogitClip | 是 | 截断数值对照差值为 0；恢复选模规则 |
| PosteL | 否 | 25 条精度匹配 no-IPL 集成记录；需提供实际代码 |
| ALS | 否 | 25 条匹配适配记录；需提供实际适配代码 |
| SALS | 否 | 25 条匹配适配记录；需提供实际适配代码 |
| Graph-MLP | 否 | 25 条匹配旧代理，非声明的 NContrast；优先修复 |
| GSDN | 否 | 25 条匹配当前声明的旧适配；不建议此次换算法，接入该版本即可 |
| GCN(X)+CE / GLNN-style | 否 | 原始路线存在；投稿入口缺失。本轮未认证全部 50 条重新溯源 |
| FF-G2M | 否 | 50 条来源一致；权重说明错误；需完整入口 |
| KRD | 否 | 50 条来源一致；扰动可靠性、50% 选择对应机制适配；需完整入口 |
| VQGraph | 否 | 50 条来源一致；软码本对应机制适配；需完整入口 |
| GB-LCSS | 是 | Cora 25 设置目标完全一致；恢复选模规则 |
| GB-LCSS 目标教师 | 是 | 目标和输入定义对应；恢复初始化及学生选模行为 |

FF-G2M/KRD/VQGraph 记录来自原始 `runs/mlp55_g2m_table4_20260812/mlp55_raw.csv` 及 `runs/mlp59_final_extensions_20260814/g2m_u0p4_seeds0_4.csv`、`g2m_u0p6_seeds0_4.csv`。PosteL/ALS/SALS 当前主表行对应 `runs/nearest_work_full_grid_20260728/raw_immutable/`。本轮没有因此认证这些方法全部附录全谱行均重新溯源。

## 4. 已通过检查及覆盖边界

### 核心目标、噪声与训练

- Cora 的 5 seeds × 5 noise settings 共 25 设置：粒球归属与原实现一致 25/25；正式软目标最大差值 0。
- Cora 软目标 Brier 与归档最大绝对差约 1.19e-7，one-hot Brier 约 5.96e-8。本轮未复算全部五数据集 125 条。
- 相同严格选模规则下，用有序训练索引或布尔 mask，本轮七种基线的 Cora seed-0 精度及 epoch 相同，没有据此新增错误结论。
- 基础 MLP 初始参数及评估输出、SCE、JoCoR、LogitClip 的抽样数值对照一致；不是所有方法的全训练轨迹逐步认证。
- 本轮真实 Cora GB-LCSS 精度 56.46%，历史该种子 56.94%。目标完全相同，不能归因于目标版本混用；其余训练数值差异未全部定位。

### 可运行性

- pytest：13 passed，3 条依赖弃用警告。
- 22 个 Python 文件语法解析成功，16 个 YAML 可解析，两份 requirements 共 16 条依赖可解析。
- 合成 smoke：144 节点、36 训练节点、6 粒球，运行成功；不作为真实任务精度证据。
- 真实 Cora 七基线与 GB-LCSS 的 200-epoch 探针完成。
- 真实 Cora GRACE 2-epoch smoke 成功，输出 2708 × 128 表征。

### 表格

仓库汇总验证通过：原 Table 2 共 40 个精度单元格、Table 3 共 30 个、原 Table 4a/4b 即现 Table 7 各 30 个、验证敏感性表共 36 个。

已提取最新 Word 的 22 个表格组件并建立旧快照映射。对提取的精度均值/标准差、普通数字及 Appendix B 的 75 行统计显示检查，未见这一部分抄录差异；p 值按当前显示精度核验。

**不能由此说全文表格零问题。** Cora 学习率、FF-G2M 权重含义、Graph-MLP 身份均有错误。纯数值集合比较不能认证行身份、科学计数法和参数语义。

仓库 `verify_reported_results.py` 只检查旧快照与逐种子记录，不读取当前论文。22 张表存在的断言只检查数量。测试通过不能替代代码身份核对。

### 历史问题状态

- 每种方法入口重新设置 seed，原先方法执行顺序污染已有修复。
- Table C 的 180 条记录分为 90 clean-validation 和 90 noisy-validation，train/train_val 元数据检查通过。
- Attributed-Wiki 当前为 11,596 条唯一无向边、0.610 同质性，未把旧数值误列成当前问题。
- Co-teaching/JoCoR 测试输出来自单个 MLP，不是双网集成；这一点不推翻相同部署 MLP 的表述。

## 5. 投稿包装同步

1. 最新主文 Table 1–7，原仓库蒸馏 Table 4 已是 Table 7，旧 D.1–D.3 已部分移入正文。README 仍为旧编号，应更新当前表格到数据/命令的映射。
2. 投稿目录五个 PDF 与最终 LaTeX 对应文件均非同一文件。文件差异本身不能证明图中数据错误，但应同步最终审定图与可编辑源文件。
3. 未确认对应代码 ZIP。修复后重新打包并逐文件核对，不沿用来源未确认的压缩包。
4. 本报告、临时检查 JSON、原工程路径不进入匿名包。

## 6. 最小修复顺序

1. 修正 Graph-MLP 身份/结果、FF-G2M 权重、GRACE 学习率；不新增研究方向，不补 NRGNN，不扩 10 seeds。
2. 迁入九条外部比较和当前论文关键消融的实际实现，接上配置、完整训练与结果入口；不重新手写近似代理。
3. 恢复分阶段选模与初始化行为；先做损失、目标和真实数据短程回归，再评估是否需要更长复现实验。
4. 以当前论文核对主表、关键消融和逐种子记录；更新图、README 和表格编号。
5. 重新生成匿名目录与 ZIP，验证打包一致性。

这些工作首先是实现一致性修复。已有真正 Graph-MLP 结果可以利用，FF-G2M 与 GRACE 参数说明可直接纠正，不应先启动全量重跑。

## 7. 检查记录

工作区 `.codex_tmp/repository_audit_20260916/`：

- `audit.py`：审计辅助脚本。
- `baseline_equivalence.json`：损失/模型数值对照。
- `baseline_result_provenance.json`：外部方法结果来源。
- `checkpoint_probe.json`、`index_order_probe.json`：训练行为控制检查。
- `core_probe.json`：目标、Brier 与初始化检查。
- `current_docx_tables.json`、`final_numeric_check.json`：表格提取与数值检查。
- `package_before.json`、`package_unchanged.json`：投稿目录未修改确认。
- `grace_smoke/`：临时预训练输出，不是论文实验结果。

**最终判断：核心机制有直接一致性证据；目前尚不能说审稿人可用这个包重新运行全部主要对比。应先修复算法身份和执行行为、补齐入口，再上传。**
