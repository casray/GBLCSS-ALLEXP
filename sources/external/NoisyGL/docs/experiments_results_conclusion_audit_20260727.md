# 第 4-7 节改写与终审报告

## 1. 审阅目标

本轮以 `paper_final_zh_v27_method_revised_20260727.md` 为事实底稿，改写第 4-7 节，并生成 v28。改写遵循三个原则：

1. 实验设置只负责定义可复现协议。
2. 结果段采用“结论句、关键证据、解释”结构。
3. 局限与资源边界集中披露，不在每个结果后重复答辩式限定。

表格主体数值、噪声协议、实验种子和方法身份保持不变。

## 2. 三路审阅

### 2.1 本地 Elsevier 范本

实际阅读了以下 10 篇本地正式期刊论文的实验、讨论和结论部分：

1. `Structure-aware-granular-ball-clustering_2026_Information-Sciences.pdf`
2. `PR2025-粒球K类孪生支持向量分类器.pdf`
3. `Capturing-local-and-global-information--Multi-view-graph-convolutional network via granular-ball computing and collaborative matrix.pdf`
4. `Enhancing-graph-convolutional-networks-with-progressive-granul ball sampling fusion A novel approach to efficient and accurate GCN training.pdf`
5. `GAdaBoost--An-efficient-and-robust-AdaBoost-algorithm based on granular-ball structure.pdf`
6. `GB-RVFL--Fusion of randomized neural network and granular ball computing.pdf`
7. `Beneficial-noise-learning-for-open-intent-classification-v_2026_Pattern-Reco.pdf`
8. `G-GBC--A-Gaussian-mixture-based-granular-ball-computing-me_2026_Knowledge-Ba.pdf`
9. `Like-draws-to-like--A-Multi-granularity-Ball-Intra-Fusion_2024_Advanced-Engi.pdf`
10. `Granular-ball-computing-based-Random-Walk-for-anomaly-_2025_Pattern-Recognit.pdf`

共同写法是：正文直接报告总体趋势与例外，机制解释紧跟相应消融，统计上无法区分时直述结果，局限集中在讨论或结论末段。

### 2.2 联网规范与近邻论文

联网审阅参考了 Elsevier Results Guide、Nature ML reporting checklist、PLOS discussion guidance，以及 GLNN、NoisyGL、NRGNN、G-GBC、BNO-GB、SAGBC 等正式论文。

获得的主要规则：

- 结果按研究问题和论证顺序组织，不按实验发生顺序组织。
- 正文解释表格中的关键模式，不逐格复述。
- 每段首句先给结论，随后给数据和解释。
- 限制需要明确，但不使用道歉式、自我否定式语言。
- 配置、资产和运行故障等复现细节应进入补充材料或 manifest。

### 2.3 技术事实审计

技术审计回查了实验报告、配置和实现，确认：

- 标签噪声只注入训练节点。
- uniform、pair 和 instance-dependent 噪声定义与代码一致。
- 主结果使用五个配对随机种子。
- `24/25`、`14/15`、相对 CE 的 `24/25`、相对 JoCoR 的 `8/25`、相对随机分组的 `25/25` 均有原始统计报告支持。
- 3×3 GRACE 种子实验中的 `0.53/0.07/0.44pp` 和 ARI `0.25-0.37` 与结果文件一致。
- 蒸馏教师使用 `target_beta=0`，学生输入为原始 `X`。

## 3. 主要修改

### 3.1 第 4 节

- 重组为数据集与范围、噪声与划分、对比方法、统计与复现四个小节。
- 删除实验历史和统计方法辩护。
- 明确 PubMed 也完成完整噪声谱，但作为边界结果分析。
- 明确使用双侧配对 t 检验，并在三个 25-setting 检验族内分别进行 Holm 校正。
- 保留 adapted 方法身份和缺失 checkpoint 的复现边界，但压缩为事实陈述。

### 3.2 第 5 节

- 表 1 后只保留总体性能、随机分组归因和噪声类型趋势。
- 将“结构化噪声普遍更有利”改为数据集依赖结论。
- 将原 5.3 拆分为局部共识与平滑、分组与票数、共识空间、学生容量、H-access、标签率与输入扰动六个问题。
- 删除“修正早期版本”“重新定性”“两点限定”等内部修订语言。
- 补充 LOO 同批复跑口径、kNN 查询节点自包含和低标签率数据集范围。
- 修正表 2 粗体规则。

### 3.3 第 6 节

- 删除方法流程的重复描述，集中回答蒸馏是否缩小精度差距。
- 明确表 5 数值是不同教师对应的 X-only 学生结果。
- 在表注中披露粒球教师使用完整球内分布目标 `beta_T=0`。
- 区分 H 输入、全节点监督和局部共识带来的增量。
- 将图方法表改写为跨部署资源参照。
- 修正表 5 和表 6 的最高均值粗体。

### 3.4 第 7 节

- 压缩为方法与主结果、机制认识、适用范围与未来工作三段。
- 保留同骨干比较范围、参数冻结口径和 transductive 限制。
- 删除对每项消融和负结果的重复罗列。

## 4. 最终技术修正

本轮除语言改写外，修复了以下实质问题：

1. pair/instance 的优势并非五个数据集普遍成立。
2. 表 5 的 `80.60/69.40` 属于蒸馏学生，而非教师。
3. 表 5 CiteSeer 两列的粗体错误。
4. 表 6 Cora、CiteSeer 和 Amazon-Photo 的粗体遗漏。
5. 表 2 Coauthor-Physics 的单独粗体缺少一致规则。
6. `14/15` 与 `24/25` 补充了“七个正式同骨干对照”的比较范围。

## 5. 自动核验

- 第 4-7 节共 87 行 Markdown 表格，去除粗体标记后，所有表格数据与 v27 逐行一致。
- 表 0-6 编号完整且唯一。
- 所有 Markdown 表格列数一致。
- 全文中文分号数量为 0。
- 全文不含本机绝对路径。
- 第 4-7 节非表格文字相对 v27 压缩约 21%。
- 摘要中的“五个完整噪声谱数据集”改为“五个主结果数据集”，避免与同样完成全谱评测的 PubMed 冲突。

## 6. 终审判断

v28 的第 4-7 节已从实验备案式写法转为论文叙述。主要证据、负结果、复现限制和资源边界仍然保留，但限定条件被放入相应结论内部，不再以连续否定句出现。当前版本适合作为后续英文翻译和作者逐句通读的中文底稿。
