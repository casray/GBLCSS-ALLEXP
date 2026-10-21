# 来源与阅读边界

核查日期：2026-10-21。这是针对选题的定向检索，不是穷尽式系统综述。未按引用数排名，未使用搜索引擎的相对时间代替正式出版日期。方法事实优先来自用户本地原文、作者代码、出版社页面、会议论文或作者预印本；搜索聚合站只用于发现线索。

## 本地材料

根目录：`C:/Users/Admin/Desktop/实验8去噪5.19-/文献文件/`。

1. `导师相关/_extracted_text/Structure-aware-granular-ball-clustering_2026_Information-Sciences.txt`：重读引言、§3.1–3.4、§4.4、结论。
2. `_literature_review_work/extracted_text/粒球聚类10篇代表文献/GB-DBSCAN--A fast granular-ball based DBSCAN clustering algorithm.txt`：重读§3.1–3.5、邻域/半径/可达性定义。
3. `_literature_review_work/extracted_text/粒球聚类10篇代表文献/Granular_Ball_Sampling_for_Noisy_Label_Classification_or_Imbalanced_Classification.txt`：重读贡献、采样描述、§III-D与结论。
4. `_literature_review_work/extracted_text/粒球聚类10篇代表文献/GBG++_A_Fast_and_Stable_Granular_Ball_Generation_Method_for_Classification.txt`：核对orphan定义与删除；结合已有逐篇报告了解其监督构球和分类规则。
5. `_literature_review_work/extracted_text/粒球计算-粒球生成与分类聚类/INGB-Informed Nonlinear Granular Ball Oversampling Framework for Noisy Imbalanced Classification.txt`：读取摘要与引言；未完整审计其采样实现。首页模板含2022，不据此认定发表年份。
6. `_literature_review_work/extracted_text/粒球计算-粒球生成与分类聚类/An_Efficient_Fuzzy_Stream_Clustering_Method_Based_on_Granular-Ball_Structure.txt`：读取摘要、漂移/衰减相关段落与结论；不声称完成全算法复现。

先读过的目录总结仅用于定位；其“对GRACE的启发”是旧任务下的解释，不作为本轮传统任务的证据。还参照工作区 `research/literature_reassessment_20261019/A_granular/REPORT.md`。

## 核心论文卡片

### S1 SAGBC

- Wang, Q.; Du, M. Structure-aware granular ball clustering. Information Sciences 740 (2026), 123224。
- 出版社：https://doi.org/10.1016/j.ins.2026.123224
- 作者全文：https://dumingjing.github.io/files/paper-22_SAGBC/2026_INS_SAGBC.pdf
- 官方代码：https://github.com/Du-Team/SAGBC
- 阅读：本地原文关键方法/实验设置/结论；在线核题录和公开代码。
- 事实：抽样产生中心；全数据与多个中心软关联；用空间和结构两条件构建球连接分量。
- 用途：第一候选直接基线。稀有簇被漏掉尚属待验证缺口。
- 代码审计入口：https://github.com/Du-Team/SAGBC/blob/main/SAGBC/SAGBC.py 。`getAnchor` 无放回抽样；`division` 的阈值8；`spilt_ball`用完整成对距离找最远对；示例 `k=5000`。当前main可变，正式复现时必须固定commit。

### S2 GB-DBSCAN

- Cheng et al. GB-DBSCAN: A fast granular-ball based DBSCAN clustering algorithm. Information Sciences 674 (2024), 120731。
- https://doi.org/10.1016/j.ins.2024.120731
- 阅读：本地方法全文相关段落＋出版社题录/摘要。
- 事实：KNN构球，实际用最大半径；按半径区分Core-GB；中心距≤半径和判直接可达。
- 用途：第二候选具体缺口来源。不把它当作严格等价点级DBSCAN的算法。

### S3 NaGB-DBSCAN

- NaGB-DBSCAN: An improved DBSCAN clustering algorithm by natural neighbor and granular-ball. Information Sciences (2025), 122445。
- https://doi.org/10.1016/j.ins.2025.122445
- 阅读：出版社检索摘要与Highlights；直接打开一次403，未取得全文。
- 事实边界：自然邻居构球、减少参数依赖、针对形状及变密度；没有完成其连边实现审计。
- 用途：不能把“自然邻居＋粒球/适应变密度”当新方法；须补全文作为现代粒球基线。

### S4 DBSCAN++

- Jennifer Jang, Heinrich Jiang. DBSCAN++: Towards fast and scalable density clustering. ICML 2019, PMLR97:3019–3029。
- https://proceedings.mlr.press/v97/jang19a.html
- 全文：https://proceedings.mlr.press/v97/jang19a/jang19a.pdf
- 阅读：会议页与论文前部算法/初始化。
- 事实：通过子集密度评估加速，含均匀和k-center初始化。
- 用途：第一候选必须超过覆盖型代表选择；抽样加速本身不是贡献。

### S5 ARP-GB

- Zhonglin Wang, Ping Zhu. An adaptive representative point-driven granular-ball clustering algorithm. International Journal of Machine Learning and Cybernetics 17,304 (2026)。出版社显示2026-05-16。
- https://link.springer.com/article/10.1007/s13042-026-03131-3
- 阅读：出版社摘要、题录；全文为订阅内容，未获取。
- 事实边界：基于KNN密度选代表，半径缩减和球重叠传播。不能据摘要判断所有少数簇处理细节。
- 用途：第一、二候选直接近邻先例。

### S6 MG-GBC（预印本，重要）

- Multi-granularity granular-ball clustering for preserving rare cores。SSRN 7358001；索引显示2026-08-27发布。
- https://ssrn.com/abstract=7358001
- https://doi.org/10.2139/ssrn.7358001
- 阅读：作者提交摘要及检索索引；正文链接未拿到全文，直接页面403。
- 摘要已明确：在ARP-GB基础上保护小而密的孤立核心，允许链接稀有碎片。
- 用途：不能说“首次稀有簇保护”。与第一候选差异暂定为抽样遗漏发现/预算重分配，必须精读全文再确认；不据它的摘要成绩承诺复现效果。

### S7 BLOCK-DBSCAN

- Yewang Chen et al. BLOCK-DBSCAN: Fast clustering for large scale data. Pattern Recognition109 (2021),107624。
- https://doi.org/10.1016/j.patcog.2020.107624
- 阅读：出版社摘要、引言与方法概述。
- 事实：球形Inner Core Blocks、cover tree、块间可达性近似。
- 用途：第二候选必要对照。批量处理球与点级回查不构成首次提出。

### S8 AnyDBC

- Son T. Mai, Ira Assent, Martin Storgaard. AnyDBC: An Efficient Anytime Density-based Clustering Algorithm for Very Large Complex Datasets. KDD2016:1025–1034。
- https://doi.org/10.1145/2939672.2939750
- 官方会议全文：https://www.kdd.org/kdd2016/papers/files/rfp0144-maiA.pdf
- 阅读：摘要与引言的算法保证/活动查询框架；本轮未完整复现。
- 事实：primitive clusters压缩、主动选择查询、逐步细化并保证最终DBSCAN结果。
- 用途：第二候选的最强概念先例之一。任何“只细查重要区域”的创新声明需具体区分。

### S9 On Metric DBSCAN with Low Doubling Dimension

- Hu Ding, Fan Yang, Mingyue Wang. IJCAI2020。
- https://www.ijcai.org/proceedings/2020/0426.pdf
- 阅读：题名、摘要与导言定位；理论细节待精读。
- 用途：提醒维度/几何假设与覆盖方法先例，不据此声称本方案更优。

### S10 GBG++

- Qin Xie et al. GBG++: A Fast and Stable Granular Ball Generation Method for Classification. TETCI，DOI2024；作者版2023首发、2025更新。
- https://doi.org/10.1109/TETCI.2024.3359091
- https://arxiv.org/abs/2305.18450
- 阅读：本地正式版orphan删除与已有方法报告；线上作者版核查。
- 用途：第三候选具体入口。单点删除误伤真实子群是预测，尚未实测。

### S11 GBS

- Shuyin Xia et al. Granular Ball Sampling for Noisy Label Classification or Imbalanced Classification. TNNLS34(4),2144–2155,2023；DOI登记2021。
- https://doi.org/10.1109/TNNLS.2021.3105984
- 阅读：本地贡献、实验§III-D、结论。
- 事实：采样真实成员/边界；已有不平衡与20%标签噪声同时出现的实验。
- 用途：第三候选必须超过它，不声称噪声不平衡新组合。原文将SMOTE误写为undersampling的一处表述不予沿用。

### S12 INGB

- Min Li et al. INGB: Informed Nonlinear Granular Ball Oversampling Framework for Noisy Imbalanced Classification。
- 作者版2023：https://arxiv.org/abs/2307.01224
- 阅读：本地摘要与引言＋作者题录；本轮不认定正式期刊年份。
- 事实：粒球空间、informed entropy、非线性过采样；已讨论噪声、不平衡、小子群等复杂性。
- 用途：第三候选近邻先例。区别应落到固定压缩预算、真实样本保留与子群恢复，不能只换采样公式。

### S13 EvoSampling

- Wenbin Pei et al. EvoSampling: A Granular Ball-Based Evolutionary Hybrid Sampling With Knowledge Transfer for Imbalanced Learning. IEEE Computational Intelligence Magazine21(2),55–67,2026。
- https://doi.org/10.1109/MCI.2026.3657694
- 作者机构题录：https://scholars.hkbu.edu.hk/en/publications/evosampling-a-granular-ball-based-evolutionary-hybrid-sampling-wi/
- 阅读：作者机构摘要/题录，未完整读发表版。
- 事实：进化过采样＋粒球欠采样去噪。2024预印本摘要写20数据集，2026正式题录摘要写18；不混用版本数字。

### S14 MDL-GBG / MDL-GBC（两篇2026预印本）

- https://arxiv.org/html/2605.08759v1
- https://arxiv.org/html/2605.11406v1
- 阅读：作者全文摘要、方法目录及部分方法/复杂度段落，未完整实验审计。
- 前者以单球/双球/核心加残差竞争并重分配残差；后者结合类条件正证据和负类边界证据。
- 用途：不能把统一分裂评分、核心边界分解或者MDL自身写成新贡献。是否具有固定预算下覆盖修复优势待直接比较。

### S15 GB-FuzzyStream

- Jiang Xie et al. An Efficient Fuzzy Stream Clustering Method Based on Granular-Ball Structure. ICDE2024:901–913。
- https://doi.org/10.1109/ICDE60146.2024.00074
- 作者代码：https://github.com/xjnine/GBFuzzyStream
- 阅读：本地摘要、衰减/漂移段落、结论。
- 事实：已经引入加权衰减和模糊边界；结论提参数适应性问题。
- 用途：本轮不推荐泛泛“粒球适应概念漂移”；没有完成新缺口诊断。

## 后续必须补的全文与强对照

- MG-GBC、ARP-GB、NaGB-DBSCAN：方法全文/代码，优先于新算法开发。
- 对抽样覆盖：k-means++、k-center、BIRCH与合适coreset。不能只打随机抽样。
- 对连接精化：AnyDBC完整查询选择规则、BLOCK-DBSCAN/KNN-BLOCK、优化的索引DBSCAN；不能把普通球树作为粒球独有创新。
- 对分类小子群：MDL-GBC、最新边界驱动粒球分类、小子群过采样与类别平衡强表格分类器。边界驱动TWD-BDGB的出版社页面卷期日期为2026-11-05，晚于本次日期；已能检索到，但首次在线日期本轮未核清，只作线索，不用它下优先权结论。

## 搜索与判断审计

关键词族：granular ball sampling noisy imbalanced；granular ball DBSCAN boundary density；rare clusters sampling；adaptive representative granular ball；AnyDBC/BLOCK-DBSCAN；MDL granular ball；granular ball stream concept drift。

部分网页访问遇403，一次搜索连接失败后重试成功。未使用付费绕过。未声称检索穷尽、引用量最高、所有代码可运行或已复现论文结果。未将聚合站的模型摘要作为方法证明。

本轮只验证了抽样概率数学例子，没有新模型训练、数据集性能实验或精度改进结果。
