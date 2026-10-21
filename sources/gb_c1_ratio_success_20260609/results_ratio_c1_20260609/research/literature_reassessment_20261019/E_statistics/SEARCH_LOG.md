# E 领域检索日志

更新日期：2026-10-19（Asia/Shanghai）。一次来源优先；实际阅读版本与未核实内容分别标出。该文件随每2篇左右的阅读批次更新。

## 检索与纳入（当前已读3篇）

### 1) Ghosh, Manwani, Sastry (2017)

- 查询：`On the Robustness of Decision Tree Learning under Label Noise PAKDD 2017`; `arXiv 1605.06296`。
- 纳入：作者arXiv全文，阅读§2.2–3.4、§4与附录A；正式题录按PAKDD/LNCS与DOI记录核对。重点记录Gini/twoing/entropy排序结论、假设、样本界与实验协议。
- 来源：[arXiv HTML](https://arxiv.org/html/1605.06296)、[arXiv PDF](https://arxiv.org/pdf/1605.06296)、[Springer DOI](https://link.springer.com/chapter/10.1007/978-3-319-57454-7_53)。正式全文若需逐字版本核对应再从Springer核验；本次方法/结果正文来自作者预印本，非逐行正式版比对。
- 不纳入：未找到的任何“引用次数”或二手摘要，不作数量声称。

### 2) Wilton, Ye (AAAI 2024)

- 查询：题名+AAAI；核对AAAI正式PDF、论文页和官方代码。
- 纳入：[AAAI正式PDF](https://ojs.aaai.org/index.php/AAAI/article/download/29516/30856) §2–4、定理3–5、表1、图2–6和实验附录。阅读理论损失定义、定理4的固定样本噪声假设、定理5的早停行为，并记录具体噪声协议和多次重复设置。
- 版本：官方AAAI正式版，AAAI 38(14), 15859–15867，DOI 10.1609/aaai.v38i14.29516；官方代码入口为https://github.com/jonathanwilton/RobustDecisionTrees。
- 访问限制：没有单独运行作者代码或复现实验（本任务只做文献阅读）。

### 3) Hothorn, Hornik, Zeileis (2006)

- 查询：DOI、作者网站论文PDF、`Unbiased Recursive Partitioning: A Conditional Inference Framework`。
- 纳入：[作者PDF](https://www.zeileis.org/papers/Hothorn%2BHornik%2BZeileis-2006.pdf) §2–6，算法步骤与条件置换部分、Table 1/Figure 4和Figures 6–7。确认作者所称“无偏”指对协变量切分数/测量尺度的选择偏差，不是噪声标签稳健性；基准实验证明预测竞争力，不证明图节点可交换。
- 版本：DOI正式记录为JCGS 15(3):651–674；全文使用作者合法稿。
- 排除误读：不把“若p值不显著则停止”迁移为本项目球合并规则；不把其关于变量重要性/协变量的检验当作父子区域标签均值差的嵌套检验。

### 4) Domingos, Hulten (KDD 2000)

- 来源：[作者PDF](https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf)，KDD 2000, pp.71–80, DOI 10.1145/347090.347107。
- 实读范围：§2 Hoeffding bound、脚注2候选差距假设、Theorem 1、§3 VFDT refinements、§4合成与校园Web流实验、Figures 1–7。核对的是作者正式会议版全文。
- 重要界限：Hoeffding阈值给出流式分裂时的决策置信，并不度量单位计算预算的预测收益；实验比较样本数量、内存和硬件均不相同，故不用于图粗化加速主张。
- 未做：未复现论文代码/历史硬件实验；不影响阅读完成计数。

### 5) Meinshausen (Biometrika 2008)

- 查询：题名+Biometrika/DOI；作者主页PDF与正式期刊记录交叉核对。
- 来源：[作者PDF](https://www.stat.math.ethz.ch/~nicolai/hierarchical.pdf)、[期刊页](https://academic.oup.com/biomet/article-abstract/95/2/265/230264?login=false)，Biometrika 95(2):265–278, DOI 10.1093/biomet/asn007。
- 实读范围：§1–3、式(1)–(12)、Theorems 1–2、Figures 2–4、数值模拟。确认父子null嵌套是因为变量簇null定义为集合内所有回归系数为零；因此父簇H0为真会逻辑推出子簇H0为真。图上标签均值差的检验对象不自动具有相同逻辑结构。
- 实验边界：四种相关协变量的合成设计，没有真实图标签、实例噪声或粗化预算实验。

### 6) Athey, Imbens (PNAS 2016)

- 来源：[PNAS/PMC正式全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4941430/)，[arXiv v3](https://arxiv.org/html/1504.01132v3)，DOI 10.1073/pnas.1510489113。
- 实读范围：正式全文§2–4（honest setup、EMSE criterion、causal-tree criterion），Table 1与三组模拟。核对PNAS正式题录、样本拆分、MSE成本和区间覆盖。
- 结论边界：只迁移结构选择与叶估计样本分离的防适应偏差原则；因果处理效应、随机分配/无混杂条件和覆盖率理论不迁移到噪声图标签。报告诚实拆分对标注预算的成本。

### 7) Lin et al. (ICML 2020, GOSDT)

- 来源：[PMLR正式页面](https://proceedings.mlr.press/v119/lin20g.html)、[正式PDF](https://proceedings.mlr.press/v119/lin20g/lin20g.pdf)、[作者算法仓库](https://github.com/Jimmy-Lin/GeneralizedOptimalSparseDecisionTrees)。正式题录ICML 2020, PMLR 119:6150–6160。
- 实读范围：正式PDF§2–5，目标式(1)、DPB思想、式(7)递归、Algorithm 1、Figures 2–6；关注加性/非加性损失差异、叶数正则、连续阈值搜索及准确率—稀疏/计算实验。
- 访问限制：PMLR网页列出的 supplementary PDF `lin20g-supp.pdf` 本次返回UnexpectedStatusCode；主文提到附录I的完整benchmark细节但未能读取。因此不补充主文未列出的全套数据集名、硬件和重复次数；方法+主实验已从正式正文核读。
- 可迁移范围：仅固定候选树上可加叶成本的cut-DP是本项目算法推断；GOSDT本身在特征树搜索空间中优化训练目标，不能据此声称图传播误差加性或粗图端到端目标可DP精确优化。

## 七篇核心完成

7/7核心项达到methods_results或theory_methods。主报告按论文机制/条件/实验区分直接证据与本项目推断，并集中给出固定无标签候选树的预算cut v2建议。
