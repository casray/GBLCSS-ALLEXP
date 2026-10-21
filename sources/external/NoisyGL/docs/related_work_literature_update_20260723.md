# Related Work 文献更新与取舍备案

日期：2026-07-23

对应稿件：`paper_final_zh_v25_related_work_updated_20260723.md`

## 1. 修改原则

本轮不按“是否最新”机械增删文献，而按三个条件筛选：

1. 与 GB-LCSS 的任务、机制或部署约束有直接关系。
2. 能从出版社页面、正式会议论文集、PMLR 或 DOI 核验发表状态。
3. 能改变相关工作的技术脉络，而不是只增加引用数量。

长期未接收并不等价于研究价值低，但在正式期刊稿中不宜让此类预印本承担关键 novelty 论证。若预印本已经作为实验基线，或是一个方向的早期代表，可以保留，但正文篇幅应低于正式发表工作。

## 2. 预印本处理

| 工作 | 截至 2026-07-23 的状态 | 本轮处理 | 理由 |
|---|---|---|---|
| Graph-MLP [4] | arXiv 2021，未检索到正式接收版本 | 保留，压缩为一句 | 它是早期 X-only 节点分类代表，也是正文表 3 的实际实验基线 |
| GSDN [14] | arXiv 2022，DBLP 仍列为 CoRR | 保留，压缩为一句 | 它是邻域自蒸馏代表，也是正文表 3 的机制适配基线 |
| RKD-MLP / Double Wins [原 33] | OpenReview 的 ICLR 2023 submission，未检索到正式接收版本 | 从正文和参考文献删除 | 未进入本文实验，且其作用可由正式发表的 KRD、AdaGMLP 等工作覆盖 |
| Pitfalls of GNN Evaluation [39] | NeurIPS 2018 Relational Representation Learning Workshop | 更正书目信息 | 原稿误写为单纯 arXiv，实际有正式 workshop 版本 |
| SCGNN | arXiv 2026-05 | 暂不加入投稿稿 | 与粒球监督增强很接近，但发表时间短且尚无正式接收信息，列入后续跟踪 |
| DREAM | arXiv 2026-01，OpenReview 页面可见，但本轮未能从正式 proceedings 独立核验出版记录 | 暂不加入投稿稿 | 机制很接近，应持续跟踪。正式论文集可核验后再补入，不以当前预印本承担 novelty 论证 |
| Efficient and Scalable Granular-ball Graph Coarsening | arXiv 2026-03 | 暂不加入投稿稿 | 已有 AAAI 2025 与 IJCAI 2025 的正式粒球图粗化论文可支撑该脉络 |

核验入口：

- [Graph-MLP arXiv](https://arxiv.org/abs/2106.04051)
- [GSDN arXiv](https://arxiv.org/abs/2210.02097)
- [RKD-MLP OpenReview](https://openreview.net/forum?id=NGIFt6BNvLe)
- [Pitfalls of GNN Evaluation 项目页](https://www.cs.cit.tum.de/daml/gnn-benchmark/)
- [SCGNN arXiv](https://arxiv.org/abs/2605.02617)

## 3. 新增正式发表工作

### 3.1 图标签噪声

1. **MILAB，Pattern Recognition 2026**  
   通过信息瓶颈约束邻居聚合，并利用特征与边权扰动下的预测一致性估计节点可靠性。它代表了 2026 年图标签噪声研究中“可靠聚合 + 节点可靠性估计”的路线。  
   [出版社页面](https://www.sciencedirect.com/science/article/abs/pii/S0031320326012367)  
   DOI: `10.1016/j.patcog.2026.114271`

2. **HALF，Information Sciences 2026**  
   根据节点局部同质性融合低通与高通损失，在同质图和异质图中识别干净与含噪训练节点。它补充了现有相关工作对异质图标签噪声的覆盖。  
   [出版社页面](https://www.sciencedirect.com/science/article/abs/pii/S0020025526006456)  
   DOI: `10.1016/j.ins.2026.123714`

3. **BO-NNC，AAAI 2025**  
   使用多个自监督图教师，通过双层优化动态学习教师权重，并结合标签改进模块完成含噪节点分类。它要求本文不得将“噪声图上的教师蒸馏”写成首次贡献，真正的区别应落在粒球共识教师和严格 X-only 学生。  
   [AAAI 正式页面](https://ojs.aaai.org/index.php/AAAI/article/view/34095)  
   DOI: `10.1609/aaai.v39i18.34095`

4. **GTA，Knowledge-Based Systems 2025**  
   通过 EdgeBoost 增补可信边，并由 EdgePrune 删除可能传播错误监督的边，直接从拓扑调整角度提高标签噪声下的鲁棒性。GB-LCSS 不修改图，而是将训练期结构信息转化为软监督。  
   [出版社页面](https://www.sciencedirect.com/science/article/abs/pii/S0950705125012031)  
   DOI: `10.1016/j.knosys.2025.114162`

这些方法都在训练和最终预测中保留图依赖，因此不会削弱 GB-LCSS 的 X-only 推理定位，但会限制对局部一致性和教师蒸馏的过强新颖性表述。

### 3.2 无图推理

1. **ES-MLP，LoG 2025**  
   将边拆分机制引入 Graph-MLP，在同质图和异质图上训练不执行消息传递的 MLP。它是比长期未接收的 RKD-MLP 更适合补入正文的正式无图推理工作。  
   [PMLR 正式页面](https://proceedings.mlr.press/v269/kohn25a.html)

### 3.3 粒球与标签噪声

1. **VPGB，Information Sciences 2022**  
   从粗粒度视角联合进行属性约简与含噪分类，是粒球处理标签噪声的重要早期正式工作。  
   [出版社页面](https://www.sciencedirect.com/science/article/pii/S0020025522009641)  
   DOI: `10.1016/j.ins.2022.08.066`

2. **GEAF，Information Sciences 2026**  
   采用不依赖标签纯度的自适应粒椭球划分，再在局部粒内检测和修正错误标签。它与 GB-LCSS 的“标签无关划分 + 局部标签处理”最接近，但处理的是独立样本分类而非图节点分类。  
   [出版社页面](https://www.sciencedirect.com/science/article/abs/pii/S0020025526002550)  
   DOI: `10.1016/j.ins.2026.123324`

3. **GBRAIN，ACM ICMR 2024**  
   在文本分类中动态构造粒球，以球中心和球级标签形成粗粒度训练表示，直接研究粒球如何缓解标签噪声。它比一般的粒球图粗化工作更接近本文的抗噪动机，但面向独立文本样本，不涉及图节点分类或 X-only 推理。  
   [ACM DOI 页面](https://doi.org/10.1145/3652583.3658084)

### 3.4 粒球与图学习

1. **Supervised Granular-Ball Graph Coarsening，AAAI 2025**  
   依据标签纯度递归划分粒球，并将粒球作为超节点压缩训练图。  
   [AAAI 正式页面](https://ojs.aaai.org/index.php/AAAI/article/view/33404)

2. **GBGC，IJCAI 2025**  
   根据图的多粒度结构生成粒球超节点，用于高效、自适应图粗化。  
   [IJCAI 正式页面](https://www.ijcai.org/proceedings/2025/388)

3. **MGGR，Neurocomputing 2026**  
   用粒球递归分解图内子区域，并通过层次编码器完成图级分类。  
   [出版社页面](https://www.sciencedirect.com/science/article/pii/S0925231226011100)  
   DOI: `10.1016/j.neucom.2026.133713`

这些工作证明粒球已进入图粗化与图级表示，但它们不研究“含噪节点标签 + X-only 推理”的组合问题。

## 4. 对稿件结构的影响

- 删除了相关工作开头的报告式分类说明，直接进入四个技术小节。
- Graph-MLP 与 GSDN 仍保留，但只承担历史与实验基线说明，不承担核心 novelty 论证。
- 删除 RKD-MLP，使用正式发表的 ES-MLP 替换参考文献编号 [33]。
- 图标签噪声小节补入 MILAB 与 HALF。
- 图标签噪声小节进一步补入 BO-NNC 与 GTA，分别覆盖噪声图多教师蒸馏和拓扑自适应路线。
- 粒球小节形成三层脉络：传统含噪分类、粒球图学习、粒球与深度抗噪学习。其中 GBRAIN 补足了“粒球直接用于深度标签噪声训练”的正式近邻工作。
- 删除节末重复总结，由粒球小节最后一段直接说明 GB-LCSS 的任务与机制差异。

## 5. 尚未纳入投稿稿的近期预印本

`SCGNN` 与本文机制高度接近，值得持续跟踪。它使用粒球建模组级语义一致性，并结合模型伪标签增强监督，但当前仍为 2026 年 5 月发布的预印本。若投稿前出现正式接收版本，应重新核验并在相关工作中补一至两句。当前不将其写入正式稿，可以避免以未稳定发表的工作主导 novelty 定位。
