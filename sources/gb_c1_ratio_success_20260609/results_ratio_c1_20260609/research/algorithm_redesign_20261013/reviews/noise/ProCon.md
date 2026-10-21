# ProCon：Leveraging Peer-Informed Label Consistency for Robust Graph Neural Networks with Noisy Labels

## 基本信息与阅读状态

- **作者与出处**：Kailai Li, Jiawei Sun, Jiong Lou, Zhanbo Feng, Hefeng Zhou, Chentao Wu, Guangtao Xue, Wei Zhao, Jie Li；IJCAI 2025 Main Track，pp. 5598–5606；DOI [10.24963/ijcai.2025/623](https://doi.org/10.24963/ijcai.2025/623)。
- **原始来源**：[IJCAI 官方论文页](https://www.ijcai.org/proceedings/2025/623)、[官方 PDF](https://www.ijcai.org/proceedings/2025/0623.pdf)。
- **阅读状态**：已核官方全文中的方法流程、主要实验设置和结论。**相关度**：最高（图标签噪声 + peer / prototype / clean-sample selection；非核心对比损失）。

## 问题与机制

ProCon 用图内相似 peers 的伪标签一致性识别疑似错标节点。peer 由特征余弦相似与图邻接共同确定；模型用原型产生节点伪标签，比较目标节点与 peers 的标签一致性。作者根据一致性分布拟合 GMM 以筛选较干净节点，再用选出的节点及其权重更新类别原型，形成迭代反馈。全文包含自监督表示学习部分，但论文的核心标签去噪判据是 peer-informed prototypical consistency，不能仅凭“self-supervised”称其为图对比学习主方法。

**目标变量**是带噪图节点分类及干净节点筛选/原型质量；不是粒球划分，也不是以正负 pair 对比损失为主要创新。摘要未将一条可独立复用的对比损失列为核心贡献，本卡不拟写对比损失公式。

## 实验、限制与迁移

官方正文在 Cora、WikiCS 等图数据上报告对称与非对称标签噪声实验，并消融 prototype 更新、样本选择等组件；不同噪声和数据集下结果有差异。方法依赖 feature similarity 与图邻接共同选 peer，若图存在异配边或噪声边，peer 同类假设会变弱；原型与 GMM 筛样也会受早期表示质量影响。需把这些视为适用条件/迁移风险，而非作者声称该方法已解决结构噪声。

**本项目用途**：这是“图拓扑 + 特征相似 + 原型一致性”处理噪声标签的直接相关先例；它不使用粒球，因此不能用它证明或否定粒球先例。它削弱的是“首次使用图邻域/原型一致性做图标签去噪”这类宽泛主张。更准确的比较是，粒球关系置信度能否比邻居/相似 peer 一致性更稳，尤其在错误关系下是否避免硬拉近。

