# STCMC-UR

## 元数据

Shizhe Hu, Binyan Tian, Weibo Liu, Yangdong Ye. “Self-supervised Trusted Contrastive Multi-view Clustering with Uncertainty Refined.” AAAI 2025, pp. 17305–17313. [AAAI 官方论文页](https://ojs.aaai.org/index.php/AAAI/article/view/33902)；[官方 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/33902/36057)。

## 问题与任务

完整、已对齐多视图数据上的无监督多视图聚类。论文针对聚类伪标签和对比关系不可靠、固定视图权重忽略视图质量等问题。它没有输入图结构、节点分类或带噪训练标签实验；结论部分明确说当前方法不能处理不完整多视图数据。

## 创新与机制

每个视图的编码器生成表示和类别证据。作者以两种 BBA（Basic Belief Assignment）生成证据，并用 Dempster–Shafer 融合得到每个样本的视图信念与不确定性；随后再跨视图融合类别证据，并依据聚类结果学习视图权重。对比模块按不确定性阈值筛选伪标签：不确定性高的样本不把伪标签用于该轮特征级对比，以减少错误负关系；其余样本的伪标签参与对比关系修正。总目标由证据生成/聚类、视图融合和特征级对比部分组成。

## 实验证据

五个数据集为 Caltech-2V、Caltech-3V、COIL20、Event8 和 NUS22，报告 ACC、NMI，包含组件消融、参数和收敛分析。作者称多个数据集取得较强结果；消融显示证据生成和对比模块均有贡献。实验不是噪声标签协议，且没有直接评价不确定性校准质量或伪标签关系的真实正确率，因此“可信”是方法目标，不能直接视为对关系可靠性的外部验证。

## 对本项目的启发与迁移边界

可借鉴把不确定性用于调节粒球关系的强弱或暂缓使用边界粒球，而不是把同球成员直接当作硬正标签。需将几何紧致度与语义可信度分开，并在节点标签噪声设置下独立检验置信度是否校准、关系错误率是否下降。

该论文是多视图聚类而非图节点分类，模型基于同一实例跨视图对应；其伪标签不等于受污染的人工训练标签。它不证明不确定性门控能抵抗图边噪声或标签噪声。

## 阅读深度与证据

已读 AAAI 官方 PDF 全文的证据生成、视图融合、不确定性筛选对比、实验设置、主结果、消融和结论（9 页）。证据：[官方 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/33902/36057)，题录据 [AAAI 官方论文页](https://ojs.aaai.org/index.php/AAAI/article/view/33902)。阅读级别：全文方法与实验核读。
