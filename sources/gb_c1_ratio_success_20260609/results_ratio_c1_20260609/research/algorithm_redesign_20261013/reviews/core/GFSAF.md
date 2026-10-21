# GFSAF：Views Attention Fusion of Granular-ball Fuzzy Representations Split for Improved Multi-view Clustering

## 文献信息与阅读层级
作者：Shuaiyu Liu, Song Wu, Jie Xu, Yazhou Ren, Yang Yang, Xiaorong Pu, Guoyin Wang。AAAI-26，DOI 10.1609/aaai.v40i28.39556。阅读层级：官方全文级，重点方法 pp. 3–5。[AAAI记录](https://ojs.aaai.org/index.php/AAAI/article/view/39556)；[官方PDF](https://ojs.aaai.org/index.php/AAAI/article/download/39556/43517)；[代码](https://github.com/Lsy235/GFSAF)。

## 方法与原文公式
多视图聚类，含 AE/DAE 和分别训练的两个阶段，不能说两阶段都持续运行对比学习。

SES 阶段：L_SES=L_GFCL+λ1 L_NS+λ2 L_rec。GFCL 用 PGB/NGB 构造球级对比：跨视图同一样本、同视图成员重叠球对是正关联，不相似球对是负关联。关联由离散集合定义，并非连续关系置信度。NS 分支获得互补表征 Z_i、N_i；Eq. (4)–(6)给出分支及 L_NS=1/2 Σ_i(cos(Z_i,N_i)+1) 类型的相似度惩罚。作者明确说 LNS 不能严格视作噪声分离，只是对分离损失的近似拟合。命名不能当作已辨识真实噪声。

VFS 阶段包含 CVA 与 K-means。Eq. (12) 给 L_VFS=L_rec；该阶段不是对比损失。

## 数据、结果与迁移判断
摘要称八个数据库，而 Table 1 列九项：WebKB、Multi-COIL-10、Multi-COIL-20、Caltech101-7、Prokaryotic、NUSWIDE、Reuters、DHA、UCI-Digits；记录为内部数量不一致。指标 ACC、PUR。论文报告 CVA 单独效果较小、各组件消融有帮助；LNS 权重太大会损害效果。结果表以点估计为主，未提供可直接复核的误差条/seed 标准差。

对当前图对比噪声节点分类，只建议借鉴“H/局部结构参与表示”的抽象想法，作为弱先验或置信信号探索。不要复制完整 AE/decoder、噪声剥离支路、CVA 注意力或双阶段架构；这些不是本任务已证明必要的部分。GFSAF 不是节点噪声分类研究。

## 证据与未核实
方法公式：官方 PDF pp. 3–5，Eq. (4)–(6)、Eq. (12)；数据 Table 1；效能/消融在实验表。精确数值、误差报告及超参敏感性尚未逐表重抄。方法原文以 PDF 为准。
