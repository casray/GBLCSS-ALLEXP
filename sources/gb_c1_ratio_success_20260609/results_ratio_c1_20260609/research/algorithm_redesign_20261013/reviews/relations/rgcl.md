# RGCL

## 元数据

Deyin Zhuang, Jian Dai, Xingfeng Li, Xi Wu, Yuan Sun, Zhenwen Ren. “Robust Graph Contrastive Learning for Incomplete Multi-view Clustering.” IJCAI 2025, pp. 7282–7290. DOI: 10.24963/ijcai.2025/810. [IJCAI 官方论文页](https://www.ijcai.org/proceedings/2025/810)；[官方 PDF](https://www.ijcai.org/proceedings/2025/0810.pdf)；[作者代码](https://github.com/DYZ163/RGCL)。

## 问题与任务

不完整多视图数据上的无监督聚类。论文重点讨论图对比聚类中的 Graph Noisy Correspondence：图内样本被当作正对、图外样本被当作负对时，图外同类样本会成为假负样本。数据缺失是视图数据缺失，不是节点标签错标。

## 创新与机制

先通过多视图关系图转移补全缺失视图，再用各视图自编码器学习表示。对比关系分成三档：关系图内样本是绝对正对；图外但当前表示相似度高的样本是半正对；图外且相似度低的样本是负对。作者对半正对构造相似度目标分布，并以目标分布锐化和交叉熵式损失引导关系学习，以减轻把相似样本作为负对的惩罚。另有跨视图图级分布对齐，用来利用视图间互补信息。

## 实验证据

HandWritten、COIL20、BDGP、LandUse-21、ALOI-100、AWA 六个多视图数据集；与九种不完整多视图聚类方法比较，指标为 ACC、NMI、ARI。实验改变视图缺失率，并包含对关系鲁棒损失、补全和跨视图对齐的消融。论文的结果支持其在所设缺失视图聚类协议下有效缓解假负关系；不支持外部标签噪声分类主张。

## 对本项目的启发与迁移边界

“绝对正—半正—负”的分层比把粒球成员二值化成正关系更稳妥：可将跨球但关系可信的候选作为弱关系，对边界/不确定球降低权重或暂缓使用。必须与普通相似度权重、最近邻和同球硬标签对照，测关系质量而不只看最终准确率。

RGCL 的关系图是多视图样本关系结构，不是噪声标签图；其半正关系来自表示相似度，仍依赖已有表示及候选图质量。该论文没有图节点监督分类或带噪训练标签实验，不能直接作为本项目鲁棒分类证据。

## 阅读深度与证据

已读 IJCAI 官方 PDF 全文的问题定义、关系图补全、噪声鲁棒对比、跨视图对齐、数据集、主结果与消融（9 页）。证据：[IJCAI 官方 PDF](https://www.ijcai.org/proceedings/2025/0810.pdf)；题录及摘要见 [IJCAI 官方论文页](https://www.ijcai.org/proceedings/2025/810)。阅读级别：全文方法与实验核读。
