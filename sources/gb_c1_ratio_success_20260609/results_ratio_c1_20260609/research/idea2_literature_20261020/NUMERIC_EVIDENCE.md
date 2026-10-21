# 原论文数字与协议边界

日期：2026-10-20。以下是**文献摘录，不是本地复现结果**。ACC、R@1均按百分数记录；不插值缺失档位、不补造标准差。

NR：所检查主实验未报告该档位。U：尚未核实。NA：任务/指标不适用。

## 1. RMCNC Table III

[作者正文及Table III](https://xlearning-lab.com/markdowns/2024-tkde-robust-multi-view-clustering-with-noisy-correspondence/)。

第一视图作为锚，注入NC；每项五次运行均值。这里只摘Scene15、Caltech101的ACC。该文Caltech为9146样本/102类别，不能和下节深度特征版拼成排名。

| 方法 | Scene15 20% | 40% | 60% | Caltech 20% | 40% | 60% |
|---|---:|---:|---:|---:|---:|---:|
| RMCNC | 41.00 | 39.55 | 38.36 | 30.60 | 27.57 | 24.02 |
| SURE（该文实现） | 40.02 | 36.94 | 31.05 | 29.68 | 19.86 | 10.83 |
| MvCLN（该文实现） | 39.31 | 37.01 | 30.40 | 30.43 | 18.45 | 10.91 |
| PVC（该文实现） | 30.28 | 22.41 | 17.88 | 19.12 | 13.31 | 9.43 |

SURE/MvCLN/PVC这些数字来自RMCNC对比表，不是其各自原论文的NC实验。

## 2. CorreGen Table 1

[作者正文及Table 1](https://xlearning-lab.com/markdowns/2026-iclr-uncover-underlying-correspondence-for-robust-multi-view-clustering/)。

五次运行均值，MR=20%，未额外引入CR损坏。摘录ACC。Caltech为8677样本/101类、DECAF与VGG19视图；评估含批内重匹配。

| 方法 | Scene15 | Caltech101 |
|---|---:|---:|
| SURE（该文实现） | 37.26 | 35.91 |
| DIVIDE（该文实现） | 41.91 | 55.65 |
| CANDY（该文实现） | 41.05 | 65.79 |
| ROLL（该文实现） | 44.86 | 20.39 |
| CorreGen | 48.04 | 68.01 |

该表档位0/20/50/80%；40%和60%为NR。ROLL在此Caltech设置较低，不能据此认定ROLL在所有协议下弱。

## 3. PLCI Appendix C, Table 4

[预印本Table 4](https://arxiv.org/html/2606.04343)。这是**NC-only、IVR=0**，不是混合缺失与错配的Table 1。Scene15，NCR=20%，ACC：

| 方法 | ACC |
|---|---:|
| CANDY（该文实现） | 40.14 |
| ROLL（该文实现） | 45.87 |
| CorreGen（该文实现） | 42.94 |
| PLCI | 44.92 |

40%和60%为NR（所查表档位20/50/80%）。这个表与CorreGen原论文对同一名称数据集的结果不同，进一步说明必须统一输入特征、训练与评估，不能拼取各论文最好数字。

## 4. DECL Table 1：只作检索任务参照

[原PDF](https://xlearning-lab.com/assets/2022-ACMMM-Deep-Evidential-Learning-with-Noisy-Correspondence-for-Cross-modal-Retrieval.pdf)，Table 1，Flickr30K 1K测试，图像→文本R@1：

| 方法 | 20% | 40% | 60% |
|---|---:|---:|---:|
| DECL-SGRAF | 77.5 | 72.7 | 65.2 |
| NCR（该文对照） | 73.5 | 68.1 | 13.9 |

这是检索召回率，**不能和聚类ACC比较**；NCR一行不是本轮从其原论文抄取的数字。

## 5. 尚不能填入的数字

| 方法/组 | 20% | 40% | 60% | 原因 |
|---|---|---|---|---|
| CANDY原论文 | U | U | U | 主PDF获取受限；后续论文数值已单列 |
| ROLL原论文 | U | NR | NR | 方法/协议段可读；未可靠摘出20%数值 |
| MGBCC | NR | NR | NR | 所检查原文未提供该NC扫描 |
| HOARD、GFSAF、MGCF-Net | U | U | U | 本次未找到可确认的NC数字，不把一般噪声实验当NC |
| PVC/SURE/MvCLN原论文 | NA/U | NA/U | NA/U | 原任务含PVP/IV；其NC适配值见RMCNC，不能回填为原值 |
| BiCro原论文 | U | U | U | 本轮未逐格摘录检索表；无MVC ACC |
| CPSPAN、SMART、OPTION | NA/U | NA/U | NA/U | 不同任务或全文未充分核实 |

这些空缺是证据状态，不能被补成0，也不能用相邻档位线性插值。后续实验应保存实际数据文件哈希、视图维数、样本数、类别数与重匹配设置。
