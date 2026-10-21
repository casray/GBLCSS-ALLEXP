# 五 seed 主表候选基线：完整交接

## 本轮目标与完成状态

按用户要求先检查实现和主实验协议，通过后跑五个主数据集 seed0--4。
SimMLP：uniform0.4；TINED：uniform0.4、uniform0.6。GPU 运行已全部结束。
75/75 result 记录齐全，75/75 检查点/概率/指纹/选模核验通过，重载最大绝对差0。
新目录之外的历史源码、结果、论文均未改。GSSC 未放行；DGD 没有本轮可验证适配版。

## 修正了什么

1. 老 common.py 固定 split/noise/评估 seed0、rate0.4，不能直接外套五seed。
   新 protocol.py 传递 seed/rate，保留原随机抽样顺序；原训练标签注噪后才屏蔽非训练标签。
2. 旧 TINED 三处 RNG 重置为0；新 tined_formal.py 三处都使用实际 seed。
3. 官方 TINED distill_run_transductive 先排序 idx_l 再做CE minibatch。
   旧 GCN port 漏了排序；现在仅在注噪后、学生CE分批前排序，不改变主实验噪声分配。
   所以 TINED seed0 学生全部重跑，不能混用旧未排序成绩。
4. 输出按方法/数据集/噪声率/seed隔离；复用前检查协议、源码与文件内容指纹。
   模型训练只接收训练噪声标签、独立的clean validation标签；测试成绩只在选模后计算。

## 配置与可比较边界

- 20/10/70、split seed=noise seed=downstream seed、raw X、训练集内注噪。
  uniform0.4 的25组 split/noise hash逐一匹配历史 replay。
  两档噪声共50组逐元素匹配原 NoisyGL 注噪函数；val/test改动数全为0。
- SimMLP 部署结构与共同两FC MLP一致。固定复用已验证的无标签seed0编码器，
  对每个下游seed初始化新分类头、训练200轮；编码器权重、BN统计、dropout被冻结。
  原无标签预训练200轮不重复；旧seed0整次运行核验后复用。其余20个head新训练。
  统计不包括无标签预训练随机性的变化，不能写成五次独立SSL预训练。
  作者 stop-gradient/alignment/reconstruction保留；Coauthor-CS原preset重构系数为0。
- TINED 共同两层GCN教师、raw X/A、hidden64、teacher dropout0.5、Adam0.01/wd0.0005、
  200轮、教师验证并列末个最大值；学生200轮、验证并列首个最大值。
  保留原生4FC FT/GP模拟、权重注入、CE/KL/DE阶段、作者student dropout和已锁定参数。
  不能写成共同两FC学生；也不能声称正式复现了作者原SAGE协议或完整DGL数值路径。
  无GCN专属作者best preset，方法参数继续使用预先声明的SAGE presets/fallback。
  能量取样仍为已声明的torch GPU randperm port，原作者使用numpy choice；均为均匀边取样。
- GSSC 当前为3FC、100监督warmup+200主训练、last tie，且inference参数元数据含辅助头。
  其核心CPU测试通过，不代表符合共同骨干/200监督轮预算。需要另行明确适配设计后再跑。
- 没有根据本轮测试精度调参，没有丢掉任何失败或低精度seed；本轮未出现运行失败。

## 五 seed 结果

单位：准确率百分数；mean ± 样本std（ddof=1）。

| 方法与噪声 | Cora | CiteSeer | CoraFull | Coauthor-CS | Amazon-Photo |
|---|---|---|---|---|---|
| SimMLP matched, 0.4 | 75.46 ± 1.41 | 65.31 ± 1.78 | 40.24 ± 0.34 | 86.47 ± 1.79 | 88.69 ± 0.61 |
| TINED common GCN, 0.4 | 78.92 ± 2.83 | 64.18 ± 1.16 | 52.07 ± 1.36 | 92.19 ± 0.46 | 85.46 ± 1.47 |
| TINED common GCN, 0.6 | 66.36 ± 2.50 | 51.32 ± 5.07 | 32.07 ± 9.33 | 88.93 ± 1.11 | 64.64 ± 6.36 |

## 实验结论与负结果

已支持：SimMLP在共同两FC结构下仍有较高准确率。与正文已有GB-LCSS uniform0.4均值
57.49/55.24/37.39/90.71/82.12相比，四个数据集均值更高，Coauthor-CS更低。
这些是均值比较，不代表已做显著性检验，也不能忽略训练机制与预训练的差异。

已支持：TINED参数固定时存在明显训练退化。50次中24次末轮验证精度比最佳轮低超过10pp。
CoraFull uniform0.6的五次末轮验证精度仅4.24--5.05%，最佳选模学生测试为20.51--43.46%，
mean32.07±9.33；相应教师测试mean56.02±0.97。不能把学生落差全归于教师精度低。
这也不是数据划分/噪声/hash/检查点取错的证据：这些核验均已通过。

合理猜测：共同优化器与固定的跨骨干方法参数可能使DE约束和KL/CE不平衡。
本轮没有控制实验，不能当作确定原因，不能自动按测试结果调参。

未验证：完整DGL原始运行逐步数值等价、官方最优协议的五seed表现、新GSSC适配设计。
因此适配版可以形成审查后的候选表格行，但不是“所有方法均完全相同骨干”的结论。

## 证据与下一步

- CODE_REVIEW.md：具体协议问题、公式和实现审查、GSSC暂缓理由。
- PLAN.md：预先确定的范围、参数、成功/失败标准和命令。
- protocol_preflight.json：50组协议核验及资产指纹。
- per_seed_results.csv：75条原始成绩，包含最佳与末轮validation、教师精度。
- summary.csv：15组mean/std、最小最大值及教师mean/std。
- artifact_audit.json：75次重载/选择/指纹核验，全部passed。
- table_rows.tex：可用于Table4/8的候选新行，尚未写入论文。
- runs/：逐格config/protocol/environment/source hashes/epochs/console/checkpoint/probabilities。

26项CPU/小规模测试通过，1项GSSC opt-in GPU测试跳过；2次新适配GPU smoke通过。
训练与汇总命令为 PLAN.md 中 run_grid.py --grid 和 report_results.py，二者exit0。

下一步应先明确TINED原生4FC例外和GSSC适配边界，再同步正文的基线说明及表格。
勿直接重新跑本目录：已完成项必须通过身份核验后复用，禁止覆盖旧数据或筛seed。
