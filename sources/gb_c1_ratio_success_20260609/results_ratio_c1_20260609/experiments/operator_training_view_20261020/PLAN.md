# 粒球投影训练视图：资格赛计划

状态：用户于2026-10-20授权开始；本文件在新数据集训练前编写。先执行资格赛，不预先扩展强方法180次矩阵。

## 研究假设与粒球角色

H1：训练时使用S=Pi P Pi会改变含噪监督下的参数学习，收益不完全来自B软目标或dropout共享。H2：投影与GCE可互补，S+GCE需相对P+GCE有增量。H3：分区语义影响效果，但粒球未必独有；平铺k-means是必要强控制。

粒球是训练传播投影单元；不做R3/O2评分，不追求纯度，不生成LLP比例损失，不新增蒸馏或粒度门控。主推理仍使用原图P。粗图是S+共享mask的一种等价实现，不据此宣称端到端高效。

## 冻结范围

四单元：CiteSeer u0.6/clean；CoraFull u0.4/clean。种子0/1/2，20/10/70划分、uniform翻转到其他类、pair映射c→c−1，与v2一致。GCN hidden256、dropout .5、Adam lr.01/wd.0005、最多300轮、每5轮原图验证、patience10次检查。主预算round(.1N)。不依测试选模。

复用v2已审计x/y/P、候选树和R0切面，只读。新分区与运行缓存写入独立目录；缓存键采用结构化输入哈希，不对含小数噪声名使用with_suffix。k-means使用同一U/PU/P²U坐标，n_init=10、max_iter=300、算法lloyd、seed20261019，球数相同；无标签生成。随机分区按全节点球大小匹配，seed900000+协议seed。

## 15个预定臂，共180次新训练

1. P/独立dropout/自身标签CE，A复验。
2. P/独立dropout/B目标CE，B复验。
3. P/球共享dropout/B目标CE。
4. P Pi/球共享dropout/B目标CE。
5. Pi P/球共享dropout/B目标CE。
6. Pi P Pi/球共享dropout/B目标CE，节点实现。
7. Pi P Pi/独立dropout/B目标CE。
8. 粗图RX/RPH/球级dropout/B计数CE，C复验与实现对照。
9. P/独立dropout/自身标签GCE，q=.7。
10. P/球共享dropout/自身标签GCE，插件匹配控制。
11. Pi P Pi/球共享dropout/自身标签GCE，插件。
12. P/独立dropout/标签平滑epsilon=.1，目标=.9onehot+.1/C。
13. P/独立dropout/传播后倒数特征层mixup，alpha=1、每步一个Beta系数、仅训练节点随机配对；所有图传播后混合表示/标签，不改变原图边。最终线性分类器下等价于混合训练logits；包含原GCN的传播后偏置分量。明确为本项目manifold-mixup适配，不冒称复现官方图Mixup。
14. k-means分区/Pi P Pi/共享dropout/B目标CE。
15. 大小匹配随机分区/Pi P Pi/共享dropout/B目标CE。

除实现对照，所有主比较在同一新引擎重跑，不混用历史均值。原节点B与C目标源/分区严格配对。单侧算子每层均应用；主算子分解统一共享mask。P/S×独立/共享mask组成四角。

## 校验与成本

先写测试，观察缺失功能失败，再实现。验证：显式稠密小矩阵与无N²投影实现一致；带偏置GCN的前向、loss、梯度和数步Adam等价；CE自身标签与B目标在球内恒定logits时等价；GCE计数风险等于逐标签风险；共享mask索引正确；反向梯度有限；CPU float64与GPU float32均检查。GPU原生dropout与共享mask的单步配对需明确传入同一mask。长期准确率不要求逐数相同。

每臂独立进程、GPU串行。保存主要checkpoint，在同一checkpoint下分别报告P与训练视图预测；训练视图只作诊断，不另选checkpoint。记录配置/源码/数据/分区/训练目标/预测哈希，raw log、每次验证、accuracy、macro-F1/逐类召回，训练/验证/推理/准备时间、完整GPU峰值和进程墙钟。缓存复用次数和历史预处理成本单列，不将每轮速度冒充总加速。

## 资格赛门槛与后续

本轮是四单元筛查，不声称五图普适性。噪声单元分母固定为2。

- 算子价值：S_shared相对P_shared的B目标CE，两个带噪单元至少一个mean提升≥1pp且≥2/3同seed差为正；其余负单元与clean损害均单列，不平均掩盖。
- 插件价值：S_shared+GCE相对P_shared+GCE，两个带噪单元至少一个mean提升≥1pp且≥2/3正；不得以P+CE为插件基线。
- 至少一条成立，才有资格扩完整11条件；若单独不超过通用正则且插件无上述增量，停止当前规格，不调参救援。
- 两图clean任何一图相对原节点A下降>2pp，则标记“有明显信息损失、不能作为统一替代”，即使噪声通过也不宣布方法成功；只能进入完整11条件机制确认，不设计粒度门控。
- k-means与粒球±.5pp内仅作实践接近描述，三seed不证明统计等价。
- 若扩展：四正信号+两负信号+五图clean=11条件。强鲁棒方法和插件须另行冻结适配后运行；最终五图三噪声插件门槛的分母固定15，至少8个mean增量≥1pp并报告seed差/std及clean损害。资格赛不代替该门槛。

## 实现步骤与实际文件

- [x] operators.py/test_operators.py：投影、GCN、共享mask与损失，先测试失败再实现。
- [x] experiment.py/test_experiment.py：变体清单、缓存键、复用缓存准备和结果配对。
- [x] run_job.py：复用旧协议辅助函数，单臂GPU训练与工件。
- [x] run_queue.py：冻结源码/配置/缓存指纹、独立进程串行、可核验续跑、超时停止。
- [x] config.json、CPU/GPU核验、短轮次冒烟与结果审计。27项测试、18次六轮冒烟及全部预测/监督审计通过。
- [x] 冻结D盘独立运行源，启动正式队列并记录进度。2026-10-20 10:54:37（北京时间）后台启动，Newidea执行记录见同目录说明。
- [x] 完成后逐运行审计、完整正负结果与方向历史，按门槛裁决。180/180审计、60组汇总和60行配对复算通过；结论见analysis/final_20261020/REPORT.md。门槛通过但CoraFull clean损害超过2pp；尚未启动扩展训练。

最小命令（GPU环境）：`D:/codex_temp/noisygl-rocm721/Scripts/python.exe -m unittest discover -s experiments/operator_training_view_20261020 -p test_*.py -v`；正式队列使用独立source/run_queue.py及冻结config.json，具体绝对路径写入LAUNCH.json。新增代码不修改v2源文件。
