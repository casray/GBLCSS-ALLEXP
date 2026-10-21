# 实验历史与经验 23：Soft Target + Robust Selection 的适用域终判

## 1. 研究问题

本阶段围绕下一篇论文候选 `LCSS+soft-small-loss` 回答三个投稿前必须明确的问题：

1. uniform 0.6 最强区间中，X-only 方法与 NRGNN/RTGNN 等图原生方法还差多少？
2. small-loss 使用已知注入噪声率时，对噪声率误设是否敏感？
3. uniform 之外，pair/instance noise 是否仍存在“局部共识修复 + 小损失筛选”的独立协同？

粒球继续只扮演 frozen GRACE H 中的局部共识单元，不参与图原生模型训练，不统计 val/test 标签。

## 2. 方法与协议

- Split：percent 0.2/0.1/0.7。
- Noise scope：train-only。
- GPU：ROCm `cuda:0`。
- 图原生 seed0：Cora、CiteSeer、CoraFull、Amazon-Photo、PubMed；NRGNN、NRGNN[X||H]、RTGNN（CoraFull 无配置，记 NA）。
- 噪声率误设：真实 uniform 0.6，训练假设 0.4/0.5/0.6；Cora、CiteSeer、CoraFull seed0。
- 噪声类型：pair 0.4 与 instance 0.4；五数据集、四臂、seeds 0-4。
- 四臂：LCSS、Small-loss、LCSS+soft-small-loss、shuffled-LCSS+soft-small-loss。

代码未修改历史 predictor 行为；只新增严格汇总与测试。

## 3. 结果

### 3.1 Uniform 0.6 图原生边界

最佳冻结 X-only 路线相对最佳图原生方法：Cora -6.33pp、CiteSeer -8.89pp、CoraFull -5.83pp、Amazon-Photo -3.75pp、PubMed +0.78pp。仅 1/5 数据集进入“胜出或差距不超过 2pp”窗口，按预注册停止扩展。

### 3.2 噪声率误设

假设 0.5 相对正确 0.6 的差值为 -0.26/-1.03/+0.43pp；假设 0.4 为 -1.48/-0.52/-1.14pp。误差 0.1 通常约 1pp 内，但不是严格不敏感；低估 0.2 已产生明确损失。

### 3.3 Pair 0.4

五数据集 synergy：+1.04/+1.07/+0.90/-0.10/+4.52pp；平均 +1.49pp。4/5 数据集满足 paired 方向门，real-shuffled 5/5 稳定通过。总体 PASS。

### 3.4 Instance 0.4

五数据集 synergy：+3.55/+3.57/+2.69/+0.90/+4.47pp；平均 +3.04pp。5/5 数据集满足方向门，real-shuffled 5/5 稳定通过。总体 PASS。

## 4. 经验与边界

1. Direct X-only 鲁棒方法与图原生高噪声方法是不同资源档位；不能因为 graph-free 部署价值而回避 4–9pp 的绝对精度差距。
2. 局部共识与 small-loss 的协同不只存在于 symmetric/uniform noise；pair 与 instance 的五种子结果说明 target correction 和样本筛选确实提供不同信息。
3. Instance 的协同强于 pair；pair 中 Amazon-Photo 被 LCSS-only 吸收，说明“跨噪声类型有效”不能写成每数据集全胜。
4. 已知噪声率依赖可以用误设实验定量缓解，但尚未解决。真实未知噪声部署仍需估计器或不依赖噪声率的 schedule。

## 5. 决策

- 继续：将该组合作为下一篇论文的 graph-free high-noise robust training 主线。
- 停止：本轮不扩图原生五种子和 Coauthor-Physics，不再调整失败的蒸馏融合。
- 保留：plain distillation 作为独立 accuracy extension，不与 direct robust method 强行合并。
- Claim 上限：uniform 0.6、pair 0.4、instance 0.4 下存在稳定协同；不声称全面超过图原生 SOTA，也不声称未知噪声率问题已解决。

## 6. 可追溯资产

- 执行备案：`external/NoisyGL/docs/execution_plan_mlp33_u06_graphnative_sensitivity_noise_20260713.md`
- 原始结果与汇总：`external/NoisyGL/runs/mlp33_u06_graphnative_sensitivity_noise_20260713/`
- 汇总脚本：`external/NoisyGL/scripts/summarize_mlp33_u06_graphnative_sensitivity_noise.py`
- 测试：`external/NoisyGL/tests/test_mlp33_u06_graphnative_sensitivity_noise.py`
