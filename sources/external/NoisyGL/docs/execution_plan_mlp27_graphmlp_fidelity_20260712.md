# MLP-27 Graph-MLP 忠实度补跑

## 固定协议

- uniform 0.4、train-only、percent 0.2/0.1/0.7。
- 专用 Graph-MLP 网络：Linear-GELU-LayerNorm-Dropout-Linear + classifier。
- NContrast 使用归一化含自环邻接的二阶矩阵，以及完整 in-batch 余弦相似度矩阵。
- 每批包含全部 train 节点，batch size 至少为 2000。
- Cora 官方参数 alpha=10/tau=2；CiteSeer alpha=1/tau=0.5；其余数据集用仓库默认 alpha=2/tau=1。
- 推理仅输入 X。

## 闸门

1. Cora/CiteSeer seed0 无 OOM、loss 有限。
2. CoraFull 单独 smoke；其 3958 个 train 节点要求至少 3958×3958 相似度矩阵。
3. 单 run 不超过 30 分钟才扩 5-seed。

## 身份说明

保留官方网络和 NContrast 核心，数据加载、噪声注入和 split 迁移到本文协议，称 `Graph-MLP (protocol port)`。

## 执行结果

- 25/25 个运行成功，无 NaN 或 OOM。
- Cora：64.91±2.04；CiteSeer：59.87±1.73；CoraFull：44.51±0.55。
- Coauthor-CS：89.97±0.39；Amazon-Photo：80.44±0.65。
- 忠实移植比旧 edge-contrastive proxy 高 1.02--7.46 个百分点，说明旧代理低估了 Graph-MLP。
- LCSS 蒸馏扩展在五个数据集上仍高 3.78--15.69 个百分点。
- 最终决策：用 `Graph-MLP (protocol port)` 替换旧 proxy；不声称官方 tuned reproduction。

完整结果见 `runs/mlp27_graphmlp_fidelity_20260712/mlp27_report.md`。
