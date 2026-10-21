# Graph-free baseline identity audit

## Purpose

本审计限定论文表 3 中各行可以使用的方法名称与 claim。所有结果均使用 percent 0.2/0.1/0.7、uniform 0.4 train-only 噪声和 5 seeds，但“协议一致”不等于“官方复现”。

| Manuscript row | Preserved identity | Local adaptation | Allowed label |
|---|---|---|---|
| GLNN-style | GCN teacher soft responses supervise all-node X-only MLP | Common teacher/student sizes, optimizer and noisy-label protocol | `GLNN-style distillation`; not official tuned reproduction |
| GSDN-inspired | Dual MLP heads, edge feature mixup, train-source label propagation, negative regularization | Reimplemented losses and common MLP protocol | `GSDN-inspired (mechanism adaptation)` |
| Graph-MLP | Dedicated MLP architecture, r-hop adjacency NContrast, official Cora/CiteSeer alpha/tau | NoisyGL loader/split/noise; repository defaults for added datasets | `Graph-MLP (protocol port)` |
| FF-G2M | Noisy hard-label CE, neighborhood low-frequency KL, edge-difference high-frequency KL | Common 2-layer width-64 GCN/MLP and validation-only model selection | `FF-G2M (common-backbone reimplementation)` |
| KRD proxy | Perturbation-derived reliability signal only | Does not preserve the paper's full sampling dynamics | `KRD-inspired reliability proxy`; excluded from published-method ranking |
| VQGraph proxy | Locally designed codebook/token signal only | Does not reproduce the official tokenizer/codebook training pipeline | `VQGraph-inspired token proxy`; excluded from published-method ranking |

## Code trace

- GLNN-style: `predictor/FrozenGraceGB_GCN_Predictor.py`, class `mlp_distill_gcn_Predictor`.
- GSDN-inspired: `predictor/GraphFreeFamily_Predictor.py`, class `gsdn_Predictor`.
- Graph-MLP port: `predictor/GraphMLPReimplemented_Predictor.py`.
- FF-G2M reimplementation and remaining proxies: `predictor/G2MBaselines_Predictor.py`.

## Paper boundary

正式排名中可以将 Graph-MLP 和 FF-G2M 作为保留核心方法身份后的同协议实现，但不能把其数值称为官方 tuned results。GLNN-style 是基础蒸馏参照，不用于宣称复现 GLNN 官方数字。GSDN、KRD 和 VQGraph 当前只能支撑机制比较；其中 KRD/VQGraph 的代理数值不得计入“超过多少个已发表方法”的统计。
