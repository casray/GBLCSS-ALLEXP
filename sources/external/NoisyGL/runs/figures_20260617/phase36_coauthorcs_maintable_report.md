# Phase 36 Coauthor-CS main-table baseline report

Protocol: percent split 0.2/0.1/0.7, uniform noise 0.4, seeds 0-4.

## Aggregate

| Dataset | Method | Test acc |
|---|---|---:|
| cora | GCN+CE | 0.7842 ± 0.0151 |
| cora | GCN+CE [X||H] | 0.8027 ± 0.0069 |
| cora | NRGNN | 0.8107 ± 0.0230 |
| cora | NRGNN [X||H] | 0.8139 ± 0.0170 |
| cora | SoftDist [X||H] | 0.8137 ± 0.0067 |
| cora | SoftDist [X||H] shuffled | 0.7921 ± 0.0097 |
| citeseer | GCN+CE | 0.6530 ± 0.0121 |
| citeseer | GCN+CE [X||H] | 0.6761 ± 0.0181 |
| citeseer | NRGNN | 0.7027 ± 0.0108 |
| citeseer | NRGNN [X||H] | 0.7090 ± 0.0187 |
| citeseer | SoftDist [X||H] | 0.6891 ± 0.0159 |
| citeseer | SoftDist [X||H] shuffled | 0.6696 ± 0.0234 |
| coauthorcs | GCN+CE | 0.9196 ± 0.0017 |
| coauthorcs | GCN+CE [X||H] | 0.9190 ± 0.0010 |
| coauthorcs | NRGNN | 0.9356 ± 0.0022 |
| coauthorcs | NRGNN [X||H] | 0.9366 ± 0.0043 |
| coauthorcs | SoftDist [X||H] | 0.9243 ± 0.0018 |
| coauthorcs | SoftDist [X||H] shuffled | 0.9188 ± 0.0013 |
| amazoncom | GCN+CE | 0.8682 ± 0.0061 |
| amazoncom | GCN+CE [X||H] | 0.8695 ± 0.0092 |
| amazoncom | NRGNN | 0.7923 ± 0.0614 |
| amazoncom | NRGNN [X||H] | 0.8230 ± 0.0150 |
| amazoncom | SoftDist [X||H] | 0.8625 ± 0.0063 |
| amazoncom | SoftDist [X||H] shuffled | 0.8443 ± 0.0196 |

## Paired deltas

| Dataset | Comparison | Delta |
|---|---|---:|
| cora | SoftDist [X||H] - GCN+CE [X||H] | +1.10pp (4/5) |
| cora | SoftDist [X||H] - SoftDist [X||H] shuffled | +2.16pp (5/5) |
| cora | SoftDist [X||H] - NRGNN [X||H] | -0.02pp (3/5) |
| cora | NRGNN [X||H] - NRGNN | +0.33pp (3/5) |
| cora | GCN+CE [X||H] - GCN+CE | +1.86pp (5/5) |
| citeseer | SoftDist [X||H] - GCN+CE [X||H] | +1.30pp (5/5) |
| citeseer | SoftDist [X||H] - SoftDist [X||H] shuffled | +1.95pp (5/5) |
| citeseer | SoftDist [X||H] - NRGNN [X||H] | -1.99pp (1/5) |
| citeseer | NRGNN [X||H] - NRGNN | +0.64pp (4/5) |
| citeseer | GCN+CE [X||H] - GCN+CE | +2.31pp (4/5) |
| coauthorcs | SoftDist [X||H] - GCN+CE [X||H] | +0.53pp (5/5) |
| coauthorcs | SoftDist [X||H] - SoftDist [X||H] shuffled | +0.55pp (5/5) |
| coauthorcs | SoftDist [X||H] - NRGNN [X||H] | -1.23pp (0/5) |
| coauthorcs | NRGNN [X||H] - NRGNN | +0.10pp (4/5) |
| coauthorcs | GCN+CE [X||H] - GCN+CE | -0.06pp (1/5) |
| amazoncom | SoftDist [X||H] - GCN+CE [X||H] | -0.70pp (0/5) |
| amazoncom | SoftDist [X||H] - SoftDist [X||H] shuffled | +1.82pp (5/5) |
| amazoncom | SoftDist [X||H] - NRGNN [X||H] | +3.95pp (5/5) |
| amazoncom | NRGNN [X||H] - NRGNN | +3.07pp (4/5) |
| amazoncom | GCN+CE [X||H] - GCN+CE | +0.14pp (3/5) |

## Boundary interpretation

- Coauthor-CS tests whether large graphs necessarily fail. A positive ball gain here rules out graph size as the sole failure variable.
- Amazon-Computers is retained as a partial boundary point: real balls beat shuffled balls, but the full-sample GCN [X||H] absorbs the gain.
- The resulting boundary is size x class count x graph regime, not a one-dimensional small-graph-only story.

Figure: `size_class_boundary.png`.
