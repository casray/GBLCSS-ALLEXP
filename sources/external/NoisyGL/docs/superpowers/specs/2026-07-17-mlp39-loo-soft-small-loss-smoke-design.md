# MLP-39 LOO-LCSS + Soft-Small-Loss Smoke Design

## 目标

检验 leave-one-out 局部共识 target 是否能进一步改善 soft-small-loss 的逐节点损失排序。实验只改变球内投票是否包含当前节点自身 noisy label，不改变 GRACE H、分组、beta、MLP、forget-rate schedule 或评估协议。

## 方法

新增两个直接训练的 X-only MLP 变体：

1. `frozengrace_gb_mlp_softdist_beta05_small_loss_loo`：使用 LOO-LCSS soft target 计算逐节点 soft CE，并按现有 small-loss schedule 选择低损失节点。
2. `frozengrace_gb_mlp_softdist_beta05_small_loss_loo_shuffled`：使用 size-matched shuffled 分组构造同样的 LOO target，作为真实局部结构归因对照。

对节点 v，从所在球的 train noisy-label 计数中移除 v 自身：

\[
p_{B\setminus v}(c)=\frac{n_{B,c}-\mathbf 1(\tilde y_v=c)}{|B_{train}|-1}.
\]

剩余外部票数少于 2 时沿用 MLP-36 的 one-hot fallback。其余 multiclass alpha、distribution target 和 beta=0.5 保持不变。

## 最小协议

- 数据集：Cora、CiteSeer、CoraFull。
- 噪声：uniform0.4，只污染 train。
- Split：percent 0.2/0.1/0.7。
- Seed：0。
- 新运行：2 methods × 3 datasets = 6 runs。
- 复用锚点：MLP-30 的默认 LCSS、Small-loss、LCSS+soft-small-loss；MLP-36 的 LOO-LCSS。只在 split/noise hash 完全匹配时复用。

## 关键比较

- `LOO-combo - default-combo`：LOO 对 robust combination 的条件增量。
- `LOO-combo - LOO-LCSS`：small-loss 在 LOO target 上的增量。
- `LOO-combo - Small-loss`：局部共识 target 相对纯样本选择的增量。
- `LOO-combo - LOO-shuffled-combo`：真实局部结构归因。

## Seed0 进入五种子的门

全部满足才扩种子：

1. 至少 2/3 数据集 `LOO-combo > default-combo`。
2. 三数据集平均条件增量至少 +0.5pp。
3. 任一数据集条件增量不低于 -0.5pp。
4. 至少 2/3 数据集 real-shuffled 至少 +1pp。
5. 全部运行无 NaN/OOM/单类坍缩，val/test 标签修改数为 0，锚点 hash 完全匹配。

若未通过，不调整 beta、forget rate、粒球参数或第二版 LOO 公式。该方向直接关闭或保留为负结果。

## 结论边界

该实验属于下一篇 robust-optimization 候选，不修改已冻结的第一篇 LCSS 论文。Seed0 通过只允许进入 paired 5-seed，不构成论文性能结论。
