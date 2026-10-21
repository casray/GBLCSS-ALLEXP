# Robust-Student Distillation Design

## Goal

检验标准 `GCN[X||H]` teacher 的图知识与 LCSS+soft-small-loss 的高噪声学生优化能否在 X-only student 中形成额外增益，同时保持局部结构归因清晰。

## Scope

- Datasets: Cora, CiteSeer, CoraFull, Amazon-Photo, PubMed, Coauthor-Physics。
- Protocol: uniform 0.4, train-only noise, percent 0.2/0.1/0.7, seed0 smoke。
- Teacher: 固定 `GCN[X||H]`，只使用 noisy train CE；teacher 不使用球或 small-loss。
- Student: 2-layer X-only MLP；推理不使用 H、邻接或 teacher。
- Existing anchors remain unchanged.

## Student objective

For all nodes, distill frozen teacher probabilities:

\[
L_{KD}=T^2\frac{1}{|V|}\sum_i H(p_i^t(T),p_i^s(T)).
\]

For train nodes, compute per-node soft CE against LCSS targets, retain the smallest-loss fraction using the same frozen schedule as MLP-31, and optimize:

\[
L_{rob}=\frac{1}{|S_e|}\sum_{i\in S_e}H(q_i^{LCSS},p_i^s).
\]

The smoke uses the preregistered equal-weight objective:

\[
L=L_{KD}+L_{rob}.
\]

No weight grid is allowed. The shuffled arm changes only ball assignments; teacher probabilities, split, noise, model, optimizer, and weights remain identical.

## Arms

1. Direct LCSS+soft-small-loss, reused from MLP-30.
2. `GCN[X||H]` teacher accuracy, emitted by the plain-distillation diagnostics.
3. Plain `GCN[X||H] -> MLP` distillation.
4. Robust-student distillation: KD + real LCSS soft-small-loss.
5. Shuffled robust-student distillation.

## Seed0 gate

The robust student proceeds to five seeds only if all conditions hold:

1. It exceeds plain distillation on at least 4/6 datasets.
2. Mean robust-student minus plain-distillation is at least +1pp.
3. It exceeds direct LCSS+soft-small-loss on at least 4/6 datasets.
4. Real minus shuffled is at least +1pp on at least 4/6 datasets.
5. Student test accuracy is at least 95% of teacher test accuracy on at least 5/6 datasets.
6. No dataset regresses more than 1pp versus plain distillation.

Failing the gate closes this exact equal-weight student fusion; no second weight or temperature is tried in this paper cycle.

## Graph-native follow-up

After the distillation smoke, audit NRGNN and NRGNN[X||H] first, followed by other already runnable graph-native baselines. Reuse only rows with exact dataset, seed, split, noise scope/type/rate, and hash compatibility. Missing rows are rerun under the MLP-32 protocol and reported in a separate cross-resource reference table.

## Integrity

- Ball targets use only noisy train labels; clean labels are evaluation-only.
- KD uses teacher predictions, not clean labels.
- Validation/test labels are never modified.
- Student inference calls only the X-only MLP.
- Every output records teacher/student dimensions, loss weights, selected count/ratio, split hash, and noise hash.

