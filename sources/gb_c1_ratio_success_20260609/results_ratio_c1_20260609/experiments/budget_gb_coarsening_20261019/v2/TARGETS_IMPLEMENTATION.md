# V2 目标构造与随机分球接口

日期：2026-10-19。范围：CPU 机制实现与单元验证；未进行训练、GPU 运算或模型性能实验。

## 接口

```python
targets, q, counts, diagnostics = build_targets(
    labels, assignment, train_indices, n_classes,
    mode="B", target_beta=0.5,
)
random_assignment = random_size_matched_assignment(assignment, seed=123)
```

- `labels`：长度 N。只检查和使用 `labels[train_indices]` 的值；非训练标签可以是 NaN 或范围外哨兵值。
- `assignment`：长度 N 的整数数组，分球 ID 必须恰为连续的 `0..M-1`。
- `train_indices`：长度 L 的非空、无重复整数索引。输出目标顺序与该数组严格一致；不接受布尔掩码。
- `n_classes`：至少为 2 的整数。训练标签必须是 `[0,C)` 中的整数值。
- `targets`：`float32 (L,C)`，每行均为概率分布。
- `q`：`float32 (M,C)`，所有模式均返回训练节点的 inclusive 球内经验分布。无训练支持的球整行为零。
- `counts`：`float32 (M,)`，每球训练节点数，不是全节点球大小。
- `diagnostics`：包括目标平均熵、训练支持总数/最小值/最大值/均值、空球/单支持球/有支持球数量；LCSS 还报告启用修正和有效平票的训练节点数。

所有模式均拒绝空训练集，以免将零行目标误作可运行监督任务。生产模块只导入 NumPy，不导入 PyTorch，不接收验证或测试掩码，不修改输入。

## 四种目标及语义边界

令训练节点 i 的噪声标签为 `y_i`，one-hot 为 `e_i`；所属球 B 的训练标签计数向量为 `h_B`，训练支持数为 `m_B`。

| mode | 训练节点目标 |
| --- | --- |
| `A` | `e_i` |
| `B` | inclusive 分布 `q_B=h_B/m_B` |
| `Bprime` | 若 `m_B>=2`，使用 `(h_B-e_i)/(m_B-1)`；否则使用 `e_i` |
| `LCSS` | 原始 GBLCSS 的置信度加权、平票安全留一目标，公式见下 |

`Bprime` 仅为纯留一平均，不是完整 GBLCSS。尤其 `m_B=2` 时，Bprime 使用另一个节点的标签，而 LCSS 保留自身 one-hot。C/D 若需要球级监督，应调用 `mode="B"` 并使用返回的 `q/counts`，不另造目标名称。

LCSS 保持 `GB-LCSS_GitHub/src/gblcss/targets.py::build_leave_one_out_targets` 的行为：

1. 若 `m_B<3`，不修正。
2. 否则 `p_i=(h_B-e_i)/(m_B-1)`；`alpha_i=clip((max(p_i)-1/C)/(1-1/C),0,1)`。
3. 若留一计数并列最大，`local_i=p_i`；不按类别索引打破平票。
4. 若唯一最大类别为 k，`local_i=(1-target_beta)*p_i+target_beta*onehot(k)`。
5. 最终 `t_i=(1-alpha_i)*e_i+alpha_i*local_i`。默认 `target_beta=.5`，其余 beta 值也与原始函数一致。

原始函数为所有 N 个节点分配目标；本接口只返回 L 个训练节点目标，但原始函数对应训练行与本接口在 `atol=1e-7` 内一致。统计只依赖训练标签；有标签但不属于 `train_indices` 的节点不参与计数。

随机分球直接计算 `default_rng(seed).permutation(assignment)`。每个球 ID 对应的全节点数都精确保留，因此也保留球大小多重集合。函数不读取标签或掩码；随机后训练支持数可能变化，这是随机成员归属产生的结果。个别节点或整个极小划分可能保持不变，不做条件重抽样。

## 球级交叉熵的等式条件

当同一球内所有训练节点共享同一个预测 `p_B` 时：

```text
sum_B m_B * CE(q_B, p_B)
= -sum_B sum_c h_Bc * log(p_Bc)
= sum_(i in train) CE(e_i, p_B(i)).
```

这是损失之和的等式。两侧均除以 `L=sum_B m_B` 才得到相同的逐训练节点平均损失。不能把按有支持球数求平均的结果直接当成逐节点平均。

节点预测不相同或在平均特征上重新计算预测时，上述等式一般不成立。测试反例：单球两个训练标签 `[0,1]`，节点预测分别为 `[.9,.1]`、`[.1,.9]`，逐节点 CE 之和约 `0.210721`；平均预测为 `[.5,.5]`，球级加权 CE 约 `1.386294`。因此不应把粗化训练描述为对任意原始节点预测的严格等价压缩。

## 验证与结论

先写 `test_mechanisms.py` 并运行：14 项均因 `build_targets API is not implemented` 断言失败。再实现 `mechanisms.py`，同一命令运行通过 14 项，无失败或跳过。

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' '.\experiments\budget_gb_coarsening_20261019\v2\test_mechanisms.py'
```

覆盖范围：

- A/B/Bprime 的手算目标、训练索引顺序、空球、单支持球、双支持球及概率与 float32 输出。
- LCSS 的唯一多数类、平票、至少两个剩余投票条件。
- 原始 GBLCSS 函数直接对照：1 个固定案例及 36 个随机案例，类别数覆盖 2/3/7；各测试 beta 0/.5/1，共 111 次目标与诊断对照。
- 将所有非训练标签置为 NaN，四种模式的三个数组输出及全部诊断均保持相同。
- 非连续分球 ID、越界或重复训练索引、空训练集、错误标签和 beta 的明确异常。
- 随机分球保留精确全节点球大小、同 seed 可复现、输入不被修改。
- 共享预测下球级加权交叉熵与原损失之和相同，以及不同节点预测下不相同的反例。

已支持的结论：本接口在上述测试范围内满足目标定义、训练标签隔离和尺寸匹配随机对照合同；LCSS 对齐原始实现。未验证的内容：完整训练的数值稳定性、性能、泛化及研究假设。这里的单元测试通过不能作为方法有效性或精度提升证据。下一步可将这些接口接入 runner；无需修改目标公式。
