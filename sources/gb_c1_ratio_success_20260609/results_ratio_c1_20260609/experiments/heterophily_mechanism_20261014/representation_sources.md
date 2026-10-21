# 新表示来源、协议与移植边界

记录日期：2026-10-14。此文档在新表示训练前建立；训练结果以每个输出目录的 `config.json`、`training.jsonl`、`summary.json` 为准。所有配置先写盘后训练，表示阶段无标签参数、不读取标签、不运行分类评估，固定末轮导出。

## Roman-empire 官方数据

- 来源：[yandex-research/heterophilous-graphs](https://github.com/yandex-research/heterophilous-graphs)，核验 commit `a431395582e929d88271309716bea4fe24ce6318`。
- 文件：[roman_empire.npz](https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/roman_empire.npz)。本地 `data/roman_empire.npz`，20,401,489 bytes。
- SHA256：`a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428`。运行加载器会核对该哈希，避免默默更换输入。
- `node_features`: float32 `[22662,300]`；`node_labels`: int64 `[22662]`；`edges`: int64 `[32927,2]`；三组 `*_masks`: bool `[10,22662]`。split0 的 train/val/test 为 11331/5665/5666；本项目使用官方 split0–4。
- 官方 `datasets.py` 第26–27行对名称不含 directed 的图调用 `dgl.to_bidirected`。本实现对称化并去重后为 65,854 条有向边，0 孤立节点，0 原生自环；节点行序不变。GRACE 的 GCNConv 自加自环；GraphACL 的 GraphConv 不加自环。
- 表示训练入口只从 NPZ 加载 node_features/edges，分类标签与划分不进入 SSL。

## GRACE

直接复用只读 `GB-LCSS_code/src/gblcss/grace.py` 的 Encoder、GRACE projector、特征丢弃、特征归一化；仅损失计算改为精确分块且按块 checkpoint 重算反向，以降低 N² 激活显存。对比集合仍为全部节点，没有负采样或 minibatch 目标替换。

Roman 缺少本项目原始预训练配置，采用预先冻结的 DBLP fallback：hidden256、projection256、2层（中间宽512）、ReLU、Adam lr.001/wd1e-5、edge drop [.1,.4]、feature mask [.1,0]、temperature.7、1000epoch、seed0。输入遵循项目原 GRACE 的逐行最小值平移与行和归一化，并非修改官方存档数据。表示导出 L2 归一化。

测试以 float64 小图比较原 dense 公式和 chunk1/3/20 的标量损失及两个输入梯度，绝对/相对容差1e-12。chunk checkpoint 只改变计算排程，GPU浮点累加仍可能产生舍入差异。

## GraphACL 官方源码审计与可移植扩展

官方：[tengxiao1/GraphACL](https://github.com/tengxiao1/GraphACL)，commit `c3aa4b2ba8b901c630140c7ab689a8cb4a693457`。`vendor/graphacl_hete_*` 保存该提交的 hete/model.py、layers.py、main.py、utils.py、dataset.py、run.sh；文件 SHA256 见 `vendor/source_hashes.json`。

已核实：

1. hete 编码器为两个 DGL GraphConv，默认 norm=both、bias=True，两层之间没有非线性，没有自环，没有激活的 PairNorm（虽实例化但 forward 未调用）。移植实现对出度和入度作 -1/2 归一化聚合，Xavier uniform 权重、零bias。
2. 在线表示经可选多层 Linear/ReLU predictor 得到投影；目标编码器复制初始化、冻结梯度，optimizer.step 后按 EMA 更新。在线/目标/投影只在目标函数入口作 L2 归一化。
3. 每节点从邻居采样K个正目标，从全部节点采样K个在线负表示；负样本不排除自己。负端在线表示保留梯度。
4. 正打分是 projected_online 与 target_neighbor；负打分是 unprojected_online 与 sampled_online。每个正样本的分母为其正分数加共享负分数之和。本实现与官方 contrastive_loss 函数通过 AST 原样提取作数值/梯度一致性测试。
5. 官方 get_emb 在低度节点上原地扩展邻居列表至K；后续轮次在这个含重复元素的列表采样。本实现保留该实际代码行为，而不将更改抽样当作同一实现。负采样用 Python random.sample，无放回；不足邻居的填充用 NumPy random.choice。
6. 官方学习器 Adam weight_decay=.0003；训练标签不进入预训练。

**限制与预注册扩展：**官方 hete loader 仅支持 cornell/texas/wisconsin/squirrel/chameleon/crocodile/film，没有 Roman-empire，也无其参数。移植入口 `train-graphacl` 是新数据集上的探索扩展，不是论文 Roman 指标复现。为控制开销，预先固定 hidden512、两层、1层predictor、lr.0001/wd.0003、tau.5、EMA.95、K5、seed0、50epochs；50轮对应官方 hete/main.py 默认轮数尺度，不根据本次标签或诊断调参。512维是本次扩展fallback，绝非官方Roman最佳配置。Roman 使用原始 float 特征，和 GRACE 的输入归一化不同。

正式预算冻结为 GRACE 1000epochs（既定DBLP fallback）和 GraphACL 50epochs。GRACE 256维、GraphACL 512维，输入预处理、训练轮数和实际计算预算均不匹配，因此本次比较只能解释为整套表示pipeline替换（pipeline swap）；不能据此隔离或声称“只替换对比损失”造成某个效果。短pilot仅核验执行、数值和速度，单独存储，不能替代这两个正式预算。

该环境未运行 DGL 端到端对照；GCN验证采用 DGL 公式的独立手算矩阵。DGL 初始化/边顺序与此移植可能影响随机路径，不能声称逐比特一致。此外本实现导出最后 optimizer.step 后的编码，而官方 hete/main.py 返回最后一步更新前的 node_embeddings；这项差异明确保存于输出配置。低度节点填充副作用保留，但采样邻居的有序列表采用 PyG coalesce 顺序，可能与 DGL/NetworkX 顺序不同。

## 运行接口和输出

先由主进程协调单GPU时段，不能并行训练。

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m pytest experiments/heterophily_mechanism_20261014/test_representations.py -q -p no:cacheprovider
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/heterophily_mechanism_20261014/prepare_representations.py --action train-grace --epochs 3 --device cuda --output experiments/heterophily_mechanism_20261014/representations/Roman-empire/grace_pilot
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/heterophily_mechanism_20261014/prepare_representations.py --action train-grace --epochs 1000 --device cuda --output experiments/heterophily_mechanism_20261014/representations/Roman-empire/grace
```

GraphACL 将 action 改为 train-graphacl，指定单独 output 并设置 `--epochs 50`。当前 runner 专门接受经哈希锁定的 Roman 输入；CoraFull/CoauthorCS GraphACL 未实现，不能把已有 GRACE 缓存改名为 GraphACL。

输出目录包括 H.pt（CPU float32 N×D、行L2归一化）、model.pt、config.json（输入/脚本哈希及配置）、training.jsonl（逐轮loss/耗时）、summary.json（总耗时、峰值GPU分配、H哈希）。已完成 H.pt 禁止覆盖。pilot 输出仅供速度/数值核验，不能混入1000epoch正式诊断。

测试包含完整CPU小图的 encoder → projection → chunk checkpoint loss → backward 参数梯度对照，与原GRACE dense loss逐个参数比较，float64容差1e-12。该检查覆盖GCN和projector参数链，不只检验独立输入张量；GPU端数值仍待pilot检验。

测试状态：10项通过；仅PyG自带 distributed deprecation warning。2026-10-14更新：Roman GRACE1000轮与GraphACL扩展50轮均已完成，pipeline_status.json为completed；完整目标诊断、学生pilot和审计结果见实验结论.md与audit/report.json。3轮GRACE pilot仍只用于执行检查。
