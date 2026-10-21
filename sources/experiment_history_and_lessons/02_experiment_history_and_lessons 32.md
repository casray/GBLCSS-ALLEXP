# MLP-42：表 4 共识教师蒸馏广度扩展

## 研究假设

现有表 4 仅覆盖 Cora、CiteSeer 与 CoraFull。若共识教师的归因具有跨域广度，它应在 Coauthor-CS 和 Amazon-Photo 的 uniform 0.4/0.6 下仍表现为真实球式优于随机分组，并与直接 CE teacher、vote-matched kNN teacher 形成可解释的配对关系。

## 粒球角色与方法动机

粒球在 frozen GRACE H 上形成 train-only 局部共识 soft target，用于训练 GCN[X‖H] teacher；最终 student 是相同的 X-only 两层 MLP，推理不使用图。扩展 coauthor 与 co-purchase 图是为了验证教师路径归因的适用范围，不用于事后挑选有利数据集。

## 代码与配置改动

- 未改 predictor、球参数、teacher/student 或已有配置。
- 新增 `external/NoisyGL/scripts/summarize_mlp42_table4_breadth.py`。
- 新增 `external/NoisyGL/tests/test_mlp42_table4_breadth.py`，覆盖完整性、哈希配对和标签零泄漏。
- 运行输出位于 `external/NoisyGL/runs/mlp42_table4_breadth_20260717/`。

## 固定协议

- 数据集：Coauthor-CS、Amazon-Photo。
- 噪声：train-only uniform 0.4/0.6。
- Split：percent 0.2/0.1/0.7。
- Seeds：0--4。
- 五臂：GCN(X)+CE、GCN[X‖H]+CE、shuffled 共识教师、vote-matched kNN 共识教师、球式共识教师。
- kNN 的 k 不按精度搜索；只匹配同 seed 平均球内训练票数。u0.6 下 Coauthor-CS/Amazon-Photo 分别为 57/24。

## 预期机制与判定

如果局部球结构真实有贡献，球式应高于 shuffled；如果球式 target 相对直接 teacher 有去噪增量，应高于 GCN[X‖H]+CE；ball−kNN 只回答分组实现差异，不作为粒球必须获胜的门槛。完整性要求为 100 个唯一单元、五臂同 seed 哈希配对、无 val/test 污染。

## 结果

| 教师 | Coauthor-CS u0.4 | Amazon-Photo u0.4 | Coauthor-CS u0.6 | Amazon-Photo u0.6 |
|---|---:|---:|---:|---:|
| GCN(X)+CE | 93.20±0.09 | 89.86±0.77 | 91.13±0.58 | 84.80±1.48 |
| GCN[X‖H]+CE | 93.35±0.55 | 90.51±1.08 | 91.16±1.32 | 85.10±3.27 |
| Shuffled | 93.11±0.39 | 89.29±0.26 | 91.23±0.50 | 80.11±9.78 |
| Vote-matched kNN | 93.69±0.43 | 90.15±0.93 | 91.66±0.90 | **86.82±0.61** |
| Ball | **93.75±0.22** | 90.49±1.05 | **92.25±0.69** | 85.87±2.25 |

Coauthor-CS：ball−shuffled 为 `+0.64/+1.02pp`（u0.4/u0.6，均 5/5 正）；ball−GCN[X‖H] 为 `+0.40/+1.09pp`；ball−kNN 为 `+0.06/+0.59pp`。u0.6 的证据最清晰。

Amazon-Photo：u0.4 ball−shuffled `+1.20pp`、ball−kNN `+0.34pp`，但 ball−GCN[X‖H] `−0.02pp`；u0.6 ball−shuffled `+5.76pp`，但 ball−kNN `−0.95pp`，且 shuffled/XH teacher 的 seed 方差很大。

完整性：100/100 单元成功，零 val/test 标签修改，五臂同 seed split/noise hash 一致；汇总回归测试 3/3 通过。

## 最终决策

- **Coauthor-CS：继续/可纳入扩展表。** 两个噪声率均支持球式共识，u0.6 证据尤其稳定。
- **Amazon-Photo：边界结果。** 真实球优于 shuffled，但不能稳定超过直接 XH teacher 或 kNN；必须完整披露，不能作为球式独占优势。
- **论文处置：暂不自动修改 v15。** 作者决定扩表时，两数据集应整体加入；不可只报告有利的 Coauthor-CS。
