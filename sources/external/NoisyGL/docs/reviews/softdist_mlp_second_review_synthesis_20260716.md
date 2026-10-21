# SoftDist-MLP / LCSS 二审综合结论

日期：2026-07-16  
审稿对象：`paper_final_zh_v14_second_review_ready_20260716.md`

## 1. 总判定

两位二审均给出 **Major Revision，但已进入 KBS 可修范围**。

- `academic-paper-review`：Major Revision / Borderline，贡献等级 Moderate。
- `peer-review`：Major Revision；完成投稿前必需项后可进入 Borderline / Weak Accept。

这与一审的“存在核心报告缺陷”不同。二审认为方法定义、主要对照、统计口径和负结果披露已经基本成立；剩余问题集中在确认性证据、完整管线方差和 artifact，而不是继续追求更高精度。

## 2. 两位审稿人的共同意见

### 已通过二审的部分

1. noisy-label + graph-free inference 的交叉问题成立。
2. H-access 资源不对称已经通过独立资源表和收窄 claim 处理。
3. 表 1 已展开正式 baseline，不再只展示最强对照。
4. LOO、普通平滑、shuffled 和 target-quality 足以证明收益不只是自票或全局标签平滑。
5. Holm、paired delta、CI 和方向计数比一审统计口径可靠；未再把“未显著”写成等效。
6. kNN/kmeans 可替代性和失败数据集披露充分，粒球不再被包装为不可替代精度来源。
7. KBS 比 Neural Networks / Pattern Recognition 更匹配当前技术新颖性与证据类型。

### 仍需处理的共同问题

1. **开发结果与确认结果仍混合。** beta 使用 Cora/CiteSeer/PubMed 的 test-informed pilot；24/25 把部分开发数据和冻结后数据合并。现有结果可直接单列参数冻结后的 CoraFull、Coauthor-CS、Amazon-Photo：相对 7 个正式 baseline 为 **14/15 个设置最高均值**（5/5、4/5、5/5）。这比继续只强调 24/25 更适合作为确认性 headline。
2. **单一 H 没有覆盖完整管线方差。** 下游 5 seeds 不能回答 GRACE seed 改变后分组、target 与增量是否稳定。
3. **三个核心数据集缺 checkpoint。** 归档 H 能复现下游，但不能从 clean checkout 重建预训练；匿名代码链接也仍是占位。
4. **完整训练成本缺失。** 当前效率表只比较构球、下游与 full-batch inference，未报告 GRACE 预训练时间、显存和 H/checkpoint 存储。

## 3. 两位审稿人的差异

- `academic-paper-review` 更关注贡献结构：train-only LCSS 与 teacher 蒸馏扩展的主次关系、现实噪声和算法新颖性。
- `peer-review` 更关注报告合规：pair/instance 生成细节、split/noise seed 规则、artifact URL、完整训练成本和确认/探索分层。
- 两者都认为真实噪声值得做，但在严格收窄 claim 后可以留作局限；它不是当前唯一的录用前置条件。
- 两者都不建议继续增加普通数据集或再堆 adapted baseline。

## 4. 下一轮最小修改包

### 零/低成本，投稿前必须做

1. 摘要、贡献和结论并列报告：开发筛选后的 24/25 与参数冻结后三数据集的 14/15；把后者作为确认性结果。
2. 附录补 pair/instance 噪声生成公式、类映射、随机种子和代码版本。
3. 补 GRACE 预训练时间、峰值显存、H/checkpoint 大小；明确 one-shot 与摊销成本。
4. 建立真实匿名仓库/归档链接，发布逐种子 CSV、配置、统计脚本、H manifest 与 SHA256。
5. 明确 LOO 平均略高但不替换默认方法的原因：其为审稿后机制诊断，未在全部噪声谱和所有数据集上预注册验证。

### 唯一优先的新实验

在 Cora、CiteSeer、CoraFull 上做 `3 GRACE seeds × 3 downstream seeds` 的分层方差。至少报告：

- H seed 间 test accuracy 方差；
- 球数与 assignment 稳定性；
- target Brier/NCR/CDR；
- LCSS 相对 CE、shuffled 和 JoCoR 的 paired delta。

如果该实验不做，则必须在标题/摘要/Highlights 的醒目位置使用 fixed-H conditional qualifier，而不只在实验设置和局限中说明。

### 可留给 rebuttal 或 future work

- 真实/弱标注图噪声；
- 噪声率误设敏感性；
- CPU/在线单节点 latency；
- unseen-node inductive 评测。

## 5. 投稿判断

当前 v14 不再需要推翻方法或重跑主表，但仍不建议立刻投稿。完成零/低成本修改后，如果补出最小多 GRACE-seed 结果，稿件的主要风险将从“实验可信度不足”转为“算法新颖性中等”；后者在 KBS 可由问题设定、系统实验和适用域分析支撑。
