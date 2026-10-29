# 带噪单位化三图扩展计划

用户授权：补齐未运行的图，CS/Photo已有结果复用，两个seed，节省token。

- 研究假设：CS的R2单位化带噪收益可在其他图或特定噪声条件重复。
- 粒球角色：固定原GraphECL SUPPORT32软目标（LOO/tie-safe/beta=.5/min_votes=2）。
- 动机：C批分离初始化后确认CS增益，Photo分化；按新的带噪优先目标扩展覆盖。
- 代码改动：独立contract/test_contract、run、analyze、launch；只读复用C批fit.py和原数据桥接，旧代码不改。
- 配置改动：Cora/CiteSeer/CoraFull×2seed×3噪声×CE/SUPPORT=36单位化新头；原始36头引用；CS/Photo的raw旧初值/unit旧初值48头引用，最终120行五图表。
- 固定协议：原split/noisy标签/目标，旧AlignedMLP分类头初值，冻结64维R2，Adam .01/wd .0005/200轮/首次最高验证计数；0次SSL/R2重训，0次clean新训练，无kNN/KMeans。
- 预期机制：控制特征尺度影响噪声梯度/有效正则；按Uniform与Pair分解。
- 成功标准：单图三噪声验证均值≥+.5pp、两个seed正为继续信号；报告五图等权均值、覆盖、最差条件。clean退出否决门，保留历史记录。
- 失败标准：来源/目标/复现/选模核验异常为工程失败；研究门失败保留负结果。旧GRACE/ProGCL失败seed不处理。
- 工程门：六套新图R2提取与初值核验；每图每seed U.4 CE/SUPPORT共12次原始头复现，200轮验证计数及最佳epoch一致，概率误差≤2e-5。
- 评分：所有新拟合锁定后才评分test；120行独立重算Accuracy/Macro-F1，保存两seed样本SD和配对差。
- 实验结果：36/36新头、12/12复现、120评分完整；Cora/CS的SUPPORT过带噪门，其余三图未过。详见REPORT_zh.md。
- 最终决策：单位化保留分图机制对照，暂缓统一替换；本轮扩展结束，下一阶段为匹配读出的弱粒球关系项。未自动启动新SSL。

- [x] 矩阵/新带噪门测试RED→GREEN。
- [x] 核验旧来源并锁定六套特征/初值及84个引用头。
- [x] 12次原始头复现通过。
- [x] 36个单位化头完成并锁定。
- [x] 120项评分/模型/目标/选模审计、五图均值SD/门/逐噪声差。
- [x] 独立复核与详细报告、历史和接续记录归档。
