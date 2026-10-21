# GB-LCSS 论文投稿上下文压缩记录

更新时间：2026-08-18

## 1. 当前任务与目标

本项目正在将 GB-LCSS 中文终稿转换并润色为 Information Sciences 投稿英文稿。当前阶段不再扩展实验，重点是英文表达、术语统一、证据边界、数字一致性和投稿材料整理。

## 2. 当前主稿与文件谱系

当前推荐继续使用的英文主稿：

- `paper_final_en_v4_information_sciences_sentence_revised_20260818.md`
- SHA256：`3F85700B12BDA342AB5A48036031AD99C9535D82E2289F0DA6CB72BD80E51C33`

对应可读 HTML：

- `paper_final_en_v4_information_sciences_sentence_revised_20260818.html`
- SHA256：`420C496C194670A3F7EAB8DE1485511561147FD934648634FB3D23D07F6AD093`

主要上游文件：

- 中文底稿：`paper_final_zh_v74_flow_scope_revised_20260818.md`
- Claude Fable 5 翻译：`paper_final_en_v1_from_zh_v74_20260818.md`
- 先前审计稿：`paper_final_en_v3_information_sciences_audit_revised_20260818.md`
- Claude 对比报告：`paper_final_en_v4_claude_comparison_report_20260818.md`
- V3 审计报告：`paper_final_en_v3_information_sciences_audit_revision_report_20260818.md`

Claude V1 不能整体替换当前主稿。它有少量更简洁的句式可以借鉴，但存在术语漂移、协议身份弱化和 claim 偏强的问题。V4 已在 V3 的证据与术语基础上逐句吸收其可取表达。

## 3. 论文的贡献与层级

核心问题：带噪训练标签下的转导式 X-only 节点分类。训练阶段允许利用图结构，推理阶段分类器仅接收节点属性。

贡献层级固定如下：

1. 主要贡献：利用同一粒球内其他训练节点的标签构造留一局部共识软监督。
2. 方法实现：粒球提供固定预算、互斥且可追溯的局部区域接口。
3. 部署结果：训练期可使用图和冻结表征，部署时仅保留 X-only MLP。
4. 次要扩展：图教师蒸馏扩大监督覆盖，提高 X-only 学生的绝对精度。

不得让蒸馏扩展取代基础方法主线，也不得把粒球写成普遍优于 kNN 或 k-means 的分组算法。

## 4. 允许的结果表述

- 正式全谱实验覆盖五个数据集和五种噪声设置。
- 在匹配 MLP 骨干的七种抗噪基线中，GB-LCSS 在 25 个设置中的 24 个取得最高或并列最高平均精度。
- 在统一协议适配下，GB-LCSS 整体优于 PosteL、ALS 和 SALS；这些结果不得称为官方复现。
- 粒球相对最佳 kNN/k-means 替代在 4/5 个数据集上的差距小于 1 个百分点，但在 CoraFull 上低于 kNN 3.11 个百分点。
- 粒球的价值是固定预算、互斥、统计复用和可追溯，而非普遍精度优势。
- 粒球共识教师在表 4 的十个设置中均提高基础 GB-LCSS，并在教师路线比较的 8/10 个设置中取得最高均值；不得扩展为普遍最优教师。

禁止将结论外推到真实标签噪声、归纳式新节点、未知噪声率模型选择、所有图数据集或所有 graph-free 方法。

## 5. 必须保留的证据边界

- 训练阶段使用图；只有推理阶段是 X-only。
- 主实验使用合成标签噪声和干净验证标签。
- 每个数据集固定一份 GRACE 表征，五个下游种子主要反映下游随机性。
- 粒球不是不可替代的精度来源。
- PosteL、ALS、SALS、Graph-MLP 和 GSDN 是协议适配实现，不能写成官方复现。
- 附录中的广度实验和负结果必须保留其真实证据身份，不并入正式五数据集全谱成功率。
- 真实噪声、归纳式新节点和无干净验证标签的模型选择属于当前局限。

## 6. 术语表

全文固定使用：

- `GB-LCSS`
- `granular ball`
- `leave-one-out local consensus`
- `X-only inference`
- `graph-free inference`
- `transductive setting`
- `protocol adaptation` / `protocol-adapted`
- `granular-ball consensus teacher`
- `pretraining`
- `reimplementation`
- `multiclass`

不要轮换为：

- `attribute-only`
- `feature-only`
- `ball-consensus`
- `graph-free training`

`graph-free inference` 只表示推理不用图，不能暗示训练不用图。

## 7. V4 的主要语言改动

- 摘要结论改为机制性陈述，删除宣传式的 `simple and effective`。
- 摘要结果句拆分，保持结果范围不变。
- 引言贡献 3 拆分为较短句子。
- 相关工作明确 NOSMOG、SA-MLP 等方法的推理资源差异。
- 区分直接软标签构造与教师蒸馏，不混写两条路线。
- 方法节统一公式后的语法衔接，简化共识权重和蒸馏描述。
- 结果节简化 Brier 与 PosteL/ALS/SALS 比较的长句。
- 未修改公式、表格、算法、数字、引用和图路径。

## 8. 已完成的一致性核验

V3 到 V4 的回归检查均通过：

- 25 个展示公式完全一致。
- 317 行 Markdown 表格完全一致。
- 2 个算法代码块完全一致。
- 115 处正文引用完全一致。
- 63 条参考文献完全一致。
- 5 个图片路径完全一致。
- 58 个标题完全一致。
- 2752 个数字 token 的顺序与内容完全一致。

V4 HTML 检查：

- `lang="en"`
- 目录标题为 `Contents`
- 58 个标题
- 22 张表
- 5 张图
- 2 个代码块
- MathJax 已启用

Highlights 字符数分别为 78、77、79、80、74，均不超过 85 字符。

术语扫描未发现 `attribute-only`、`ball-consensus`、`graph-free training`、`state-of-the-art`、`significantly`、`robustly`、`notably` 或 `furthermore` 等问题表达。

当前仅有 14 条长段落提示，多为摘要、贡献、相关工作概述以及方法和实验设置中的高密度段落。它们是风格建议，不是错误，未进行机械拆分。

## 9. 已作出的关键决策

- 以 V4 为当前英文主稿，不整体采用 Claude V1。
- 不再为提高表面优势而调整 kNN、删除负结果或扩大 claim。
- 不把训练期使用图写成 graph-free training。
- 不把协议适配基线写成 official reproduction。
- 不因语言润色改变实验身份、方法定义或证据范围。
- 当前任务不包含新增实验。

## 10. 下一步

1. 作者通过 V4 HTML 逐节通读，重点检查专业术语和句意，而非再次改动实验叙事。
2. 对英文稿做最终投稿格式整理，包括作者信息、Highlights 文件、cover letter、匿名工件链接和期刊模板。
3. 最终检查只聚焦语言、数字引用和排版，不再开启新的算法或实验议题。
4. 任何后续改稿都从 V4 新建版本，不覆盖 V4。

## 11. 新窗口快速核验问题

开始后续工作前，应能直接回答：

1. 当前主稿是哪一份？
   - `paper_final_en_v4_information_sciences_sentence_revised_20260818.md`
2. 为什么不整体采用 Claude V1？
   - 因为存在术语漂移、协议身份弱化和过强 claim；其优点已被选择性吸收到 V4。
3. 哪些内容不得因语言修改而改变？
   - 公式、数字、表格、算法、引用、图路径、协议身份和 claim 边界。
4. 当前是否有待跑实验？
   - 没有。本阶段只做投稿级语言与材料整理。

