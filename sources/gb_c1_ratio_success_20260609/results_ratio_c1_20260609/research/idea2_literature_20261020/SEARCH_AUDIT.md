# 检索范围与可追溯性

日期：2026-10-20。用途：保存本轮文献核查的范围、证据层级与缺口，供后续会话接续。

## 范围

按用户附件A–E组织：对应噪声MVC、部分对齐/原型匹配、粒球MVC、跨模态对应噪声、机会校正与无放回抽样。

重点2020–2026；正式会议/出版社/作者论文及作者代码为依据。PLCI、SMART保持预印本标识；polyClustR、Hoeffding、ARI/信息论校正作为必要历史项；MGCF-Net、DECL按用户点名纳入，不伪装为同一发表范围或同一任务。

## 覆盖的检索主题

以下是覆盖主题，不冒充搜索引擎完整逐条日志：

- noisy correspondence multi-view clustering；mismatched pairs。
- partially view-aligned / unaligned clustering；prototype alignment。
- cluster-level / set-level / group-level correspondence；many-to-many。
- granular-ball multi-view clustering；MGBCC、HOARD、GFSAF及对应噪声。
- chance-corrected cluster overlap；hypergeometric overlap；Fisher；ARI；PMI。
- 各篇精确标题、官方会议页、作者仓库与核心方法文件。
- 2025–2026的新作追踪发现PLCI、SMART、OPTION等；未核实材料不据以作结论。

## 证据分层

1. **正式页面**用于核实发表身份，不自动代表已读全文。
2. **作者全文/HTML/PDF相关段落**用于方法、公式和协议。
3. **作者代码**用于核查实际负号、配对生成、推理重匹配等实现细节。
4. **后续论文对照表**只能作为该后续论文协议下的实现结果，不冒充原论文成绩。
5. **本轮数学推导和枚举**独立标识，不归给已有论文，不当作训练实证。

逐篇URL及定位见[PAPER_CARDS.md](PAPER_CARDS.md)。资料共26卡，其中20篇方法论文、6项统计/历史资料或官方文档。全文受限者有明确状态；没有把搜索摘要包装成完整精读。

## 下载与环境限制

[fetch_sources.py](fetch_sources.py)尝试将15个公共PDF保存并提取文本。记录见[source_manifest.json](source_manifest.json)。所有下载失败，主要为连接中断/SSL错误；重试未解决。此为网络获取失败，不是用户拒绝或自动审批拒绝。

已通过浏览工具读取的作者HTML、部分PDF、slides及源码不受此本地批量下载结果等同限制。sources目录当前没有成功保存的PDF；引用链由卡片URL与段落/表号保存。

## 本轮唯一执行的计算

脚本：

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' 'research/idea2_literature_20261020/theory_check.py'
```

输出132个有限配置均通过精确检查。代码未训练网络、未读取实验标签、未修改旧实验，不构成新算法性能实验。

## 下次接续

1. 先读REPORT及THEORY_AND_PROTOCOL，保留“修改后有限诊断”的决定。
2. 实验前回查项目01/02文献与历史，以及现有原始日志，遵守项目协议。
3. 锁定数据版本、原版复现与重匹配评估；不要直接拼文献成绩。
4. 未读到的正文、未验证的代码与未知数字继续保持未知。
5. 不重启Idea 1，不把未通过的旧角色改名重新跑；本轮新问题是NC局部关系证据。
