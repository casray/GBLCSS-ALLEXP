# v31 中文稿完成与一致性审计

日期：2026-07-29

目标稿件：`paper_final_zh_v31_tiesafe_nearestwork_20260728.md`

## 1. 完成结论

v31 已以 v30 的完整静态修订为底稿，合并当前 tie-safe v2 主网格、正式
最近工作共同协议网格和更新后的运行时证据。该版本没有启动审稿流程，也
没有生成英文稿。

正式最近工作结果改变了性能定位。GB-LCSS 在完整 25-setting 比较中相对
PosteL、ALS 和 SALS 的平均配对差分别为 +5.202、+7.215 和 +4.202 个
百分点。相对 GSDN 的平均配对差为 -0.698 个百分点。GB-LCSS 在 10/25 个
设置取得最高均值，GSDN 在 15/25 个设置取得最高均值。因此，v31 不再把
GB-LCSS 表述为结构辅助 X-only 最近工作中的整体最优方法，而是报告其对
三种软目标方法的稳定优势，以及与 GSDN 之间的数据集依赖差异。

## 2. 稿件改动

1. 摘要、Highlights、引言、贡献和结论加入正式 GSDN 校准结果。
2. 正文表 3 替换为 GB-LCSS、PosteL、GSDN、ALS 和 SALS 的 uniform 0.4
   共同协议表。
3. 历史 Graph-MLP、GLNN、FF-G2M、KRD 和 VQGraph 表移至附录 I 的资源
   分层诊断，不参与当前最近工作排名。
4. 第 4.3 节明确四个最近工作实现的身份。GSDN 是方程驱动的共同协议
   适配，不是官方或精确复现。
5. 第 4.4 节加入 25 个 immutable block、500 条原生结果、625 条五方法
   联合证据和 125 个通过审计的协议组。
6. 表 4 的 GB-LCSS 下游墙钟与构球时间更新为当前 tie-safe uniform 0.4、
   seed 0 结果。Cora、CiteSeer 和 CoraFull 的墙钟为 3.58、5.11 和
   17.43 秒，构球时间为 1.59、2.51 和 6.08 秒。
7. 全文移除中文分号。数字、公式、参考文献顺序和未受影响表格保持不变。

## 3. 证据与谱系

- `claim_evidence_ledger.md` 已扩展至 C-031。
  - SUPPORTED 27 条。
  - PARTIAL 3 条。
  - CONTRADICTED 1 条。
- C-030 约束正式最近工作性能结论。
- C-031 约束正式网格完整性和断点恢复结论。
- `manuscript_table_lineage.csv` 已重建：
  - Table 1 为 20 个当前 tie-safe 单元，全部 CLOSED。
  - Table 3 为 25 个正式五方法单元，全部 CLOSED。
  - Table 4 的 6 个 GB-LCSS 墙钟/构球单元已改指 tie-safe v2 raw。
- 最近工作目录包含：
  - 25/25 个 immutable raw block。
  - 500/500 条原生最近工作结果。
  - 625/625 条五方法联合结果。
  - 125/125 个通过协议审计的 dataset--noise--seed 组。

## 4. 自动检查

### 4.1 代码与编排器

- `py -3 -m unittest paper_pipeline.tests.test_nearest_work_full_grid`
  - 14 tests，OK。
- `py -3 -m py_compile paper_pipeline/scripts/run_nearest_work_full_grid.py`
  - PASS。
- `py -3 -m unittest tests.test_gsdn_faithful tests.test_standard_runner_baselines`
  - 31 tests，OK。
  - 测试过程出现现有的 sparse invariant、未关闭配置文件和
    `ruamel.yaml` loader 警告，没有测试失败。

### 4.2 稿件与数据

- 正文表 1 的 20 个单元与 tie-safe v2 aggregate 逐项一致。
- 正文表 3 的 25 个单元与 five-method aggregate 逐项一致。
- 表 4 的三个 GB-LCSS 墙钟和构球值与 tie-safe v2 seed-0 raw 一致。
- 11 个 Markdown 表格块的列数一致。
- 正文表标题编号为 0--6，连续且无重复。
- 参考文献定义为 1--66，连续。正文引用覆盖 66 条，无未定义引用。
- 中文分号数量为 0。
- `TODO`、`TBD`、`PENDING` 和 `PAUSED_BY_USER` 标记数量为 0。
- 未发现“最近工作结果尚未产生”“共同网格仍暂停”等旧状态语句。

## 5. 文件指纹

- v31 稿件 SHA-256：
  `29711CB0412F77D5FD1B86FCA5806C2572AF41907DEF719B033B7E73D0322B2D`
- claim ledger SHA-256：
  `CCD5E6E09225CFEE055343FEF86A5D5E2BA830CB89B629E130DE5E30157CDC26`
- manuscript table lineage SHA-256：
  `A2766BDD4673DDE189A5539E48EE3CB3DCA12AA76E702D42528F3DE196829F8A`

## 6. 尚未关闭的投稿材料

以下事项不属于 v31 中文稿内容或实验阻塞：

1. 实际匿名审稿仓库、固定 tag、软件许可和长期归档 DOI。
2. 作者姓名、单位、通讯地址、CRediT、基金号和利益冲突确认。
3. 独立 Highlights 投稿文件。
4. 英文翻译、Elsevier 排版和正式提交前作者通读。

本轮按要求不执行审稿 skill，不生成英文稿。
