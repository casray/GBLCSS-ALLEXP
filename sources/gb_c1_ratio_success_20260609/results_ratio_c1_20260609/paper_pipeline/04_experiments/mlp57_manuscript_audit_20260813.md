# MLP-57 稿件与证据一致性审计

审计对象：`paper_final_zh_v57_xteacher_loo_formalized_20260813.md`

## 代码与协议

- 相关 pytest：21 passed，19 subtests passed。
- X-only 球式教师新增记录：30/30，全部 `status=ok`，`teacher_use_xh=0`，教师与学生输入维度均等于原始 X 维度。
- 正式 LOO 验证协议新增记录：60/60，全部 `distribution_only`、`leave_one_out=1`，测试标签修改数为 0。
- 正式 LOO GRACE-seed 记录：27/27，全部 `distribution_only`、`leave_one_out=1`。

## 稿件与 CSV

- 表 4 的 6 个 X-only 球式教师单元格均与 `xteacher_aggregate.csv` 一致。
- 附录 C 的 12 个正式 LOO real/shuffled 单元格均与 `loo_validation_aggregate.csv` 一致。
- 附录 D.3 的跨 H seed 标准差与 `grace_seed_tiesafe_variance.csv` 一致。
- 附录 D.3 的 GRACE 表征三数据集结果与 `representation_grace_aggregate.csv` 一致。
- 稿件不再包含 `w/o LOO`、`[X||H]` 教师、非留一广度表或旧教师输入消融。

## 结构检查

- 公式编号 1--23 连续。
- 附录 A--H 连续。
- Markdown 表格列数一致。
- 参考文献 1--63 连续，正文无未定义引用或未引用条目。
- 图 1 文件存在。
- Elsevier 写作 linter：未发现已配置的 AI 写作模式。

## 旧审计器说明

历史脚本 `audit_submission_integrity_20260710.py` 假定稿件存在固定的“表 5”标记，而当前稿件只有四张正文表，因此在解析阶段终止。该失败属于旧解析器与当前稿件结构不兼容，不是实验或数字校验失败。其不完整输出已标记为不可使用，当前审计采用 MLP-57 汇总测试、CSV 定点核验和稿件结构检查。

## 结论

本轮修改通过代码、协议、数字和文稿结构检查。V56 保留未覆盖，V57 的 SHA256 已写入证据登记表。
