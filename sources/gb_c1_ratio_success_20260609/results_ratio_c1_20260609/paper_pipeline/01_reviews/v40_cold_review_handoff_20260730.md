# V40 冷审交付说明

## 审阅对象

`paper_final_zh_v40_cold_review_ready_20260730.md`

SHA256：

`F6CDA518B6402ADB963FE065CCEF14BD481958099FF33CFEF4BF63423D30AC03`

该文件由 v39 原样冻结得到。v39 已整合最近正式实验，v40 不再修改数据、
公式、表格或 claim。

## 当前方法身份

- 正式主方法为 tie-safe、leave-one-out GB-LCSS。
- `tie_mode=distribution_only`。
- `leave_one_out=true`。
- 每个训练节点使用
  \(p_{j,-v},q_{j,-v},\alpha_{j,-v}\) 构造节点相关软目标。
- 推理阶段仅保留 X-only MLP。

## 最近实验的论文位置

1. 正式 LOO 五数据集全噪声谱：正文表 2、附录 A/B。
2. 留一粒球、kNN、k-means 与成本 Pareto：第 5.3 节。
3. 干净标签 \(\rho=0\) 控制：第 5.3 节。
4. NRGNN 与 NRGNN[\(X\Vert H\)] 鲁棒教师蒸馏：正文表 5、附录 D.4。

## 冷审时应区分的两条结论

- 基础主线：GB-LCSS 在五个正式数据集的 25 个噪声设置中均高于随机
  等规模分组，并在当前同资源比较范围内稳定改善 X-only MLP。
- 二级扩展：图教师蒸馏可继续提高 X-only student。球式共识教师不是
  统一最强教师，NRGNN 教师在 uniform 0.6 的三个数据集上更高。

鲁棒教师蒸馏不参与基础方法的 25-setting headline，也不改变基础
GB-LCSS 的方法身份。

## 冷审前自动核验

- v39/v40 稿件与证据审计：0 errors。
- 内部实验编号、smoke 表述、私有绝对路径、旧五臂表述：0 matches。
- v40 与 v39 SHA256 完全相同。
- v38 历史稿未被覆盖。

