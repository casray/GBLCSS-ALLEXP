# 1. Current Overall Judgment

日期2026-10-14。本文件整理1个完整方向：新增强与上下文空间→冻结GB-LCSS，当前defer。

- Confirmed：24预训练/684分类、324目标完成；严格审核和独立公式重算通过。
- Confirmed：B64上下文R相对原GRACE带噪平均MLP−6.332pp、GCN−1.888pp。
- Inconclusive：GCN对PiNGDA-H+.516pp，通过基础数值筛查，但对固定mean仅+.154pp，两seed不能确立独立贡献。
- Likely：增强与最终原图表示分布差异、C低秩是候选原因，未做因果干预。
- Risk：支持率/票数增加不保证正确类别多数；有效秩、VICReg loss和Brier均不能单独替代分类。
- Next priority：若恢复context，先保留原GRACE增强做单因素检验，不直接扫多个参数。

# 2. Direction Index

| ID | 方向 | GB角色 | 版本 | 数据 | 正信号 | 主要失败 | 决策 | 论文价值 |
|---|---|---|---|---|---|---|---|---|
| CTX-01 | 新增强与上下文构球空间 | 冻结局部软监督区域 | GRACE/PiNGDA/mean/context/Hreg | Cora、CiteSeer、CoraFull | GCN对弱基座+.516pp | 对原GRACE总体下降、投票语义变差 | defer | negative result、diagnostic |

# 3. Direction-Level Detailed Records

## CTX-01

### 3.1 Direction Summary

目标：改善粒球需要的空间。假设：新增强和独立C改善标签证据。baseline为同协议重跑的GRACE与CE；当前defer。[PLAN](PLAN.md)、run_experiment.py/core.py/pingda.py固定方案；[综合结论](综合结论与接续.md)及[完整表](实验结论.md)给出事实、解释与决策；raw log为runs/raw.log，逐seed/配对表在runs/results.csv、summary.csv、paired_deltas.csv。

### 3.2 Why This Direction Was Tried

用户要求从需要的空间反推，比较GRACE后续增强及局部上下文任务。04方案避免同球硬正例、强中心拉近和球控制增强闭环。预期claim为可靠局部估计，当前未建立。

### 3.3 Versions / Attempts Included

四个SSL arm产生六空间：旧GRACE-H、新PiNGDA-H、固定邻域均值R、context-H/R、同参数节点VICReg-H。独立完整seed0/1、三图、三噪声、MLP/GCN、B64/128/256。主预算64，不逐图选最优。统一200轮，不与旧CoraFull1000轮或其他分支单seed混比。

### 3.4 Role of Granular Ball

H/R离线标签无关分球，训练标签LOO投票生成软目标；R无效节点回退同编码器完整H目标。球不控制SSL loss、增强、图结构、采样或evaluation。独立KM/KNN本轮按用户要求排除，GB内部二分保留。

### 3.5 Result Summary

B64带噪MLP相对GRACE12/12下降，平均−6.332pp；GCN−1.888pp，CoraFull Uniform局部+1.62pp。对PiNGDA-H为+.055/+.516pp；对mean为−.526/+.154pp。GCN CoraFull Uniform上下文61.51±.56、CE61.52±.90，clean66.89±.30、CE67.61±.03。全mean/std在summary.csv，aggregate_comparisons.json保留配对平均。大图上下文峰值7.56GiB、GRACE2.89GiB，耗时波动大，不作速度主张。

### 3.6 What Is Directly Supported

新空间B64多数类别正确覆盖低：GRACE/context为Cora73.11/43.81%、CiteSeer66.24/30.68%、CoraFull25.37/5.52%，同时票数更多。六个带噪条件Brier均劣于GRACE。128维C有效秩1.10–2.09。证据锚点runs/audit/exact_evidence.csv、target_formula_report.json、pretrain complete.json。

### 3.7 Reasonable Interpretation

增强与原图空间偏移可能损伤构球；VICReg在增强输入上保证的方差不能自动迁移到原图；均值过平滑/结构统计主导也是候选。来自view_diagnostics.csv、generator_diagnostics.csv，尚无因果干预，不写成确定原因。

### 3.8 What Cannot Be Concluded

不是PiNGDA完整官方复现；不能否定全部子图、VICReg理论或粒球。缺KM/KNN不能主张不可替代性。无SSL与noisy CE联训，不能拿PLReMix分类/对比冲突解释此次失败。两seed不支持显著性或普遍鲁棒性。

### 3.9 Failure Analysis

已定位的失败环节是新空间→局部证据退化；主因Unknown。CoraFull GCN目标更差而相对GRACE分类更好，说明分类器鲁棒性/修改强度还有解释空间。最小隔离检验为原GRACE增强下只加相同context。

### 3.10 Lessons Learned

基座要在本协议先实测；更多票≠正确多数；有效秩≠分类信息；原图读出需与训练视图对齐。以完整CSV和精确投票/视图诊断为证据，未来检查空间、目标、分类三段链路。

### 3.11 What Is Still Useful

精确分块全负样本loss、公式审计、R回退、Hreg对照和全部负结果可以复用；GCN对弱基座小幅正信号仅作探索，不作为主算法。

### 3.12 Decision

defer当前完整组合，不直接扩大seed或扫λ/温度/扰动。若恢复，需以原GRACE及固定mean为强对照先证明跨图收益，再增加seed及普通分组比较。该后续未实施。

# 4. Cross-Direction Failed Pattern Ledger

confirmed：支持票多但错误类别关系仍主导，继承旧“紧凑≠语义好”边界。此次目标本身退化，不能全归为旧目标改善但分类不涨的转化失败。

# 5. Cross-Direction Positive Signal Ledger

possible：有图GCN路线保留；架构差异不算context创新。GCN对PiNGDA两个seed条件平均均正，需强基座/mean/更多seed验证，当前仅diagnostic。

# 6. Anti-Repetition Memory

| 暂不重试 | 证据 | 恢复条件 |
|---|---|---|
| 当前PiNGDA+context直接扩大seed | 对原GRACE总体退化 | 隔离基座与context并改善证据 |
| 只加Bmax救小图 | 多数新空间仅4–18球停止 | 明确空间/停止原因，事前匹配容量支持 |
| 只看rank、variance或票数 | 与错误多数及分类退化同现 | 原图空间/LOO证据/分类同时验证 |

失败形式不等于上下文思想被证伪。主线最低要求：可信基线、多完整seed、多图、必要消融、无泄漏、普通分组对照。

# 7. Direction-Level Lessons for Future Experiments

先建立基座，再证明context增量；保留clean锚点；两噪声共享SSL不算额外独立seed；标签诊断冻结后进行；不通过test选空间预算。

# 8. Current Multi-Direction Summary

当前方向defer；异配/自适应生成器分支见项目根历史及各自目录，用各自协议判断。

# 9. Recommended Next Directions

1. 优先：原GRACE增强单因素context检验，保持LCSS/分类器/预算，保留mean/Hreg，证据与分类均需改善。
2. 独立基座审计：作者数据预处理/归一化/骨干与预算，先冻结差异再实验，不从当前test追调。
3. 真子图基座：单独审计LS-GCL/CL-GCL入口协议，普通邻域读出不代替独立子图编码；另行预注册，目前未执行。

# 10. Paper Claim Implications

上下文空间提高局部估计not ready；GCN对弱基座小幅改善weak；票数与正确多数脱钩可作本轮诊断。宽泛首次、普遍鲁棒性和球必要性都缺证据。

# 11. TODO List

主队列、日志/config/mean/std/targets/logits审计完整，无待续训练。缺真实子图、打乱context/冻结θ、同空间KM/KNN、因果干预与更多seed。后者仅在受控候选成功后推进，当前不自动续跑。
