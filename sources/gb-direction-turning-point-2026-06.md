# GB Direction Turning Point（2026-06）

## 当前状态

截至 2026-07-01，SoftDist 分类线、噪声检测线、跨视角冲突线、椭球/GEAF 修复线、免K图聚类新主线，以及弱 MLP 双鲁棒重评估线均已完成 go/no-go 验证。MLP 目标硬度谱系已完成 seed0 与 5-seed 确认，用于修正 SoftDist-MLP 的主算法候选。

## 关键结论

1. **SoftDist 分类不是 SOTA 主线。**  
   控制同一 frozen GRACE `H` 后，SoftDist 的真实粒球软目标能稳定优于 `gcn_xh_ce` 与 shuffled control，但给强 baseline 同样的 `[X||H]` 后不再领先。因此只能写受控增量与适用边界，不能写强性能 claim。

2. **H-space 粒球变体基本被吸收。**  
   椭球、GMM、GEAF-style hard correction、检测集成、跨视角冲突等方向均显示：offline signal 可以存在，但很难稳定转成 downstream 或独立检测优势。主要原因是粒球在 `H` 空间中接近 H-kNN/局部聚类的粗粒化近似。

3. **图聚类新主线也未过最小 gate。**  
   Phase 41 用 `GRACE H + ordinary balls + ball-graph modularity agglomeration` 验证免K聚类。结果三数据集 G1/G2 全失败：免K精度明显低于 true-K kmeans，且没有 wrong-K 鲁棒优势。G4 也失败，说明球凝聚没有超过 `kmeans_H_autoK`。

4. **弱 MLP 噪声鲁棒重评估线过 G1。**  
   Phase MLP-1 按 MgCNL 式“固定弱 backbone，比抗噪策略”重新评估 SoftDist-MLP。协议：Cora/CiteSeer，percent split 0.2/0.1/0.7，train-only label noise，uniform {0.2,0.4,0.6}+pair0.4，seeds 0-4，所有对照均为同一 X-only 2-layer MLP。结果：SoftDist-MLP 在预注册 G1 口径下通过（两数据集 uniform 2/3 通过，pair0.4 双通过），可作为“不换域、弱下游噪声鲁棒”SCI 路线起点。边界：严格口径不全胜，Cora uniform0.4/0.6 对 small-loss/JoCoR 只有边际或不稳定优势，不能写成全面碾压所有抗噪方法。

5. **弱 MLP 双鲁棒扩展成篇，但 claim 必须收窄。**  
   Phase MLP-2 将 MLP-1 扩展为“标签噪声 + 结构噪声”的 graph-less 节点分类路线。标签噪声主表补齐 Coauthor-CS 与 instance0.4，结构噪声表补 edge_drop/isolated 推理扰动，消融补 w/o GRACE、hard majority、shuffled。完整性：标签 600/600、结构 315/315、消融 75/75，均无缺失。C1 标签鲁棒扩展通过 5/7 新设定；C2 结构鲁棒强成立（SoftDist-MLP 在所有结构扰动下 0.00pp drop，GCN/GIN 最高掉 23.26pp）；C3 部分成立（GRACE、真实球、球监督必要），但“soft distribution 优于 hard majority”被三数据集稳定负结果否定。论文可写普通 SCI 双鲁棒版本，但不能写 soft 必然优于 hard，也不能写全面优于所有 MLP 抗噪策略。

6. **MLP 目标硬度谱系显示：软目标不是最优，但固定 hard 也未定型。**  
   Phase MLP-3 在 Cora/CiteSeer/PubMed、uniform {0.2,0.4}、seed0 下比较 beta=0 soft distribution、beta=0.5 hybrid、beta=1 hard-majority target、beta=1 shuffled。结果：beta=0 没有在任何设置最好；beta=1 在 Cora 两个噪声率和 PubMed uniform0.4 最好；beta=0.5 在 CiteSeer 两个噪声率和 PubMed uniform0.2 最好。beta=1 相对 soft 在 4/6 设置为正，uniform0.4 上 3/3 为正；beta=1 real 相对 shuffled 在 6/6 为正（+8.12pp 到 +15.25pp）。结论是“目标硬化有价值，但最优硬度存在数据集/噪声依赖”，不能凭 seed0 直接把主算法改成固定 beta=1。

7. **MLP-4 5-seed 确认：默认主算法应转为 beta=0.5 半硬目标，beta=1 只作中噪声强变体。**  
   Phase MLP-4 在 Cora/CiteSeer/PubMed、uniform {0.2,0.4}、seeds 0-4 下完成 120/120 run。beta=0.5 是唯一在 6/6 设置中稳定高于旧 beta=0 soft target 的硬度（每格 ≥4/5 seeds 为正）。beta=1 是 4/6 设置的 mean-best，且 uniform0.4 上很强：Cora +4.95pp (5/5)、CiteSeer +2.95pp (5/5)、PubMed +4.13pp (4/5)；但 PubMed uniform0.2 明确为负：beta1-soft = -0.98pp，0/5 seeds 为正。因此固定 hard-majority 不适合作为跨噪声默认。beta=1 real > shuffled 在 6/6 设置中均为 5/5 正，说明硬化收益仍依赖真实球结构。

8. **MLP-5a 重验椭球/GMM 精细加权：未翻案，主算法保持 beta=0.5。**  
   Phase MLP-5a 在弱 MLP + beta=0.5 半硬目标下重验 ellipsoid/gmm_resp。实现上已确认正确组合顺序：先构造加权 soft distribution，再做 `(1-β)·weighted_dist + β·onehot(majority)`。Cora/CiteSeer uniform0.4 seed0 结果：ellipsoid 相对 beta=0.5 为 +1.16pp / -0.47pp，gmm_resp 为 -3.17pp / -0.82pp，均不满足“两数据集都正”的 5-seed 触发条件。ellipsoid 相对 shuffled 在两数据集均约 +12pp，说明真实球结构有信号，但没有稳定超过简单半硬目标。结论：ellipsoid/gmm_resp 进入消融/负结果，主方法不加入精细加权。

9. **MLP-5b 标签率敏感性 seed0：Cora/CiteSeer 有低标签率优势，PubMed 是边界。**  
   Phase MLP-5b 在 Cora/CiteSeer/PubMed、uniform0.4、train_percent {0.05,0.10,0.20,0.35,0.50,0.65}、seed0 下完成 90/90 run。方法为同一 X-only weak MLP：CE、small-loss、JoCoR、SoftDist β=0.5、β=0.5 shuffled。Cora/CiteSeer 上，β=0.5 相对 small-loss/JoCoR 的低标签率增益明显高于高标签率：Cora low-high 为 +3.58/+5.01pp，CiteSeer 为 +5.90/+6.07pp。PubMed 相反：β=0.5 在所有标签率均输 small-loss/JoCoR，但负差距随标签率升高缩小。β=0.5 相对 shuffled 在三数据集所有标签率均为正，说明真实球结构始终有信号。结论：不能写“全数据集统一低标签率优势”；可以写“Cora/CiteSeer 型 citation 数据的标签稀缺优势 + PubMed 边界”。若要上论文主图，建议只对 Cora/CiteSeer 的 train0.05/0.65 关键端点补 5-seed。

10. **MLP-5 主表：β=0.5 在 Cora/CiteSeer 强成立，PubMed 明确边界，真实球结构全表稳定必要。**  
    Phase MLP-5 在 Cora/CiteSeer/PubMed、noise={uniform0.2/0.4/0.6, pair0.4, instance0.4}、seeds 0-4 下完成主表，并补 Cora/CiteSeer train_percent={0.05,0.65} 低标签端点；总计 850/850 rows，missing=0。所有方法使用同一 X-only 2-layer weak MLP backbone，train-only label noise，显式 percent split。新增对照为 MLP-adapted FINE 与 MLP-adapted DivideMix-lite（非图像域原版复现）。主表结果：Cora 上 β=0.5 vs classic 25/25 setting 胜（平均 +6.83pp），vs FINE/DivideMix-lite 10/10 胜（+9.41pp），vs shuffled 5/5 胜（+10.62pp）；CiteSeer 上 vs classic 25/25 胜（+6.65pp），vs latest 10/10 胜（+8.24pp），vs shuffled 5/5 胜（+9.22pp）。PubMed 上 vs classic 13/25 胜，平均 -0.33pp，主要输 small-loss / co-teaching / JoCoR；vs latest 5/10 胜，DivideMix-lite 在 PubMed 很强；但 vs shuffled 5/5 胜（+8.18pp）。整体上 β=0.5 vs shuffled 15/15 胜（平均 +9.34pp），说明真实球结构是稳定必要因素。低标签 5-seed 端点确认 Cora/CiteSeer train=0.05 时 β=0.5 对 small-loss/JoCoR/FINE/shuffled 均为 5/5 胜；train=0.65 优势缩小，Cora vs JoCoR 仅 +0.18pp 且 2/5 seeds 正。论文 claim 应写为“Cora/CiteSeer 型 citation 数据 + weak MLP backbone 下跨噪声类型稳定强于同 backbone 抗噪策略”，同时明确 PubMed 边界：球结构有信号但不足以压过 strong sample-selection / DivideMix-lite。

## Phase 41 数字锚点

主 split `0.05`：

| Dataset | GB ACC | KMeans trueK ACC | ACC gap | GB NMI | KMeans trueK NMI | wrong-K ACC adv | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Cora | 0.4823 | 0.6317 | -0.1493 | 0.3404 | 0.5193 | -0.1208 | FAIL |
| CiteSeer | 0.5342 | 0.6395 | -0.1053 | 0.2942 | 0.3782 | +0.0240 | FAIL |
| PubMed | 0.5248 | 0.6766 | -0.1518 | 0.1918 | 0.3107 | +0.0096 | FAIL |

## 决策

不要继续开新的粒球形状、投票、椭球、GMM、GEAF、跨视角冲突或 H-space 图聚类变体。除非出现全新的、能先离线超过 `kmeans_H_autoK` 或 H-kNN 的机制证据，否则继续调参大概率重复“offline 有信号但主贡献不成立”的循环。

当前最稳路线有两种可选写法：
- 保守写法：回到 SoftDist 分类论文，定位为“图神经表示下粒球标签去噪的适用性边界”，贡献包括受控粒球增量、低/中标签率机制、shuffled/同H对照、以及多条负结果证明何时粒球会被强表示吸收。
- 新增 SCI 快速路线：把 SoftDist-MLP β=0.5 作为弱下游 graph-less 双鲁棒方法展开，核心 claim 是“同一 X-only MLP backbone 下，GRACE-H 粒球半硬监督在 Cora/CiteSeer 型 citation 数据的多数/全部标签噪声设置中优于常见 MLP 抗噪策略和 MLP-adapted FINE/DivideMix-lite，同时 MLP 推理天然免疫测试时结构扰动”。该路线必须保留 strict 边界：不写模型无关或 SOTA 分类性能；不写 soft target 必然优于 hard majority；PubMed 上 small-loss / JoCoR / DivideMix-lite 更强，应作为适用域边界写清。

MLP 主算法的下一步决策：不要继续维护“soft distribution 是默认最优”的假设。默认主算法采用 beta=0.5 半硬目标，beta=1 作为 aggressive / medium-noise variant 进入消融或附表；不加入 ellipsoid/gmm_resp 精细加权；暂不做自适应硬度，因为 5-seed 未满足跨数据集稳定触发条件。标签率结果可以作为论文机制图候选，但 claim 必须限定为 Cora/CiteSeer 型 citation 数据的 seed0 信号；PubMed 作为边界。下一步 MLP-5 应用 beta=0.5 重跑 MLP-2 受影响主表格，并保留 beta=1 在 uniform0.4 的强增益作为机制分析。
