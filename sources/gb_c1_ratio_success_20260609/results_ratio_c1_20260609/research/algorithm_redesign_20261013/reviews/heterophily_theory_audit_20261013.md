# 异配图中多尺度加权投票方案：理论与历史审查

日期：2026-10-13。范围：审查“嵌套多尺度粒球并集 + semTopK 候选、无标签几何权重、加权噪声标签投票、有效样本量与尺度 JS 惩罚、置信混合软目标”的理论条件和项目内邻近证据。没有训练模型，没有改实验源码、日志或其他研究材料。

## 1. 结论

**加权投票的 Hoeffding 界在一组明确条件下成立；Brier 不等式也成立，而且可以写成更强的精确恒等式。两条公式都不证明该置信分数校准、能处理异配，也不保证学生性能提高。**

这个提案与 2026-10-12 的多尺度/增强稳定性筛查机制高度相邻：两者都用多尺度留一标签分布、跨尺度 JS 一致性、支持量和置信排序。但新提案把它们改成按无标签权重投票，再决定软目标中观测标签与邻域分布的混合比例；旧筛查没有测试这条精确公式，也没有训练最终分类器。因此不能说“完整新方法已经做过且失败”，也不能忽略旧筛查的负面信号，直接把新公式当成全新方向。

建议定位为**尚未通过门槛的低成本、冻结表示候选**。先完成公式与数据依赖定义，再做同候选、同拟标签的离线目标质量筛查，并与等权投票及匹配 kNN 比较；若没有稳健的目标质量增量，或只在弱对照上领先，就不应进入学生训练或主张粒球独立贡献。

## 2. 记号、候选集与“并集”退化

固定一个目标节点 (i)。其唯一投票节点集合记为 (U_i\subseteq\mathcal T\setminus\{i\})，其中 \(\mathcal T\) 是训练节点集。每个投票节点 (j\) 的干净标签为 \(y_j\)，观测标签为 \(\tilde y_j\)，非负几何权重为 \(w_{ij}\ge 0\)。令 \(W_i=\sum_{j\in U_i}w_{ij}>0\)，则

\[
r_i^*(c)=\frac{1}{W_i}\sum_{j\in U_i}w_{ij}\,\mathbf1[\tilde y_j=c],\qquad
p_i(c)=\frac{1}{W_i}\sum_{j\in U_i}w_{ij}\,\mathbf1[y_j=c].
\]

若同一根节点的粒球集合真的嵌套，例如 \(B^{(1)}(i)\subseteq\cdots\subseteq B^{(S)}(i)\)，那么

\[
\bigcup_s B^{(s)}(i)=B^{(S)}(i).
\]

因此，嵌套球的**并集本身只等于最大尺度球**。在这个定义下，多尺度额外信息只可能来自分别计算 \(r_s\) 后的尺度分歧统计，或来自 \(\mathrm{semTopK}(i)\) 增补的节点；不能把并集本身说成多尺度集成。若实际各尺度不是同根、同一层次的嵌套成员集，必须另行定义集合关系。

跨尺度投票前应先按训练节点 ID 去重。若节点 \(j\) 在多个尺度出现，最终一次投票可为它定义一个固定聚合权重，但不能将同一 \(\tilde y_j\) 复制成数张“独立票”。权重和候选节点也须排除目标自身，并且只含训练节点。

## 3. 加权对称翻转下的正确错误界

设候选集合和权重在给定无标签输入后固定，并与噪声翻转随机量独立。对每个唯一投票节点，假设独立对称翻转：

\[
\Pr(\tilde y_j=y_j)=1-\eta,\qquad
\Pr(\tilde y_j=c\mid y_j)=\frac{\eta}{C-1}\quad(c\ne y_j).
\]

这里 \(\eta\) 是**实际翻错概率**，不是“以概率 \(\eta\) 从全部 \(C\) 类均匀重采样”的参数。定义加权干净 plurality 类 \(c^*=\arg\max_c p_i(c)\)，并定义到最强竞争类的归一化加权 margin

\[
\Delta_i=\min_{c\ne c^*}\{p_i(c^*)-p_i(c)\}.
\]

对称转移把类分布向均匀分布收缩：

\[
\mathbb E[r_i^*(c)\mid y,w,U]
=\frac{\eta}{C-1}+\gamma p_i(c),\qquad
\gamma=1-\frac{C\eta}{C-1}.
\]

因此每个竞争类的期望加权票差为 \(W_i\gamma[p_i(c^*)-p_i(c)]\)。对每个投票节点定义 \(X_{jc}=w_{ij}(\mathbf1[\tilde y_j=c^*]-\mathbf1[\tilde y_j=c])\)；其区间长度是 \(2w_{ij}\)。独立 Hoeffding 不等式再对 \(C-1\) 个竞争类作 union bound，得到

\[
\Pr\{\arg\max_c r_i^*(c)\ne c^*\mid y,w,U\}
\le \min\!\left\{1,(C-1)\exp\!\left(-\frac{n_{\mathrm{eff},i}\gamma^2\Delta_i^2}{2}\right)\right\},
\qquad
n_{\mathrm{eff},i}=\frac{(\sum_jw_{ij})^2}{\sum_jw_{ij}^2}.
\]

原提案形式在这些定义下是正确的。但可用范围有硬边界：

- 必须有 \(\gamma>0\)，也就是 \(\eta<(C-1)/C\)，且目标加权 plurality margin \(\Delta_i>0\)。若 \(\gamma=0\)，噪声标签期望为均匀分布；若 \(\gamma<0\)，期望票会反转。平方 \(\gamma^2\) 不能隐藏这个符号条件。
- 该界说明恢复**加权干净邻域 plurality** 的概率，不自动说明恢复根节点 \(y_i\)。要把它用于节点去噪，还需 \(c^*=y_i\) 或者直接定义根节点类的 margin \(p_i(y_i)-\max_{c\ne y_i}p_i(c)>0\)。异配区域可能让该 margin 为零或负；高权重、低分歧也可能稳定地支持错误类别。
- \(U_i,w_{ij}\) 必须对投票噪声独立。若 semTopK、表示、几何或权重使用了同一批含噪训练标签训练/筛选，候选条件化后不再是独立的固定设计，以上 Hoeffding 推导不能直接沿用。冻结的无标签表示、标签无关的距离以及显式 leave-one-out 有助于满足条件；仅仅在最后一行代码里没有读取 \(\tilde y_i\) 不够。
- 若固定 \(X,A,Y_{\mathrm{true}}\)，并对每个训练节点独立抽取人工翻转变量，那么 GNN 感受野重叠、节点共享邻居或表示之间相关**本身不会破坏这些人工标签翻转变量的条件独立性**。它们会让干净标签组成、局部 margin 和不同目标的误差相关；这与噪声随机量是否独立是两回事。若表示/权重由含噪标签训练得到，则权重又会依赖翻转变量，属于另一条依赖路径，不能用这条观察豁免。
- 该界是逐目标节点的条件概率界，不表示不同目标节点的投票结果相互独立。目标之间共享邻居时，跨节点误差会相关。
- 对项目的 `uniform` 噪声，现有 `GB-LCSS_code/src/gblcss/data.py` 定义的转移矩阵是对角 \(1-\eta\)、非对角 \(\eta/(C-1)\)，并逐训练标签独立抽样，所以该 \(\gamma\) 与此协议吻合。`pair` 噪声和 instance-dependent 噪声不是这个对称转移；需改用其转移矩阵直接计算每个类的期望差，不能把同一个 \(\gamma\) 套过去。

更一般地，设噪声转移矩阵是 \(T\)，固定的加权干净 plurality 仍为 \(c^*\)。对每个竞争类定义未归一化期望差

\[
M_{ic}=\sum_jw_{ij}\,[T_{y_j,c^*}-T_{y_j,c}].
\]

若所有 \(M_{ic}>0\)，同样有

\[
\Pr(\mathrm{vote\ failure})\le
\min\!\left\{1,\sum_{c\ne c^*}\exp\!\left(-\frac{M_{ic}^2}{2\sum_jw_{ij}^2}\right)\right\}.
\]

这才是处理 pair/类不对称噪声的可复用版本；它要求已知或可信估计 \(T\)，而不是凭标签置信分数假设存在单一正的 \(\gamma\)。

## 4. \(n_{\mathrm{eff}}\) 与跨尺度 JS 的解释边界

\(n_{\mathrm{eff}}\) 是 Kish 权重有效样本量，也是此 Hoeffding 指数中 \((\sum w)^2/\sum w^2\) 的自然写法。若有 \(m\) 个等权且不同的投票节点，它等于 \(m\)；权重越集中，它越接近 1。名称宜写“权重有效票数”或“Kish 有效样本量”。在当前模型中，它表示独立翻转噪声下加权投票的浓缩尺度，**不等于** \(n_{\mathrm{eff}}\) 个独立的语义证据，也不证明球内干净类别标签是从目标类别独立同分布抽样。

跨尺度分布 \(r_s\) 若重复使用同一个节点标签，它们是相关估计，不是独立视图。平均 JS

\[
d_i=\operatorname{mean}_s\mathrm{JS}(r_{is},\bar r_i)
\]

可以描述票分布对尺度的敏感性，但不能降低 Hoeffding 界中的方差，也不能因为 \(d_i\) 小就推导更高的正确概率。相同或高度重叠票源会自然产生低 JS；祖先/子球还共享大量同一批噪声标签。把每尺度票数相加会重复计数。项目 10/12 的 screen 只去重完全相同的投票集合，已明确说明其他重叠集合仍相关；新算法应按唯一投票节点处理相关性，而不是用尺度数扩充 \(n_{\mathrm{eff}}\)。

此外须明确：JS 使用自然对数时范围是 \([0,\log 2]\)，使用 log₂ 时范围是 \([0,1]\)。\(\tau\) 与该选择绑定，不能跨定义复用同一数值。需将 `concentration(r*)` 明确规定为 \([0,1]\) 内的函数；例如直接用 \(\max_c r^*(c)\) 会让均匀分布也有 \(1/C\) 的非零置信度，而重标化浓度则有不同语义。也应预先定义零权重、少票、并列票、缺失尺度和 \(\tau\le0\) 的处理。否则 \(\alpha_i^*\) 不是可复现的规则。

即使这些细节固定，

\[
\alpha_i^*=\operatorname{concentration}(r_i^*)
\frac{n_{\mathrm{eff},i}}{n_{\mathrm{eff},i}+\kappa}
\exp(-d_i/\tau)
\]

仍是一个启发式缩放，不是 \(\Pr(q_i=y_i\mid\text{observed data})\) 的估计。它缺少从三因子到标签正确概率的校准论证。低分歧可能来自嵌套重叠；高浓度可能来自异配邻居一致地指向另一个类；高有效票数也只压低独立噪声方差，不压低局部类别偏差。需在冻结规则后检验其与目标正确率/Brier 风险的关系。

## 5. Brier 混合不等式：正确条件与精确恒等式

令 \(e_y\) 是目标干净类别的 one-hot 向量，\(z=e_{\tilde y_i}\) 是该目标自己的 noisy one-hot，\(q_i\) 是概率向量，\(a=\alpha_i\in[0,1]\)。按未除以 2 的多类 Brier 定义

\[
\mathrm{BS}(p,e_y)=\|p-e_y\|_2^2,
\]

有逐点的**精确平方范数恒等式**：

\[
\boxed{\mathrm{BS}((1-a)z+aq_i,e_y)
=(1-a)\mathrm{BS}(z,e_y)+a\mathrm{BS}(q_i,e_y)
-a(1-a)\|z-q_i\|_2^2.}
\]

因此，只要在条件信息 \(\mathcal F_i\) 下 \(q_i,a\) 固定，且 \(\mathcal F_i\) 不包含目标自身的噪声翻转，目标标签以概率 \(\eta_i\) 被替换为错误类，就有

\[
\mathbb E[\mathrm{BS}((1-a)e_{\tilde y_i}+aq_i,e_{y_i})\mid\mathcal F_i]
=2\eta_i(1-a)+a\,\mathrm{BS}(q_i,e_{y_i})
-a(1-a)\,\mathbb E[\|e_{\tilde y_i}-q_i\|_2^2\mid\mathcal F_i]
\]

\[
\le 2\eta_i(1-a)+a\,\mathrm{BS}(q_i,e_{y_i}).
\]

这直接验证了提案中的上界。它不需要错误类均匀对称，只需要 noisy one-hot 以 \(\eta_i\) 的概率与干净 one-hot 不同，并且其翻转与 \(a,q_i\) 条件独立。若采用除以 2 的 Brier 记分，右侧所有项也除以 2。

若 \(a,q_i\) 是随机量，但与目标自身的翻转独立，正确的无条件写法是

\[
\mathbb E\,\mathrm{BS}(t_i,e_{y_i})
\le 2\eta_i\,\mathbb E(1-a_i)
+\mathbb E[a_i\,\mathrm{BS}(q_i,e_{y_i})].
\]

不能一般地把随机量换成 \(2\eta_i(1-\mathbb E a_i)+\mathbb E[a_i]\,\mathbb E[\mathrm{BS}(q_i)]\)，因为 \(a_i\) 与 \(q_i\) 的质量可能相关。更不能只写常数 \(\alpha\) 而不说明这是给定单个节点、条件于其他投票标签后的量。

条件“没有直接读取自己的 noisy label”也要按**信息路径**解释。如果表示、语义检索器或权重曾用所有训练 noisy labels 拟合，那么目标 \(\tilde y_i\) 可能已通过模型参数影响 \(q_i,a_i\)，公式的条件独立假设不成立。稳妥选项是无标签冻结几何、对每个目标的 label leave-one-out，或按节点交叉拟合并证明目标噪声不进入其目标生成器。

这个上界不保证优于原 noisy label。基线 Brier 风险是 \(2\eta_i\)；右侧仅在 \(q_i\) 的风险足够低、或混合比例足够小的情况下有信息。若 \(q_i\) 是稳定但错误的邻居分布，取较大 \(a_i\) 仍可能伤害训练目标。JS 稳定性和 \(n_{\mathrm{eff}}\) 目前没有为 \(\mathrm{BS}(q_i,e_{y_i})\) 给出上界。

## 6. 全局验证选择与异配数据依赖

全局超参数是否独立于训练目标自身的翻转，取决于**完整的参数选择路径**。validation 标签干净且与训练标签的人工翻转独立，本身还不够：若先用全部 noisy training labels 训练学生，再用该学生的 validation 分数选择 \(\kappa,\tau,K\) 或候选公式，那么分数、入选参数和后续 \(a_i,q_i\) 都可能通过学生参数间接依赖 \(\tilde y_i\)。此时不能把已选参数简单当作与目标噪声独立的常数，上一节的条件 Brier 上界不再由原假设推出。

要保留该条件，需要所选参数对目标 \(i\) 的翻转随机量可条件独立，例如参数在该轮训练标签生成前固定，或对目标所属 fold 做完整 cross-fitting：学生训练、validation 打分、超参数选择和目标构造都不接触该 fold 的 noisy labels。对单个预注册、固定的规则，若 \(q_i,a_i\) 的构造只使用与 \(\tilde y_i\) 独立的标签噪声之外信息，也可条件化后应用恒等式。需核查数据与模型整条路径，而非只检查最后的投票函数是否显式读入自身标签。

但这不让同一 validation 上报告的最好分数成为无偏确认。多组 \(K,\kappa,\tau\)、候选集与打分公式共用一份 validation，会造成选择乐观偏差；应冻结配置后留出测试集或做嵌套/外层配对确认。若在看训练 clean label 后选权重或阈值，则会直接破坏可部署规则及理论解释。全局训练造成的参数耦合也不能被“每节点 leave-one-out 投票”自动消除。

“异配”也不是该界自身覆盖的条件。证明只控制 noisy vote 相对加权干净邻域分布的随机翻转误差；它不控制邻居真值相对根节点真值的结构偏差。应在标签冻结之后按真实的局部同配/异配分层作诊断，并报告 \(p_i(y_i)-\max_{c\ne y_i}p_i(c)\) 及其与 \(\alpha_i\)、目标 Brier 的关系。不能在运行前把真实标签用于建候选集或调权重。

## 7. 已有结果：重叠在哪里，尚未检验什么

| 已有来源 | 直接检验的机制与结果 | 对新方案的含义 | 未检验部分 |
|---|---|---|---|
| [`gb_signal_increment_diagnostic_20261012/实验结论.md`](../../../experiments/gb_signal_increment_diagnostic_20261012/实验结论.md) 与 `run_diagnostic.py` | 固定 GRACE 表示上，强教师/同伴/kNN 统计后加入固定粒球几何、支持量和 leave-one-out 投票特征。Base+kNN 宏 AUROC 0.955829，+GB 为 0.954895，差 −0.000935；Precision@20% 差 −0.2248 pp，干净误标率增加 0.0663 pp。无层级树，多尺度不在此轮。 | 削弱“在强邻域/教师信号上再加固定 GB 投票统计就自然有增量”的预期。 | 未测新权重函数、跨尺度 JS、目标混合 \(\alpha\)、学生或分类结果；目标还是污染检测探针，使用 clean train corruption flag 作响应。不能把它说成新软目标实验。 |
| [`gb_multiscale_stability_20261012/实验结论.md`](../../../experiments/gb_multiscale_stability_20261012/实验结论.md)、`PLAN.md`、`screen.py`、`decision.json` | 真递归 16/32/64 预算树、两个冻结增强视图、leave-one-out 观测票；跨尺度 JS 稳定性乘以 margin 与唯一训练票源支持。NoStability 与 Full 共享候选和拟标签。G1 预设需 +3 pp 且不增加 clean corruption；实际 precision +0.5452 pp、clean corruption +0.00757 pp，G1 未过。统一训练节点 5% 覆盖时 GB 79.1141%，低于 KM 81.5826% 与匹配票源 kNN 81.8627%；G2 未过。没有训练分类器。 | 这是机制上最近的负向证据：多尺度/增强 JS 稳定排序只带来小幅目标筛选信号，且 GB 未胜匹配 kNN。对新提案最强地削弱“稳定性×支持量就足以建立可靠 GB 置信度”的直觉。 | 未测加权唯一节点集合、上述 Hoeffding 所需的正 margin 分层、新 \(n_{\mathrm{eff}}\) 因子、软目标插值、校准误差或最终学生收益。它筛选固定拟标签，不是以 \(\alpha_i\) 连续混合 noisy label 与 \(q_i\)。不能称同一个算法已被否决。 |
| [`supervision_recovery_EF_20261012/实验结论.md`](../../../experiments/supervision_recovery_EF_20261012/实验结论.md) | 非粒球教师流程：E 将置信度筛出的 20% 可疑训练标签替换为教师 argmax，F 删除同一批监督。18/18 组错标都减少，但学生 E−A 宏精度 −0.0391 pp，E−F −0.0819±0.4565 pp；现实恢复门槛未通过。 | 说明“错标变少/修正标签变准”不等于学生分类改善。即使新目标 Brier 变好，也需单独验证训练后的节点分类。 | 没有 GB、加权邻域、尺度 JS 或本次软混合；不能作为该规则的直接反证。 |
| [`04_粒球所需空间与子图增强方案.md`](../04_粒球所需空间与子图增强方案.md) 与研究记忆 | 当前主建议仍是先检验强节点基座/局部上下文表示，再比较同空间 GB、KM、kNN；要求分开“空间变好→局部证据变好→最终分类变好”。 | 新提案若只替换后处理的标签权重，应作为低成本分支与该优先级并列验证，不能凭公式跳过粒球必要性、空间与分类转化对照。 | 04 是研究设计，不是新算法结果；其上下文学习、PiNGDA/LS-GCL 对照尚未实施。本审查不把它算成负结果。 |

还有一条结构边界：`gb_multiscale_stability` 的 JS 统计是由实际重叠的 scope 计算，报告明确指出不同但重叠的投票集相关；它只去重完全相同的 voter set，未把不同集合错误称为独立。新提案应继承该限制，并进一步按唯一训练节点聚合权重。C1 的历史负结果也只限制对应训练目标/表示/配置；现存项目记忆同时保留了 C1 相对其 no-ball 弱基线的正收益，不能笼统总结为“粒球都失败”。

## 8. 值不值得继续：有界门槛

目前值得保留的只是一个**公式上成立、尚未展示预测价值**的加权去噪假设。若要推进，先满足以下条件：

1. **先固定可复现定义。** 明确 \(U_i\)、semTopK 如何从 label-free 输入生成、每个唯一 voter 的聚合权重、\(r_s\) 的尺度定义、`concentration`、JS 对数底、\(\kappa,\tau\) 合法范围、少票/并列的回退。嵌套并集退化为最大球这一事实必须写入方法，而不是当作新聚合。
2. **做无训练的目标筛查。** 在冻结表示和固定协议下，先检验 weighted \(q_i\) 对 clean target 的 Brier/正确类概率是否优于相同候选集上的 unweighted \(q_i\)，并与匹配 kNN、容量/支持票相当的 KM 比较。权重定义和打分必须在读取诊断真值前冻结。异配 claim 需要独立报告真实局部 margin 分层；若目标类 margin 非正，不能借 Hoeffding 界宣称投票有恢复保证。
3. **分离三个贡献。** 相同候选、相同拟标签下比较 `alpha=0`、常数 alpha、仅 concentration、仅 Kish support、仅 scale-JS、完整 alpha。否则看不出增量来自投票分布、权重、有效支持量还是稳定性惩罚。报告 clean label 被误改风险、目标 Brier 和按节点/数据集配对结果。
4. **只在质量门通过后训练学生。** `supervision_recovery_EF` 已表明标签错误减少可能不转化为分类收益。正式学生比较需固定 `q,alpha` 生成协议，包含 noisy-label baseline、原 LCSS、等权多尺度、加权版本，以及同表示的 kNN/KM 对照；训练与测试必须沿用同一 split、噪声协议、验证选模和独立完整协议 seed。

若 weighted 版本不能在相同候选集和匹配 kNN 上提高目标质量，停止 GB 特有主张；若仅高分辨率 top 子集提高、全覆盖退化，则只能提出覆盖—风险取舍，不能说全节点去噪有效。若软目标质量改善但学生不涨，结论与 E/F 一致：监督目标改好不是充分收益。只有目标质量、最终分类收益和同空间 GB 必要性都过预先登记门槛，才值得升级为论文主线。

## 9. 证据等级

- **数学上可复用：** 对称独立翻转下，正加权干净 margin 的 plurality error 有上述 Hoeffding 上界；一般转移矩阵也可通过期望类差得到对应界。Brier 混合有精确恒等式与条件上界。
- **项目中已观测：** 固定统计的 GB 增量未过门；跨尺度/增强 JS 稳定性有小幅正排序信号，但未过预设门且 GB 低于 kNN；较准的标签纠正没有普遍转成学生收益。
- **合理解释而非结果：** 嵌套/增强 scope 共用 voter 可能让 JS 低估有效不确定性；邻域标签与根节点标签的偏差可能主导异配错误；当前教师/kNN 可能已吸收区域信号。这些尚未被因果隔离。
- **仍未知：** 新的 label-free 权重是否让 root-class margin 更正、JS 惩罚是否有增量、alpha 是否校准、是否优于同空间 kNN/KM，以及最终有图/无图节点分类是否获益。

### 本审查读取的直接材料

- [`99_研究记忆与接续.md`](../99_研究记忆与接续.md)
- [`01_项目与旧实验复盘.md`](../01_项目与旧实验复盘.md)
- [`04_粒球所需空间与子图增强方案.md`](../04_粒球所需空间与子图增强方案.md)
- [`experiment_history_43_multiscale_stability_20261012.md`](../../../docs/experiment_history_43_multiscale_stability_20261012.md)
- [`gb_multiscale_stability_20261012` sources](../../../experiments/gb_multiscale_stability_20261012/实验结论.md) (`PLAN.md`, `screen.py`, `raw_log.txt`, `decision.json` in the same folder)
- [`gb_signal_increment_diagnostic_20261012` sources](../../../experiments/gb_signal_increment_diagnostic_20261012/实验结论.md) (`PLAN.md`, `run_diagnostic.py`, `raw_log.txt`, `decision.json` in the same folder)
- [`supervision_recovery_EF_20261012` sources](../../../experiments/supervision_recovery_EF_20261012/实验结论.md) (`raw_log.txt`, `decision.json` in the same folder)
- [`data.py`](../../../GB-LCSS_code/src/gblcss/data.py), for the uniform/pair transition matrices and independent per-training-label noise sampling.

本文件是历史与理论审查，不是新实验、显著性检验、算法有效性确认或论文新颖性结论。
