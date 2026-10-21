# D领域检索日志

日期：2026-10-19（Asia/Shanghai）

## 纳入与核查

1. **H2GCN / Beyond Homophily in Graph Neural Networks**：检索NeurIPS官方论文与作者代码；纳入NeurIPS 2020正式PDF。深读§2、§3.1–3.2、§5.1–5.2及相关表/图，覆盖方法、假设、真实图协议、复杂度。
2. **GPR-GNN / Adaptive Universal Generalized PageRank Graph Neural Network**：检索ICLR 2021目录、OpenReview官方页面、作者arXiv和作者代码；确认正式发表ICLR 2021。OpenReview PDF访问时被challenge拦截，方法/结果改用作者arXiv v6（页面标注2021-10-26），并明确记录版本差别。读§3–5、Theorems 4.1–4.2、Appendix A.3/A.5/A.9。

## 查询

- `Beyond Homophily in Graph Neural Networks Current Limitations and Effective Designs H2GCN NeurIPS 2020 official PDF`
- `Adaptive Universal Generalized PageRank Graph Neural Network ICLR 2021 official PDF`
- `GPR-GNN ICLR 2021 OpenReview accepted paper`
- `H2GCN author code GemsLab`

## 访问失败与边界

- OpenReview正式PDF读取在搜索服务中重定向到challenge页；因此不把其正文阅读状态虚报为成功，GPR-GNN正文核对采用作者arXiv v6全文。ICLR官方目录仅用于确认会议版本。
- 未执行作者代码、未安装依赖、未运行训练。表格/公式定位以PDF页码与HTML行定位记载；不同版本的排版行号可能变化。
3. **ACM / Revisiting Heterophily For Graph Neural Networks**：检索并读NeurIPS正式PDF与官方摘要页/DOI，再核作者代码仓库。深读§2.2–3、§4.1–4.2与§6的消融、主比较和耗时；纳入NeurIPS 2022正式版本。
4. **A Critical Look at the Evaluation of GNNs under Heterophily**：查ICLR 2023官方Downloads目录确认正式发表；全文采用作者arXiv v2（页面标注2024-03-02）和官方代码/数据仓库。深读重复节点问题、adjusted homophily/label informativeness、新数据集与benchmark setup/results。OpenReview/ICLR具体文章跳转抓取失败，未用第三方总结替代方法证据。

新增查询：

- A Critical Look at Evaluation of GNNs under Heterophily ICLR 2023 author arxiv code
- site:iclr.cc/Downloads/2023 A Critical Look Evaluation GNNs Heterophily
- site:papers.neurips.cc Revisiting Heterophily For Graph Neural Networks ACM
- Revisiting Heterophily ACM official code Sitao Luan
- 2026-10-19（第5篇）：核对ICML 2024《Understanding Heterophily for GNNs》：PMLR正式元数据（41st ICML、PMLR 235、pp.50489–50529）与作者arXiv v2全文；读§2–5/定理1–4/Eq.4,7,8,9及Appendix A.2–A.4、B.4、C.1–C.3。重点记录HSBM的高斯特征、类别平均度同阶和密度假设；核对作者声明的模型外推限制。Sources: https://proceedings.mlr.press/v235/wang24u.html ; https://arxiv.org/html/2401.09125v2
- 2026-10-19（第6篇）：核对KDD 2025 CPF正式DOI（10.1145/3711896.3737075）、作者v2全文和公开代码；细读§2 RSA，§3.1–3.6/Eq.6–11，§4.1–4.3/Tables 1、3、5、6，Appendix F.1与G.1。专门逐式核Eq.7–8及Eq.10–11，确认粗化分区用于原图节点的滤波系数共享，传播和监督输出仍是原图节点；整理作者复杂度、超大图预传播和100 split×初始化协议。Sources: https://arxiv.org/html/2505.14033v2 ; https://doi.org/10.1145/3711896.3737075 ; https://github.com/vasile-paskardlgm/CPF

## 第7篇与跨文献核查：ACE（2026-10-19）

- 题录与正式版本：以[PMLR UAI 2026正式条目](https://proceedings.mlr.press/v337/li26j.html)核对正式会议、作者、卷页337:3738–3761，并用[UAI官方接收列表](https://auai.org/uai2026/accepted_papers)交叉确认；方法正文用作者[arXiv v1 HTML](https://arxiv.org/html/2607.21885v1)及PDF。发表时间按正式会议记录，不按预印本页面抓取时间判断。
- 正文深读范围：§2.1–2.2、Proposition 2.2、Table 1；§3.1/Eq.5；§3.2.1–3.2.3/Eq.6–11/Proposition 3.1；§4.1–4.5/Tables 3–4；Appendix D.1–D.4。记录了初始10%粗化损失、ASR机制、粗图目标与lift原节点辅助CE、同配/异配比较及每轮效率。
- 代码静态核查：[ACE作者仓库](https://github.com/vasile-paskardlgm/ACE)的README、backbones.py、coarse_train.py、refined_projector.py。确认代码每轮粗图前向，lift logits只在原train mask算辅助损失；训练后原图前向测试。README依赖外部提供标签和各split mask，仓库当前未见数据预处理脚本；粗标签生成时的split隔离以及论文全部split比例因此标作未核实。当前coarse_train.py列出的模型为gcn/gat/GPRGNN，未见LINKX训练入口。
- LINKX维数交叉核对：查看[Non-Homophily-Large-Scale作者实现](https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/models.py)，邻接输入MLP宽度依赖num_nodes；据此只得出直接参数复制时shape不匹配，未推断任何改写版LINKX无法迁移。
- 需要保留的公式边界：ACE预印本Proposition 2.2附近Θ标记疑似互换，Eq.6/11矩阵方向按协调人已核对意见存在歧义；卡片按损失实际执行路径概述，不将疑似有误矩阵写为复现公式。Prop.3.1只描述局部ASR行为，不升级成分类风险保证。
- 查询词：ACE Remedying Coarsening-Based GNN Training under Heterophily UAI 2026；site:proceedings.mlr.press/v337/li26j.html；github vasile-paskardlgm ACE coarse_train.py；LINKX models.py num_nodes adjacency MLP。
- 补充综合时引用协调人核查的[SIGN workshop 2020正文](https://grlplus.github.io/papers/77.pdf)§2–3/Table 1、3关于多通道预计算成本；该文未计入本轮七篇核心文献，也未作为本代理独立深读篇数。
- 未安装依赖、未运行论文代码、未训练模型。预处理总成本与原图测试推理成本均需未来按相同硬件/协议实测；作者每epoch开销数字不等于端到端加速。
