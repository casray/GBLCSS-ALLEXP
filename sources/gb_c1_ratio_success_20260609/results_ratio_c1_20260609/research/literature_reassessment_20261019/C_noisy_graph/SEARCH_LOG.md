# C领域检索日志

检索日期：2026-10-19（Asia/Shanghai）。本日志记录本轮定向检索与逐篇核读；不是穷尽系统综述。

## 范围与策略

- 主题：noisy labels in graph node classification; robust graph condensation/coarsening under label corruption; label-noise-aware clustering/aggregation.
- 来源优先级：会议/期刊论文官方页面与正式PDF；作者/机构存档全文；作者官方代码仓库。预印本用来获得可读全文时，题录和最终版本另核。
- 已用查询：
  - `NoisyGL A Comprehensive Benchmark for Graph Neural Networks under Label Noise NeurIPS 2024 official arxiv 2406.04299`
  - `NRGNN Noise Resistant Graph Neural Networks label noise KDD 2021 arxiv 2106.04714`
  - `RTGNN Robust Training of Graph Neural Networks via Noise-robust Loss arxiv 2211.06614 venue`
  - `Divide and Denoise Empowering Simple Models for Robust Semi-supervised Node Classification against Label Noise KDD 2024 GD2`
  - `"Robust Training of Graph Neural Networks via Noise Governance" WSDM 2023 PDF`
  - `"RTGNN" "Noise Governance" github`
  - `"RTGNN" noise governance label noise theorem experiments 2211.06614 pdf`
  - `"RTGNN" "inter-view" "intra-view" noise governance graph neural networks`
- 检索入口：arXiv作者论文页/HTML、NeurIPS proceedings、ACM/DOI、作者高校存档、作者GitHub。常规web打开个别arXiv/ACM HTML失败时，改用作者PDF和arXiv HTML/正式元数据交叉核验；未执行代码。
- 截止边界：2026-10-19。只读、未训练、未安装依赖、未访问秘密或私有文件。

## 纳入（8篇独立核心，已读方法与实验）

1. **NoisyGL**，Wang et al., NeurIPS 2024 Datasets & Benchmarks。纳入原因：统一协议及噪声类型；全文正式PDF与作者代码README均核读。定位：§2-4、RQ2/3/5、App.A6；论文8数据集主实验，代码当前接口另有扩展。官方来源：`https://proceedings.neurips.cc/paper_files/paper/2024/hash/436ffa18e7e17be336fd884f8ebb5748-Abstract-Datasets_and_Benchmarks_Track.html`、`https://proceedings.neurips.cc/paper_files/paper/2024/file/436ffa18e7e17be336fd884f8ebb5748-Paper-Datasets_and_Benchmarks_Track.pdf`、`https://github.com/eaglelab-zju/NoisyGL`。
2. **NRGNN**，Dai et al., KDD 2021。纳入原因：稀疏标签+噪声、伪标签和补边机制，与粗化中有限标签利用高度相关。全文核读：§3定理3.1/3.2、§4式(8)-(14)、§5.1-5.5表1-3。正式DOI `10.1145/3447548.3467364`；全文：`https://arxiv.org/html/2106.04714`；KDD来源：`https://dl.acm.org/doi/10.1145/3447548.3467364`。
3. **RTGNN**，Qian et al., WSDM 2023。纳入原因：逐节点噪声分组、soft correction及置信策略。全文核读作者PDF §4.2-4.4式(7)-(16)、§5.1-5.4表2-4和附录；正式DOI由题录页核验。预印本/作者存档首页的DOI为占位符，不作为最终DOI。正式DOI `10.1145/3539597.3570369`；全文：`https://zhoujingbo.github.io/paper/2023robustWSDM.pdf`；作者arXiv：`https://arxiv.org/abs/2211.06614`。

## 纳入记录与阅读边界（续）

4. **PLGC**，Nandy et al., TMLR 2026。正式TMLR目录核验刊载与作者；全文读arXiv v1 §3、§4及§5.1.3–5.1.4。OpenReview全文入口遇验证码，故没有对正式PDF逐段对照；报告明确不推定预印本与正式版全部一致。作者代码README用于核验下游干净标签用法。重点协议：凝缩不依赖GT标签，但噪声实验下游预测头用少量干净标签，故不能等同本项目无额外干净训练标签。来源：https://jmlr.org/tmlr/papers/；https://arxiv.org/html/2601.10358v1；https://github.com/jayjaynandy/PLGC。
5. **D&D / DnD-Net**，Ding et al., KDD 2024。核正式DOI 10.1145/3637528.3671798、机构存档PDF和官方代码。读§4.1–4.4、式(4)–(12)、§5.1–5.3、表1/2及图2/4；机构PDF的普通web直接打开受限，但搜索索引包含全文片段，并用官方代码核实训练标签污染仅写入idx_train、验证保持干净。该论文代码名DnD-Net；GD²是NeurIPS 2025独立工作，本轮不计入核心。来源：https://doi.org/10.1145/3637528.3671798；https://research-repository.griffith.edu.au/bitstreams/6c7b28ba-e3a2-4d0e-a75d-0fadbe5ed0e3/download；https://github.com/kaize0409/DND-NET。
6. **RobGC**，Gao et al., TKDE 2025。核正式DOI、卷期页码；读arXiv HTML v2 §II–IV式(7)–(14)、主要表图、消融和复杂度以及官方仓库。论文脚注明确假设label clean来研究结构噪声，因此只作为结构鲁棒凝缩近邻证据，不支持带噪标签抗性。来源：https://doi.org/10.1109/TKDE.2025.3569629；https://arxiv.org/html/2406.13200；https://github.com/XYGaoG/RobGC。
7. **MRGC**，Luo et al., NeurIPS 2025 Main Conference。正式NeurIPS PDF、arXiv HTML和官方代码交叉核读§3–4、定理1/2、表1/2与附录C.1。方法包含训练标签随机翻转，属于压缩+标签扰动直接近邻证据，但其实验是明确预算的adversarial poisoning（常与特征、结构攻击联合），不能泛化成未知自然/实例相关噪声。分别记录联合攻击表和原图攻击单列标签扰动行，避免混合协议。来源：https://proceedings.neurips.cc/paper_files/paper/2025/file/ce1a7774429d5c06cc744ff4260a8e9f-Paper-Conference.pdf；https://arxiv.org/html/2510.26451；https://github.com/RingBDStack/MRGC。
8. **ICGNN**，Ju et al., ICML 2026, PMLR 306:54828–54846。核PMLR正式页面；方法与表格读arXiv v2 §2–4式(2)–(8)、定理3.1、表1–3、图2–7、附录B–E；官方train.py和utils.py核验噪声仅施加idx_train，默认五次运行。正式PMLR PDF与arXiv v2未逐段比对，报告明确保留此不确定性。定理分离界依赖联合结构/属性同质性δ，δ>0.5才无重叠。来源：https://proceedings.mlr.press/v306/ju26c.html；https://arxiv.org/html/2601.17469；https://github.com/wayc04/ICGNN。

## 题名、协议与分类边界

- D&D是KDD 2024论文，实现名DnD-Net；GD²是不同论文，不混用名称、实验或模块。
- PLGC的噪声实验在凝缩阶段不使用真实标签，但下游有少量干净标签微调；需标为协议不等价。
- RobGC是结构噪声凝缩，且标签默认干净；MRGC是有预算标签翻转的对抗投毒/联合攻击，具有直接标签扰动证据但不是一般实例依赖噪声保证。
- NoisyGL的RNCGLN 66.8x/2945.8x与PubMed OOM是单一方法及特定运行协议数字，不代表图标签鲁棒领域总体耗时。
- 本轮未阅读SGBGC正式版/扩展附录，不将其计入核心8篇；不影响已满足的最低7篇要求。
## 版本/失败说明

- RTGNN arXiv条目显示预印本2022提交；正式会议信息为WSDM 2023。作者PDF存在`XXXXXX`占位DOI，采用ACM/正式元数据DOI，避免把作者存档错误视为正式题录。
- `https://arxiv.org/html/2211.06614`在普通web打开返回内部错误；改用可检索的作者PDF `https://zhoujingbo.github.io/paper/2023robustWSDM.pdf`，全文10页，配合正式DOI核验。
- D&D机构存档PDF全文可被搜索索引读取，普通web直接打开受限；实验协议另由官方代码验证。未运行论文代码。
- PLGC和ICGNN正式出版目录页面可访问，正式全文未能做逐段版本diff；分析内容明确以arXiv v1/v2为全文依据。
- RobGC及MRGC全文来源可读；MRGC的均匀标签翻转须保留攻击预算和是否伴随其他模态攻击的协议说明。

