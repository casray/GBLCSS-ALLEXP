# 2026-10-06 实验交接

## 本轮目标与状态

按照 GB-LCSS 主实验的划分、训练标签噪声与验证选择协议，以 seed 0 筛查 TFR、ERASE 鲁棒图教师及其属性 MLP 蒸馏学生。40/40 组完成，最终失败 0 组。投稿包和历史实验代码未修改。

## 环境与入口

- GPU：AMD Radeon RX 7700 XT。
- Python：`D:/codex_temp/noisygl-rocm721/Scripts/python.exe`。
- PyTorch：2.9.1+rocm7.2.1，PyG 2.7.0。
- 入口：`run_screen.py --grid`。已成功的任务会跳过。
- 最新主实验源代码：工作区 `GB-LCSS_GitHub/src/gblcss`，指纹见 `environment.json`。
- 作者代码：`vendor/tfr` 与 `vendor/erase`，下载提交号见 `environment.json`。

## 协议与已完成核验

五数据集，uniform 0.4/0.6，seed 0，20/10/70 的有序 NoisyGL 划分，只污染训练标签，干净验证精度选择检查点，最后评估测试集。十个设定的划分和噪声均匹配历史指纹；每个设定的四种方法指纹一致。GB-LCSS 复用主实验粒球成员缓存并以默认参数重新在 GPU 构造目标。

四种教师接相同的两层属性 MLP，hidden 64、dropout 0.5、Adam 0.01、weight decay 0.0005、200 epochs、全节点软交叉熵、temperature 1。每个学生训练前统一重置 seed 0。因此教师复现历史精度，学生与论文原学生的随机数消费路径不同，不能混用旧学生数字。

`test_adapters.py` 七项通过，日志见 `tests.log`；`audit_outputs.py` 核验 40 组保存概率、指标、检查点及标签未污染，见 `output_audit.json`。原主实验源文件指纹未变。

## 已观察结果

- ERASE 学生高于本轮 GB-LCSS 学生 8/10，平均 +3.36 个百分点；uniform 0.6 下 5/5，平均 +5.81；uniform 0.4 下 3/5，平均 +0.91。
- ERASE 在 CiteSeer uniform 0.4 和 Coauthor-CS uniform 0.4 低于 GB-LCSS，分别约 -1.07 与 -0.19 个百分点。
- TFR 在 Cora、CiteSeer uniform 0.6 优于 GB-LCSS；CiteSeer 为 68.94% 对 52.88%，约 +16.07 个百分点。但在其他数据集上迁移 DBLP 预设表现较弱。
- GB-LCSS 学生高于 GCN+CE 学生 7/10；CoraFull 两种噪声及 Amazon-Photo uniform 0.6 为负结果。
- 完整教师、学生精度与配对差值分别见 `REPORT.md`、`results.csv`、`paired_deltas.csv`。

## 实现纠错与结论边界

第一批八组 TFR 错误预加图级自环，已移动至 `superseded_tfr_selfloops/` 并全部重跑。最终十组 TFR 都是 revision 2，遵循作者 DBLP 配置 `add_self_loop: false`；GCN 与解码器在内部添加自环。保留作者第一次验证后解码器持续 eval 的行为。

ERASE 原型、传播和等价多类压缩公式做了 GPU 适配；逻辑回归探针在 GPU 上采用与 sklearn C=1 多项式目标相同的 LBFGS 目标，合成等价测试通过。本轮输入为主实验原特征，而不是作者 Cora/CiteSeer 默认行归一化特征。

TFR 统一迁移作者 DBLP 预设；ERASE 使用作者三个引用图预设，其余两图固定迁移 Cora 预设。教师架构、容量、预算未完全统一。这是共同评估协议下的官方核心实现适配，不是原论文 SOTA 数值复现。单 seed 没有标准差或显著性证据，不得把差值当作稳定排名，不得将 TFR 负结果当作方法无效。

## 后续优先级

1. 在获得用户授权后，为 ERASE 和两条 GCN 对照补 seeds 1-4。
2. 同容量 GAT+CE 对照用于分辨 ERASE 的鲁棒机制和教师容量；输出置信度差异仍属待检验解释。
3. TFR 优先验证 CiteSeer 强噪声优势；迁移预设下的其他负结果先核查配置适用性。

没有未完成训练进程，也没有启动下一批实验。
