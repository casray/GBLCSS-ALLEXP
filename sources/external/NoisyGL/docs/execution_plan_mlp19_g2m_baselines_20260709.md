# 执行计划：MLP-19 graph-free GNN-to-MLP baseline 扩展

日期：2026-07-09  
定位：Table B / graph-free family 扩展表补充，不进入主表 A，不改变 LCSS 主算法。

## 1. 目标

把三篇 GNN-to-MLP / graph-free inference 方法纳入同一 noisy-label 协议下的定位表：

- `ffg2m_adapted`：FF-G2M adapted。GCN teacher + X-only MLP student；在普通 teacher distillation 之外加入 low/high-frequency edge-level distillation。
- `krd_adapted`：KRD adapted。GCN teacher + X-only MLP student；用 teacher 在特征扰动下的 entropy 稳定性估计可靠 knowledge nodes，并按可靠性加权 distillation。
- `vqgraph_adapted`：VQGraph adapted。先检查官方仓库，确认其包含 teacher/tokenizer/student 训练路径；本轮不搬官方 DGL/CPF split 管线，而是在 NoisyGL 同协议下实现其核心思想：GCN teacher 学结构 tokenizer/codebook 和 soft token assignments，X-only MLP student 同时蒸馏 teacher label 与 token assignment。写作时必须标注为 adapted core implementation，不声称 official reproduction。

## 2. 协议

- 数据集：Cora / CiteSeer / PubMed。
- 噪声：`noise_scope=train`，`noise_type=uniform`，`noise_rate=0.4`。
- split：`split_type=percent`，`train_percent=0.2`，`val_percent=0.1`，`test_percent=0.7`。
- seeds：0-4。
- device：ROCm `cuda:0`。
- 口径：adapted to our protocol，不声称官方复现。

## 3. 实现边界

- `ffg2m_adapted` 和 `krd_adapted` 都使用同一个 NoisyGL split/noise pipeline，teacher 只用 train noisy labels 训练，val noisy labels 早停；clean test labels 只评估。
- student 推理只调用 `MLP(X)`，不读 `adj`。
- KRD reliability 只用 teacher logits 在扰动前后的 entropy 变化，不读 clean label。
- FF-G2M frequency loss 只用图边和 teacher/student 输出，不读标签。
- VQGraph official repo 已通过临时克隆检查，存在 tokenizer/codebook 训练路径；因此本轮接入 adapted core version。若未来要做 official reproduction，需要另起独立复现实验，不能把本 adapted 数字写成官方复现。

## 4. 运行命令

Smoke：

```powershell
cd C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" scripts/run_standard_downstream.py `
  --datasets cora,citeseer,pubmed `
  --methods ffg2m_adapted,krd_adapted,vqgraph_adapted `
  --seeds 0 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --noise_scope train `
  --device cuda:0 `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --output runs/mlp19_g2m_baselines_20260709/smoke_seed0.csv `
  --stop_on_error
```

Full：

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" scripts/run_standard_downstream.py `
  --datasets cora,citeseer,pubmed `
  --methods ffg2m_adapted,krd_adapted,vqgraph_adapted `
  --seeds 0,1,2,3,4 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --noise_scope train `
  --device cuda:0 `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --output runs/mlp19_g2m_baselines_20260709/uniform0p4_seeds_0_1_2_3_4.csv
```

Summarize：

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" scripts/summarize_mlp19_g2m_baselines_20260709.py `
  --run_dir runs/mlp19_g2m_baselines_20260709 `
  --graphfree_table runs/mlp14_graphfree_family_20260703/mlp14_combined_raw.csv
```

## 5. 产出

- `runs/mlp19_g2m_baselines_20260709/mlp19_raw.csv`
- `runs/mlp19_g2m_baselines_20260709/mlp19_combined_raw.csv`
- `runs/mlp19_g2m_baselines_20260709/mlp19_aggregate.csv`
- `runs/mlp19_g2m_baselines_20260709/mlp19_paired_deltas.csv`
- `runs/mlp19_g2m_baselines_20260709/mlp19_report.md`

## 6. 判读纪律

- 三方法只进 Table B，不进主表 A。
- `ffg2m_adapted` / `krd_adapted` 若赢现有 adapted graph-free 方法，写作时必须注明同协议 adapted 实现。
- VQGraph 结果进入 Table B，但必须和 FF-G2M/KRD 一样标注 adapted to our noisy-label protocol；小于 1 个 std 的差距写 comparable，不写稳定胜出。
- 小于 1 个 std 的差距写 comparable，不写稳定胜出。
