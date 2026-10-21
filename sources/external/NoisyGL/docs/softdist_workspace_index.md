# SoftDist 工作区索引

更新时间：2026-06-18。

本文档只描述本地 NoisyGL 研究扩展。上游 NoisyGL 的安装和基础用法仍以根目录
`README.md` 为准。完整研究结论和 claim 边界以项目根目录的
`02_experiment_history_and_lessons 11.md`、`02_experiment_history_and_lessons 12.md`
为准。

## 当前状态

- 冻结主方法：`SoftDist [X||H]`。
- 核心受控结论：在相同 frozen-GRACE `H` 下，真实球软目标需要同时对比
  `GCN+CE [X||H]` 和 size-matched shuffled balls。
- 主要协议：percent split `0.2/0.1/0.7`，默认噪声 `uniform 0.4`，正式结果使用
  seeds `0-4`；单 seed 仅用于 smoke test。
- 2026-06-18 最终验证：ROCm Python 下 `102 passed`、`14 subtests passed`。
  warning 来自 PyG 2.7 的 deprecated distributed import 和既有 YAML loader。
- MLP 诊断已完成：`MLP(X) + SoftDist(H-ball)` 在 Cora/CiteSeer、uniform 0.4、
  seeds 0-4 下均以 5/5 seeds 胜过 matched CE 与 shuffled control。这里的分类器
  不接收 H，但 H 仍用于建球，不能表述为端到端无 H。
- MLP 噪声谱已完成：Cora/CiteSeer 在 6/6 个 noisy regimes 均达到预注册稳定
  正增量，clean safety 也通过；结论仍限于“可迁移到 X-only MLP”，不是模型无关。
- MLP 效率已完成：SoftDist-MLP 的分类器训练与显存低于 GCN/NRGNN，但建球和软
  目标构建使下游总时间不低于 plain GCN；GRACE 预训练成本继续单列。
- 完全无 H 的固定 X-ball 诊断已停止：Cora/CiteSeer 都只生成 1 个球，real 与
  shuffled 的 purity gap 均为 0，离线门槛失败，因此未运行 seed-0 下游。

## 核心实现

| 位置 | 职责 |
|---|---|
| `predictor/FrozenGraceGB_GCN_Predictor.py` | SoftDist、shuffled、`[X||H]`、GAT/APPNP 主机与诊断记录 |
| `predictor/module/GNNs.py` | GCN/GAT/APPNP/MLP 下游实现；MLP 忽略邻接矩阵 |
| `utils/frozen_grace_targets.py` | frozen embedding 加载、内存特征/X 建球接口、SoftDist target、alpha 和诊断量 |
| `utils/ordinary_gb.py` | 原始粒球构建与划分逻辑 |
| `utils/soft_targets.py` | 软目标辅助函数 |
| `predictor/NRGNN_Predictor.py` | NRGNN、NRGNN `[X||H]` 和已判负 BallAux 对照 |
| `predictor/GBSelect_Predictor.py` / `utils/gbselect.py` | 双视角 clean-sample selection 诊断分支 |
| `single_exp.py` | 单次实验方法注册与 CLI 入口 |
| `scripts/run_ssl_swap_downstream.py` | 同协议替换 H、批量运行 SoftDist/公平性对照并写 CSV |
| `scripts/run_standard_downstream.py` | 标准 NoisyGL predictor 的显式 split/seed CSV runner |
| `scripts/diagnose_featuregb_mlp.py` | 固定 X-space real/shuffled purity、NCR/CDR、alpha 与离线门槛 |

## 配置族

- 主线：`config/frozengrace_gb_gcn_softdist_xh/`。
- 主线随机对照：`config/frozengrace_gb_gcn_softdist_xh_shuffled/`。
- H 公平对照：`config/gcn_xh_ce/`、`config/nrgnn_xh/`。
- 下游主机消融：`config/frozengrace_gb_gcn_softdist_xh_gat/`、
  `config/frozengrace_gb_gcn_softdist_xh_appnp/` 及对应 CE 配置。
- 无 H 输入/无图卷积诊断：`config/mlp_ce/`、
  `config/frozengrace_gb_mlp_softdist/`、
  `config/frozengrace_gb_mlp_softdist_shuffled/`。
- 完全无 H 的固定 X-space 负结果：`config/featuregb_mlp_softdist/`、
  `config/featuregb_mlp_softdist_shuffled/`。这些配置用于复现门槛失败，不是推荐方法。
- 历史机制负结果：`adaptive`、`dynalpha`、`bc`、`expand`、`nrgnn_ballaux`。
  这些目录用于复现负结果，不是推荐主方法。
- 大图诊断分支：`config/gbselect*/`。

## 实验与证据索引

| 主题 | 计划 | 主要产物目录 |
|---|---|---|
| SoftDist 与 `[X||H]` 归因 | `docs/execution_plan_xh_expansion_20260610.md` | `runs/softdist_5seed_rocm_20260610/`、`runs/xh_5seed_rocm_20260611/` |
| 噪声率/类型边界 | 同上 Phase 6；`docs/execution_plan_noise_robustness_completion_20260618.md` | `runs/noise_grid_rocm_20260611/`、`runs/noise_breadth_20260618/` |
| SSL 与 backbone 泛化 | `docs/execution_plan_ssl_upgrade_20260612.md` | `runs/ssl_probe_20260612/`、`runs/ssl_swap_seed0_20260612/`、`runs/phase11_backbone_20260612/` |
| 公平性和外部方法 | `docs/execution_plan_2025_comparison_20260615.md` | `runs/phase24_nrgnn_xh_20260616/`、`runs/tfr_comparison_20260616/` |
| 大图球构造/吸收 | `docs/execution_plan_bigball_scaling_20260616.md` | `runs/ball_scaling_diag_20260616/`、`runs/bigball_downstream_20260616/` |
| GBSelect 负结果 | `docs/execution_plan_gbselect_20260616.md` | `runs/gbselect_20260616/` |
| 数据集广度和适用域 | `docs/execution_plan_applicability_grid_20260617.md` | `runs/applicability_grid_20260617/`、`runs/evidence_matrix_20260617/` |
| LastFM 边界诊断 | `docs/execution_plan_lastfmasia_20260618.md` | `runs/lastfmasia_20260618/` |
| MLP(X) 独立归因、噪声谱、效率和 X-ball 诊断 | `docs/superpowers/plans/2026-06-18-softdist-mlp-diagnostic.md` | `runs/softdist_mlp_seed0_20260618/`、`runs/softdist_mlp_5seed_20260618/`、`runs/softdist_mlp_noise_spectrum_20260618/`、`runs/softdist_mlp_efficiency_20260618/`、`runs/featuregb_mlp_diagnostic_20260618/` |

## 运行与验证

PowerShell：

```powershell
$py = $env:NOISYGL_ROCM_PYTHON
& $py -m pytest tests -q
```

若环境变量未设置，当前机器已验证的解释器为
`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`。
正式实验继续使用 `cuda:0`，并在实验报告中记录 PyTorch/PyG 版本。

## 文件保留策略

- 应进入代码版本：`predictor/`、`utils/`、`scripts/`、`tests/`、`config/`、`docs/`。
- 应保留用于科研追溯：最终 CSV、汇总 Markdown、图、关键 raw log、实际运行配置。
- 不作为科研证据：`__pycache__/`、`*.pyc`、`.pytest_cache/`、`*.pid`。
- `runs/` 当前保持原位，未做删除或压缩。后续提交时应按实验阶段拆分，避免把
  800 个产物与核心代码压成一个不可审查的提交。

## 继续开发前的边界

1. 不改变 split、noise injection、seed 或 baseline evaluation。
2. 新方法必须同时有 matched CE 和 shuffled-ball 对照。
3. smoke 通过后才扩 5 seeds；小于既有 std 的变化不作为稳定提升。
4. frozen-GRACE 预训练成本必须与下游训练成本分开报告。
5. MLP 诊断首先回答“球软目标能否脱离消息传播”，不等价于“完全不用 H”。
6. 不得使用“模型无关/classifier-independent”；GAT/APPNP 的 CiteSeer 负结果必须保留。
7. 固定 X-space 参数下的单球退化已经记录，不得在看到结果后调球数或 split
   threshold 再包装成预注册验证。topology-ball 如要继续，必须独立预注册。
