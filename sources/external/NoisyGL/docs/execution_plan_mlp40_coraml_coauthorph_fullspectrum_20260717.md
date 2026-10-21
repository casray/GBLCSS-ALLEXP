# MLP-40：Cora_ML 与 Coauthor-Physics 缺失噪声谱补全

## 研究假设

Cora_ML 与 Coauthor-Physics 在 uniform0.4 下均显示真实分组结构贡献，但相对最强同骨干对照仅数值接近。补齐其余噪声设置可以判断这种贡献是否跨噪声体制稳定，以及它们应进入核心全谱主表、全谱边界表，还是继续保留为标准设置广度数据集。

## 已有数据审计

- Cora_ML 已完成 uniform0.2 与 uniform0.4 的 13 方法×5 seeds；真正缺失 uniform0.6、pair0.4、instance0.4。
- Cora_ML 在 u0.2 落后正式最强 Small-loss 1.71pp，在 u0.4 领先 Small-loss 0.22pp，但仅 2/5 paired seeds 正向。
- Coauthor-Physics 已完成 uniform0.2/0.4/0.6；真正缺失 pair0.4、instance0.4。
- Coauthor-Physics 在 u0.2/u0.4/u0.6 相对最强正式对照为 −0.69/+0.02/−2.29pp，因此即使 pair/instance 都赢，也最多取得 3/5 设置最高均值，不能进入“核心成功数据集”计数，只可能升级为全谱边界数据集。

## 粒球角色

LCSS 继续使用 frozen GRACE H 上 label-blind 粒球分组生成 train-only soft target。参数与主方法完全冻结，不为两个数据集调整。

## 方法集合

每个设置只运行正式全谱排名的九臂：

1. LCSS beta=0.5。
2. size-matched shuffled LCSS。
3. MLP+CE。
4. Small-loss。
5. Co-teaching。
6. JoCoR。
7. SCE。
8. ELR。
9. LogitClip。

ANL-CE、GMM+CT、FINE 与历史代理不扩到全谱。

## 固定协议

- Split：percent 0.2/0.1/0.7。
- Noise scope：train。
- GPU：ROCm `cuda:0`。
- Seed0 阶段新增 5 settings×9 methods=45 runs。
- 已有 u0.2/u0.4/u0.6 行按 dataset/seed/split/noise hash 复用，不重跑。

## Seed0 扩种子门

### Cora_ML

三个缺失设置中必须同时满足：

1. 至少 2/3 设置 LCSS 高于该设置最强正式 baseline。
2. 三设置平均 `LCSS−best baseline >= +1pp`。
3. 任一设置不得低于最强 baseline 超过 1pp。
4. 三设置均满足 `LCSS−shuffled >= +1pp`。

通过后才为三设置补 seeds1-4。

### Coauthor-Physics

由于已有 uniform 两项明确落后，本数据集不再竞争核心成功计数。pair/instance 只有同时满足：

1. 两设置均 `LCSS−best baseline > 0`。
2. 两设置平均领先至少 +0.5pp。
3. 两设置均 `LCSS−shuffled >= +1pp`。

才补 seeds1-4，用于构建全谱边界表；否则保留现有三档 uniform 边界，不继续扩种子。

## 五种子最终处置

- Cora_ML 只有在 5 个设置中至少 4 个取得最高均值、至少 4 个 real−shuffled 达 +1pp 且 4/5 seeds 正向、任一设置不落后最强对照超过 1pp时，才可加入核心全谱数据集；否则只进全谱边界或广度表。
- Coauthor-Physics 无论 pair/instance 结果如何，都不加入核心成功计数；若 seed0 门通过并补齐 5-seed，则进入全谱边界表。
- 不因结果修改 beta、球参数、MLP、forget-rate 或 baseline 配置。

## 完整性要求

全部新行必须 `status=ok`、split/noise hash 在同 dataset/setting/seed 内一致、`modified_val_count=modified_test_count=0`、无 NaN/OOM/单类坍缩。

## 实验结果

seed0 阶段已完成 45/45 个 ROCm GPU run，总墙钟约 457 秒。全部行
`status=ok`，同 dataset/setting 内 split/noise hash 一致，且
`modified_val_count=modified_test_count=0`、`modified_count=modified_train_count`。

| 数据集 | 设置 | LCSS | 最强正式 baseline | Shuffled | LCSS−best | LCSS−shuffled |
|---|---|---:|---|---:|---:|---:|
| Cora_ML | uniform0.6 | 37.88 | JoCoR 42.08 | 31.73 | −4.20pp | +6.15pp |
| Cora_ML | pair0.4 | 51.91 | Small-loss 50.33 | 46.37 | +1.57pp | +5.53pp |
| Cora_ML | instance0.4 | 52.91 | Small-loss 55.53 | 46.56 | −2.62pp | +6.35pp |
| Coauthor-Physics | pair0.4 | 89.39 | JoCoR 89.31 | 83.62 | +0.09pp | +5.78pp |
| Coauthor-Physics | instance0.4 | 93.40 | SCE 92.71 | 87.49 | +0.69pp | +5.91pp |

Cora_ML 只有 1/3 设置高于最强 baseline，三设置平均为 −1.75pp，且
uniform0.6 与 instance0.4 的回退均超过 1pp；仅 real−shuffled 的 3/3 结构门通过。
Coauthor-Physics 两设置均数值略高于最强 baseline，但平均仅 +0.39pp，未达到预注册
+0.5pp 扩种子线；real−shuffled 的 2/2 结构门通过。

## 最终决策

**两个数据集均未通过冻结 seed0 扩种子门，停止，不补 seeds 1–4。**

- Cora_ML 不进入核心全谱主表，继续作为附录广度/边界数据集。
- Coauthor-Physics 不能进入核心成功计数；由于平均领先未达门槛，本轮也不补成 pair/instance
  五种子全谱表，继续保留已有 uniform 三档边界证据。
- 两数据集真实分组相对 shuffled 均有明显 seed0 增量，但这只能支持“局部结构有信号”，
  不能支持“稳定超过强鲁棒 baseline”。
- 不放宽门槛，不修改 beta、球参数、MLP 或 baseline 配置。

证据：`runs/mlp40_coraml_coauthorph_fullspectrum_20260717/mlp40_report.md`。
