# C批：R2尺度×初始化2×2结果

新增64/64；历史引用64；失败0。所有头同训练协议，分别报告两seed样本SD。

|图|监督|尺度|初始化|噪声Accuracy%|SD|clean Accuracy%|噪声验证%|
|---|---|---|---|---:|---:|---:|---:|
|coauthorcs|CE|raw|legacy|84.805|0.444|92.936|84.970|
|coauthorcs|CE|raw|unified|83.857|0.525|92.866|84.006|
|coauthorcs|CE|unit|legacy|88.469|0.252|91.974|87.961|
|coauthorcs|CE|unit|unified|88.166|0.485|92.075|87.807|
|coauthorcs|SUPPORT32|raw|legacy|87.676|0.134|92.060|87.834|
|coauthorcs|SUPPORT32|raw|unified|87.636|0.125|92.211|87.898|
|coauthorcs|SUPPORT32|unit|legacy|89.782|0.584|91.545|89.798|
|coauthorcs|SUPPORT32|unit|unified|89.700|0.527|91.627|89.698|
|amazonpho|CE|raw|legacy|81.198|2.144|91.905|82.593|
|amazonpho|CE|raw|unified|81.534|1.659|91.914|82.484|
|amazonpho|CE|unit|legacy|82.303|2.104|88.422|82.941|
|amazonpho|CE|unit|unified|82.219|2.020|88.627|82.789|
|amazonpho|SUPPORT32|raw|legacy|85.801|1.048|91.485|87.124|
|amazonpho|SUPPORT32|raw|unified|85.798|1.061|91.401|86.754|
|amazonpho|SUPPORT32|unit|legacy|85.910|1.501|88.058|86.688|
|amazonpho|SUPPORT32|unit|unified|86.078|1.184|88.049|86.688|

## 成对效应

|图|监督|效应|带噪测试pp|clean测试pp|带噪验证pp|clean验证pp|
|---|---|---|---:|---:|---:|---:|
|coauthorcs|CE|normalization_legacy|+3.664|-0.962|+2.991|-0.846|
|coauthorcs|CE|normalization_unified|+4.309|-0.791|+3.801|-1.064|
|coauthorcs|CE|initialization_raw|-0.948|-0.070|-0.964|+0.273|
|coauthorcs|CE|initialization_unit|-0.303|+0.101|-0.155|+0.055|
|coauthorcs|CE|normalization_main|+3.986|-0.877|+3.396|-0.955|
|coauthorcs|CE|initialization_main|-0.625|+0.016|-0.559|+0.164|
|coauthorcs|CE|interaction|+0.645|+0.171|+0.809|-0.218|
|coauthorcs|CE|diagonal|+3.361|-0.861|+2.837|-0.791|
|coauthorcs|SUPPORT32|normalization_legacy|+2.105|-0.514|+1.964|-0.846|
|coauthorcs|SUPPORT32|normalization_unified|+2.064|-0.584|+1.800|-0.900|
|coauthorcs|SUPPORT32|initialization_raw|-0.040|+0.152|+0.064|+0.027|
|coauthorcs|SUPPORT32|initialization_unit|-0.082|+0.082|-0.100|-0.027|
|coauthorcs|SUPPORT32|normalization_main|+2.084|-0.549|+1.882|-0.873|
|coauthorcs|SUPPORT32|initialization_main|-0.061|+0.117|-0.018|+0.000|
|coauthorcs|SUPPORT32|interaction|-0.042|-0.070|-0.164|-0.055|
|coauthorcs|SUPPORT32|diagonal|+2.023|-0.432|+1.864|-0.873|
|amazonpho|CE|normalization_legacy|+1.105|-3.483|+0.349|-3.399|
|amazonpho|CE|normalization_unified|+0.685|-3.287|+0.305|-3.595|
|amazonpho|CE|initialization_raw|+0.336|+0.009|-0.109|-0.065|
|amazonpho|CE|initialization_unit|-0.084|+0.205|-0.153|-0.261|
|amazonpho|CE|normalization_main|+0.895|-3.385|+0.327|-3.497|
|amazonpho|CE|initialization_main|+0.126|+0.107|-0.131|-0.163|
|amazonpho|CE|interaction|-0.420|+0.196|-0.044|-0.196|
|amazonpho|CE|diagonal|+1.021|-3.277|+0.196|-3.660|
|amazonpho|SUPPORT32|normalization_legacy|+0.109|-3.427|-0.436|-3.595|
|amazonpho|SUPPORT32|normalization_unified|+0.280|-3.352|-0.065|-3.529|
|amazonpho|SUPPORT32|initialization_raw|-0.003|-0.084|-0.370|-0.131|
|amazonpho|SUPPORT32|initialization_unit|+0.168|-0.009|-0.000|-0.065|
|amazonpho|SUPPORT32|normalization_main|+0.195|-3.389|-0.251|-3.562|
|amazonpho|SUPPORT32|initialization_main|+0.082|-0.047|-0.185|-0.098|
|amazonpho|SUPPORT32|interaction|+0.171|+0.075|+0.370|+0.065|
|amazonpho|SUPPORT32|diagonal|+0.277|-3.436|-0.436|-3.660|

## 预登记门

[
  {
    "dataset": "coauthorcs",
    "head": "CE",
    "initialization": "legacy",
    "validation_noisy_seed_deltas_pp": [
      2.2913256955810133,
      3.6915802873249675
    ],
    "validation_noisy_mean_delta_pp": 2.9914529914529906,
    "validation_clean_seed_deltas_pp": [
      -0.9819967266775791,
      -0.7092198581560325
    ],
    "validation_clean_mean_delta_pp": -0.8456082924168058,
    "noise_gate": true,
    "full_gate": false
  },
  {
    "dataset": "coauthorcs",
    "head": "CE",
    "initialization": "unified",
    "validation_noisy_seed_deltas_pp": [
      3.5279141662120423,
      4.0734679032551355
    ],
    "validation_noisy_mean_delta_pp": 3.800691034733589,
    "validation_clean_seed_deltas_pp": [
      -0.9274413529732612,
      -1.200218221494822
    ],
    "validation_clean_mean_delta_pp": -1.0638297872340416,
    "noise_gate": true,
    "full_gate": false
  },
  {
    "dataset": "coauthorcs",
    "head": "SUPPORT32",
    "initialization": "legacy",
    "validation_noisy_seed_deltas_pp": [
      1.472995090016364,
      2.454991816693943
    ],
    "validation_noisy_mean_delta_pp": 1.9639934533551533,
    "validation_clean_seed_deltas_pp": [
      -0.818330605564654,
      -0.8728859792689576
    ],
    "validation_clean_mean_delta_pp": -0.8456082924168058,
    "noise_gate": true,
    "full_gate": false
  },
  {
    "dataset": "coauthorcs",
    "head": "SUPPORT32",
    "initialization": "unified",
    "validation_noisy_seed_deltas_pp": [
      1.4184397163120603,
      2.182214948172392
    ],
    "validation_noisy_mean_delta_pp": 1.800327332242226,
    "validation_clean_seed_deltas_pp": [
      -0.7637752318603361,
      -1.0365521003818827
    ],
    "validation_clean_mean_delta_pp": -0.9001636661211094,
    "noise_gate": true,
    "full_gate": false
  },
  {
    "dataset": "amazonpho",
    "head": "CE",
    "initialization": "legacy",
    "validation_noisy_seed_deltas_pp": [
      0.2614379084967311,
      0.43572984749455657
    ],
    "validation_noisy_mean_delta_pp": 0.34858387799564383,
    "validation_clean_seed_deltas_pp": [
      -4.444444444444443,
      -2.35294117647058
    ],
    "validation_clean_mean_delta_pp": -3.3986928104575114,
    "noise_gate": false,
    "full_gate": false
  },
  {
    "dataset": "amazonpho",
    "head": "CE",
    "initialization": "unified",
    "validation_noisy_seed_deltas_pp": [
      -0.174291938997816,
      0.7843137254901981
    ],
    "validation_noisy_mean_delta_pp": 0.305010893246191,
    "validation_clean_seed_deltas_pp": [
      -4.967320261437919,
      -2.2222222222222143
    ],
    "validation_clean_mean_delta_pp": -3.594771241830067,
    "noise_gate": false,
    "full_gate": false
  },
  {
    "dataset": "amazonpho",
    "head": "SUPPORT32",
    "initialization": "legacy",
    "validation_noisy_seed_deltas_pp": [
      -0.8278867102396532,
      -0.04357298474945045
    ],
    "validation_noisy_mean_delta_pp": -0.43572984749455185,
    "validation_clean_seed_deltas_pp": [
      -4.705882352941188,
      -2.4836601307189454
    ],
    "validation_clean_mean_delta_pp": -3.594771241830067,
    "noise_gate": false,
    "full_gate": false
  },
  {
    "dataset": "amazonpho",
    "head": "SUPPORT32",
    "initialization": "unified",
    "validation_noisy_seed_deltas_pp": [
      -0.3485838779956462,
      0.21786492374727592
    ],
    "validation_noisy_mean_delta_pp": -0.06535947712418515,
    "validation_clean_seed_deltas_pp": [
      -4.705882352941188,
      -2.35294117647058
    ],
    "validation_clean_mean_delta_pp": -3.529411764705884,
    "noise_gate": false,
    "full_gate": false
  }
]

## 审计

{
  "new_heads": 64,
  "aliases": 64,
  "total_scored": 128,
  "failures": [],
  "source_hashes": 791,
  "matching_scale_initialization_pairs": 64,
  "targets_unchanged": true,
  "first_validation_max_verified": true,
  "max_new_cpu_replay_error": 9.5367431640625e-07
}