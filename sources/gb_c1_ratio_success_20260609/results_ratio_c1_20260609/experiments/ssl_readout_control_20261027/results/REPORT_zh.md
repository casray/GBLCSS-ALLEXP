# B批：GraphECL统一读出结果

新拟合64/64；原生引用32；旧R2引用32；失败0。

每个seed先平均三个噪声条件，再报告两个seed均值和样本SD。所有路径使用相同原始GraphECL粒球目标。
旧raw R2还具有不同分类头初始化；其差异记作特征归一化与统一读出合并效果。PCA同时含中心化、截断与行归一化。

|图|读出特征|监督|噪声Accuracy%|SD|验证%|clean Accuracy%|噪声MacroF1%|
|---|---|---|---:|---:|---:|---:|---:|
|coauthorcs|native|CE|88.206|0.020|88.407|91.171|81.391|
|coauthorcs|pca64|CE|86.532|0.220|86.570|91.471|81.293|
|coauthorcs|r2_unit64|CE|88.166|0.485|87.807|92.075|82.548|
|coauthorcs|r2_raw64|CE|84.805|0.444|84.970|92.936|79.088|
|coauthorcs|native|SUPPORT32|89.198|0.501|89.462|90.871|84.612|
|coauthorcs|pca64|SUPPORT32|88.613|0.511|88.552|90.953|83.910|
|coauthorcs|r2_unit64|SUPPORT32|89.700|0.527|89.698|91.627|85.201|
|coauthorcs|r2_raw64|SUPPORT32|87.676|0.134|87.834|92.060|82.669|
|amazonpho|native|CE|86.813|0.453|86.906|93.371|82.446|
|amazonpho|pca64|CE|85.711|0.295|85.773|93.380|82.239|
|amazonpho|r2_unit64|CE|82.219|2.020|82.789|88.627|77.784|
|amazonpho|r2_raw64|CE|81.198|2.144|82.593|91.905|77.114|
|amazonpho|native|SUPPORT32|89.153|0.084|89.172|92.549|85.526|
|amazonpho|pca64|SUPPORT32|89.872|0.158|90.022|92.764|86.912|
|amazonpho|r2_unit64|SUPPORT32|86.078|1.184|86.688|88.049|81.344|
|amazonpho|r2_raw64|SUPPORT32|85.801|1.048|87.124|91.485|82.023|

## 验证门

[
  {
    "dataset": "coauthorcs",
    "head": "CE",
    "gate": "PCA_above_R2",
    "passed": false,
    "validation_noisy": {
      "seeds": [
        -1.2365884706310282,
        -1.2365884706310208
      ],
      "mean": -1.2365884706310246,
      "sd": 5.181305113656458e-15
    },
    "validation_clean_delta_pp": -0.8456082924167996
  },
  {
    "dataset": "coauthorcs",
    "head": "CE",
    "gate": "PCA_near_native",
    "passed": false,
    "validation_noisy": {
      "seeds": [
        -2.0185488270594667,
        -1.6548463356973981
      ],
      "mean": -1.8366975813784325,
      "sd": 0.2571764979765604
    },
    "validation_clean_delta_pp": -0.3546099290780147
  },
  {
    "dataset": "coauthorcs",
    "head": "SUPPORT32",
    "gate": "PCA_above_R2",
    "passed": false,
    "validation_noisy": {
      "seeds": [
        -0.9456264775413725,
        -1.345699218039642
      ],
      "mean": -1.1456628477905073,
      "sd": 0.2828941477742123
    },
    "validation_clean_delta_pp": -0.7364975450081857
  },
  {
    "dataset": "coauthorcs",
    "head": "SUPPORT32",
    "gate": "PCA_near_native",
    "passed": false,
    "validation_noisy": {
      "seeds": [
        -1.145662847790511,
        -0.672849609019821
      ],
      "mean": -0.9092562284051661,
      "sd": 0.3343294473695292
    },
    "validation_clean_delta_pp": -0.6001091107474099
  },
  {
    "dataset": "amazonpho",
    "head": "CE",
    "gate": "PCA_above_R2",
    "passed": true,
    "validation_noisy": {
      "seeds": [
        4.183006535947713,
        1.78649237472767
      ],
      "mean": 2.9847494553376914,
      "sd": 1.6945914146082832
    },
    "validation_clean_delta_pp": 4.052287581699349
  },
  {
    "dataset": "amazonpho",
    "head": "CE",
    "gate": "PCA_near_native",
    "passed": false,
    "validation_noisy": {
      "seeds": [
        -1.1764705882352937,
        -1.0893246187363796
      ],
      "mean": -1.1328976034858367,
      "sd": 0.061621505985758177
    },
    "validation_clean_delta_pp": -0.5228758169934677
  },
  {
    "dataset": "amazonpho",
    "head": "SUPPORT32",
    "gate": "PCA_above_R2",
    "passed": true,
    "validation_noisy": {
      "seeds": [
        4.226579520697164,
        2.440087145969505
      ],
      "mean": 3.3333333333333344,
      "sd": 1.2632408727079862
    },
    "validation_clean_delta_pp": 3.9215686274509833
  },
  {
    "dataset": "amazonpho",
    "head": "SUPPORT32",
    "gate": "PCA_near_native",
    "passed": true,
    "validation_noisy": {
      "seeds": [
        0.6535947712418314,
        1.0457516339869282
      ],
      "mean": 0.8496732026143798,
      "sd": 0.2772967769359001
    },
    "validation_clean_delta_pp": 0.0
  }
]

## 审计

{
  "new_heads": 64,
  "native_aliases": 32,
  "raw_r2_aliases": 32,
  "total_scored": 128,
  "failures": [],
  "source_hashes_verified": 497,
  "feature_signature_verified": true,
  "all_targets_identical_to_original": true,
  "all_selections_first_validation_max": true,
  "matching_64dim_initialization_pairs": 32,
  "max_new_cpu_replay_error": 4.76837158203125e-07
}