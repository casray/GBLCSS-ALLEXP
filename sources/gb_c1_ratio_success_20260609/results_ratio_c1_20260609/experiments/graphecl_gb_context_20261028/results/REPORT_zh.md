# GB-ContextECL 五图两seed：预算修订结果

全部58个保留SSL端点与492个分类头拟合完成后评分。GB100及匹配对照覆盖五图；小剂量覆盖前三图；Roman延后。CS/Photo完整底座复用冻结结果，新增分支仍按原总轮数训练。先在每seed内平均三噪声，再计算两seed均值/样本SD。

|数据|方法|监督|测试Accuracy±SD %|Macro-F1 %|验证 %|
|---|---|---|---:|---:|---:|
|cora|ecl|CE|75.251 ± 0.199|71.186|77.222|
|cora|ecl|fixed|78.953 ± 0.833|75.651|79.877|
|cora|gb025|CE|75.145 ± 0.100|71.192|77.284|
|cora|gb025|fixed|78.980 ± 0.771|75.682|79.938|
|cora|gb025|own|78.857 ± 0.672|75.442|79.938|
|cora|cos025|CE|75.233 ± 0.224|71.190|77.222|
|cora|cos025|fixed|78.980 ± 0.871|75.660|79.938|
|cora|cos025|own|78.399 ± 0.771|74.521|79.691|
|cora|shuffle025|CE|75.479 ± 0.124|71.343|76.914|
|cora|shuffle025|fixed|79.024 ± 0.585|75.649|79.877|
|cora|shuffle025|own|78.751 ± 0.423|75.397|79.568|
|cora|gb100|CE|75.541 ± 0.211|71.785|77.037|
|cora|gb100|fixed|78.989 ± 0.833|75.656|79.630|
|cora|gb100|own|78.813 ± 0.983|75.286|79.568|
|cora|cos100|CE|75.374 ± 0.199|71.534|77.222|
|cora|cos100|fixed|78.945 ± 0.846|75.612|79.938|
|cora|cos100|own|78.839 ± 0.572|76.096|80.802|
|cora|shuffle100|CE|75.374 ± 0.224|71.487|77.037|
|cora|shuffle100|fixed|78.830 ± 0.460|75.421|79.630|
|cora|shuffle100|own|78.663 ± 0.522|75.163|79.383|
|citeseer|ecl|CE|59.436 ± 3.139|51.247|60.141|
|citeseer|ecl|fixed|58.985 ± 3.938|50.133|59.588|
|citeseer|gb025|CE|59.228 ± 3.453|50.914|60.191|
|citeseer|gb025|fixed|59.006 ± 3.888|50.175|59.538|
|citeseer|gb025|own|58.956 ± 3.311|50.202|59.839|
|citeseer|cos025|CE|59.429 ± 3.169|51.237|60.141|
|citeseer|cos025|fixed|58.956 ± 3.878|50.109|59.538|
|citeseer|cos025|own|58.978 ± 3.847|50.141|59.639|
|citeseer|shuffle025|CE|59.386 ± 3.250|51.170|60.191|
|citeseer|shuffle025|fixed|58.985 ± 3.756|50.166|59.639|
|citeseer|shuffle025|own|59.171 ± 4.100|50.074|59.990|
|citeseer|gb100|CE|59.422 ± 3.442|51.060|60.191|
|citeseer|gb100|fixed|58.906 ± 3.787|49.886|59.388|
|citeseer|gb100|own|58.913 ± 3.615|50.311|59.388|
|citeseer|cos100|CE|59.414 ± 3.169|51.227|60.141|
|citeseer|cos100|fixed|58.942 ± 3.878|50.096|59.538|
|citeseer|cos100|own|58.928 ± 3.837|50.093|59.538|
|citeseer|shuffle100|CE|59.278 ± 3.503|50.976|60.141|
|citeseer|shuffle100|fixed|59.056 ± 3.959|50.024|59.639|
|citeseer|shuffle100|own|59.436 ± 4.111|50.721|60.392|
|corafull|ecl|CE|53.644 ± 0.510|34.550|54.346|
|corafull|ecl|fixed|53.799 ± 0.267|34.132|54.640|
|corafull|gb025|CE|53.509 ± 0.221|34.447|54.480|
|corafull|gb025|fixed|53.748 ± 0.165|34.136|54.767|
|corafull|gb025|own|53.788 ± 0.633|34.495|54.666|
|corafull|cos025|CE|53.578 ± 0.577|34.470|54.497|
|corafull|cos025|fixed|53.725 ± 0.153|34.142|54.682|
|corafull|cos025|own|53.842 ± 0.359|34.676|54.977|
|corafull|shuffle025|CE|53.612 ± 0.456|34.579|54.362|
|corafull|shuffle025|fixed|53.728 ± 0.286|34.205|54.607|
|corafull|shuffle025|own|53.727 ± 0.226|34.471|54.682|
|corafull|gb100|CE|53.734 ± 0.580|34.709|54.506|
|corafull|gb100|fixed|53.746 ± 0.437|34.228|54.716|
|corafull|gb100|own|54.071 ± 0.910|34.797|54.784|
|corafull|cos100|CE|53.664 ± 0.427|34.602|54.497|
|corafull|cos100|fixed|53.728 ± 0.211|34.083|54.750|
|corafull|cos100|own|53.602 ± 0.002|34.081|54.497|
|corafull|shuffle100|CE|53.532 ± 0.451|34.640|54.413|
|corafull|shuffle100|fixed|53.602 ± 0.243|34.176|54.413|
|corafull|shuffle100|own|53.679 ± 0.526|33.982|54.438|
|coauthorcs|ecl|CE|88.206 ± 0.020|81.391|88.407|
|coauthorcs|ecl|fixed|89.198 ± 0.501|84.612|89.462|
|coauthorcs|gb100|CE|88.575 ± 0.020|82.670|88.871|
|coauthorcs|gb100|fixed|89.340 ± 0.551|84.194|89.534|
|coauthorcs|gb100|own|89.540 ± 0.467|85.483|89.871|
|coauthorcs|cos100|CE|88.315 ± 0.031|81.963|88.507|
|coauthorcs|cos100|fixed|89.246 ± 0.452|84.577|89.471|
|coauthorcs|cos100|own|89.335 ± 0.327|84.661|89.698|
|coauthorcs|shuffle100|CE|87.928 ± 0.031|81.929|88.325|
|coauthorcs|shuffle100|fixed|89.039 ± 0.331|83.911|89.334|
|coauthorcs|shuffle100|own|89.202 ± 0.114|83.949|89.534|
|amazonpho|ecl|CE|86.813 ± 0.453|82.446|86.906|
|amazonpho|ecl|fixed|89.153 ± 0.084|85.526|89.172|
|amazonpho|gb100|CE|86.082 ± 0.053|81.636|86.863|
|amazonpho|gb100|fixed|88.992 ± 0.445|85.292|89.020|
|amazonpho|gb100|own|88.425 ± 0.462|84.173|88.889|
|amazonpho|cos100|CE|86.782 ± 0.427|82.366|86.797|
|amazonpho|cos100|fixed|89.051 ± 0.035|85.292|89.129|
|amazonpho|cos100|own|88.615 ± 0.493|84.605|88.780|
|amazonpho|shuffle100|CE|86.355 ± 0.414|80.975|86.906|
|amazonpho|shuffle100|fixed|88.764 ± 0.202|84.874|88.911|
|amazonpho|shuffle100|own|88.718 ± 0.154|84.664|88.758|

## 五主图固定目标验证门
[
  {
    "arm": "gb100",
    "mean_validation_pp": -0.09033717699116356,
    "seed_validation_pp": [
      -0.16553100838813267,
      -0.015143345594194436
    ],
    "positive_primary_graphs": 2,
    "vs_cos_validation_pp": -0.1076431916325545,
    "vs_shuffle_validation_pp": 0.07222964840624643,
    "advance": false
  }
]

## 判读
GB对ECL衡量附加机制增量；同剂量Cos和Shuffle用于解释增量来源。fixed与own的同H比较定位监督目标变化。强剂量为五图主比较；弱剂量和异配范围按预算修订分别缩减与延后。clean不进入推进门。详细逐seed/逐条件及目标Brier见scores；梯度和权重剂量见formal/warmup。

{
  "status": "passed",
  "scored": 492,
  "ssl_endpoints": 58,
  "revision": "context_ecl_budget_revision_20261028",
  "baseline_replayed": 6,
  "baseline_cached": 4,
  "max_cpu_probability_error": 2.6226043701171875e-06,
  "max_independent_target_error": 1.0202742073417426e-07,
  "sources": 570,
  "gpu_bitwise_reloads": 492,
  "cpu_rank_disagreement_heads": 2,
  "cpu_rank_disagreement_nodes": 2,
  "cpu_test_rank_disagreements": 2,
  "repair_receipt_sha256": "3f87857ec4d6af778c039b4254dfe8e36b90c35a239408bdbae57b9eff682d88"
}