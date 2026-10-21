"""Summarize frozen diagnostic evidence and preregistered screening decisions."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    frame=pd.read_csv(ROOT/"target_metrics.csv")
    delta=pd.read_csv(ROOT/"paired_deltas.csv")
    noise=delta[delta.noise!="clean"]
    decisions={}
    for rep,rows in noise.groupby("representation"):
        decisions[rep]={}
        for comparison,label in [("GB128 - GB64","capacity"),("GB128 - KNN128","ball_vs_knn"),
                                  ("GB128 - KM128","ball_vs_km"),("GB128_full - GB128_mean_matched","reliability")]:
            selected=rows[rows.comparison==comparison]
            datasets={}
            for dataset,group in selected.groupby("dataset"):
                per_seed=group.groupby("seed").brier_delta.mean()
                datasets[dataset]=dict(mean_brier_delta=float(per_seed.mean()),std_brier_delta=float(per_seed.std(ddof=1)) if len(per_seed)>1 else None,
                    positive_seed_count=int((per_seed<0).sum()),n_seeds=len(per_seed),
                    clean_harm_delta=float(group.clean_harm_delta.mean()),
                    screening_threshold_pass=bool(per_seed.mean()<=-.005 and (per_seed<0).sum()>=4))
            decisions[rep][label]=datasets
    (ROOT/"decision.json").write_text(json.dumps(dict(stage="exploratory",decisions=decisions,
        criterion="negative brier_delta favors first arm; pooled across uniform/pair at each paired protocol seed",
        no_significance_claim=True,reliability_training="disabled unless per-dataset gates and shuffled comparator pass",
        multiple_heterophily_graphs_tested=False),ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 异配机制诊断与首轮训练：实验结论","",
      "日期：2026-10-14。计划及判据见PLAN.md；本文件由已完成的记录生成，缺失运行不视作成功。","",
      "## 范围和协议","",
      f"- 已完成目标诊断：{len(frame[['dataset','representation','seed','noise']].drop_duplicates())}个图/表示/协议seed/噪声单元，{len(frame)}行方法结果。",
      "- 旧图20/10/70，Roman官方划分；每表示目前只有一个SSL预训练seed，五个协议seed不等于五次独立表示预训练。",
      "- 训练目标先保存并hash冻结，再用训练真值评价目标质量；图性质统计另行读取真值。真值不进入构球、近邻或权重。",
      "- kNN逐节点唯一训练票数匹配GB，此处不是原稿固定K版本；跨论文数值不可直接混比。",
      "- 负Brier差表示前者更好；Brier越低越好。正margin比例分母是全部训练节点，平均margin仅限票数合格节点。",
      "", "## 数据性质",""]
    for path in sorted((ROOT/"runs").glob("*/*/graph_stats.json")):
        info=json.loads(path.read_text())
        lines.append(f"- {path.parent.parent.name}/{path.parent.name}: N={info['nodes']}, C={info['classes']}, edge homophily={info['edge_homophily']:.4f}, adjusted={info['adjusted_homophily']:.4f}。")
    lines+=["","## 原目标与容量/邻域对照","",
      "| 图/表示 | 噪声 | CE | GB64 | GB128 | KM128 | 匹配kNN128 |","|---|---|---:|---:|---:|---:|---:|"]
    for key,group in frame.groupby(["dataset","representation","noise"]):
        values=[]
        for arm in ["CE","GB64","GB128","KM128","KNN128"]:
            x=group[group.arm==arm].target_brier
            values.append(f"{x.mean():.5f}±{x.std(ddof=1):.5f}" if len(x)>1 else f"{x.mean():.5f} (n=1)")
        lines.append("| "+"/".join(key[:2])+" | "+key[2]+" | "+" | ".join(values)+" |")
    lines+=["","## 配对机制增量","",
      "| 图/表示 | 比较 | Uniform/Pair均值Brier差 | 配对seed std | 改善seed | .005探索门槛 |",
      "|---|---|---:|---:|---:|---|"]
    for rep,tests in decisions.items():
        for label,ds in tests.items():
            for dataset,item in ds.items():
                std=item["std_brier_delta"]
                lines.append(f"| {dataset}/{rep} | {label} | {item['mean_brier_delta']:+.5f} | {std:.5f}" +
                    f" | {item['positive_seed_count']}/{item['n_seeds']} | {item['screening_threshold_pass']} |" if std is not None else
                    f"| {dataset}/{rep} | {label} | {item['mean_brier_delta']:+.5f} | n=1 | {item['positive_seed_count']}/1 | False |")
    lines+=["","## 解释和决策边界","",
      "- 容量、区域和收缩分别解释，不能把GRACE→其他表示的变化算成粒球贡献。",
      "- full不胜mean_matched/shuffled时，暂缓ESS×JS可靠性作为新方法；不凭减少误修正就称总体目标更好。",
      "- GB不胜同空间kNN/KM时，不支持粒球不可替代性。容量改善仍可作为原稿瓶颈诊断。",
      "- 干净条件CE的目标Brier必为0；任何非零邻域混合都可能增加目标误差，这不等于下游准确率必降低。",
      "- 这里只支持固定几何与这些配置下的观察，不否定所有异配表示/粒球方法。",
      "", "## 下游分类状态",""]
    results=ROOT/"student_results.csv"
    if results.exists():
        students=pd.read_csv(results)
        lines.append(f"已完成{len(students)}次200epoch学生训练，完整记录见student_results.csv。")
        lines+=["","| 图/表示 | seed | 噪声 | CE | GB64 | GB128 | KNN128 |","|---|---:|---|---:|---:|---:|---:|"]
        for key,g in students.groupby(["dataset","representation","seed","noise"]):
            vals=[]
            for arm in ["CE","GB64","GB128","KNN128"]:
                x=g[g.arm==arm].test_accuracy
                vals.append(f"{100*x.iloc[0]:.2f}" if len(x) else "未完成")
            lines.append(f"| {key[0]}/{key[1]} | {key[2]} | {key[3]} | "+" | ".join(vals)+" |")
        lines+=["","仅seed0时是pilot；不计算虚假的std、不主张稳定性能。不同表示下CE相同来源的重复运行不得当独立样本。"]
    else:
        lines.append("尚无已完成学生结果；是否正在运行请查student_raw.log及session交接，不能将启动等同完成。")
    lines+=["","## 文件和复核","",
       "config.json、各run source_manifest/config/split/protocol、冻结targets、diagnostic_raw.log、target_metrics/summary、paired_deltas/summary、decision.json；表示分支另有逐epoch日志和来源说明。",
       "audit_diagnostic.py提供已保存记录的独立核对，实际审计结果见audit/；不因脚本存在就宣称审计通过。",
       "未来扩大实验需先核已有负结果；不在读到测试结果后改变本轮阈值。",""]
    (ROOT/"实验结论.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(cells=len(frame[["dataset","representation","seed","noise"]].drop_duplicates()),
                         rows=len(frame),representations=list(decisions)),ensure_ascii=True))
if __name__=="__main__":main()

