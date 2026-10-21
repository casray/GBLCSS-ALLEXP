"""Audit completed v2 artifacts and produce descriptive, seed-paired evidence."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo

import numpy as np

DATASETS = ["cora", "citeseer", "pubmed", "amazonpho", "corafull", "coauthorcs"]
PRIMARY = DATASETS[:-1]
NAMES = dict(cora="Cora", citeseer="CiteSeer", pubmed="PubMed", amazonpho="Photo", corafull="CoraFull", coauthorcs="CS")
NOISES = ["clean", "u0.4", "u0.6", "pair0.4"]
BACKBONES = ["cached_mlp", "gcn"]
METRICS = ["test_accuracy", "macro_f1", "worst_class_recall", "cold_total_s", "run_total_s", "gpu_peak_allocated_gib"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(8*1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def meanstd(values):
    return statistics.mean(values), statistics.stdev(values)


def fmt(mean, std):
    return f"{mean:.2f} ± {std:.2f}"


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"]*len(headers)) + "|\n" + "\n".join("| " + " | ".join(str(x) for x in row) + " |" for row in rows) + "\n"


def main(run, source, output):
    run, source, output = Path(run), Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    from run_suite import source_fingerprint, validate_completed_result
    from sklearn.metrics import adjusted_rand_score

    state = json.loads((run/"status.json").read_text(encoding="utf-8-sig"))
    protocol = json.loads((run/"protocol.json").read_text(encoding="utf-8"))
    config = json.loads((run/"config.json").read_text(encoding="utf-8"))
    jobs = json.loads((run/"jobs.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed" and state["completed"] == state["total"] == len(jobs) == 1080
    assert source_fingerprint(source)[1] == protocol["source_sha256"]
    assert sha(run/"config.json") == sha(source/"config_v2.json") == protocol["config_sha256"]
    assert len(list((run/"results").glob("*.json"))) == 1080
    assert len({j["id"] for j in jobs}) == 1080

    cache = {}
    cache_files = 0
    for dataset in DATASETS:
        folder = run/"cache"/dataset
        meta = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
        assert meta["config_sha256"] == protocol["config_sha256"]
        for name, expected in meta["cache_files"].items():
            assert (folder/name).stat().st_size == expected["bytes"]
            assert sha(folder/name) == expected["sha256"], (dataset, name)
            cache_files += 1
        assert sha(Path(meta["source"])) == meta["source_sha256"], dataset
        y = np.load(folder/"y.npy", mmap_mode="r")
        cache[dataset] = (meta, y)
        print("Cache/source hashes verified:", dataset, flush=True)

    results, index = [], {}
    max_accuracy_error = max_f1_error = 0.
    for job in jobs:
        r = json.loads((run/"results"/(job["id"]+".json")).read_text(encoding="utf-8"))
        validate_completed_result(r, job, protocol["config_sha256"], protocol["source_sha256"], run)
        assert "parent_process_wall_s" in r and r["source_hashes"] == protocol["source_hashes"]
        meta, y = cache[job["dataset"]]
        assert r["source_data_sha256"] == meta["source_sha256"]
        pred = np.load(run/"results"/r["prediction_file"])
        labels, predicted = y[pred["indices"]], pred["prediction"]
        assert len(labels) == len(predicted) and len(np.unique(pred["indices"])) == len(labels)
        accuracy = float(np.mean(labels == predicted))
        max_accuracy_error = max(max_accuracy_error, abs(accuracy-r["test_accuracy"]))
        assert abs(accuracy-r["test_accuracy"]) < 1e-12
        matrix = np.bincount(labels*r["C"]+predicted, minlength=r["C"]**2).reshape(r["C"], r["C"])
        denominator = matrix.sum(0)+matrix.sum(1)
        f1 = np.divide(2*np.diag(matrix), denominator, out=np.zeros(r["C"], dtype=float), where=denominator>0)
        max_f1_error = max(max_f1_error, abs(float(f1.mean())-r["macro_f1"]))
        assert abs(float(f1.mean())-r["macro_f1"]) < 1e-12
        recall = np.divide(np.diag(matrix), matrix.sum(1), out=np.zeros(r["C"], dtype=float), where=matrix.sum(1)>0)
        assert np.max(np.abs(recall-np.asarray(r["per_class_recall"]))) < 1e-12
        assert abs(float(recall.min())-r["worst_class_recall"]) < 1e-12
        if r["secondary_lift_test_accuracy"] is not None:
            assert "secondary_lift_prediction" in pred.files
            assert abs(float(np.mean(pred["secondary_lift_prediction"]==labels))-r["secondary_lift_test_accuracy"]) < 1e-12
            assert r["best_secondary_epoch"] in [h["epoch"] for h in r["validation_history"]]
        assert r["best_primary_epoch"] in [h["epoch"] for h in r["validation_history"]]
        assert abs(r["best_primary_val"]-max(h["primary_val"] for h in r["validation_history"])) < 1e-12
        assert r["amortized_total_s"]["1"] == r["cold_total_s"]
        assert r["cold_total_s"] >= r["parent_process_wall_s"]
        assert r["gpu_peak_allocated_gib"] >= r["gpu_primary_inference_peak_gib"]
        if job["arm"] != "A":
            assert r["M"] == round(r["N"]*job["ratio"])
        for metric in METRICS:
            assert np.isfinite(r[metric])
        key = tuple(job[k] for k in ["dataset", "noise", "seed", "backbone", "ratio", "arm", "rule"])
        assert key not in index
        index[key] = r
        results.append(r)

    def get(ds, no, bb, arm, rule="R0", ratio=.1):
        return [index[(ds, no, seed, bb, ratio, arm, rule)] for seed in [0,1,2]]

    def paired(ds, no, bb, arm="C", ref="A", rule="R0", ref_rule="R0", ratio=.1):
        a, b = get(ds,no,bb,arm,rule,ratio), get(ds,no,bb,ref,ref_rule,.1 if ref=="A" else ratio)
        for x,y in zip(a,b):
            assert x["split_sha256"]==y["split_sha256"] and x["noise_sha256"]==y["noise_sha256"]
            if rule==ref_rule=="R0" and arm=="C" and ref in ["B","Bprime","LCSS"]:
                assert x["assignment_sha256"]==y["assignment_sha256"]
            if rule==ref_rule=="R0" and arm=="C" and ref=="B":
                assert x["target_brier_clean_diagnostic"]==y["target_brier_clean_diagnostic"]
        differences = [100*(x["test_accuracy"]-y["test_accuracy"]) for x,y in zip(a,b)]
        mean, std = meanstd(differences)
        return dict(dataset=ds,noise=no,backbone=bb,ratio=ratio,arm=arm,rule=rule,reference=ref,reference_rule=ref_rule,
                    delta_pp_mean=mean,delta_pp_std=std,seed_deltas_pp=differences,
                    positive_seeds=sum(d>0 for d in differences),negative_seeds=sum(d<0 for d in differences))

    main_rows, pair_rows, e2_rows, cost_rows, sensitivity = [], [], [], [], []
    for bb in BACKBONES:
        for ds in DATASETS:
            for no in NOISES:
                row = dict(dataset=ds,noise=no,backbone=bb)
                arms=["A","B","Bprime","LCSS","C","D"] if ds in PRIMARY else ["A","C","D"]
                for arm in arms:
                    observations=get(ds,no,bb,arm)
                    for metric in ["test_accuracy","macro_f1","worst_class_recall"]:
                        mean,std=meanstd([x[metric]*100 for x in observations])
                        row[f"{arm}_{metric}_mean_percent"]=mean
                        row[f"{arm}_{metric}_std_pp"]=std
                main_rows.append(row)
                for ref in (["A","B","Bprime","LCSS","D"] if ds in PRIMARY else ["A","D"]):
                    pair_rows.append(paired(ds,no,bb,ref=ref))
                for ratio in ([.05,.1,.3] if ds=="corafull" else [.1]):
                    a,c=get(ds,no,bb,"A"),get(ds,no,bb,"C",ratio=ratio)
                    cost=dict(dataset=ds,noise=no,backbone=bb,ratio=ratio)
                    for metric in ["cold_total_s","gpu_peak_allocated_gib","parent_process_wall_s"]:
                        am,as_=meanstd([x[metric] for x in a]);cm,cs=meanstd([x[metric] for x in c])
                        cost.update({"A_"+metric:am,"C_"+metric:cm,"A_"+metric+"_std":as_,"C_"+metric+"_std":cs,"C_over_A_"+metric:cm/am})
                    cost["accuracy_delta_pp"]=paired(ds,no,bb,ratio=ratio)["delta_pp_mean"]
                    cost["training_time_C_over_A"]=statistics.mean(x["times"]["training_s"] for x in c)/statistics.mean(x["times"]["training_s"] for x in a)
                    for k in [1,3,10,30]:
                        am=statistics.mean(x["amortized_total_s"][str(k)] for x in a)
                        cm=statistics.mean(x["amortized_total_s"][str(k)] for x in c)
                        cost[f"K{k}_C_over_A"]=cm/am
                    cost["G0_observed_pass"]=no=="clean" and cost["accuracy_delta_pp"]>=-1 and (cost["C_over_A_cold_total_s"]<=.5 or cost["C_over_A_gpu_peak_allocated_gib"]<=.5)
                    cost_rows.append(cost)
        for ds in ["cora","corafull"]:
            for no in ["u0.4","pair0.4"]:
                for rule in ["R3","O2"]:
                    p=paired(ds,no,bb,rule=rule,ref="C")
                    for ru,label in [("R0","baseline"),(rule,"candidate")]:
                        m,s=meanstd([x["test_accuracy"]*100 for x in get(ds,no,bb,"C",ru)])
                        p[label+"_mean_percent"]=m;p[label+"_std_pp"]=s
                    for seed in [0,1,2]:
                        folder=run/"cache"/ds/"cuts"
                        budget=round(cache[ds][0]["N"]*.1)
                        r0=np.load(folder/f"R0_M{budget}.npy")
                        # Match the executed worker's Path.with_suffix convention.
                        # The locked matrix has only one uniform rate in E2.
                        other=np.load((folder/f"{rule}_M{budget}_s{seed}_{no}").with_suffix(".npy"))
                        p[f"ARI_seed{seed}"]=float(adjusted_rand_score(r0,other))
                    e2_rows.append(p)
        for no in NOISES:
            for ratio in [.05,.1,.3]:
                row=dict(backbone=bb,noise=no,ratio=ratio)
                for arm in ["B","LCSS","C"]:
                    m,s=meanstd([x["test_accuracy"]*100 for x in get("corafull",no,bb,arm,ratio=ratio)])
                    row[arm+"_mean_percent"]=m;row[arm+"_std_pp"]=s
                row["A_mean_percent"]=statistics.mean(x["test_accuracy"]*100 for x in get("corafull",no,bb,"A"))
                sensitivity.append(row)

    aggregates=[]
    for bb in BACKBONES:
        for ref in ["A","B","Bprime","LCSS","D"]:
            cells=[p for p in pair_rows if p["backbone"]==bb and p["dataset"] in PRIMARY and p["noise"]!="clean" and p["reference"]==ref]
            assert len(cells)==15
            aggregates.append(dict(backbone=bb,reference=ref,unit="15 dataset x noise cell means; descriptive, not independent repetitions",mean_delta_pp=statistics.mean(p["delta_pp_mean"] for p in cells),positive_cells=sum(p["delta_pp_mean"]>0 for p in cells),all3_positive_cells=sum(p["positive_seeds"]==3 for p in cells),cells=15))
    e2_aggregate={rule:statistics.mean(p["delta_pp_mean"] for p in e2_rows if p["rule"]==rule) for rule in ["R3","O2"]}
    audit=dict(status="passed",verification_status="ANALYZED: all saved prediction/metric artifacts verified; no independent retraining",jobs=1080,seed_counts=dict(Counter(r["job"]["seed"] for r in results)),experiment_counts=dict(Counter(r["job"]["experiment"] for r in results)),missing=0,cache_files_verified=cache_files,source_datasets_verified=6,prediction_artifacts_verified=1080,max_accuracy_recompute_error=max_accuracy_error,max_macro_f1_recompute_error=max_f1_error,config_sha256=protocol["config_sha256"],source_sha256=protocol["source_sha256"],finished_at_china=datetime.fromisoformat(state["finished_at"]).astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),worker_wall_sum_minutes=sum(r["parent_process_wall_s"] for r in results)/60,G0_pass_observed_cells=sum(c["G0_observed_pass"] for c in cost_rows),E2_equal_cell_mean_delta_pp=e2_aggregate,limitations=["3 seeds, descriptive screening only","reconstructed cold costs, not isolated cold end-to-end trials","cached MLP is not official SIGN","original-graph inference; not graph-free deployment","LCSS is old formula on new partition/backbone, not original full GB-LCSS","no SGBGC or robust-method direct comparisons in this run","E2 limited two datasets and two noise settings; no ordinary-Gini R1 arm","same-split independent-noise stability experiment was not run"])
    save_json(output/"audit.json",audit)
    save_json(output/"aggregates.json",dict(main=aggregates,E2=e2_aggregate))
    save_csv(output/"main_accuracy_and_f1.csv",main_rows)
    save_csv(output/"paired_accuracy.csv",pair_rows)
    save_csv(output/"E2_diagnostic.csv",e2_rows)
    save_csv(output/"costs.csv",cost_rows)
    save_csv(output/"corafull_sensitivity.csv",sensitivity)
    with gzip.open(output/"raw_results.jsonl.gz","wt",encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")

    report=["# v2机制分解与精简E2：1080项结果及决策", "日期：2026-10-20。实验完成时间（北京时间）："+audit["finished_at_china"],
            "Verification Status: ANALYZED。已核对全部保存预测与指标、来源及配置哈希，没有独立重训。三种子仅作描述性筛查。", "## 判断",
            "球级训练在部分高噪声图上具有超出相同分区目标平滑的增量，尤其是GCN；这一线索值得保留。含噪留一不纯度分配粒度R3没有显示一致收益，当前主设定暂缓。完整效率门槛未通过，CoraFull多类损失也未修复。不能将这批结果写成完整新算法已经成功。",
            "## 实验目的、假设与粒球角色",
            "研究在有限预算下以共享训练单元抑制逐点错标拟合，同时保留真实分类差异。A=原节点自身含噪标签；B=相同分区含自身球平均目标；Bprime=纯留一平均；LCSS=完整旧目标公式在相同新分区/骨干上；C=球级输入/传播训练；D=同全节点球大小的随机分组。R0仅D评分；R3用带噪留一继承评分；O2只用干净训练标签选择切面，下游仍同一带噪监督。",
            "代码变化：独立进程、原图/原节点验证每5轮、主次checkpoint分离、exact-M树DP、固定树seed、目标隔离、阶段成本及工件哈希。配置：20/10/70划分，三种子0/1/2，hidden256，最多300轮，主M/N=.1；CoraFull额外.05/.3。WikiCS已锁为未来确认图，本轮未用。两骨干为custom GCN与缓存多跳特征MLP，当前GCN部署仍使用原图。",
            "成功要求：球训练在匹配条件下改善精度/成本，并优于目标平滑及随机分组；R3需有足够相对R0增量。失败或暂缓：收益不能跨条件保持、干净标签切面诊断也难获收益、完整成本没有优势。具体预先定义与修订见PLAN_V2.md及历史PLAN。",
            "## 1. 完整性",
            f"1080/1080完成：每个seed360项，1032项E0+48项新增E2。缺失0；30个准备缓存文件、6份原始图文件哈希及1080份预测工件核验通过。逐一重算accuracy、macro-F1、各类recall及secondary accuracy；最大误差accuracy={max_accuracy_error:.3g}、macro-F1={max_f1_error:.3g}。验证主checkpoint、预算、B/C同标签源与配对分区、配置/源码、K1成本一致性。训练进程墙钟求和约{audit['worker_wall_sum_minutes']:.1f}分钟，不包括全部准备、暂停与被中断未记账运行。",
            "## 2. u0.4主结果（均值±样本标准差，%）"]
    for bb in BACKBONES:
        report.append("### "+bb)
        report.append(table(["图","A自身标签","B平均目标","LCSS旧公式","C球训练","C−B配对差pp"],[[NAMES[ds]]+[fmt(*meanstd([x["test_accuracy"]*100 for x in get(ds,"u0.4",bb,a)])) for a in ["A","B","LCSS","C"]]+[fmt(*[paired(ds,"u0.4",bb,ref="B")[k] for k in ["delta_pp_mean","delta_pp_std"]])] for ds in PRIMARY]))
    report += ["### 较明确的额外机制线索与负结果", "以下按预先指定条件描述，不将挑选的正单元作为全域成功；全部条件与种子见CSV及压缩原始JSON。差值标准差按同seed配对计算。"]
    chosen=[("gcn","cora","u0.6"),("gcn","citeseer","u0.4"),("gcn","citeseer","u0.6"),("gcn","pubmed","u0.6"),("cached_mlp","amazonpho","u0.4"),("cached_mlp","amazonpho","u0.6"),("gcn","corafull","u0.4")]
    report.append(table(["骨干/图/噪声","C−A pp","C−B pp","C−LCSS pp","C−B三个seed差"],[[f"{bb}/{NAMES[ds]}/{no}"]+[fmt(*[paired(ds,no,bb,ref=ref)[k] for k in ["delta_pp_mean","delta_pp_std"]]) for ref in ["A","B","LCSS"]]+[", ".join(f"{v:+.2f}" for v in paired(ds,no,bb,ref="B")["seed_deltas_pp"])] for bb,ds,no in chosen]))
    report += ["GCN的C−B较强，但C同时改变传播、训练图和迁移，不能唯一归因于输入共享。缓存分支更接近输入/共享预测效应的分解：Photo u0.4 C−B为正，但仍接近或弱于完整LCSS；Photo u0.6反而明显下降。纯LOO平均不是旧论文完整目标；不能仅凭C胜Bprime声称胜旧GB-LCSS。",
               "在两个骨干的五图×三种带噪条件中，C均优于随机D，但这只证明当前语义分区优于破坏结构的随机控制；不能单独证明粒球比其他合理分区更必要。没有k-means/kNN等新的强分区对照。"]
    report.append(table(["骨干","C−A等权单元均差pp","C−B等权单元均差pp","C−LCSS等权单元均差pp","C−LCSS正单元/15"],[[bb]+[f"{next(a for a in aggregates if a['backbone']==bb and a['reference']==ref)['mean_delta_pp']:+.3f}" for ref in ["A","B","LCSS"]]+[str(next(a for a in aggregates if a['backbone']==bb and a['reference']=='LCSS')['positive_cells'])+"/15"] for bb in BACKBONES]))
    report += ["这些跨数据/噪声均差仅作描述，不把相关单元当独立重复，也不让正负平均抵消掩盖特定图的损害。",
               "## 3. E2：R3未显示一致增量"]
    report.append(table(["图/噪声/骨干","R0 accuracy%","R3−R0 pp","O2−R0 pp"],[[f"{NAMES[ds]}/{no}/{bb}",fmt(*meanstd([x['test_accuracy']*100 for x in get(ds,no,bb,'C')]))]+[fmt(*[paired(ds,no,bb,ref='C',rule=rule)[k] for k in ['delta_pp_mean','delta_pp_std']]) for rule in ['R3','O2']] for bb in BACKBONES for ds in ['cora','corafull'] for no in ['u0.4','pair0.4']]))
    report += [f"八个数据×噪声×骨干单元的等权平均：R3−R0={e2_aggregate['R3']:+.3f}pp，O2−R0={e2_aggregate['O2']:+.3f}pp。R3主增量门槛未过；干净切面诊断整体也不到1pp。CoraFull pair/GCN中O2有局部+1.828pp信号，三个seed均为正；但Cora pair/GCN的O2为−3.184pp。不能称O2性能上界，也不能否定所有标签感知划分，只能暂缓当前固定树、beta和预算目标。",
               "诊断量与分类不一致：CoraFull球内干净训练Gini由R0约.3554降到O2约.2990，但u0.4分类只小幅改变；Cora约.1511降至.1145，pair/GCN反而下降。说明降低球内混类不是最终分类优化的充分目标。这是观测到的代理量—最终任务脱节，不是数学期望等式被推翻。切面ARI列衡量规则间差异，不冒充固定split独立噪声重复的稳定性实验。",
               "决策：暂缓当前R3标签评分主线，不直接展开大规模交叉拟合救援；整体clean-label诊断未过推进门槛。局部CoraFull-pair空间保留为证据，不逐图挑beta继续搜索。R1普通Gini比较本轮未实施，无法评定经典校正相对Gini的必要性。",
               "## 4. CoraFull数量敏感性与迁移"]
    report.append(table(["骨干/条件","A","C .05","C .1","C .3"],[[f"{bb}/{no}",f"{statistics.mean(x['test_accuracy']*100 for x in get('corafull',no,bb,'A')):.2f}"]+[f"{statistics.mean(x['test_accuracy']*100 for x in get('corafull',no,bb,'C',ratio=rt)):.2f}" for rt in [.05,.1,.3]] for bb in BACKBONES for no in NOISES]))
    report += ["增加球数能缓解一部分压缩损失，但两骨干在所有四种标签条件下，.3的C均仍低于相应A。不能把‘生成足够多的球’视为已经解决多类问题；LCSS/原节点B在部分大预算条件更好，也指向输入/传播压缩损失而非只缺软目标。比较中没有按测试集逐图挑预算并作为最终方法成绩。",
               "## 5. 效率门槛未通过"]
    cost_selected=[c for c in cost_rows if c['noise']=='clean' and c['ratio']==.1 and c['dataset'] in ['corafull','coauthorcs']]
    report.append(table(["图/骨干/clean","A冷成本s","C冷成本s","C/A时间","A完整GPU峰值GiB","C完整GPU峰值GiB","K30时间比"],[[f"{NAMES[c['dataset']]}/{c['backbone']}",f"{c['A_cold_total_s']:.2f}",f"{c['C_cold_total_s']:.2f}",f"{c['C_over_A_cold_total_s']:.2f}",f"{c['A_gpu_peak_allocated_gib']:.3f}",f"{c['C_gpu_peak_allocated_gib']:.3f}",f"{c['K30_C_over_A']:.2f}"] for c in cost_selected]))
    report += ["所有当前可观察clean预算单元都未同时满足‘精度损失≤1pp且总时间或完整GPU峰值≤0.5×’。这里只是筛查：原计划的大图arxiv尚未运行，不能据此否定所有大图粗化的效率空间；本批已无法支持高效为主要实证卖点。重建冷成本含建树等阶段，非独立无缓存完整冷跑；K30收益较小，不再出现此前忽略成本时的6～11倍。",
               "完整GCN峰值包含原图验证/推理及粗图驻留输入，因而当前实现经常没有显存收益。这并不等价于粗图训练算子本身毫无节省；每轮/训练阶段通常更快，但完整流程未形成所要求优势。three seeds不是专门计时三重复，不将小时间差当稳定加速。缓存MLP不是官方SIGN，不能声称已战胜正式SIGN。",
               "## 6. 最终决策与下一步",
               "继续：保留无标签真实分区下球级GCN训练在Cora/CiteSeer和PubMed高噪声上的额外正则化线索。其价值已经超过‘仅把旧软目标搬到新代码’的猜测，但作用范围有限。",
               "暂缓：当前D+留一继承不纯度的粒度评分；不继续扫beta、加门控或用局部最优成绩改写总体失败。效率主张同样暂缓。",
               "下一步建议按既定第三批条件做有限、同协议的ERASE/ProCon/GD²强鲁棒方法比较：固定当前主预算与骨干，不重新调参挑测试成绩；保留五图全表，以Cora/CiteSeer的机制信号为重点，同时披露Photo/CoraFull损害。旧教师训练124秒等不同协议历史数字不可直接作当前效率基线。若不能接近鲁棒方法精度并形成完整成本取舍，就不把当前版本扩成完整新论文。WikiCS仍留作此前未开发的确认数据。尚未自动启动第三批。",
               "## 7. 证据强度与统计检查",
               "已支持：部分高噪声单元C优于匹配B/LCSS，随机D明显退化；CoraFull损失未消失；当前R3无一致增益；完整效率阈值未过。合理解释：共享表示与粗传播限制噪声拟合；候选评分代理与最终任务不匹配。未验证：唯一因果机制、普适异配适应性、相对强鲁棒方法价值、大图效率和官方SIGN对比。被负结果削弱：单靠当前留一不纯度分配预算就能统一改善分类。",
               "11类统计风险已检查：分组反向结果单列，避免总体平均掩盖；不由平均准确率推出每节点去噪；无仅保留成功运行的存活者选择（1080全收）；类别基率用macro-F1和召回补充；针对低分图修改存在回归均值风险，故保留WikiCS；多个条件/机制与分析路径均公开且不事后换门槛；对照实验支持组合变化的影响但不唯一归因；碰撞变量/反向因果对本固定实验不直接适用。无p值与显著性声明，3seed不证明等价或稳定泛化。",
               "工件审计的额外实现记录：分析脚本显式按UTF-8读取含中文路径的JSON，避免Windows默认编码误解路径；未改写训练工件。执行版切面缓存通过Path.with_suffix生成文件名，小数噪声字符串的末尾会被当扩展名截断。当前E2每种噪声类型只有一个率（u0.4、pair0.4），两者不会相互碰撞，全部运行的标签源哈希也一致；未来若扩展同类型多个率，必须先修缓存键并新建运行目录，不能直接复用当前键。",
               "## 8. 文件与可复核范围",
               "audit.json=完整性；main_accuracy_and_f1.csv=全部主预算均值/std；paired_accuracy.csv=同seed差值/std；E2_diagnostic.csv=两规则对照/种子差/ARI；costs.csv=成本/峰值/摊销；corafull_sensitivity.csv=预算敏感性；raw_results.jsonl.gz=1080份原始逐运行JSON副本。原始预测与日志、缓存仍在运行根目录。脚本不重新训练、不修改冻结生产实现。"]
    (output/"REPORT.md").write_text("\n\n".join(report)+"\n",encoding="utf-8")
    evidence={p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in output.iterdir() if p.is_file() and p.name not in ["EVIDENCE_MANIFEST.json"]}
    save_json(output/"EVIDENCE_MANIFEST.json",dict(date="2026-10-20",source_run=str(run),analysis_script_sha256=sha(Path(__file__)),files=evidence))
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",required=True)
    ap.add_argument("--source",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    main(args.run,args.source,args.output)
