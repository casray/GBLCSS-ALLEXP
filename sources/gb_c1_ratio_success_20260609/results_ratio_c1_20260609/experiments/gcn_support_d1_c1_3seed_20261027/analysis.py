"""Independent frozen-file analysis. No training-module import and no GPU access."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
WORKSPACE = CODE.parents[1]
DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
CONDITIONS = ("clean", "uniform0.4", "uniform0.6")
METHODS = ("CE", "OLD64", "LEGACY_SUPPORT32", "DEEP_S32", "FIXED_D1", "JOINT")
CONTROLS = METHODS[:-1]
METRICS = ("test_accuracy", "test_macro_f1", "validation")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".postprocess.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def csv_save(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def relative_input(root, name):
    path = (Path(root)/name).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Manifest path escapes root: "+str(name))
    return path


class Reader:
    """Track every input separately from the immutable fitting manifest."""
    def __init__(self):
        self.inputs = {}

    def hash(self, path):
        path = str(Path(path).resolve())
        if path not in self.inputs:
            self.inputs[path] = sha(path)
        return self.inputs[path]

    def expect(self, path, digest):
        require(self.hash(path) == digest, "Hash mismatch: "+str(path))

    def json(self, path):
        self.hash(path)
        return read(path)

    def tensor(self, path, *, objects=False):
        self.hash(path)
        import torch
        return torch.load(path, map_location="cpu", weights_only=not objects)

    def logs(self, path):
        self.hash(path)
        return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]

    def verify(self):
        for path, digest in self.inputs.items():
            if sha(path) != digest:
                raise RuntimeError("Input changed during postprocessing: "+path)


def validate_config(config):
    require(tuple(config["datasets"]) == DATASETS, "Expected approved five graphs")
    require(config["seeds"] == [0, 1, 2], "Expected seeds 0/1/2")
    require(tuple(config["conditions"]) == CONDITIONS, "Expected clean/U.4/U.6")
    require(tuple(config["teacher_routes"]) == METHODS, "Expected six approved routes")
    require(config["joint_race"] == [[25, 14], [75, 4], [200, 1]], "Race changed")
    require(config["supports"] == [4, 8, 16, 32, 64, 128], "Supports changed")


def historical(dataset, seed):
    return dataset in DATASETS[:3] and seed == 0


def cell_id(dataset, seed, condition):
    return f"{dataset}/seed{seed}/{condition}"


def selection_path(root, stage, dataset, seed, condition, *, old=False):
    folder = Path(root)/"selections"/stage/dataset
    return folder/f"{condition}.json" if old else folder/f"seed{seed}"/f"{condition}.json"


def check_freeze(root, reader, *, historical=False):
    root = Path(root)
    frozen = reader.json(root/"selections_frozen.json")
    reader.expect(root/"refs.json", frozen["refs_sha256"])
    reader.expect(root/"teachers_frozen.json", frozen["teacher_freeze_sha256"])
    tf = reader.json(root/"teachers_frozen.json")
    reader.expect(root/"teacher_refs_frozen.json", tf["refs_sha256"])
    for group in ("selections", "races"):
        for name, digest in tf[group].items():
            reader.expect(relative_input(root, name), digest)
    for name, digest in frozen["choices"].items():
        reader.expect(relative_input(root, name), digest)
    require(len(frozen["choices"]) == (18 if historical else 90), "Selection freeze incomplete")
    require(len(tf["selections"]) == (9 if historical else 45), "Teacher freeze incomplete")
    expected = {selection_path(root, stage, dataset, seed, condition, old=historical).relative_to(root).as_posix()
                for dataset in (DATASETS[:3] if historical else DATASETS)
                for seed in ((0,) if historical else range(3)) for condition in CONDITIONS
                for stage in ("teacher", "student")}
    require(set(frozen["choices"]) == expected, "Frozen selection path matrix differs from approved scope")
    require(set(tf["selections"]) == {p for p in expected if p.startswith("selections/teacher/")},
            "Frozen teacher selection paths differ from approved scope")
    refs = reader.json(root/"refs.json")
    teacher_refs = reader.json(root/"teacher_refs_frozen.json")
    require(teacher_refs == {k: v for k, v in refs.items() if v["kind"] == "teacher"},
            "Teacher references changed after freeze")
    if "historical_manifest_sha256" in tf:
        reader.expect(root/"historical_manifest.json", tf["historical_manifest_sha256"])
    return frozen, refs


def check_choices(root, reader, refs):
    records = []
    for dataset in DATASETS:
        for seed in range(3):
            for condition in CONDITIONS:
                teacher = reader.json(selection_path(root, "teacher", dataset, seed, condition))
                student = reader.json(selection_path(root, "student", dataset, seed, condition))
                require(student["teacher_choices"] == teacher["choices"], "Student teacher choice drift")
                for stage, selection in (("teacher", teacher), ("student", student)):
                    require(set(selection["choices"]) == set(METHODS), "Missing logical route")
                    require((selection["dataset"], selection["seed"], selection["condition"]) ==
                            (dataset, seed, condition), "Selection cell identity mismatch")
                    for method, key in selection["choices"].items():
                        ref = refs[key]
                        require((ref["dataset"], ref["seed"], ref["condition"], ref["kind"]) ==
                                (dataset, seed, condition, stage), "Cross-cell/stage alias: "+key)
                        require(key == cell_id(dataset, seed, condition)+f"/{stage}/"+ref["arm"], "Invalid ref key")
                        require(ref["historical"] is historical(dataset, seed), "Historical marker mismatch")
                        for field in ("folder", "target_path"):
                            require(Path(ref[field]).is_absolute(), "Expected absolute "+field)
                        records.append(dict(dataset=dataset, seed=seed, condition=condition,
                                            stage=stage, method=method, ref=key, **{"reference": ref}))
    require(len(records) == 540, "Expected 540 logical endpoints")
    return records


def metrics(probabilities, labels, index, classes):
    p, y, index = np.asarray(probabilities), np.asarray(labels), np.asarray(index, dtype=np.int64)
    require(p.shape == (len(y), classes) and len(index) > 0, "Invalid metric shape")
    require(np.isfinite(p).all() and (p >= -1e-7).all() and (p <= 1+1e-7).all(), "Invalid probabilities")
    require(np.allclose(p.sum(1), 1, atol=1e-6, rtol=0), "Unnormalized probabilities")
    pred, truth = p[index].argmax(1), y[index].astype(np.int64)
    matrix = np.bincount(truth*classes+pred, minlength=classes**2).reshape(classes, classes)
    den = matrix.sum(0)+matrix.sum(1)
    f1 = np.divide(2*np.diag(matrix), den, out=np.zeros(classes), where=den > 0)
    recalls = [dict(class_id=k, count=int(matrix[k].sum()),
                    recall=float(matrix[k, k]/matrix[k].sum()) if matrix[k].sum() else None) for k in range(classes)]
    return dict(accuracy=float(np.mean(pred == truth)), macro_f1=float(f1.mean())), recalls


def summarize(rows, ids, metric_names, seeds=(0, 1, 2)):
    """Average cells within each seed, then mean/sample-SD over seeds."""
    buckets = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        identity = tuple(row[k] for k in ids)+(int(row["seed"]),)
        require(identity not in seen, "Duplicate seed cell: "+str(identity))
        seen.add(identity)
        variants = [row, dict(row, dataset="ALL")]
        if row["condition"] != "clean":
            variants += [dict(row, condition="NOISY_MACRO"), dict(row, dataset="ALL", condition="NOISY_MACRO")]
        for variant in variants:
            buckets[tuple(variant[k] for k in ids)][int(row["seed"])].append(row)
    out = []
    for key, by_seed in buckets.items():
        require(set(by_seed) == set(seeds), "Missing seed for "+str(key))
        cells = [{(r["dataset"], r["condition"]) for r in by_seed[s]} for s in seeds]
        require(all(c == cells[0] for c in cells), "Unbalanced seed cells")
        row = dict(zip(ids, key), n_seeds=len(seeds), cells_per_seed=len(cells[0]), ddof=1,
                   aggregation="equal conditions within seed, equal graphs within seed; sample SD across seeds")
        for metric in metric_names:
            values = [statistics.mean(float(r[metric]) for r in by_seed[s]) for s in seeds]
            require(all(math.isfinite(v) for v in values), "Nonfinite statistic")
            row.update({f"{metric}_seed{s}": v for s, v in zip(seeds, values)})
            row[metric+"_mean"] = statistics.mean(values)
            row[metric+"_std"] = statistics.stdev(values) if len(values) > 1 else None
        out.append(row)
    return out


def comparisons(rows, controls=CONTROLS, methods=METHODS):
    index = {(r["dataset"], int(r["seed"]), r["condition"], r["stage"], r["method"]): r for r in rows}
    require(len(index) == len(rows), "Duplicate comparison endpoint")
    cells = sorted({k[:3] for k in index})
    pairs, gaps = [], []
    for stage in ("teacher", "student"):
        for dataset, seed, condition in cells:
            base = dict(dataset=dataset, seed=seed, condition=condition)
            def get(method, which=stage):
                key = (dataset, seed, condition, which, method)
                require(key in index, "Missing matched endpoint: "+str(key))
                return index[key]
            a = get("JOINT")
            for control in controls:
                b = get(control)
                pairs.append(dict(base, stage=stage, method="JOINT", reference=control,
                                  **{m+"_diff_pp": 100*(float(a[m])-float(b[m])) for m in METRICS}))
            if "FIXED_D1" in methods and "DEEP_S32" in methods:
                a, b = get("FIXED_D1"), get("DEEP_S32")
                pairs.append(dict(base, stage=stage, method="FIXED_D1", reference="DEEP_S32",
                                  **{m+"_diff_pp": 100*(float(a[m])-float(b[m])) for m in METRICS}))
            if stage == "teacher":
                for method in methods:
                    a, b = get(method, "student"), get(method, "teacher")
                    gaps.append(dict(base, method=method,
                        student_minus_teacher_accuracy_pp=100*(float(a["test_accuracy"])-float(b["test_accuracy"])),
                        student_minus_teacher_macro_f1_pp=100*(float(a["test_macro_f1"])-float(b["test_macro_f1"]))))
    return pairs, gaps


def data_paths(workspace, reader):
    """Read common.DATA_FILES without executing common or its import side effects."""
    source = workspace/"experiments/new_graph_mlp_seed0_20261009/common.py"
    reader.hash(source)
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    expr = next(n.value for n in tree.body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == "DATA_FILES" for t in n.targets))
    def path_expr(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id == "PROJECT":
            return workspace.parents[1]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return Path(path_expr(node.left))/path_expr(node.right)
        raise ValueError("Unsupported DATA_FILES expression")
    require(isinstance(expr, ast.Dict), "Expected DATA_FILES dictionary")
    return {ast.literal_eval(k): Path(path_expr(v)) for k, v in zip(expr.keys, expr.values)}


def geometry_path(workspace, dataset, seed):
    return workspace/"experiments/sc_support_channel_pilot_20261023/diagnostics/e1"/dataset/f"seed{seed}/geometry.pt"


def raw_graph(path, reader):
    raw = reader.tensor(path, objects=True)[0]
    return tuple((raw[k] if isinstance(raw, dict) else getattr(raw, k)).cpu()
                 for k in ("x", "y", "edge_index"))


def source_freeze(root, reader):
    files = [CODE/"analysis.py", CODE/"audit.py", CODE/"tests/test_analysis.py"]
    value = {str(p.resolve()): reader.hash(p) for p in files}
    path = root/"postprocessing_sources.json"
    if path.exists():
        require(reader.json(path) == value, "Postprocessing sources changed; preserve prior run before versioning")
    else:
        save(path, value)
        reader.hash(path)


def costs(root, old, reader, records):
    physical = reader.json(root/"physical_fits.json")
    oldphysical = reader.json(old/"physical_fits.json")
    manifest = reader.json(root/"historical_manifest.json")
    require(manifest["physical_manifest"] == oldphysical, "Historical physical cost manifest drift")
    require(manifest["cells"] == 9 and manifest["logical_students"] == 54, "Historical scope mismatch")
    out = []
    for dataset in DATASETS:
        for seed in range(3):
            for condition in CONDITIONS:
                oldcell = historical(dataset, seed)
                source = oldphysical if oldcell else physical
                fits = [v for v in source.values() if (v["dataset"], int(v.get("seed", 0)), v["condition"]) ==
                        (dataset, seed, condition)]
                row = dict(dataset=dataset, seed=seed, condition=condition, historical=oldcell,
                           student_logical=6, teacher_selected_logical=6, candidate_teacher_logical=45,
                           new_teacher_candidate_trajectories=0 if oldcell else 45,
                           new_student_logical=0 if oldcell else 6)
                for kind in ("teacher", "student"):
                    subset = [f for f in fits if f["kind"] == kind]
                    row[kind+"_physical"] = len(subset)
                    row[kind+"_updates"] = sum(f["epoch"] for f in subset)
                    row[kind+"_seconds"] = sum(f["seconds"] for f in subset)
                folders = {Path(r["reference"]["folder"]) for r in records if
                           (r["dataset"], r["seed"], r["condition"]) == (dataset, seed, condition)}
                row["selected_artifact_bytes"] = sum(p.stat().st_size for f in folders for p in f.rglob("*") if p.is_file())
                row["joint_race_candidate_updates"] = 2250
                row["joint_with_ce_updates"] = 2450
                row["fixed_expanded_with_ce_updates"] = 1600
                out.append(row)
    require(sum(r["teacher_physical"] for r in out if not r["historical"]) == 1620, "Expected 1620 new teachers")
    return out


def selected_spec(arm):
    if arm in ("CE", "OLD64", "LEGACY_SUPPORT32"):
        return dict(requested_support=32 if arm == "LEGACY_SUPPORT32" else None,
                    ball_count_cap=64 if arm == "OLD64" else None,
                    strength_kind="observed" if arm == "CE" else "uniform", uniform_strength=0. if arm == "CE" else 1.)
    prefix, suffix = arm.split("_", 1)
    out = dict(requested_support=int(prefix[1:]), ball_count_cap=None)
    if suffix.startswith("L"):
        out.update(strength_kind="uniform", uniform_strength=float(suffix[1:]))
    else:
        _, agree, conflict = suffix.split("_")
        out.update(strength_kind="group", agree_strength=float(agree[1:]), conflict_strength=float(conflict[1:]), tie_strength=1.)
    return out


def tex_table(path, rows, ids, metrics_to_show):
    def escape(value):
        return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
    lines = [r"% n=3; sample SD (ddof=1). Macro averages are computed inside each seed.",
             r"\begin{longtable}{"+"l"*len(ids)+"r"*len(metrics_to_show)+"}",
             " & ".join(escape(k) for k in (*ids, *metrics_to_show))+r" \\", r"\hline"]
    for r in rows:
        values = [escape(r[k]) for k in ids]
        for metric in metrics_to_show:
            scale = 1 if metric.endswith("_pp") else 100
            values.append(f"${scale*r[metric+'_mean']:.3f} \\pm {scale*r[metric+'_std']:.3f}$")
        lines.append(" & ".join(values)+r" \\")
    lines.append(r"\end{longtable}")
    Path(path).write_text("\n".join(lines)+"\n", encoding="utf-8")


def report(root, summaries, paired, cost_rows):
    lines = ["# GCN SUPPORT/D1/C1 三 seed 扩展", "",
             "45 cells，五图 x seed0/1/2 x clean/U.4/U.6；六路教师与同 seed DIST 学生。",
             "固定历史 GRACE 几何；旧三图 seed0 九格只读复用，其余 36 格新拟合。",
             "全部选择冻结后测试；宏平均先 seed 内平均条件和图，再对三个 seed 求均值与样本 SD(ddof=1)。",
             "所有 seed 原值、负差、clean、逐类召回、配置身份、成本保留于 CSV。", "",
             "## 跨图带噪结果", "", "|阶段|方法|accuracy % mean +/- SD|macro-F1 % mean +/- SD|",
             "|---|---|---:|---:|"]
    for r in summaries:
        if r["dataset"] == "ALL" and r["condition"] == "NOISY_MACRO":
            lines.append(f"|{r['stage']}|{r['method']}|{100*r['test_accuracy_mean']:.3f} +/- {100*r['test_accuracy_std']:.3f}|"
                         f"{100*r['test_macro_f1_mean']:.3f} +/- {100*r['test_macro_f1_std']:.3f}|")
    lines += ["", "## 配对增量及负结果", "", "|图|条件|阶段|参照|accuracy pp mean +/- SD|seed0/1/2 pp|",
              "|---|---|---|---|---:|---|"]
    for r in paired:
        if r["method"] == "JOINT" and r["condition"] in ("clean", "NOISY_MACRO"):
            m = "test_accuracy_diff_pp"
            seeds = "/".join(f"{r[f'{m}_seed{s}']:+.3f}" for s in range(3))
            lines.append(f"|{r['dataset']}|{r['condition']}|{r['stage']}|{r['reference']}|"
                         f"{r[m+'_mean']:+.3f} +/- {r[m+'_std']:.3f}|{seeds}|")
    lines += ["", "## 成本与判读", "",
              f"新教师轨迹 {sum(r['teacher_physical'] for r in cost_rows if not r['historical'])}；"
              f"新物理学生 {sum(r['student_physical'] for r in cost_rows if not r['historical'])}；逻辑学生 270(旧54/新216)。",
              f"新拟合累计更新 {sum(r['teacher_updates']+r['student_updates'] for r in cost_rows if not r['historical'])}。"
              "epochs 为各物理轨迹最终累计轮数，前缀与续训合计一次。",
              "成本表将历史复用与新拟合分列；计时沿用 fit.seconds，存储另列选中产物与全运行目录总量。",
              "已支持结论：以上表格及逐 seed 配对差；教师增量与学生传递分别判读，负差完整呈现。",
              "机制解释：区域监督可能改变教师概率结构与蒸馏效果，该解释仍需机制证据。",
              "尚待验证：跨未开发划分、更多随机种子及其他 SSL 几何的稳定性。",
              "负结果削弱方向：逐图/条件/seed 负差直接记录于 paired_differences.csv，保留其参照身份。",
              "最终决策：保留本次全部结果并停止自动扩展；结合教师、学生、clean、F1、逐类召回及成本表做下一步研究决策。",
              "CPU 审计单独执行，最终状态见 complete.json。"]
    (root/"REPORT_zh.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def analyze(root=None, workspace=WORKSPACE, threads=4):
    config = read(CODE/"config.json")
    validate_config(config)
    root = Path(root or config["runtime"]).resolve()
    workspace = Path(workspace).resolve()
    old = workspace/"experiments"/config["historical"]
    reader = Reader()
    require(not (root/"run.lock").exists(), "Fitting lock still held")
    require(reader.json(root/"fits_complete.json")["status"] == "FITTING_COMPLETE_TEST_DEFERRED", "Fitting incomplete")
    frozen, refs = check_freeze(root, reader)
    check_freeze(old, reader, historical=True)
    records = check_choices(root, reader, refs)
    source_freeze(root, reader)
    import torch
    torch.set_num_threads(threads)
    started = now()
    marker = dict(started=started, selection_freeze_sha256=reader.hash(root/"selections_frozen.json"), device="cpu")
    if (root/"testing_started.json").exists():
        previous = reader.json(root/"testing_started.json")
        require(previous["selection_freeze_sha256"] == marker["selection_freeze_sha256"], "Test freeze changed")
    else:
        save(root/"testing_started.json", marker)
    require(read(root/"testing_started.json")["started"] >= frozen["frozen"], "Testing before final freeze")
    paths = data_paths(workspace, reader)
    rows, recalls = [], []
    for dataset in DATASETS:
        _, y, _ = raw_graph(paths[dataset], reader)
        y = y.numpy()
        classes = int(y.max())+1
        for seed in range(3):
            geo = reader.tensor(geometry_path(workspace, dataset, seed))
            cache = {}
            for endpoint in records:
                if (endpoint["dataset"], endpoint["seed"]) != (dataset, seed):
                    continue
                ref = endpoint["reference"]
                folder = Path(ref["folder"])
                reader.expect(folder/"result.json", ref["result_sha256"])
                result = reader.json(folder/"result.json")
                reader.expect(folder/"probabilities.pt", result["probabilities_sha256"])
                require(result["epoch"] == 200 and result["status"] == "complete", "Selected prefix endpoint")
                if str(folder) not in cache:
                    cache[str(folder)] = reader.tensor(folder/"probabilities.pt").numpy()
                p = cache[str(folder)]
                test, per_class = metrics(p, y, geo["test"].numpy(), classes)
                valid, _ = metrics(p, y, geo["valid"].numpy(), classes)
                require(abs(valid["accuracy"]-result["validation"]) < 1e-12, "Validation probability drift")
                row = {k: v for k, v in endpoint.items() if k != "reference"}
                row.update(folder=str(folder), historical=ref["historical"],
                           test_accuracy=test["accuracy"], test_macro_f1=test["macro_f1"],
                           validation=valid["accuracy"], valid_macro_f1=valid["macro_f1"],
                           valid_nll=result["valid_nll"], best_epoch=result["best_epoch"], seconds=result["seconds"],
                           arm=ref["arm"], partition=ref["partition"], target_hash=result["target_hash"],
                           probabilities_sha256=result["probabilities_sha256"], initial_hash=result["initial_hash"])
                if endpoint["stage"] == "student":
                    row["teacher_probability_sha256"] = result["teacher_probability_sha256"]
                    teacher_choices = reader.json(selection_path(root, "teacher", dataset, seed, endpoint["condition"]))["choices"]
                    teacher_ref = refs[teacher_choices[endpoint["method"]]]
                else:
                    teacher_ref = ref
                row["teacher_arm"] = teacher_ref["arm"]
                row.update(selected_spec(teacher_ref["arm"]))
                rows.append(row)
                for rec in per_class:
                    recalls.append({**{k: row[k] for k in ("dataset", "seed", "condition", "stage", "method")}, **rec})
        print(f"ANALYSIS {dataset}: {sum(r['dataset']==dataset for r in rows)} endpoints", flush=True)
    pairs, gaps = comparisons(rows)
    ids = ("dataset", "condition", "stage", "method")
    summary = summarize(rows, ids, METRICS)
    paired_summary = summarize(pairs, (*ids, "reference"), tuple(m+"_diff_pp" for m in METRICS))
    gap_metrics = ("student_minus_teacher_accuracy_pp", "student_minus_teacher_macro_f1_pp")
    gap_summary = summarize(gaps, ("dataset", "condition", "method"), gap_metrics)
    cost_rows = costs(root, old, reader, records)
    outputs = dict(selected_results=rows, summary=summary, paired_differences=pairs,
                   paired_summary=paired_summary, transfer_gaps=gaps, gap_summary=gap_summary,
                   per_class_recall=recalls, costs=cost_rows)
    for name, values in outputs.items():
        csv_save(root/(name+".csv"), values)
    tex_table(root/"summary.tex", summary, ids, METRICS[:2])
    tex_table(root/"paired_summary.tex", paired_summary, (*ids, "reference"), ("test_accuracy_diff_pp", "test_macro_f1_diff_pp"))
    tex_table(root/"gap_summary.tex", gap_summary, ("dataset", "condition", "method"), gap_metrics)
    report(root, summary, paired_summary, cost_rows)
    reader.verify()
    save(root/"postprocessing_inputs.json", reader.inputs)
    filenames = [*(name+".csv" for name in outputs), "summary.tex", "paired_summary.tex", "gap_summary.tex", "REPORT_zh.md"]
    storage = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    done = dict(status="ANALYSIS_COMPLETE", finished=now(), device="cpu", selected=540, student_logical=270,
                n_seeds=3, ddof=1, test_after_freeze=True, runtime_bytes=storage,
                source_manifest_sha256=sha(root/"postprocessing_sources.json"),
                input_manifest_sha256=sha(root/"postprocessing_inputs.json"),
                outputs={name: sha(root/name) for name in filenames})
    save(root/"analysis_complete.json", done)
    if root != CODE:
        for name in ("REPORT_zh.md", "summary.csv", "summary.tex", "paired_summary.tex", "gap_summary.tex", "analysis_complete.json"):
            (CODE/name).write_bytes((root/name).read_bytes())
    print(json.dumps(done, indent=2), flush=True)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--runtime", type=Path, default=None)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    analyze(args.root, args.workspace, args.threads)


if __name__ == "__main__":
    main()
