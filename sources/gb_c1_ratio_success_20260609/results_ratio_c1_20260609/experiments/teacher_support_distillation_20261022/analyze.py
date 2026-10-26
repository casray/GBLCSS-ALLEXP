from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORKSPACE / "GB-LCSS_GitHub" / "src"))
sys.path.insert(0, str(WORKSPACE / "experiments" / "new_graph_mlp_seed0_20261009"))

import run as runner  # noqa: E402
from common import raw_data  # noqa: E402
from gblcss.models import MLP  # noqa: E402

CONFIG = runner.CONFIG
ARMS = CONFIG["teacher_arms"]
CONTRASTS = (("SUPPORT_LCSS", "OLD_LCSS"), ("SUPPORT_LCSS", "CE"), ("OLD_LCSS", "CE"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def make_result_map(rows: list[dict]) -> dict:
    result = {}
    for row in rows:
        key = (row["dataset"], int(row["seed"]), row["condition"], row["teacher_arm"])
        if key in result:
            raise RuntimeError(f"Duplicate student result: {key}")
        if row["status"] != "complete":
            raise RuntimeError(f"Incomplete student result: {key}")
        result[key] = row
    return result


def audit_reload(result_map: dict) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audit = []
    active_dataset = None
    raw = None
    for key in sorted(result_map):
        dataset, seed, condition, arm = key
        if dataset != active_dataset:
            raw = raw_data(dataset)
            active_dataset = dataset
        folder = ROOT / "runs" / dataset / f"seed{seed}" / condition / arm
        record = result_map[key]
        pack = torch.load(runner.protocol_cell(dataset, seed, condition), map_location="cpu", weights_only=False)
        valid = pack["valid_index"].long()
        test = pack["test_index"].long()
        log = [json.loads(line) for line in (folder / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        max_valid = max(float(item["valid_accuracy"]) for item in log)
        expected_epoch = next(int(item["epoch"]) for item in log if float(item["valid_accuracy"]) == max_valid)
        if expected_epoch != int(record["best_epoch"]):
            raise RuntimeError(f"Checkpoint selection mismatch for {key}: {expected_epoch} vs {record['best_epoch']}")
        if abs(max_valid - float(record["best_valid_accuracy"])) > 1e-8:
            raise RuntimeError(f"Validation metric mismatch for {key}")

        x = raw.x.float().to(device)
        y = raw.y.long().to(device)
        classes = int(y.max().item()) + 1
        checkpoint = torch.load(folder / "student.pt", map_location="cpu", weights_only=False)
        model = MLP(x.shape[1], 64, classes, 2, 0.5).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            probabilities = model(x).softmax(1).cpu()
        saved = torch.load(folder / "probabilities.pt", map_location="cpu", weights_only=True).float()
        max_error = float((probabilities - saved).abs().max().item())
        if max_error > 1e-6:
            raise RuntimeError(f"Checkpoint reload probability mismatch for {key}: {max_error}")
        prediction = saved.argmax(1)
        recomputed_accuracy = float((prediction[test] == y.cpu()[test]).float().mean().item())
        recomputed_f1 = runner.macro_f1(y.cpu()[test], prediction[test], classes)
        if abs(recomputed_accuracy - float(record["test_accuracy"])) > 1e-6:
            raise RuntimeError(f"Test accuracy mismatch for {key}")
        if abs(recomputed_f1 - float(record["test_macro_f1"])) > 1e-6:
            raise RuntimeError(f"Test macro-F1 mismatch for {key}")
        audit.append({"dataset": dataset, "seed": seed, "condition": condition, "teacher_arm": arm,
                      "selected_epoch_matches_first_max": True, "checkpoint_probability_max_error": max_error,
                      "test_accuracy_recomputed": recomputed_accuracy, "test_macro_f1_recomputed": recomputed_f1})
    return audit


def summarize() -> None:
    runner.verify_frozen()
    expected = runner.expected_keys()
    rows = read_csv(ROOT / "results.csv")
    result_map = make_result_map(rows)
    if set(result_map) != expected:
        raise RuntimeError(f"Expected {len(expected)} student fits; found {len(result_map)}")

    audit = audit_reload(result_map)
    write_csv(ROOT / "reload_audit.csv", audit)

    arm_rows = []
    for arm in ARMS:
        selected = [row for key, row in result_map.items() if key[3] == arm]
        teacher_acc = [float(row["teacher_test_accuracy"]) for row in selected]
        student_acc = [float(row["test_accuracy"]) for row in selected]
        student_f1 = [float(row["test_macro_f1"]) for row in selected]
        arm_rows.append({
            "teacher_arm": arm, "cells": len(selected),
            "teacher_test_accuracy_mean_pct": mean(teacher_acc) * 100,
            "student_test_accuracy_mean_pct": mean(student_acc) * 100,
            "student_test_accuracy_sd_across_cells_pp": sd(student_acc) * 100,
            "student_test_macro_f1_mean_pct": mean(student_f1) * 100,
            "student_minus_teacher_accuracy_mean_pp": mean([s - t for s, t in zip(student_acc, teacher_acc)]) * 100,
            "mean_selected_epoch": mean([int(row["best_epoch"]) for row in selected]),
        })
    write_csv(ROOT / "arm_summary.csv", arm_rows)

    paired_rows = []
    grouped = defaultdict(list)
    for dataset in CONFIG["datasets"]:
        for seed in CONFIG["seeds"]:
            for condition in CONFIG["conditions"]:
                for new_arm, old_arm in CONTRASTS:
                    new = result_map[(dataset, seed, condition, new_arm)]
                    old = result_map[(dataset, seed, condition, old_arm)]
                    if new["initial_state_sha256"] != old["initial_state_sha256"] or new["initial_rng_sha256"] != old["initial_rng_sha256"]:
                        raise RuntimeError(f"Student initialization is not paired: {dataset}/{seed}/{condition}/{new_arm}/{old_arm}")
                    paired = {
                        "dataset": dataset, "seed": seed, "condition": condition,
                        "contrast": f"{new_arm}-{old_arm}",
                        "student_accuracy_delta_pp": (float(new["test_accuracy"]) - float(old["test_accuracy"])) * 100,
                        "student_macro_f1_delta_pp": (float(new["test_macro_f1"]) - float(old["test_macro_f1"])) * 100,
                        "teacher_accuracy_delta_pp": (float(new["teacher_test_accuracy"]) - float(old["teacher_test_accuracy"])) * 100,
                        "best_epoch_new": int(new["best_epoch"]), "best_epoch_old": int(old["best_epoch"]),
                    }
                    paired_rows.append(paired)
                    grouped[(paired["contrast"], dataset, seed, condition)].append(paired)
    write_csv(ROOT / "paired_results.csv", paired_rows)

    contrasts_summary = []
    primary = None
    for new_arm, old_arm in CONTRASTS:
        contrast = f"{new_arm}-{old_arm}"
        selected = [row for row in paired_rows if row["contrast"] == contrast]
        deltas = [float(row["student_accuracy_delta_pp"]) for row in selected]
        f1_deltas = [float(row["student_macro_f1_delta_pp"]) for row in selected]
        item = {
            "contrast": contrast, "cells": len(selected),
            "mean_student_accuracy_delta_pp": mean(deltas), "sd_student_accuracy_delta_pp": sd(deltas),
            "positive_tie_negative_cells": f"{sum(v > 1e-12 for v in deltas)}/{sum(abs(v) <= 1e-12 for v in deltas)}/{sum(v < -1e-12 for v in deltas)}",
            "mean_student_macro_f1_delta_pp": mean(f1_deltas),
            "mean_teacher_accuracy_delta_pp": mean([float(row["teacher_accuracy_delta_pp"]) for row in selected]),
        }
        for seed in CONFIG["seeds"]:
            seed_values = [float(row["student_accuracy_delta_pp"]) for row in selected if int(row["seed"]) == seed]
            item[f"seed{seed}_mean_delta_pp"] = mean(seed_values)
        for dataset in CONFIG["datasets"]:
            values = [float(row["student_accuracy_delta_pp"]) for row in selected if row["dataset"] == dataset]
            item[f"{dataset}_mean_delta_pp"] = mean(values)
        for condition in CONFIG["conditions"]:
            values = [float(row["student_accuracy_delta_pp"]) for row in selected if row["condition"] == condition]
            item[f"{condition}_mean_delta_pp"] = mean(values)
        contrasts_summary.append(item)
        if contrast == "SUPPORT_LCSS-OLD_LCSS":
            primary = item
    write_csv(ROOT / "paired_summary.csv", contrasts_summary)

    cell_rows = []
    for dataset in CONFIG["datasets"]:
        for condition in CONFIG["conditions"]:
            for arm in ARMS:
                selected = [result_map[(dataset, seed, condition, arm)] for seed in CONFIG["seeds"]]
                acc = [float(row["test_accuracy"]) * 100 for row in selected]
                f1 = [float(row["test_macro_f1"]) * 100 for row in selected]
                cell_rows.append({"dataset": dataset, "condition": condition, "teacher_arm": arm,
                                  "student_accuracy_mean_pct": mean(acc), "student_accuracy_sd_pp": sd(acc),
                                  "student_macro_f1_mean_pct": mean(f1), "student_macro_f1_sd_pp": sd(f1)})
    write_csv(ROOT / "cell_summary.csv", cell_rows)

    gate = CONFIG["success_gate"]
    seed_means = [float(primary[f"seed{seed}_mean_delta_pp"]) for seed in CONFIG["seeds"]]
    dataset_means = [float(primary[f"{dataset}_mean_delta_pp"]) for dataset in CONFIG["datasets"]]
    decision = {
        "primary_contrast": "SUPPORT_LCSS-OLD_LCSS",
        "mean_accuracy_gain_pp": primary["mean_student_accuracy_delta_pp"],
        "seed_means_pp": dict(zip((f"seed{s}" for s in CONFIG["seeds"]), seed_means)),
        "dataset_means_pp": dict(zip(CONFIG["datasets"], dataset_means)),
        "threshold_mean_gain_pp": gate["mean_support_minus_old_accuracy_pp_at_least"],
        "positive_seed_means_required": gate["positive_seed_means_required"],
        "positive_seed_means_observed": sum(value > 0 for value in seed_means),
        "positive_dataset_means_required": gate["positive_dataset_means_required"],
        "positive_dataset_means_observed": sum(value > 0 for value in dataset_means),
        "gate_passed": (
            primary["mean_student_accuracy_delta_pp"] >= gate["mean_support_minus_old_accuracy_pp_at_least"]
            and sum(value > 0 for value in seed_means) >= gate["positive_seed_means_required"]
            and sum(value > 0 for value in dataset_means) >= gate["positive_dataset_means_required"]
        ),
        "interpretation_limit": "descriptive screening only; 3 protocol seeds, no significance claim; no clean-label condition",
    }
    runner.write_json(ROOT / "decision.json", decision)

    support_arm = next(row for row in arm_rows if row["teacher_arm"] == "SUPPORT_LCSS")
    old_arm = next(row for row in arm_rows if row["teacher_arm"] == "OLD_LCSS")
    report = [
        "# SUPPORT-LCSS 教师蒸馏实验结论",
        "",
        "## 实验范围",
        "",
        f"完成 {len(result_map)} 个 MLP 学生拟合：5 个数据集 × 3 个协议 seed × 3 种噪声条件 × 3 个教师臂。教师输出复用自 `teacher_support_lcss_20261022`，没有重训教师。学生只输入节点属性；使用全节点、温度 1 的软交叉熵，训练 200 轮，并按干净验证集首次最大准确率选模。",
        "",
        "主比较是 SUPPORT-LCSS 教师蒸馏学生减去旧 LCSS 教师蒸馏学生。CE 教师蒸馏为参照。每个配对使用相同学生初始化和 RNG。",
        "",
        "## 主要结果",
        "",
        f"- SUPPORT 蒸馏学生平均测试准确率：{support_arm['student_test_accuracy_mean_pct']:.3f}%；对应旧 LCSS：{old_arm['student_test_accuracy_mean_pct']:.3f}%。",
        f"- SUPPORT − 旧 LCSS 学生准确率：{primary['mean_student_accuracy_delta_pp']:+.3f} ± {primary['sd_student_accuracy_delta_pp']:.3f} pp（45 个配对单元；描述性标准差）。胜/平/负：{primary['positive_tie_negative_cells']}。",
        f"- 学生 macro-F1 配对差：{primary['mean_student_macro_f1_delta_pp']:+.3f} pp；教师准确率配对差：{primary['mean_teacher_accuracy_delta_pp']:+.3f} pp。",
        "- 三个 seed 的学生准确率配对均值（pp）：" + ", ".join(f"seed{s} {primary[f'seed{s}_mean_delta_pp']:+.3f}" for s in CONFIG["seeds"]) + ".",
        "- 按数据集平均的学生准确率配对差（pp）：" + ", ".join(f"{d} {primary[f'{d}_mean_delta_pp']:+.3f}" for d in CONFIG["datasets"]) + ".",
        "",
        "## 判定与边界",
        "",
        f"预设实用门槛{'通过' if decision['gate_passed'] else '未通过'}：总体均值需至少 +{gate['mean_support_minus_old_accuracy_pp_at_least']:.1f} pp、seed 均值至少 {gate['positive_seed_means_required']}/{gate['seed_count']} 为正、数据集均值至少 {gate['positive_dataset_means_required']}/{gate['dataset_count']} 为正。",
        "这只回答固定教师概率下，SUPPORT-LCSS 是否更适合蒸馏；不能替代教师端准确率，也不能推出 SUPPORT 算法本身带来普遍提升。不同条件和数据集仍需结合 `paired_results.csv`、`cell_summary.csv` 查看。三个协议 seed 不构成显著性检验；本实验未包含 clean 条件，不能据此声称没有干净数据损害。",
        "",
        "## 审计",
        "",
        f"检查了 {len(audit)} 个学生检查点：首次最大验证轮次匹配，模型重载概率最大误差不超过 1e-6，测试准确率与 macro-F1 均由保存概率独立复算。配对初始化哈希一致性在汇总时逐单元检查。",
        "",
        "## 文件",
        "",
        "- `results.csv`: 每个拟合的结果与配置指纹。",
        "- `paired_results.csv`, `paired_summary.csv`: 配对差及总体/seed/数据集/噪声分组结果。",
        "- `arm_summary.csv`, `cell_summary.csv`: 教师与学生指标。",
        "- `reload_audit.csv`: checkpoint、选模及指标复核。",
        "- `decision.json`: 预设门槛的机械判定。",
    ]
    (ROOT / "REPORT_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "primary": primary, "reload_audits": len(audit)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    summarize()
