"""Score the fixed R2 + SUPPORT + LCSS grid after all fits are complete."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

import run

ROOT = run.ROOT
CONFIG = run.CONFIG


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def log_best_epoch(folder):
    rows = [json.loads(line) for line in (folder / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = CONFIG["model"]["head_epochs"]
    if len(rows) != expected or [row["epoch"] for row in rows] != list(range(1, expected + 1)):
        raise RuntimeError(f"Incomplete epoch log: {folder}")
    best = max(row["validation_correct"] for row in rows)
    first = next(row["epoch"] for row in rows if row["validation_correct"] == best)
    return rows, first, best


def cell_folder(name, seed, condition):
    if name in CONFIG["reused_datasets"]:
        return run.old_run_dir(name, seed, condition), "reused"
    return run.new_run_dir(name, seed, condition), "new"


def main():
    if not (ROOT / "progress.json").exists() or read_json(ROOT / "progress.json").get("status") != "completed":
        raise RuntimeError("The 18 new model fits must finish before test scoring")
    run.verify_frozen()
    rows = []
    metric_cache = {}
    for name in CONFIG["datasets"]:
        raw = run.raw_data(name)
        labels = raw.y.long().cpu().numpy()
        classes = int(raw.y.max()) + 1
        for seed in CONFIG["seeds"]:
            split = run.percent_split_indices(len(raw.x), seed)
            test_ids = np.asarray(split[2], dtype=np.int64)
            for condition in CONFIG["conditions"]:
                folder, provenance = cell_folder(name, seed, condition)
                result = read_json(folder / "result.json")
                if result.get("status") != "ok" or result.get("route") != "R2" or result.get("arm") != "SUPPORT":
                    raise RuntimeError(f"Invalid result identity: {folder}")
                if result.get("condition") != condition or result.get("seed") != seed or result.get("dataset") != name:
                    raise RuntimeError(f"Result grid key mismatch: {folder}")
                if run.sha(folder / "probabilities.pt") != result["probabilities_sha256"]:
                    raise RuntimeError(f"Probability hash mismatch: {folder}")
                if run.sha(folder / "student.pt") != result["checkpoint_sha256"]:
                    raise RuntimeError(f"Checkpoint hash mismatch: {folder}")
                history, first, best = log_best_epoch(folder)
                if first != result["best_epoch"] or best != result["valid_correct"]:
                    raise RuntimeError(f"Best-validation selection mismatch: {folder}")
                probabilities = torch.load(folder / "probabilities.pt", map_location="cpu", weights_only=True)
                if probabilities.shape != (len(labels), classes) or not torch.isfinite(probabilities).all():
                    raise RuntimeError(f"Invalid probability tensor: {folder}")
                torch.testing.assert_close(probabilities.sum(1), torch.ones(len(labels)), atol=1e-5, rtol=0)
                prediction = probabilities.argmax(1).numpy()
                accuracy = float((prediction[test_ids] == labels[test_ids]).mean())
                macro_f1 = float(f1_score(
                    labels[test_ids], prediction[test_ids], labels=np.arange(classes),
                    average="macro", zero_division=0
                ))
                row = {
                    "dataset": name, "seed": seed, "condition": condition,
                    "route": "R2", "method": "SUPPORT_LCSS",
                    "accuracy": accuracy, "macro_f1": macro_f1,
                    "validation": result["validation"], "best_epoch": result["best_epoch"],
                    "epochs": result["epochs"], "provenance": provenance,
                    "source_folder": str(folder),
                    "probabilities_sha256": result["probabilities_sha256"],
                    "checkpoint_sha256": result["checkpoint_sha256"]
                }
                rows.append(row)
                metric_cache[(name, seed, condition)] = row
    expected = len(CONFIG["datasets"]) * len(CONFIG["seeds"]) * len(CONFIG["conditions"])
    if len(rows) != expected or expected != CONFIG["expected_cells"]:
        raise RuntimeError(f"Expected {CONFIG['expected_cells']} result cells, found {len(rows)}")
    write_csv(ROOT / "results.csv", rows)

    condition_rows = []
    for name in CONFIG["datasets"]:
        for condition in CONFIG["conditions"]:
            selected = [metric_cache[(name, seed, condition)] for seed in CONFIG["seeds"]]
            acc_mean, acc_std = mean_std([row["accuracy"] * 100 for row in selected])
            f1_mean, f1_std = mean_std([row["macro_f1"] * 100 for row in selected])
            condition_rows.append({
                "dataset": name, "condition": condition, "seeds": len(selected),
                "accuracy_mean_pct": acc_mean, "accuracy_sample_std_pp": acc_std,
                "macro_f1_mean_pct": f1_mean, "macro_f1_sample_std_pp": f1_std
            })
    write_csv(ROOT / "condition_summary.csv", condition_rows)

    dataset_rows, seed_rows = [], []
    for name in CONFIG["datasets"]:
        per_seed = []
        for seed in CONFIG["seeds"]:
            selected = [metric_cache[(name, seed, condition)] for condition in CONFIG["conditions"]]
            seed_row = {
                "dataset": name, "seed": seed,
                "accuracy_mean_over_noisy_conditions_pct": float(np.mean([r["accuracy"] for r in selected]) * 100),
                "macro_f1_mean_over_noisy_conditions_pct": float(np.mean([r["macro_f1"] for r in selected]) * 100)
            }
            seed_rows.append(seed_row)
            per_seed.append(seed_row)
        acc_mean, acc_std = mean_std([row["accuracy_mean_over_noisy_conditions_pct"] for row in per_seed])
        f1_mean, f1_std = mean_std([row["macro_f1_mean_over_noisy_conditions_pct"] for row in per_seed])
        dataset_rows.append({
            "dataset": name, "seeds": len(per_seed), "conditions_per_seed": len(CONFIG["conditions"]),
            "accuracy_mean_pct": acc_mean, "accuracy_sample_std_pp": acc_std,
            "macro_f1_mean_pct": f1_mean, "macro_f1_sample_std_pp": f1_std
        })
    write_csv(ROOT / "seed_summary.csv", seed_rows)
    write_csv(ROOT / "dataset_summary.csv", dataset_rows)

    overall_seed = []
    for seed in CONFIG["seeds"]:
        selected = [metric_cache[(name, seed, condition)]
                    for name in CONFIG["datasets"] for condition in CONFIG["conditions"]]
        overall_seed.append({
            "seed": seed,
            "accuracy_mean_over_15_cells_pct": float(np.mean([r["accuracy"] for r in selected]) * 100),
            "macro_f1_mean_over_15_cells_pct": float(np.mean([r["macro_f1"] for r in selected]) * 100)
        })
    overall = {
        "result_cells": len(rows), "datasets": len(CONFIG["datasets"]),
        "seeds": len(CONFIG["seeds"]), "noisy_conditions": len(CONFIG["conditions"]),
        "new_fits": CONFIG["expected_new_fits"],
        "reused_fits": CONFIG["expected_reused_cells"],
        "accuracy_mean_pct": float(np.mean([r["accuracy_mean_over_15_cells_pct"] for r in overall_seed])),
        "accuracy_seed_sample_std_pp": float(np.std(
            [r["accuracy_mean_over_15_cells_pct"] for r in overall_seed], ddof=1
        )),
        "macro_f1_mean_pct": float(np.mean([r["macro_f1_mean_over_15_cells_pct"] for r in overall_seed])),
        "macro_f1_seed_sample_std_pp": float(np.std(
            [r["macro_f1_mean_over_15_cells_pct"] for r in overall_seed], ddof=1
        )),
        "interpretation": "Descriptive absolute performance; no baseline or clean-training comparison."
    }
    run.write(ROOT / "overall_summary.json", overall)
    report = [
        "# R2 + SUPPORT + LCSS：五数据集结果",
        "",
        "## 协议",
        "",
        "覆盖 Cora、CiteSeer、CoraFull、Coauthor-CS、Amazon-Photo；种子 0–2；噪声条件为 uniform 0.4、uniform 0.6、pair 0.4。没有运行 clean 训练条件，也没有加入 KNN 或 K-means 臂。R2 仍用 clean validation 选择首个最佳 epoch；test 仅在全部模型完成后评分。",
        "",
        f"共 {len(rows)} 个数据集—种子—噪声结果单元：复用已审计的前三个数据集 {CONFIG['expected_reused_cells']} 个 R2+SUPPORT 运行，新训练 Coauthor-CS 与 Amazon-Photo {CONFIG['expected_new_fits']} 个模型，并为新增数据集训练 {CONFIG['expected_new_alignments']} 个无标签 R2 对齐模型。各数据集的 GRACE H 固定复用，没有做独立 GRACE 重预训练。",
        "",
        "## 各噪声条件",
        "",
        "| Dataset | Condition | Accuracy mean ± SD (%) | Macro-F1 mean ± SD (%) |",
        "|---|---|---:|---:|"
    ]
    for row in condition_rows:
        report.append(
            f"| {row['dataset']} | {row['condition']} | "
            f"{row['accuracy_mean_pct']:.2f} ± {row['accuracy_sample_std_pp']:.2f} | "
            f"{row['macro_f1_mean_pct']:.2f} ± {row['macro_f1_sample_std_pp']:.2f} |"
        )
    report.extend([
        "",
        "## 数据集平均",
        "",
        "下表先在每个 seed 内对三种噪声条件取平均，再计算三个 seed 的均值和样本标准差。",
        "",
        "| Dataset | Accuracy mean ± SD (%) | Macro-F1 mean ± SD (%) |",
        "|---|---:|---:|"
    ])
    for row in dataset_rows:
        report.append(
            f"| {row['dataset']} | {row['accuracy_mean_pct']:.2f} ± {row['accuracy_sample_std_pp']:.2f} | "
            f"{row['macro_f1_mean_pct']:.2f} ± {row['macro_f1_sample_std_pp']:.2f} |"
        )
    report.extend([
        "",
        "## 总体描述",
        "",
        f"按每个 seed 跨五个数据集和三种噪声条件的 15 个单元取宏平均，Accuracy 为 "
        f"{overall['accuracy_mean_pct']:.2f} ± {overall['accuracy_seed_sample_std_pp']:.2f}% ，"
        f"Macro-F1 为 {overall['macro_f1_mean_pct']:.2f} ± {overall['macro_f1_seed_sample_std_pp']:.2f}%。",
        "",
        "这是单方法的绝对表现汇总，没有对照臂，不能据此估计 SUPPORT 或 LCSS 的增益，也不能推出它们优于其他分组方式。三个 seed 的 GRACE 表示并非独立预训练；结果主要反映固定 H 下 split/noise、R2 对齐初始化和分类训练的变化。",
        "",
        "## 文件",
        "",
        "- results.csv：45 个逐单元结果及来源。",
        "- condition_summary.csv：每个数据集和噪声条件的 seed 均值与样本标准差。",
        "- dataset_summary.csv、seed_summary.csv、overall_summary.json：跨噪声描述统计。",
        "- new_alignments.csv、geometry.csv、protocol_checks.json、frozen_inputs.json：新增对齐模型、SUPPORT 分区、协议检查与输入哈希。"
    ])
    (ROOT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    audit = {
        "passed": True, "result_cells": len(rows),
        "reused_cells": sum(row["provenance"] == "reused" for row in rows),
        "new_cells": sum(row["provenance"] == "new" for row in rows),
        "best_epoch_logs_checked": len(rows),
        "probability_and_checkpoint_hashes_checked": len(rows),
        "knn_kmeans_or_clean_comparison_rows": 0
    }
    run.write(ROOT / "audit.json", audit)
    print(json.dumps({"audit": audit, "overall": overall}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
