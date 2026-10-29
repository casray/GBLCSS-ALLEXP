"""Read-only audit of completed artifacts and label-free midpoint diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(root):
    started = datetime.now().isoformat(timespec="seconds")
    receipt = read(root / "receipt.json")
    receipt_hash = digest(root / "receipt.json")
    progress = read(root / "formal/completed.json")
    problems = []
    checked = 0

    def check(path, expected):
        nonlocal checked
        checked += 1
        if not path.is_file():
            problems.append({"path": str(path), "issue": "missing"})
        elif digest(path) != expected:
            problems.append({"path": str(path), "issue": "hash mismatch"})

    for name, expected in receipt["signature"].items():
        check(Path(name), expected)

    baselines, legacy, gradients, weights = [], [], [], []
    head_count = 0
    endpoint_keys = set()
    for task in progress["completed"]:
        d, seed, arm = task["dataset"], task["seed"], task["arm"]
        key = (d, seed, arm)
        if key in endpoint_keys:
            problems.append({"task": task, "issue": "duplicate completed endpoint"})
        endpoint_keys.add(key)
        ssl = root / "formal/ssl" / d / f"seed{seed}" / arm
        folders = [ssl]
        for condition in receipt["config"]["conditions"]:
            for head in (["CE", "fixed"] if arm == "ecl" else ["CE", "fixed", "own"]):
                folders.append(root / "formal/heads" / d / f"seed{seed}" / arm / condition / head)
                head_count += 1
        for folder in folders:
            manifest_path = folder / "complete.json"
            if not manifest_path.is_file():
                problems.append({"path": str(manifest_path), "issue": "missing commit marker"})
                continue
            manifest = read(manifest_path)
            if manifest["status"] != "complete":
                problems.append({"path": str(manifest_path), "issue": "invalid status"})
            for name, expected in manifest["files"].items():
                check(folder / name, expected)
        if arm == "ecl" and d != "roman":
            a = read(ssl / "result.json")["audit"]
            baselines.append({**task, **a})
            if not a["old_state_equal"] or a["old_h_max_error"] > 1e-7:
                problems.append({"task": task, "issue": "legacy SSL replay failed"})
            for folder in folders[1:]:
                a = read(folder / "legacy_replay.json")
                legacy.append(a)
                if not (a["validation_trajectory_equal"] and a["best_epoch_equal"] and a["probability_error"] <= 2e-5):
                    problems.append({"path": str(folder), "issue": "legacy head replay failed"})

    warmup = root / "formal/warmup"
    for path in sorted(warmup.rglob("gradient_diagnostics.json")):
        d, seed = path.relative_to(warmup).parts[:2]
        g = read(path)
        gradients.extend({"dataset": d, "seed": int(seed[4:]), "baseline_gradient_norm": g["baseline_gradient_norm"], **row} for row in g["rows"])
        stats = path.parent / "context/stats.json"
        weights.append({"dataset": d, "seed": int(seed[4:]), **read(stats)})
    return {
        "snapshot_started": started,
        "audit_finished": datetime.now().isoformat(timespec="seconds"),
        "status": "passed" if not problems else "failed",
        "root": str(root), "receipt_sha256": receipt_hash,
        "frozen_source_files": len(receipt["signature"]),
        "total_hash_checks": checked,
        "completed_endpoints": len(progress["completed"]),
        "committed_heads_for_completed_endpoints": head_count,
        "queue_failures": progress["failures"],
        "baseline_replays": baselines,
        "legacy_head_replays": len(legacy),
        "max_legacy_head_probability_error": max((v["probability_error"] for v in legacy), default=None),
        "midpoint_gradients": gradients, "midpoint_weights": weights,
        "live_status_after_audit": read(root / "status.json"),
        "problems": problems,
    }


def report(data):
    lines = ["# GB-ContextECL 运行中核验", "", f"快照时间：{data['snapshot_started']}。此报告读取已完成产物和无标签中点统计；未读取分类分数或真实类别。", "",
             f"端点 {data['completed_endpoints']}/84；已提交端点对应分类头 {data['committed_heads_for_completed_endpoints']}/720。队列记录失败 {len(data['queue_failures'])} 项。",
             f"冻结来源 {data['frozen_source_files']} 项，总计哈希检查 {data['total_hash_checks']} 次，核验状态 {data['status']}。",
             f"历史底座复现 {len(data['baseline_replays'])} 项，历史分类头复现 {data['legacy_head_replays']} 项，预测最大误差 {data['max_legacy_head_probability_error']}。", "",
             "## 实际预热中点的干预梯度", "",
             "下表为 η=1 时，附加项参数梯度范数 / 原始 GraphECL 梯度范数。η=.25 时即时比值为表中四分之一；训练轨迹会随优化变化。", "",
             "|数据|seed|GB %|Cos %|Shuffle %|GB与原梯度余弦|", "|---|---:|---:|---:|---:|---:|"]
    groups = {}
    for row in data["midpoint_gradients"]:
        groups.setdefault((row["dataset"], row["seed"]), {})[row["kind"]] = row
    for (d, s), group in groups.items():
        ratios = [100 * group[k]["gradient_ratio_at_eta1"] for k in ["gb", "cos", "shuffle"]]
        lines.append(f"|{d}|{s}|{ratios[0]:.6f}|{ratios[1]:.6f}|{ratios[2]:.6f}|{group['gb']['gradient_cosine']:.6f}|")
    lines += ["", "## 当前判读", "",
              "实际干预强度随图和权重构造变化。同一 η 约束最大可能偏移，仍需结合实际梯度大小解释 GB 与 Cos 的结果。Shuffle 保留每个节点的权重多重集，用于检验权重与边的对应关系；参数梯度也受对应关系影响。",
              "CiteSeer 的 Cos 行内偏移很小，GB 附加梯度方向在两个 seed 上均与原梯度呈负夹角余弦。此处记录局部优化方向，分类价值由冻结方案的完整对照结果判断。",
              "维持本轮配置和训练预算；待拟合完成后联合查看 fixed/own SUPPORT、CE、目标 Brier、逐图逐噪声和两 seed 配对变化，再决定是否设计匹配实际强度的后续对照。", "",
              "## 当前队列位置", "", "```json", json.dumps(data["live_status_after_audit"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:\codex_temp\graphecl_gb_context_20261028"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "progress_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output / "PROGRESS_zh.md").write_text(report(result), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "completed_endpoints", "committed_heads_for_completed_endpoints", "frozen_source_files", "total_hash_checks", "legacy_head_replays", "problems", "live_status_after_audit"]}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
