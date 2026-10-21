from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.data import split_name
from noisy_label_gb_grace.smoothing.targets import CONFLICT_GATE_REPAIR_CONFIGS, REPAIR_CONTROL_CONFIGS


DEFAULT_PRETRAIN_SEEDS = {
    "Cora": 39788,
    "CiteSeer": 38108,
}


WEIGHTED_METHODS = {"gb_weighted_ce", "gagb_weighted_ce", "kmeans_weighted_ce", "topk_matched_weighted_ce"}
DEFAULT_METHODS = [
    "ce",
    "gb_weighted_ce",
    "gagb_weighted_ce",
    "kmeans_weighted_ce",
    "topk_matched_weighted_ce",
    "global",
    "kmeans",
    "topk_matched",
    "gb",
    "gb_prior",
    "gagb",
    "gagb_ca_ce",
    "gagb_qb_pred_mix",
    "gagb_qb_teacher_prior",
    "gagb_qb_conflict_gate",
    "gagb_qb_conflict_gate_alpha03",
    "gagb_qb_conflict_gate_alpha06",
    "gagb_qb_conflict_gate_alpha07",
    "gagb_qb_conflict_gate_alpha08",
    "gagb_qb_conflict_gate_alpha09",
    "gagb_qb_conflict_gate_replace_qmix",
    "gagb_class_capped_alpha09",
    "gagb_class_balanced_alpha09",
    "gagb_boundary_safe_alpha09",
    "gagb_replay_random_same_count",
    "gagb_qb_conflict_gate_sharpen_qmix_T05",
    "gagb_qb_conflict_gate_margin_replace",
    "grace_gagb",
    "grace_gagb_emb_only",
    "grace_gagb_topo_only",
    "kmeans_grace_gagb_matched",
    "topk_grace_gagb_train_matched",
    "grace_gagb_shuffled_alpha_matched",
    "grace_gagb_qb_conflict_gate_alpha03",
    "grace_gagb_qb_conflict_gate_alpha07",
    "grace_gagb_qb_conflict_gate_alpha09",
    "grace_gagb_qfilter_alpha07_q50",
    "grace_gagb_qfilter_alpha07_q70",
    "grace_gagb_qweight_alpha07",
    "grace_gagb_qsafe_boundary_alpha09_q50",
    "grace_gagb_qsafe_boundary_alpha09_q70",
    "grace_gagb_qsafe_boundary_alpha07_q50",
    "grace_gagb_qsafe_class_capped_alpha09_q50",
    "grace_gagb_qsafe_class_balanced_alpha09_q50",
    "gagb_qsafe_boundary_alpha09_q50",
    "khop_only_qsafe_boundary_alpha09_q50",
    "grace_gagb_shuffled_qsafe_boundary_alpha09_q50",
    "gagb_qfilter_alpha07_q50",
    "khop_only_qfilter_alpha07_q50",
    "grace_gagb_shuffled_qfilter_alpha07_q50",
    "dcgb",
    "dcgb_emb_only",
    "dcgb_topo_only",
    "kmeans_dcgb_matched",
    "topk_dcgb_train_matched",
    "dcgb_shuffled_alpha_matched",
    "dcgb_qsafe_boundary_alpha09_q50",
    "emb_only_gagb_local",
    "emb_only_gagb_global",
    "khop_only",
    "kmeans_gagb_matched",
    "topk_gagb_train_matched",
    "topk_gagb_allnode_matched",
    "gagb_shuffled_size_only",
    "gagb_shuffled_alpha_matched",
    "gagb_shuffled_alpha09_matched",
    "gagb_shuffled_replace_matched",
    "kmeans_gagb_matched_alpha09",
    "kmeans_gagb_matched_replace",
    "topk_gagb_train_matched_alpha09",
    "topk_gagb_train_matched_replace",
    "khop_only_alpha09",
    "khop_only_replace",
]
METHOD_CHOICES = sorted({*DEFAULT_METHODS, "topk", "topk_gagb_matched"})
REQUIRED_PACKAGES = ["torch", "torch_geometric", "sklearn", "yaml"]


def checkpoint_path(
    dataset: str,
    split_seed: int,
    pretrain_seed: int,
    split_mode: str = "planetoid",
    train_ratio: float | None = None,
    val_ratio: float | None = None,
) -> Path:
    split_token = split_name(split_mode, split_seed, train_ratio, val_ratio)
    return ROOT / "checkpoints" / "grace" / dataset / f"{split_token}_pretrain{pretrain_seed}" / "checkpoint.pt"


def embedding_path(
    dataset: str,
    split_seed: int,
    pretrain_seed: int,
    split_mode: str = "planetoid",
    train_ratio: float | None = None,
    val_ratio: float | None = None,
) -> Path:
    split_token = split_name(split_mode, split_seed, train_ratio, val_ratio)
    return ROOT / "checkpoints" / "embeddings" / dataset / f"{split_token}_pretrain{pretrain_seed}" / "H.pt"


def result_rows_path(
    dataset: str,
    split_seed: int,
    pretrain_seed: int,
    noise_seed: int,
    classifier_seed: int,
    noise_type: str,
    noise_rate: float,
    split_mode: str = "planetoid",
    train_ratio: float | None = None,
    val_ratio: float | None = None,
) -> Path:
    rate_name = f"{noise_rate:g}".replace(".", "p")
    split_token = split_name(split_mode, split_seed, train_ratio, val_ratio)
    return (
        ROOT
        / "results"
        / "gate_runs"
        / dataset
        / f"{noise_type}_{rate_name}"
        / f"{split_token}_pretrain{pretrain_seed}_noise{noise_seed}_clf{classifier_seed}"
        / "result_rows.csv"
    )


def resolve_python(args: argparse.Namespace) -> Path:
    if args.python is not None:
        candidate = args.python.resolve()
        if not args.allow_external_python:
            try:
                candidate.relative_to(ROOT.resolve())
            except ValueError as exc:
                raise SystemExit(
                    f"Python interpreter must be inside this project. Refusing external interpreter: {candidate}\n"
                    "Pass --allow-external-python only when using a controlled local research environment."
                ) from exc
        return candidate
    exe_name = "python.exe" if os.name == "nt" else "python"
    return (ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / exe_name).resolve()


def check_environment(python_path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if not python_path.exists():
        raise SystemExit(
            f"Project-local Python not found: {python_path}\n"
            "Create noisy_label_gb_grace/.venv or pass --python to a Python executable inside this project."
        )
    check_code = "import " + ", ".join(REQUIRED_PACKAGES)
    completed = subprocess.run(
        [str(python_path), "-c", check_code],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "Project-local Python is missing required packages; formal gate was not started.\n"
            f"Python: {python_path}\n"
            f"Required: {', '.join(REQUIRED_PACKAGES)}\n"
            f"Output:\n{completed.stdout}"
        )


def method_alpha_values(method: str, alpha_max: list[float]) -> list[float]:
    config = CONFLICT_GATE_REPAIR_CONFIGS.get(method.lower())
    if config is None and method.lower() in REPAIR_CONTROL_CONFIGS:
        config = CONFLICT_GATE_REPAIR_CONFIGS[REPAIR_CONTROL_CONFIGS[method.lower()]["base_method"]]
    if config is not None:
        return [float(config["alpha_max"])]
    return alpha_max


def expected_result_complete(path: Path, methods: list[str], alpha_max: list[float], expected_meta: dict) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False
    expected = set()
    for method in methods:
        if method == "ce":
            expected.add(("ce", "0"))
        elif method in WEIGHTED_METHODS:
            expected.add((method, "0"))
        else:
            for alpha in method_alpha_values(method, alpha_max):
                expected.add((method, f"{float(alpha):g}"))
    observed = set()
    for row in rows:
        if not row.get("method"):
            continue
        metadata_matches = True
        for key, expected_value in expected_meta.items():
            actual_value = row.get(key)
            if key == "noise_rate":
                metadata_matches = metadata_matches and actual_value not in (None, "") and float(actual_value) == float(expected_value)
            elif key.endswith("_seed"):
                metadata_matches = metadata_matches and actual_value not in (None, "") and int(float(actual_value)) == int(expected_value)
            elif key in {"train_ratio", "val_ratio"}:
                if expected_value is None:
                    metadata_matches = metadata_matches and actual_value in (None, "")
                else:
                    metadata_matches = metadata_matches and actual_value not in (None, "") and float(actual_value) == float(expected_value)
            elif key == "split_mode":
                metadata_matches = metadata_matches and str(actual_value or "planetoid") == str(expected_value)
            elif key == "split_name":
                fallback = f"split{int(expected_meta.get('split_seed', 0))}"
                metadata_matches = metadata_matches and str(actual_value or fallback) == str(expected_value)
            else:
                metadata_matches = metadata_matches and str(actual_value) == str(expected_value)
        if metadata_matches:
            observed.add((row.get("method", ""), f"{float(row.get('alpha_max', 0.0)):g}"))
    return expected.issubset(observed)


def task_command(task: dict) -> list[str]:
    return [str(task["python"]), *task["args"]]


def build_tasks(args: argparse.Namespace) -> list[dict]:
    if args.mode != "frozen":
        raise ValueError("Gate experiment v1 only supports --mode frozen")
    python_path = resolve_python(args)
    methods = args.methods or DEFAULT_METHODS
    tasks = []
    for dataset in args.datasets:
        for split_config in build_split_configs(args):
            for seed in args.seeds:
                split_seed = 0
                pretrain_seed = args.pretrain_seed if args.pretrain_seed is not None else seed
                split_token = split_name(
                    split_config["split_mode"],
                    split_seed,
                    split_config["train_ratio"],
                    split_config["val_ratio"],
                )
                checkpoint = checkpoint_path(
                    dataset,
                    split_seed,
                    pretrain_seed,
                    split_config["split_mode"],
                    split_config["train_ratio"],
                    split_config["val_ratio"],
                )
                embedding = embedding_path(
                    dataset,
                    split_seed,
                    pretrain_seed,
                    split_config["split_mode"],
                    split_config["train_ratio"],
                    split_config["val_ratio"],
                )
                tasks.append({
                    "kind": "pretrain",
                    "output": checkpoint,
                    "python": python_path,
                    "args": [
                        "scripts/run_vanilla_grace.py",
                        "--dataset",
                        dataset,
                        "--split-seed",
                        str(split_seed),
                        "--split-mode",
                        split_config["split_mode"],
                        "--pretrain-seed",
                        str(pretrain_seed),
                        *ratio_args(split_config),
                    ],
                })
                tasks.append({
                    "kind": "embedding",
                    "output": embedding,
                    "python": python_path,
                    "args": [
                        "scripts/extract_embeddings.py",
                        "--dataset",
                        dataset,
                        "--split-seed",
                        str(split_seed),
                        "--split-mode",
                        split_config["split_mode"],
                        "--pretrain-seed",
                        str(pretrain_seed),
                        *ratio_args(split_config),
                    ],
                })
                for noise_rate in args.noise_rates:
                    result_path = result_rows_path(
                        dataset,
                        split_seed,
                        pretrain_seed,
                        seed,
                        seed,
                        args.noise_type,
                        noise_rate,
                        split_config["split_mode"],
                        split_config["train_ratio"],
                        split_config["val_ratio"],
                    )
                    tasks.append({
                        "kind": "classifier",
                        "output": result_path,
                        "python": python_path,
                        "methods": methods,
                        "metadata": {
                            "dataset": dataset,
                            "split_mode": split_config["split_mode"],
                            "split_name": split_token,
                            "train_ratio": split_config["train_ratio"],
                            "val_ratio": split_config["val_ratio"],
                            "split_seed": split_seed,
                            "pretrain_seed": pretrain_seed,
                            "noise_seed": seed,
                            "classifier_seed": seed,
                            "noise_type": args.noise_type,
                            "noise_rate": noise_rate,
                        },
                        "args": [
                            "scripts/run_noisy_classifier.py",
                            "--dataset",
                            dataset,
                            "--split-seed",
                            str(split_seed),
                            "--split-mode",
                            split_config["split_mode"],
                            "--pretrain-seed",
                            str(pretrain_seed),
                            *ratio_args(split_config),
                            "--noise-seed",
                            str(seed),
                            "--classifier-seed",
                            str(seed),
                            "--noise-type",
                            args.noise_type,
                            "--noise-rate",
                            str(noise_rate),
                            "--classifier-epochs",
                            str(args.classifier_epochs),
                            "--patience",
                            str(args.patience),
                            "--alphas",
                            *[str(value) for value in args.alpha_max],
                            "--primary-alpha",
                            str(args.primary_alpha),
                            "--teacher-epochs",
                            str(args.teacher_epochs),
                            "--teacher-lr",
                            str(args.teacher_lr),
                            "--teacher-weight-decay",
                            str(args.teacher_weight_decay),
                            "--teacher-temperature",
                            str(args.teacher_temperature),
                            "--qb-pred-beta",
                            str(args.qb_pred_beta),
                            "--qb-teacher-gamma",
                            str(args.qb_teacher_gamma),
                            "--conflict-noisy-support-max",
                            str(args.conflict_noisy_support_max),
                            "--conflict-target-conf-min",
                            str(args.conflict_target_conf_min),
                            "--methods",
                            *methods,
                        ],
                    })
    return tasks


def build_split_configs(args: argparse.Namespace) -> list[dict]:
    if args.split_mode == "planetoid":
        return [{"split_mode": "planetoid", "train_ratio": None, "val_ratio": None}]
    return [
        {"split_mode": "ratio", "train_ratio": train_ratio, "val_ratio": args.val_ratio}
        for train_ratio in args.train_ratios
    ]


def ratio_args(split_config: dict) -> list[str]:
    if split_config["split_mode"] != "ratio":
        return []
    return [
        "--train-ratio",
        str(split_config["train_ratio"]),
        "--val-ratio",
        str(split_config["val_ratio"]),
    ]


def should_skip(task: dict, args: argparse.Namespace) -> bool:
    if args.force:
        return False
    output = task["output"]
    if task["kind"] in {"pretrain", "embedding"}:
        return output.exists()
    if task["kind"] == "classifier":
        return expected_result_complete(output, task["methods"], args.alpha_max, task["metadata"])
    return False


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    return [task_command(task) for task in build_tasks(args)]


def run(args: argparse.Namespace) -> None:
    python_path = resolve_python(args)
    check_environment(python_path, dry_run=args.dry_run)
    tasks = build_tasks(args)
    if args.max_commands is not None:
        tasks = tasks[: args.max_commands]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir / run_id
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
        manifest = [
            {
                "kind": task["kind"],
                "output": str(task["output"]),
                "skip": should_skip(task, args),
                "command": " ".join(task_command(task)),
            }
            for task in tasks
        ]
        (log_dir / "commands.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    for command_idx, task in enumerate(tasks):
        command = task_command(task)
        printable = " ".join(command)
        if should_skip(task, args):
            print(f"(K) | skip existing {task['kind']}: {task['output']}")
            continue
        if args.dry_run:
            print(printable)
            continue
        print(f"(X) | {printable}")
        log_path = log_dir / f"cmd_{command_idx:04d}_{task['kind']}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(printable + "\n")
            handle.flush()
            subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
        print(f"(L) | raw log: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["Cora", "CiteSeer"], default=["Cora", "CiteSeer"])
    parser.add_argument("--noise-type", choices=["symmetric"], default="symmetric")
    parser.add_argument("--noise-rates", nargs="+", type=float, default=[0.0, 0.2, 0.4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--pretrain-seed",
        type=int,
        default=None,
        help="Override pretrain seed for debugging. Formal gate leaves this unset so pretrain_seed=seed.",
    )
    parser.add_argument("--split-mode", choices=["planetoid", "ratio"], default="planetoid")
    parser.add_argument("--train-ratios", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--mode", choices=["frozen"], default="frozen")
    parser.add_argument("--save-diagnostics", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CHOICES,
        default=DEFAULT_METHODS,
    )
    parser.add_argument("--alpha-max", "--alphas", dest="alpha_max", nargs="+", type=float, default=[0.3, 0.5])
    parser.add_argument("--primary-alpha", type=float, default=0.3)
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--teacher-lr", type=float, default=0.01)
    parser.add_argument("--teacher-weight-decay", type=float, default=5e-4)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--qb-pred-beta", type=float, default=0.5)
    parser.add_argument("--qb-teacher-gamma", type=float, default=2.0)
    parser.add_argument("--conflict-noisy-support-max", type=float, default=0.35)
    parser.add_argument("--conflict-target-conf-min", type=float, default=0.55)
    parser.add_argument("--classifier-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--allow-external-python", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-commands", type=int, default=None)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "gate")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
