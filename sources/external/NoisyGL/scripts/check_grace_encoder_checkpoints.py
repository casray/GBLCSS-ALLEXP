from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [
    REPO_ROOT.parents[1] / "gb_graph_aware_minival_6_1",
    Path(r"C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master"),
]


def is_candidate(path: Path) -> bool:
    lower_name = path.name.lower()
    lower_path = str(path).lower()
    if lower_name == "h.pt":
        return False
    if "\\embeddings" in lower_path or "/embeddings" in lower_path:
        return False
    if path.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
        return False
    tokens = ("encoder", "model", "checkpoint", "ckpt", "best", "grace")
    return any(token in lower_path for token in tokens)


def scan_roots(roots: list[Path], max_rows: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in roots:
        root = root.expanduser()
        if not root.exists():
            rows.append({
                "root": str(root),
                "status": "missing_root",
                "path": "",
                "size_bytes": "",
            })
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not is_candidate(path):
                continue
            rows.append({
                "root": str(root),
                "status": "candidate",
                "path": str(path),
                "size_bytes": path.stat().st_size,
            })
            if len([row for row in rows if row["status"] == "candidate"]) >= max_rows:
                return rows
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["root", "status", "path", "size_bytes"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan for reusable GRACE encoder checkpoints for MLP-7 7c-step0.")
    parser.add_argument("--roots", default=";".join(str(root) for root in DEFAULT_ROOTS))
    parser.add_argument("--output", default=str(REPO_ROOT / "runs" / "mlp7_baselines_20260701" / "grace_encoder_checkpoint_scan.csv"))
    parser.add_argument("--max_rows", type=int, default=200)
    args = parser.parse_args()

    roots = [Path(item.strip()) for item in str(args.roots).split(";") if item.strip()]
    rows = scan_roots(roots, max_rows=max(1, int(args.max_rows)))
    write_csv(Path(args.output), rows)
    candidates = sum(1 for row in rows if row["status"] == "candidate")
    print(f"WROTE {args.output}")
    print(f"GRACE_ENCODER_CANDIDATES={candidates}")


if __name__ == "__main__":
    main()
