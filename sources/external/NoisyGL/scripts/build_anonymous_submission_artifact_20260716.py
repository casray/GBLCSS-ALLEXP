from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def find_forbidden_matches(text: str, *, hostname: str = "") -> list[str]:
    username = os.environ.get("USERNAME", "").strip()
    patterns = [
        (r"(?i)c:\\users\\", r"C:\Users"),
        (r"(?i)/users/", "/Users/"),
        (r"(?i)/home/", "/home/"),
        (r"实验8", "实验8"),
        (r"(?i)administrator", "Administrator"),
    ]
    if username:
        patterns.append((re.escape(username), username))
    if hostname:
        patterns.append((re.escape(hostname), hostname))
    matches = []
    for pattern, label in patterns:
        if re.search(pattern, text):
            matches.append(label)
    return sorted(set(matches))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_allowlist(path: Path) -> list[Path]:
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        relative = Path(line)
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
            raise ValueError(f"unsafe allowlist entry: {line}")
        entries.append(relative)
    if not entries:
        raise ValueError("artifact allowlist is empty")
    return entries


def _scan_text(path: Path, *, hostname: str) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = find_forbidden_matches(text, hostname=hostname)
    if matches:
        raise ValueError(
            f"forbidden identity/path token in {path}: {', '.join(matches)}"
        )


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def build_artifact(
    *,
    source_root: str | Path,
    output_root: str | Path,
    allowlist_path: str | Path,
    hostname: str = "",
) -> list[dict[str, object]]:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    allowlist_path = Path(allowlist_path).resolve()
    if not source_root.is_dir():
        raise ValueError(f"source root does not exist: {source_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"artifact output must be new or empty: {output_root}")
    validated: list[tuple[Path, Path]] = []
    for relative in _read_allowlist(allowlist_path):
        source = (source_root / relative).resolve()
        try:
            source.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"allowlist entry escapes source root: {relative}") from exc
        if not source.is_file():
            raise ValueError(f"allowlisted file is missing: {relative}")
        _scan_text(source, hostname=hostname)
        validated.append((relative, source))

    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for relative, source in validated:
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        rows.append(
            {
                "path": relative.as_posix(),
                "size_bytes": int(destination.stat().st_size),
                "sha256": _sha256(destination),
            }
        )

    rows.sort(key=lambda row: str(row["path"]))
    _write_manifest(output_root / "ARTIFACT_MANIFEST.csv", rows)
    (output_root / "README.md").write_text(
        "# Anonymous LCSS Reproducibility Artifact\n\n"
        "This directory contains the code, fixed configurations, audited result tables, and statistical "
        "scripts needed to inspect the LCSS experiments. It is generated from an explicit allowlist and "
        "contains no repository history. Every copied file is recorded in `ARTIFACT_MANIFEST.csv` with "
        "its SHA256 digest.\n\n"
        "## Environment\n\n"
        "The versions used in the reported ROCm runs are listed in "
        "`docs/anonymous_requirements_20260716.txt`. Install the platform-appropriate PyTorch build.\n\n"
        "## GRACE preprocessing\n\n"
        "Frozen embeddings are identified by SHA256 in `docs/h_embedding_manifest_20260712.md`. "
        "The MLP-37 pretraining script accepts `--upstream-root` pointing to the archived GRACE checkout; "
        "the downstream runner only needs the resulting `H.pt` path. GRACE preprocessing is label-free.\n\n"
        "## Scope\n\n"
        "The copied configs cover the five full-spectrum datasets, LCSS, its size-matched shuffled control, "
        "and the seven same-backbone baselines used for the formal ranking.\n",
        encoding="utf-8",
    )
    for path in output_root.rglob("*"):
        if path.is_file():
            _scan_text(path, hostname=hostname)
    if (output_root / ".git").exists():
        raise ValueError("artifact unexpectedly contains git history")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean, allowlisted anonymous submission artifact.")
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--hostname", default=os.environ.get("COMPUTERNAME", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_artifact(
        source_root=args.source_root,
        output_root=args.output_root,
        allowlist_path=args.allowlist,
        hostname=args.hostname,
    )
    print(f"WROTE {args.output_root} ({len(rows)} allowlisted files)")


if __name__ == "__main__":
    main()
