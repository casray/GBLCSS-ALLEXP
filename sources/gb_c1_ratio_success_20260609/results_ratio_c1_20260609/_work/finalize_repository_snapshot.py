"""Check archived reference rows and package the existing reviewer repository."""

from collections import defaultdict
import csv
import hashlib
import io
from pathlib import Path
import re
import statistics
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "GB-LCSS_GitHub"
ORIGINAL = Path("D:/papercode_final_v4.zip")
SOURCE_MEMBER = "results/main_tables/table2_per_seed.csv"
DATASETS = ["cora", "citeseer", "corafull", "coauthorcs", "amazonpho"]
SETTINGS = [("uniform", "0.2"), ("uniform", "0.4"), ("uniform", "0.6"),
            ("pair", "0.4"), ("instance", "0.4")]


def main():
    with zipfile.ZipFile(ORIGINAL) as original:
        archived = original.read(SOURCE_MEMBER)
    assert archived == (ROOT / "GB-LCSS_code" / SOURCE_MEMBER).read_bytes()
    rows = list(csv.DictReader(io.StringIO(archived.decode("utf-8-sig"))))
    assert len(rows) == 1000
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"], row["noise_type"], row["noise_rate"])].append(row)
    assert len(groups) == 200
    checked = 0
    for index, dataset in enumerate(DATASETS, 1):
        table = (ROOT / "GB-LCSS_submission_notation_final_20260925" / "tables" /
                 f"table_A_{index}.tex").read_text(encoding="utf-8-sig")
        for line in table.splitlines():
            cells = line.split(" & ")
            if len(cells) != 6:
                continue
            method = cells[0].strip()
            if (dataset, method, *SETTINGS[0]) not in groups:
                continue
            for cell, (noise, rate) in zip(cells[1:], SETTINGS):
                group = groups[(dataset, method, noise, rate)]
                assert len(group) == 5 and {row["seed"] for row in group} == set("01234")
                values = [100 * float(row["test_accuracy"]) for row in group]
                numbers = re.findall(r"\d+\.\d+", cell)
                actual = [f"{statistics.mean(values):.2f}", f"{statistics.stdev(values):.2f}"]
                assert numbers == actual, (dataset, method, noise, rate, numbers, actual)
                checked += 1
    assert checked == 200
    print(f"Reference verified: {len(rows)} archived rows; {checked} paper mean/std cells match.")
    (REPO / "reference_results" / "matched_per_seed.csv").write_bytes(archived)

    files = [REPO / name for name in [
        "README.md", "DATASETS.md", "LICENSE", "THIRD_PARTY_NOTICES.md", ".gitignore",
        "requirements.txt", "pytest.ini", "run.py", "scripts/prepare_representation.py",
        "reference_results/README.md", "reference_results/matched_per_seed.csv",
    ]]
    for directory, extension in [("src", "*.py"), ("tests", "*.py"), ("configs", "*.yaml"),
                                 ("representations", "*.pt"), ("licenses", "*.txt")]:
        files.extend(sorted((REPO / directory).rglob(extension)))
    files = sorted(set(files), key=lambda path: path.relative_to(REPO).as_posix())
    checksums = {path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in files}
    manifest = REPO / "SHA256SUMS.txt"
    manifest.write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()), encoding="ascii")
    files.append(manifest)
    archive_path = ROOT / "GB-LCSS_GitHub.zip"
    staging = ROOT / "GB-LCSS_GitHub.pending.zip"
    with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, "GB-LCSS/" + path.relative_to(REPO).as_posix())
    with zipfile.ZipFile(staging) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files)
        for name, digest in checksums.items():
            assert hashlib.sha256(archive.read("GB-LCSS/" + name)).hexdigest() == digest
        assert not any("__pycache__" in name or "/outputs/" in name for name in archive.namelist())
    staging.replace(archive_path)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_path.with_suffix(".zip.sha256").write_text(f"{digest}  {archive_path.name}\n", encoding="ascii")
    print(f"Verified ZIP: {len(files)} files; {archive_path.stat().st_size / 1024**2:.1f} MiB")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
