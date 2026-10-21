from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "GB-LCSS_Elsevier_LaTeX_five_dataset_final_20260920"
WORK = Path(__file__).resolve().parent

for stem in ["main", "supplementary"]:
    aux = (OUT / f"{stem}.aux").read_text(encoding="utf-8")
    label_list = re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}", aux)
    labels = dict(label_list)
    assert len(labels) == len(label_list), "Duplicate labels"
    text = (OUT / f"{stem}.tex").read_text(encoding="utf-8")
    refs = re.findall(r"\\(?:ref\*?|eqref)\{([^}]+)\}", text)
    assert set(refs) <= set(labels), set(refs) - set(labels)
    if stem == "main":
        assert all(labels[f"eq:{n}"] == str(n) for n in range(1, 24))
        assert labels["sec:3.4"] == "3.4"
        assert labels["sec:4.3.4"] == "4.3.4"
        assert labels["tab:6"] == "7"
        assert labels["tab:7"] == "8"
    else:
        assert all(labels[f"eq:H.{n}"] == f"H.{n}" for n in range(1, 7))
        assert labels["tab:I.1"] == "I.1" and labels["tab:I.2"] == "I.2"

path = OUT.with_suffix(".zip")
allowed = {".tex", ".bib", ".bst", ".bbl", ".pdf", ".md"}
files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.suffix in allowed)
with ZipFile(path, "w", ZIP_DEFLATED) as archive:
    for p in files:
        archive.write(p, p.relative_to(OUT).as_posix())
with ZipFile(path) as archive:
    assert archive.testzip() is None
    for p in files:
        assert archive.read(p.relative_to(OUT).as_posix()) == p.read_bytes()
report = json.loads((WORK / "checks.json").read_text(encoding="utf-8"))
print(f'Main: {report["main"]["pages"]} pages; supplement: {report["supplementary"]["pages"]} pages.')
print(f"Cross-references, equation numbering, ZIP integrity: passed.")
print(f"ZIP: {path}\nFiles: {len(files)}\nBytes: {path.stat().st_size}")
