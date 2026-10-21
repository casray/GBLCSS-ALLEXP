import difflib
import hashlib
import json
import re
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "GB-LCSS_submission_final_20260924"
NEW = ROOT / "GB-LCSS_submission_notation_final_20260925"
OUT = Path(__file__).resolve().parent
EDITED = {
    "main.tex", "supplementary.tex", "tables/notation.tex",
    "algorithms/algorithm1.tex", "algorithms/algorithm2.tex", "README.md",
}
GENERATED = {".aux", ".log", ".out", ".pdf", ".spl", ".bbl", ".blg"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels(text):
    return re.findall(r"\\label\{([^}]+)\}", text)


def equation_numbers(text):
    return dict(re.findall(r"\\newlabel\{(eq:[^}]+)\}\{\{([^}]+)\}", text))


report = {"unchanged_files": [], "edited_files": [], "documents": {}}
patch = []
for original in sorted(OLD.rglob("*")):
    if not original.is_file():
        continue
    relative = original.relative_to(OLD).as_posix()
    revised = NEW / relative
    assert revised.is_file(), relative
    if relative in EDITED:
        before = original.read_text(encoding="utf-8")
        after = revised.read_text(encoding="utf-8")
        patch.extend(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="before/" + relative, tofile="after/" + relative,
        ))
        if relative.endswith(".tex"):
            assert labels(before) == labels(after), (relative, "labels changed")
            for command in ("cite", "ref", "eqref"):
                pattern = r"\\" + command + r"\*?\{([^}]+)\}"
                assert re.findall(pattern, before) == re.findall(pattern, after), (relative, command)
            float_pattern = r"\\(?:begin\{(?:table|figure|algorithm)\}\[[^]]+\]|FloatBarrier|Needspace\{[^}]+\})"
            assert re.findall(float_pattern, before) == re.findall(float_pattern, after), (relative, "float controls")
        report["edited_files"].append(relative)
    elif original.parent == OLD and original.suffix in GENERATED:
        continue
    else:
        assert digest(original) == digest(revised), (relative, "unexpected change")
        report["unchanged_files"].append(relative)

for name in ("main", "supplementary", "highlights"):
    before = (OLD / (name + ".tex")).read_text(encoding="utf-8")
    after = (NEW / (name + ".tex")).read_text(encoding="utf-8")
    if name == "main":
        before = before.split(r"\section{Experiments}")[1]
        after = after.split(r"\section{Experiments}")[1]
    elif name == "supplementary":
        theory = r"\\section\{Analysis of local-consensus supervision\}.*?(?=\\section\{Parameter sensitivity and target-error decomposition\})"
        before = re.sub(theory, "", before, flags=re.S)
        after = re.sub(theory, "", after, flags=re.S)
    number_pattern = r"(?<![A-Za-z])\d+(?:\.\d+)?"
    if name in ("main", "supplementary"):
        assert re.findall(number_pattern, before) == re.findall(number_pattern, after), (name, "result numbers")
    aux_before = (OLD / (name + ".aux")).read_text(encoding="utf-8")
    aux_after = (NEW / (name + ".aux")).read_text(encoding="utf-8")
    assert equation_numbers(aux_before) == equation_numbers(aux_after), (name, "equation numbering")
    log = (NEW / (name + ".log")).read_text(encoding="utf-8", errors="replace")
    assert not re.search(r"(?:Reference|Citation) .*undefined|There were undefined|multiply defined", log), name
    pdf = fitz.open(NEW / (name + ".pdf"))
    report["documents"][name] = {
        "pages": len(pdf),
        "equations": equation_numbers(aux_after),
        "overfull": re.findall(r"Overfull .*", log),
    }
    for index, page in enumerate(pdf):
        text = page.get_text()
        if name == "main":
            render = bool(re.search(r"Main notation|Definition [1-6]|Proposition 1|Corollary 1|Graph-teacher distillation extension|\(21\)", text))
        elif name == "supplementary":
            render = bool(re.search(r"\(H\.[1-6]\)", text))
        else:
            render = False
        if render:
            page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35)).save(OUT / f"{name}_page_{index + 1:02d}.png")

(OUT / "revision.patch").write_text("".join(patch), encoding="utf-8")
(OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({
    "edited_files": report["edited_files"],
    "unchanged_file_count": len(report["unchanged_files"]),
    "documents": report["documents"],
}, ensure_ascii=False, indent=2))
