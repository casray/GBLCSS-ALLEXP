from pathlib import Path
import json
import re
from zipfile import ZipFile, ZIP_DEFLATED

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / "GB-LCSS_Elsevier_LaTeX_integrated_final_20260920"
BASE = ROOT / "GB-LCSS_Elsevier_LaTeX_five_dataset_final_20260920"
CLAUDE = ROOT / "GB-LCSS_Elsevier_LaTeX_polished_20260920"


def read(path):
    return path.read_text(encoding="utf-8-sig")


def equations(text):
    return [re.sub(r"\s+", "", s) for s in re.findall(
        r"\\begin\{equation\}(.*?)\\end\{equation\}", text, re.S)]


def data_rows(path):
    return [re.sub(r"\\ensuremath\{([-+])\}", r"\1", line.strip()).replace(r"\\*", r"\\")
            for line in read(path).splitlines() if " & " in line]


main = read(OUT / "main.tex")
assert "Unless stated otherwise" not in main
assert not any(s in main for s in ["[repository URL]", "[tool name]", "recovers none", "reinforce themselves"])
for stem in ["main", "supplementary"]:
    assert equations(read(OUT / f"{stem}.tex")) == equations(read(CLAUDE / f"{stem}.tex"))
for name in ["references.bib", "gblcss-numeric.bst"]:
    assert (OUT / name).read_bytes() == (BASE / name).read_bytes()
for folder in ["algorithms", "figures"]:
    for p in (BASE / folder).iterdir():
        if p.is_file():
            assert p.read_bytes() == (OUT / folder / p.name).read_bytes(), p
for p in (BASE / "tables").glob("*.tex"):
    assert data_rows(p) == data_rows(OUT / "tables" / p.name), p

keys = set(re.findall(r"@\w+\{([^,]+),", read(OUT / "references.bib")))
cited = {key for group in re.findall(r"\\cite\{([^}]+)\}", main) for key in group.split(",")}
assert keys == cited and len(keys) == 74
highlights = re.findall(r"\\item (.*)", read(OUT / "highlights.tex"))
assert len(highlights) == 4 and all(len(s) <= 85 for s in highlights)
report = {"source_checks": "passed", "references": len(keys), "highlight_lengths": list(map(len, highlights))}
report["abstract_words"] = len(re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S).group(1).split())

for stem in ["main", "supplementary", "highlights"]:
    log = read(OUT / f"{stem}.log")
    errors = re.findall(r"^.*(?:Undefined control sequence|LaTeX Error|undefined references|multiply defined|Missing character).*$", log, re.M)
    assert not errors, errors
    if stem != "highlights":
        aux = read(OUT / f"{stem}.aux")
        labels_list = re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}", aux)
        labels = dict(labels_list)
        assert len(labels_list) == len(labels)
        refs = re.findall(r"\\(?:ref\*?|eqref)\{([^}]+)\}", read(OUT / f"{stem}.tex"))
        assert set(refs) <= set(labels), set(refs) - set(labels)
        if stem == "main":
            assert all(labels[f"eq:{i}"] == str(i) for i in range(1, 24))
            assert labels["tab:6"] == "7" and labels["tab:7"] == "8"
        else:
            assert all(labels[f"eq:H.{i}"] == f"H.{i}" for i in range(1, 7))
    with pdfplumber.open(OUT / f"{stem}.pdf") as pdf:
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        assert "??" not in text and "\ufffd" not in text
        (WORK / f"{stem}_text.txt").write_text(text, encoding="utf-8")
        report[stem] = {"pages": len(pdf.pages)}
    document = pdfium.PdfDocument(str(OUT / f"{stem}.pdf"))
    thumbs, layout = [], []
    pages_dir = WORK / f"{stem}_pages"
    pages_dir.mkdir(exist_ok=True)
    for i in range(len(document)):
        page = document[i]
        bitmap = page.render(scale=1.5)
        im = bitmap.to_pil().convert("RGB")
        im.save(pages_dir / f"page_{i+1:02d}.png")
        # Pixel bounds include text inside vector figures, unlike PDF text extraction.
        gray = im.convert("L")
        x0, x1 = int(45*1.5), im.width-int(45*1.5)
        used = []
        for y in range(int(76*1.5), min(im.height, int(715*1.5))):
            if gray.crop((x0,y,x1,y+1)).getextrema()[0] < 180:
                used.append(y)
        gaps = sorted(((b-a-1)/1.5, a/1.5, b/1.5) for a,b in zip(used, used[1:]))
        layout.append({"page": i+1, "bottom": round(max(used)/1.5,1) if used else None,
                       "max_internal_gap": gaps[-1] if gaps else None})
        thumb = ImageOps.contain(im, (304, 398))
        panel = Image.new("RGB", (326, 430), "#eeeeee")
        panel.paste(thumb, ((326-thumb.width)//2, 10))
        ImageDraw.Draw(panel).text((12, 411), f"{stem} {i+1}", fill="black")
        thumbs.append(panel)
        page.close()
    document.close()
    report[stem]["layout"] = layout
    for offset in range(0, len(thumbs), 12):
        group = thumbs[offset:offset+12]
        contact = Image.new("RGB", (1304, 430*((len(group)+3)//4)), "#dddddd")
        for j, panel in enumerate(group):
            contact.paste(panel, ((j%4)*326, (j//4)*430))
        contact.save(WORK / f"{stem}_contact_{offset//12+1}.png")

(WORK / "checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.suffix in {".tex", ".bib", ".bst", ".bbl", ".pdf", ".md"})
with ZipFile(OUT.with_suffix(".zip"), "w", ZIP_DEFLATED) as archive:
    for p in files:
        archive.write(p, p.relative_to(OUT).as_posix())
with ZipFile(OUT.with_suffix(".zip")) as archive:
    assert archive.testzip() is None
    for p in files:
        assert archive.read(p.relative_to(OUT).as_posix()) == p.read_bytes()
print(f"ZIP verified: {len(files)} files")
