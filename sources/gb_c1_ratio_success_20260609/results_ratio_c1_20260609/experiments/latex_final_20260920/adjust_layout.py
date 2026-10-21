from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "GB-LCSS_Elsevier_LaTeX_five_dataset_final_20260920"

for path in [OUT / "main.tex", *(OUT / "tables").glob("table_*.tex")]:
    text = path.read_text(encoding="utf-8")
    text = text.replace(r"\begin{figure}[htbp]", r"\begin{figure}[!htb]")
    text = text.replace(r"\begin{table}[htbp]", r"\begin{table}[!htb]")
    if path.name == "main.tex":
        text = text.replace(r"\Needspace{85pt}", r"\Needspace{64pt}")
        text = text.replace(r"\setlength{\bibsep}{2pt plus 0.3pt minus 0.3pt}",
                            r"\setlength{\bibsep}{0.5pt plus 0.2pt minus 0.2pt}")
    if path.name.startswith("table_A_"):
        text = text.replace("\\Needspace{230pt}\n", "")
    if path.name.startswith(("table_F_", "table_G_")):
        import re
        text = re.sub(r"\\Needspace\{\d+pt\}\n", "", text)
    if any(path.name.startswith(f"table_{letter}_") for letter in "ABCDEFGHI"):
        text = text.replace(r"\renewcommand{\arraystretch}{1.18}", r"\renewcommand{\arraystretch}{1.12}")
    if path.name.startswith(("table_A_", "table_G_")) or path.name in {"table_F_1.tex", "table_F_3.tex", "table_C_1.tex"}:
        lines = text.splitlines()
        body_start = next(i for i, line in enumerate(lines) if "\\endlastfoot" in line) + 1
        row_indices = [i for i in range(body_start, len(lines)) if " & " in lines[i] and lines[i].endswith((r"\\", r"\\*"))]
        for position, i in enumerate(row_indices):
            if 1 <= position < len(row_indices) - 2:
                lines[i] = lines[i].removesuffix("*")
        text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")

path = OUT / "supplementary.tex"
text = path.read_text(encoding="utf-8")
text = text.replace(r"\Needspace{265pt}", r"\Needspace{120pt}")
text = text.replace(r"\Needspace{240pt}", r"\Needspace{120pt}")
text = text.replace(r"\setlength{\LTpre}{9pt}", r"\setlength{\LTpre}{6pt}")
text = text.replace(r"\setlength{\LTpost}{9pt}", r"\setlength{\LTpost}{6pt}")
path.write_text(text, encoding="utf-8", newline="\n")
print("Adjusted float placement and oversized space reservations; body and table values unchanged.")
