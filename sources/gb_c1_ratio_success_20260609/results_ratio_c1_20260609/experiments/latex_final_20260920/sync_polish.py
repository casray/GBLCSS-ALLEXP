from pathlib import Path
import csv
import re
import shutil


ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
BASE = ROOT / "GB-LCSS_Elsevier_LaTeX_discussion_layout_final_20260918"
OUT = ROOT / "GB-LCSS_Elsevier_LaTeX_five_dataset_final_20260920"
DATA = ROOT / "experiments/submission_checks_20260920_extension/combined_five_datasets"
DATASETS = ["cora", "citeseer", "corafull", "coauthorcs", "amazonphoto"]
NAMES = ["Cora", "CiteSeer", "CoraFull", "Coauthor-CS", "Amazon-Photo"]


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected one match, found {count}: {old[:100]}")
    return text.replace(old, new, 1)


def read(path):
    return path.read_text(encoding="utf-8-sig")


def write(path, text):
    path.write_text(text, encoding="utf-8", newline="\n")


OUT.mkdir(exist_ok=True)
for folder in ["tables", "algorithms", "figures"]:
    (OUT / folder).mkdir(exist_ok=True)
for pattern in ["*.tex", "*.bib", "*.bst"]:
    for path in BASE.glob(pattern):
        shutil.copy2(path, OUT / path.name)
for folder in ["tables", "algorithms"]:
    for path in (BASE / folder).glob("*.tex"):
        shutil.copy2(path, OUT / folder / path.name)
for path in (BASE / "figures").glob("*.pdf"):
    shutil.copy2(path, OUT / "figures" / path.name)

main = read(BASE / "main.tex")
# Shift existing labels before inserting the new analysis to avoid collisions.
main = re.sub(r"eq:(1[4-9]|20)(?=\})", lambda m: f"eq:{int(m[1]) + 3}", main)
main = re.sub(r"sec:3\.([45])(?=[\]}])", lambda m: f"sec:3.{int(m[1]) + 1}", main)
main = replace_once(main, r"\newtheorem{definition}{Definition}",
                    r"\newtheorem{definition}{Definition}" + "\n" + r"\newtheorem{proposition}{Proposition}")
old_contrib = r"\item We develop consensus-weighted soft supervision within these regions. Each training node combines its observed label with the labels of the other training nodes in its ball, with their agreement determining the contribution of local consensus."
main = replace_once(main, old_contrib, old_contrib + " We analyze the reliability of local voting under independent symmetric label noise and derive an exact decomposition of the squared probability error of the resulting soft target.")
main = replace_once(main, "presents GB-LCSS and its graph-teacher extension.",
                    "presents GB-LCSS, analyzes local-consensus supervision, and introduces the graph-teacher extension.")
marker = r"\subsection{Graph-teacher distillation extension}\label{sec:3.5}"
main = replace_once(main, marker, read(WORK / "theory.tex") + "\n" + marker)
main = replace_once(main, "Appendices A--G are provided", "Appendices A--I are provided")
main = replace_once(main, r"\(\beta = 0.5\) throughout the experiments.",
                    r"\(\beta = 0.5\) as the default settings.")

start = main.index(r"\subsubsection{Effect of the representation space}")
end = main.index(r"\subsection{Graph-teacher extension and computational cost}", start)
main = main[:start] + read(WORK / "experiments.tex") + "\n" + main[end:]
marker = r"\subsubsection{Sensitivity to the grouping rule}"
main = replace_once(main, marker,
    "Across the five datasets under uniform 0.4 noise, we also separate the reduction in target error on corrupted labels from the increase on initially correct labels (Appendix~I.2). The reduction exceeds the increase in all 25 dataset--seed runs.\n\n" + marker)

start = main.index("GB-LCSS uses labels from related training nodes", main.index(r"\subsection{Discussion}"))
end = main.index("\n\n", start)
main = main[:start] + r"""GB-LCSS supplements an individual annotation with labels from related training nodes. Label-free grouping keeps corrupted annotations from shaping the groups, while excluding the current node prevents its label from reinforcing the consensus used to adjust it. Proposition~\ref{prop:local_vote} bounds the probability that a competing class matches or exceeds the true class in local votes under independent symmetric noise and a positive clean-class margin. In the random-grouping and smoothing controls (Table~\ref{tab:4}), preserving local membership yields higher accuracy than the alternative targets. Appendix~I.2 shows that the reduction in squared probability error on corrupted labels exceeds the increase on initially correct labels in all evaluated runs.""" + main[end:]
main = replace_once(main,
    "These findings place GB-LCSS as a method for improving noisy-label supervision while retaining graph-free inference.",
    "GB-LCSS is designed to improve noisy-label supervision while retaining graph-free inference.")
main = replace_once(main,
    r"Its mixed ranking against Graph-MLP in Table~\ref{tab:3} indicates that this approach and graph-based training objectives offer different benefits across datasets.",
    r"Its mixed ranking against Graph-MLP in Table~\ref{tab:3} shows that the relative accuracy of local-consensus supervision and graph-based training objectives varies across datasets.")
main = replace_once(main,
    "The prescribed budget also bounds the number of regions to be maintained.",
    r"The prescribed budget also bounds the number of regions to be maintained. In Section~\ref{sec:4.3.4}, the largest accuracy losses occurred when all nodes remained in one ball or substantially fewer nodes received consensus supervision.")
conclusion = "This separates region construction from noisy-label aggregation and provides soft targets for an MLP that uses only node attributes at inference."
main = replace_once(main, conclusion, conclusion + " The analysis characterizes local-vote reliability under independent symmetric noise and decomposes the squared probability error of the resulting soft target.")
write(OUT / "main.tex", main)

supp = read(BASE / "supplementary.tex")
supp = supp.replace("Equation references refer to the main paper.",
    "Equations numbered (1)--(23) refer to the main paper; equations numbered (H.1)--(H.6) are given in Appendix~H below.")
supp = replace_once(supp, r"\end{document}", read(WORK / "appendices.tex") + "\n\n" + r"\end{document}")
# Unstretched text avoids turning available page space into large vertical gaps.
supp = replace_once(supp, r"\setlength{\textfloatsep}{14pt plus 3pt minus 3pt}",
    r"\setlength{\textfloatsep}{12pt plus 2pt minus 2pt}" + "\n" +
    r"\setlength{\floatsep}{12pt plus 2pt minus 2pt}" + "\n" +
    r"\setlength{\intextsep}{12pt plus 2pt minus 2pt}" + "\n" + r"\raggedbottom")
write(OUT / "supplementary.tex", supp)

f2 = read(OUT / "tables/table_F_2.tex")
f2 = replace_once(f2, "Chosen on Cora and CiteSeer, then fixed",
                  "Set during method development on Cora and CiteSeer; used unchanged in the main comparisons")
f2 = replace_once(f2, r"Eq.~(14)", r"Eq.~(17)")
write(OUT / "tables/table_F_2.tex", f2)

with (DATA / "summary_verified.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
summary = {(r["dataset"], r["arm"]): r for r in rows}
assert len(summary) == 90
assert all(int(r["seeds"]) == 5 for r in rows)


def accuracy(dataset, arm, bold=False):
    r = summary[dataset, arm]
    value = f'{100*float(r["accuracy_mean"]):.2f}' + r"\ensuremath{\pm}" + f'{100*float(r["accuracy_sd"]):.2f}'
    return r"\textbf{" + value + "}" if bold else value


def balls(dataset, arm):
    r = summary[dataset, arm]
    low, high = int(float(r["balls_min"])), int(float(r["balls_max"]))
    return str(low) if low == high else f"{low}--{high}"


table7 = r"""\begin{table}[htbp]
\centering
\caption{Classification accuracy (\%) and ball counts for three representation spaces under uniform 0.4 noise. Accuracy is mean \(\pm\) sample standard deviation over five paired seeds; ball counts give the range across seeds.}\label{tab:6}
\begingroup
\fontsize{9}{11}\selectfont
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.18}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
\textbf{Dataset} & \multicolumn{3}{c}{\textbf{Accuracy}} & \multicolumn{3}{c}{\textbf{Number of balls}}\\
\cmidrule(lr){2-4}\cmidrule(l){5-7}
& GRACE & Raw \(X\) & \(A^2X\) & GRACE & Raw \(X\) & \(A^2X\)\\
\midrule
"""
for dataset, name in zip(DATASETS, NAMES):
    values = [name] + [accuracy(dataset, arm, arm == "default") for arm in ["default", "raw_x", "a2x"]]
    values += [balls(dataset, arm) for arm in ["default", "raw_x", "a2x"]]
    table7 += " & ".join(values) + r" \\" + "\n"
table7 += "\\bottomrule\n\\end{tabular*}\n\\endgroup\n\\end{table}\n"
write(OUT / "tables/table_6.tex", table7)

arms = [
    ("default", "Default"),
    ("beta_0", r"\(\beta=0\)"),
    ("beta_0.25", r"\(\beta=0.25\)"),
    ("beta_0.75", r"\(\beta=0.75\)"),
    ("beta_1", r"\(\beta=1\)"),
    ("max_balls_32", r"\(B_{\max}=32\)"),
    ("max_balls_128", r"\(B_{\max}=128\)"),
    ("min_train_per_ball_5", r"\(s_{\min}=5\)"),
    ("min_train_per_ball_10", r"\(s_{\min}=10\)"),
    ("min_ball_size_1", r"\(n_{\min}=1\)"),
    ("min_ball_size_16", r"\(n_{\min}=16\)"),
    ("min_ball_size_64", r"\(n_{\min}=64\)"),
    ("split_improvement_0.01", r"\(\delta=0.01\)"),
    ("split_improvement_0.1", r"\(\delta=0.10\)"),
    ("mlp_ce", "MLP+CE"),
    ("shuffled", "Random grouping"),
    ("raw_x", r"Raw \(X\)"),
    ("a2x", r"\(A^2X\)"),
]
best = {d: max(float(summary[d, a]["accuracy_mean"]) for a, _ in arms) for d in DATASETS}
header = r"\textbf{Configuration} & " + " & ".join(r"\textbf{" + n + "}" for n in NAMES) + r" \\"
tablei1 = r"""\begingroup
\fontsize{9}{11}\selectfont
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}lccccc@{}}
\caption{Parameter sensitivity and controls under uniform 0.4 noise. Accuracy (\%) is mean \(\pm\) sample standard deviation over five paired seeds. Bold denotes the highest mean in each column.}\label{tab:I.1}\\
\toprule
""" + header + "\n\\midrule\n\\endfirsthead\n"
tablei1 += r"\multicolumn{6}{l}{\small Table \thetable\ continued.}\\" + "\n\\toprule\n" + header + "\n\\midrule\n\\endhead\n"
tablei1 += r"\midrule\multicolumn{6}{r}{\small Continued on next page.}\\" + "\n\\endfoot\n\\bottomrule\n\\endlastfoot\n"
for arm, title in arms:
    if arm == "mlp_ce":
        tablei1 += "\\midrule\n"
    values = [title] + [accuracy(d, arm, float(summary[d, arm]["accuracy_mean"]) == best[d]) for d in DATASETS]
    tablei1 += " & ".join(values) + r" \\" + "\n"
tablei1 += "\\end{longtable}\n\\endgroup\n"
write(OUT / "tables/table_I_1.tex", tablei1)

with (DATA / "theory_diagnostics.csv").open(encoding="utf-8-sig", newline="") as f:
    theory = {r["dataset"]: r for r in csv.DictReader(f)}
tablei2 = r"""\begin{table}[htbp]
\centering
\caption{Decomposition of the reduction in training-target Brier score under uniform 0.4 noise.}\label{tab:I.2}
\begingroup
\fontsize{9}{11}\selectfont
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.2}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
\textbf{Dataset} & \textbf{\shortstack{One-hot\\label}} & \textbf{\shortstack{Soft\\target}} & \textbf{\shortstack{Reduction on\\corrupted labels}} & \textbf{\shortstack{Error increase on\\correct labels}} & \textbf{\shortstack{Net\\reduction}}\\
\midrule
"""
for dataset, name in zip(DATASETS, NAMES):
    values = [name] + [f'{float(theory[dataset][key]):.4f}' for key in ["noisy_brier", "target_brier", "corrupted_gain", "clean_cost", "net_gain"]]
    tablei2 += " & ".join(values) + r" \\" + "\n"
tablei2 += "\\bottomrule\n\\end{tabular*}\n\\endgroup\n\\end{table}\n"
write(OUT / "tables/table_I_2.tex", tablei2)

write(OUT / "README.md", """# GB-LCSS manuscript

Compile `main.tex` for the article and `supplementary.tex` for Appendices A-I.
`highlights.tex` contains the separate Highlights document.

The source uses the Elsevier `elsarticle` class and standard TeX Live packages.
It can be uploaded directly to Overleaf with pdfLaTeX and BibTeX, or compiled
with Tectonic. Figures are embedded vector PDFs; table and algorithm sources
are in `tables/` and `algorithms/`.
""")
print(OUT)
print("Main equations:", re.findall(r"\\label\{eq:(\d+)\}", main))
print("New tables generated from verified five-seed CSVs.")
