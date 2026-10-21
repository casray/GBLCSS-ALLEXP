from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASETS = ["cora", "citeseer", "pubmed", "coauthorcs", "amazonpho"]
DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "coauthorcs": "Coauthor-CS",
    "amazonpho": "Amazon-Photo",
}
NOISES = [
    ("uniform", 0.2, "Uniform 0.2"),
    ("uniform", 0.4, "Uniform 0.4"),
    ("uniform", 0.6, "Uniform 0.6"),
    ("pair", 0.4, "Pair 0.4"),
    ("instance", 0.4, "Instance 0.4"),
]
METHOD_ORDER = [
    "mlp_ce",
    "mlp_coteaching",
    "mlp_small_loss",
    "mlp_sce",
    "mlp_elr",
    "mlp_jocor",
    "mlp_dividemix_lite",
    "mlp_fine",
    "mlp_logitclip",
    "mlp_anl",
    "mlp_ct",
    "frozengrace_gb_mlp_softdist_beta05",
    "frozengrace_gb_mlp_softdist_beta05_shuffled",
]
METHOD_LABELS = {
    "mlp_ce": "MLP+CE",
    "mlp_coteaching": "Co-teaching (2018)",
    "mlp_small_loss": "Small-loss (2018)",
    "mlp_sce": "SCE (2019)",
    "mlp_elr": "ELR (2020)",
    "mlp_jocor": "JoCoR (2020)",
    "mlp_dividemix_lite": "DivideMix-lite (2020, adapted)",
    "mlp_fine": "FINE (2021, adapted)",
    "mlp_logitclip": "LogitClip (2023, adapted)",
    "mlp_anl": "ANL (2023, adapted)",
    "mlp_ct": "CT (2025, adapted)",
    "frozengrace_gb_mlp_softdist_beta05": "LCSS",
    "frozengrace_gb_mlp_softdist_beta05_shuffled": "Shuffled",
}
GRAPHFREE_METHOD_ORDER = [
    "mlp_ce",
    "graphmlp",
    "gsdn",
    "frozengrace_gb_mlp_softdist_beta05",
    "gcn_ce",
    "gcn_xh_ce",
    "mlp_distill_gcn",
    "mlp_distill_gcn_xh",
    "mlp_distill_softdist_xh",
    "mlp_distill_knn_xh",
]
GRAPHFREE_LABELS = {
    "mlp_ce": "Vanilla MLP",
    "graphmlp": "Graph-MLP (adapted)",
    "gsdn": "GSDN (adapted)",
    "frozengrace_gb_mlp_softdist_beta05": "LCSS train-only",
    "gcn_ce": "GCN(X)+CE reference",
    "gcn_xh_ce": "GCN[X||H]+CE reference",
    "mlp_distill_gcn": "GLNN-style GCN teacher",
    "mlp_distill_gcn_xh": "GLNN-style GCN[X||H] teacher",
    "mlp_distill_softdist_xh": "LCSS-distill (T3 ball teacher)",
    "mlp_distill_knn_xh": "kNN-consensus distill (T4)",
}
PALETTE = {
    "LCSS": "#0072B2",
    "LCSS train-only": "#0072B2",
    "Shuffled": "#999999",
    "MLP+CE": "#555555",
    "Graph-MLP": "#D55E00",
    "GSDN": "#CC79A7",
    "Small-loss": "#009E73",
    "JoCoR": "#E69F00",
    "kNN-consensus distill": "#56B4E9",
    "LCSS-distill": "#0072B2",
}


@dataclass
class ManifestEntry:
    artifact: str
    sources: str
    check: str
    note: str = ""


def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def pct(mean: float | str, std: float | str | None = None) -> str:
    if pd.isna(mean):
        return "NA"
    m = float(mean) * 100.0
    if std is None or pd.isna(std):
        return f"{m:.2f}"
    return f"{m:.2f}±{float(std) * 100.0:.2f}"


def fmt_cell(mean: float, std: float, bold: bool = False, star: bool = False) -> str:
    text = pct(mean, std)
    if star:
        text = f"{text}*"
    if bold:
        return f"**{text}**"
    return text


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def latex_cell(mean: float, std: float, bold: bool = False, star: bool = False) -> str:
    text = pct(mean, std)
    if star:
        text = f"{text}$^*$"
    if bold:
        return rf"\textbf{{{text}}}"
    return text


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_booktabs(df: pd.DataFrame, caption: str | None = None, label: str | None = None) -> str:
    latex = df.to_latex(index=False, escape=False)
    lines = latex.splitlines()
    if caption or label:
        wrapped = ["\\begin{table*}[t]", "\\centering"]
        if caption:
            wrapped.append(f"\\caption{{{latex_escape(caption)}}}")
        if label:
            wrapped.append(f"\\label{{{label}}}")
        wrapped.extend(lines)
        wrapped.append("\\end{table*}")
        return "\n".join(wrapped) + "\n"
    return latex + "\n"


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    columns = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([str(row[c]) for c in df.columns])
    def clean(value: str) -> str:
        return value.replace("\n", "<br>").replace("|", "\\|")
    lines = []
    lines.append("| " + " | ".join(clean(c) for c in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(clean(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def combine_main_aggregates() -> pd.DataFrame:
    mlp10 = read_csv(REPO_ROOT / "runs/mlp10_baseline_refresh_20260702/mlp10_aggregate.csv")
    mlp12 = read_csv(REPO_ROOT / "runs/mlp12_amazonphoto_20260702/mlp12_aggregate.csv")
    combined = pd.concat([mlp10, mlp12], ignore_index=True)
    combined["noise_rate"] = combined["noise_rate"].astype(float)
    combined = combined.drop_duplicates(["dataset", "noise_type", "noise_rate", "method"], keep="last")
    return combined


def load_p0b() -> pd.DataFrame:
    path = REPO_ROOT / "runs/mlp15_review_defense_20260704/p0b_paired_significance.csv"
    df = read_csv(path)
    df["noise_rate"] = df["noise_rate"].astype(float)
    return df


def p0b_star(p0b: pd.DataFrame, dataset: str, noise_type: str, noise_rate: float, baseline: str) -> bool:
    rows = p0b[
        (p0b["dataset"].astype(str).str.lower() == dataset)
        & (p0b["noise_type"].astype(str).str.lower() == noise_type)
        & (p0b["noise_rate"].astype(float).round(6) == round(float(noise_rate), 6))
        & (p0b["baseline"].astype(str) == baseline)
    ]
    if rows.empty:
        return False
    return bool(float(rows.iloc[0]["paired_t_p"]) < 0.05)


def generate_main_table(out_dir: Path, manifest: list[ManifestEntry]) -> None:
    df = combine_main_aggregates()
    p0b = load_p0b()
    md_parts = [
        "# Appendix A. Complete main table\n",
        "Values are 5-seed test accuracy mean±std (%). Bold marks the best value in each dataset/noise column. "
        "`*` marks LCSS vs the strongest non-LCSS non-shuffled baseline with paired t-test p<0.05.\n",
    ]
    tex_parts = [
        "% Appendix A. Complete main table. Auto-generated by scripts/generate_mlp17_paper_assets.py\n",
    ]
    checks = []
    for dataset in DATASETS:
        rows_md = []
        rows_tex = []
        sub = df[df["dataset"].astype(str).str.lower() == dataset]
        for method in METHOD_ORDER:
            row_md = {"Method": METHOD_LABELS[method]}
            row_tex = {"Method": latex_escape(METHOD_LABELS[method])}
            for noise_type, noise_rate, noise_label in NOISES:
                col = sub[
                    (sub["noise_type"].astype(str).str.lower() == noise_type)
                    & (sub["noise_rate"].astype(float).round(6) == round(noise_rate, 6))
                ]
                method_row = col[col["method"].astype(str) == method]
                if method_row.empty:
                    row_md[noise_label] = "NA"
                    row_tex[noise_label] = "NA"
                    continue
                max_mean = col[col["method"].isin(METHOD_ORDER)]["mean"].astype(float).max()
                mean = float(method_row.iloc[0]["mean"])
                std = float(method_row.iloc[0]["std"])
                bold = math.isclose(mean, max_mean, rel_tol=0, abs_tol=5e-12)
                star = False
                if method == "frozengrace_gb_mlp_softdist_beta05":
                    baselines = col[
                        (~col["method"].isin(["frozengrace_gb_mlp_softdist_beta05", "frozengrace_gb_mlp_softdist_beta05_shuffled"]))
                        & (col["method"].isin(METHOD_ORDER))
                    ]
                    if not baselines.empty and mean >= float(baselines["mean"].max()) - 5e-12:
                        best_base = baselines.sort_values("mean", ascending=False).iloc[0]["method"]
                        star = p0b_star(p0b, dataset, noise_type, noise_rate, str(best_base))
                row_md[noise_label] = fmt_cell(mean, std, bold=bold, star=star)
                row_tex[noise_label] = latex_cell(mean, std, bold=bold, star=star)
            rows_md.append(row_md)
            rows_tex.append(row_tex)
        table_md = pd.DataFrame(rows_md)
        table_tex = pd.DataFrame(rows_tex)
        md_parts.append(f"\n## {DATASET_LABELS[dataset]}\n\n")
        md_parts.append(df_to_md(table_md))
        tex_parts.append(f"\n% {DATASET_LABELS[dataset]}\n")
        tex_parts.append(to_booktabs(table_tex, caption=f"Complete main table on {DATASET_LABELS[dataset]}.", label=f"tab:main-{dataset}"))
        expected = len(METHOD_ORDER) * len(NOISES)
        observed = int(sub[sub["method"].isin(METHOD_ORDER)].shape[0])
        checks.append((dataset, observed, expected))
    md_path = out_dir / "tables/appendix_A_complete_main_table.md"
    tex_path = out_dir / "tables/appendix_A_complete_main_table.tex"
    write_text(md_path, "\n".join(md_parts))
    write_text(tex_path, "\n".join(tex_parts))
    failed = [f"{d}:{o}/{e}" for d, o, e in checks if o != e]
    status = "pass" if not failed else "fail " + ", ".join(failed)
    manifest.append(ManifestEntry(str(md_path.relative_to(out_dir)), "mlp10_aggregate.csv + mlp12_aggregate.csv + p0b", status))
    manifest.append(ManifestEntry(str(tex_path.relative_to(out_dir)), "mlp10_aggregate.csv + mlp12_aggregate.csv + p0b", status))


def generate_significance_table(out_dir: Path, manifest: list[ManifestEntry]) -> None:
    df = load_p0b()
    columns = [
        "source",
        "dataset",
        "noise_type",
        "noise_rate",
        "ours_label",
        "baseline_label",
        "n",
        "mean_delta_pp",
        "positive_seeds",
        "paired_t_p",
        "wilcoxon_p",
    ]
    present = [c for c in columns if c in df.columns]
    out = df[present].copy()
    out["noise_rate"] = out["noise_rate"].astype(float).map(lambda x: f"{x:.1f}")
    if "mean_delta_pp" in out:
        out["mean_delta_pp"] = out["mean_delta_pp"].astype(float).map(lambda x: f"{x:+.2f}")
    for col in ["paired_t_p", "wilcoxon_p"]:
        if col in out:
            out[col] = out[col].astype(float).map(lambda x: f"{x:.3g}")
    md = (
        "# Appendix B. Paired significance table\n\n"
        "n=5 paired seeds. Wilcoxon p-value lower bound is 0.0625 under five paired seeds.\n\n"
        + df_to_md(out)
    )
    tex = to_booktabs(out, caption="Paired significance tests for LCSS comparisons.", label="tab:paired-significance")
    md_path = out_dir / "tables/appendix_B_paired_significance.md"
    tex_path = out_dir / "tables/appendix_B_paired_significance.tex"
    write_text(md_path, md)
    write_text(tex_path, tex)
    status = "pass" if len(out) > 0 else "fail empty"
    manifest.append(ManifestEntry(str(md_path.relative_to(out_dir)), "p0b_paired_significance.csv", status))
    manifest.append(ManifestEntry(str(tex_path.relative_to(out_dir)), "p0b_paired_significance.csv", status))


def find_csvs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.csv"))


def generate_graph_dependent_reference(out_dir: Path, manifest: list[ManifestEntry]) -> None:
    source_dirs = [
        REPO_ROOT / "runs/phase24_nrgnn_xh_20260616",
        REPO_ROOT / "runs/tfr_comparison_20260616",
        REPO_ROOT / "runs/coauthorcs_baseline_20260617",
        REPO_ROOT / "runs/2025_comparison_20260616",
    ]
    candidates = []
    hashes = []
    for src in source_dirs:
        for path in find_csvs(src):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            cols = {c.lower(): c for c in df.columns}
            method_col = cols.get("method") or cols.get("method_label")
            dataset_col = cols.get("dataset")
            test_col = cols.get("test") or cols.get("test_acc") or cols.get("mean")
            if not method_col or not dataset_col or not test_col:
                continue
            method_text = " ".join(df[method_col].astype(str).str.lower().unique().tolist())
            if not any(key in method_text for key in ["nrgnn", "tfr", "gcn_xh", "gcn"]):
                continue
            hashes.append(f"{path.relative_to(REPO_ROOT)}:{file_hash(path)}")
            keep_cols = [dataset_col, method_col, test_col]
            for optional in ["noise_type", "noise_rate", "seed", "n", "std", "train_percent", "split_type"]:
                if optional in df.columns:
                    keep_cols.append(optional)
            part = df[keep_cols].copy()
            part["source_file"] = str(path.relative_to(REPO_ROOT))
            candidates.append(part)
    if candidates:
        ref = pd.concat(candidates, ignore_index=True, sort=False)
        ref = ref.head(300)
        md = (
            "# Appendix C. Graph-dependent references\n\n"
            "**Inference requires the graph — different deployment setting; numbers are same-protocol references, not paired comparisons.**\n\n"
            + df_to_md(ref)
            + "\n\n## H1 source hashes\n\n"
            + "\n".join(f"- `{x}`" for x in hashes)
            + "\n"
        )
        tex_df = ref.astype(str)
        status = "pass hash-recorded"
    else:
        ref = pd.DataFrame({"Status": ["No compatible NRGNN/TFR reference CSV found by the extractor."]})
        md = (
            "# Appendix C. Graph-dependent references\n\n"
            "**Inference requires the graph — different deployment setting; numbers are same-protocol references, not paired comparisons.**\n\n"
            "No compatible source rows were found by the extractor. This table is intentionally marked missing; no new experiment was run.\n"
        )
        tex_df = ref
        status = "skipped missing-compatible-source"
    md_path = out_dir / "tables/appendix_C_graph_dependent_references.md"
    tex_path = out_dir / "tables/appendix_C_graph_dependent_references.tex"
    write_text(md_path, md)
    write_text(tex_path, to_booktabs(tex_df, caption="Graph-dependent reference results.", label="tab:graph-dependent-ref"))
    manifest.append(ManifestEntry(str(md_path.relative_to(out_dir)), "phase24/tfr/coauthorcs/2025 comparison CSV scan", status))
    manifest.append(ManifestEntry(str(tex_path.relative_to(out_dir)), "phase24/tfr/coauthorcs/2025 comparison CSV scan", status))


def generate_sensitivity_and_beta(out_dir: Path, manifest: list[ManifestEntry]) -> None:
    stage_a = read_csv(REPO_ROOT / "runs/mlp13_ballparam_sensitivity_20260703/mlp13_stageA_grid_summary.csv")
    stage_b = read_csv(REPO_ROOT / "runs/mlp13_ballparam_sensitivity_20260703/mlp13_stageB_summary.csv")
    beta = read_csv(REPO_ROOT / "runs/mlp15_review_defense_20260704/p0a_beta_validation_audit.csv")
    for df in [stage_a, stage_b, beta]:
        for col in df.columns:
            if df[col].dtype.kind in "fc":
                df[col] = df[col].map(lambda x: round(float(x), 6) if pd.notna(x) else x)
    md = "# Appendix D1. Construction-parameter sensitivity\n\n## Stage A seed0 grid\n\n"
    md += df_to_md(stage_a)
    md += "\n## Stage B 5-seed selected configurations\n\n"
    md += df_to_md(stage_b)
    tex = to_booktabs(stage_a.astype(str), caption="Stage-A construction parameter grid.", label="tab:param-stage-a")
    tex += "\n" + to_booktabs(stage_b.astype(str), caption="Stage-B 5-seed selected construction parameters.", label="tab:param-stage-b")
    md_path = out_dir / "tables/appendix_D1_parameter_sensitivity.md"
    tex_path = out_dir / "tables/appendix_D1_parameter_sensitivity.tex"
    write_text(md_path, md)
    write_text(tex_path, tex)
    beta_md = "# Appendix D2. Beta validation audit\n\n" + df_to_md(beta)
    beta_tex = to_booktabs(beta.astype(str), caption="Beta validation audit.", label="tab:beta-audit")
    beta_md_path = out_dir / "tables/appendix_D2_beta_audit.md"
    beta_tex_path = out_dir / "tables/appendix_D2_beta_audit.tex"
    write_text(beta_md_path, beta_md)
    write_text(beta_tex_path, beta_tex)
    stage_status = "pass" if len(stage_a) >= 27 and len(stage_b) > 0 else "fail incomplete"
    beta_status = "pass" if len(beta) > 0 else "fail empty"
    manifest.append(ManifestEntry(str(md_path.relative_to(out_dir)), "mlp13_stageA_grid_summary.csv + mlp13_stageB_summary.csv", stage_status))
    manifest.append(ManifestEntry(str(tex_path.relative_to(out_dir)), "mlp13_stageA_grid_summary.csv + mlp13_stageB_summary.csv", stage_status))
    manifest.append(ManifestEntry(str(beta_md_path.relative_to(out_dir)), "p0a_beta_validation_audit.csv", beta_status))
    manifest.append(ManifestEntry(str(beta_tex_path.relative_to(out_dir)), "p0a_beta_validation_audit.csv", beta_status))


def save_figure(fig: plt.Figure, base: Path, manifest: list[ManifestEntry], sources: str, check: str) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    manifest.append(ManifestEntry(str(base.with_suffix(".pdf").relative_to(base.parents[1])), sources, check))
    manifest.append(ManifestEntry(str(base.with_suffix(".png").relative_to(base.parents[1])), sources, check))


def short_label(method_label: str) -> str:
    label = str(method_label)
    if "Ours" in label or "SoftDist" in label:
        return "LCSS"
    if "Shuffled" in label:
        return "Shuffled"
    if "Small" in label:
        return "Small-loss"
    if "JoCoR" in label:
        return "JoCoR"
    if "MLP+CE" in label:
        return "MLP+CE"
    return label.replace(" beta=0.5 train-only", "")


def generate_feature_noise_curves(out_dir: Path, manifest: list[ManifestEntry]) -> None:
    df = read_csv(REPO_ROOT / "runs/mlp16_cutvalue_featnoise_20260706/mlp16_feature_noise_aggregate.csv")
    datasets = [d for d in DATASETS if d in set(df["dataset"].astype(str))]
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.1, 1.75), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"].astype(str) == dataset].copy()
        for method, part in sub.groupby("method"):
            part = part.sort_values("alpha")
            label = short_label(part.iloc[0]["method_label"])
            color = PALETTE.get(label, None)
            x = part["alpha"].astype(float).to_numpy()
            y = part["mean"].astype(float).to_numpy() * 100.0
            s = part["std"].astype(float).to_numpy() * 100.0
            ax.plot(x, y, marker="o", markersize=2.8, linewidth=1.1, label=label, color=color)
            ax.fill_between(x, y - s, y + s, alpha=0.12, color=color)
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=7)
        ax.set_xlabel(r"Feature noise $\alpha$")
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    axes[0].set_ylabel("Test accuracy (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=5, fontsize=6)
    fig.tight_layout(w_pad=0.8)
    save_figure(fig, out_dir / "figures/appendix_E_feature_noise_curves", manifest, "mlp16_feature_noise_aggregate.csv", "pass")


def generate_cut_value_table(out_dir: Path, manifest: list[ManifestEntry]) -> None:
    agg = read_csv(REPO_ROOT / "runs/mlp16_cutvalue_featnoise_20260706/mlp16_cut_value_aggregate.csv")
    raw = read_csv(REPO_ROOT / "runs/mlp16_cutvalue_featnoise_20260706/mlp16_cut_value_raw.csv")
    table = agg.copy()
    for col in ["cv_mean", "cv_std", "mean", "std"]:
        if col in table:
            table[col] = table[col].astype(float).map(lambda x: f"{x:.4f}")
    md = (
        "# Appendix F. Cut Value table\n\n"
        "Footnote: high CV for CE/shuffled can be contaminated by prediction collapse under noisy-label protocols. "
        "CV is therefore not used as a clean structure metric here; the distill-vs-train-only rows are the only mechanism evidence quoted in the manuscript.\n\n"
        + df_to_md(table)
        + "\n\nRaw per-seed rows are preserved in `mlp16_cut_value_raw.csv`.\n"
    )
    tex = to_booktabs(table.astype(str), caption="Cut Value audit under label noise.", label="tab:cut-value")
    md_path = out_dir / "tables/appendix_F_cut_value.md"
    tex_path = out_dir / "tables/appendix_F_cut_value.tex"
    write_text(md_path, md)
    write_text(tex_path, tex)
    status = "pass" if len(agg) > 0 and len(raw) > 0 else "fail empty"
    manifest.append(ManifestEntry(str(md_path.relative_to(out_dir)), "mlp16_cut_value_aggregate.csv + raw", status))
    manifest.append(ManifestEntry(str(tex_path.relative_to(out_dir)), "mlp16_cut_value_aggregate.csv + raw", status))


def resolve_embedding_path(dataset: str) -> Path:
    base = REPO_ROOT.parent.parent / "gb_graph_aware_minival_6_1" / "checkpoints"
    mapping = {
        "cora": base / "embeddings/Cora/ratio_train0p2_val0p1_split0_pretrain0/H.pt",
        "citeseer": base / "embeddings/CiteSeer/ratio_train0p2_val0p1_split0_pretrain0/H.pt",
        "pubmed": base / "embeddings_v2/PubMed/grace/H.pt",
    }
    return mapping[dataset]


def generate_tsne_figure(out_dir: Path, manifest: list[ManifestEntry]) -> None:
    try:
        import torch
        from scripts.run_ssl_swap_downstream import _load_dataset
        from utils.frozen_grace_targets import load_frozen_embeddings
        from utils.labelnoise import label_process
        from utils.ordinary_gb import build_ordinary_granular_balls
        from utils.tools import setup_seed
    except Exception as exc:
        md = f"# Appendix G. t-SNE\n\nSkipped: dependency import failed: `{exc}`.\n"
        write_text(out_dir / "figures/appendix_G_tsne_skipped.md", md)
        manifest.append(ManifestEntry("figures/appendix_G_tsne_skipped.md", "GRACE H + NoisyGL loader", "skipped import-failed", str(exc)))
        return
    panels = []
    for dataset in ["cora", "citeseer"]:
        try:
            setup_seed(0)
            data = _load_dataset(dataset, device="cpu", split_type="percent", train_percent=0.2, val_percent=0.1, test_percent=0.7)
            h_path = resolve_embedding_path(dataset)
            h = load_frozen_embeddings(h_path, expected_nodes=int(data.n_nodes))
            labels = data.labels.detach().cpu().long()
            noisy, _ = label_process(
                labels=data.labels,
                features=data.feats,
                n_classes=data.n_classes,
                noise_type="uniform",
                noise_rate=0.4,
                random_seed=0,
                eligible_mask=data.train_masks,
                debug=False,
            )
            gb = build_ordinary_granular_balls(
                h,
                max_balls=64,
                min_ball_size=32,
                split_improvement=0.05,
                random_state=0,
            )
            n = min(2500, int(h.shape[0]))
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(int(h.shape[0]), size=n, replace=False)) if int(h.shape[0]) > n else np.arange(int(h.shape[0]))
            z = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=0).fit_transform(h[idx].numpy())
            y_clean = labels[idx].numpy()
            y_noisy = noisy.detach().cpu().long()[idx].numpy()
            train_mask = data.train_masks.detach().cpu().bool().numpy()[idx]
            changed_train = train_mask & (y_clean != y_noisy)
            panels.append((dataset, z, y_clean, y_noisy, np.asarray(gb.assignment)[idx], train_mask, changed_train))
        except Exception as exc:
            write_text(out_dir / f"figures/appendix_G_tsne_{dataset}_skipped.md", f"Skipped {dataset}: `{exc}`\n")
            manifest.append(ManifestEntry(f"figures/appendix_G_tsne_{dataset}_skipped.md", "GRACE H + NoisyGL loader", "skipped", str(exc)))
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 3, figsize=(7.1, 2.2 * len(panels)), squeeze=False)
    titles = ["Clean labels (all nodes)", "Noisy labels on train nodes", "Local groups"]
    for row, (dataset, z, y_clean, y_noisy, groups, train_mask, changed_train) in enumerate(panels):
        values = [y_clean, y_noisy, groups]
        for col in range(3):
            ax = axes[row][col]
            if col == 1:
                ax.scatter(z[:, 0], z[:, 1], c="#d5dbe3", s=2, alpha=0.55, linewidths=0)
                if np.any(train_mask):
                    ax.scatter(
                        z[train_mask, 0],
                        z[train_mask, 1],
                        c=y_noisy[train_mask],
                        s=8,
                        cmap="tab20",
                        alpha=0.88,
                        linewidths=0,
                    )
                if np.any(changed_train):
                    ax.scatter(
                        z[changed_train, 0],
                        z[changed_train, 1],
                        facecolors="none",
                        edgecolors="black",
                        s=22,
                        linewidths=0.45,
                    )
            else:
                ax.scatter(z[:, 0], z[:, 1], c=values[col], s=3, cmap="tab20", alpha=0.82, linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{DATASET_LABELS[dataset]} · {titles[col]}", fontsize=7)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.tight_layout(w_pad=0.15, h_pad=0.45)
    save_figure(fig, out_dir / "figures/appendix_G_tsne_three_coloring", manifest, "GRACE H + seed0 uniform0.4 labels + recomputed balls", "pass illustrative")


def generate_labelrate_curve(out_dir: Path, manifest: list[ManifestEntry]) -> None:
    df = read_csv(REPO_ROOT / "runs/mlp6_fairness_labelrate_20260701/mlp6_aggregate.csv")
    df = df[df["table"].astype(str).eq("labelrate")].copy()
    df["train_percent"] = df["train_percent"].astype(float)
    datasets = ["cora", "citeseer"]
    methods = [
        "frozengrace_gb_mlp_softdist_beta05",
        "frozengrace_gb_mlp_softdist_beta05_shuffled",
        "mlp_small_loss",
        "mlp_jocor",
        "mlp_fine",
    ]
    labels = {
        "frozengrace_gb_mlp_softdist_beta05": "LCSS",
        "frozengrace_gb_mlp_softdist_beta05_shuffled": "Shuffled",
        "mlp_small_loss": "Small-loss",
        "mlp_jocor": "JoCoR",
        "mlp_fine": "FINE",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.15), sharey=False)
    monotonic = {}
    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"].astype(str).eq(dataset)]
        for method in methods:
            part = sub[sub["method"].astype(str).eq(method)].sort_values("train_percent")
            if part.empty:
                continue
            label = labels[method]
            color = PALETTE.get(label, None)
            x = part["train_percent"].astype(float).to_numpy() * 100.0
            y = part["mean"].astype(float).to_numpy() * 100.0
            s = part["std"].astype(float).to_numpy() * 100.0
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.1, label=label, color=color)
            ax.fill_between(x, y - s, y + s, alpha=0.12, color=color)
        # H2: LCSS delta to strongest selection family.
        deltas = []
        for tp in sorted(sub["train_percent"].astype(float).unique()):
            cur = sub[sub["train_percent"].astype(float).eq(tp)]
            ours = cur[cur["method"].astype(str).eq("frozengrace_gb_mlp_softdist_beta05")]
            sels = cur[cur["method"].astype(str).isin(["mlp_small_loss", "mlp_jocor", "mlp_fine"])]
            if ours.empty or sels.empty:
                continue
            deltas.append(float(ours.iloc[0]["mean"]) - float(sels["mean"].astype(float).max()))
        monotonic[dataset] = all(deltas[i] >= deltas[i + 1] - 1e-12 for i in range(len(deltas) - 1)) if len(deltas) >= 2 else False
        ax.set_title(DATASET_LABELS[dataset], fontsize=7)
        ax.set_xlabel("Train ratio (%)")
        ax.set_ylabel("Test accuracy (%)")
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    handles, labels_list = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=5, fontsize=6)
    fig.tight_layout(w_pad=1.0)
    note = "H2 strict monotonic decrease toward high-label end: " + ", ".join(f"{k}={v}" for k, v in monotonic.items())
    save_figure(fig, out_dir / "figures/appendix_H_labelrate_curves", manifest, "mlp6_aggregate.csv", "pass; " + note)
    write_text(out_dir / "tables/appendix_H_labelrate_monotonicity.md", "# H2 monotonicity check\n\n" + note + "\n")
    manifest.append(ManifestEntry("tables/appendix_H_labelrate_monotonicity.md", "mlp6_aggregate.csv", "pass", note))


def generate_graphfree_table(out_dir: Path, manifest: list[ManifestEntry]) -> None:
    agg = read_csv(REPO_ROOT / "runs/mlp14_graphfree_family_20260703/mlp14_aggregate.csv")
    agg["noise_rate"] = agg["noise_rate"].astype(float)
    sub = agg[(agg["noise_type"].eq("uniform")) & (agg["noise_rate"].round(6).eq(0.4))].copy()
    mlp9 = read_csv(REPO_ROOT / "runs/mlp9_denoised_teacher_20260702/mlp9_aggregate.csv")
    mlp9["noise_rate"] = mlp9["noise_rate"].astype(float)
    mlp9 = mlp9[(mlp9["noise_type"].eq("uniform")) & (mlp9["noise_rate"].round(6).eq(0.4))].copy()
    rows = []
    for method in GRAPHFREE_METHOD_ORDER:
        row = {"Method": GRAPHFREE_LABELS[method]}
        for dataset in DATASETS:
            if method in {"mlp_distill_gcn", "mlp_distill_gcn_xh", "mlp_distill_softdist_xh", "mlp_distill_knn_xh"}:
                part = mlp9[(mlp9["dataset"].eq(dataset)) & (mlp9["method"].eq(method))]
                row[DATASET_LABELS[dataset]] = (
                    pct(part.iloc[0]["student_mean"], part.iloc[0]["student_std"]) if not part.empty else "NA"
                )
            else:
                part = sub[(sub["dataset"].eq(dataset)) & (sub["method"].eq(method))]
                row[DATASET_LABELS[dataset]] = pct(part.iloc[0]["mean"], part.iloc[0]["std"]) if not part.empty else "NA"
        rows.append(row)
    table = pd.DataFrame(rows)
    md = "# Table B. Graph-free family comparison (uniform 0.4)\n\n" + df_to_md(table)
    tex = to_booktabs(table, caption="Graph-free family comparison under uniform 0.4.", label="tab:graphfree-family")
    md_path = out_dir / "tables/table_B_graphfree_family.md"
    tex_path = out_dir / "tables/table_B_graphfree_family.tex"
    write_text(md_path, md)
    write_text(tex_path, tex)
    t3 = table[table["Method"].str.contains("LCSS-distill", regex=False)]
    t4 = table[table["Method"].str.contains("kNN-consensus", regex=False)]
    t3_has_values = (not t3.empty) and all(t3.iloc[0][DATASET_LABELS[d]] != "NA" for d in ["cora", "citeseer", "pubmed"])
    t4_has_values = (not t4.empty) and all(t4.iloc[0][DATASET_LABELS[d]] != "NA" for d in ["cora", "citeseer", "pubmed"])
    status = "pass T3/T4-separated" if t3_has_values and t4_has_values else "fail T3/T4-not-separated-or-missing"
    manifest.append(ManifestEntry(str(md_path.relative_to(out_dir)), "mlp14_aggregate.csv", status))
    manifest.append(ManifestEntry(str(tex_path.relative_to(out_dir)), "mlp14_aggregate.csv", status))


def generate_distill_extension_table(out_dir: Path, manifest: list[ManifestEntry]) -> None:
    agg = read_csv(REPO_ROOT / "runs/mlp9_denoised_teacher_20260702/mlp9_aggregate.csv")
    rows = []
    for _, r in agg.sort_values(["dataset", "noise_rate", "method"]).iterrows():
        rows.append(
            {
                "Dataset": DATASET_LABELS.get(str(r["dataset"]), str(r["dataset"])),
                "Noise": f"{r['noise_type']}{float(r['noise_rate']):.1f}",
                "Arm": r["method_label"],
                "Student": pct(r["student_mean"], r["student_std"]),
                "Teacher": pct(r["teacher_mean"], r["teacher_std"]),
                "n": int(r["n"]),
            }
        )
    table = pd.DataFrame(rows)
    decomp_path = REPO_ROOT / "runs/mlp9_denoised_teacher_20260702/mlp9_decomposition_deltas.csv"
    decomp = read_csv(decomp_path) if decomp_path.exists() else pd.DataFrame()
    md = "# Extension table. Denoised-teacher distillation\n\n" + df_to_md(table)
    if not decomp.empty:
        md += "\n## Decomposition deltas\n\n" + df_to_md(decomp)
    tex = to_booktabs(table, caption="Denoised-teacher distillation arms.", label="tab:distill-arms")
    if not decomp.empty:
        tex += "\n" + to_booktabs(decomp.astype(str), caption="Distillation decomposition deltas.", label="tab:distill-decomp")
    md_path = out_dir / "tables/extension_denoised_teacher_distill.md"
    tex_path = out_dir / "tables/extension_denoised_teacher_distill.tex"
    write_text(md_path, md)
    write_text(tex_path, tex)
    methods = set(agg["method"].astype(str))
    status = "pass T3/T4-present" if {"mlp_distill_softdist_xh", "mlp_distill_knn_xh"}.issubset(methods) else "fail missing T3/T4"
    manifest.append(ManifestEntry(str(md_path.relative_to(out_dir)), "mlp9_aggregate.csv + decomposition", status))
    manifest.append(ManifestEntry(str(tex_path.relative_to(out_dir)), "mlp9_aggregate.csv + decomposition", status))


def write_manifest(out_dir: Path, entries: list[ManifestEntry]) -> None:
    rows = []
    for entry in entries:
        rows.append(
            {
                "artifact": entry.artifact,
                "sources": entry.sources,
                "check": entry.check,
                "note": entry.note,
            }
        )
    df = pd.DataFrame(rows)
    text = "# MLP-17 paper asset manifest\n\n"
    text += "All assets are generated from existing CSV/report artifacts. No new training was run.\n\n"
    text += df_to_md(df)
    write_text(out_dir / "asset_manifest.md", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="paper_assets_20260706")
    args = parser.parse_args()

    setup_matplotlib()
    out_dir = REPO_ROOT / args.output_dir
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    manifest: list[ManifestEntry] = []

    generate_main_table(out_dir, manifest)
    generate_significance_table(out_dir, manifest)
    generate_graph_dependent_reference(out_dir, manifest)
    generate_sensitivity_and_beta(out_dir, manifest)
    generate_feature_noise_curves(out_dir, manifest)
    generate_cut_value_table(out_dir, manifest)
    generate_tsne_figure(out_dir, manifest)
    generate_labelrate_curve(out_dir, manifest)
    generate_graphfree_table(out_dir, manifest)
    generate_distill_extension_table(out_dir, manifest)
    write_manifest(out_dir, manifest)

    print(f"Wrote assets to {out_dir}")
    print(f"Manifest entries: {len(manifest)}")
    failures = [m for m in manifest if str(m.check).startswith("fail")]
    skipped = [m for m in manifest if "skipped" in str(m.check)]
    print(f"Failures: {len(failures)}")
    print(f"Skipped: {len(skipped)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
