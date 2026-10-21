from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "mlp18_graphdomain_tablec_20260708"


LABEL_MAP = {
    "gcn": "GCN(X)",
    "gcn_xh_ce": "GCN[X||H]",
    "nrgnn": "NRGNN",
    "nrgnn_xh": "NRGNN[X||H]",
    "tfr_gcn": "TFR-GCN",
    "procon": "ProCon",
    "rncgln": "RNCGLN",
    "rtgnn": "RTGNN",
    "clnode": "CLNode",
    "crgnn": "CRGNN",
}


def _append_standard(rows, path, datasets=None, methods=None):
    path = Path(path)
    if not path.exists():
        return
    df = pd.read_csv(path)
    if datasets is not None:
        df = df[df["dataset"].isin(datasets)]
    if methods is not None:
        df = df[df["method"].isin(methods)]
    if "status" in df.columns:
        df = df[df["status"].astype(str).eq("ok")]
    for col in ["modified_val_count", "modified_test_count"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").fillna(-1)
            if values.ne(0).any():
                raise RuntimeError(f"protocol violation in {path}: {col} is nonzero")
    for _, row in df.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "method": LABEL_MAP.get(row["method"], row["method"]),
                "raw_method": row["method"],
                "seed": int(row["seed"]),
                "test": float(row["test"]),
                "inference_graph": "yes",
                "source_file": str(path.relative_to(ROOT)),
                "protocol_audit": (
                    "modified_val_count=0; modified_test_count=0"
                    if "modified_val_count" in df.columns
                    else "not_available"
                ),
            }
        )


def _append_mlp9_distill(rows, path):
    path = Path(path)
    df = pd.read_csv(path)
    df = df[
        df["method"].eq("mlp_distill_softdist_xh")
        & df["noise_type"].eq("uniform")
        & (pd.to_numeric(df["noise_rate"]) == 0.4)
    ]
    for _, row in df.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "method": "LCSS-distill X-only student",
                "raw_method": row["method"],
                "seed": int(row["seed"]),
                "test": float(row["student_test"]),
                "inference_graph": "no",
                "source_file": str(path.relative_to(ROOT)),
                "protocol_audit": "distill runner source; noise-count columns not present",
            }
        )


def _append_standard_distill(rows, path):
    path = Path(path)
    df = pd.read_csv(path)
    df = df[
        df["method"].eq("mlp_distill_softdist_xh")
        & df["noise_type"].eq("uniform")
        & (pd.to_numeric(df["noise_rate"]) == 0.4)
    ]
    for col in ["modified_val_count", "modified_test_count"]:
        values = pd.to_numeric(df[col], errors="coerce").fillna(-1)
        if values.ne(0).any():
            raise RuntimeError(f"protocol violation in {path}: {col} is nonzero")
    for _, row in df.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "method": "LCSS-distill X-only student",
                "raw_method": row["method"],
                "seed": int(row["seed"]),
                "test": float(row["test"]),
                "inference_graph": "no",
                "source_file": str(path.relative_to(ROOT)),
                "protocol_audit": "modified_val_count=0; modified_test_count=0",
            }
        )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    _append_standard(rows, RUN_DIR / "clean_core_cora_citeseer_uniform0p4_5seed.csv")
    _append_standard(rows, RUN_DIR / "clean_core_coauthorcs_uniform0p4_5seed.csv")
    _append_standard(rows, RUN_DIR / "fill_core_pubmed_amazon_uniform0p4_5seed.csv")
    _append_standard(rows, RUN_DIR / "fill_nrgnn_xh_pubmed_amazon_uniform0p4_5seed.csv")
    _append_standard(
        rows,
        ROOT
        / "runs"
        / "procon_tfr_same_protocol_20260707"
        / "procon_tfr_uniform0p4_seeds_0_1_2_3_4.csv",
    )
    _append_standard(rows, RUN_DIR / "phaseB_uniform0p4_small_5seed.csv")
    _append_standard(
        rows,
        RUN_DIR / "phaseB_uniform0p4_large_5seed.csv",
        datasets=["pubmed"],
        methods=["rtgnn", "clnode", "crgnn"],
    )
    _append_standard(
        rows,
        RUN_DIR / "phaseA_seed0_smoke_rest.csv",
        datasets=["coauthorcs", "amazonpho"],
        methods=["rtgnn", "clnode", "crgnn"],
    )
    _append_standard(rows, RUN_DIR / "phaseA_seed0_smoke.csv", methods=["rncgln"])
    _append_standard(rows, RUN_DIR / "phaseA_seed0_smoke_rncgln_amazonpho.csv", methods=["rncgln"])
    _append_standard(rows, RUN_DIR / "clean_coauthorcs_rtgnn_clnode_crgnn_seeds1_4.csv")
    _append_standard(rows, RUN_DIR / "fill_coauthorcs_clnode_crgnn_seed1.csv")
    _append_standard(rows, RUN_DIR / "fill_coauthorcs_rtgnn_clnode_crgnn_seeds2_4.csv")
    _append_standard(rows, RUN_DIR / "fill_amazon_rtgnn_clnode_crgnn_rncgln_seeds1_4.csv")
    _append_standard(rows, RUN_DIR / "fill_rncgln_cora_citeseer_seeds1_4.csv", methods=["rncgln"])
    _append_standard(rows, RUN_DIR / "fill_coauthorcs_tfr_procon_seed0.csv")
    _append_standard(rows, RUN_DIR / "fill_coauthorcs_tfr_procon_seeds1_4.csv")
    _append_standard(rows, RUN_DIR / "fill_pubmed_amazon_tfr_procon_seed0.csv")
    _append_standard(rows, RUN_DIR / "fill_pubmed_amazon_tfr_procon_seeds1_4.csv")
    _append_standard(rows, RUN_DIR / "fill_rncgln_pubmed_coauthorcs_seed0.csv")
    _append_mlp9_distill(rows, ROOT / "runs" / "mlp9_denoised_teacher_20260702" / "mlp9_combined_raw.csv")
    _append_standard_distill(
        rows,
        ROOT
        / "runs"
        / "mlp16_distill_tableb_completion_20260707"
        / "distill_tableb_uniform0p4_seeds_0_1_2_3_4.csv",
    )

    raw = pd.DataFrame(rows)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RUN_DIR / "clean_tablec_raw.csv", index=False)

    aggregate = raw.groupby(["dataset", "method", "raw_method", "inference_graph"], as_index=False).agg(
        n=("test", "count"),
        mean=("test", "mean"),
        std=("test", "std"),
        source_file=("source_file", lambda values: "; ".join(sorted(set(values)))),
    )
    aggregate["mean_pct"] = aggregate["mean"] * 100
    aggregate["std_pct"] = aggregate["std"].fillna(0) * 100
    aggregate.to_csv(RUN_DIR / "clean_tablec_aggregate.csv", index=False)

    method_order = [
        "GCN(X)",
        "GCN[X||H]",
        "NRGNN",
        "NRGNN[X||H]",
        "TFR-GCN",
        "ProCon",
        "RNCGLN",
        "RTGNN",
        "CLNode",
        "CRGNN",
        "LCSS-distill X-only student",
    ]
    dataset_order = ["cora", "citeseer", "coauthorcs", "pubmed", "amazonpho"]
    view = aggregate.copy()
    view["cell"] = view.apply(
        lambda row: f"{row['mean_pct']:.2f}±{row['std_pct']:.2f}"
        if row["n"] > 1
        else f"{row['mean_pct']:.2f}†",
        axis=1,
    )

    lines = [
        "# Clean Table C candidate (uniform 0.4, train-only noise)",
        "",
        "Old phase24/tfr/coauthorcs-baseline graph-dependent numbers are invalidated because they were generated before train-only `eligible_mask` support and show full-graph label corruption.",
        "",
        "| Method | Inference graph? | Cora | CiteSeer | Coauthor-CS | PubMed | Amazon-Photo |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in method_order:
        sub = view[view["method"].eq(method)]
        if sub.empty:
            continue
        inference_graph = sub["inference_graph"].iloc[0]
        cells = []
        for dataset in dataset_order:
            hit = sub[sub["dataset"].eq(dataset)]
            cells.append(hit["cell"].iloc[0] if len(hit) else "—")
        lines.append(f"| {method} | {inference_graph} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "Protocol audit: all newly rerun graph-dependent rows used `scripts/run_standard_downstream.py --noise_scope train` and have `modified_val_count=0`, `modified_test_count=0`. MLP-9 distill rows do not carry noise-count columns in their raw file; MLP-16 distill rows do and pass the same audit.",
            "",
            "## Feasibility / exclusions",
            "",
            "- Cells with n=1 are seed0-only feasibility entries; all n values are available in `clean_tablec_aggregate.csv`.",
            "- RNCGLN on PubMed and Coauthor-CS is kept as seed0-only because PubMed is slow and Coauthor-CS collapses to 0.00 in seed0.",
            "- TFR-GCN/ProCon on PubMed/Amazon-Photo and Coauthor-CS are adapted template runs, not official tuned results; they are included to avoid NA cells and should be footnoted as adapted.",
            "",
            "## Aggregate CSV",
            "",
            "See `clean_tablec_aggregate.csv` and `clean_tablec_raw.csv`.",
        ]
    )
    report = "\n".join(lines)
    (RUN_DIR / "clean_tablec_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
