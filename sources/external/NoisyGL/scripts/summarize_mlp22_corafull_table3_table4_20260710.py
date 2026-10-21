from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "mlp22_corafull_table3_table4_20260710"

TABLE3_FILES = [
    RUN_DIR / "table3_corafull_uniform0p4_seeds_0_4.csv",
    RUN_DIR / "table3_corafull_uniform0p6_seeds_0_4.csv",
]
TABLE4_FILES = [
    RUN_DIR / "table4_corafull_gcn_nrgnn_uniform0p4_seeds_0_4.csv",
    RUN_DIR / "table4_corafull_nrgnn_xh_uniform0p4_seeds_0_4.csv",
    RUN_DIR / "table4_corafull_tfr_gcn_uniform0p4_seeds_0_4.csv",
    RUN_DIR / "table4_corafull_gd2_uniform0p4_seeds_0_4.csv",
    RUN_DIR / "table4_corafull_procon_uniform0p4_seeds_0_4.csv",
]

TABLE3_LABELS = {
    "mlp_distill_gcn": "GCN(X)+CE（GLNN 式）",
    "mlp_distill_gcn_xh": "GCN[X‖H]+CE",
    "mlp_distill_softdist_xh_shuffled": "随机分组共识教师",
    "mlp_distill_knn_xh": "kNN 共识教师",
    "mlp_distill_softdist_xh": "球式共识教师（本文）",
}

TABLE4_LABELS = {
    "gcn": "GCN(X)",
    "gcn_xh_ce": "GCN[X‖H]",
    "nrgnn": "NRGNN",
    "nrgnn_xh": "NRGNN[X‖H]",
    "tfr_gcn": "TFR-GCN",
    "procon": "ProCon",
    "gd2": "GD2",
    "mlp_distill_softdist_xh": "LCSS 蒸馏扩展（学生）",
}


def _read_many(paths: list[Path]) -> pd.DataFrame:
    frames = []
    missing = []
    for path in paths:
        if not path.exists():
            missing.append(str(path.relative_to(REPO_ROOT)))
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = str(path.relative_to(REPO_ROOT))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No input CSV files found.")
    df = pd.concat(frames, ignore_index=True)
    if missing:
        df.attrs["missing"] = missing
    return df


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "method", "noise_type", "noise_rate"], as_index=False)
        .agg(
            n=("test", "count"),
            mean=("test", "mean"),
            std=("test", "std"),
            ok=("status", lambda s: int((s == "ok").sum())),
            modified_val_count=("modified_val_count", "max"),
            modified_test_count=("modified_test_count", "max"),
        )
        .sort_values(["dataset", "noise_rate", "method"])
    )


def _cell(agg: pd.DataFrame, method: str, noise_rate: float) -> str:
    rows = agg[(agg["method"] == method) & (agg["noise_rate"] == noise_rate)]
    if rows.empty:
        return "NA"
    row = rows.iloc[0]
    return f"{100 * row['mean']:.2f}±{100 * row['std']:.2f}"


def _markdown_table3(agg: pd.DataFrame) -> str:
    order = [
        "mlp_distill_gcn",
        "mlp_distill_gcn_xh",
        "mlp_distill_softdist_xh_shuffled",
        "mlp_distill_knn_xh",
        "mlp_distill_softdist_xh",
    ]
    lines = [
        "| 教师 | CoraFull u0.4 | CoraFull u0.6 |",
        "|---|---:|---:|",
    ]
    for method in order:
        lines.append(
            f"| {TABLE3_LABELS[method]} | {_cell(agg, method, 0.4)} | {_cell(agg, method, 0.6)} |"
        )
    return "\n".join(lines)


def _markdown_table4(agg4: pd.DataFrame, agg3: pd.DataFrame) -> str:
    order = [
        "gcn",
        "gcn_xh_ce",
        "nrgnn",
        "nrgnn_xh",
        "tfr_gcn",
        "procon",
        "gd2",
        "mlp_distill_softdist_xh",
    ]
    combined = pd.concat([agg4, agg3[agg3["method"].eq("mlp_distill_softdist_xh")]], ignore_index=True)
    lines = [
        "| 方法 | 推理需图 | CoraFull u0.4 |",
        "|---|---|---:|",
    ]
    for method in order:
        infer_graph = "否" if method == "mlp_distill_softdist_xh" else "是"
        lines.append(f"| {TABLE4_LABELS[method]} | {infer_graph} | {_cell(combined, method, 0.4)} |")
    return "\n".join(lines)


def _paired_delta(df: pd.DataFrame, base: str) -> pd.DataFrame:
    wide = df.pivot_table(index=["noise_rate", "seed"], columns="method", values="test")
    out = []
    for method in wide.columns:
        if method == base:
            continue
        delta = wide[method] - wide[base]
        valid = delta.dropna()
        if valid.empty:
            continue
        out.append(
            {
                "method": method,
                "base": base,
                "n": len(valid),
                "mean_delta": valid.mean(),
                "positive": int((valid > 0).sum()),
                "negative": int((valid < 0).sum()),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    table3 = _read_many(TABLE3_FILES)
    table4 = _read_many(TABLE4_FILES)

    agg3 = _agg(table3)
    agg4 = _agg(table4)
    agg = pd.concat([agg3.assign(table="table3"), agg4.assign(table="table4")], ignore_index=True)
    agg.to_csv(RUN_DIR / "mlp22_aggregate.csv", index=False)

    paired3 = _paired_delta(table3, "mlp_distill_softdist_xh")
    paired3.to_csv(RUN_DIR / "mlp22_table3_paired_delta_vs_ball_teacher.csv", index=False)

    report = [
        "# MLP-22 CoraFull Table 3/4 Completion",
        "",
        "## Table 3 Replacement Cells",
        "",
        _markdown_table3(agg3),
        "",
        "## Table 4 Replacement Cells",
        "",
        _markdown_table4(agg4, agg3),
        "",
        "## Table 3 Paired Deltas Against Ball Teacher",
        "",
        paired3.to_markdown(index=False),
        "",
        "## Protocol Audit",
        "",
        f"- Table 3 rows: {len(table3)}; all status ok: {bool((table3['status'] == 'ok').all())}.",
        f"- Table 4 rows: {len(table4)}; all status ok: {bool((table4['status'] == 'ok').all())}.",
        f"- Table 3 max modified val/test: {int(table3['modified_val_count'].max())}/{int(table3['modified_test_count'].max())}.",
        f"- Table 4 max modified val/test: {int(table4['modified_val_count'].max())}/{int(table4['modified_test_count'].max())}.",
    ]
    (RUN_DIR / "mlp22_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
