import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is available in the experiment env
    stats = None


METHOD_TO_BETA = {
    "frozengrace_gb_mlp_softdist": 0.0,
    "frozengrace_gb_mlp_softdist_beta05": 0.5,
    "frozengrace_gb_mlp_softdist_beta1": 1.0,
}

LABELS = {
    "frozengrace_gb_mlp_softdist_beta05": "SoftDist-MLP beta=0.5",
    "frozengrace_gb_mlp_softdist_beta05_shuffled": "SoftDist-MLP shuffled",
    "frozengrace_gb_linear_softdist_beta05": "SoftDist-Linear beta=0.5",
    "frozengrace_gb_linear_softdist_beta05_shuffled": "SoftDist-Linear shuffled",
    "linear_ce": "Linear+CE",
    "mlp_ce": "MLP+CE",
    "mlp_anl": "ANL",
    "mlp_coteaching": "Co-teaching",
    "mlp_ct": "CT",
    "mlp_dividemix_lite": "DivideMix-lite",
    "mlp_elr": "ELR",
    "mlp_fine": "FINE",
    "mlp_jocor": "JoCoR",
    "mlp_logitclip": "LogitClip",
    "mlp_sce": "SCE",
    "mlp_small_loss": "Small-loss",
    "graphmlp": "Graph-MLP adapted",
    "gsdn": "GSDN adapted",
    "gcn": "GCN(X)+CE",
    "gcn_xh_ce": "GCN[X||H]+CE",
    "mlp_distill_gcn": "GLNN-style GCN teacher",
    "mlp_distill_gcn_xh": "GLNN-style GCN[X||H] teacher",
    "mlp_distill_knn_xh": "Ours-distill kNN teacher",
    "lp_mlp_softdist_beta05": "LP-target MLP beta=0.5",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _to_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns:
        return df[df["status"].fillna("ok").eq("ok")].copy()
    return df.copy()


def mean_std(x: pd.Series) -> str:
    return f"{x.mean() * 100:.2f}±{x.std(ddof=1) * 100:.2f}"


def _format_pp(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:+.2f}"


def paired_stats(df: pd.DataFrame, group_cols, ours: str, baseline: str, source: str) -> dict | None:
    ours_df = df[df["method"].eq(ours)]
    base_df = df[df["method"].eq(baseline)]
    keys = list(group_cols) + ["seed"]
    merged = ours_df[keys + ["test"]].merge(
        base_df[keys + ["test"]],
        on=keys,
        suffixes=("_ours", "_baseline"),
    )
    if merged.empty:
        return None
    delta = merged["test_ours"].astype(float) - merged["test_baseline"].astype(float)
    t_p = np.nan
    w_p = np.nan
    if stats is not None and len(delta) >= 2:
        try:
            t_p = float(stats.ttest_rel(merged["test_ours"], merged["test_baseline"]).pvalue)
        except Exception:
            t_p = np.nan
        try:
            if not np.allclose(delta.to_numpy(), 0):
                w_p = float(stats.wilcoxon(delta).pvalue)
        except Exception:
            w_p = np.nan
    out = {col: merged[col].iloc[0] for col in group_cols}
    out.update(
        {
            "source": source,
            "ours": ours,
            "baseline": baseline,
            "ours_label": LABELS.get(ours, ours),
            "baseline_label": LABELS.get(baseline, baseline),
            "n": int(len(delta)),
            "mean_delta": float(delta.mean()),
            "mean_delta_pp": float(delta.mean() * 100),
            "std_delta_pp": float(delta.std(ddof=1) * 100) if len(delta) > 1 else 0.0,
            "positive_seeds": int((delta > 0).sum()),
            "paired_t_p": t_p,
            "wilcoxon_p": w_p,
            "seeds": " ".join(str(int(s)) for s in merged["seed"].tolist()),
            "deltas_pp": " ".join(f"{v * 100:.2f}" for v in delta.tolist()),
        }
    )
    return out


def beta_validation_audit(beta_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(beta_dir.glob("beta_uniform0p*_seeds_0_1_2_3_4.csv")):
        df = _read_csv(path)
        df = _ok(_to_numeric(df, ["seed", "valid", "test", "noise_rate"]))
        df = df[df["method"].isin(METHOD_TO_BETA)]
        if df.empty:
            continue
        df["target_beta"] = df["method"].map(METHOD_TO_BETA)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    agg = (
        raw.groupby(["dataset", "noise_type", "noise_rate", "method", "target_beta"], as_index=False)
        .agg(valid_mean=("valid", "mean"), valid_std=("valid", "std"), test_mean=("test", "mean"), test_std=("test", "std"), n=("seed", "nunique"))
    )

    rows = []
    for key, grp in agg.groupby(["dataset", "noise_type", "noise_rate"]):
        val_row = grp.sort_values(["valid_mean", "test_mean"], ascending=False).iloc[0]
        test_row = grp.sort_values(["test_mean", "valid_mean"], ascending=False).iloc[0]
        beta05 = grp[grp["target_beta"].eq(0.5)].iloc[0] if grp["target_beta"].eq(0.5).any() else None
        rows.append(
            {
                "dataset": key[0],
                "noise_type": key[1],
                "noise_rate": key[2],
                "val_best_method": val_row["method"],
                "val_best_beta": val_row["target_beta"],
                "val_best_valid": val_row["valid_mean"],
                "val_best_test": val_row["test_mean"],
                "test_best_method": test_row["method"],
                "test_best_beta": test_row["target_beta"],
                "test_best_valid": test_row["valid_mean"],
                "test_best_test": test_row["test_mean"],
                "beta05_valid": beta05["valid_mean"] if beta05 is not None else np.nan,
                "beta05_test": beta05["test_mean"] if beta05 is not None else np.nan,
                "val_best_minus_beta05_valid_pp": (val_row["valid_mean"] - beta05["valid_mean"]) * 100 if beta05 is not None else np.nan,
                "test_best_minus_beta05_test_pp": (test_row["test_mean"] - beta05["test_mean"]) * 100 if beta05 is not None else np.nan,
                "val_and_test_choose_same_beta": bool(val_row["target_beta"] == test_row["target_beta"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "noise_rate"])


def significance_tables(mlp10_raw: Path, mlp14_raw: Path) -> pd.DataFrame:
    rows = []

    main = _read_csv(mlp10_raw)
    main = _ok(_to_numeric(main, ["seed", "noise_rate", "test"]))
    ours = "frozengrace_gb_mlp_softdist_beta05"
    group_cols = ["dataset", "noise_type", "noise_rate"]
    for key, grp in main.groupby(group_cols):
        if ours not in set(grp["method"]):
            continue
        for baseline in sorted(set(grp["method"]) - {ours}):
            stat = paired_stats(grp, group_cols, ours, baseline, "mlp10_main")
            if stat is not None:
                rows.append(stat)

    graphfree = _read_csv(mlp14_raw)
    graphfree = _ok(_to_numeric(graphfree, ["seed", "noise_rate", "test"]))
    key_methods = {
        "train_only": (
            "frozengrace_gb_mlp_softdist_beta05",
            ["mlp_ce", "graphmlp", "gsdn"],
        ),
        "distill": (
            "mlp_distill_knn_xh",
            ["mlp_distill_gcn", "mlp_distill_gcn_xh", "graphmlp", "gsdn"],
        ),
    }
    for key, grp in graphfree.groupby(group_cols):
        for family, (method, baselines) in key_methods.items():
            if method not in set(grp["method"]):
                continue
            for baseline in baselines:
                if baseline not in set(grp["method"]):
                    continue
                stat = paired_stats(grp, group_cols, method, baseline, f"mlp14_{family}")
                if stat is not None:
                    rows.append(stat)

    return pd.DataFrame(rows)


def complete_linear_capacity(linear_dir: Path, mlp10_raw: Path, mlp12_raw: Path):
    linear_frames = []
    for path in sorted(linear_dir.glob("linear_ablation_uniform0p4_seeds_*.csv")):
        df = _read_csv(path)
        linear_frames.append(df)
    linear = pd.concat(linear_frames, ignore_index=True)
    linear = _ok(_to_numeric(linear, ["seed", "noise_rate", "valid", "test"]))

    ref_frames = []
    for path in [mlp10_raw, mlp12_raw]:
        df = _read_csv(path)
        df = _ok(_to_numeric(df, ["seed", "noise_rate", "valid", "test"]))
        df = df[
            df["method"].isin(
                [
                    "mlp_ce",
                    "frozengrace_gb_mlp_softdist_beta05",
                    "frozengrace_gb_mlp_softdist_beta05_shuffled",
                ]
            )
        ]
        ref_frames.append(df)
    refs = pd.concat(ref_frames, ignore_index=True)
    refs = refs[
        refs["noise_type"].eq("uniform")
        & refs["noise_rate"].eq(0.4)
        & refs["dataset"].isin(["cora", "citeseer", "pubmed", "coauthorcs", "amazonpho"])
    ]

    cols = ["dataset", "method", "seed", "noise_type", "noise_rate", "valid", "test", "status"]
    combined = pd.concat([linear[cols], refs[cols]], ignore_index=True)
    combined = combined.drop_duplicates(["dataset", "method", "seed", "noise_type", "noise_rate"], keep="last")

    aggregate = (
        combined.groupby(["dataset", "method"], as_index=False)
        .agg(n=("seed", "nunique"), mean=("test", "mean"), std=("test", "std"), valid_mean=("valid", "mean"))
        .sort_values(["dataset", "method"])
    )
    aggregate["method_label"] = aggregate["method"].map(lambda x: LABELS.get(x, x))
    aggregate["mean_std"] = aggregate.apply(lambda r: f"{r['mean'] * 100:.2f}±{r['std'] * 100:.2f}", axis=1)

    comparisons = [
        ("frozengrace_gb_linear_softdist_beta05", "linear_ce"),
        ("frozengrace_gb_linear_softdist_beta05", "frozengrace_gb_linear_softdist_beta05_shuffled"),
        ("frozengrace_gb_linear_softdist_beta05", "frozengrace_gb_mlp_softdist_beta05"),
        ("frozengrace_gb_mlp_softdist_beta05", "mlp_ce"),
        ("frozengrace_gb_mlp_softdist_beta05", "frozengrace_gb_mlp_softdist_beta05_shuffled"),
    ]
    rows = []
    for dataset, grp in combined.groupby(["dataset"]):
        for ours, baseline in comparisons:
            if ours in set(grp["method"]) and baseline in set(grp["method"]):
                stat = paired_stats(grp, ["dataset"], ours, baseline, "linear_capacity")
                if stat is not None:
                    rows.append(stat)
    deltas = pd.DataFrame(rows)
    return combined, aggregate, deltas


def p2_clean_safety(out_dir: Path):
    path = out_dir / "p2_clean_uniform0p0_seeds_0_1_2_3_4.csv"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    raw = _read_csv(path)
    raw = _ok(_to_numeric(raw, ["seed", "valid", "test", "noise_rate"]))
    aggregate = (
        raw.groupby(["dataset", "method"], as_index=False)
        .agg(n=("seed", "nunique"), mean=("test", "mean"), std=("test", "std"), valid_mean=("valid", "mean"))
        .sort_values(["dataset", "method"])
    )
    aggregate["method_label"] = aggregate["method"].map(lambda x: LABELS.get(x, x))
    aggregate["mean_std"] = aggregate.apply(lambda r: f"{r['mean'] * 100:.2f}±{r['std'] * 100:.2f}", axis=1)

    rows = []
    comparisons = [
        ("frozengrace_gb_mlp_softdist_beta05", "mlp_ce"),
        ("frozengrace_gb_mlp_softdist_beta05", "frozengrace_gb_mlp_softdist_beta05_shuffled"),
    ]
    for dataset, grp in raw.groupby("dataset"):
        for ours, baseline in comparisons:
            if ours in set(grp["method"]) and baseline in set(grp["method"]):
                stat = paired_stats(grp, ["dataset"], ours, baseline, "p2_clean")
                if stat is not None:
                    rows.append(stat)
    return aggregate, pd.DataFrame(rows)


def p3_lp_target(out_dir: Path, mlp10_raw: Path):
    path = out_dir / "p3_lp_target_uniform0p4_seeds_0_1_2_3_4.csv"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    lp = _read_csv(path)
    lp = _ok(_to_numeric(lp, ["seed", "valid", "test", "noise_rate"]))

    ref = _read_csv(mlp10_raw)
    ref = _ok(_to_numeric(ref, ["seed", "valid", "test", "noise_rate"]))
    ref = ref[
        ref["dataset"].isin(["cora", "citeseer", "pubmed"])
        & ref["noise_type"].eq("uniform")
        & ref["noise_rate"].eq(0.4)
        & ref["method"].isin(
            [
                "mlp_ce",
                "frozengrace_gb_mlp_softdist_beta05",
                "frozengrace_gb_mlp_softdist_beta05_shuffled",
            ]
        )
    ].copy()

    cols = ["dataset", "method", "seed", "noise_type", "noise_rate", "valid", "test", "status"]
    raw = pd.concat([ref[cols], lp[cols]], ignore_index=True)
    raw = raw.drop_duplicates(["dataset", "method", "seed", "noise_type", "noise_rate"], keep="last")

    aggregate = (
        raw.groupby(["dataset", "method"], as_index=False)
        .agg(n=("seed", "nunique"), mean=("test", "mean"), std=("test", "std"), valid_mean=("valid", "mean"))
        .sort_values(["dataset", "method"])
    )
    aggregate["method_label"] = aggregate["method"].map(lambda x: LABELS.get(x, x))
    aggregate["mean_std"] = aggregate.apply(lambda r: f"{r['mean'] * 100:.2f}±{r['std'] * 100:.2f}", axis=1)

    rows = []
    comparisons = [
        ("frozengrace_gb_mlp_softdist_beta05", "lp_mlp_softdist_beta05"),
        ("lp_mlp_softdist_beta05", "mlp_ce"),
        ("lp_mlp_softdist_beta05", "frozengrace_gb_mlp_softdist_beta05_shuffled"),
    ]
    for dataset, grp in raw.groupby("dataset"):
        for ours, baseline in comparisons:
            if ours in set(grp["method"]) and baseline in set(grp["method"]):
                stat = paired_stats(grp, ["dataset"], ours, baseline, "p3_lp_target")
                if stat is not None:
                    rows.append(stat)
    return aggregate, pd.DataFrame(rows)


def p4_pubmed_linear(out_dir: Path, linear_dir: Path):
    frames = []
    for path in sorted(out_dir.glob("p4_pubmed_linear_*_seeds_0_1_2_3_4.csv")):
        frames.append(_read_csv(path))
    for path in sorted(out_dir.glob("p4_pubmed_softdist_linear_*_seeds_0_1_2_3_4.csv")):
        frames.append(_read_csv(path))

    # Uniform 0.4 target arms were produced by the earlier linear ablation.
    for path in sorted(linear_dir.glob("linear_ablation_uniform0p4_seeds_*.csv")):
        df = _read_csv(path)
        df = df[
            df["dataset"].eq("pubmed")
            & df["method"].isin(
                [
                    "linear_ce",
                    "frozengrace_gb_linear_softdist_beta05",
                    "frozengrace_gb_linear_softdist_beta05_shuffled",
                ]
            )
        ]
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = _ok(_to_numeric(raw, ["seed", "valid", "test", "noise_rate"]))
    raw = raw.drop_duplicates(["dataset", "method", "seed", "noise_type", "noise_rate"], keep="last")

    aggregate = (
        raw.groupby(["dataset", "noise_type", "noise_rate", "method"], as_index=False)
        .agg(n=("seed", "nunique"), mean=("test", "mean"), std=("test", "std"), valid_mean=("valid", "mean"))
        .sort_values(["noise_type", "noise_rate", "method"])
    )
    aggregate["method_label"] = aggregate["method"].map(lambda x: LABELS.get(x, x))
    aggregate["mean_std"] = aggregate.apply(lambda r: f"{r['mean'] * 100:.2f}±{r['std'] * 100:.2f}", axis=1)

    rows = []
    target = "frozengrace_gb_linear_softdist_beta05"
    baselines = [
        "linear_ce",
        "frozengrace_gb_linear_softdist_beta05_shuffled",
        "linear_small_loss",
        "linear_jocor",
        "linear_dividemix_lite",
        "linear_ct",
    ]
    for _, grp in raw.groupby(["dataset", "noise_type", "noise_rate"]):
        if target not in set(grp["method"]):
            continue
        for baseline in baselines:
            if baseline not in set(grp["method"]):
                continue
            stat = paired_stats(grp, ["dataset", "noise_type", "noise_rate"], target, baseline, "p4_pubmed_linear")
            if stat is not None:
                rows.append(stat)
    deltas = pd.DataFrame(rows)

    best_rows = []
    robust_methods = ["linear_small_loss", "linear_jocor", "linear_dividemix_lite", "linear_ct"]
    for key, grp in aggregate.groupby(["dataset", "noise_type", "noise_rate"]):
        robust = grp[grp["method"].isin(robust_methods)]
        target_row = grp[grp["method"].eq(target)]
        if robust.empty or target_row.empty:
            continue
        best = robust.sort_values("mean", ascending=False).iloc[0]
        best_rows.append(
            {
                "dataset": key[0],
                "noise_type": key[1],
                "noise_rate": key[2],
                "softdist_linear": float(target_row.iloc[0]["mean"]),
                "best_linear_robust_method": best["method"],
                "best_linear_robust": float(best["mean"]),
                "delta_vs_best_linear_robust_pp": (float(target_row.iloc[0]["mean"]) - float(best["mean"])) * 100,
            }
        )
    best_summary = pd.DataFrame(best_rows)
    return aggregate, deltas, best_summary


def write_report(
    out_dir: Path,
    beta: pd.DataFrame,
    sig: pd.DataFrame,
    linear_agg: pd.DataFrame,
    linear_delta: pd.DataFrame,
    p2_agg: pd.DataFrame,
    p2_delta: pd.DataFrame,
    p3_agg: pd.DataFrame,
    p3_delta: pd.DataFrame,
    p4_agg: pd.DataFrame,
    p4_delta: pd.DataFrame,
    p4_best: pd.DataFrame,
):
    lines = []
    lines.append("# MLP-15 Review Defense P0 Report\n")
    lines.append("Protocol: P0 pure recomputation; no model retraining in this report.\n")

    lines.append("## P0a Beta Validation Audit\n")
    if beta.empty:
        lines.append("No beta audit rows found.\n")
    else:
        view = beta.copy()
        for col in ["val_best_valid", "val_best_test", "test_best_valid", "test_best_test", "beta05_valid", "beta05_test"]:
            view[col] = view[col].map(lambda x: f"{x * 100:.2f}" if pd.notna(x) else "")
        for col in ["val_best_minus_beta05_valid_pp", "test_best_minus_beta05_test_pp"]:
            view[col] = view[col].map(_format_pp)
        lines.append(view.to_markdown(index=False))
        lines.append("\n")

    lines.append("## P0b Paired Significance Summary\n")
    if sig.empty:
        lines.append("No significance rows found.\n")
    else:
        key = sig.copy()
        key = key[
            key["source"].isin(["mlp10_main", "mlp14_train_only", "mlp14_distill"])
            & (
                key["baseline"].isin(["mlp_ce", "frozengrace_gb_mlp_softdist_beta05_shuffled", "graphmlp", "gsdn", "mlp_distill_gcn", "mlp_distill_gcn_xh"])
            )
        ]
        key = key.sort_values(["source", "dataset", "noise_type", "noise_rate", "baseline"])
        cols = ["source", "dataset", "noise_type", "noise_rate", "ours_label", "baseline_label", "n", "mean_delta_pp", "positive_seeds", "paired_t_p", "wilcoxon_p"]
        show = key[cols].copy()
        show["mean_delta_pp"] = show["mean_delta_pp"].map(_format_pp)
        show["paired_t_p"] = show["paired_t_p"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
        show["wilcoxon_p"] = show["wilcoxon_p"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
        lines.append(show.to_markdown(index=False))
        lines.append("\n")

    lines.append("## P2 Clean-Label Safety\n")
    if p2_agg.empty:
        lines.append("P2 clean run not found.\n")
    else:
        pivot = p2_agg.pivot(index="dataset", columns="method_label", values="mean_std").reset_index()
        preferred = ["dataset", "MLP+CE", "SoftDist-MLP beta=0.5", "SoftDist-MLP shuffled"]
        cols = [c for c in preferred if c in pivot.columns]
        lines.append(pivot[cols].to_markdown(index=False))
        lines.append("\n")
        d = p2_delta.sort_values(["dataset", "baseline_label"]).copy()
        show = d[["dataset", "ours_label", "baseline_label", "n", "mean_delta_pp", "positive_seeds", "paired_t_p", "wilcoxon_p"]].copy()
        show["mean_delta_pp"] = show["mean_delta_pp"].map(_format_pp)
        show["paired_t_p"] = show["paired_t_p"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
        show["wilcoxon_p"] = show["wilcoxon_p"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
        lines.append(show.to_markdown(index=False))
        lines.append("\n")

    lines.append("## P3 LP-Target Graph-Space Consensus Control\n")
    if p3_agg.empty:
        lines.append("P3 LP-target run not found.\n")
    else:
        pivot = p3_agg.pivot(index="dataset", columns="method_label", values="mean_std").reset_index()
        preferred = [
            "dataset",
            "MLP+CE",
            "LP-target MLP beta=0.5",
            "SoftDist-MLP beta=0.5",
            "SoftDist-MLP shuffled",
        ]
        cols = [c for c in preferred if c in pivot.columns]
        lines.append(pivot[cols].to_markdown(index=False))
        lines.append("\n")
        d = p3_delta.sort_values(["dataset", "baseline_label"]).copy()
        show = d[["dataset", "ours_label", "baseline_label", "n", "mean_delta_pp", "positive_seeds", "paired_t_p", "wilcoxon_p"]].copy()
        show["mean_delta_pp"] = show["mean_delta_pp"].map(_format_pp)
        show["paired_t_p"] = show["paired_t_p"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
        show["wilcoxon_p"] = show["wilcoxon_p"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
        lines.append(show.to_markdown(index=False))
        lines.append("\n")

    lines.append("## P4 PubMed Linear Robust-Baseline Fairness\n")
    if p4_agg.empty:
        lines.append("P4 PubMed linear run not found.\n")
    else:
        pivot = p4_agg.pivot_table(
            index=["noise_type", "noise_rate"],
            columns="method_label",
            values="mean_std",
            aggfunc="first",
        ).reset_index()
        preferred = [
            "noise_type",
            "noise_rate",
            "Linear+CE",
            "SoftDist-Linear beta=0.5",
            "SoftDist-Linear shuffled",
            "Small-loss",
            "JoCoR",
            "DivideMix-lite",
            "CT",
        ]
        cols = [c for c in preferred if c in pivot.columns]
        lines.append(pivot[cols].to_markdown(index=False))
        lines.append("\n")
        if not p4_best.empty:
            best = p4_best.copy()
            best["softdist_linear"] = best["softdist_linear"].map(lambda x: f"{x * 100:.2f}")
            best["best_linear_robust"] = best["best_linear_robust"].map(lambda x: f"{x * 100:.2f}")
            best["delta_vs_best_linear_robust_pp"] = best["delta_vs_best_linear_robust_pp"].map(_format_pp)
            lines.append(best.to_markdown(index=False))
            lines.append("\n")
        d = p4_delta[p4_delta["baseline"].isin(["linear_ce", "frozengrace_gb_linear_softdist_beta05_shuffled", "linear_small_loss", "linear_jocor", "linear_dividemix_lite", "linear_ct"])].copy()
        d = d.sort_values(["noise_type", "noise_rate", "baseline_label"])
        show = d[["noise_type", "noise_rate", "ours_label", "baseline_label", "n", "mean_delta_pp", "positive_seeds", "paired_t_p", "wilcoxon_p"]].copy()
        show["mean_delta_pp"] = show["mean_delta_pp"].map(_format_pp)
        show["paired_t_p"] = show["paired_t_p"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
        show["wilcoxon_p"] = show["wilcoxon_p"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
        lines.append(show.to_markdown(index=False))
        lines.append("\n")

    lines.append("## P0c Completed Linear Capacity Ablation\n")
    if linear_agg.empty:
        lines.append("No linear rows found.\n")
    else:
        pivot = linear_agg.pivot(index="dataset", columns="method_label", values="mean_std").reset_index()
        preferred = [
            "dataset",
            "MLP+CE",
            "Linear+CE",
            "SoftDist-MLP beta=0.5",
            "SoftDist-Linear beta=0.5",
            "SoftDist-MLP shuffled",
            "SoftDist-Linear shuffled",
        ]
        cols = [c for c in preferred if c in pivot.columns]
        lines.append(pivot[cols].to_markdown(index=False))
        lines.append("\n")

        d = linear_delta.copy()
        d = d[d["baseline"].isin(["linear_ce", "frozengrace_gb_linear_softdist_beta05_shuffled", "frozengrace_gb_mlp_softdist_beta05", "mlp_ce", "frozengrace_gb_mlp_softdist_beta05_shuffled"])]
        d = d.sort_values(["dataset", "ours", "baseline"])
        show = d[["dataset", "ours_label", "baseline_label", "n", "mean_delta_pp", "positive_seeds", "paired_t_p", "wilcoxon_p"]].copy()
        show["mean_delta_pp"] = show["mean_delta_pp"].map(_format_pp)
        show["paired_t_p"] = show["paired_t_p"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
        show["wilcoxon_p"] = show["wilcoxon_p"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
        lines.append(show.to_markdown(index=False))
        lines.append("\n")

    report = "\n".join(lines)
    (out_dir / "mlp15_report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="runs/mlp15_review_defense_20260704")
    parser.add_argument("--beta_dir", default="runs/mlp_beta_hardness_5seed_20260701")
    parser.add_argument("--mlp10_raw", default="runs/mlp10_baseline_refresh_20260702/mlp10_combined_raw.csv")
    parser.add_argument("--mlp12_raw", default="runs/mlp12_amazonphoto_20260702/mlp12_combined_raw.csv")
    parser.add_argument("--mlp14_raw", default="runs/mlp14_graphfree_family_20260703/mlp14_combined_raw.csv")
    parser.add_argument("--linear_dir", default="runs/linear_ablation_20260704")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    beta = beta_validation_audit(Path(args.beta_dir))
    sig = significance_tables(Path(args.mlp10_raw), Path(args.mlp14_raw))
    linear_raw, linear_agg, linear_delta = complete_linear_capacity(Path(args.linear_dir), Path(args.mlp10_raw), Path(args.mlp12_raw))
    p2_agg, p2_delta = p2_clean_safety(out_dir)
    p3_agg, p3_delta = p3_lp_target(out_dir, Path(args.mlp10_raw))
    p4_agg, p4_delta, p4_best = p4_pubmed_linear(out_dir, Path(args.linear_dir))

    beta.to_csv(out_dir / "p0a_beta_validation_audit.csv", index=False)
    sig.to_csv(out_dir / "p0b_paired_significance.csv", index=False)
    linear_raw.to_csv(out_dir / "p0c_linear_capacity_raw.csv", index=False)
    linear_agg.to_csv(out_dir / "p0c_linear_capacity_aggregate.csv", index=False)
    linear_delta.to_csv(out_dir / "p0c_linear_capacity_paired_deltas.csv", index=False)
    p2_agg.to_csv(out_dir / "p2_clean_aggregate.csv", index=False)
    p2_delta.to_csv(out_dir / "p2_clean_paired_deltas.csv", index=False)
    p3_agg.to_csv(out_dir / "p3_lp_target_aggregate.csv", index=False)
    p3_delta.to_csv(out_dir / "p3_lp_target_paired_deltas.csv", index=False)
    p4_agg.to_csv(out_dir / "p4_pubmed_linear_aggregate.csv", index=False)
    p4_delta.to_csv(out_dir / "p4_pubmed_linear_paired_deltas.csv", index=False)
    p4_best.to_csv(out_dir / "p4_pubmed_linear_best_robust_summary.csv", index=False)

    linear_dir = Path(args.linear_dir)
    linear_agg.to_csv(linear_dir / "linear_ablation_completed_aggregate.csv", index=False)
    linear_delta.to_csv(linear_dir / "linear_ablation_completed_paired_deltas.csv", index=False)

    write_report(out_dir, beta, sig, linear_agg, linear_delta, p2_agg, p2_delta, p3_agg, p3_delta, p4_agg, p4_delta, p4_best)
    print(f"beta: {out_dir / 'p0a_beta_validation_audit.csv'}")
    print(f"significance: {out_dir / 'p0b_paired_significance.csv'}")
    print(f"linear aggregate: {out_dir / 'p0c_linear_capacity_aggregate.csv'}")
    print(f"linear deltas: {out_dir / 'p0c_linear_capacity_paired_deltas.csv'}")
    print(f"p2 clean: {out_dir / 'p2_clean_aggregate.csv'}")
    print(f"p3 lp target: {out_dir / 'p3_lp_target_aggregate.csv'}")
    print(f"p4 pubmed linear: {out_dir / 'p4_pubmed_linear_aggregate.csv'}")
    print(f"report: {out_dir / 'mlp15_report.md'}")


if __name__ == "__main__":
    main()
