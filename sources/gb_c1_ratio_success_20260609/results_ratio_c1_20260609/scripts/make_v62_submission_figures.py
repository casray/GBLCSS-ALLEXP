from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parents[1]
SOURCE_CSV = (
    ROOT
    / "paper_pipeline"
    / "00_evidence"
    / "v38_loo_summary"
    / "formal_aggregate.csv"
)
LOO_RAW_CSV = (
    EXPERIMENT_ROOT
    / "external"
    / "NoisyGL"
    / "runs"
    / "mlp44_loo_tiesafe_fullspectrum_20260730"
    / "mlp44_combined_raw.csv"
)
GROUPING_COST_CSV = (
    EXPERIMENT_ROOT
    / "external"
    / "NoisyGL"
    / "runs"
    / "mlp45_loo_grouping_pareto_20260730"
    / "grouping_cost_benchmark.csv"
)
GROUPING_ACCURACY_CSV = (
    EXPERIMENT_ROOT
    / "external"
    / "NoisyGL"
    / "runs"
    / "mlp45_loo_grouping_pareto_20260730"
    / "mlp45_accuracy_aggregate.csv"
)
FIGURE_DIR = ROOT / "figures"
SOURCE_DATA_DIR = FIGURE_DIR / "source_data"

OURS = "frozengrace_gb_mlp_softdist_beta05_loo"
BASELINES = [
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_jocor",
    "mlp_sce",
    "mlp_elr",
    "mlp_logitclip",
]
METHOD_LABELS = {
    "mlp_ce": "MLP+CE",
    "mlp_small_loss": "Small-loss",
    "mlp_coteaching": "Co-teaching",
    "mlp_jocor": "JoCoR",
    "mlp_sce": "SCE",
    "mlp_elr": "ELR",
    "mlp_logitclip": "LogitClip",
}
DATASETS = ["cora", "citeseer", "corafull", "coauthorcs", "amazonpho"]
DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "corafull": "CoraFull",
    "coauthorcs": "Coauthor-CS",
    "amazonpho": "Amazon-Photo",
}
SETTINGS = [
    ("uniform", 0.2, "Uniform\n0.2"),
    ("uniform", 0.4, "Uniform\n0.4"),
    ("uniform", 0.6, "Uniform\n0.6"),
    ("pair", 0.4, "Pair\n0.4"),
    ("instance", 0.4, "Instance\n0.4"),
]

PALETTE = {
    "ink": "#26323D",
    "muted": "#6B7785",
    "light": "#E9EEF2",
    "blue": "#2F6B9A",
    "blue_soft": "#D7E7F3",
    "red": "#C95A49",
    "orange": "#D97706",
    "green": "#118A6A",
    "pink": "#C9799A",
    "gray": "#9AA6B2",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: mpl.figure.Figure, stem: Path, *, tight: bool = True) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(stem.with_suffix(".svg"), bbox_inches=bbox)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=bbox)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches=bbox)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches=bbox)


def build_gain_source_data() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV)
    methods = [OURS, *BASELINES]
    selected = df[
        df["dataset"].isin(DATASETS)
        & df["method"].isin(methods)
        & df["noise_type"].isin({"uniform", "pair", "instance"})
    ].copy()

    expected_rows = len(DATASETS) * len(SETTINGS) * len(methods)
    if len(selected) != expected_rows:
        raise ValueError(f"Expected {expected_rows} aggregate rows, found {len(selected)}")
    if not (selected["n"] == 5).all():
        raise ValueError("Every heatmap entry must aggregate exactly five seeds")

    pivot = selected.pivot(
        index=["dataset", "noise_type", "noise_rate"],
        columns="method",
        values="mean",
    )
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for noise_type, noise_rate, setting_label in SETTINGS:
            key = (dataset, noise_type, noise_rate)
            values = pivot.loc[key]
            best_method = values[BASELINES].astype(float).idxmax()
            ours_mean = float(values[OURS])
            baseline_mean = float(values[best_method])
            rows.append(
                {
                    "dataset": DATASET_LABELS[dataset],
                    "noise_type": noise_type,
                    "noise_rate": noise_rate,
                    "setting": setting_label.replace("\n", " "),
                    "gb_lcss_mean_accuracy": ours_mean,
                    "best_baseline": METHOD_LABELS[best_method],
                    "best_baseline_mean_accuracy": baseline_mean,
                    "gain_pp": 100.0 * (ours_mean - baseline_mean),
                    "seeds": 5,
                }
            )

    source = pd.DataFrame(rows)
    if len(source) != 25:
        raise ValueError(f"Expected 25 cells, found {len(source)}")
    positive_or_tied = int((source["gain_pp"] >= -1e-12).sum())
    if positive_or_tied != 24:
        raise ValueError(f"Expected 24/25 non-negative cells, found {positive_or_tied}/25")
    negative = source[source["gain_pp"] < 0]
    if len(negative) != 1 or negative.iloc[0]["dataset"] != "Coauthor-CS":
        raise ValueError("The expected sole negative cell was not recovered")
    return source


def plot_gain_heatmap(source: pd.DataFrame) -> None:
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_path = SOURCE_DATA_DIR / "fig2_gblcss_gain_heatmap_source.csv"
    source.to_csv(source_path, index=False, encoding="utf-8-sig")

    matrix = np.empty((len(DATASETS), len(SETTINGS)), dtype=float)
    for i, dataset in enumerate(DATASETS):
        dataset_label = DATASET_LABELS[dataset]
        for j, (_, _, setting_label) in enumerate(SETTINGS):
            setting = setting_label.replace("\n", " ")
            value = source.loc[
                (source["dataset"] == dataset_label) & (source["setting"] == setting),
                "gain_pp",
            ]
            if len(value) != 1:
                raise ValueError(f"Missing heatmap value for {dataset_label}, {setting}")
            matrix[i, j] = float(value.iloc[0])

    cmap = LinearSegmentedColormap.from_list(
        "gain_diverging",
        [PALETTE["red"], "#F6F3EF", PALETTE["blue"]],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=6.0)

    fig, ax = plt.subplots(figsize=(7.15, 3.55), constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(SETTINGS)))
    ax.set_xticklabels([label for _, _, label in SETTINGS], fontsize=8)
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_yticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=8)
    ax.tick_params(which="both", length=0, pad=5)
    ax.set_xlabel("Noise setting", fontsize=8.5, labelpad=7)
    ax.set_ylabel("Dataset", fontsize=8.5, labelpad=7)
    ax.set_title(
        "GB-LCSS gain over the strongest common-backbone baseline",
        fontsize=10,
        fontweight="bold",
        color=PALETTE["ink"],
        pad=10,
    )

    ax.set_xticks(np.arange(-0.5, len(SETTINGS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.57 else PALETTE["ink"]
            ax.text(
                j,
                i,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight="bold" if value < 0 else "normal",
            )
            if value < 0:
                ax.add_patch(
                    Rectangle(
                        (j - 0.48, i - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#7A2D23",
                        linewidth=1.5,
                    )
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.025)
    cbar.set_label("Accuracy gain (percentage points)", fontsize=8, labelpad=7)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_linewidth(0.6)

    fig.text(
        0.01,
        0.01,
        "Each cell compares five-seed means under paired splits and noise instances.",
        fontsize=6.7,
        color=PALETTE["muted"],
    )
    save_figure(fig, FIGURE_DIR / "fig2_gblcss_gain_heatmap_v1")
    plt.close(fig)


def build_uniform_rate_source_data() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV)
    methods = [
        OURS,
        "mlp_jocor",
        "mlp_ce",
        "frozengrace_gb_mlp_softdist_beta05_loo_shuffled",
    ]
    selected = df[
        df["dataset"].isin(DATASETS)
        & df["method"].isin(methods)
        & (df["noise_type"] == "uniform")
        & df["noise_rate"].isin({0.2, 0.4, 0.6})
    ].copy()
    expected_rows = len(DATASETS) * 3 * len(methods)
    if len(selected) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} uniform-rate rows, found {len(selected)}"
        )
    if not (selected["n"] == 5).all():
        raise ValueError("Every uniform-rate point must aggregate exactly five seeds")

    labels = {
        OURS: "GB-LCSS",
        "mlp_jocor": "JoCoR",
        "mlp_ce": "MLP+CE",
        "frozengrace_gb_mlp_softdist_beta05_loo_shuffled": "Random grouping",
    }
    selected["dataset_label"] = selected["dataset"].map(DATASET_LABELS)
    selected["method_label"] = selected["method"].map(labels)
    selected["mean_accuracy_pct"] = 100.0 * selected["mean"]
    selected["std_accuracy_pct"] = 100.0 * selected["std"]
    source = selected[
        [
            "dataset",
            "dataset_label",
            "noise_rate",
            "method",
            "method_label",
            "n",
            "mean_accuracy_pct",
            "std_accuracy_pct",
        ]
    ].sort_values(["dataset", "method", "noise_rate"])
    return source


def plot_uniform_rate_curves(source: pd.DataFrame) -> None:
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_path = SOURCE_DATA_DIR / "fig3_uniform_noise_rate_curves_source.csv"
    source.to_csv(source_path, index=False, encoding="utf-8-sig")

    styles = {
        "GB-LCSS": {"color": PALETTE["blue"], "marker": "o", "lw": 2.0, "ls": "-", "zorder": 5},
        "JoCoR": {"color": PALETTE["red"], "marker": "s", "lw": 1.25, "ls": "-", "zorder": 4},
        "MLP+CE": {"color": PALETTE["gray"], "marker": "^", "lw": 1.15, "ls": "-", "zorder": 3},
        "Random grouping": {"color": PALETTE["orange"], "marker": "D", "lw": 1.15, "ls": "--", "zorder": 2},
    }
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 4.45),
        sharey=True,
    )
    axes = axes.ravel()
    rates = np.array([0.2, 0.4, 0.6])

    for panel_idx, dataset in enumerate(DATASETS):
        ax = axes[panel_idx]
        panel = source[source["dataset"] == dataset]
        for method_label, style in styles.items():
            values = panel[panel["method_label"] == method_label].sort_values("noise_rate")
            if len(values) != 3:
                raise ValueError(f"Missing rate point for {dataset}, {method_label}")
            ax.errorbar(
                values["noise_rate"],
                values["mean_accuracy_pct"],
                yerr=values["std_accuracy_pct"],
                capsize=2.2,
                markersize=4.2,
                markeredgewidth=0.6,
                **style,
                label=method_label,
            )

        ax.set_ylim(18.0, 98.0)
        ax.set_yticks([20, 40, 60, 80])
        ax.set_xticks(rates)
        ax.set_xticklabels(["0.2", "0.4", "0.6"])
        ax.grid(axis="y", color=PALETTE["light"], linewidth=0.75)
        ax.set_title(DATASET_LABELS[dataset], fontsize=8.5, fontweight="bold", pad=5)
        ax.text(
            -0.12,
            1.07,
            chr(ord("a") + panel_idx),
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )
        if panel_idx in {0, 3}:
            ax.set_ylabel("Test accuracy (%)", fontsize=7.5)
        if panel_idx >= 3:
            ax.set_xlabel("Uniform label-noise rate", fontsize=7.5)
        ax.tick_params(labelsize=7, length=2.5)

    legend_ax = axes[-1]
    legend_ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        fontsize=8,
        handlelength=2.4,
        labelspacing=1.0,
    )
    legend_ax.text(
        0.5,
        0.17,
        "Mean +/- sample SD\n5 paired seeds",
        transform=legend_ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color=PALETTE["muted"],
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.105,
        top=0.92,
        wspace=0.16,
        hspace=0.24,
    )

    save_figure(fig, FIGURE_DIR / "fig3_uniform_noise_rate_curves_v1")
    plt.close(fig)


def build_target_quality_source_data() -> pd.DataFrame:
    df = pd.read_csv(LOO_RAW_CSV)
    selected = df[
        (df["method"] == OURS)
        & df["dataset"].isin(DATASETS)
        & df["noise_type"].isin({"uniform", "pair", "instance"})
    ].copy()
    selected = selected[
        selected.apply(
            lambda row: any(
                row["noise_type"] == noise_type
                and np.isclose(float(row["noise_rate"]), noise_rate)
                for noise_type, noise_rate, _ in SETTINGS
            ),
            axis=1,
        )
    ]

    expected_rows = len(DATASETS) * len(SETTINGS) * 5
    if len(selected) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} formal LOO rows, found {len(selected)}"
        )
    if not selected["status"].eq("ok").all():
        raise ValueError("Target-quality source contains failed runs")
    if not selected["leave_one_out"].eq(1).all():
        raise ValueError("Target-quality source must use leave-one-out targets")
    if not selected["tie_mode"].eq("distribution_only").all():
        raise ValueError("Target-quality source must use tie-safe targets")
    if not (
        selected["modified_val_count"].eq(0)
        & selected["modified_test_count"].eq(0)
    ).all():
        raise ValueError("Validation or test labels were modified")
    if selected[
        ["soft_target_brier", "noisy_onehot_brier", "soft_brier_gain"]
    ].isna().any().any():
        raise ValueError("Target-quality diagnostics contain missing values")
    reconstructed_gain = (
        selected["noisy_onehot_brier"] - selected["soft_target_brier"]
    )
    if not np.allclose(
        reconstructed_gain,
        selected["soft_brier_gain"],
        atol=1e-9,
        rtol=1e-7,
    ):
        raise ValueError("Stored Brier gains do not match their components")
    if not selected["soft_brier_gain"].gt(0).all():
        raise ValueError("Expected every formal run to have positive Brier gain")

    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for noise_type, noise_rate, setting_label in SETTINGS:
            group = selected[
                (selected["dataset"] == dataset)
                & (selected["noise_type"] == noise_type)
                & np.isclose(selected["noise_rate"].astype(float), noise_rate)
            ].sort_values("seed")
            if len(group) != 5 or group["seed"].nunique() != 5:
                raise ValueError(
                    f"Expected five unique seeds for {dataset}, {noise_type} {noise_rate}"
                )
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "noise_type": noise_type,
                    "noise_rate": noise_rate,
                    "setting": setting_label.replace("\n", " "),
                    "seeds": 5,
                    "noisy_onehot_brier_mean": group["noisy_onehot_brier"].mean(),
                    "noisy_onehot_brier_sd": group["noisy_onehot_brier"].std(ddof=1),
                    "soft_target_brier_mean": group["soft_target_brier"].mean(),
                    "soft_target_brier_sd": group["soft_target_brier"].std(ddof=1),
                    "brier_gain_mean": group["soft_brier_gain"].mean(),
                    "brier_gain_sd": group["soft_brier_gain"].std(ddof=1),
                }
            )

    source = pd.DataFrame(rows)
    if len(source) != 25 or not source["brier_gain_mean"].gt(0).all():
        raise ValueError("Expected 25 positive aggregate Brier gains")
    return source


def plot_target_quality(source: pd.DataFrame) -> None:
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_path = SOURCE_DATA_DIR / "fig4_target_quality_source.csv"
    source.to_csv(source_path, index=False, encoding="utf-8-sig")

    dataset_colors = {
        "cora": PALETTE["blue"],
        "citeseer": PALETTE["orange"],
        "corafull": PALETTE["green"],
        "coauthorcs": PALETTE["pink"],
        "amazonpho": PALETTE["gray"],
    }
    fig, (ax_scatter, ax_heatmap) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.25),
        gridspec_kw={"width_ratios": [1.0, 1.18]},
    )

    brier_min = min(
        source["noisy_onehot_brier_mean"].min(),
        source["soft_target_brier_mean"].min(),
    )
    brier_max = max(
        source["noisy_onehot_brier_mean"].max(),
        source["soft_target_brier_mean"].max(),
    )
    margin = 0.055 * (brier_max - brier_min)
    limits = (max(0.0, brier_min - margin), brier_max + margin)
    ax_scatter.plot(
        limits,
        limits,
        color="#9EA8B1",
        linewidth=0.9,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    for dataset in DATASETS:
        panel = source[source["dataset"] == dataset]
        ax_scatter.errorbar(
            panel["noisy_onehot_brier_mean"],
            panel["soft_target_brier_mean"],
            xerr=panel["noisy_onehot_brier_sd"],
            yerr=panel["soft_target_brier_sd"],
            fmt="o",
            markersize=4.5,
            markerfacecolor=dataset_colors[dataset],
            markeredgecolor="white",
            markeredgewidth=0.5,
            ecolor=dataset_colors[dataset],
            elinewidth=0.65,
            capsize=1.5,
            alpha=0.9,
            label=DATASET_LABELS[dataset],
            zorder=3,
        )
    ax_scatter.set_xlim(limits)
    ax_scatter.set_ylim(limits)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_xlabel("Noisy one-hot target Brier score", fontsize=7.5)
    ax_scatter.set_ylabel("GB-LCSS soft-target Brier score", fontsize=7.5)
    ax_scatter.set_title("Target quality across 25 settings", fontsize=8.7, fontweight="bold")
    ax_scatter.grid(color=PALETTE["light"], linewidth=0.65)
    ax_scatter.tick_params(labelsize=7, length=2.5)
    ax_scatter.text(
        0.97,
        0.05,
        "below diagonal = lower error",
        transform=ax_scatter.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=PALETTE["muted"],
    )
    ax_scatter.legend(
        loc="upper left",
        fontsize=6.3,
        handletextpad=0.35,
        borderaxespad=0.35,
        labelspacing=0.35,
    )

    matrix = np.empty((len(DATASETS), len(SETTINGS)), dtype=float)
    for i, dataset in enumerate(DATASETS):
        for j, (_, _, setting_label) in enumerate(SETTINGS):
            setting = setting_label.replace("\n", " ")
            value = source.loc[
                (source["dataset"] == dataset) & (source["setting"] == setting),
                "brier_gain_mean",
            ]
            if len(value) != 1:
                raise ValueError(f"Missing Brier gain for {dataset}, {setting}")
            matrix[i, j] = float(value.iloc[0])

    cmap = LinearSegmentedColormap.from_list(
        "brier_gain",
        ["#F5F7F8", "#BFD6E6", PALETTE["blue"]],
        N=256,
    )
    x_edges = np.arange(len(SETTINGS) + 1) - 0.5
    y_edges = np.arange(len(DATASETS) + 1) - 0.5
    image = ax_heatmap.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=cmap,
        vmin=0.0,
        vmax=0.52,
        shading="flat",
        edgecolors="white",
        linewidth=1.3,
    )
    ax_heatmap.set_xticks(np.arange(len(SETTINGS)))
    ax_heatmap.set_xticklabels([label for _, _, label in SETTINGS], fontsize=7)
    ax_heatmap.set_yticks(np.arange(len(DATASETS)))
    ax_heatmap.set_yticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=7)
    ax_heatmap.set_xlim(-0.5, len(SETTINGS) - 0.5)
    ax_heatmap.set_ylim(len(DATASETS) - 0.5, -0.5)
    ax_heatmap.tick_params(which="both", length=0, pad=4)
    ax_heatmap.set_xlabel("Noise setting", fontsize=7.5, labelpad=6)
    ax_heatmap.set_title("Mean Brier-score reduction", fontsize=8.7, fontweight="bold")
    for spine in ax_heatmap.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value >= 0.34 else PALETTE["ink"]
            ax_heatmap.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )
    cbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.042, pad=0.025)
    cbar.solids.set_rasterized(False)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    cbar.set_label("Noisy target - soft target", fontsize=6.8, labelpad=5)
    cbar.outline.set_linewidth(0.55)

    ax_scatter.text(-0.16, 1.06, "a", transform=ax_scatter.transAxes, fontsize=9, fontweight="bold")
    ax_heatmap.text(-0.16, 1.06, "b", transform=ax_heatmap.transAxes, fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.88, wspace=0.34)
    save_figure(fig, FIGURE_DIR / "fig4_target_quality_v1")
    plt.close(fig)


def build_grouping_profile_source_data() -> pd.DataFrame:
    cost = pd.read_csv(GROUPING_COST_CSV)
    accuracy = pd.read_csv(GROUPING_ACCURACY_CSV)
    cost_method_map = {
        "loo_gb": "GB partition",
        "exact_knn": "Exact kNN",
        "loo_kmeans": "k-means partition",
    }
    accuracy_method_map = {
        OURS: "GB partition",
        "knn_mlp_softdist_beta05": "Exact kNN",
        "kmeans_mlp_softdist_beta05_loo": "k-means partition",
    }
    cost = cost[cost["dataset"].isin(DATASETS) & cost["method"].isin(cost_method_map)].copy()
    accuracy = accuracy[
        accuracy["dataset"].isin(DATASETS)
        & accuracy["method"].isin(accuracy_method_map)
    ].copy()
    if len(cost) != 15 or len(accuracy) != 15:
        raise ValueError(
            f"Expected 15 cost and 15 accuracy rows, found {len(cost)} and {len(accuracy)}"
        )
    if not cost["seed"].eq(0).all():
        raise ValueError("Grouping cost measurements must be the registered seed-0 runs")
    if not accuracy["seeds"].eq(5).all():
        raise ValueError("Grouping accuracy must aggregate five seeds")
    if "approximate_knn_available" in cost and not cost[
        "approximate_knn_available"
    ].eq(0).all():
        raise ValueError("This profile is limited to exact kNN")

    cost["method_label"] = cost["method"].map(cost_method_map)
    accuracy["method_label"] = accuracy["method"].map(accuracy_method_map)
    profile = cost.merge(
        accuracy[["dataset", "method_label", "mean", "std", "seeds"]],
        on=["dataset", "method_label"],
        how="inner",
        validate="one_to_one",
    )
    if len(profile) != 15:
        raise ValueError("Grouping cost and accuracy rows did not align one-to-one")
    profile["dataset_label"] = profile["dataset"].map(DATASET_LABELS)
    profile["accuracy_mean_pct"] = 100.0 * profile["mean"]
    profile["accuracy_sd_pct"] = 100.0 * profile["std"]
    profile["structure_storage_kib"] = profile["structure_storage_bytes"] / 1024.0
    profile["peak_rss_delta_mib"] = profile["peak_rss_delta_bytes"] / (1024.0**2)

    gb_accuracy = profile[profile["method_label"] == "GB partition"].set_index(
        "dataset"
    )["accuracy_mean_pct"]
    profile["accuracy_delta_vs_gb_pp"] = profile.apply(
        lambda row: row["accuracy_mean_pct"] - gb_accuracy.loc[row["dataset"]],
        axis=1,
    )
    return profile[
        [
            "dataset",
            "dataset_label",
            "method_label",
            "seed",
            "seeds",
            "n_nodes",
            "n_train",
            "groups",
            "matched_k",
            "end_to_end_structure_seconds",
            "structure_storage_kib",
            "peak_rss_delta_mib",
            "accuracy_mean_pct",
            "accuracy_sd_pct",
            "accuracy_delta_vs_gb_pp",
        ]
    ].sort_values(["dataset", "method_label"])


def plot_grouping_profile(source: pd.DataFrame) -> None:
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_path = SOURCE_DATA_DIR / "fig5_grouping_profile_source.csv"
    source.to_csv(source_path, index=False, encoding="utf-8-sig")

    styles = {
        "GB partition": {"color": PALETTE["blue"], "marker": "o"},
        "Exact kNN": {"color": PALETTE["orange"], "marker": "s"},
        "k-means partition": {"color": PALETTE["green"], "marker": "^"},
    }
    x = np.arange(len(DATASETS))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))

    method_offsets = {
        "GB partition": -0.16,
        "Exact kNN": 0.0,
        "k-means partition": 0.16,
    }
    for method_label, style in styles.items():
        panel = (
            source[source["method_label"] == method_label]
            .set_index("dataset")
            .loc[DATASETS]
        )
        axes[0].scatter(
            x + method_offsets[method_label],
            panel["end_to_end_structure_seconds"],
            label=method_label,
            color=style["color"],
            marker=style["marker"],
            s=31,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        axes[1].scatter(
            x + method_offsets[method_label],
            panel["structure_storage_kib"],
            label=method_label,
            color=style["color"],
            marker=style["marker"],
            s=31,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )

    axes[0].set_ylabel("Construction time (s)", fontsize=7.4)
    axes[0].set_ylim(0, 15.5)
    axes[0].set_yticks([0, 5, 10, 15])
    axes[0].set_title("End-to-end construction", fontsize=8.4, fontweight="bold")

    axes[1].set_yscale("log", base=2)
    axes[1].set_ylim(24, 2600)
    axes[1].set_yticks([32, 64, 128, 256, 512, 1024, 2048])
    axes[1].set_yticklabels(["32", "64", "128", "256", "512", "1024", "2048"])
    axes[1].set_ylabel("Structure storage (KiB, log2)", fontsize=7.4)
    axes[1].set_title("Stored grouping structure", fontsize=8.4, fontweight="bold")

    accuracy_offsets = {"Exact kNN": -0.09, "k-means partition": 0.09}
    for method_label in ["Exact kNN", "k-means partition"]:
        style = styles[method_label]
        panel = (
            source[source["method_label"] == method_label]
            .set_index("dataset")
            .loc[DATASETS]
        )
        axes[2].scatter(
            x + accuracy_offsets[method_label],
            panel["accuracy_delta_vs_gb_pp"],
            label=method_label,
            color=style["color"],
            marker=style["marker"],
            s=31,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
    axes[2].axhline(0, color="#7D8994", linewidth=0.8, linestyle=(0, (4, 3)))
    axes[2].axhspan(-1.0, 1.0, color=PALETTE["blue_soft"], alpha=0.45, zorder=0)
    axes[2].set_ylim(-1.35, 3.55)
    axes[2].set_yticks([-1, 0, 1, 2, 3])
    axes[2].set_ylabel("Accuracy difference vs GB (pp)", fontsize=7.4)
    axes[2].set_title("Alternative grouping accuracy", fontsize=8.4, fontweight="bold")
    axes[2].annotate(
        "+3.11",
        xy=(2 + accuracy_offsets["Exact kNN"], 3.1094),
        xytext=(2.22, 2.65),
        fontsize=6.8,
        color=PALETTE["orange"],
        arrowprops={"arrowstyle": "-", "color": PALETTE["orange"], "lw": 0.7},
    )

    dataset_ticks = ["Cora", "CiteSeer", "CoraFull", "Coauthor\n-CS", "Amazon\n-Photo"]
    for panel_idx, ax in enumerate(axes):
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_ticks, rotation=28, ha="right", fontsize=6.7)
        ax.grid(axis="y", color=PALETTE["light"], linewidth=0.7)
        ax.tick_params(labelsize=6.7, length=2.5)
        ax.text(
            -0.18,
            1.08,
            chr(ord("a") + panel_idx),
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        fontsize=7,
        handlelength=2.1,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.26, top=0.79, wspace=0.40)
    save_figure(fig, FIGURE_DIR / "fig5_grouping_profile_v2")
    plt.close(fig)


def add_arrow(ax, start, end, label=None, color=None, y_offset=0.0) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.25,
        color=color or PALETTE["ink"],
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color=PALETTE["muted"],
        )


def draw_graph(ax, center=(0.12, 0.50), scale=0.10) -> None:
    nodes = np.array(
        [
            [-0.85, 0.10],
            [-0.55, 0.78],
            [-0.10, 0.35],
            [0.20, 0.92],
            [0.55, 0.28],
            [0.82, 0.72],
            [0.02, -0.50],
            [-0.62, -0.58],
            [0.72, -0.42],
        ]
    )
    edges = [(0, 1), (0, 2), (0, 7), (1, 2), (1, 3), (2, 3), (2, 4), (2, 6),
             (3, 5), (4, 5), (4, 6), (4, 8), (6, 7), (6, 8)]
    xy = np.column_stack((center[0] + scale * nodes[:, 0], center[1] + scale * nodes[:, 1]))
    for a, b in edges:
        ax.plot(
            [xy[a, 0], xy[b, 0]],
            [xy[a, 1], xy[b, 1]],
            color="#8996A3",
            linewidth=0.85,
            zorder=1,
        )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=35,
        facecolor="#E1E7EC",
        edgecolor="#647585",
        linewidth=0.8,
        zorder=2,
    )


def draw_mlp(ax, x=0.82, y=0.50) -> None:
    layers = [
        [(x, y - 0.08), (x, y - 0.025), (x, y + 0.03), (x, y + 0.085)],
        [(x + 0.055, y - 0.065), (x + 0.055, y), (x + 0.055, y + 0.065)],
        [(x + 0.11, y - 0.035), (x + 0.11, y + 0.035)],
    ]
    for left, right in zip(layers[:-1], layers[1:]):
        for x1, y1 in left:
            for x2, y2 in right:
                ax.plot([x1, x2], [y1, y2], color="#9FB7CE", linewidth=0.45, zorder=1)
    for layer in layers:
        for px, py in layer:
            ax.add_patch(Circle((px, py), 0.009, facecolor="white", edgecolor=PALETTE["blue"], linewidth=0.9, zorder=2))


def plot_graphical_abstract() -> None:
    rng = np.random.default_rng(21)
    fig, ax = plt.subplots(figsize=(7.38, 2.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel_specs = [
        (0.015, 0.12, 0.285, 0.78, "1", "Label-free graph encoding"),
        (0.355, 0.12, 0.315, 0.78, "2", "Granular-ball LOO targets"),
        (0.725, 0.12, 0.26, 0.78, "3", "X-only prediction"),
    ]
    for x, y, w, h, number, title in panel_specs:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.018",
                facecolor="#FBFCFD",
                edgecolor="#B9C3CC",
                linewidth=0.9,
            )
        )
        ax.add_patch(Circle((x + 0.024, y + h - 0.045), 0.018, facecolor=PALETTE["ink"], edgecolor="none"))
        ax.text(x + 0.024, y + h - 0.045, number, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        ax.text(x + 0.05, y + h - 0.047, title, ha="left", va="center", fontsize=8.1, fontweight="bold", color=PALETTE["ink"])

    draw_graph(ax, center=(0.105, 0.51), scale=0.105)
    ax.text(0.105, 0.30, "Training graph  (X, A)", ha="center", fontsize=7.2, color=PALETTE["ink"])
    ax.add_patch(
        FancyBboxPatch(
            (0.205, 0.40),
            0.065,
            0.21,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=PALETTE["blue_soft"],
            edgecolor=PALETTE["blue"],
            linewidth=0.9,
        )
    )
    ax.text(0.2375, 0.505, "GRACE", ha="center", va="center", fontsize=7.7, fontweight="bold", color=PALETTE["blue"])
    ax.text(0.2375, 0.455, "no labels", ha="center", va="center", fontsize=6.4, color=PALETTE["muted"])
    add_arrow(ax, (0.175, 0.51), (0.202, 0.51))
    ax.text(0.155, 0.20, "Frozen representation H", ha="center", fontsize=7.2, color=PALETTE["blue"])

    ball_centers = [(0.415, 0.62), (0.505, 0.64), (0.585, 0.60), (0.435, 0.39), (0.545, 0.39), (0.625, 0.38)]
    ball_radii = [0.060, 0.058, 0.057, 0.064, 0.068, 0.052]
    class_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["pink"], PALETTE["gray"]]
    for idx, ((cx, cy), radius) in enumerate(zip(ball_centers, ball_radii)):
        selected = idx == 4
        ax.add_patch(
            Circle(
                (cx, cy),
                radius,
                fill=False,
                edgecolor=PALETTE["orange"] if selected else "#7E95AA",
                linewidth=1.5 if selected else 0.8,
                linestyle="-" if selected else (0, (3, 2)),
            )
        )
        points = []
        while len(points) < (10 if selected else 7):
            candidate = rng.uniform(-radius * 0.72, radius * 0.72, size=2)
            if np.linalg.norm(candidate) <= radius * 0.72:
                points.append(candidate)
        for p_idx, (dx, dy) in enumerate(points):
            is_labeled = selected and p_idx < 5
            face = class_colors[[2, 2, 2, 1, 4][p_idx]] if is_labeled else "white"
            edge = face if is_labeled else "#A8B5C0"
            ax.add_patch(Circle((cx + dx, cy + dy), 0.0062, facecolor=face, edgecolor=edge, linewidth=0.65))
        if selected:
            target_x, target_y = cx - 0.018, cy + 0.006
            ax.add_patch(Circle((target_x, target_y), 0.011, fill=False, edgecolor=PALETTE["orange"], linewidth=1.2))
            ax.plot([target_x - 0.008, target_x + 0.008], [target_y - 0.008, target_y + 0.008], color=PALETTE["orange"], linewidth=1.0)
            ax.plot([target_x - 0.008, target_x + 0.008], [target_y + 0.008, target_y - 0.008], color=PALETTE["orange"], linewidth=1.0)

    ax.text(0.512, 0.23, "Partition all nodes; only training labels vote", ha="center", fontsize=6.9, color=PALETTE["muted"])
    bars_x = np.linspace(0.608, 0.648, 5)
    heights = [0.015, 0.035, 0.080, 0.022, 0.014]
    for x, h, color in zip(bars_x, heights, class_colors):
        ax.add_patch(Rectangle((x, 0.68), 0.0065, h, facecolor=color, edgecolor="none"))
    ax.text(0.628, 0.79, "soft target", ha="center", fontsize=6.5, color=PALETTE["muted"])
    add_arrow(ax, (0.575, 0.69), (0.603, 0.71), label="LOO votes", y_offset=0.014, color=PALETTE["orange"])

    feature_x = 0.748
    for idx, value in enumerate([0.9, 0.45, 0.75, 0.28]):
        ax.add_patch(Rectangle((feature_x + idx * 0.012, 0.47), 0.010, 0.13, facecolor=PALETTE["blue"], alpha=0.25 + 0.65 * value, edgecolor="none"))
    ax.text(feature_x + 0.018, 0.43, r"$x_v$", ha="center", fontsize=7.2, color=PALETTE["ink"])
    add_arrow(ax, (0.802, 0.535), (0.815, 0.535))
    draw_mlp(ax, x=0.825, y=0.535)
    add_arrow(ax, (0.945, 0.535), (0.956, 0.535))
    prob_x = [0.958, 0.966, 0.974, 0.982]
    prob_h = [0.025, 0.055, 0.12, 0.035]
    for x, h in zip(prob_x, prob_h):
        ax.add_patch(Rectangle((x, 0.455), 0.005, h, facecolor=PALETTE["green"] if h == max(prob_h) else "#C8D2DB", edgecolor="none"))
    ax.text(0.875, 0.30, "Trained MLP", ha="center", fontsize=7.3, fontweight="bold", color=PALETTE["ink"])
    ax.text(0.875, 0.235, "Inference uses X only", ha="center", fontsize=7.5, color=PALETTE["blue"], fontweight="bold")

    add_arrow(ax, (0.302, 0.51), (0.349, 0.51), label="frozen H", y_offset=0.02, color=PALETTE["blue"])
    add_arrow(ax, (0.672, 0.51), (0.719, 0.51), label=r"targets $t_v$", y_offset=0.02, color=PALETTE["orange"])
    ax.text(
        0.5,
        0.045,
        "Graph-assisted supervision, X-only deployment",
        ha="center",
        va="center",
        fontsize=8.2,
        color=PALETTE["ink"],
        fontweight="bold",
    )

    save_figure(fig, FIGURE_DIR / "graphical_abstract_gblcss_v1", tight=False)
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    source = build_gain_source_data()
    plot_gain_heatmap(source)
    uniform_source = build_uniform_rate_source_data()
    plot_uniform_rate_curves(uniform_source)
    target_quality_source = build_target_quality_source_data()
    plot_target_quality(target_quality_source)
    grouping_profile_source = build_grouping_profile_source_data()
    plot_grouping_profile(grouping_profile_source)
    print(f"Created manuscript figures 2-5 in {FIGURE_DIR}")
    print(f"Heatmap source data: {SOURCE_DATA_DIR / 'fig2_gblcss_gain_heatmap_source.csv'}")
    print(
        "Uniform-rate source data: "
        f"{SOURCE_DATA_DIR / 'fig3_uniform_noise_rate_curves_source.csv'}"
    )
    print(
        "Target-quality source data: "
        f"{SOURCE_DATA_DIR / 'fig4_target_quality_source.csv'}"
    )
    print(
        "Grouping-profile source data: "
        f"{SOURCE_DATA_DIR / 'fig5_grouping_profile_source.csv'}"
    )


if __name__ == "__main__":
    main()
