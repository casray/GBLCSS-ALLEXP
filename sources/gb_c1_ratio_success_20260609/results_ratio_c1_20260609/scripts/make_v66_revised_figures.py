from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parents[1]
FIGURE_DIR = ROOT / "figures"
SOURCE_DATA_DIR = FIGURE_DIR / "source_data"
FIG3_SOURCE = SOURCE_DATA_DIR / "fig3_uniform_noise_rate_curves_source.csv"
LOO_RAW_CSV = (
    EXPERIMENT_ROOT
    / "external"
    / "NoisyGL"
    / "runs"
    / "mlp44_loo_tiesafe_fullspectrum_20260730"
    / "mlp44_combined_raw.csv"
)

OURS = "frozengrace_gb_mlp_softdist_beta05_loo"
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

COLORS = {
    "ink": "#26323D",
    "muted": "#6B7785",
    "grid": "#E6EBEF",
    "blue": "#1578B7",
    "orange": "#E39A00",
    "green": "#169C76",
    "pink": "#C9799A",
    "dark_gray": "#58595B",
    "mid_gray": "#8F969C",
    "light_gray": "#B8BEC3",
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


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def load_fig3_source() -> pd.DataFrame:
    source = pd.read_csv(FIG3_SOURCE)
    expected_methods = {"GB-LCSS", "JoCoR", "MLP+CE", "Random grouping"}
    expected_rows = len(DATASETS) * 3 * len(expected_methods)
    if len(source) != expected_rows:
        raise ValueError(f"Expected {expected_rows} Fig. 3 rows, found {len(source)}")
    if set(source["dataset"]) != set(DATASETS):
        raise ValueError("Fig. 3 dataset set does not match the manuscript")
    if set(source["method_label"]) != expected_methods:
        raise ValueError("Fig. 3 method set does not match the manuscript")
    if not source["n"].eq(5).all():
        raise ValueError("Every Fig. 3 point must aggregate five paired seeds")
    return source


def plot_uniform_rate_curves(source: pd.DataFrame) -> None:
    styles = {
        "GB-LCSS": {
            "color": COLORS["blue"],
            "linewidth": 2.0,
            "linestyle": "-",
            "zorder": 5,
        },
        "JoCoR": {
            "color": COLORS["orange"],
            "linewidth": 1.45,
            "linestyle": "-",
            "zorder": 4,
        },
        "MLP+CE": {
            "color": COLORS["dark_gray"],
            "linewidth": 1.35,
            "linestyle": "-",
            "zorder": 3,
        },
        "Random grouping": {
            "color": COLORS["light_gray"],
            "linewidth": 1.3,
            "linestyle": "--",
            "zorder": 2,
        },
    }
    y_specs = {
        "cora": ((28, 66), [30, 40, 50, 60]),
        "citeseer": ((27, 67), [30, 40, 50, 60]),
        "corafull": ((18, 46), [20, 30, 40]),
        "coauthorcs": ((76, 94), [80, 85, 90]),
        "amazonpho": ((54, 90), [60, 70, 80, 90]),
    }

    fig, axes = plt.subplots(1, 5, figsize=(10.6, 3.15), sharex=True)
    for ax, dataset in zip(axes, DATASETS):
        panel = source[source["dataset"] == dataset]
        for method, style in styles.items():
            values = panel[panel["method_label"] == method].sort_values("noise_rate")
            if len(values) != 3:
                raise ValueError(f"Missing noise-rate point for {dataset}, {method}")
            x = values["noise_rate"].to_numpy(dtype=float)
            mean = values["mean_accuracy_pct"].to_numpy(dtype=float)
            sd = values["std_accuracy_pct"].to_numpy(dtype=float)
            ax.fill_between(
                x,
                mean - sd,
                mean + sd,
                color=style["color"],
                alpha=0.13 if method == "GB-LCSS" else 0.09,
                linewidth=0,
                zorder=style["zorder"] - 1,
            )
            ax.plot(
                x,
                mean,
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                marker="o",
                markersize=4.4,
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                markeredgewidth=0.4,
                zorder=style["zorder"],
                label=method,
            )

        limits, ticks = y_specs[dataset]
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        ax.set_xticks([0.2, 0.4, 0.6])
        ax.set_xticklabels(["0.2", "0.4", "0.6"])
        ax.set_title(DATASET_LABELS[dataset], fontsize=9, fontweight="bold", pad=7)
        ax.set_xlabel("Uniform noise rate", fontsize=7.5, labelpad=5)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
        ax.tick_params(labelsize=7.2, length=2.6, width=0.7)
    axes[0].set_ylabel("Test accuracy (%)", fontsize=8, labelpad=6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        fontsize=7.8,
        handlelength=2.4,
        columnspacing=1.7,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.18, top=0.78, wspace=0.34)
    save_figure(fig, FIGURE_DIR / "fig3_uniform_noise_rate_curves_v2")
    plt.close(fig)


def load_target_quality_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(LOO_RAW_CSV)
    selected = raw[
        (raw["method"] == OURS)
        & raw["dataset"].isin(DATASETS)
        & raw["noise_type"].isin({"uniform", "pair", "instance"})
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
        raise ValueError(f"Expected {expected_rows} target-quality rows, found {len(selected)}")
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
    reconstructed = selected["noisy_onehot_brier"] - selected["soft_target_brier"]
    if not np.allclose(reconstructed, selected["soft_brier_gain"], atol=1e-9, rtol=1e-7):
        raise ValueError("Stored Brier gains do not match their components")
    if not selected["soft_brier_gain"].gt(0).all():
        raise ValueError("Every paired run must have positive Brier gain")

    selected["dataset_label"] = selected["dataset"].map(DATASET_LABELS)
    setting_map = {(kind, rate): label.replace("\n", " ") for kind, rate, label in SETTINGS}
    selected["setting"] = selected.apply(
        lambda row: setting_map[(row["noise_type"], float(row["noise_rate"]))], axis=1
    )
    raw_source = selected[
        [
            "dataset",
            "dataset_label",
            "noise_type",
            "noise_rate",
            "setting",
            "seed",
            "noisy_onehot_brier",
            "soft_target_brier",
            "soft_brier_gain",
        ]
    ].sort_values(["dataset", "noise_type", "noise_rate", "seed"])

    aggregate = (
        raw_source.groupby(
            ["dataset", "dataset_label", "noise_type", "noise_rate", "setting"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            noisy_onehot_brier_mean=("noisy_onehot_brier", "mean"),
            noisy_onehot_brier_sd=("noisy_onehot_brier", "std"),
            soft_target_brier_mean=("soft_target_brier", "mean"),
            soft_target_brier_sd=("soft_target_brier", "std"),
            brier_gain_mean=("soft_brier_gain", "mean"),
            brier_gain_sd=("soft_brier_gain", "std"),
        )
    )
    if len(aggregate) != 25 or not aggregate["seeds"].eq(5).all():
        raise ValueError("Expected 25 aggregate settings with five unique seeds")
    return raw_source, aggregate


def plot_target_quality(raw_source: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    dataset_colors = {
        "cora": COLORS["blue"],
        "citeseer": COLORS["orange"],
        "corafull": COLORS["green"],
        "coauthorcs": COLORS["pink"],
        "amazonpho": COLORS["mid_gray"],
    }
    noise_markers = {"uniform": "o", "pair": "s", "instance": "^"}
    marker_sizes = {"uniform": 36, "pair": 34, "instance": 42}
    fig, (ax_scatter, ax_heatmap) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.25),
        gridspec_kw={"width_ratios": [1.0, 1.18]},
    )

    lower = min(raw_source["noisy_onehot_brier"].min(), raw_source["soft_target_brier"].min())
    upper = max(raw_source["noisy_onehot_brier"].max(), raw_source["soft_target_brier"].max())
    margin = 0.045 * (upper - lower)
    limits = (max(0.0, lower - margin), upper + margin)
    ax_scatter.plot(
        limits,
        limits,
        color="#9EA8B1",
        linewidth=0.9,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    for dataset in DATASETS:
        mean_panel = aggregate[aggregate["dataset"] == dataset]
        for noise_type, marker in noise_markers.items():
            points = mean_panel[mean_panel["noise_type"] == noise_type]
            ax_scatter.scatter(
                points["noisy_onehot_brier_mean"],
                points["soft_target_brier_mean"],
                s=marker_sizes[noise_type],
                marker=marker,
                color=dataset_colors[dataset],
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
    ax_scatter.set_xlim(limits)
    ax_scatter.set_ylim(limits)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_xlabel("Noisy one-hot target Brier score", fontsize=7.5)
    ax_scatter.set_ylabel("GB-LCSS soft-target Brier score", fontsize=7.5)
    ax_scatter.set_title("Soft targets have lower Brier error", fontsize=8.6, fontweight="bold")
    ax_scatter.grid(color=COLORS["grid"], linewidth=0.65)
    ax_scatter.tick_params(labelsize=7, length=2.5)
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5.3,
            markerfacecolor=dataset_colors[dataset],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=DATASET_LABELS[dataset],
        )
        for dataset in DATASETS
    ]
    noise_handles = [
        Line2D(
            [0],
            [0],
            marker=noise_markers[noise_type],
            linestyle="none",
            markersize=5.0,
            markerfacecolor="#7F8992",
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
        )
        for noise_type, label in [
            ("uniform", "Uniform"),
            ("pair", "Pair"),
            ("instance", "Instance"),
        ]
    ]
    dataset_legend = ax_scatter.legend(
        handles=dataset_handles,
        loc="upper left",
        fontsize=6.1,
        handletextpad=0.35,
        borderaxespad=0.35,
        labelspacing=0.32,
    )
    ax_scatter.add_artist(dataset_legend)
    ax_scatter.legend(
        handles=noise_handles,
        loc="lower right",
        fontsize=6.0,
        handletextpad=0.30,
        borderaxespad=0.45,
        labelspacing=0.28,
        title="Noise type",
        title_fontsize=6.1,
    )

    matrix = np.empty((len(DATASETS), len(SETTINGS)), dtype=float)
    for i, dataset in enumerate(DATASETS):
        for j, (_, _, setting_label) in enumerate(SETTINGS):
            setting = setting_label.replace("\n", " ")
            value = aggregate.loc[
                (aggregate["dataset"] == dataset) & (aggregate["setting"] == setting),
                "brier_gain_mean",
            ]
            if len(value) != 1:
                raise ValueError(f"Missing Brier gain for {dataset}, {setting}")
            matrix[i, j] = float(value.iloc[0])

    cmap = LinearSegmentedColormap.from_list(
        "brier_gain",
        ["#F5F7F8", "#BFD6E6", "#2F6B9A"],
        N=256,
    )
    image = ax_heatmap.pcolormesh(
        np.arange(len(SETTINGS) + 1) - 0.5,
        np.arange(len(DATASETS) + 1) - 0.5,
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
    ax_heatmap.set_title("Mean Brier-score reduction", fontsize=8.6, fontweight="bold")
    for spine in ax_heatmap.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax_heatmap.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value >= 0.34 else COLORS["ink"],
            )
    cbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.042, pad=0.025)
    cbar.solids.set_rasterized(False)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    cbar.set_label("Noisy target - soft target", fontsize=6.8, labelpad=5)
    cbar.outline.set_linewidth(0.55)

    ax_scatter.text(-0.16, 1.06, "a", transform=ax_scatter.transAxes, fontsize=9, fontweight="bold")
    ax_heatmap.text(-0.16, 1.06, "b", transform=ax_heatmap.transAxes, fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.88, wspace=0.34)
    save_figure(fig, FIGURE_DIR / "fig4_target_quality_v4")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    fig3_source = load_fig3_source()
    plot_uniform_rate_curves(fig3_source)
    raw_source, aggregate = load_target_quality_data()
    raw_source.to_csv(
        SOURCE_DATA_DIR / "fig4_target_quality_raw_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        SOURCE_DATA_DIR / "fig4_target_quality_source_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_target_quality(raw_source, aggregate)
    print("Created revised manuscript Figures 3 and 4")
    print(f"Fig. 3: {FIGURE_DIR / 'fig3_uniform_noise_rate_curves_v2.png'}")
    print(f"Fig. 4: {FIGURE_DIR / 'fig4_target_quality_v4.png'}")


if __name__ == "__main__":
    main()
