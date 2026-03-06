"""
Paper-grade plots from the aggregated paper_comparison.csv.

Plot 1 – Best single model (lowest mean_rmse_val):
    Grouped bar chart of per-target RMSE for Validation vs Test,
    with baseline dummy-predictor markers and a reference line at 1.0.

Plot 2 – All-model comparison:
    Two side-by-side subplots (Validation / Test).  Each subplot shows a
    grouped bar chart with one bar per model for the mean RMSE, plus
    individual target scores as semi-transparent scatter points to convey
    the spread without mis-representing statistics.
"""

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
#  Global style for publication-quality figures                       #
# ------------------------------------------------------------------ #
_PAPER_RC = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Colour-blind-friendly qualitative palette (Tol's muted)
_CB_PALETTE = [
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
    "#DDDDDD",  # grey
]

# Target column prefixes in the CSV (order matters for x-axis)
TARGET_PREFIXES: List[str] = [
    "robotrainer_front",
    "robotrainer_left",
    "robotrainer_right",
    "hand_grip_left",
    "hand_grip_right",
    "jump_and_reach",
    "figure_8_walk",
    "ruler_drop_test",
]

# Readable display names
TARGET_LABELS: List[str] = [
    "Max F\nFront",
    "Max F\nLeft",
    "Max F\nRight",
    "Hand\nGrip Left",
    "Hand Grip\nRight",
    "Jump &\nReach",
    "Figure-8\nWalk",
    "Ruler\nDrop",
]


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
def _apply_style() -> None:
    plt.rcParams.update(_PAPER_RC)


def _get_target_values(
    row: pd.Series, metric: str, split: str
) -> np.ndarray:
    """Return an array of per-target values for *row*."""
    return np.array(
        [row.get(f"{tp}_{metric}_{split}", np.nan) for tp in TARGET_PREFIXES]
    )


# ------------------------------------------------------------------ #
#  Plot 1 – Best model detail                                        #
# ------------------------------------------------------------------ #
def plot_best_model(
    df: pd.DataFrame,
    output_dir: str,
    filename: str = "best_model_rmse.png",
) -> str:
    """
    Grouped bar chart for the best model (lowest ``mean_rmse_val``).

    * Two bars per target: Validation RMSE and Test RMSE
    * Diamond markers for baseline (dummy mean predictor) per split
    * Horizontal dashed line at RMSE = 1.0
    """
    _apply_style()

    best_idx = df["mean_rmse_val"].idxmin()
    best = df.loc[best_idx]
    model_name = best["model"]

    val_rmse = _get_target_values(best, "rmse", "val")
    test_rmse = _get_target_values(best, "rmse", "test")
    val_bl = _get_target_values(best, "baseline_rmse", "val")
    test_bl = _get_target_values(best, "baseline_rmse", "test")

    n = len(TARGET_PREFIXES)
    x = np.arange(n) * 0.75  # tighter spacing between target groups
    width = 0.32

    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Bars
    color_val = "#4477AA"
    color_test = "#EE6677"

    bars_val = ax.bar(
        x - width / 2, val_rmse, width,
        label="Validation", color=color_val, edgecolor="none",
    )
    bars_test = ax.bar(
        x + width / 2, test_rmse, width,
        label="Test", color=color_test, edgecolor="none",
    )

    # Baseline markers
    ax.scatter(
        x - width / 2, val_bl,
        marker="D", s=40, zorder=5,
        facecolors=color_val, edgecolors="black", linewidths=0.7,
        label="Val Mean Baseline",
    )
    ax.scatter(
        x + width / 2, test_bl,
        marker="D", s=40, zorder=5,
        facecolors=color_test, edgecolors="black", linewidths=0.7,
        label="Test Mean Baseline",
    )

    # Reference line at 1.0
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, alpha=0.7, label="RMSE = 1.0")

    # Bar value labels
    # For the last 3 targets the val baseline diamond sits right above the
    # val bar, so place those labels *inside* the bar instead.
    _obstructed_val = set(range(n - 3, n))  # indices 5, 6, 7

    for bar_idx, bar in enumerate(bars_val):
        h = bar.get_height()
        if not np.isnan(h):
            if bar_idx in _obstructed_val:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h - 0.02,
                    f"{h:.2f}",
                    ha="center", va="top", fontsize=7,
                    color="white", fontweight="bold",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.02,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

    for bar in bars_test:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_LABELS)
    ax.set_ylabel("RMSE (scaled)")
    y_max = max(np.nanmax(val_bl), np.nanmax(test_bl)) * 1.03
    ax.set_ylim(0, y_max)
    ax.set_xlim(x[0] - width - 0.08, x[-1] + width)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="lightgrey")

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[PaperPlot] Saved best-model plot → {save_path}")
    return save_path


# ------------------------------------------------------------------ #
#  Plot 2 – All-model comparison                                     #
# ------------------------------------------------------------------ #
def plot_all_models(
    df: pd.DataFrame,
    output_dir: str,
    filename: str = "all_models_comparison.png",
) -> str:
    """
    Two subplots (Validation | Test).

    Each subplot: one bar per model showing **mean RMSE**, with individual
    per-target scores overlaid as jittered dots so the reader can judge
    spread without a misleading box-plot median.
    """
    _apply_style()

    # Sort models by mean_rmse_val for consistent ordering
    df = df.sort_values("mean_rmse_val").reset_index(drop=True)
    models = df["model"].tolist()
    n_models = len(models)

    x = np.arange(n_models)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    split_info: List[Tuple[str, str, str]] = [
        ("val", "Validation", "#4477AA"),
        ("test", "Test", "#EE6677"),
    ]

    for ax, (split, split_label, base_color) in zip(axes, split_info):
        means = df[f"mean_rmse_{split}"].values

        # Bars for mean RMSE
        bars = ax.bar(
            x, means, width=0.55,
            color=base_color, alpha=0.85,
            edgecolor="white", linewidth=0.6,
            label=f"Mean RMSE ({split_label})",
        )

        # Overlay individual target scores as jittered dots
        rng = np.random.default_rng(42)
        for i, (_, row) in enumerate(df.iterrows()):
            target_vals = _get_target_values(row, "rmse", split)
            jitter = rng.uniform(-0.18, 0.18, size=len(target_vals))
            ax.scatter(
                np.full_like(target_vals, i) + jitter,
                target_vals,
                color="black", alpha=0.45, s=18, zorder=5,
                edgecolors="none",
            )

        # Reference line
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, alpha=0.7, label="RMSE = 1.0")

        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=40, ha="right")
        ax.set_title(f"{split_label} Split")
        ax.set_ylabel("RMSE (scaled)" if split == "val" else "")
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="lightgrey")

    y_max_global = 0
    for split in ("val", "test"):
        col_max = df[f"mean_rmse_{split}"].max()
        target_max = max(
            df[[f"{tp}_rmse_{split}" for tp in TARGET_PREFIXES]].max().max(),
            col_max,
        )
        y_max_global = max(y_max_global, target_max)
    axes[0].set_ylim(0, y_max_global * 1.15)

    fig.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[PaperPlot] Saved all-models plot → {save_path}")
    return save_path


# ------------------------------------------------------------------ #
#  Convenience entry point                                           #
# ------------------------------------------------------------------ #
def generate_paper_plots(
    csv_path: str,
    output_dir: Optional[str] = None,
) -> None:
    """Read *csv_path* and produce both paper plots."""
    df = pd.read_csv(csv_path)

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    plot_best_model(df, output_dir)
    plot_all_models(df, output_dir)


# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate paper-grade comparison plots from paper_comparison.csv."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="/workspace/automatic_assessment/experiment_results/final_best/paper_comparison.csv",
        help="Path to the paper_comparison.csv file.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory for output plots (default: same as CSV).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        print(f"CSV not found: {args.csv_path}")
        raise SystemExit(1)

    generate_paper_plots(args.csv_path, args.output_dir)
