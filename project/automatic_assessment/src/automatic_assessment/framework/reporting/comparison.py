"""
Cross-run comparison of experiment results (artifact schema v3, with a
fallback for legacy flat-key folders).

Point it at a directory that contains run folders (each with config.yaml +
metrics.yaml) and it aggregates three tidy tables:

- per-target metrics  (run x split x target: rmse / r2 / baseline_rmse)
- per-run summaries   (mode, ratio, mean metrics, best_trial_val_loss,
                       final_epochs)
- per-fold summaries  (run x held-out user: val_loss, best_epoch — v3 only)

Plots (written to <root>/comparison_results/):
- ratio_summary_<split>.png            : mean RMSE vs augmentation ratio
                                         (bold mean + per-target spaghetti,
                                         baseline reference)
- ratio_fold_distributions.png         : per-user fold val-loss distribution
                                         per ratio (does the ratio shift the
                                         whole distribution or just the mean?)
- augmentation_ratio_comparison_*.png  : per-target bars vs ratio
- detailed_model_comparison_*.png      : per-target bars across model classes
- aggregated_model_comparison_*.png    : RMSE distribution per model class
- multitarget_comparison_*.png         : across target-set configurations
- <split>_aggregated_metrics.csv, run_summaries.csv, fold_summaries.csv
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


class ExperimentComparison:
    def __init__(self, experiments_root_path: str, split: str = "val"):
        """
        Args:
            experiments_root_path: Directory containing run folders.
            split: Which split the plots show ("val" is the project's
                   primary metric; "test" is the 3-user normative spot check).
        """
        if split not in ("test", "val"):
            raise ValueError(f"split must be 'test' or 'val', got '{split}'")
        self.root_path = experiments_root_path
        self.split = split
        self.df = pd.DataFrame()      # per (run, split, target)
        self.runs = pd.DataFrame()    # per run
        self.folds = pd.DataFrame()   # per (run, fold)  [v3 only]

    # ------------------------------------------------------------------
    # Loading (v3 + legacy)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_model_name(folder_name: str) -> str:
        parts = folder_name.split('_')
        return "_".join(parts[2:]) if len(parts) >= 3 else "UnknownModel"

    @staticmethod
    def _config_type(targets: List[str]) -> str:
        n = len(targets)
        if n == 1:
            return "Single Target"
        if n == 8:
            return "Reduced (8 Targets)"
        if n > 10:
            return "All Targets"
        return f"{n} Targets"

    def _load_run_v3(self, folder: str, config: dict, metrics: dict,
                     rows: list, run_rows: list, fold_rows: list) -> None:
        exp = config.get("experiment", {})
        targets = exp.get("targets", [])
        base = {
            "Folder": folder,
            "Model Class": self._parse_model_name(folder),
            "Mode": exp.get("hyperparameter_mode", "default"),
            "Augmentation Ratio": exp.get("augmentation_ratio"),
            "Config Type": self._config_type(targets),
        }

        for split in ("val", "test"):
            block = metrics.get(split, {}).get("scaled", {})
            for target, m in block.get("per_target", {}).items():
                rows.append({**base, "Split": split, "Target": target,
                             "RMSE": float(m["rmse"]), "R2": float(m["r2"]),
                             "Baseline RMSE": float(m.get("baseline_rmse", np.nan))})

        val = metrics.get("val", {})
        test = metrics.get("test", {})
        run_rows.append({**base,
                         "val_rmse_mean": val.get("scaled", {}).get("rmse_mean"),
                         "val_r2_mean": val.get("scaled", {}).get("r2_mean"),
                         "val_baseline_rmse_mean": val.get("scaled", {}).get("baseline_rmse_mean"),
                         "val_loss": val.get("loss"),
                         "test_rmse_mean": test.get("scaled", {}).get("rmse_mean"),
                         "best_trial_val_loss": metrics.get("best_trial_val_loss"),
                         "final_epochs": config.get("final_epochs")})

        for f in metrics.get("folds", []):
            fold_rows.append({**base, "fold": f.get("fold"), "user_id": f.get("user_id"),
                              "val_loss": f.get("val_loss"), "best_epoch": f.get("best_epoch")})

    def _load_run_legacy(self, folder: str, config: dict, metrics: dict,
                         rows: list, run_rows: list) -> None:
        pipeline_cfg = config.get("pipeline_config", {})
        targets = pipeline_cfg.get("targets", [])
        base = {
            "Folder": folder,
            "Model Class": self._parse_model_name(folder),
            "Mode": pipeline_cfg.get("hyperparameter_mode", "default"),
            "Augmentation Ratio": pipeline_cfg.get("augmentation_ratio"),
            "Config Type": self._config_type(targets),
        }
        for split in ("val", "test"):
            for i, target in enumerate(targets):
                rmse = metrics.get(f"{split}_rmse_target_{i}")
                if rmse is None:
                    continue
                rows.append({**base, "Split": split, "Target": target,
                             "RMSE": float(rmse),
                             "R2": float(metrics.get(f"{split}_r2_target_{i}", np.nan)),
                             "Baseline RMSE": float(metrics.get(f"{split}_baseline_rmse_target_{i}", np.nan))})
        run_rows.append({**base,
                         "val_rmse_mean": metrics.get("val_rmse_mean"),
                         "val_r2_mean": metrics.get("val_r2_mean"),
                         "val_baseline_rmse_mean": metrics.get("val_baseline_rmse_mean"),
                         "val_loss": metrics.get("val_loss"),
                         "test_rmse_mean": metrics.get("test_rmse_mean"),
                         "best_trial_val_loss": metrics.get("best_trial_val_loss"),
                         "final_epochs": config.get("final_epochs")})

    def load_data(self) -> pd.DataFrame:
        """Scans all run folders and builds the tidy tables."""
        rows, run_rows, fold_rows = [], [], []

        for folder_name in sorted(os.listdir(self.root_path)):
            folder_path = os.path.join(self.root_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            config_path = os.path.join(folder_path, "config.yaml")
            metrics_path = os.path.join(folder_path, "metrics.yaml")
            if not (os.path.exists(config_path) and os.path.exists(metrics_path)):
                continue

            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}
                with open(metrics_path) as f:
                    metrics = yaml.safe_load(f) or {}

                if metrics.get("schema_version", 0) >= 3:
                    self._load_run_v3(folder_name, config, metrics, rows, run_rows, fold_rows)
                else:
                    self._load_run_legacy(folder_name, config, metrics, rows, run_rows)
            except Exception as e:
                print(f"Error processing {folder_name}: {e}")

        self.df = pd.DataFrame(rows)
        self.runs = pd.DataFrame(run_rows)
        self.folds = pd.DataFrame(fold_rows)
        if not self.df.empty:
            print(f"Loaded {self.runs.shape[0]} runs / {self.df.shape[0]} per-target rows "
                  f"/ {self.folds.shape[0]} fold rows from {self.root_path}")
        return self.df

    def _ensure_loaded(self):
        if self.df.empty:
            self.load_data()

    def _out_dir(self, output_dir: Optional[str]) -> str:
        out = output_dir or os.path.join(self.root_path, "comparison_results")
        os.makedirs(out, exist_ok=True)
        return out

    def _split_df(self) -> pd.DataFrame:
        return self.df[self.df["Split"] == self.split]

    # ------------------------------------------------------------------
    # NEW: ratio summary (headline figure for the augmentation experiment)
    # ------------------------------------------------------------------

    def plot_ratio_summary(self, output_dir: Optional[str] = None) -> None:
        """
        Mean RMSE vs augmentation ratio: bold mean line, per-target spaghetti,
        dummy-baseline reference. (The best-trial vs re-evaluated CV losses
        of 'optimize' runs are NOT plotted here — compare them via
        run_summaries.csv: columns best_trial_val_loss vs val_loss.)
        """
        self._ensure_loaded()
        runs = self.runs.dropna(subset=["Augmentation Ratio"])
        if runs.empty or runs["Augmentation Ratio"].nunique() <= 1:
            print("Ratio summary skipped: no varying augmentation ratio.")
            return
        out = self._out_dir(output_dir)

        df_t = self._split_df().dropna(subset=["Augmentation Ratio"]).copy()
        df_t["Augmentation Ratio"] = df_t["Augmentation Ratio"].astype(int)
        runs = runs.copy()
        runs["Augmentation Ratio"] = runs["Augmentation Ratio"].astype(int)
        runs = runs.sort_values("Augmentation Ratio")

        mean_col = f"{self.split}_rmse_mean"

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        # per-target spaghetti
        for target, sub in df_t.groupby("Target"):
            sub = sub.sort_values("Augmentation Ratio")
            ax.plot(sub["Augmentation Ratio"], sub["RMSE"], alpha=0.35, lw=1,
                    marker='o', ms=3, label=None)
            ax.annotate(target, (sub["Augmentation Ratio"].iloc[-1], sub["RMSE"].iloc[-1]),
                        fontsize=7, alpha=0.6, xytext=(4, 0), textcoords="offset points")

        # bold mean
        ax.plot(runs["Augmentation Ratio"], runs[mean_col], color="black", lw=2.5,
                marker='o', ms=6, label=f"Mean RMSE ({self.split})")

        # baseline reference (val only; constant across ratios by construction)
        if self.split == "val" and runs["val_baseline_rmse_mean"].notna().any():
            bl = float(runs["val_baseline_rmse_mean"].mean())
            ax.axhline(bl, color="red", linestyle="--", lw=1.5, alpha=0.7,
                       label=f"Dummy baseline ({bl:.2f})")

        ax.set_xlabel("Augmentation Ratio (clones per training user)")
        ax.set_ylabel(f"RMSE ({self.split}, scaled)")
        ax.set_xticks(sorted(runs["Augmentation Ratio"].unique()))
        mode_label = "/".join(sorted(runs["Mode"].unique()))
        ax.set_title(f"Augmentation Ratio vs RMSE — {mode_label} mode ({self.split})")
        ax.legend(loc="upper left", fontsize=9)
        plt.tight_layout()
        path = os.path.join(out, f"ratio_summary_{self.split}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")

    # ------------------------------------------------------------------
    # NEW: per-fold distributions (v3 runs only)
    # ------------------------------------------------------------------

    def plot_fold_distributions(self, output_dir: Optional[str] = None) -> None:
        """
        Distribution of per-user fold val losses per augmentation ratio.
        Shows whether a ratio improves users broadly or only shifts the mean
        via a few users.
        """
        self._ensure_loaded()
        if self.folds.empty:
            print("Fold distributions skipped: no v3 fold data.")
            return
        folds = self.folds.dropna(subset=["Augmentation Ratio"]).copy()
        if folds.empty or folds["Augmentation Ratio"].nunique() <= 1:
            print("Fold distributions skipped: no varying augmentation ratio.")
            return
        out = self._out_dir(output_dir)
        folds["Augmentation Ratio"] = folds["Augmentation Ratio"].astype(int)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=folds, x="Augmentation Ratio", y="val_loss", color="lightsteelblue")
        sns.stripplot(data=folds, x="Augmentation Ratio", y="val_loss",
                      color="black", alpha=0.45, jitter=0.15, size=4)
        plt.ylabel("Per-user fold val loss (Huber, scaled)")
        plt.xlabel("Augmentation Ratio (clones per training user)")
        plt.title("Fold-level validation loss distribution per augmentation ratio")
        plt.tight_layout()
        path = os.path.join(out, "ratio_fold_distributions.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")

    # ------------------------------------------------------------------
    # Per-target bars vs augmentation ratio
    # ------------------------------------------------------------------

    def plot_augmentation_ratio_comparison(self, output_dir: Optional[str] = None) -> None:
        """Per-target RMSE bars across augmentation ratios (one figure per model)."""
        self._ensure_loaded()
        df = self._split_df().dropna(subset=["Augmentation Ratio"]).copy()
        if df.empty or df["Augmentation Ratio"].nunique() <= 1:
            print("Augmentation ratio comparison skipped: ratio does not vary.")
            return
        out = self._out_dir(output_dir)
        df["Augmentation Ratio"] = df["Augmentation Ratio"].astype(int)
        sorted_ratios = sorted(df["Augmentation Ratio"].unique())
        palette = dict(zip(sorted_ratios, sns.color_palette("coolwarm", len(sorted_ratios))))

        for model_class, model_df in df.groupby("Model Class"):
            targets = sorted(model_df["Target"].unique())
            if not targets:
                continue
            y_limit = max(1.1, model_df["RMSE"].max() * 1.15)
            n_cols = 3
            n_rows = (len(targets) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
            axes = axes.flatten()

            for i, target in enumerate(targets):
                ax = axes[i]
                subset = model_df[model_df["Target"] == target].sort_values("Augmentation Ratio")
                sns.barplot(data=subset, x="Augmentation Ratio", y="RMSE",
                            hue="Augmentation Ratio", ax=ax,
                            palette=palette, legend=False)
                ax.axhline(1.0, color="red", linestyle="--", lw=1.2, alpha=0.6)
                bl = subset["Baseline RMSE"].mean()
                if not np.isnan(bl):
                    ax.axhline(bl, color="orange", linestyle=":", lw=1.5, alpha=0.8,
                               label=f"Baseline {bl:.2f}")
                    ax.legend(fontsize=8)
                ax.set_ylim(0, y_limit)
                ax.set_title(target)
                ax.set_xlabel("Augmentation Ratio")
                ax.set_ylabel("RMSE")
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.2f", fontsize=7)

            for j in range(len(targets), len(axes)):
                axes[j].axis("off")

            split_label = "Validation" if self.split == "val" else "Test"
            plt.suptitle(f"RMSE vs Augmentation Ratio per Target ({split_label}) — {model_class}",
                         fontsize=15)
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])
            path = os.path.join(out, f"augmentation_ratio_comparison_{model_class}_{self.split}.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"Saved {path}")

    # ------------------------------------------------------------------
    # Model comparison (across model classes)
    # ------------------------------------------------------------------

    def plot_model_comparison(self, output_dir: Optional[str] = None) -> None:
        """RMSE distribution per model class + per-target bars across models."""
        self._ensure_loaded()
        df = self._split_df()
        if df.empty:
            print("No data found to plot.")
            return
        if df["Model Class"].nunique() <= 1:
            print("Model comparison skipped: only one model class present.")
            return
        out = self._out_dir(output_dir)
        split_label = "Validation" if self.split == "val" else "Test"

        # 1. Aggregated distribution
        plt.figure(figsize=(10, 8))
        order = df.groupby("Model Class")["RMSE"].mean().sort_values().index
        sns.boxplot(data=df, x="Model Class", y="RMSE", hue="Model Class",
                    order=order, palette="Set2", legend=False)
        sns.stripplot(data=df, x="Model Class", y="RMSE", order=order,
                      color="black", alpha=0.3, jitter=0.2)
        plt.axhline(1.0, color="black", linestyle="--", lw=2, label="Ref 1.0")
        plt.title(f"Model Performance Distribution across Targets ({split_label})")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"aggregated_model_comparison_{self.split}.png"), dpi=150)
        plt.close()

        # 2. Per-target bars
        targets = sorted(df["Target"].unique())
        n_cols = 3
        n_rows = (len(targets) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        axes = axes.flatten()
        y_limit = max(1.2, df["RMSE"].max() * 1.1)

        for i, target in enumerate(targets):
            ax = axes[i]
            subset = df[df["Target"] == target]
            sns.barplot(data=subset, x="Model Class", y="RMSE", hue="Model Class",
                        order=order, palette="Set2", errorbar=None, ax=ax, legend=False)
            ax.axhline(1.0, color="black", linestyle="--", lw=1.2, label="Ref 1.0")
            bl = subset["Baseline RMSE"].mean()
            if not np.isnan(bl):
                ax.axhline(bl, color="red", linestyle=":", lw=1.8, label=f"Baseline {bl:.2f}")
            ax.set_ylim(0, y_limit)
            ax.set_title(target)
            ax.set_xlabel("")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            ax.legend(fontsize=8)
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", fontsize=7)

        for j in range(len(targets), len(axes)):
            axes[j].axis("off")

        plt.suptitle(f"Model Comparison per Target ({split_label})", fontsize=16)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.savefig(os.path.join(out, f"detailed_model_comparison_{self.split}.png"), dpi=150)
        plt.close()
        print("Saved model comparison plots.")

    # ------------------------------------------------------------------
    # Target-set configuration comparison
    # ------------------------------------------------------------------

    def plot_multitarget_comparison(self, output_dir: Optional[str] = None) -> None:
        """Per-target RMSE across target-set configurations (single/reduced/all)."""
        self._ensure_loaded()
        df = self._split_df()
        if df.empty or df["Config Type"].nunique() <= 1:
            print("Multitarget comparison skipped: only one target-set configuration.")
            return
        out = self._out_dir(output_dir)

        config_order = [c for c in ["Single Target", "Reduced (8 Targets)", "All Targets"]
                        if c in df["Config Type"].unique()]
        config_order += [c for c in df["Config Type"].unique() if c not in config_order]
        palette = dict(zip(config_order, sns.color_palette("viridis", len(config_order))))

        for model_class, model_df in df.groupby("Model Class"):
            valid_targets = sorted(
                t for t, sub in model_df.groupby("Target")
                if sub["Config Type"].nunique() > 1
            )
            if not valid_targets:
                print(f"Skipping {model_class}: no targets with >1 config type.")
                continue

            y_limit = max(1.1, model_df["RMSE"].max() * 1.1)
            n_cols = 3
            n_rows = (len(valid_targets) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
            axes = axes.flatten()

            for i, target in enumerate(valid_targets):
                ax = axes[i]
                subset = model_df[model_df["Target"] == target]
                sns.barplot(data=subset, x="Config Type", y="RMSE", hue="Config Type",
                            order=[c for c in config_order if c in subset["Config Type"].unique()],
                            palette=palette, ax=ax, legend=False)
                ax.axhline(1.0, color="red", linestyle="--", lw=1.2, alpha=0.7)
                ax.set_ylim(0, y_limit)
                ax.set_title(target)
                ax.set_xlabel("")
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.2f", fontsize=7)

            for j in range(len(valid_targets), len(axes)):
                axes[j].axis("off")

            split_label = "Validation" if self.split == "val" else "Test"
            plt.suptitle(f"RMSE per Target-Set Configuration ({split_label}) — {model_class}",
                         fontsize=15)
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])
            path = os.path.join(out, f"multitarget_comparison_{model_class}_{self.split}.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"Saved {path}")

    # ------------------------------------------------------------------
    # CSV reports + convenience runner
    # ------------------------------------------------------------------

    def generate_aggregated_report(self, output_dir: Optional[str] = None) -> None:
        """Writes the tidy per-target table, run summaries, and fold summaries."""
        self._ensure_loaded()
        out = self._out_dir(output_dir)

        self._split_df().to_csv(os.path.join(out, f"{self.split}_aggregated_metrics.csv"), index=False)
        if not self.runs.empty:
            self.runs.sort_values(["Mode", "Augmentation Ratio"]).to_csv(
                os.path.join(out, "run_summaries.csv"), index=False)
        if not self.folds.empty:
            self.folds.to_csv(os.path.join(out, "fold_summaries.csv"), index=False)
        print(f"Saved CSV reports to {out}")

    def generate_all(self, output_dir: Optional[str] = None) -> None:
        """Runs every applicable comparison (methods skip themselves if N/A)."""
        self._ensure_loaded()
        self.plot_ratio_summary(output_dir)
        self.plot_fold_distributions(output_dir)
        self.plot_augmentation_ratio_comparison(output_dir)
        self.plot_model_comparison(output_dir)
        self.plot_multitarget_comparison(output_dir)
        self.generate_aggregated_report(output_dir)


if __name__ == "__main__":
    results_path = "/workspace/automatic_assessment/experiment_results/new_models"
    
    if not os.path.exists(results_path):
        print(f"Path {results_path} does not exist.")
        exit(1)
        
    comp = ExperimentComparison(results_path, split="val")  # val = primary metric
    comp.generate_all()
    # test-split view of the same runs (3-user normative spot check)
    comp_test = ExperimentComparison(results_path, split="test")
    comp_test.plot_ratio_summary()
    comp_test.plot_augmentation_ratio_comparison()
    comp_test.generate_aggregated_report()
