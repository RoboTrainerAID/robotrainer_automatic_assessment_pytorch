import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.metrics import mean_squared_error, r2_score


class VisualizationModule:
    """
    Generates plots from saved experiment artifacts (schema v3).

    Reads the TIDY predictions.csv (one row per split/user/target) and the
    nested metrics.yaml. Legacy result folders (pre-v3: wide predictions,
    flat metric keys) are converted on the fly so old runs stay plottable.

    Output layout (validation first):
        plots/val/parity_plots.png      plots/test/parity_plots.png
        plots/val/residuals.png         plots/test/residuals.png
        plots/val/user_performance.png  plots/test/user_performance.png
        plots/learning_curve.png
        plots/tuning_convergence.png
        plots/rmse_comparison.png
    """

    def __init__(self, experiment_path):
        self.path = experiment_path
        self.figures_path = os.path.join(experiment_path, "plots")
        os.makedirs(self.figures_path, exist_ok=True)

        self.metrics_raw = self._load_yaml("metrics.yaml")
        self.config_raw = self._load_yaml("config.yaml")
        self.is_v3 = bool(self.metrics_raw.get("schema_version", 0) >= 3)

        self.targets = self._load_targets()
        self.preds = self._load_predictions()  # tidy DataFrame (possibly empty)

    # ------------------------------------------------------------------
    # Loading (v3 + legacy fallback)
    # ------------------------------------------------------------------

    def _load_yaml(self, name):
        p = os.path.join(self.path, name)
        if os.path.exists(p):
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_targets(self):
        if self.is_v3:
            return list(self.config_raw.get("experiment", {}).get("targets", []))
        return list(self.config_raw.get("pipeline_config", {}).get("targets", []))

    def _load_predictions(self) -> pd.DataFrame:
        p = os.path.join(self.path, "predictions.csv")
        if not os.path.exists(p):
            return pd.DataFrame()
        df = pd.read_csv(p)

        if "target" in df.columns and "y_true" in df.columns:
            return df  # v3 tidy format

        # ---- legacy wide format -> tidy ----
        if "split" not in df.columns:
            df["split"] = "test"
        actual_cols = [c for c in df.columns if c.startswith("actual_target_")]
        rows = []
        for _, r in df.iterrows():
            for i in range(len(actual_cols)):
                target = self.targets[i] if i < len(self.targets) else f"target_{i}"
                rows.append({
                    "split": r["split"],
                    "fold": r.get("fold", -1),
                    "user_id": r["user_id"],
                    "target": target,
                    "y_true": r[f"actual_target_{i}"],
                    "y_pred": r[f"pred_target_{i}"],
                    "y_true_unscaled": np.nan,
                    "y_pred_unscaled": np.nan,
                })
        return pd.DataFrame(rows)

    def _metric(self, split: str, target: str, name: str):
        """Reads a per-target metric (scaled) from v3 nested or legacy flat metrics."""
        if self.is_v3:
            block = self.metrics_raw.get(split, {}).get("scaled", {})
            if target == "_mean":
                return block.get(f"{name}_mean")
            return block.get("per_target", {}).get(target, {}).get(name)
        # legacy flat keys use target indices
        if target == "_mean":
            return self.metrics_raw.get(f"{split}_{name}_mean")
        if target in self.targets:
            idx = self.targets.index(target)
            return self.metrics_raw.get(f"{split}_{name}_target_{idx}")
        return None

    def _splits(self):
        if self.preds.empty:
            return []
        splits = list(self.preds["split"].unique())
        return sorted(splits, key=lambda s: 0 if s == "val" else 1)

    def _split_dir(self, split: str) -> str:
        d = os.path.join(self.figures_path, split)
        os.makedirs(d, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Prediction-based plots (per split)
    # ------------------------------------------------------------------

    def generate_parity_plots(self):
        """Actual vs Predicted per target (scaled), one figure per split."""
        for split in self._splits():
            df = self.preds[self.preds["split"] == split]
            targets = [t for t in self.targets if t in df["target"].unique()] or sorted(df["target"].unique())
            if df.empty or not targets:
                continue

            fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5), squeeze=False)

            for i, target in enumerate(targets):
                sub = df[df["target"] == target]
                ax = axes[0, i]

                sns.regplot(x="y_true", y="y_pred", data=sub,
                            ax=ax, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})

                min_val = min(sub["y_true"].min(), sub["y_pred"].min())
                max_val = max(sub["y_true"].max(), sub["y_pred"].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)

                r2 = self._metric(split, target, "r2")
                rmse = self._metric(split, target, "rmse")
                if r2 is None:
                    r2 = r2_score(sub["y_true"], sub["y_pred"])
                if rmse is None:
                    rmse = float(np.sqrt(mean_squared_error(sub["y_true"], sub["y_pred"])))

                ax.set_title(f"{target} ({split})\n$R^2$: {r2:.2f} | RMSE: {rmse:.2f}")
                ax.set_xlabel("Ground Truth (Scaled)")
                ax.set_ylabel("Prediction (Scaled)")

            plt.tight_layout()
            plt.savefig(os.path.join(self._split_dir(split), "parity_plots.png"))
            plt.close()

    def generate_error_distribution(self):
        """Residual histograms per target and split (scaled)."""
        for split in self._splits():
            df = self.preds[self.preds["split"] == split]
            targets = [t for t in self.targets if t in df["target"].unique()] or sorted(df["target"].unique())
            if df.empty or not targets:
                continue

            fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5), squeeze=False)

            for i, target in enumerate(targets):
                sub = df[df["target"] == target]
                residuals = sub["y_true"] - sub["y_pred"]
                ax = axes[0, i]

                sns.histplot(residuals, kde=True, ax=ax, color='purple')
                ax.axvline(0, color='black', linestyle='--')
                ax.set_title(f"Residuals: {target} ({split})")
                ax.set_xlabel("Error (Actual - Predicted)")

            plt.tight_layout()
            plt.savefig(os.path.join(self._split_dir(split), "residuals.png"))
            plt.close()

    def generate_user_performance_report(self):
        """Per-user RMSE per target (scaled)."""
        for split in self._splits():
            df = self.preds[self.preds["split"] == split]
            if df.empty:
                continue

            err = df.assign(sq=(df["y_true"] - df["y_pred"]) ** 2)
            per_user = err.groupby(["user_id", "target"])["sq"].mean().pow(0.5).reset_index(name="rmse")

            plt.figure(figsize=(10, 6))
            sns.barplot(x="user_id", y="rmse", hue="target", data=per_user)
            title = "Per-User Error (LOGO CV Validation)" if split == "val" else f"Per-User Error ({split})"
            plt.title(title)
            plt.ylabel("RMSE (Scaled)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self._split_dir(split), "user_performance.png"))
            plt.close()

    # ------------------------------------------------------------------
    # Split-independent plots
    # ------------------------------------------------------------------

    def generate_learning_curve(self):
        lc_path = os.path.join(self.path, "learning_curve.csv")
        if not os.path.exists(lc_path):
            print("No learning curve data found.")
            return

        df = pd.read_csv(lc_path)

        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['train_loss'], label='Training Loss', marker='o', markersize=3)

        if 'val_loss' in df.columns and df['val_loss'].notna().any():
            plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='o', markersize=3)

        plt.title("Learning Curve (Fold 0)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Huber)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, "learning_curve.png"))
        plt.close()

    def generate_tuning_plot(self):
        trials_file = os.path.join(self.path, "tuning_trials.csv")
        if not os.path.exists(trials_file):
            return

        df = pd.read_csv(trials_file)
        if 'value' not in df.columns or 'number' not in df.columns:
            return

        df = df.sort_values('number')
        scores = df['value'].values
        best_so_far = np.minimum.accumulate(scores)

        plt.figure(figsize=(10, 6))
        plt.scatter(df['number'], scores, c='gray', alpha=0.5, label='Individual Trial')
        plt.plot(df['number'], best_so_far, c='red', lw=2, label='Best So Far')

        best_val = float(np.min(scores))
        plt.axhline(best_val, color='green', linestyle=':', label=f'Best: {best_val:.4f}')

        n_trials = len(df)
        if n_trials >= 20:
            exploration_end = int(n_trials * 0.2)
            plt.axvline(exploration_end, color='blue', linestyle='--', alpha=0.5)
            plt.text(exploration_end, max(scores), " Exploration End ", rotation=90, verticalalignment='top')

        plt.title("Hyperparameter Tuning Convergence")
        plt.xlabel("Trial Number")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, "tuning_convergence.png"))
        plt.close()
        print(f"Saved tuning convergence plot to {self.figures_path}/tuning_convergence.png")

    def generate_rmse_comparison(self):
        """Grouped bars: Validation (and Test) RMSE per target, baseline dots."""
        targets = self.targets
        if not targets:
            print("No targets found for RMSE comparison.")
            return

        data = []
        splits_present = []
        for split, label in (("val", "Validation"), ("test", "Test")):
            for target in targets:
                rmse = self._metric(split, target, "rmse")
                if rmse is not None:
                    data.append({"Target": target, "RMSE": float(rmse), "Split": label})
                    if label not in splits_present:
                        splits_present.append(label)

        if not data:
            print("No per-target RMSE metrics found for plotting.")
            return

        df_plot = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(max(10, len(targets) * 2.5), 6))
        sns.set_theme(style="whitegrid")

        barplot = sns.barplot(
            data=df_plot, x='Target', y='RMSE',
            hue='Split', hue_order=splits_present, palette="muted", ax=ax
        )

        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
                   label='Scaled Mean Baseline (≈1.0)')

        # Baseline dots aligned to the rendered bar positions
        patches = barplot.patches
        n_targets = len(targets)
        colors = {"Validation": "#1f77b4", "Test": "#ff7f0e"}
        for s_idx, (split, label) in enumerate([(s, l) for s, l in (("val", "Validation"), ("test", "Test"))
                                                if l in splits_present]):
            xs, ys = [], []
            for i, target in enumerate(targets):
                bl = self._metric(split, target, "baseline_rmse")
                patch_idx = s_idx * n_targets + i
                if bl is not None and patch_idx < len(patches):
                    p = patches[patch_idx]
                    xs.append(p.get_x() + p.get_width() / 2)
                    ys.append(float(bl))
            if xs:
                ax.scatter(xs, ys, color=colors[label], marker='D', s=70, zorder=5,
                           edgecolors='black', linewidths=0.8, label=f'{label} Baseline (Dummy)')

        ax.set_title("RMSE per Target: Validation" + (" vs Test" if len(splits_present) > 1 else ""),
                     fontsize=16)
        ax.set_ylabel("RMSE (Scaled)", fontsize=12)
        ax.set_xlabel("")
        plt.xticks(rotation=30, ha='right')
        ax.legend(title="Legend", loc='upper right')
        plt.tight_layout()

        save_path = os.path.join(self.figures_path, "rmse_comparison.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

    # ------------------------------------------------------------------

    def generate_all_plots(self):
        """Generates all available plots and reports."""
        self.generate_parity_plots()
        self.generate_error_distribution()
        self.generate_learning_curve()
        self.generate_user_performance_report()
        self.generate_tuning_plot()
        self.generate_rmse_comparison()


if __name__ == "__main__":
    experiment_path = "/workspace/automatic_assessment/experiment_results/20260706_132402_BASE_BaselineFLAT"
    viz = VisualizationModule(experiment_path)
    viz.generate_all_plots()
