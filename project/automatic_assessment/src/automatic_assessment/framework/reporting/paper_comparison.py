"""
Paper-ready experiment comparison table generator.

Scans experiment result folders, reads config.yaml and metrics.yaml from each,
and produces a single CSV where each row is one experiment and columns contain
per-target scores for both the validation and test splits.

Column layout
─────────────
  folder, model, augmentation_ratio, note,
  mean_rmse_val, mean_rmse_test,
  <target_name>_rmse_val,  <target_name>_r2_val,  <target_name>_baseline_rmse_val,
  <target_name>_rmse_test, <target_name>_r2_test, <target_name>_baseline_rmse_test,
  ... (repeated for every target)
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


class PaperComparison:
    """Aggregate experiment metrics into a single paper-ready CSV."""

    SPLITS = ("val", "test")

    def __init__(self, experiments_root_path: str):
        """
        Args:
            experiments_root_path: Directory that contains the individual
                experiment folders (each with config.yaml + metrics.yaml).
        """
        self.root_path = experiments_root_path

    # ------------------------------------------------------------------ #
    #  helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_model_name(folder_name: str) -> str:
        """Extract the model name from *YYYYMMDD_HHMMSS_ModelName* folders."""
        parts = folder_name.split("_")
        if len(parts) >= 3:
            return "_".join(parts[2:])
        return "UnknownModel"

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Convert a value to float, returning NaN on failure."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    # ------------------------------------------------------------------ #
    #  core                                                               #
    # ------------------------------------------------------------------ #
    def load_data(self) -> pd.DataFrame:
        """
        Walk every subfolder of *root_path*, read its config + metrics,
        and return a DataFrame with one row per experiment.
        """
        rows: List[Dict[str, Any]] = []

        for folder_name in sorted(os.listdir(self.root_path)):
            folder_path = os.path.join(self.root_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            config_path = os.path.join(folder_path, "config.yaml")
            metrics_path = os.path.join(folder_path, "metrics.yaml")

            if not (os.path.exists(config_path) and os.path.exists(metrics_path)):
                continue

            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                with open(metrics_path, "r") as f:
                    metrics = yaml.safe_load(f)
            except Exception as exc:
                print(f"[PaperComparison] Error reading {folder_name}: {exc}")
                continue

            # --- meta info ------------------------------------------------
            model_name = self._parse_model_name(folder_name)

            pipeline_cfg = config.get("pipeline_config", {})
            target_names: List[str] = pipeline_cfg.get("targets", [])
            augmentation_ratio = pipeline_cfg.get("augmentation_ratio", None)
            note = pipeline_cfg.get("note", "")

            if not target_names:
                print(f"[PaperComparison] Skipping {folder_name}: no targets in config.")
                continue

            row: Dict[str, Any] = {
                "folder": folder_name,
                "model": model_name,
                "augmentation_ratio": augmentation_ratio,
                "note": note,
                "n_targets": len(target_names),
            }

            # --- aggregate mean RMSE per split ----------------------------
            for split in self.SPLITS:
                row[f"mean_rmse_{split}"] = self._safe_float(
                    metrics.get(f"{split}_rmse_mean")
                )

            # --- per-target metrics ---------------------------------------
            for idx, target_name in enumerate(target_names):
                # Sanitise the target name so it works as a column suffix
                t = self._sanitize_column_name(target_name)

                for split in self.SPLITS:
                    rmse_key = f"{split}_rmse_target_{idx}"
                    r2_key = f"{split}_r2_target_{idx}"
                    baseline_rmse_key = f"{split}_baseline_rmse_target_{idx}"

                    row[f"{t}_rmse_{split}"] = self._safe_float(
                        metrics.get(rmse_key)
                    )
                    row[f"{t}_r2_{split}"] = self._safe_float(
                        metrics.get(r2_key)
                    )
                    row[f"{t}_baseline_rmse_{split}"] = self._safe_float(
                        metrics.get(baseline_rmse_key)
                    )

            rows.append(row)

        df = pd.DataFrame(rows)

        if not df.empty:
            # Ensure a deterministic, readable column order
            df = self._reorder_columns(df, target_names)

        return df

    # ------------------------------------------------------------------ #
    #  column helpers                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sanitize_column_name(name: str) -> str:
        """Lower-case, replace spaces & special chars with underscores."""
        return name.lower().replace(" ", "_").replace("&", "and")

    def _reorder_columns(
        self, df: pd.DataFrame, target_names: List[str]
    ) -> pd.DataFrame:
        """
        Put columns in a logical order:
        meta → mean scores → per-target (val then test for each target).
        """
        meta_cols = ["folder", "model", "augmentation_ratio", "note", "n_targets"]
        mean_cols = [f"mean_rmse_{s}" for s in self.SPLITS]

        target_cols: List[str] = []
        for target_name in target_names:
            t = self._sanitize_column_name(target_name)
            for split in self.SPLITS:
                target_cols.append(f"{t}_rmse_{split}")
                target_cols.append(f"{t}_r2_{split}")
                target_cols.append(f"{t}_baseline_rmse_{split}")

        ordered = [c for c in meta_cols + mean_cols + target_cols if c in df.columns]
        # Append any remaining columns that we might have missed
        remaining = [c for c in df.columns if c not in ordered]
        return df[ordered + remaining]

    # ------------------------------------------------------------------ #
    #  I/O                                                                #
    # ------------------------------------------------------------------ #
    def generate_csv(self, output_path: Optional[str] = None) -> str:
        """
        Load data and write to CSV.

        Args:
            output_path: Full path for the CSV file.  Defaults to
                ``<root_path>/paper_comparison.csv``.

        Returns:
            The path the CSV was written to.
        """
        df = self.load_data()

        if output_path is None:
            output_path = os.path.join(self.root_path, "paper_comparison.csv")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[PaperComparison] Saved {len(df)} rows → {output_path}")
        return output_path


# ---------------------------------------------------------------------- #
#  CLI convenience                                                       #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a paper-ready comparison CSV from experiment results."
    )
    parser.add_argument(
        "experiments_dir",
        nargs="?",
        default="/workspace/automatic_assessment/experiment_results/final_best",
        help="Root directory containing experiment folders.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: <experiments_dir>/comparison/paper_comparison.csv).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.experiments_dir):
        print(f"Directory not found: {args.experiments_dir}")
        raise SystemExit(1)

    comp = PaperComparison(args.experiments_dir)
    comp.generate_csv(args.output)
