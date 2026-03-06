"""
Generate a LaTeX comparison table from paper_comparison.csv.

Produces a booktabs table with one row per model, sorted by validation RMSE,
with per-target test RMSE columns and bold highlights for column-wise bests.
"""

import csv
import os
import statistics
from typing import List, Optional, Tuple


# Target column prefixes and their short LaTeX-safe display names
TARGETS: List[Tuple[str, str]] = [
    ("robotrainer_front", "Max F Front"),
    ("robotrainer_left", "Max F Left"),
    ("robotrainer_right", "Max F Right"),
    ("hand_grip_left", "Hand Grip Left"),
    ("hand_grip_right", "Hand Grip Right"),
    ("jump_and_reach", r"Jump \&Reach"),
    ("figure_8_walk", "Figure-8 Walk"),
    ("ruler_drop_test", "Ruler Drop"),
]


def _fmt(val: float, best: float, prec: int = 2) -> str:
    """Format a number; bold if it equals the column best."""
    s = f"{val:.{prec}f}"
    if abs(val - best) < 1e-6:
        return r"\textbf{" + s + "}"
    return s


def generate_latex_table(
    csv_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Read *csv_path* and write a LaTeX table to *output_path*.

    Args:
        csv_path: Path to paper_comparison.csv.
        output_path: Where to save the .tex file.
            Defaults to ``model_comparison_table.tex`` next to the CSV.

    Returns:
        The path the file was written to.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Compute mean baseline test per row
    bl_keys = [f"{t[0]}_baseline_rmse_test" for t in TARGETS]
    for r in rows:
        r["mean_baseline_test"] = statistics.mean(float(r[k]) for k in bl_keys)

    # Sort by validation RMSE (ascending)
    rows.sort(key=lambda r: float(r["mean_rmse_val"]))

    # Column-wise bests for bolding
    best = {
        "mean_rmse_val": min(float(r["mean_rmse_val"]) for r in rows),
        "mean_rmse_test": min(float(r["mean_rmse_test"]) for r in rows),
    }
    for prefix, _ in TARGETS:
        best[prefix] = min(float(r[f"{prefix}_rmse_test"]) for r in rows)

    # Header
    hdr_targets = " & ".join(t[1] for t in TARGETS)
    header = (
        r"Model & $\overline{\mathrm{RMSE}}_{\mathrm{val}}$"
        r" & $\overline{\mathrm{RMSE}}_{\mathrm{test}}$"
        r" & $\overline{\mathrm{BL}}_{\mathrm{test}}$"
        f" & {hdr_targets} \\\\"
    )

    ncols = 4 + len(TARGETS)
    col_spec = "l" + "r" * (ncols - 1)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Test RMSE per target for all models, sorted by validation RMSE.}",
        r"\label{tab:model_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for r in rows:
        model = r["model"].replace("_", r"\_")
        cells = [model]
        cells.append(_fmt(float(r["mean_rmse_val"]), best["mean_rmse_val"]))
        cells.append(_fmt(float(r["mean_rmse_test"]), best["mean_rmse_test"]))
        cells.append(f"{r['mean_baseline_test']:.2f}")
        for prefix, _ in TARGETS:
            cells.append(_fmt(float(r[f"{prefix}_rmse_test"]), best[prefix]))
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        r"\end{table}",
    ]

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(csv_path), "model_comparison_table.tex"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[PaperTable] Saved {len(rows)} rows → {output_path}")
    return output_path


# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a LaTeX comparison table from paper_comparison.csv."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="/workspace/automatic_assessment/experiment_results/final_best/paper_comparison.csv",
        help="Path to the paper_comparison.csv file.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output .tex path (default: model_comparison_table.tex next to CSV).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        print(f"CSV not found: {args.csv_path}")
        raise SystemExit(1)

    generate_latex_table(args.csv_path, args.output)
