"""
Standalone script to analyze the generated features CSV.
Checks for NaNs, feature coverage, and dataset statistics.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import automatic_assessment.dataset.config as config
import os
from typing import Optional
from scipy.stats import pearsonr

def load_csv_data(file_path: str = None) -> pd.DataFrame:
    """Generic function to load a CSV file."""

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()

    print(f"Loading data from: {file_path}")
    return pd.read_csv(file_path)

def compare_target_csvs(df_old: pd.DataFrame, df_new: pd.DataFrame, key_col: str = 'Code') -> None:
    """
    Compares two target CSVs ensuring they contain the same data.
    """
    print("-" * 40)
    print("Comparing Target CSVs...")
    
    if df_old.empty or df_new.empty:
        print("One of the dataframes is empty, skipping comparison.")
        return

    # Normalize key column
    if key_col not in df_old.columns and 'user' in df_old.columns: key_col_old = 'user'
    else: key_col_old = key_col
        
    if key_col not in df_new.columns and 'user' in df_new.columns: key_col_new = 'user'
    else: key_col_new = key_col

    if key_col_old not in df_old.columns or key_col_new not in df_new.columns:
        print(f"Error: Key column '{key_col}' not found for comparison.")
        return

    # Set index for alignment
    df1 = df_old.set_index(key_col_old)
    df2 = df_new.set_index(key_col_new)

    # 1. Check structural differences
    users_1 = set(df1.index)
    users_2 = set(df2.index)
    
    missing_in_new = users_1 - users_2
    added_in_new = users_2 - users_1
    
    if missing_in_new: print(f"  Users missing in new CSV: {missing_in_new}")
    if added_in_new: print(f"  Users added in new CSV: {added_in_new}")
    
    common_users = users_1.intersection(users_2)
    common_cols = set(df1.columns).intersection(set(df2.columns))
    
    print(f"  Comparing {len(common_users)} users and {len(common_cols)} columns.")

    # 2. Check value differences
    diff_count = 0
    for user in common_users:
        for col in common_cols:
            val1 = df1.loc[user, col]
            val2 = df2.loc[user, col]
            
            # Handle NaN equality
            if pd.isna(val1) and pd.isna(val2):
                continue
            
            # Handle numeric differences with tolerance
            is_diff = False
            if pd.api.types.is_numeric_dtype(type(val1)) and pd.api.types.is_numeric_dtype(type(val2)):
                if not np.isclose(val1, val2, equal_nan=True):
                    is_diff = True
            else:
                if val1 != val2:
                    is_diff = True
            
            if is_diff:
                diff_count += 1
                print(f"  Difference for User {user}, Col '{col}': Old={val1} | New={val2}")

    if diff_count == 0:
        print("  SUCCESS: Common data matches exactly.")
    else:
        print(f"  Found {diff_count} discrepancies.")
    print("-" * 40)

def print_feature_summary(df: pd.DataFrame) -> None:
    """Prints a summary analysis of the feature dataframe."""
    if df.empty:
        return

    # 1. basic shape
    num_rows, num_cols = df.shape
    feature_cols = [c for c in df.columns if c not in ['user', 'path']]
    
    print("-" * 40)
    print(f"Dataset Shape: {num_rows} paths x {num_cols} columns")
    print(f"Feature Columns: {len(feature_cols)}")
    
    # 2. NaN Analysis
    total_cells = num_rows * len(feature_cols)
    total_nans = df[feature_cols].isna().sum().sum()
    nan_perc = (total_nans / total_cells) * 100 if total_cells > 0 else 0
    
    print("-" * 40)
    print(f"Missing Values (NaNs): {total_nans} ({nan_perc:.2f}%)")
    
    # Features with most NaNs
    nan_per_col = df[feature_cols].isna().sum()
    nan_cols = nan_per_col[nan_per_col > 0].sort_values(ascending=False)
    if not nan_cols.empty:
        print("Top 5 features with missing values:")
        for name, count in nan_cols.head(5).items():
            print(f"  - {name}: {count} NaNs")
    else:
        print("  No missing values found in feature columns.")

    # 3. Features per Path (Non-null count)
    # How many valid features does each path have?
    valid_feats_per_row = df[feature_cols].notna().sum(axis=1)
    min_feats = valid_feats_per_row.min()
    max_feats = valid_feats_per_row.max()
    
    print("-" * 40)
    print("Valid Features per Path:")
    print(f"  Min: {min_feats}")
    print(f"  Max: {max_feats}")
    
    if min_feats == 0:
        empty_paths = df[valid_feats_per_row == 0]['path'].tolist()
        print(f"  Warning: {len(empty_paths)} paths have 0 valid features.")

    # 4. Paths per User
    if 'user' in df.columns:
        paths_per_user = df.groupby('user')['path'].count()
        print("-" * 40)
        print("Paths per User:")
        print(f"  Total Users: {len(paths_per_user)}")
        print(f"  Min paths: {paths_per_user.min()}")
        print(f"  Max paths: {paths_per_user.max()}")
    else:
        print("Warning: 'user' column not found, skipping user stats.")

    print("-" * 40)

def analyze_correlations(features_df: pd.DataFrame, targets_df: pd.DataFrame, threshold: float = 0.3) -> None:
    """
    Computes and plots the correlation matrix between features and regression targets.
    Selects the specific path (trial) index that yields the highest correlation for each feature.
    
    Args:
        threshold: Minimum absolute correlation required with at least one target 
                   to include the feature in the plot.
    """
    if features_df.empty or targets_df.empty:
        return

    print("Running Correlation Analysis...")

    # Identify common user column
    target_user_col = None
    if 'user' in targets_df.columns:
        target_user_col = 'user'
    elif 'Code' in targets_df.columns:
        target_user_col = 'Code'
    
    if not target_user_col:
        print("Error: Could not identify user column in targets CSV (expected 'user' or 'Code').")
        return

    # Add 'trial_index' to features based on path order per user
    # Assuming standard sorting of paths implies temporal order
    features_sorted = features_df.sort_values(by=['user', 'path'])
    features_sorted['trial_index'] = features_sorted.groupby('user').cumcount() + 1
    max_trials = features_sorted['trial_index'].max()
    print(f"  Detected up to {max_trials} paths (trials) per user.")

    feature_cols = [c for c in features_df.columns if c not in ['user', 'path', 'class', 'trial_index']]
    
    # Identify target columns
    target_cols = [c for c in targets_df.columns if c not in [target_user_col, 'user', 'path']]
    target_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(targets_df[c])]

    print(f"  Correlating {len(feature_cols)} features with targets: {target_cols}")

    # Storage for best correlations: Key = (feature, target), Value = (abs_r, r, p, trial_idx)
    best_corrs = {}

    # Iterate over each trial index (1, 2, 3...) separately
    for t_idx in range(1, max_trials + 1):
        # Filter features for this trial
        trial_data = features_sorted[features_sorted['trial_index'] == t_idx]
        
        # Merge with targets
        merged = pd.merge(trial_data, targets_df, left_on='user', right_on=target_user_col, how='inner')
        
        if len(merged) < 3:
            continue

        for f_col in feature_cols:
            if merged[f_col].std() == 0: continue
            
            for t_col in target_cols:
                # Drop NaNs
                tmp = merged[[f_col, t_col]].dropna()
                if len(tmp) < 3: continue
                
                r, p = pearsonr(tmp[f_col], tmp[t_col])
                
                key = (f_col, t_col)
                current_abs = abs(r)
                
                # Update if better or new
                if key not in best_corrs or current_abs > best_corrs[key][0]:
                    best_corrs[key] = (current_abs, r, p, t_idx)

    # --- Construct Final Matrix ---
    # Logic: For each feature, find the single Path ID that yields the highest absolute correlation 
    # with ANY target variable. This path then represents the feature in the heatmap (complete row).
    
    all_rows = [] # Store all results before thresholding
    
    for f_col in feature_cols:
        
        # 1. Gather max correlation for this feature across all paths
        # We store (max_r_abs, path_index) tuples
        f_candidates = []
        
        for t_idx in range(1, max_trials + 1):
             trial_data = features_sorted[features_sorted['trial_index'] == t_idx]
             
             # Need enough data points
             merged = pd.merge(trial_data, targets_df, left_on='user', right_on=target_user_col, how='inner')
             if len(merged) < 3: continue
             if merged[f_col].std() == 0: continue
             
             # Find the highest correlation this specific path achieves with ANY target
             path_max_r = 0.0
             for t_col in target_cols:
                 tmp = merged[[f_col, t_col]].dropna()
                 if len(tmp) >= 3:
                     r, _ = pearsonr(tmp[f_col], tmp[t_col])
                     if abs(r) > path_max_r:
                         path_max_r = abs(r)
             
             f_candidates.append((path_max_r, t_idx))
        
        if not f_candidates: continue
        
        # Pick the path that had the single highest correlation peak
        f_candidates.sort(key=lambda x: x[0], reverse=True)
        best_r_global, best_path_idx = f_candidates[0]
        
        # REMOVED: if best_r_global < threshold: continue
        # We now keep all, and filter later for specific plots if needed
        
        # 2. Compute the full row for this Best Path
        row_dict = {}
        annot_dict = {}
        
        # Re-fetch data for this specific best path to fill the row
        trial_data = features_sorted[features_sorted['trial_index'] == best_path_idx]
        merged = pd.merge(trial_data, targets_df, left_on='user', right_on=target_user_col, how='inner')
        
        for t_col in target_cols:
            tmp = merged[[f_col, t_col]].dropna()
            if len(tmp) < 3 or merged[f_col].std() == 0:
                row_dict[t_col] = 0.0
                annot_dict[t_col] = "" # No data
            else:
                r, p = pearsonr(tmp[f_col], tmp[t_col])
                row_dict[t_col] = r
                
                stars = ""
                if p < 0.001: stars = "***"
                elif p < 0.01: stars = "**"
                elif p < 0.05: stars = "*"
                annot_dict[t_col] = f"{abs(r):.2f}\n{stars}" if stars else f"{abs(r):.2f}"

        # Feature label marks which path was selected
        row_label = f"{f_col} (Path {best_path_idx})"
        
        all_rows.append({
            "label": row_label,
            "values": row_dict,
            "annots": annot_dict,
            "max_abs": best_r_global # for sorting the plot
        })

    if not all_rows:
        print(f"No correlations found.")
        return

    # --- PLOT 1: FILTERED BY THRESHOLD ---
    final_rows = [row for row in all_rows if row['max_abs'] >= threshold]

    if final_rows:
        # Convert to DataFrames
        final_rows.sort(key=lambda x: x['max_abs'], reverse=True)
        
        df_plot_data = pd.DataFrame([x['values'] for x in final_rows], index=[x['label'] for x in final_rows])
        df_plot_annot = pd.DataFrame([x['annots'] for x in final_rows], index=[x['label'] for x in final_rows])

        print(f"  Plotting {len(df_plot_data)} features (Best Path Selection) > {threshold}.")

        # --- Plotting ---
        width = max(10, len(target_cols) * 1.2)
        height = max(8, len(df_plot_data) * 0.4 + 2) 
        plt.figure(figsize=(width, height)) 
        
        sns.heatmap(
            df_plot_data.abs(),   # Colors based on ABSOLUTE correlation
            annot=df_plot_annot.values, 
            fmt="",              
            cmap="Reds",         
            vmin=0, 
            vmax=1,
            linewidths=0.5,
            annot_kws={"size": 9, "va": "center"} 
        )
        plt.title(f"Best-Path Feature Correlation. Threshold > {threshold} \n * p<0.05, ** p<0.01, *** p<0.001")
        plt.xlabel("Targets")
        plt.ylabel("Extracted Features")
        plt.xticks(rotation=45, ha='right') 
        plt.tight_layout()
        
        output_dir = os.path.dirname("/workspace/automatic_assessment/figures/dataset/feature_correlation_matrix.png")
        os.makedirs(output_dir, exist_ok=True)

        output_plot = f"/workspace/automatic_assessment/figures/dataset/feature_correlation_matrix_best_path.png"
        plt.savefig(output_plot)
        print(f"Correlation matrix saved to: {output_plot}")
    else:
        print(f"  No features satisfied the threshold {threshold} for the first plot.")

    # --- SECOND PLOT: Best Feature PER Target (NO THRESHOLD) ---
    print("\n  Generating 'Best Feature per Target' plot (No Threshold)...")
    
    # We want to find the feature (row) from ALL rows that maximizes correlation for each specific target col
    best_rows_indices = set()
    
    # Identify the best feature row for each target using all_rows (unfiltered)
    for t_col in target_cols:
        best_r = -1.0
        best_row_idx = -1
        
        for i, row_data in enumerate(all_rows):
            # value for this target
            val = abs(row_data['values'].get(t_col, 0.0))
            if val > best_r:
                best_r = val
                best_row_idx = i
        
        if best_row_idx != -1:
            best_rows_indices.add(best_row_idx)
            # Print info
            print(f"    Target '{t_col}': Best Feature is '{all_rows[best_row_idx]['label']}' (R={best_r:.2f})")

    if not best_rows_indices:
        print("    No features found for best-per-target plot.")
        return

    # Filter rows from all_rows
    best_target_rows = [all_rows[i] for i in best_rows_indices]
    
    # Sort by overall max abs correlation for visual hierarchy
    best_target_rows.sort(key=lambda x: x['max_abs'], reverse=True)

    # Convert to DataFrames
    df_best_data = pd.DataFrame([x['values'] for x in best_target_rows], index=[x['label'] for x in best_target_rows])
    df_best_annot = pd.DataFrame([x['annots'] for x in best_target_rows], index=[x['label'] for x in best_target_rows])

    # Plotting
    width = max(10, len(target_cols) * 1.2)
    height = max(6, len(df_best_data) * 0.5 + 2) 
    plt.figure(figsize=(width, height)) 
    
    sns.heatmap(
        df_best_data.abs(),   
        annot=df_best_annot.values, 
        fmt="",              
        cmap="Reds",         
        vmin=0, 
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": 10, "va": "center", "weight": "bold"} 
    )
    plt.title(f"Top Features (Best per Target)\n * p<0.05, ** p<0.01, *** p<0.001")
    plt.xlabel("Targets")
    plt.ylabel("Extracted Features")
    plt.xticks(rotation=45, ha='right') 
    plt.tight_layout()
    
    output_plot_best = f"/workspace/automatic_assessment/figures/dataset/feature_correlation_matrix_best_per_target.png"
    plt.savefig(output_plot_best)
    print(f"Best-per-target matrix saved to: {output_plot_best}")

def analyze_feature_retention(features_df: pd.DataFrame, targets_df: pd.DataFrame, thresholds: list = np.arange(0.4, 0.8, 0.01)) -> None:
    """
    Analyzes how many features remain after filtering by different correlation thresholds.
    Prints a report.
    """
    if features_df.empty or targets_df.empty:
        return

    print("\n" + "="*40)
    print("Running Feature Retention Analysis...")
    
    # Identify common user column
    target_user_col = None
    if 'user' in targets_df.columns:
        target_user_col = 'user'
    elif 'Code' in targets_df.columns:
        target_user_col = 'Code'
    
    if not target_user_col:
        print("Error: Could not identify user column in targets CSV (expected 'user' or 'Code').")
        return

    feature_cols = [c for c in features_df.columns if c not in ['user', 'path', 'class', 'trial_index']]
    target_cols = [c for c in targets_df.columns if c not in [target_user_col, 'user', 'path']]
    target_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(targets_df[c])]

    print(f"  Total Initial Features: {len(feature_cols)}")
    print(f"  Targets considered: {len(target_cols)}")

    # Pre-calculate Max Correlation for each feature
    feature_max_corrs = {f: 0.0 for f in feature_cols}
    
    # Merge data once if possible, or per path (trial) like before
    features_sorted = features_df.sort_values(by=['user', 'path'])
    # Assign path index
    if 'trial_index' not in features_sorted.columns:
        features_sorted['trial_index'] = features_sorted.groupby('user').cumcount() + 1
    max_trials = features_sorted['trial_index'].max()

    # Iterate trials
    # We take the max correlation a feature achieves in ANY trial across ANY target.
    # This is "best-case" relevance.
    
    for t_idx in range(1, max_trials + 1):
        trial_data = features_sorted[features_sorted['trial_index'] == t_idx]
        merged = pd.merge(trial_data, targets_df, left_on='user', right_on=target_user_col, how='inner')
        
        if len(merged) < 3: continue

        for f_col in feature_cols:
            if merged[f_col].std() == 0: continue
            
            for t_col in target_cols:
                tmp = merged[[f_col, t_col]].dropna()
                if len(tmp) < 3: continue
                
                r, _ = pearsonr(tmp[f_col], tmp[t_col])
                current_abs = abs(r)
                
                if current_abs > feature_max_corrs[f_col]:
                    feature_max_corrs[f_col] = current_abs

    # Determine counts for thresholds
    print("\n--- Feature Retention Report ---")
    print(f"{'Threshold':<10} | {'Retained Features':<20} | {'% of Total':<10}")
    print("-" * 45)
    
    results = []

    for thresh in thresholds:
        count = sum(1 for val in feature_max_corrs.values() if val >= thresh)
        perc = (count / len(feature_cols)) * 100
        print(f"{thresh:<10.2f} | {count:<20} | {perc:<10.2f}%")
        results.append((thresh, count))
    
    print("="*40 + "\n")
    
    # Optional: Plot retention curve
    plt.figure(figsize=(8, 5))
    x_vals = [r[0] for r in results]
    y_vals = [r[1] for r in results]
    plt.plot(x_vals, y_vals, marker='o', linestyle='-')
    plt.title("Feature Retention vs. Correlation Threshold")
    plt.xlabel("Correlation Threshold (Absolute)")
    plt.ylabel("Number of Retained Features")
    plt.grid(True)
    
    output_plot = "/workspace/automatic_assessment/figures/dataset/feature_retention_curve.png"
    plt.savefig(output_plot)
    print(f"Retention curve saved to: {output_plot}")

def main():
    # 1. Load Features
    features = load_csv_data(config.CSV_OUTPUT_PATH) # Uses config default
    print_feature_summary(features)
    
    # 2. Comparison of Targets
    # targets_old = load_csv_data("/data/raw/motoric_test_old.csv")
    targets = load_csv_data("/data/raw/motoric_test.csv")
    # compare_target_csvs(targets_old, targets)
    
    # 3. Correlation Plot
    # analyze_correlations(features, targets, threshold=0.5)

    # 4. Feature Retention Analysis
    analyze_feature_retention(features, targets)

if __name__ == "__main__":
    main()
