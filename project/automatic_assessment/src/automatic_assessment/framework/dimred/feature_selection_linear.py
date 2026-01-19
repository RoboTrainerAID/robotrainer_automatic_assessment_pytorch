# ============================================================================
# IDEAL FEATURE SELECTION PIPELINE (REORDERED)
# 1) Raw correlation screening (model-free)
# 2) Time-series aggregation
# 3) LOSO permutation importance (model-aware)
# 4) Target pruning (top-K)
# 5) Feature pruning (time-series + path + user)
# 6) Export pruned datasets
# ============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# ---------------- CONFIG ----------------
PRUNE_CORR_RATIO = 0.5          # remove bottom 50% raw TS channels
TOP_K_TARGETS = 8
TOP_K_FEATURES = 40             # final model input size
RIDGE_ALPHA = 1.0
N_PERM = 5
OUT_DIR = "/data/dataset_conv@1s/pruned_linear"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- LOSO SRMSE ----------------

def loso_scaled_rmse(X, y, groups):
    logo = LeaveOneGroupOut()
    rmses = []

    for tr, te in logo.split(X, y, groups):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X.iloc[tr])
        Xte = scaler.transform(X.iloc[te])

        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(Xtr, y.iloc[tr])
        pred = model.predict(Xte)

        rmse = np.sqrt(mean_squared_error(y.iloc[te], pred))
        rmses.append(rmse / (y.iloc[tr].std() + 1e-8))

    return np.mean(rmses)


# ---------------- STEP 1: RAW CORRELATION ----------------

def raw_ts_correlation(ts_df, target_df, target_name):
    feats = [c for c in ts_df.columns if c not in ["user", "path", "time"]]
    ts_user_mean = ts_df.groupby("user")[feats].mean().reset_index()
    merged = ts_user_mean.merge(target_df[["user", target_name]], on="user")
    scores = {}
    for f in feats:
        rho, _ = spearmanr(merged[f], merged[target_name])
        scores[f] = abs(rho)
    return pd.Series(scores).sort_values(ascending=False)

# ---------------- STEP 2: TS AGGREGATION ----------------
AGG_FUNCS = {
    "mean": np.mean,
    "std": np.std,
    "p25": lambda x: np.percentile(x, 25),
    "p75": lambda x: np.percentile(x, 75),
    "p95": lambda x: np.percentile(x, 95),
    "diff": lambda x: x.iloc[-1] - x.iloc[0],
}

def aggregate_timeseries(ts_df, features=None):
    """
    Aggregate raw time-series into summary features per (user, path).
    
    Parameters
    ----------
    ts_df : pd.DataFrame
        Columns must include ['user', 'path', 'time'] + features
    features : list, optional
        Subset of columns to aggregate. If None, all except ['user', 'path', 'time'].

    Returns
    -------
    X_ts : pd.DataFrame
        Aggregated features, MultiIndex [user, path], semantic column names
    """
    if features is None:
        features = [c for c in ts_df.columns if c not in ["user", "path", "time"]]

    rows = []
    index = []

    for (user, path), g in ts_df.groupby(["user", "path"]):
        feats = {}
        for col in features:
            for name, fn in AGG_FUNCS.items():
                feats[f"{col}_{name}"] = fn(g[col])
        rows.append(feats)
        index.append((user, path))

    X_ts = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=["user", "path"]))
    return X_ts

def prepare_path_features(path_df):
    Xp = path_df.set_index(["user", "path"])
    return Xp

def prepare_user_features(user_df):
    Xu = user_df.set_index("user")
    return Xu

def build_feature_matrix(ts_df, path_df, user_df):
    X_ts = aggregate_timeseries(ts_df)
    X_path = prepare_path_features(path_df)
    X_user = prepare_user_features(user_df)

    X = (
        X_ts
        .join(X_path, how="left")
        .reset_index(level="path", drop=True)
        .join(X_user, how="left")
    )

    # Final safety checks
    X = X.copy()
    X.columns = X.columns.astype(str)
    X.index.name = "user"

    assert X.columns.map(type).nunique() == 1
    assert "user" not in X.columns
    assert "path" not in X.columns

    return X

def get_target_series(target_df, target_name, X):
    y = target_df.set_index("user")[target_name]
    y = y.loc[X.index]
    return y


# ---------------- STEP 3: MERGE STATIC ----------------

def merge_all(X_ts, path_df, user_df):
    """
    Merge time-series aggregates, path-level features, and user-level features
    into a single DataFrame, preserving user and path indices.
    
    Parameters
    ----------
    X_ts : pd.DataFrame
        Aggregated time-series, MultiIndex [user, path]
    path_df : pd.DataFrame
        Path-level features, must include ['user', 'path']
    user_df : pd.DataFrame
        User-level features, must include ['user']

    Returns
    -------
    X_all : pd.DataFrame
        Full feature matrix with string column names
    groups : pd.Series
        Array/Series of user indices for LOSO grouping
    """
    # Path features: set MultiIndex
    path_features = path_df.set_index(["user", "path"])
    
    # User features: set index, will broadcast to paths
    user_features = user_df.set_index("user")
    
    # Join time-series + path
    X = X_ts.join(path_features, how="left")
    
    # Join user features
    X = X.join(user_features, on="user", how="left")
    
    # Ensure string column names
    X.columns = X.columns.astype(str)
    
    # LOSO groups are the user IDs
    groups = X.index.get_level_values("user")
    
    return X, groups

# ---------------- STEP 4: PERMUTATION ----------------

def permutation_importance_loso(X, y, groups):
    base = loso_scaled_rmse(X, y, groups)
    imps = []

    for col in X.columns:
        Xp = X.copy()
        Xp[col] = np.random.permutation(Xp[col].values)
        score = loso_scaled_rmse(Xp, y, groups)
        imps.append((col, score - base))

    return (
        pd.DataFrame(imps, columns=["feature", "importance"])
        .sort_values("importance", ascending=False)
    )

# ======================== MAIN ========================

def main():
    # -----------------------------
    # Load datasets
    # -----------------------------
    ts = pd.read_csv("/data/dataset_conv@1s/timeseries.csv")
    path = pd.read_csv("/data/dataset_conv@1s/path_related.csv")
    user = pd.read_csv("/data/dataset_conv@1s/user_related.csv")
    target = pd.read_csv("/data/dataset_conv@1s/target.csv")

    target_names = [c for c in target.columns if c != "user"]

    # -----------------------------
    # AGGREGATE TIME-SERIES FEATURES
    # -----------------------------
    X_ts = aggregate_timeseries(ts)
    # X_ts index: MultiIndex (user, path)
    # Columns: fully semantic names

    # -----------------------------
    # MERGE ALL FEATURES
    # -----------------------------
    X_full, groups = merge_all(X_ts, path, user)
    # Ensure string column names for sklearn
    X_full.columns = X_full.columns.astype(str)

    # -----------------------------
    # SCALE FEATURES
    # -----------------------------
    scaler = StandardScaler()
    Xs = pd.DataFrame(
        scaler.fit_transform(X_full),
        columns=X_full.columns,
        index=X_full.index
    )

    # -----------------------------
    # TARGET SELECTION (LOSO scaled RMSE)
    # -----------------------------
    # Merge targets on user index
    target_aligned = target.set_index("user").reindex(Xs.index.get_level_values("user"))
    target_scores = {
        t: loso_scaled_rmse(
            Xs,
            target_aligned[t],
            groups
        )
        for t in target_names
    }
    ranked_targets = pd.Series(target_scores).sort_values()
    best_targets = ranked_targets.head(TOP_K_TARGETS)

    print("\n=== SELECTED TARGETS ===")
    print(best_targets)

    # -----------------------------
    # RAW CORRELATION PRUNING
    # -----------------------------
    # Use first best target as reference
    corr = raw_ts_correlation(ts, target, best_targets.index[0])
    keep_ts = corr.head(int(len(corr) * PRUNE_CORR_RATIO)).index.tolist()
    print(f"\nKept {len(keep_ts)} / {len(corr)} raw TS channels after correlation screening")

    # -----------------------------
    # AGGREGATE PRUNED TIME-SERIES
    # -----------------------------
    X_ts_pruned = aggregate_timeseries(ts, features=keep_ts)
    X_all, groups = merge_all(X_ts_pruned, path, user)
    X_all.columns = X_all.columns.astype(str)
    X_all = pd.DataFrame(
        StandardScaler().fit_transform(X_all),
        columns=X_all.columns,
        index=X_all.index
    )

    # -----------------------------
    # ALIGN TARGET FOR PERMUTATION
    # -----------------------------
    y_ref = target.set_index("user").reindex(X_all.index.get_level_values("user"))[best_targets.index[0]]

    # -----------------------------
    # PERMUTATION IMPORTANCE
    # -----------------------------
    imp_df = permutation_importance_loso(X_all, y_ref, groups)
    imp_df = imp_df.sort_values("importance", ascending=False)
    selected_features = imp_df.head(TOP_K_FEATURES)["feature"].tolist()

    print("\n=== TOP FEATURES ===")
    print(imp_df.head(25))

    # -----------------------------
    # EXPORT PRUNED DATASETS
    # -----------------------------
    # Only keep original TS columns that contributed to selected features
    
    # 1. Identify available source columns
    ts_source_cols = set(ts.columns) - {"user", "path", "time"}
    path_source_cols = set(path.columns) - {"user", "path"}
    user_source_cols = set(user.columns) - {"user"}

    # 2. Buckets for selected columns
    keep_ts = set()
    keep_path = set()
    keep_user = set()

    # 3. Distribute selected features back to their source
    for feat in selected_features:
        # Check Path
        if feat in path_source_cols:
            keep_path.add(feat)
            continue
        
        # Check User
        if feat in user_source_cols:
            keep_user.add(feat)
            continue
        
        # Check Time-Series (reverse engineering the aggregation name)
        # Feature format is "{original_col}_{agg_func}"
        for col in ts_source_cols:
            if feat.startswith(f"{col}_"):
                suffix = feat[len(col)+1:]
                if suffix in AGG_FUNCS:
                    keep_ts.add(col)
                    break

    # 4. Create Pruned DataFrames
    ts_pruned = ts[["user", "path", "time"] + sorted(list(keep_ts))]
    path_pruned = path[["user", "path"] + sorted(list(keep_path))]
    user_pruned = user[["user"] + sorted(list(keep_user))]
    target_pruned = target[["user"] + list(best_targets.index)]

    # Ensure output folder exists
    os.makedirs(OUT_DIR, exist_ok=True)
    ts_pruned.to_csv(f"{OUT_DIR}/timeseries.csv", index=False)
    path_pruned.to_csv(f"{OUT_DIR}/path.csv", index=False)
    user_pruned.to_csv(f"{OUT_DIR}/user.csv", index=False)
    target_pruned.to_csv(f"{OUT_DIR}/target.csv", index=False)

    print("\n=== PRUNED DATASETS SAVED TO ===")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
