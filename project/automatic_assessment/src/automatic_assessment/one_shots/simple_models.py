"""
experiment_pipeline_with_tuning.py

Enhanced experiment pipeline:
- Randomized hyperparameter search (manual) with global budget
- Per-fold scalers to prevent leakage
- Safer target scaling with small-std protections
- Save/load hyperparameters to CSV
- Diagnostics for unusually large predictions (Ridge issue)
- Improved plotting (legends, error bars, boxplots)
"""

import os
import csv
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
import xgboost as xgb
from tqdm import tqdm

from automatic_assessment.framework.data.dataset import DatasetConv1s, DatasetFreq1hzAugmentedx4
from automatic_assessment.framework.dimred.lasso import select_top_n_features_lasso

# -------------------- Global settings --------------------
HYPERPARAM_SEARCH_STEPS = 20  # global budget for randomized tuning (change as needed)
USE_HPARAM_TUNING = False      # set False to load saved hyperparams instead of tuning
HYPERPARAM_DIR = "experiment_results/hyperparams"
RESULTS_DIR = "experiment_results"
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Protection threshold for tiny y std (to avoid exploding scaled values)
Y_STD_MIN = 1e-3


# -------------------- Model configuration dataclass --------------------
@dataclass
class ModelConfig:
    skip: bool
    name: str
    base_estimator_cls: Any
    search_space: List[Any]  # skopt space objects
    default_params: Dict[str, Any] = field(default_factory=dict)
    supports_native_multioutput: bool = True

    def instantiate(self, params: Dict[str, Any]):
        """Instantiate the base estimator with params."""
        return self.base_estimator_cls(**params)

    def multioutput_wrapper(self, estimator):
        """Wrap estimator for multi-output if needed."""
        if self.supports_native_multioutput:
            return estimator
        else:
            return MultiOutputRegressor(estimator)

    def singleoutput_wrapper(self, estimator):
        """Return a wrapper for independent single-output training (MultiOutputRegressor)."""
        return MultiOutputRegressor(estimator)


# -------------------- Define models and parameter ranges --------------------
MODEL_CONFIGS = [
    # Simple Linear Model (Baseline)
    ModelConfig(
        skip=False,
        name="LinearRegression",
        base_estimator_cls=LinearRegression,
        search_space=[
            # Dummy parameter to satisfy optimizer loop, effectively no tuning
            Categorical([True], name="fit_intercept"), 
        ],
        default_params={"n_jobs": -1},
        supports_native_multioutput=True,
    ),
    # ElasticNet (Robust Linear Model: L1 + L2 regularization)
    ModelConfig(
        skip=False,
        name="ElasticNet",
        base_estimator_cls=ElasticNet,
        search_space=[
            Real(0.001, 10.0, prior="log-uniform", name="alpha"),  # Regularization strength
            Real(0.1, 0.9, name="l1_ratio"),  # Balance between L1 (Lasso) and L2 (Ridge)
        ],
        default_params={"random_state": RANDOM_SEED, "max_iter": 5000, "selection": "random"},
        supports_native_multioutput=True,
    ),
    # SVR: Reworked for stronger regularization on small dataset (560 samples)
    ModelConfig(
        skip=False,
        name="SVR",
        base_estimator_cls=SVR,
        search_space=[
            # Lower C range (0.01 to 10) to enforce simpler decision boundaries and prevent overfitting
            Real(0.01, 10.0, prior="log-uniform", name="C"),
            # Epsilon tube width
            Real(0.01, 0.5, prior="log-uniform", name="epsilon"),
            Categorical(["rbf", "linear"], name="kernel"),
            # Gamma control for RBF kernel radius
            Categorical(["scale", "auto"], name="gamma"),
        ],
        default_params={"cache_size": 500},
        supports_native_multioutput=False,  # requires wrapper
    ),
    # RandomForest: User-optimized parameters for small data
    ModelConfig(
        skip=False,
        name="RandomForest",
        base_estimator_cls=RandomForestRegressor,
        search_space=[
            Integer(200, 500, name="n_estimators"),
            Integer(2, 4, name="max_depth"),
            Integer(3, 5, name="min_samples_leaf"), # Higher min_samples_leaf prevents overfitting
            Categorical(["sqrt", "log2"], name="max_features"),
        ],
        default_params={"random_state": RANDOM_SEED, "n_jobs": -1},
        supports_native_multioutput=True,
    ),
    # XGBoost: Gradient Boosting with regularization
    ModelConfig(
        skip=False,
        name="XGBoost",
        base_estimator_cls=xgb.XGBRegressor if xgb else None,
        search_space=[
            Integer(50, 300, name="n_estimators"),
            Integer(2, 4, name="max_depth"),  # Keep trees shallow to prevent overfitting
            Real(0.01, 0.1, prior="log-uniform", name="learning_rate"),
            Real(0.5, 0.9, name="subsample"),      # Row sampling to prevent overfitting
            Real(0.5, 0.9, name="colsample_bytree"), # Feature sampling
            Real(0.01, 10.0, prior="log-uniform", name="reg_alpha"),  # L1 regularization
            Real(0.01, 10.0, prior="log-uniform", name="reg_lambda"), # L2 regularization
        ],
        default_params={"random_state": RANDOM_SEED, "n_jobs": -1, "objective": "reg:squarederror"},
        supports_native_multioutput=False, # Use wrapper for independent targets
    ),
    # Simple MLP: Conservative architecture (small capacity + high regularization) to prevent overfitting
    ModelConfig(
        skip=False,
        name="SimpleMLP",
        base_estimator_cls=MLPRegressor,
        search_space=[
            Categorical([(16,), (32,), (16, 8)], name="hidden_layer_sizes"),
            Real(0.01, 5.0, prior="log-uniform", name="alpha"),  # Strong L2 regularization
            Real(0.001, 0.01, prior="log-uniform", name="learning_rate_init"),
        ],
        default_params={
            "random_state": RANDOM_SEED, 
            "max_iter": 500, 
            "early_stopping": True, # Crucial for preventing overfitting
            "validation_fraction": 0.1,
            "n_iter_no_change": 10
        },
        supports_native_multioutput=True,
    ),
    # Medium MLP: Deeper architecture for capturing non-linear complexities
    ModelConfig(
        skip=False,
        name="MediumMLP",
        base_estimator_cls=MLPRegressor,
        search_space=[
            Categorical([(64, 32), (128, 64), (64, 32, 16)], name="hidden_layer_sizes"),
            Categorical(["relu", "tanh"], name="activation"),
            Real(0.0001, 0.1, prior="log-uniform", name="alpha"),
            Real(0.0001, 0.005, prior="log-uniform", name="learning_rate_init"),
        ],
        default_params={
            "random_state": RANDOM_SEED, 
            "max_iter": 1000,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 15
        },
        supports_native_multioutput=True,
    ),
]


# -------------------- Utilities --------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_hyperparams(model_name: str, params: Dict[str, Any], folder: str = HYPERPARAM_DIR):
    ensure_dir(folder)
    fname = os.path.join(folder, f"{model_name}_best_params.json")
    
    # Convert numpy types to native python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_params = {k: convert_to_native(v) for k, v in params.items()}

    with open(fname, "w") as fh:
        json.dump(clean_params, fh, indent=2)
    print(f"[HYP] Saved best hyperparams for {model_name} to {fname}")


def load_hyperparams(model_name: str, folder: str = HYPERPARAM_DIR) -> Dict[str, Any]:
    fname = os.path.join(folder, f"{model_name}_best_params.json")
    if os.path.exists(fname):
        with open(fname, "r") as fh:
            params = json.load(fh)
        print(f"[HYP] Loaded hyperparams for {model_name} from {fname}")
        return params
    else:
        print(f"[HYP] No saved hyperparams for {model_name} at {fname}")
        return {}


def sample_param_set_from_grid(param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Uniform random sampling from lists inside param_grid."""
    sampled = {}
    for k, v in param_grid.items():
        if not isinstance(v, list):
            raise ValueError("Param grid values must be lists")
        sampled[k] = random.choice(v)
    return sampled


def get_real_user(user_id: int) -> int:
    """
    Resolves the original user ID from an augmented ID.
    Augmentation scheme: new_id = original_user_id * 100 + clone_index
    Assumption: Original user IDs are < 100.
    """
    if user_id < 100:
        return user_id
    return user_id // 100


# -------------------- Hyperparameter tuning (Bayesian Optimization) --------------------
def bayesian_optimization(
    X: np.ndarray,
    y: np.ndarray,
    users: List[int],
    cfg: ModelConfig,
    n_iter: int = HYPERPARAM_SEARCH_STEPS,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Bayesian optimization using scikit-optimize.
    Evaluates configurations using Leave-One-Group-Out CV (LOGO) where groups are users.
    Returns the best parameter dict (lowest mean RMSE across user folds).
    """
    opt = Optimizer(cfg.search_space, random_state=seed)
    
    best_score = float("inf")
    best_params = None

    # Identify unique users for LOGO
    unique_users = sorted(list(set(users)))
    n_groups = len(unique_users)
    
    if n_groups <= 1:
        raise ValueError("Need at least 2 unique users for LOGO tuning")

    print(f"[HYP] Starting Bayesian optimization for {cfg.name} with {n_iter} iterations (LOGO over {n_groups} users)...")

    # Convert users list to numpy array for easier indexing
    users_arr = np.array(users)

    for it in range(n_iter):
        # Ask for a new set of parameters
        suggested = opt.ask()
        # Map values to names
        params = {dim.name: val for dim, val in zip(cfg.search_space, suggested)}
        
        # Handle special cases (e.g. max_depth=-1 -> None)
        if "max_depth" in params and params["max_depth"] == -1:
            params["max_depth"] = None

        # merge defaults but do not overwrite sampled keys
        merged_params = {**cfg.default_params, **params}

        # Collect predictions for LOGO
        preds = np.zeros_like(y, dtype=float)
        y_true_all = np.zeros_like(y, dtype=float)

        # LOGO over unique users
        for u_val in unique_users:
            # Boolean masks for current user group
            val_mask = (users_arr == u_val)
            train_mask = ~val_mask

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            # Fit scalers on the fold's train -> avoid leakage
            x_scaler = StandardScaler().fit(X_tr)
            X_tr_s = x_scaler.transform(X_tr)
            X_val_s = x_scaler.transform(X_val)

            # For targets: use StandardScaler (center+scale)
            y_scaler = StandardScaler().fit(y_tr)
            y_tr_s = y_scaler.transform(y_tr)
            # Beware tiny std: protect
            stds = y_scaler.scale_.copy()
            stds_safe = np.where(stds < Y_STD_MIN, 1.0, stds)
            # if tiny std encountered, hack back weights by replacing scale_
            if np.any(stds < Y_STD_MIN):
                y_scaler.scale_ = stds_safe

            y_val_s = y_scaler.transform(y_val)

            # instantiate estimator with merged_params
            base_est = cfg.instantiate(merged_params)
            # For multi-output: wrap if necessary (for training multi-target at once)
            est_multi = cfg.multioutput_wrapper(base_est)
            # fit
            try:
                est_multi.fit(X_tr_s, y_tr_s)
                y_val_pred_s = est_multi.predict(X_val_s)
            except Exception as e:
                # if estimator failed (e.g., invalid hyperparam combination), penalize heavily
                print(f"[HYP] Iter {it+1}/{n_iter} params {merged_params} failed during fit: {e}")
                y_val_pred_s = np.zeros_like(y_val_s) + 1e6

            # Store predictions and truth (in scaled space for optimization metric)
            preds[val_mask] = y_val_pred_s
            y_true_all[val_mask] = y_val_s

        # compute aggregated RMSE across all samples in scaled space
        score = rmse(y_true_all, preds)
        
        # Tell the optimizer the result
        opt.tell(suggested, score)

        # Lower is better
        if score < best_score:
            best_score = score
            best_params = merged_params
        
        if (it + 1) % max(1, n_iter // 5) == 0:
            print(f"[HYP] Iter {it+1}/{n_iter}: current best score {best_score:.4f}")

    print(f"[HYP] Best params for {cfg.name}: {best_params} (score {best_score:.4f})")
    return best_params


# -------------------- LOOCV evaluator (uses per-fold scalers; expects best_params provided) --------------------
def run_loocv_with_params(
    X: np.ndarray, y: np.ndarray, users: List[int], cfg: ModelConfig, best_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs Leave-One-Group-Out CV (LOGO) across the provided users. 
    Groups are defined by unique user IDs.
    For each user fold:
    - fits scalers on fold train (all other users)
    - trains estimator(s) with best_params
    - collects per-fold predictions and metrics (scaled-space RMSE)
    Returns dictionary with per-fold and aggregated metrics and per-user error records.
    """
    # Identify unique users for LOGO
    unique_users = sorted(list(set(users)))
    users_arr = np.array(users)
    
    n_targets = y.shape[1]
    preds_multi = np.zeros_like(y, dtype=float)
    preds_single = np.zeros_like(y, dtype=float)
    preds_baseline = np.zeros_like(y, dtype=float)
    
    # Store the scaled ground truth for each fold to compute aggregated scaled metrics later
    y_scaled_composite = np.zeros_like(y, dtype=float)

    per_fold_records = []
    per_user_error_records = {u: [] for u in unique_users}

    for idx, u_val in enumerate(tqdm(unique_users, desc=f"LOOCV {cfg.name}")):
        # Boolean masks for current user group
        val_mask = (users_arr == u_val)
        train_mask = ~val_mask
        
        if idx == 0:
            train_unique = sorted(list(set(users_arr[train_mask])))
            tqdm.write(f"[LOOCV FOLD 0] Test User: {u_val} | Train Users ({len(train_unique)}): {train_unique}")
            
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        # Fit fold scalers (prevent leakage)
        x_scaler = StandardScaler().fit(X_tr)
        X_tr_s = x_scaler.transform(X_tr)
        X_val_s = x_scaler.transform(X_val)

        y_scaler = StandardScaler().fit(y_tr)
        # Protect tiny std (avoid dividing by zero)
        stds = y_scaler.scale_.copy()
        stds_safe = np.where(stds < Y_STD_MIN, 1.0, stds)
        if np.any(stds < Y_STD_MIN):
            # warning for diagnostics
            warnings.warn(
                f"[LOOCV] tiny target std detected on fold train; replacing small stds with {Y_STD_MIN} to avoid huge scaling"
            )
            y_scaler.scale_ = stds_safe
        y_tr_s = y_scaler.transform(y_tr)
        y_val_s = y_scaler.transform(y_val)
        
        # Store scaled truth for this fold (fill corresponding rows)
        y_scaled_composite[val_mask] = y_val_s

        # baseline (mean of train in scaled space)
        baseline_mean = np.mean(y_tr_s, axis=0)[np.newaxis, :]
        # Broadcast baseline to match validation set size for RMSE calculation
        baseline_pred = np.tile(baseline_mean, (y_val_s.shape[0], 1))
        preds_baseline[val_mask] = baseline_pred

        # multi-output estimator
        base_est = cfg.instantiate(best_params)
        est_multi = cfg.multioutput_wrapper(base_est)
        est_multi.fit(X_tr_s, y_tr_s)
        y_pred_multi_s = est_multi.predict(X_val_s)
        preds_multi[val_mask] = y_pred_multi_s

        # single-output approach: train independent estimators per target via MultiOutputRegressor wrapper
        base_est_2 = cfg.instantiate(best_params)
        est_single = cfg.singleoutput_wrapper(base_est_2)
        est_single.fit(X_tr_s, y_tr_s)
        y_pred_single_s = est_single.predict(X_val_s)
        preds_single[val_mask] = y_pred_single_s

        # metrics (scaled space) - computed over the validation set (which may have multiple samples)
        rmse_multi = rmse(y_val_s, y_pred_multi_s)
        rmse_single = rmse(y_val_s, y_pred_single_s)
        rmse_baseline = rmse(y_val_s, baseline_pred)

        per_target_rmse_multi = np.sqrt(np.mean((y_val_s - y_pred_multi_s) ** 2, axis=0))
        per_target_rmse_single = np.sqrt(np.mean((y_val_s - y_pred_single_s) ** 2, axis=0))
        per_target_rmse_baseline = np.sqrt(np.mean((y_val_s - baseline_mean) ** 2, axis=0))

        per_fold_records.append(
            {
                "val_user": u_val,
                "rmse_multi": rmse_multi,
                "rmse_single": rmse_single,
                "rmse_baseline": rmse_baseline,
                "per_target_rmse_multi": per_target_rmse_multi,
                "per_target_rmse_single": per_target_rmse_single,
                "per_target_rmse_baseline": per_target_rmse_baseline,
            }
        )

        # collect per-user absolute scaled errors for boxplots
        abs_err_multi = np.abs(y_pred_multi_s - y_val_s).flatten().tolist()
        abs_err_single = np.abs(y_pred_single_s - y_val_s).flatten().tolist()
        abs_err_baseline = np.abs(baseline_mean - y_val_s).flatten().tolist()
        per_user_error_records[u_val].extend(abs_err_multi + abs_err_single + abs_err_baseline)

    # aggregate across folds using the composite scaled truth
    # This ensures we are comparing scaled predictions to scaled truth
    per_target_rmse_multi = np.sqrt(np.mean((y_scaled_composite - preds_multi) ** 2, axis=0))
    per_target_rmse_single = np.sqrt(np.mean((y_scaled_composite - preds_single) ** 2, axis=0))
    per_target_rmse_baseline = np.sqrt(np.mean((y_scaled_composite - preds_baseline) ** 2, axis=0))

    # per-user RMSE (mean across targets)
    per_user_rmse_multi = {}
    per_user_rmse_single = {}
    
    # Calculate per-user RMSE using the masks again
    for u_val in unique_users:
        val_mask = (users_arr == u_val)
        # Extract predictions and truth for this user
        u_true = y_scaled_composite[val_mask]
        u_pred_multi = preds_multi[val_mask]
        u_pred_single = preds_single[val_mask]
        
        per_user_rmse_multi[u_val] = float(np.sqrt(np.mean((u_true - u_pred_multi) ** 2)))
        per_user_rmse_single[u_val] = float(np.sqrt(np.mean((u_true - u_pred_single) ** 2)))

    aggregated = {
        "per_target_rmse_multi": per_target_rmse_multi,
        "per_target_rmse_single": per_target_rmse_single,
        "per_target_rmse_baseline": per_target_rmse_baseline,
        "per_user_rmse_multi": per_user_rmse_multi,
        "per_user_rmse_single": per_user_rmse_single,
        "preds_multi": preds_multi,
        "preds_single": preds_single,
        "preds_baseline": preds_baseline,
        "per_user_error_records": per_user_error_records,
        "per_fold_records": per_fold_records,
    }
    return aggregated


# -------------------- Test evaluation (train on full training pool, evaluate on outer test) --------------------
def evaluate_on_test_pool(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: ModelConfig,
    best_params: Dict[str, Any],
):
    """
    Fit on full training pool with scalers fitted on full training pool,
    evaluate on outer test users. Return predictions and RMSEs in scaled space.
    """
    # Fit scalers on full training pool
    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)
    X_test_s = x_scaler.transform(X_test)

    y_scaler = StandardScaler().fit(y_train)
    # protect tiny stds
    stds = y_scaler.scale_.copy()
    stds_safe = np.where(stds < Y_STD_MIN, 1.0, stds)
    if np.any(stds < Y_STD_MIN):
        warnings.warn("[TEST] tiny y std in full training pool; applying Y_STD_MIN protection")
        y_scaler.scale_ = stds_safe

    y_train_s = y_scaler.transform(y_train)
    y_test_s = y_scaler.transform(y_test)

    # train multi-output estimator on full training pool
    base_est = cfg.instantiate(best_params)
    est_multi = cfg.multioutput_wrapper(base_est)
    est_multi.fit(X_train_s, y_train_s)
    y_pred_multi_s = est_multi.predict(X_test_s)

    # single-output (independent models)
    base_est2 = cfg.instantiate(best_params)
    est_single = cfg.singleoutput_wrapper(base_est2)
    est_single.fit(X_train_s, y_train_s)
    y_pred_single_s = est_single.predict(X_test_s)

    # baseline
    baseline_mean = np.mean(y_train_s, axis=0)[np.newaxis, :]
    y_pred_baseline_s = np.tile(baseline_mean, (X_test_s.shape[0], 1))

    # compute per-target RMSEs in scaled space
    per_target_rmse_multi = np.sqrt(np.mean((y_test_s - y_pred_multi_s) ** 2, axis=0))
    per_target_rmse_single = np.sqrt(np.mean((y_test_s - y_pred_single_s) ** 2, axis=0))
    per_target_rmse_baseline = np.sqrt(np.mean((y_test_s - y_pred_baseline_s) ** 2, axis=0))

    # overall RMSE (mean across test users of RMSE across targets)
    overall_rmse_multi = float(np.mean(np.sqrt(np.mean((y_test_s - y_pred_multi_s) ** 2, axis=1))))
    overall_rmse_single = float(np.mean(np.sqrt(np.mean((y_test_s - y_pred_single_s) ** 2, axis=1))))
    overall_rmse_baseline = float(np.mean(np.sqrt(np.mean((y_test_s - y_pred_baseline_s) ** 2, axis=1))))

    # diagnostics: check unusually large predictions or NaNs
    def diagnostics(name, arr):
        if np.isnan(arr).any():
            print(f"[DIAG] NaN in predictions for {name}")
        max_abs = float(np.nanmax(np.abs(arr)))
        if max_abs > 1e6:
            print(f"[DIAG] Very large prediction magnitude ({max_abs:.2e}) for {name}")

    diagnostics(cfg.name + " multi", y_pred_multi_s)
    diagnostics(cfg.name + " single", y_pred_single_s)

    return {
        "y_pred_multi_s": y_pred_multi_s,
        "y_pred_single_s": y_pred_single_s,
        "y_pred_baseline_s": y_pred_baseline_s,
        "per_target_rmse_multi": per_target_rmse_multi,
        "per_target_rmse_single": per_target_rmse_single,
        "per_target_rmse_baseline": per_target_rmse_baseline,
        "overall_rmse_multi": overall_rmse_multi,
        "overall_rmse_single": overall_rmse_single,
        "overall_rmse_baseline": overall_rmse_baseline,
    }


# -------------------- Plotting --------------------
def plot_overall_model_comparison(agg_results: Dict[str, Dict[str, Any]], out_path: str):
    ensure_dir(out_path)
    model_names = []
    means = []
    stds = []
    for mname, res in agg_results.items():
        model_names.append(mname)
        per_fold = [r["rmse_multi"] for r in res["per_fold_records"]]
        means.append(np.mean(per_fold))
        stds.append(np.std(per_fold))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(model_names, means, yerr=stds, capsize=6)
    ax.set_title("LOOCV Overall Scaled RMSE (Multi-Output)\n(Mean across users; Error bars = Std Dev across users)")
    ax.set_ylabel("Scaled RMSE")
    ax.set_xlabel("Model")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend().set_visible(False)
    fname = os.path.join(out_path, "overall_model_comparison_multi_rmse.png")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    print(f"[PLOT] Saved {fname}")


def plot_per_target_comparison(agg_results: Dict[str, Dict[str, Any]], target_names: List[str], out_path: str):
    ensure_dir(out_path)
    n_targets = len(target_names)
    n_models = len(agg_results)
    fig, axes = plt.subplots(n_models, 1, figsize=(max(10, 1.5 * n_targets), 4 * n_models), squeeze=False)
    for r, (mname, res) in enumerate(agg_results.items()):
        ax = axes[r][0]
        # compute per-target mean and std across folds (we stored per_target_rmse per fold)
        per_target_rmse_multi_per_fold = np.array([fold["per_target_rmse_multi"] for fold in res["per_fold_records"]])
        per_target_rmse_single_per_fold = np.array([fold["per_target_rmse_single"] for fold in res["per_fold_records"]])
        per_target_rmse_baseline_per_fold = np.array([fold["per_target_rmse_baseline"] for fold in res["per_fold_records"]])

        mean_multi = per_target_rmse_multi_per_fold.mean(axis=0)
        std_multi = per_target_rmse_multi_per_fold.std(axis=0)
        mean_single = per_target_rmse_single_per_fold.mean(axis=0)
        std_single = per_target_rmse_single_per_fold.std(axis=0)
        mean_base = per_target_rmse_baseline_per_fold.mean(axis=0)
        std_base = per_target_rmse_baseline_per_fold.std(axis=0)

        x = np.arange(n_targets)
        width = 0.25
        ax.bar(x - width, mean_single, width=width, yerr=std_single, label="single-output", capsize=4)
        ax.bar(x, mean_multi, width=width, yerr=std_multi, label="multi-output", capsize=4)
        ax.bar(x + width, mean_base, width=width, yerr=std_base, label="baseline", capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(target_names, rotation=45, ha="right")
        ax.set_title(f"Per-Target Scaled RMSE (LOOCV) - {mname}\n(Mean across users; Error bars = Std Dev across users)")
        ax.set_ylabel("Scaled RMSE")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fname = os.path.join(out_path, "per_target_comparison_with_errorbars.png")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    print(f"[PLOT] Saved {fname}")


def plot_per_user_boxplot(all_models_error_records: Dict[str, Dict[int, List[float]]], out_path: str):
    ensure_dir(out_path)
    users = sorted({u for recs in all_models_error_records.values() for u in recs.keys()})
    combined_errors = []
    labels = []
    for u in users:
        vals = []
        for m in all_models_error_records:
            vals.extend(all_models_error_records[m].get(u, []))
        combined_errors.append(vals)
        labels.append(str(u))

    fig, ax = plt.subplots(figsize=(max(10, len(users) * 0.4), 6))
    ax.boxplot(combined_errors, tick_labels=labels, showmeans=True)
    ax.set_title("Per-User Distribution of Scaled Absolute Errors\n(Aggregated across all targets & models for each user)")
    ax.set_xlabel("User ID")
    ax.set_ylabel("Scaled Absolute Error")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fname = os.path.join(out_path, "per_user_error_boxplot.png")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    print(f"[PLOT] Saved {fname}")


# -------------------- High-level experiment runner --------------------
def run_experiment(save_dir: str = RESULTS_DIR, recreate_dataset: bool = False):
    ensure_dir(save_dir)
    ensure_dir(HYPERPARAM_DIR)

    # Load dataset
    ds = DatasetFreq1hzAugmentedx4(recreate=recreate_dataset)
    # X_all, y_all, users_all, feature_names = ds.get_path_level_dataset()
    X_all, y_all, users_all, feature_names = ds.get_user_level_dataset()

    X_all, feature_names = select_top_n_features_lasso(X_all, y_all, feature_names, n=50, alpha=0.1)
    
    # Filter 'user' from target names to match y_all shape
    if hasattr(ds, "target_df"):
        target_names = [c for c in ds.target_df.columns if c != "user"]
    else:
        target_names = [f"target_{i}" for i in range(y_all.shape[1])]

    print(f"[DATA] {len(users_all)} samples, X shape {X_all.shape}, y shape {y_all.shape}")

    # Outer split: pick deterministic test users (LOGO split)
    rng = np.random.RandomState(RANDOM_SEED)
    
    # 1. Map every sample to its 'Real' underlying user ID
    # This groups augmented clones (e.g., 501, 502) with the original user (5)
    users_all_real = np.array([get_real_user(u) for u in users_all])
    unique_real_users = sorted(list(set(users_all_real)))
    
    n_test = 4
    
    # 2. Select Test Users from the unique REAL users
    if len(unique_real_users) < n_test:
        raise ValueError(f"Not enough unique real users ({len(unique_real_users)}) for test set of size {n_test}")

    test_users = rng.choice(unique_real_users, size=n_test, replace=False).tolist()
    train_users_set = [u for u in unique_real_users if u not in test_users]
    
    print(f"[SPLIT] Test Real Users: {test_users}")
    print(f"[SPLIT] Train Real Users (count={len(train_users_set)})")

    # 3. Create Masks based on REAL User IDs
    # This ensures that if user 5 is in Test, 501 and 502 are ALSO in Test (and not Train)
    train_mask = np.isin(users_all_real, train_users_set)
    test_mask = np.isin(users_all_real, test_users)

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    
    # Pass REAL user IDs for CV grouping. 
    # This prevents leakage between augmented clones during internal cross-validation (LOGO).
    train_users_samples = users_all_real[train_mask].tolist()
    
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]

    # Verify sample counts
    print(f"[SPLIT] Total Train Samples: {X_train.shape[0]}")
    print(f"[SPLIT] Total Test Samples: {X_test.shape[0]}")
    total_samples = X_train.shape[0] + X_test.shape[0]
    print(f"[SPLIT] Total Samples Check: {total_samples} / {len(users_all)}")
    
    if total_samples != len(users_all):
        raise ValueError(f"Sample mismatch! Lost {len(users_all) - total_samples} samples during split.")

    results_summary = {}
    all_models_error_records = {}
    for cfg in MODEL_CONFIGS:
        if cfg.skip:
            print(f"[MODEL] Skipping {cfg.name} as per configuration.")
            continue
        print(f"\n[MODEL] Processing {cfg.name}")
        # load saved hyperparams if tuning disabled
        best_params = {}
        if USE_HPARAM_TUNING:
            best_params = bayesian_optimization(X_train, y_train, train_users_samples, cfg, n_iter=HYPERPARAM_SEARCH_STEPS)
            save_hyperparams(cfg.name, best_params)
        else:
            loaded = load_hyperparams(cfg.name)
            if loaded:
                best_params = loaded
            else:
                # fallback to default params if none saved
                print(f"[HYP] No saved params for {cfg.name}; using defaults")
                best_params = cfg.default_params.copy()

        # Ensure n_jobs is -1 for RandomForest to use all cores
        if cfg.name == "RandomForest":
            best_params["n_jobs"] = -1

        # Run LOGO-CV with best_params (per-fold scalers inside)
        aggregated = run_loocv_with_params(X_train, y_train, train_users_samples, cfg, best_params)
        # store aggregated and per-fold
        agg_out = aggregated.copy()
        agg_out["per_fold_records"] = aggregated["per_fold_records"]
        results_summary[cfg.name] = agg_out
        all_models_error_records[cfg.name] = aggregated["per_user_error_records"]

        # Calculate mean scores for printing
        folds = aggregated["per_fold_records"]
        mean_cv_multi = float(np.mean([f["rmse_multi"] for f in folds]))
        mean_cv_single = float(np.mean([f["rmse_single"] for f in folds]))
        mean_cv_baseline = float(np.mean([f["rmse_baseline"] for f in folds]))

        print(f"\n[LOOCV EVAL] {cfg.name} Mean Cross-Validation Scores (Scaled RMSE):")
        print(f"  Multi-Output: {mean_cv_multi:.4f}")
        print(f"  Single-Output: {mean_cv_single:.4f}")
        print(f"  Baseline:     {mean_cv_baseline:.4f}")

        # Save per-target LOOCV metrics to CSV (scaled-space RMSE)
        df_targets = pd.DataFrame(
            {
                "target": target_names,
                "rmse_multi": aggregated["per_target_rmse_multi"],
                "rmse_single": aggregated["per_target_rmse_single"],
                "rmse_baseline": aggregated["per_target_rmse_baseline"],
            }
        )
        df_targets.to_csv(os.path.join(save_dir, f"loocv_per_target_rmse_{cfg.name}.csv"), index=False)

        # Save per-fold summary
        per_fold_summary = []
        for f in aggregated["per_fold_records"]:
            per_fold_summary.append(
                {
                    "val_user": f["val_user"],
                    "rmse_multi": f["rmse_multi"],
                    "rmse_single": f["rmse_single"],
                    "rmse_baseline": f["rmse_baseline"],
                }
            )
        pd.DataFrame(per_fold_summary).to_csv(os.path.join(save_dir, f"loocv_folds_summary_{cfg.name}.csv"), index=False)

        # Evaluate on outer test pool
        test_eval = evaluate_on_test_pool(X_train, y_train, X_test, y_test, cfg, best_params)
        
        print(f"\n[TEST EVAL] {cfg.name} Final Test Scores (Scaled RMSE):")
        print(f"  Multi-Output: {test_eval['overall_rmse_multi']:.4f}")
        print(f"  Single-Output: {test_eval['overall_rmse_single']:.4f}")
        print(f"  Baseline:     {test_eval['overall_rmse_baseline']:.4f}")

        # Save test per-target RMSE (scaled)
        df_test = pd.DataFrame(
            {
                "target": target_names,
                "rmse_multi_test": test_eval["per_target_rmse_multi"],
                "rmse_single_test": test_eval["per_target_rmse_single"],
                "rmse_baseline_test": test_eval["per_target_rmse_baseline"],
            }
        )
        df_test.to_csv(os.path.join(save_dir, f"test_per_target_rmse_{cfg.name}.csv"), index=False)

        # Save best params JSON (already saved by save_hyperparams) and summary
        results_summary[cfg.name]["test_eval"] = {
            "overall_rmse_multi": test_eval["overall_rmse_multi"],
            "overall_rmse_single": test_eval["overall_rmse_single"],
            "overall_rmse_baseline": test_eval["overall_rmse_baseline"],
        }

    # Plotting
    plot_overall_model_comparison(results_summary, out_path=save_dir)
    plot_per_target_comparison(results_summary, target_names, out_path=save_dir)
    plot_per_user_boxplot(all_models_error_records, out_path=save_dir)

    # Save overall summary CSV
    rows = []
    for mname, res in results_summary.items():
        # compute mean LOOCV rmse across folds for multi & single
        fold_multi = [f["rmse_multi"] for f in res["per_fold_records"]]
        fold_single = [f["rmse_single"] for f in res["per_fold_records"]]
        rows.append(
            {
                "model": mname,
                "loocv_mean_rmse_multi": float(np.mean(fold_multi)),
                "loocv_std_rmse_multi": float(np.std(fold_multi)),
                "loocv_mean_rmse_single": float(np.mean(fold_single)),
                "loocv_std_rmse_single": float(np.std(fold_single)),
                "test_overall_rmse_multi": res["test_eval"]["overall_rmse_multi"],
                "test_overall_rmse_single": res["test_eval"]["overall_rmse_single"],
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "summary_overall_results.csv"), index=False)
    print(f"\n[FIN] All results saved in {os.path.abspath(save_dir)}")

    # Diagnostics for Ridge unusual numbers:
    # If you saw huge numbers in earlier run (1.97e+08), that was likely due to a near-zero target std somewhere
    # causing extremely large scaled values OR leakage/clipping outside training folds. With per-fold scaling and
    # small-std protection this should be resolved. If extreme values still occur, check the loocv_folds_summary_Ridge.csv
    # and the test_per_target_rmse_Ridge.csv files for outlier folds and inspect the raw predictions saved there.
    return results_summary


if __name__ == "__main__":
    # Run with tuning enabled. Turn USE_HPARAM_TUNING=False to reuse saved params.
    run_experiment(save_dir=RESULTS_DIR, recreate_dataset=False)
