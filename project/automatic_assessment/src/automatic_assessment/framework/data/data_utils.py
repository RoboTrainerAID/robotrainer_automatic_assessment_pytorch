import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from automatic_assessment.framework.dimred.lars import select_multitarget_top_features_lars
from automatic_assessment.framework.dimred.correlation import select_features_by_correlation
from automatic_assessment.framework.data import schema

# X tuple convention: see framework/data/schema.py (single source of truth)


def apply_feature_selection(X_train: tuple, y_train: torch.Tensor, X_val: tuple, n_features: int = None, correlation_threshold: float = None) -> tuple:
    """
    Selects path features from X_train[0] based on y_train using LARS or Correlation.
    Automatic method selection:
    - If n_features is provided -> Use LARS.
    - If n_features is None and correlation_threshold is provided -> Use Correlation.
    - If neither -> No selection.

    Args:
        X_train: Input tuple (x_path, x_user, *ts_groups)
        y_train: Tensor of targets
        X_val: Validation tuple with the same layout
        n_features: Number of features to keep (trigger for LARS).
        correlation_threshold: Threshold for correlation (trigger for Correlation if LARS not used).

    Returns:
        (X_train_new, X_val_new) with modified path feature tensors.
    """

    selected_indices = None

    # Determine method
    if n_features is not None:
        # Mode: LARS
        x_path_train = X_train[schema.X_PATH].numpy()
        y_train_np = y_train.numpy()
        selected_indices = select_multitarget_top_features_lars(x_path_train, y_train_np, top_n_features=n_features)

    elif correlation_threshold is not None:
        # Mode: Correlation
        x_path_train = X_train[schema.X_PATH].numpy()
        y_train_np = y_train.numpy()
        selected_indices = select_features_by_correlation(x_path_train, y_train_np, correlation_threshold=correlation_threshold)

    else:
        # Mode: None
        return X_train, X_val

    if selected_indices is None or len(selected_indices) == 0:
        return X_train, X_val

    # Convert to tensor for indexing
    feat_indices = torch.tensor(selected_indices, dtype=torch.long)

    # Update Train Path Features: (N, P, F) -> (N, P, F_sel)
    Xt_list = list(X_train)
    Xt_list[schema.X_PATH] = Xt_list[schema.X_PATH][:, :, feat_indices]
    X_train_new = tuple(Xt_list)

    # Update Val Path Features
    Xv_list = list(X_val)
    Xv_list[schema.X_PATH] = Xv_list[schema.X_PATH][:, :, feat_indices]
    X_val_new = tuple(Xv_list)

    return X_train_new, X_val_new


def get_dataloader(*tensors, batch_size, shuffle=True):
    """
    Creates a DataLoader from arbitrary number of tensors.

    Args:
        *tensors: Variable number of tensors (X parts..., y)
        batch_size (int): Size of batches
        shuffle (bool): Whether to shuffle
    """
    dataset = TensorDataset(*tensors)
    # Essential for fast GPU transfer: pin_memory=True
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )


def slice_data(X: tuple, indices: np.ndarray) -> tuple:
    """Helper to slice tuple of arrays."""
    return tuple(x[indices] for x in X)


def _masked_channel_scaling(x_ts: torch.Tensor, mask: torch.Tensor):
    """
    Computes per-channel mean/std over VALID entries only.

    Args:
        x_ts: (N, P, C, T) float tensor, 0.0 at invalid positions.
        mask: (N, P, C, T) bool tensor, True where the bin holds data.

    Returns:
        (mean, std): tensors of shape (C,). Channels without any valid
        entry (or with zero variance) get mean=0 / std=1 so they pass
        through unchanged.
    """
    m = mask.float()
    counts = m.sum(dim=(0, 1, 3))                                   # (C,)
    safe_counts = counts.clamp(min=1.0)
    mean = (x_ts * m).sum(dim=(0, 1, 3)) / safe_counts              # (C,)
    var = (((x_ts - mean.view(1, 1, -1, 1)) * m) ** 2).sum(dim=(0, 1, 3)) / safe_counts
    std = var.sqrt()
    # Degenerate channels: no data or constant signal -> identity scaling
    mean = torch.where(counts > 0, mean, torch.zeros_like(mean))
    std = torch.where((counts > 0) & (std > 1e-8), std, torch.ones_like(std))
    return mean, std


def _apply_masked_scaling(x_ts: torch.Tensor, mask: torch.Tensor, mean, std) -> torch.Tensor:
    """Standardizes valid entries and RE-ZEROES invalid positions."""
    scaled = (x_ts - mean.view(1, 1, -1, 1)) / std.view(1, 1, -1, 1)
    return scaled * mask.float()


def _assert_finite(name: str, tensor: torch.Tensor):
    if tensor.numel() > 0 and not torch.isfinite(tensor).all():
        raise ValueError(
            f"'{name}' contains NaN/Inf after scaling. Upstream data is invalid — "
            "run dataset/process_dataset.py and check the validation report."
        )


def prepare_fold_data(X_t: tuple, y_t, X_v: tuple, y_v) -> tuple:
    """
    Scaling within the CV loop (fit on the fold-train split only).

    X structure: (x_path, x_user, g0_x, g0_mask, g1_x, g1_mask, ...)
    - Each timeseries group is standardized per channel using ONLY valid
      (masked) entries, and invalid positions are re-zeroed afterwards.
      This keeps the padding at exactly 0 and keeps scaler statistics
      free of padding.
    - The masks are passed through unchanged (models use them for pooling).
    """
    scaler_y = StandardScaler()

    # Ensure y is numpy for fit_transform (handles tensors if passed)
    y_t_np = y_t.cpu().numpy() if isinstance(y_t, torch.Tensor) else y_t
    y_v_np = y_v.cpu().numpy() if isinstance(y_v, torch.Tensor) else y_v

    yt_s = torch.FloatTensor(scaler_y.fit_transform(y_t_np))
    yv_s = torch.FloatTensor(scaler_y.transform(y_v_np))
    feat_idx = None

    # 1. X_PATH (N, P, Fp) -> Scale Per feature Fp across N, P
    xt_path, xv_path = X_t[schema.X_PATH], X_v[schema.X_PATH]
    if isinstance(xt_path, torch.Tensor):
        xt_path = xt_path.cpu().numpy()
        xv_path = xv_path.cpu().numpy()

    N, P, Fp = xt_path.shape
    xt_path_flat = xt_path.reshape(-1, Fp)
    xv_path_flat = xv_path.reshape(-1, Fp)

    scaler_path = StandardScaler()
    xt_path_s = scaler_path.fit_transform(xt_path_flat).reshape(N, P, Fp)
    xv_path_s = scaler_path.transform(xv_path_flat).reshape(xv_path.shape[0], P, Fp)

    # 2. X_USER (N, Fu) -> Scale Per feature Fu across N
    xt_user, xv_user = X_t[schema.X_USER], X_v[schema.X_USER]
    if isinstance(xt_user, torch.Tensor):
        xt_user = xt_user.cpu().numpy()
        xv_user = xv_user.cpu().numpy()

    scaler_user = StandardScaler()
    xt_user_s = scaler_user.fit_transform(xt_user)
    xv_user_s = scaler_user.transform(xv_user)

    Xt_parts = [torch.FloatTensor(xt_path_s), torch.FloatTensor(xt_user_s)]
    Xv_parts = [torch.FloatTensor(xv_path_s), torch.FloatTensor(xv_user_s)]

    # 3. TS groups (N, P, C_g, T_g) -> masked standardization per channel
    for i in range(schema.GROUPS_START, len(X_t), 2):
        xt_g, mt_g = X_t[i], X_t[i + 1]
        xv_g, mv_g = X_v[i], X_v[i + 1]
        if xt_g.numel() > 0:
            mean_c, std_c = _masked_channel_scaling(xt_g, mt_g)
            xt_g_s = _apply_masked_scaling(xt_g, mt_g, mean_c, std_c).float()
            xv_g_s = _apply_masked_scaling(xv_g, mv_g, mean_c, std_c).float()
        else:
            xt_g_s, xv_g_s = xt_g, xv_g
        Xt_parts.extend([xt_g_s, mt_g])
        Xv_parts.extend([xv_g_s, mv_g])

    Xt_final = tuple(Xt_parts)
    Xv_final = tuple(Xv_parts)

    # Fail loudly on NaN/Inf instead of silently poisoning the training (analysis 3.5)
    _assert_finite("x_path (train)", Xt_final[schema.X_PATH])
    _assert_finite("x_user (train)", Xt_final[schema.X_USER])
    _assert_finite("y (train)", yt_s)
    _assert_finite("x_path (val)", Xv_final[schema.X_PATH])
    _assert_finite("x_user (val)", Xv_final[schema.X_USER])
    _assert_finite("y (val)", yv_s)
    for i, xt_g, _m in schema.iter_group_tensors(Xt_final):
        _assert_finite(f"ts group {(i - schema.GROUPS_START) // 2} (train)", xt_g)
    for i, xv_g, _m in schema.iter_group_tensors(Xv_final):
        _assert_finite(f"ts group {(i - schema.GROUPS_START) // 2} (val)", xv_g)

    return Xt_final, yt_s, Xv_final, yv_s, scaler_y, feat_idx


def get_root_groups(users):
    """
    Maps augmented user IDs back to original user IDs.
    Assumes augmentation user_id * 100 + i and original IDs < 100.
    """
    us = np.array(users)
    return np.array([u if u < 100 else u // 100 for u in us])


class AugmentedLOGO:
    """
    Custom splitter for Augmented Data.

    Behavior:
    1. Iterates over each 'real' user (ID < 100) as the Test Set.
    2. Drops all augmented versions (clones) of the current Test User from the Train Set.
       (Ensures we don't train on augmented versions of the subject we are testing on).
    3. Train set includes all other users AND their augmented versions.

    Args:
        include_augmented_in_test (bool): If True, the test set includes the original user
            AND their augmented clones. If False, only the original user is tested.
            In both cases, clones are excluded from the training set.
    """
    def __init__(self, include_augmented_in_test: bool = False):
        self.include_augmented_in_test = include_augmented_in_test

    def split(self, groups):
        groups = np.array(groups)
        root_groups = get_root_groups(groups)
        unique_roots = np.unique(root_groups)

        for root in unique_roots:
            # Identification of relevant records for this root (original + augmented)
            root_family_mask = (root_groups == root)

            if self.include_augmented_in_test:
                test_mask = root_family_mask
            else:
                # Test: Specifically records that match the ROOT ID exactly (original data only)
                test_mask = (groups == root)

            if np.sum(test_mask) == 0:
                continue

            # Train: Everything that is NOT part of the current root's family
            train_mask = ~root_family_mask

            yield np.where(train_mask)[0], np.where(test_mask)[0]
