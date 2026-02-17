import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

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

def prepare_fold_data(X_t: tuple, y_t, X_v: tuple, y_v) -> tuple:
    """Helper to handle scaling within the CV loop."""        
    scaler_y = StandardScaler()
    
    # Ensure y is numpy for fit_transform (handles tensors if passed)
    y_t_np = y_t.cpu().numpy() if isinstance(y_t, torch.Tensor) else y_t
    y_v_np = y_v.cpu().numpy() if isinstance(y_v, torch.Tensor) else y_v

    yt_s = torch.FloatTensor(scaler_y.fit_transform(y_t_np))
    yv_s = torch.FloatTensor(scaler_y.transform(y_v_np))
    feat_idx = None

    # Structure: (x_ts, x_path, x_user)
    # 1. X_TS (N, P, F, T) -> Scale Per feature F across N, P, T
    xt_ts, xv_ts = X_t[0], X_v[0]
    
    # Get dimensions for Training data
    N_t, P, F, T_t = xt_ts.shape
    
    # Get dimensions for Validation/Test data (T might differ or just be safe)
    N_v, _, _, T_v = xv_ts.shape
    
    # Handle both Tensor and Numpy inputs for reshaping/transposing
    # if isinstance(xt_ts, torch.Tensor):
    xt_ts_flat = xt_ts.permute(0,1,3,2).reshape(-1, F).cpu().numpy()
    xv_ts_flat = xv_ts.permute(0,1,3,2).reshape(-1, F).cpu().numpy()
    # else:
    #     xt_ts_flat = xt_ts.transpose(0,1,3,2).reshape(-1, F)
    #     xv_ts_flat = xv_ts.transpose(0,1,3,2).reshape(-1, F)
    
    scaler_ts = StandardScaler()
    # Scaling - returns numpy
    # Reshape back using the specific dimensions of each set
    xt_ts_s = scaler_ts.fit_transform(xt_ts_flat).reshape(N_t, P, T_t, F).transpose(0,1,3,2)
    xv_ts_s = scaler_ts.transform(xv_ts_flat).reshape(N_v, P, T_v, F).transpose(0,1,3,2)
    
    # 2. X_PATH (N, P, Fp) -> Scale Per feature Fp across N, P
    xt_path, xv_path = X_t[1], X_v[1]
    if isinstance(xt_path, torch.Tensor):
        xt_path = xt_path.cpu().numpy()
        xv_path = xv_path.cpu().numpy()

    N, P, Fp = xt_path.shape
    xt_path_flat = xt_path.reshape(-1, Fp)
    xv_path_flat = xv_path.reshape(-1, Fp)
    
    scaler_path = StandardScaler()
    xt_path_s = scaler_path.fit_transform(xt_path_flat).reshape(N, P, Fp)
    xv_path_s = scaler_path.transform(xv_path_flat).reshape(xv_path.shape[0], P, Fp)
    
    # 3. X_USER (N, Fu) -> Scale Per feature Fu across N
    xt_user, xv_user = X_t[2], X_v[2]
    if isinstance(xt_user, torch.Tensor):
        xt_user = xt_user.cpu().numpy()
        xv_user = xv_user.cpu().numpy()

    scaler_user = StandardScaler()
    xt_user_s = scaler_user.fit_transform(xt_user)
    xv_user_s = scaler_user.transform(xv_user)

    Xt_final = (torch.FloatTensor(xt_ts_s), torch.FloatTensor(xt_path_s), torch.FloatTensor(xt_user_s))
    Xv_final = (torch.FloatTensor(xv_ts_s), torch.FloatTensor(xv_path_s), torch.FloatTensor(xv_user_s))
    
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
            # We want to exclude ALL of them from training if we are testing on the root
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