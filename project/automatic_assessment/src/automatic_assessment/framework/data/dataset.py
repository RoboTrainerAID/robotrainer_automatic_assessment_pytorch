import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset
from typing import List, Tuple, Dict, Any, Union

import automatic_assessment.dataset.config as config
from automatic_assessment.dataset.timeseries_loader import TimeseriesLoader, TimeseriesDataset

def prepare_and_split_data(
    numpy_folder_name: str,
    input_root: str,
    output_root: str,
    test_user_ids: List[int]
):
    """
    Loads raw numpy data and related CSVs, performs data alignment and merging,
    splits the dataset based on test users, and saves the resulting splits 
    to separate folders (train/test).
    """
    print(f"\nPreparing dataset from {numpy_folder_name}...")
    root = Path(output_root).resolve()
    if not root.exists():
        print(f"Directory {root} does not exist. Check your paths")
        exit(1)
    input_path = Path(input_root)

    # 1. Load Timeseries Data (Raw numpy structure)
    loader = TimeseriesLoader(config) 
    full_ts_dataset = loader.load(input_path / numpy_folder_name)

    # 2. Load CSVs
    targets_df = pd.read_csv(input_path / "motoric_test.csv")
    user_feat_df = pd.read_csv(input_path / "demographics.csv")
    path_static_df = pd.read_csv(input_path / "task_difficulty.csv")
    path_dynamic_df = pd.read_csv(input_path / "timeseries_features.csv")

    # 3. Column Standardization & Merging
    # Ensure standard names 'user' and 'path'
    def standardize_cols(df):
        cols = df.columns
        if 'user_id' in cols: df.rename(columns={'user_id': 'user'}, inplace=True)
        if 'path_id' in cols: df.rename(columns={'path_id': 'path'}, inplace=True)

    standardize_cols(targets_df)
    standardize_cols(user_feat_df)
    standardize_cols(path_static_df)
    standardize_cols(path_dynamic_df)

    # Exclude known non-numeric columns like 'sex'
    if 'sex' in user_feat_df.columns:
        print("Dropping non-numeric column 'sex' from user features.")
        user_feat_df.drop(columns=['sex'], inplace=True)

    # Check for remaining non-numeric columns and warn
    for df, name in [(user_feat_df, 'User Features'), (path_static_df, 'Task Difficulty'), (path_dynamic_df, 'Timeseries Features')]:
        non_numeric = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"Warning: {name} contains non-numeric columns which may cause issues during tensor conversion: {list(non_numeric)}")

    # Merge Static Path features (task_difficulty) into Dynamic Path features (timeseries_features)
    # Result: Dataframe with (User, Path) granularity containing all path-related info
    merged_path_df = pd.merge(path_dynamic_df, path_static_df, on='path', how='left')

    # 4. Define Splits
    all_users = sorted(targets_df['user'].unique())
    train_users = [u for u in all_users if u not in test_user_ids]
    
    # Convert numpy ints to python ints for cleaner printing
    train_users_print = [int(u) for u in train_users]

    print(f"Total Users: {len(all_users)}")
    print(f"Test Split Users ({len(test_user_ids)}): {test_user_ids}")
    print(f"Train Split Users ({len(train_users)}): {train_users_print}")

    # 5. Save Splits Helper
    def save_split_data(split_name: str, user_ids: List[int]):
        split_dir = root / split_name
        split_dir.mkdir(exist_ok=True)
        
        # Filter DataFrames
        u_sub = user_feat_df[user_feat_df['user'].isin(user_ids)].copy()
        t_sub = targets_df[targets_df['user'].isin(user_ids)].copy()
        p_sub = merged_path_df[merged_path_df['user'].isin(user_ids)].copy()
        
        # Filter Timeseries
        # TimeseriesDataset behaves like a dictionary {user_id: {path_id: PathData}}
        ts_map_sub = {uid: full_ts_dataset[uid] for uid in user_ids if uid in full_ts_dataset}
        ts_dataset_sub = TimeseriesDataset(ts_map_sub)
        
        # Save to split folder
        if not u_sub.empty:
            print(f"Saving {split_name} split to {split_dir}...")
            u_sub.to_csv(split_dir / "user_features.csv", index=False)
            t_sub.to_csv(split_dir / "targets.csv", index=False)
            p_sub.to_csv(split_dir / "path_features.csv", index=False)
            
            ts_dir = split_dir / "timeseries"
            ts_dataset_sub.save(ts_dir)
        else:
            print(f"Warning: Empty split for {split_name}")

    save_split_data("train", train_users)
    save_split_data("test", test_user_ids)
    print(f"Preparation complete. Data saved to {root}")


def _load_dataset_split(
    folder_path: str, 
    load_timeseries: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, Dict]:
    """
    Internal helper to load a saved split into memory as Tensors.
    Returns: x_ts, x_path, x_user, y, user_ids, feature_names
    """
    root = Path(folder_path)
    if not (root / "targets.csv").exists():
        raise FileNotFoundError(f"Dataset split not found at {root}")

    # Load CSVs
    targets_df = pd.read_csv(root / "targets.csv")
    user_feat_df = pd.read_csv(root / "user_features.csv")
    path_feat_df = pd.read_csv(root / "path_features.csv")
    
    # Sort to ensure alignment
    targets_df.sort_values('user', inplace=True)
    user_feat_df.sort_values('user', inplace=True)
    path_feat_df.sort_values(['user', 'path'], inplace=True)
    
    users = targets_df['user'].values
    unique_paths = sorted(path_feat_df['path'].unique())
    
    n_samples = len(users)
    n_paths = len(unique_paths)

    # --- 1. Y Targets ---
    y_cols = [c for c in targets_df.columns if c != 'user']
    y = torch.tensor(targets_df[y_cols].values, dtype=torch.float32)
    
    # --- 2. X User (N, UserFeats) ---
    x_user_cols = [c for c in user_feat_df.columns if c != 'user']
    x_user = torch.tensor(user_feat_df[x_user_cols].values, dtype=torch.float32)
    
    # --- 3. X Path (N, Paths, PathFeats) ---
    # Realign to grid (User x Path)
    mi = pd.MultiIndex.from_product([users, unique_paths], names=['user', 'path'])
    path_feat_df = path_feat_df.set_index(['user', 'path']).reindex(mi)
    
    x_path_cols = path_feat_df.columns.tolist()
    n_path_feats = len(x_path_cols)
    
    # Reshape
    x_path_np = path_feat_df.values.astype(np.float32)
    x_path = torch.tensor(x_path_np.reshape(n_samples, n_paths, n_path_feats))
    
    # --- 4. X Timeseries (N, Paths, Channels, Time) ---
    x_ts_list = [] # Fallback if not padding, but we target tensor
    feature_names = {
        "user": x_user_cols,
        "path": x_path_cols,
        "targets": y_cols,
        "ts": []
    }
    
    if load_timeseries:
        ts_dir = root / "timeseries"
        if ts_dir.exists():
            loader = TimeseriesLoader(config)
            ts_data = loader.load(str(ts_dir))
            
            # Determine dimensions
            # Identify active channels from first valid entry
            sample_channels = []
            max_time = 0
            
            # Scan for consistency
            for u_id, u_paths in ts_data.items():
                for p_id, p_data in u_paths.items():
                    if not sample_channels and p_data.timeseries:
                        sample_channels = sorted(list(p_data.timeseries.keys()))
                    
                    # Find max length
                    for ts_name, arr in p_data.timeseries.items():
                         # numpy arrays shape logic: (Time,) or (Time, Cols)
                         ln = arr.shape[0]
                         if ln > max_time: max_time = ln
            
            if not sample_channels:
                print("Warning: No timeseries channels found.")
                x_ts = torch.empty(n_samples, n_paths, 0, 0)
            else:
                n_channels = len(sample_channels)
                feature_names['ts'] = sample_channels
                
                # Create Padded Tensor
                x_ts_np = np.zeros((n_samples, n_paths, n_channels, max_time), dtype=np.float32)
                
                user_map = {u: i for i, u in enumerate(users)}
                path_map = {p: i for i, p in enumerate(unique_paths)}
                
                for u_id, u_paths in ts_data.items():
                    if u_id not in user_map: continue
                    idx_u = user_map[u_id]
                    
                    for p_id, p_data in u_paths.items():
                        if p_id not in path_map: continue
                        idx_p = path_map[p_id]
                        
                        for c_idx, channel in enumerate(sample_channels):
                            if channel in p_data.timeseries:
                                arr = p_data.timeseries[channel]
                                # Extract value column if multi-column, assuming time is dim 0
                                if arr.ndim > 1:
                                    # Fallback: take col specified in config or 0
                                    val = arr[:, config.COL_VALUE] if arr.shape[1] > config.COL_VALUE else arr[:, 0]
                                else:
                                    val = arr
                                
                                ln = min(len(val), max_time)
                                x_ts_np[idx_u, idx_p, c_idx, :ln] = val[:ln]
                
                x_ts = torch.tensor(x_ts_np)
        else:
            x_ts = torch.empty(0)
    else:
        x_ts = torch.empty(0)
        
    return x_ts, x_path, x_user, y, users, feature_names


class AssessmentDataset(TorchDataset):
    """
    Dataset class that loads a prepared split (train/test) from disk.
    
    Structure:
    - x_ts: (n_samples, n_paths, n_timeseries, variable_timesteps [padded])
    - x_path: (n_samples, n_paths, n_path_features)
    - x_user: (n_samples, n_user_features)
    - y: (n_samples, n_targets)
    """
    def __init__(self, folder_path: str, load_timeseries: bool = True):
        self.folder_path = folder_path
        print(f"Loading dataset from {folder_path}...")
        
        self.x_ts, self.x_path, self.x_user, self.y, self.users, self.feature_names = _load_dataset_split(
            folder_path, load_timeseries
        )
        
        # Original dataset clones for restoration if needed
        self._x_ts_orig = self.x_ts.clone()
        self._x_path_orig = self.x_path.clone()
        self._x_user_orig = self.x_user.clone()
        self._y_orig = self.y.clone()
        
        # Make a deep copy of feature names to restore from
        import copy
        self._feature_names_orig = copy.deepcopy(self.feature_names)
        
        self.n_samples = self.y.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return tuple: (Features_Tuple, Target)
        # Features Tuple: (Time Series, Path Features, User Features)
        return (self.x_ts[idx], self.x_path[idx], self.x_user[idx]), self.y[idx]
    
    def get_all(self):
        return (self.x_ts, self.x_path, self.x_user), self.y, self.users, self.feature_names

    def create_augmented_dataset(self, augmentation_ratio: int):
        """
        Placeholder for data augmentation logic.
        Should update self.x_* tensors in place or extend them.
        """
        print(f"Placeholder: Creating augmented dataset with ratio {augmentation_ratio}...")
        # Implementation to follow
        pass

    def perform_feature_selection(self, method: str = 'lars', **kwargs):
        """
        Placeholder for feature selection logic.
        Should slice self.x_path or self.x_ts features.
        """
        print(f"Placeholder: Performing feature selection using {method}...")
        # Implementation to follow
        pass

    def perform_target_selection(self, selected_targets: Union[List[str], str]):
        """
        Updates self.y and self.feature_names['targets'] to include only the specified targets.
        Always restores from self._y_orig first to allow sequential calls with different sets.
        """
        if isinstance(selected_targets, str):
            selected_targets = [selected_targets]
            
        print(f"Selecting targets: {selected_targets}")
        
        # 1. Restore from original (including any augmentation applied if logic was updated to update _orig)
        # Note: If augmentation is applied, it should update _y_orig too or handle differencing.
        # For now assuming static dataset or that augmentation updates _orig
        self.y = self._y_orig.clone()
        import copy
        self.feature_names = copy.deepcopy(self._feature_names_orig)
        
        all_targets = self.feature_names['targets']
        
        # 2. Find indices
        indices = []
        found_targets = []
        
        for t in selected_targets:
            if t in all_targets:
                indices.append(all_targets.index(t))
                found_targets.append(t)
            else:
                print(f"Warning: Target '{t}' not found in dataset. Available: {all_targets}")
        
        if not indices:
            raise ValueError(f"No valid targets selected from available: {all_targets}")
            
        # 3. Slice y
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        self.y = self.y[:, indices_tensor]
        
        # 4. Update feature names
        self.feature_names['targets'] = found_targets
        
        print(f"Dataset updated. New target shape: {self.y.shape}")


if __name__ == "__main__":
    # Example usage
    prepare_and_split_data(
        numpy_folder_name="timeseries_numpy_processed",
        input_root="/data/raw",
        output_root="/data",
        test_user_ids=[14, 19, 27]  # most average test users by normative values from IfSS
    )
    
    dataset = AssessmentDataset("/data/train")
    print(f"Loaded dataset with {len(dataset)} samples.")