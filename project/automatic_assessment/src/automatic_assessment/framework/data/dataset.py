import copy
import json
from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset
from typing import List, Tuple, Dict, Any, Union

import automatic_assessment.dataset.config as config
from automatic_assessment.dataset.timeseries_loader import TimeseriesLoader, TimeseriesDataset
from automatic_assessment.dataset.timeseries_augmentation import TimeseriesAugmenter
from automatic_assessment.dataset.timeseries_features import TimeseriesFeatureExtractor
from automatic_assessment.framework.data.schema import TSGroup, build_X, group_specs

# ============================================================
# Timeseries tensor building (grouped, full resolution)
# ============================================================
# The learned-feature models consume one tensor PER CHANNEL GROUP
# (see config.TS_MODEL_GROUPS): mechanical @ 50 Hz, physiological @ 2 Hz,
# gait events @ 2 Hz. Within a group all channels share one time grid
# (bins of 1/rate seconds, sample means per bin, boolean validity mask);
# groups are NOT aligned to each other. This keeps the fast force
# dynamics at high resolution instead of averaging everything to 1 Hz.
#
# The tensors are PRECOMPUTED once per split folder into
# `ts_tensors_cache.npz` (written at split creation and self-healing at
# load time). The statistical features (timeseries_features.csv) always
# use the raw full-resolution series — this grouping only affects the
# model tensors.
#
# X tuple convention used across the framework:
#     X = (x_path, x_user, g0_x, g0_mask, g1_x, g1_mask, ...)
# with groups in config.TS_MODEL_GROUPS order.

TS_CACHE_FILE = "ts_tensors_cache.npz"
TS_CACHE_VERSION = 2


def _build_group_tensors(ts_data, users, unique_paths) -> List[Dict[str, Any]]:
    """
    Builds one (x, mask) numpy tensor pair per channel group.

    For every (user, path) and group, a time grid with bins of
    1/rate_hz seconds starts at the earliest relative timestamp of the
    group's channels and covers their full span. Samples are averaged
    per bin; bins without samples are 0 with mask=False. Missing
    channels stay fully masked.
    """
    specs = group_specs()
    user_map = {u: i for i, u in enumerate(users)}
    path_map = {p: i for i, p in enumerate(unique_paths)}
    n_samples, n_paths = len(users), len(unique_paths)

    groups = []
    for spec in specs:
        channels = spec["channels"]
        bin_s = 1.0 / spec["rate_hz"]
        n_channels = len(channels)
        channel_set = set(channels)

        # Pass 1: per-path grid info (t0, n_bins) over this group's channels
        grid_infos = {}
        max_time = 0
        for u_id, u_paths in ts_data.items():
            for p_id, p_data in u_paths.items():
                t_start, t_end = None, None
                for ch_name, arr in p_data.timeseries.items():
                    if ch_name not in channel_set:
                        continue
                    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] <= config.COL_VALUE:
                        continue
                    rel = arr[:, config.COL_REL_TS]
                    lo, hi = float(rel[0]), float(rel[-1])
                    t_start = lo if t_start is None else min(t_start, lo)
                    t_end = hi if t_end is None else max(t_end, hi)
                if t_start is None:
                    grid_infos[(u_id, p_id)] = (0.0, 0)
                else:
                    n_bins = int(np.floor((t_end - t_start) / bin_s)) + 1
                    grid_infos[(u_id, p_id)] = (t_start, n_bins)
                    max_time = max(max_time, n_bins)

        x_np = np.zeros((n_samples, n_paths, n_channels, max(max_time, 1)), dtype=np.float32)
        mask_np = np.zeros((n_samples, n_paths, n_channels, max(max_time, 1)), dtype=bool)

        # Pass 2: bin-average every channel onto the group grid
        for u_id, u_paths in ts_data.items():
            if u_id not in user_map:
                continue
            idx_u = user_map[u_id]
            for p_id, p_data in u_paths.items():
                if p_id not in path_map:
                    continue
                idx_p = path_map[p_id]
                t0, n_bins = grid_infos[(u_id, p_id)]
                if n_bins == 0:
                    continue

                for c_idx, channel in enumerate(channels):
                    arr = p_data.timeseries.get(channel)
                    if arr is None or arr.ndim != 2 or arr.shape[0] == 0:
                        continue  # missing channel -> stays fully masked
                    if arr.shape[1] <= config.COL_VALUE:
                        raise ValueError(
                            f"Timeseries '{channel}' (user {u_id}, path {p_id}) has "
                            f"{arr.shape[1]} columns; expected value column at index "
                            f"{config.COL_VALUE} ([raw_ts, rel_ts, value])."
                        )

                    rel = arr[:, config.COL_REL_TS]
                    val = arr[:, config.COL_VALUE]
                    bins = np.floor((rel - t0) / bin_s).astype(np.int64)
                    bins = np.clip(bins, 0, n_bins - 1)

                    counts = np.bincount(bins, minlength=n_bins)
                    sums = np.bincount(bins, weights=val, minlength=n_bins)
                    valid = counts > 0
                    binned = np.zeros(n_bins, dtype=np.float32)
                    binned[valid] = (sums[valid] / counts[valid]).astype(np.float32)

                    x_np[idx_u, idx_p, c_idx, :n_bins] = binned
                    mask_np[idx_u, idx_p, c_idx, :n_bins] = valid

        groups.append({**spec, "x": x_np, "mask": mask_np})

    return groups


def _ts_cache_meta(users, unique_paths) -> Dict[str, Any]:
    return {
        "version": TS_CACHE_VERSION,
        "users": [int(u) for u in users],
        "paths": [int(p) for p in unique_paths],
        "groups": group_specs(),
    }


def _save_ts_cache(root: Path, groups: List[Dict[str, Any]], users, unique_paths) -> None:
    arrays = {}
    for i, g in enumerate(groups):
        arrays[f"g{i}_x"] = g["x"]
        arrays[f"g{i}_mask"] = g["mask"]
    meta = json.dumps(_ts_cache_meta(users, unique_paths))
    np.savez_compressed(root / TS_CACHE_FILE, meta=np.array(meta), **arrays)
    total_mb = sum(g["x"].nbytes + g["mask"].nbytes for g in groups) / 1e6
    print(f"Precomputed timeseries tensors cached to {root / TS_CACHE_FILE} "
          f"({total_mb:.1f} MB in memory).")


def _load_ts_cache(root: Path, users, unique_paths):
    """Returns cached groups if the cache matches data + config, else None."""
    cache_path = root / TS_CACHE_FILE
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
    except Exception as e:
        print(f"Warning: could not read ts tensor cache ({e!r}) — rebuilding.")
        return None

    if meta != _ts_cache_meta(users, unique_paths):
        print("Timeseries tensor cache is stale (data or group config changed) — rebuilding.")
        return None

    groups = []
    for i, spec in enumerate(meta["groups"]):
        groups.append({**spec, "x": data[f"g{i}_x"], "mask": data[f"g{i}_mask"]})
    return groups


def prepare_and_split_data(
    numpy_folder_name: str,
    input_root: str,
    output_root: str,
    test_user_ids: List[int]
):
    """
    Loads raw numpy data and related CSVs, performs data alignment and merging,
    splits the dataset based on test users, and saves the resulting splits
    to separate folders (train/test). The grouped timeseries tensors are
    precomputed per split (ts_tensors_cache.npz) so training runs never pay
    the build cost.
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
    merged_path_df = pd.merge(path_dynamic_df, path_static_df, on='path', how='left')

    # 4. Define Splits
    all_users = sorted(targets_df['user'].unique())
    train_users = [u for u in all_users if u not in test_user_ids]

    train_users_print = [int(u) for u in train_users]

    print(f"Total Users: {len(all_users)}")
    print(f"Test Split Users ({len(test_user_ids)}): {test_user_ids}")
    print(f"Train Split Users ({len(train_users)}): {train_users_print}")

    # 5. Save Splits Helper
    def save_split_data(split_name: str, user_ids: List[int], augment: bool):
        split_dir = root / split_name
        split_dir.mkdir(exist_ok=True)

        # Remove a stale tensor cache from a previous split generation
        stale_cache = split_dir / TS_CACHE_FILE
        if stale_cache.exists():
            stale_cache.unlink()

        # Filter DataFrames
        u_sub = user_feat_df[user_feat_df['user'].isin(user_ids)].copy()
        t_sub = targets_df[targets_df['user'].isin(user_ids)].copy()
        p_sub = merged_path_df[merged_path_df['user'].isin(user_ids)].copy()

        # Filter Timeseries
        ts_map_sub = {uid: full_ts_dataset[uid] for uid in user_ids if uid in full_ts_dataset}

        # --- Data augmentation: synthetic clones, TRAINING split only ---
        # (config.AUGMENTATION; test users never get clones, so augmented
        #  information cannot leak into the hold-out. Experiments select
        #  how many clones to use via AssessmentDataset.view(...,
        #  augmentation_ratio=r) — no regeneration needed for ratio sweeps.)
        if augment and config.AUGMENTATION["max_ratio"] > 0:
            augmenter = TimeseriesAugmenter(config)
            clones = augmenter.augment_users(ts_map_sub)

            if clones:
                # Duplicate user/target rows with IDENTICAL values (a clone is
                # a noisy re-measurement of the same person)
                u_rows, t_rows = [], []
                for clone_id in sorted(clones):
                    orig_id = clone_id // 100
                    u_row = u_sub[u_sub['user'] == orig_id].iloc[0].copy()
                    u_row['user'] = clone_id
                    u_rows.append(u_row)
                    t_row = t_sub[t_sub['user'] == orig_id].iloc[0].copy()
                    t_row['user'] = clone_id
                    t_rows.append(t_row)
                u_sub = pd.concat([u_sub, pd.DataFrame(u_rows)], ignore_index=True)
                t_sub = pd.concat([t_sub, pd.DataFrame(t_rows)], ignore_index=True)

                # Re-extract the statistical features from the NOISY traces
                # (validates config keys and NaNs loudly, same as originals)
                print(f"Extracting features for {len(clones)} synthetic users...")
                extractor = TimeseriesFeatureExtractor(TimeseriesDataset(clones))
                clone_dyn = extractor.extract_features()
                clone_nan = clone_dyn.drop(columns=['user', 'path']).isna().sum().sum()
                if clone_nan > 0:
                    raise RuntimeError(
                        f"Augmented features contain {int(clone_nan)} NaN values — "
                        "check the augmentation noise settings."
                    )
                # Same merge as for the originals (static task difficulty per path)
                clone_paths_df = pd.merge(clone_dyn, path_static_df, on='path', how='left')
                # Align column order with the original merged frame
                clone_paths_df = clone_paths_df[p_sub.columns]
                p_sub = pd.concat([p_sub, clone_paths_df], ignore_index=True)

                ts_map_sub = {**ts_map_sub, **clones}

        ts_dataset_sub = TimeseriesDataset(ts_map_sub)

        # Save to split folder
        if not u_sub.empty:
            print(f"Saving {split_name} split to {split_dir} "
                  f"({u_sub['user'].nunique()} users incl. clones)...")
            u_sub.to_csv(split_dir / "user_features.csv", index=False)
            t_sub.to_csv(split_dir / "targets.csv", index=False)
            p_sub.to_csv(split_dir / "path_features.csv", index=False)

            ts_dir = split_dir / "timeseries"
            ts_dataset_sub.save(ts_dir)

            # Precompute the grouped model tensors for this split
            print(f"Precomputing grouped timeseries tensors for {split_name}...")
            _load_dataset_split(str(split_dir))
        else:
            print(f"Warning: Empty split for {split_name}")

    save_split_data("train", train_users, augment=True)
    save_split_data("test", test_user_ids, augment=False)
    print(f"Preparation complete. Data saved to {root}")


def _load_dataset_split(
    folder_path: str,
    load_timeseries: bool = True
) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, Dict]:
    """
    Internal helper to load a saved split into memory as Tensors.

    Returns: ts_groups, x_path, x_user, y, user_ids, feature_names
    where ts_groups is a list of schema.TSGroup (x: (N,P,C,T) float32,
    mask: (N,P,C,T) bool) in config.TS_MODEL_GROUPS order. Grouped tensors come from the split's
    ts_tensors_cache.npz when valid, otherwise they are rebuilt from the
    raw series and the cache is (re)written.
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
    mi = pd.MultiIndex.from_product([users, unique_paths], names=['user', 'path'])
    path_feat_df = path_feat_df.set_index(['user', 'path']).reindex(mi)

    x_path_cols = path_feat_df.columns.tolist()
    n_path_feats = len(x_path_cols)

    x_path_np = path_feat_df.values.astype(np.float32)
    x_path = torch.tensor(x_path_np.reshape(n_samples, n_paths, n_path_feats))

    feature_names = {
        "user": x_user_cols,
        "path": x_path_cols,
        "targets": y_cols,
        "ts_groups": [],
    }

    # --- 4. Grouped Timeseries Tensors (cached) ---
    ts_groups: List[Dict[str, Any]] = []
    if load_timeseries:
        ts_dir = root / "timeseries"
        if ts_dir.exists():
            groups_np = _load_ts_cache(root, users, unique_paths)
            if groups_np is None:
                loader = TimeseriesLoader(config)
                ts_data = loader.load(str(ts_dir))
                groups_np = _build_group_tensors(ts_data, users, unique_paths)
                _save_ts_cache(root, groups_np, users, unique_paths)

            for g in groups_np:
                group = TSGroup(
                    name=g["name"],
                    rate_hz=g["rate_hz"],
                    channels=g["channels"],
                    x=torch.tensor(np.ascontiguousarray(g["x"])),
                    mask=torch.tensor(np.ascontiguousarray(g["mask"])),
                )
                ts_groups.append(group)
                feature_names["ts_groups"].append(group.spec)

    return ts_groups, x_path, x_user, y, users, feature_names


@dataclass
class DatasetView:
    """
    Immutable selection of an AssessmentDataset (R11): a specific target
    subset and augmentation ratio. Creating a view never mutates the
    parent dataset, so multiple experiments can share one loaded dataset.
    """
    x_path: torch.Tensor
    x_user: torch.Tensor
    ts_groups: List[TSGroup]
    y: torch.Tensor
    users: np.ndarray
    feature_names: Dict[str, Any]

    def get_all(self):
        return build_X(self.x_path, self.x_user, self.ts_groups), self.y, self.users, self.feature_names


def make_view(x_path, x_user, ts_groups, y, users, feature_names,
              targets: List[str] = None, augmentation_ratio: int = None) -> DatasetView:
    """
    Builds a DatasetView (module-level for testability).

    - targets: subset of target names (order preserved); None = all.
    - augmentation_ratio: keep original users plus clones with clone
      index <= ratio (clone id = original*100 + index). None or 0 keeps
      only the originals; on splits without clones the filter is a no-op.
    """
    users = np.asarray(users)

    # --- Row selection by augmentation ratio ---
    ratio = 0 if augmentation_ratio is None else int(augmentation_ratio)
    clone_idx = np.where(users >= 100, users % 100, 0)
    row_mask = clone_idx <= ratio
    row_idx = np.where(row_mask)[0]

    # --- Target selection ---
    all_targets = list(feature_names['targets'])
    if targets is None:
        targets = all_targets
    missing = [t for t in targets if t not in all_targets]
    if missing:
        raise ValueError(f"Unknown targets {missing}. Available: {all_targets}")
    t_idx = torch.tensor([all_targets.index(t) for t in targets], dtype=torch.long)

    fn = copy.deepcopy(feature_names)
    fn['targets'] = list(targets)

    return DatasetView(
        x_path=x_path[row_idx],
        x_user=x_user[row_idx],
        ts_groups=[TSGroup(name=g.name, rate_hz=g.rate_hz, channels=g.channels,
                           x=g.x[row_idx], mask=g.mask[row_idx])
                   for g in ts_groups],
        y=y[row_idx][:, t_idx],
        users=users[row_idx],
        feature_names=fn,
    )


class AssessmentDataset(TorchDataset):
    """
    Dataset class that loads a prepared split (train/test) from disk.

    Structure:
    - ts_groups: one tensor pair per channel group (config.TS_MODEL_GROUPS),
        each x: (n_samples, n_paths, n_group_channels, T_group) float32
        with 0.0 where no data, plus a boolean validity mask of the same
        shape. Groups have DIFFERENT rates/lengths and are not aligned.
    - x_path: (n_samples, n_paths, n_path_features)
    - x_user: (n_samples, n_user_features)
    - y: (n_samples, n_targets)

    The X tuple convention used across the framework is:
        X = (x_path, x_user, g0_x, g0_mask, g1_x, g1_mask, ...)

    The dataset itself is STATELESS after loading (R11): experiments
    select targets and augmentation ratio through `view(...)`, which
    returns an immutable DatasetView and never mutates this object.
    """
    def __init__(self, folder_path: str, load_timeseries: bool = True):
        self.folder_path = folder_path
        print(f"Loading dataset from {folder_path}...")

        (self.ts_groups, self.x_path, self.x_user,
         self.y, self.users, self.feature_names) = _load_dataset_split(
            folder_path, load_timeseries
        )

        self._validate_finite()

        self.n_samples = self.y.shape[0]

    def view(self, targets: List[str] = None, augmentation_ratio: int = None) -> DatasetView:
        """Immutable selection of targets and augmentation ratio (R11)."""
        return make_view(self.x_path, self.x_user, self.ts_groups, self.y,
                         self.users, self.feature_names,
                         targets=targets, augmentation_ratio=augmentation_ratio)

    def _validate_finite(self):
        """Fail loudly if any loaded tensor contains NaN/Inf (see analysis 3.5)."""
        problems = []
        for g in self.ts_groups:
            if g.x.numel() > 0 and not torch.isfinite(g.x).all():
                bad = (~torch.isfinite(g.x)).sum().item()
                problems.append(f"ts group '{g.name}' contains {bad} non-finite values")
        if not torch.isfinite(self.x_user).all():
            problems.append("x_user contains non-finite values")
        if not torch.isfinite(self.y).all():
            problems.append("y (targets) contains non-finite values")
        if not torch.isfinite(self.x_path).all():
            bad_cols = (~torch.isfinite(self.x_path)).reshape(-1, self.x_path.shape[-1]).any(dim=0)
            names = [n for n, b in zip(self.feature_names['path'], bad_cols.tolist()) if b]
            problems.append(
                f"x_path contains non-finite values in columns: {names}. "
                "This usually means a (user, path) row is missing in "
                "path_features.csv or a feature was NaN in timeseries_features.csv."
            )
        if problems:
            raise ValueError(
                f"Dataset split '{self.folder_path}' failed the finite-check:\n  - "
                + "\n  - ".join(problems)
            )

    def _build_X(self, idx=None):
        return build_X(self.x_path, self.x_user, self.ts_groups, idx=idx)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return tuple: (Features_Tuple, Target)
        return self._build_X(idx), self.y[idx]

    def get_all(self):
        """Full dataset without selection. Prefer view(...) in experiments."""
        return self._build_X(), self.y, self.users, self.feature_names


if __name__ == "__main__":
    # Example usage: regenerates the splits (train split gets synthetic
    # clones up to config.AUGMENTATION['max_ratio']; test split stays
    # 100% original users).
    prepare_and_split_data(
        numpy_folder_name="timeseries_numpy_processed",
        input_root="/data/raw",
        output_root="/data",
        test_user_ids=[14, 19, 27]  # most average test users by normative values from IfSS
    )

    dataset = AssessmentDataset("/data/train")
    v0 = dataset.view(augmentation_ratio=0)
    v_max = dataset.view(augmentation_ratio=config.AUGMENTATION["max_ratio"])
    print(f"Loaded train split: {len(v0.users)} original users, "
          f"{len(v_max.users)} incl. clones at max ratio.")
