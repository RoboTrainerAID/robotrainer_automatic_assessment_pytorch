"""
Data containers and loading logic for the timeseries dataset.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


@dataclass
class PathData:
    """All loaded (and preprocessed) timeseries for a single path."""
    user_id: int
    path_id: int
    meta: dict = field(default_factory=dict)
    timeseries: Dict[str, np.ndarray] = field(default_factory=dict)
    scalar_features: Dict[str, float] = field(default_factory=dict)
    motion_start_timestamp: Optional[float] = None


class TimeseriesLoader:
    """
    Loads, trims, and cleans the timeseries dataset based on the provided configuration.
    """

    def __init__(self, config: Any) -> None:
        """
        :param config: A module or object containing configuration variables:
                       DATASET_ROOT, TIMESERIES_TO_LOAD, TIMESERIES_TRIMMED_BY_MOTION_START,
                       PATHS_WITHOUT_DISTURBANCE, MOTION_START_THRESHOLD,
                       TIMESERIES_ZERO_IS_INVALID, COL_RAW_TS, COL_VALUE,
                       DATA_WITH_A_SINGLE_VALUE_PER_PATH, CSV_OUTPUT_PATH
        """
        self.cfg = config
        self.data: Dict[int, Dict[int, PathData]] = {}
        # Stores validation issues: user_id -> list of issue messages
        self.validation_report: Dict[int, List[str]] = {}
        self.features: Optional[pd.DataFrame] = None

    def load_all(self) -> None:
        """Discover and load all users and paths."""
        root = Path(self.cfg.DATASET_ROOT).resolve()
        if not root.exists():
            print(f"Error: Dataset root not found at {root}")
            return

        user_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("U")])
        
        for user_dir in user_dirs:
            try:
                user_id = int(user_dir.name[1:])
            except ValueError:
                print(f"Skipping invalid user directory: {user_dir.name}")
                continue
                
            self.data[user_id] = {}
            
            # Sort paths numerically (path_1, path_2, ...)
            path_dirs = sorted(
                [d for d in user_dir.iterdir() if d.is_dir() and d.name.startswith("path_")],
                key=lambda p: int(p.name.split("_")[1]) if "_" in p.name and p.name.split("_")[1].isdigit() else 0
            )

            for path_dir in path_dirs:
                try:
                    path_id = int(path_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue

                pd = self._load_path(user_id, path_id, path_dir)
                if pd:
                    self.data[user_id][path_id] = pd
            
            print(f"Loaded User {user_id}: {len(self.data[user_id])} paths")
            
        self._print_validation_report()

    def _print_validation_report(self):
        """Prints a summary of missing timeseries or other issues."""
        if not self.validation_report:
            return
            
        print("\n" + "="*40)
        print("MISSING / EMPTY TIMESERIES REPORT")
        print("="*40)
        
        for user_id in sorted(self.validation_report.keys()):
            issues = self.validation_report[user_id]
            if not issues:
                continue
            
            print(f"\nUser {user_id}:")
            for issue in issues:
                print(f"  - {issue}")
        print("\n" + "="*40 + "\n")

    def get_all_paths(self) -> List[PathData]:
        """Flatten the data structure to a list of PathData objects."""
        all_paths = []
        for user_data in self.data.values():
            all_paths.extend(user_data.values())
        return all_paths

    def _load_path(self, user_id: int, path_id: int, path_dir: Path) -> Optional[PathData]:
        pd = PathData(user_id=user_id, path_id=path_id)
        
        # Load Meta
        meta_path = path_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                pd.meta = json.load(f)

        # 1. Load Single Value Features
        for name in self.cfg.DATA_WITH_A_SINGLE_VALUE_PER_PATH:
            fpath = path_dir / f"{name}.npy"
            if fpath.exists():
                try:
                    # Load as array first, validate later
                    arr = np.load(str(fpath))
                    
                    # If this file has timestamps (columns), extract only the value column.
                    # This prevents counting timestamps as values during validation.
                    if arr.ndim == 2 and arr.shape[1] > self.cfg.COL_VALUE:
                         arr = arr[:, self.cfg.COL_VALUE]

                    pd.scalar_features[name] = arr
                except Exception as e:
                    print(f"Error loading scalar {fpath}: {e}")

        # Determine expected timeseries
        expected_ts = self.cfg.TIMESERIES_TO_LOAD.copy()
        
        if path_id in self.cfg.PATHS_WITHOUT_DISTURBANCE:
            expected_ts = [ts for ts in expected_ts if not ts.startswith("disturbance_force")]

        # Load .npy files
        for ts_name in expected_ts:
            fpath = path_dir / f"{ts_name}.npy"
            if not fpath.exists():
                # Missing files are handled in _validate_and_clean
                continue
            
            try:
                arr = np.load(str(fpath))
                if arr.ndim == 2 and arr.shape[1] >= 3:
                     pd.timeseries[ts_name] = arr
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

        # Preprocess special cases explicitly requested (HRV, PPG)
        self._preprocess_hrv(pd)
        self._preprocess_ppg(pd)

        # 1. Trim to motion start
        self._trim_path_data(pd)
        
        # 2. Filter invalid zeros (restore from memory)
        self._filter_invalid_zeros(pd)

        # 3. Final validation (check for empty or constant signals)
        self._validate_and_clean(pd, expected_ts)
        
        # Always return the path object, even if partially empty
        return pd

    def _preprocess_hrv(self, pd: PathData) -> None:
        """
        Keep only the first value column for HRV timeseries.
        HRV has data columns at indices 2 and 3; we keep 2 (plus timestamps 0, 1).
        """
        if "hrv" in pd.timeseries:
            arr = pd.timeseries["hrv"]
            # If there are more than 3 columns (ts_raw, ts_sync, val1), trim the rest.
            if arr.shape[1] > 3:
                pd.timeseries["hrv"] = arr[:, :3]

    def _preprocess_ppg(self, pd: PathData) -> None:
        """
        Flatten PPG channels which contain multiple samples per row.
        Each row has timestamps (cols 0,1) and multiple values (cols 2+).
        """
        target_ts = ["ppg_ch0", "ppg_ch1", "ppg_ch2", "ppg_ch3"]
        for name in target_ts:
            if name not in pd.timeseries:
                continue
            
            arr = pd.timeseries[name]
            n_cols = arr.shape[1]
            # Expecting ts_raw, ts_sync, and at least one value
            if n_cols <= 2:
                continue
            
            # Extract timestamps and repeat them for each value column
            timestamps = arr[:, :2]
            n_vals = n_cols - 2
            
            # Repeat rows: [t0, t0, t0, t1, t1, t1...]
            timestamps_expanded = np.repeat(timestamps, n_vals, axis=0)
            
            # Extract values and flatten row-wise: [v0_0, v0_1, v0_2, v1_0...]
            values = arr[:, 2:]
            values_flattened = values.reshape(-1, 1)
            
            # Stack to create (N*n_vals, 3) matrix
            pd.timeseries[name] = np.hstack((timestamps_expanded, values_flattened))

    def _trim_path_data(self, pd: PathData):
        """Trim loaded timeseries based on robot_vel_x threshold."""
        if "robot_vel_x" not in pd.timeseries:
            return

        vel_x = pd.timeseries["robot_vel_x"]
        # Find index where abs(velocity) > threshold
        mask = np.abs(vel_x[:, self.cfg.COL_VALUE]) > self.cfg.MOTION_START_THRESHOLD
        if not np.any(mask):
            return # No motion detected

        start_idx = np.argmax(mask)
        start_time = vel_x[start_idx, self.cfg.COL_RAW_TS]
        pd.motion_start_timestamp = float(start_time)

        # Trim all configured series
        for ts_name in self.cfg.TIMESERIES_TRIMMED_BY_MOTION_START:
            if ts_name in pd.timeseries:
                arr = pd.timeseries[ts_name]
                pd.timeseries[ts_name] = arr[arr[:, self.cfg.COL_RAW_TS] >= start_time]

    def _filter_invalid_zeros(self, pd: PathData):
        """Remove rows with 0.0 value for specific signals (Heart Rate, etc)."""
        for ts_name in self.cfg.TIMESERIES_ZERO_IS_INVALID:
            if ts_name in pd.timeseries:
                arr = pd.timeseries[ts_name]
                # Keep rows where value != 0
                arr = arr[arr[:, self.cfg.COL_VALUE] != 0.0]
                pd.timeseries[ts_name] = arr

    def _validate_and_clean(self, pd: PathData, expected_ts: List[str]) -> None:
        """
        Check which expected timeseries are missing or empty.
        Remove empty keys from pd.timeseries.
        Record issues in validation_report.
        """
        issues = []
        
        # 1. Check for completely missing files
        missing_files = [ts for ts in expected_ts if ts not in pd.timeseries]
        if missing_files:
            issues.append(f"Path {pd.path_id}: Missing files {missing_files}")

        # 2. Check for invalid scalars (must have exactly 1 value)
        invalid_scalars = []
        empty_scalars = []
        for name, val in pd.scalar_features.items():
            if val.size > 1:
                invalid_scalars.append(name)
            elif val.size == 1:
                # Convert to pure python float
                pd.scalar_features[name] = float(val.item())
            else:
                # Empty (size 0), valid but unusable, remove silently
                empty_scalars.append(name)
        
        for k in invalid_scalars:
            # Instead of deleting, take the first value
            first_val = float(pd.scalar_features[k].item(0))
            pd.scalar_features[k] = first_val
            issues.append(f"Path {pd.path_id}: Scalar feature '{k}' has invalid size > 1 (took first value)")

        for k in empty_scalars:
            del pd.scalar_features[k]
            issues.append(f"Path {pd.path_id}: Scalar feature '{k}' is empty (removed)")

        # 3. Check for arrays that are too short for stat processing (< 3)
        empty_keys = []
        for ts_name, arr in pd.timeseries.items():
            if len(arr) == 0:
                empty_keys.append(ts_name)
        
        for k in empty_keys:
            del pd.timeseries[k]
            issues.append(f"Path {pd.path_id}: Timeseries '{k}' is empty (removed)")
            
        if issues:
            if pd.user_id not in self.validation_report:
                self.validation_report[pd.user_id] = []
            self.validation_report[pd.user_id].extend(issues)
