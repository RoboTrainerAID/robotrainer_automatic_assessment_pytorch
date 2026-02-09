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


@dataclass
class PathData:
    """All loaded (and preprocessed) timeseries for a single path."""
    user_id: str
    path_id: str
    meta: dict = field(default_factory=dict)
    timeseries: Dict[str, np.ndarray] = field(default_factory=dict)
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
                       TIMESERIES_ZERO_IS_INVALID, COL_RAW_TS, COL_VALUE
        """
        self.cfg = config
        self.data: Dict[str, Dict[str, PathData]] = {}
        # Stores validation issues: user_id -> list of issue messages
        self.validation_report: Dict[str, List[str]] = {}

    def load_all(self) -> None:
        """Discover and load all users and paths."""
        root = Path(self.cfg.DATASET_ROOT).resolve()
        if not root.exists():
            print(f"Error: Dataset root not found at {root}")
            return

        user_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("U")])
        
        for user_dir in user_dirs:
            user_id = user_dir.name
            self.data[user_id] = {}
            
            # Sort paths numerically (path_1, path_2, ...)
            path_dirs = sorted(
                [d for d in user_dir.iterdir() if d.is_dir() and d.name.startswith("path_")],
                key=lambda p: int(p.name.split("_")[1]) if "_" in p.name and p.name.split("_")[1].isdigit() else 0
            )

            for path_dir in path_dirs:
                path_id = path_dir.name
                pd = self._load_path(user_id, path_id, path_dir)
                if pd:
                    self.data[user_id][path_id] = pd
            
            print(f"Loaded {user_id}: {len(self.data[user_id])} paths")
            
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

    def _load_path(self, user_id: str, path_id: str, path_dir: Path) -> Optional[PathData]:
        pd = PathData(user_id=user_id, path_id=path_id)
        
        # Load Meta
        meta_path = path_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                pd.meta = json.load(f)

        # Determine expected timeseries
        path_num = int(path_id.split("_")[1]) if "_" in path_id else 0
        expected_ts = self.cfg.TIMESERIES_TO_LOAD.copy()
        
        if path_num in self.cfg.PATHS_WITHOUT_DISTURBANCE:
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

        # 1. Trim to motion start
        self._trim_path_data(pd)
        
        # 2. Filter invalid zeros (restore from memory)
        self._filter_invalid_zeros(pd)

        # 3. Final validation (check for empty or constant signals)
        self._validate_and_clean(pd, expected_ts)
        
        # Always return the path object, even if partially empty
        return pd

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
            issues.append(f"Query {pd.path_id}: Missing files {missing_files}")

        # 2. Check for empty arrays (e.g. after trim or zero-removal)
        empty_keys = []
        for ts_name, arr in pd.timeseries.items():
            if len(arr) == 0:
                empty_keys.append(ts_name)
        
        for k in empty_keys:
            del pd.timeseries[k]
            issues.append(f"Query {pd.path_id}: Data for '{k}' is empty (removed)")
            
        if issues:
            if pd.user_id not in self.validation_report:
                self.validation_report[pd.user_id] = []
            self.validation_report[pd.user_id].extend(issues)
