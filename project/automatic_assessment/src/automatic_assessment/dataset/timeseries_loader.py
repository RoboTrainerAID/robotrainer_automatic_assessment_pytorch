"""
Data containers and loading logic for the timeseries dataset.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

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
        self.data: Dict[int, Dict[int, PathData]] = {} # user_id -> path_id -> PathData

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
            
            # print(f"Loaded User {user_id}: {len(self.data[user_id])} paths")

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
                    # Load as array first
                    arr = np.load(str(fpath))
                    
                    # If this file has timestamps (columns), extract only the value column.
                    if arr.ndim == 2 and arr.shape[1] > self.cfg.COL_VALUE:
                         arr = arr[:, self.cfg.COL_VALUE]

                    # Normalize scalar values immediately
                    if arr.size == 1:
                        pd.scalar_features[name] = float(arr.item())
                    elif arr.size > 1:
                        # Take first value for scalar features that accidentally have more
                        pd.scalar_features[name] = float(arr.item(0))
                        
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
        
        return pd

    def _preprocess_hrv(self, pd: PathData) -> None:
        """
        Keep only the first value column for HRV timeseries.
        HRV has data columns at indices 2 and 3; we keep 2 (plus timestamps 0, 1).
        """
        if "hrv" in pd.timeseries:
            arr = pd.timeseries["hrv"]
            if arr.shape[1] > 3:
                pd.timeseries["hrv"] = arr[:, :3]

    def _preprocess_ppg(self, pd: PathData) -> None:
        """
        Flatten PPG channels which contain multiple samples per row.
        """
        target_ts = ["ppg_ch0", "ppg_ch1", "ppg_ch2", "ppg_ch3"]
        for name in target_ts:
            if name not in pd.timeseries:
                continue
            
            arr = pd.timeseries[name]
            n_cols = arr.shape[1]
            if n_cols <= 2:
                continue
            
            timestamps = arr[:, :2]
            n_vals = n_cols - 2
            
            timestamps_expanded = np.repeat(timestamps, n_vals, axis=0)
            values = arr[:, 2:]
            values_flattened = values.reshape(-1, 1)
            
            pd.timeseries[name] = np.hstack((timestamps_expanded, values_flattened))

    def _trim_path_data(self, pd: PathData):
        """Trim loaded timeseries based on robot_vel_x threshold."""
        if "robot_vel_x" not in pd.timeseries:
            return

        vel_x = pd.timeseries["robot_vel_x"]
        mask = np.abs(vel_x[:, self.cfg.COL_VALUE]) > self.cfg.MOTION_START_THRESHOLD
        if not np.any(mask):
            return 

        start_idx = np.argmax(mask)
        start_time = vel_x[start_idx, self.cfg.COL_RAW_TS]
        pd.motion_start_timestamp = float(start_time)
        
        # Calculate new duration based on sliced times
        new_duration = 0.0

        for ts_name in self.cfg.TIMESERIES_TRIMMED_BY_MOTION_START:
            if ts_name in pd.timeseries:
                arr = pd.timeseries[ts_name]
                trimmed_arr = arr[arr[:, self.cfg.COL_RAW_TS] >= start_time]
                pd.timeseries[ts_name] = trimmed_arr
                
                # Update duration if this ts is robot_vel_x as reference
                if ts_name == "robot_vel_x" and trimmed_arr.size > 0:
                    times = trimmed_arr[:, self.cfg.COL_RAW_TS]
                    new_duration = float(times[-1] - times[0])
        
        # Update meta if we found a valid duration
        if new_duration > 0 and pd.meta:
            pd.meta["duration"] = new_duration
            pd.meta["trimmed_start_timestamp"] = float(start_time)

    def _filter_invalid_zeros(self, pd: PathData):
        """Remove rows with 0.0 value for specific signals (Heart Rate, etc)."""
        for ts_name in self.cfg.TIMESERIES_ZERO_IS_INVALID:
            if ts_name in pd.timeseries:
                arr = pd.timeseries[ts_name]
                arr = arr[arr[:, self.cfg.COL_VALUE] != 0.0]
                pd.timeseries[ts_name] = arr
