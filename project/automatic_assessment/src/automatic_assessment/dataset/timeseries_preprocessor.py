"""
Logic for cleaning and trimming the timeseries dataset.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Any
from automatic_assessment.dataset.timeseries_loader import PathData, TimeseriesDataset


class TimeseriesPreprocessor:
    """
    Trims and cleans the loaded dataset.
    """

    def __init__(self, dataset: TimeseriesDataset, config: Any) -> None:
        """
        :param dataset: The loaded TimeseriesDataset object.
        :param config: Configuration object.
        """
        self.dataset = dataset
        self.cfg = config

    def process(self) -> None:
        """
        Applies cleaning steps to the entire dataset (in-place).
        1. Fixes raw format quirks (HRV, PPG).
        2. Trims data based on motion start.
        3. Filters invalid zero values.
        """
        print("Preprocessing dataset (Formatting, Trimming & Filtering)...")
        count = 0
        for user_id, paths in self.dataset.items():
            for path_id, pd in paths.items():
                self._preprocess_hrv(pd)
                self._preprocess_ppg(pd)
                self._trim_path_data(pd)
                self._filter_invalid_zeros(pd)
                count += 1
        # print(f"  Processed {count} paths.")

    def _preprocess_hrv(self, pd: PathData) -> None:
        """
        Keep only the first value column for HRV timeseries.
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
                # Slice logic
                trimmed_arr = arr[arr[:, self.cfg.COL_RAW_TS] >= start_time]
                
                if trimmed_arr.size > 0:
                    # Note: We do NOT shift relative timestamps to maintain synchronization
                    pd.timeseries[ts_name] = trimmed_arr
                
                    # Update duration if this ts is robot_vel_x as reference
                    if ts_name == "robot_vel_x":
                        # Duration is the span of validity (end - start of trimmed region)
                        start_rel = trimmed_arr[0, self.cfg.COL_REL_TS]
                        end_rel = trimmed_arr[-1, self.cfg.COL_REL_TS]
                        new_duration = float(end_rel - start_rel)
        
        # Update scalar features if we found a valid duration
        if new_duration > 0:
            # Store as scalar feature
            pd.scalar_features["duration"] = new_duration
            
            # Remove from meta if present to avoid duplication/confusion
            if pd.meta and "duration" in pd.meta:
                del pd.meta["duration"]

            if pd.meta is not None:
                pd.meta["trimmed_start_timestamp"] = float(start_time)

    def _filter_invalid_zeros(self, pd: PathData):
        """Remove rows with 0.0 value for specific signals (Heart Rate, etc)."""
        for ts_name in self.cfg.TIMESERIES_ZERO_IS_INVALID:
            if ts_name in pd.timeseries:
                arr = pd.timeseries[ts_name]
                arr = arr[arr[:, self.cfg.COL_VALUE] != 0.0]
                pd.timeseries[ts_name] = arr
