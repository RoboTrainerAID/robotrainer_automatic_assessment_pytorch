"""
Feature extraction logic for timeseries data.
Converts raw timeseries into a flat feature vector per path.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import welch, correlate
from typing import Dict, List, Any, Callable, Tuple
import warnings

from automatic_assessment.dataset.timeseries_loader import TimeseriesDataset, PathData
import automatic_assessment.dataset.config as config

class TimeseriesFeatureExtractor:
    """
    Extracts scalar features from time-series data based on config definitions.
    """

    def __init__(self, dataset: TimeseriesDataset) -> None:
        self.dataset = dataset
        self.features_df = pd.DataFrame()

    def extract_features(self) -> pd.DataFrame:
        """
        Iterates through the dataset and extracts features for every path.
        Returns a pandas DataFrame where each row is a path.
        """
        rows = []

        print("Extracting features...")
        for user_id, paths in self.dataset.items():
            for path_id, pd_data in paths.items():
                
                # Base metadata
                row = {
                    "user": user_id,
                    "path": path_id,
                }

                # 1. Add Scalar Features (already computed or loaded single values)
                for name, val in pd_data.scalar_features.items():
                    row[name] = val

                # 2. Extract Statistical Features per Timeseries
                ts_features = self._compute_ts_features(pd_data)
                row.update(ts_features)

                # 3. Extract Correlation Features (with robust sync)
                corr_features = self._compute_correlations(pd_data)
                row.update(corr_features)

                # 4. Extract Time Delay Features
                delay_features = self._compute_time_delays(pd_data)
                row.update(delay_features)

                rows.append(row)

        self.features_df = pd.DataFrame(rows)
        if not self.features_df.empty:
            print(f"  Extracted {self.features_df.shape[1]} features for {self.features_df.shape[0]} paths.")
        return self.features_df

    def save_features_to_csv(self, output_path: str = None) -> None:
        """
        Saves the extracted features Dataframe to CSV.
        """
        if output_path is None:
            output_path = config.CSV_OUTPUT_PATH
        
        if self.features_df.empty:
            print("Warning: No features to save.")
            return

        print(f"Saving features to {output_path}...")
        # Sort columns roughly alphabetically but keep IDs first
        cols = list(self.features_df.columns)
        if "user" in cols: cols.remove("user")
        if "path" in cols: cols.remove("path")
        cols = sorted(cols)
        final_cols = ["user", "path"] + cols
        
        # Determine cols that actually exist
        final_cols = [c for c in final_cols if c in self.features_df.columns]
        
        self.features_df[final_cols].to_csv(output_path, index=False)

    def _compute_ts_features(self, pd_data: PathData) -> Dict[str, float]:
        """
        Computes robust statistics for specific timeseries defined in config.TS_FEATURES.
        """
        feat_dict = {}

        for ts_name, feature_list in config.TS_FEATURES.items():
            if ts_name not in pd_data.timeseries:
                continue

            # Extract value column (Assumes index 2 is value)
            arr = pd_data.timeseries[ts_name]
            if arr.shape[0] == 0:
                continue
            
            timestamps = arr[:, config.COL_RAW_TS]
            values = arr[:, config.COL_VALUE]

            # sampling rate estimation for freq domain features
            fs = 1.0
            if len(timestamps) > 1:
                duration = timestamps[-1] - timestamps[0]
                if duration > 0:
                    fs = (len(timestamps) - 1) / duration

            for f_name in feature_list:
                cleaned_name = f"{ts_name}_{f_name}"
                val = np.nan

                try:
                    if f_name == "trimmed_mean":
                        val = self._feat_trimmed_mean(values)
                    elif f_name == "abs_mean":
                        val = np.mean(np.abs(values))
                    elif f_name == "std":
                        val = np.std(values)
                    elif f_name == "rms":
                        val = np.sqrt(np.mean(values**2))
                    elif f_name == "p95":
                        val = np.percentile(values, 95)
                    elif f_name == "p05":
                        val = np.percentile(values, 5)
                    
                    # --- Integral Features ---
                    elif f_name in ["energy", "integral_squared"]:
                        # Signal Energy: Integral of squared signal
                        # Unit: [Unit^2 * s]
                        if len(timestamps) > 1:
                            dt = np.mean(np.diff(timestamps))
                            val = np.sum(values**2) * dt
                        else:
                            # print("## energy ##")
                            # print(f"Warning: Not enough timestamps to compute energy for {cleaned_name}.")
                            # print(f"  Timestamps: {timestamps}")
                            val = 0.0
                            
                    elif f_name in ["integral", "impulse", "work", "total_distance"]:
                        # Signed Integral: Area under curve
                        # Unit: [Unit * s]
                        # Called 'impulse' for Force, 'work' for Power, 'displacement' for Velocity
                        if len(timestamps) > 1:
                            dt = np.mean(np.diff(timestamps))
                            val = np.sum(values) * dt
                        else:
                            # print("## integral ##")
                            # print(f"Warning: Not enough timestamps to compute integral for {cleaned_name}.")
                            # print(f"  Timestamps: {timestamps}")
                            val = 0.0
                            
                    elif f_name in ["absolute_integral"]:
                        # Absolute Integral: Area under absolute curve
                        # Unit: [Unit * s]
                        if len(timestamps) > 1:
                            dt = np.mean(np.diff(timestamps))
                            val = np.sum(np.abs(values)) * dt
                        else:
                            # print("## absolute_integral ##")
                            # print(f"Warning: Not enough timestamps to compute absolute integral for {cleaned_name}.")
                            # print(f"  Timestamps: {timestamps}")
                            val = 0.0

                    elif f_name == "rms_diff":
                        # Smoothness proxy
                        diffs = np.diff(values)
                        val = np.sqrt(np.mean(diffs**2))
                    elif f_name == "band_power_voluntary":
                        val = self._feat_band_power(values, fs, config.VOLUNTARY_BAND)
                    elif f_name == "band_power_tremor":
                        val = self._feat_band_power(values, fs, config.TREMOR_BAND)
                
                except Exception as e:
                    # Ignore calc errors, leave as NaN
                    pass

                feat_dict[cleaned_name] = val
        
        return feat_dict

    def _compute_correlations(self, pd_data: PathData) -> Dict[str, float]:
        """
        Computes Pearson correlation for pairs defined in config.CORRELATION_FEATURES.
        Uses robust synchronization (interpolation) to handle signals of different frequencies.
        """
        feat_dict = {}

        for (name_a, name_b) in config.CORRELATION_FEATURES:
            out_key = f"corr_{name_a}_vs_{name_b}"
            
            if name_a not in pd_data.timeseries or name_b not in pd_data.timeseries:
                feat_dict[out_key] = np.nan
                continue

            arr_a = pd_data.timeseries[name_a]
            arr_b = pd_data.timeseries[name_b]

            # Synchronize signals to the same time basis
            vals_a, vals_b, fs = self._synchronize_signals(arr_a, arr_b)

            if len(vals_a) < 3:
                # feat_dict[out_key] = np.nan
                feat_dict[out_key] = 0.0
                continue

            try:
                # Add constant check to avoid warnings
                if np.std(vals_a) == 0 or np.std(vals_b) == 0:
                     corr = 0.0
                else:
                    corr, _ = pearsonr(vals_a, vals_b)
                feat_dict[out_key] = corr
            except Exception as e:
                feat_dict[out_key] = np.nan
                print(f"Warning: Correlation computation failed for {out_key}.")
                print(f"  Error: {e}")


        return feat_dict

    def _compute_time_delays(self, pd_data: PathData) -> Dict[str, float]:
        """
        Computes the time delay (lag) between two signals using cross-correlation.
        Positive delay means Signal B follows Signal A.
        """
        feat_dict = {}

        for (name_a, name_b) in config.TIME_DELAY_FEATURES:
            out_key = f"delay_{name_a}_to_{name_b}"
            
            if name_a not in pd_data.timeseries or name_b not in pd_data.timeseries:
                feat_dict[out_key] = np.nan
                continue

            arr_a = pd_data.timeseries[name_a]
            arr_b = pd_data.timeseries[name_b]
            
            # 1. Synchronize to common time basis (Reference is usually lower freq)
            vals_a, vals_b, fs = self._synchronize_signals(arr_a, arr_b)
            
            if len(vals_a) < 10 or fs is None:
                # feat_dict[out_key] = np.nan
                feat_dict[out_key] = 0.0
                continue
                
            # 2. Normalize (Zero Mean) - Vital for cross-correlation
            sig_a = vals_a - np.mean(vals_a)
            sig_b = vals_b - np.mean(vals_b)
            
            if np.std(sig_a) == 0 or np.std(sig_b) == 0:
                feat_dict[out_key] = 0.0
                continue
            
            # 3. Cross-Correlation
            # mode='full' returns correlation at all shifts
            xcorr = correlate(sig_b, sig_a, mode='full', method='auto')
            lags = np.arange(-(len(sig_a) - 1), len(sig_a))
            
            # 4. Find Lag at Max Correlation
            max_idx = np.argmax(xcorr)
            lag_samples = lags[max_idx]
            
            # 5. Convert to Time (seconds)
            time_delay = lag_samples / fs
            
            feat_dict[out_key] = time_delay
            
        return feat_dict

    def _synchronize_signals(self, arr_a: np.ndarray, arr_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Synchronizes two signals and returns the sampling frequency of the reference (time basis).
        """
        if arr_a.shape[0] == 0 or arr_b.shape[0] == 0:
             return np.array([]), np.array([]), None
        
        # Determine Reference (shorter length / lower freq implied)
        if arr_a.shape[0] <= arr_b.shape[0]:
            ref, sec = arr_a, arr_b
            swapped = False
        else:
            ref, sec = arr_b, arr_a
            swapped = True
            
        ref_times = ref[:, config.COL_RAW_TS]
        sec_times = sec[:, config.COL_RAW_TS]
        
        ref_vals = ref[:, config.COL_VALUE]
        sec_vals = sec[:, config.COL_VALUE]
        
        # Interpolate Secondary to Reference Times
        aligned_sec_vals = np.interp(ref_times, sec_times, sec_vals)
        
        # Estimate Frequency of Reference
        fs = 1.0
        if len(ref_times) > 1:
            duration = ref_times[-1] - ref_times[0]
            if duration > 0:
                fs = (len(ref_times) - 1) / duration
        
        if not swapped:
            return ref_vals, aligned_sec_vals, fs
        else:
            return aligned_sec_vals, ref_vals, fs

    # --- Feature Implementations ---

    def _feat_trimmed_mean(self, values: np.ndarray, proportion: float = 0.1) -> float:
        """Calculates trimmed mean of ABSOLUTE values excluding the top and bottom proportion/2."""
        # Use absolute values for magnitude estimation
        abs_values = np.abs(values)
        
        lower = np.percentile(abs_values, proportion * 50)
        upper = np.percentile(abs_values, 100 - (proportion * 50))
        
        mask = (abs_values >= lower) & (abs_values <= upper)
        if not np.any(mask):
            return float(np.mean(abs_values))
            
        return float(np.mean(abs_values[mask]))

    def _feat_band_power(self, values: np.ndarray, fs: float, band: tuple) -> float:
        """Calculates Average Spectral Power in a frequency band using Welch's method."""
        # if fs <= 0: return np.nan
        if fs <= 0: return 0.0
        
        # Welch's method
        nperseg = min(len(values), 256)
        freqs, psd = welch(values, fs, nperseg=nperseg)
        
        # Find indices
        idx_min = np.argmax(freqs >= band[0])
        idx_max = np.argmax(freqs > band[1])
        if idx_max == 0: idx_max = len(freqs) # If band goes beyond Nyquist
        
        if idx_min >= idx_max:
             return 0.0
             
        # Average power in band (integrate PSD)
        band_power = np.trapz(psd[idx_min:idx_max], freqs[idx_min:idx_max])
        return float(band_power)
