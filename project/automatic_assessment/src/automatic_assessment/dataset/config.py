"""
Configuration settings for the dataset preprocessing.
"""

from pathlib import Path

# Root folder of the numpy dataset. 
DATASET_ROOT = Path("/data/raw/timeseries_numpy").resolve()
CSV_OUTPUT_PATH = Path("/data/raw/timeseries_features.csv").resolve()

# Columns in .npy files
COL_RAW_TS = 0
COL_REL_TS = 1
COL_VALUE = 2

# List of timeseries to load
TIMESERIES_TO_LOAD = [
    "disturbance_force_x",
    "disturbance_force_y",
    "heart_rate",
    "hrv",
    "path_deviation_front",
    "path_deviation_left",
    "path_deviation_right",
    # "ppg_ch0",
    # "ppg_ch1",
    # "ppg_ch2",
    # "ppg_ch3",
    "ppi",
    "robot_pos_theta",
    "robot_pos_x",
    "robot_pos_y",
    "robot_vel_rot_z",
    "robot_vel_x",
    "robot_vel_y",
    "user_force_x",
    "user_force_y",
    "user_torque_z",
    "left_stride_duration",
    "left_stride_length",
    "left_stride_stance_time",
    "left_stride_swing_time",
    "right_stride_duration",
    "right_stride_length",
    "right_stride_stance_time",
    "right_stride_swing_time",
]

DATA_WITH_A_SINGLE_VALUE_PER_PATH = [
    "cadence_avg",
    "left_num_strides",
    "right_num_strides",
]

# Timeseries to trim based on motion start
TIMESERIES_TRIMMED_BY_MOTION_START = [
    "disturbance_force_x",
    "disturbance_force_y",
    "path_deviation_front",
    "path_deviation_left",
    "path_deviation_right",
    # "ppg_ch0",
    # "ppg_ch1",
    # "ppg_ch2",
    # "ppg_ch3",
    "robot_pos_theta",
    "robot_pos_x",
    "robot_pos_y",
    "robot_vel_rot_z",
    "robot_vel_x",
    "robot_vel_y",
    "user_force_x",
    "user_force_y",
    "user_torque_z",
]

# Timeseries where 0.0 is invalid (e.g. sensor dropout)
TIMESERIES_ZERO_IS_INVALID = [
    "heart_rate",
    "hrv",
    "ppi",
]

# Paths that do not have disturbance forces
PATHS_WITHOUT_DISTURBANCE = [4, 11, 12, 13, 14]

# Velocity threshold to detect motion start
MOTION_START_THRESHOLD = 0.01

# ============================================================
# FEATURE EXTRACTION Core robust features (applied to most TS)
# ============================================================

CORE_FEATURES = [
    "trimmed_mean",
    "std",
    "p95",
    "p05",
    "rms",
]

# ============================================================
# Feature mapping per timeseries
# ============================================================

TS_FEATURES = {

    # --- Force Magnitudes (derived) ---
    "user_force_mag": CORE_FEATURES + [
        "energy",
        "rms_diff",
        "impulse",
        "band_power_voluntary",
        "band_power_tremor",
    ],

    "disturbance_force_mag": CORE_FEATURES + [
        "energy",
        "impulse",
    ],

    # --- Velocity Magnitude (derived) ---
    "robot_vel_mag": CORE_FEATURES + [
        "rms_diff",
        "band_power_voluntary",
    ],

    # --- Path Deviation ---
    "path_deviation_front": CORE_FEATURES + [
        "abs_mean",
        "band_power_voluntary",
    ],

    # --- Physiology ---
    "heart_rate": [
        "trimmed_mean",
        "std",
        "p95",
    ],

    "ppi": [
        "trimmed_mean",
        "std",
        "p95",
    ],

    # --- Stride parameters ---
    "left_stride_duration": ["trimmed_mean", "std"],
    "right_stride_duration": ["trimmed_mean", "std"],
}

# ============================================================
# Correlation feature pairs
# ============================================================

CORRELATION_FEATURES = [
    ("user_force_mag", "path_deviation_front"),
    ("user_force_mag", "robot_vel_mag"),
    ("disturbance_force_mag", "path_deviation_front"),
]

# ============================================================
# Spectral band definitions
# ============================================================

VOLUNTARY_BAND = (0.1, 2.0)
TREMOR_BAND = (3.0, 10.0)