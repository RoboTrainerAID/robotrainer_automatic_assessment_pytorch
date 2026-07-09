"""
Configuration settings for the dataset preprocessing.
"""

from pathlib import Path

# Root folder of the numpy dataset. 
DATASET_FOLDER = Path("/data/raw/timeseries_numpy").resolve()
CSV_OUTPUT_PATH = Path("/data/raw/timeseries_features.csv").resolve()

# Columns in .npy files
COL_RAW_TS = 0
COL_REL_TS = 1
COL_VALUE = 2

# List of timeseries to load (Raw files)
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
    "duration",
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

# Lever arm length for torque to force conversion (in meters)
HANDLE_RADIUS = 0.35

# ============================================================
# Derived Timeseries Configuration
# ============================================================

# 1. Magnitude Calculations
# Assumes components within a group are already synchronized.
# Format: "output_name": {"x": "ts_x", "y": "ts_y", "rot": "ts_rot" (optional), "is_velocity": bool}
DERIVED_MAGNITUDES = {
    # --- User Force ---
    "user_force_lin_mag": {
        "x": "user_force_x", 
        "y": "user_force_y"
    },
    "user_force_total_mag": {
        "x": "user_force_x", 
        "y": "user_force_y", 
        "rot": "user_torque_z", 
        "is_velocity": False
    },
    
    # --- Robot Velocity ---
    "robot_vel_lin_mag": {
        "x": "robot_vel_x", 
        "y": "robot_vel_y"
    },
    "robot_vel_total_mag": {
        "x": "robot_vel_x", 
        "y": "robot_vel_y", 
        "rot": "robot_vel_rot_z", 
        "is_velocity": True
    },

    # --- Disturbance Force ---
    "disturbance_force_lin_mag": {
        "x": "disturbance_force_x", 
        "y": "disturbance_force_y"
    },
}

# 2. Power and Work Calculations
# Combines Force set and Velocity set which may differ in frequency.
# Format: "base_output_name": {"force_roots": [fx, fy, tz], "vel_roots": [vx, vy, wz]}
# Note: Roots are the keys to look up available components. 
DERIVED_POWER_WORK = {
    "user": {
        "force_x": "user_force_x", "force_y": "user_force_y", "torque_z": "user_torque_z",
        "vel_x": "robot_vel_x",    "vel_y": "robot_vel_y",    "vel_rot_z": "robot_vel_rot_z"
    }
}
# Outputs produced: "{base}_power", "{base}_work_cum"

# Helper to generate full list of expected loaded files (Raw + Derived)
def _get_all_expected_ts():
    all_ts = set(TIMESERIES_TO_LOAD)
    
    # Add Magnitude names
    for name in DERIVED_MAGNITUDES.keys():
        all_ts.add(name)
             
    # Add Power/Work names
    for base in DERIVED_POWER_WORK.keys():
        all_ts.add(f"{base}_power")
        all_ts.add(f"{base}_work_cum")
        
    return list(all_ts)

# Dynamic list of all allowed loadable files
ALL_VALID_TIMESERIES = _get_all_expected_ts()

# ============================================================
# Timeseries groups for the LEARNED-FEATURE models (CNN/LSTM)
# ============================================================
# Each group gets its own tensor at its own sampling rate and its own
# encoder in the models. Rationale:
# - Mechanical signals carry fast dynamics (force-field hits, tremor
#   band 3-10 Hz): keep high resolution. 200 Hz raw force is capped to
#   50 Hz (bin averaging) — Nyquist 25 Hz comfortably covers the
#   relevant bands while cutting tensor size 4x.
# - Physiological signals are natively ~1-2.2 Hz: 2 Hz preserves the
#   native HRV/PPI resolution (heart_rate at 1 Hz simply has masked
#   gaps on the 2 Hz grid).
# - Gait events occur about once per stride (~1 Hz): 2 Hz keeps
#   individual strides separate.
# Channels within a group share one time grid; groups are NOT aligned
# to each other (per-group learned features, no cross-group indexing).
# The statistical features (TS_FEATURES) always use the RAW series —
# this grouping only affects the tensors built for model training.
TS_MODEL_GROUPS = {
    "mechanical": {
        "rate_hz": 50.0,
        "channels": [
            "disturbance_force_lin_mag", "disturbance_force_x", "disturbance_force_y",
            "path_deviation_front", "path_deviation_left", "path_deviation_right",
            "robot_pos_theta", "robot_pos_x", "robot_pos_y",
            "robot_vel_lin_mag", "robot_vel_rot_z", "robot_vel_total_mag",
            "robot_vel_x", "robot_vel_y",
            "user_force_lin_mag", "user_force_total_mag", "user_force_x", "user_force_y",
            "user_power", "user_torque_z", "user_work_cum",
        ],
    },
    "physiological": {
        "rate_hz": 2.0,
        "channels": ["heart_rate", "hrv", "ppi"],
    },
    "gait": {
        "rate_hz": 2.0,
        "channels": [
            "left_stride_duration", "left_stride_length",
            "left_stride_stance_time", "left_stride_swing_time",
            "right_stride_duration", "right_stride_length",
            "right_stride_stance_time", "right_stride_swing_time",
        ],
    },
}


# ============================================================
# Data Augmentation (synthetic TRAINING users)
# ============================================================
# Clones are generated ONCE at split creation (framework/data/dataset.py,
# prepare_and_split_data) and ONLY for training users — test users never
# get clones, so no augmented information can leak into the hold-out.
#
# Semantics of a clone: a noisy RE-MEASUREMENT of the same person.
# Targets, demographics, and task difficulty stay strictly identical;
# only the RAW sensor channels receive noise. Derived series (magnitudes,
# power, work) and all statistical features are RE-COMPUTED from the
# noisy traces so the clone stays physically consistent.
#
# Clone user id = original_id * 100 + clone_index (1..max_ratio); the
# AugmentedLOGO splitter excludes a validation user's clones from the
# training fold. At experiment time, AssessmentDataset.view(...,
# augmentation_ratio=r) selects clones with index <= r, so a ratio sweep
# needs no data regeneration.
AUGMENTATION = {
    "max_ratio": 10,        # clones generated per training user at split creation
    "method": "jitter",    # currently implemented: "jitter" (extensible)
    "noise_factor": 0.05,  # sigma = noise_factor * robust per-(path,channel) std
    "seed": 42,            # base seed; each clone derives its own stream
}


def augmentation_positive_channels() -> set:
    """
    Channels that must stay strictly positive under augmentation
    (multiplicative/truncated noise instead of additive Gaussian):
    all physiological and gait-event channels.
    """
    positive = set()
    for name in ("physiological", "gait"):
        positive.update(TS_MODEL_GROUPS[name]["channels"])
    return positive


def validate_ts_groups() -> None:
    """Groups must exactly partition ALL_VALID_TIMESERIES (fail loudly)."""
    grouped = [ch for g in TS_MODEL_GROUPS.values() for ch in g["channels"]]
    duplicates = {ch for ch in grouped if grouped.count(ch) > 1}
    grouped_set = set(grouped)
    valid_set = set(ALL_VALID_TIMESERIES)
    problems = []
    if duplicates:
        problems.append(f"channels assigned to multiple groups: {sorted(duplicates)}")
    if grouped_set - valid_set:
        problems.append(f"grouped channels that are not valid timeseries: {sorted(grouped_set - valid_set)}")
    if valid_set - grouped_set:
        problems.append(f"valid timeseries missing from every group: {sorted(valid_set - grouped_set)}")
    if problems:
        raise ValueError("TS_MODEL_GROUPS configuration invalid:\n  - " + "\n  - ".join(problems))

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
    # per segment
    # NOTE: was "user_force_mag" before, which matched NO produced series
    # (silently skipped!) — the derived series is named "user_force_total_mag"
    # (Eq. 4 in the paper: linear force + tangential torque component).
    "user_force_total_mag": CORE_FEATURES + [
        "energy",             # integral_squared
        "impulse",  # integral (Total effort)
        "rms_diff",
        "band_power_voluntary",
        "band_power_tremor",
    ],

    # per segment
    "user_force_lin_mag": CORE_FEATURES + [
        "energy",             # integral_squared
        "impulse",  # integral (Total effort)
        "rms_diff",
        "band_power_voluntary",
        "band_power_tremor",
    ],

    # per segment
    "user_force_y": CORE_FEATURES + [
        "energy",             # integral_squared
        "impulse",  # integral (Total effort)
        "rms_diff",
        "band_power_voluntary",
        "band_power_tremor",
    ],

    # per segment
    "disturbance_force_lin_mag": [
        "trimmed_mean",
        "energy",
        "absolute_integral",
    ],

    # per segment
    "user_power": CORE_FEATURES + [
        "work",  # integral (Total effort)
    ],

    # --- Velocity Magnitude (derived) ---
    # per segment
    "robot_vel_lin_mag": CORE_FEATURES + [
        "rms_diff",
        "band_power_voluntary",
        "total_distance"
    ],
    
    # --- Path Deviation ---
    # per segment
    "path_deviation_front": CORE_FEATURES + [
        "absolute_integral"
    ],

    # per segment
    "path_deviation_left": CORE_FEATURES + [
        "absolute_integral"
    ],

    # --- Physiology ---
    "heart_rate": [
        "trimmed_mean"
    ],

    "ppi": [
        "trimmed_mean",
    ],

    "hrv": [
        "trimmed_mean",
        "std",
        "p95",
        "p05",
    ],

    # --- Stride parameters ---
    "left_stride_duration": ["trimmed_mean"],
    "left_stride_length": ["trimmed_mean"],
    "left_stride_stance_time": ["trimmed_mean"],
    "left_stride_swing_time": ["trimmed_mean"],
    "right_stride_duration": ["trimmed_mean"],
    "right_stride_length": ["trimmed_mean"],
    "right_stride_stance_time": ["trimmed_mean"],
    "right_stride_swing_time": ["trimmed_mean"],
}

# ============================================================
# Correlation feature pairs
# ============================================================

CORRELATION_FEATURES = [
    ("user_force_lin_mag", "path_deviation_front"),
    ("user_force_lin_mag", "robot_vel_lin_mag"),
    ("disturbance_force_lin_mag", "path_deviation_front"),
]

# ============================================================
# Time Delay / Lag feature pairs
# Format: ("Signal_A", "Signal_B") -> Calculates lag of B relative to A
# Positive lag means B happens AFTER A.
# ============================================================

TIME_DELAY_FEATURES = [
    ("disturbance_force_lin_mag", "user_force_lin_mag"),
    ("disturbance_force_lin_mag", "path_deviation_front"), # Inertial delay
    ("path_deviation_front", "user_force_lin_mag"),      # Reaction time
]

# ============================================================
# Spectral band definitions
# ============================================================

VOLUNTARY_BAND = (0.1, 2.0)
TREMOR_BAND = (3.0, 10.0)