import numpy as np
import pandas as pd
from tqdm import tqdm  # For progress visualization given the loop intensity
from scipy.stats import skew, kurtosis
from scipy.signal import periodogram
import antropy as ant

def safe_std(x):
    return np.std(x) + 1e-12


def safe_mean(x):
    return np.mean(x) + 1e-12


def rms(x):
    return np.sqrt(np.mean(x**2))


def coefficient_of_variation(x):
    return safe_std(x) / np.abs(safe_mean(x))


def autocorr_decay_time(x, max_lag=50):
    """
    Time until autocorrelation decays below 1/e.
    Proxy for motor correction timescale.
    """
    x = x - np.mean(x)
    ac = np.correlate(x, x, mode="full")[len(x)-1:]
    
    # Fix: Handle constant signal (variance=0 -> ac[0]=0)
    if ac[0] < 1e-12:
        return 0

    ac /= ac[0]

    for lag in range(1, min(len(ac), max_lag)):
        if ac[lag] < 1 / np.e:
            return lag
    return max_lag

def extract_timeseries_features(x, fs=1.0):
    """
    Tier-1 features: highest ROI for LOSO + small N.
    Input:
        x : np.ndarray, shape (T,)
    Output:
        dict of scalar features
    """
    feats = {}

    # -------------------------
    # 1. Basic statistics
    # -------------------------
    feats["mean"] = np.mean(x)
    feats["std"] = np.std(x)
    feats["p95"] = np.percentile(x, 95)
    feats["p05"] = np.percentile(x, 5)

    # -------------------------
    # 2. Shape statistics
    # -------------------------
    # Fix: Handle constant signals to avoid division by zero in skew/kurtosis
    if feats["std"] < 1e-9:
        feats["skew"] = 0.0
        feats["kurtosis"] = 0.0
    else:
        # Skew: Asymmetry (e.g., leaning more forward than backward)
        s = skew(x, nan_policy='omit')
        # Kurtosis: High = lots of sudden 'shocks' or stumbles
        k = kurtosis(x, nan_policy='omit')
        
        # Ensure finite values
        feats["skew"] = s if np.isfinite(s) else 0.0
        feats["kurtosis"] = k if np.isfinite(k) else 0.0

    # -------------------------
    # 3. Complexity / regularity
    # -------------------------
    # Sample Entropy: Measures "regularity". 
    # (Low = rigid/repetitive, High = adaptable/healthy or noisy)
    try:
        se = ant.sample_entropy(x, order=2)
        feats["sample_entropy"] = se if np.isfinite(se) else 0.0
    except Exception:
        feats["sample_entropy"] = 0.0

    # -------------------------
    # 4. Temporal persistence
    # -------------------------
    feats["autocorr_decay"] = autocorr_decay_time(x)

    # -------------------------
    # 5. Smoothness / motor control
    # -------------------------
    # Log Dimensionless Jerk (approximate version)
    dt = 1.0 / fs
    jerk = np.diff(x, n=2) / (dt ** 2)
    
    jerk_energy = np.sum(jerk**2)
    if jerk_energy <= 0:
        feats["log_jerk"] = -10.0 # Arbitrary low value for "no jerk"
    else:
        feats["log_jerk"] = np.log(jerk_energy + 1e-10)

    # -------------------------
    # 6. Energy / effort proxies
    # -------------------------
    feats["signal_energy"] = np.sum(x**2)
    feats["rms"] = rms(x)

    # -------------------------
    # 7. Spectral features
    # -------------------------
    try:
        f, Pxx = periodogram(x, fs=fs)

        voluntary_band = (f >= 0.1) & (f <= 2.0)
        tremor_band = (f >= 3.0) & (f <= 10.0)

        power_voluntary = np.sum(Pxx[voluntary_band])
        power_tremor = np.sum(Pxx[tremor_band])

        # Power in "Voluntary Motion" band (0.1 - 2 Hz)
        feats["power_voluntary"] = power_voluntary
        # Power in "Tremor" band (3 - 10 Hz)
        feats["power_tremor"] = power_tremor
        # Ratio: How much of the signal is jitter vs intentional?
        
        # Fix: Cap ratio to avoid huge values or inf
        denom = power_voluntary + 1e-12
        ratio = power_tremor / denom
        if not np.isfinite(ratio) or ratio > 1e6:
            ratio = 1e6
        feats["tremor_ratio"] = ratio

    except Exception:
        feats["power_voluntary"] = 0.0
        feats["power_tremor"] = 0.0
        feats["tremor_ratio"] = 0.0

    dx = np.diff(x)

    # -------------------------
    # 1. CV of absolute corrections
    # -------------------------
    cv = coefficient_of_variation(np.abs(dx))
    feats["cv_abs_diff"] = cv if np.isfinite(cv) else 0.0

    # -------------------------
    # 2. RMS of corrections
    # -------------------------
    feats["rms_diff"] = rms(dx)

    # -------------------------
    # 3. Gini index (amplitude inequality)
    # -------------------------
    abs_x = np.abs(x) + 1e-12
    sorted_x = np.sort(abs_x)
    n = len(sorted_x)
    index = np.arange(1, n + 1)
    
    denom_gini = n * np.sum(sorted_x)
    if denom_gini == 0:
        feats["gini_amplitude"] = 0.0
    else:
        gini = (2 * np.sum(index * sorted_x)) / denom_gini - (n + 1) / n
        feats["gini_amplitude"] = gini if np.isfinite(gini) else 0.0

    # -------------------------
    # 4. Peak-to-RMS ratio (burstiness proxy)
    # -------------------------
    rms_val = rms(x)
    if rms_val < 1e-12:
        feats["peak_to_rms"] = 0.0
    else:
        feats["peak_to_rms"] = np.max(np.abs(x)) / rms_val

    return feats

def bounded_gaussian_jitter(
    x,
    noise_ratio=0.05,
    clip_ratio=0.1,
    random_state=None
):
    """
    Apply bounded Gaussian jittering.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Original time-series
    noise_ratio : float
        Std of noise relative to signal range (default 5%)
    clip_ratio : float
        Max allowed perturbation relative to signal range
    random_state : int or None

    Returns
    -------
    x_aug : np.ndarray
        Augmented time-series
    """
    rng = np.random.default_rng(random_state)

    signal_range = np.max(x) - np.min(x)
    if signal_range == 0: signal_range = 1.0 # Handle flat signals
    sigma = noise_ratio * signal_range
    clip_val = clip_ratio * signal_range

    noise = rng.normal(0.0, sigma, size=len(x))
    noise = np.clip(noise, -clip_val, clip_val)

    return x + noise

def user_augmentation(target_df, user_df, path_df, timeseries_df, augmentation_ratio, original_path_related_cols):
    """
    Augments the dataset by creating synthetic users via jittering.
    Ensures physical consistency by re-calculating TS features from jittered signals.
    """
    print(f"\n--- Starting Data Augmentation (Ratio: {augmentation_ratio}x) ---")
    
    new_users = []
    new_targets = []
    new_paths = []
    new_ts_list = []

    # Identify sensor columns (everything in TS that isn't metadata)
    ts_meta_cols = ['user', 'path', 'time']
    sensor_cols = [c for c in timeseries_df.columns if c not in ts_meta_cols]

    # Number of new copies to generate per user
    # If ratio is 3, we generate 3 NEW copies (total 4x data) 
    n_clones = int(augmentation_ratio)

    unique_users = user_df['user'].unique()
    
    for user_id in tqdm(unique_users, desc="Augmenting Users"):
        # 1. Get User Data
        u_row = user_df[user_df['user'] == user_id].iloc[0]
        t_row = target_df[target_df['user'] == user_id].iloc[0]
        
        # Get all paths and TS for this user
        u_paths = path_df[path_df['user'] == user_id]
        u_ts = timeseries_df[timeseries_df['user'] == user_id]

        for i in range(1, n_clones + 1):
            # 6. Create Consistent ID (e.g. 5 -> 501, 502)
            new_id = int(user_id * 100 + i)
            
            # --- A. User Data Augmentation ---
            # Jitter Age (integer), Keep Sex/ID-features constant
            aug_u_row = u_row.copy()
            aug_u_row['user'] = new_id
            
            if 'age' in aug_u_row:
                # Add integer noise to age, bounded reasonably
                age_noise = int(np.random.choice([-2, -1, 0, 1, 2]))
                aug_u_row['age'] = max(18, aug_u_row['age'] + age_noise)
            
            new_users.append(aug_u_row)

            # --- B. Target Augmentation ---
            # Regression targets get slight Gaussian jitter
            aug_t_row = t_row.copy()
            aug_t_row['user'] = new_id
            
            target_cols = [c for c in target_df.columns if c != 'user']
            for t_col in target_cols:
                val = aug_t_row[t_col]
                # 2% jitter relative to value magnitude
                noise = np.random.normal(0, 0.02 * abs(val) + 1e-6) 
                aug_t_row[t_col] = val + noise
            
            new_targets.append(aug_t_row)

            # --- C. Timeseries & Path Augmentation ---
            # We must process paths and TS together to maintain consistency
            
            # 1. Clone Time-Series first
            current_user_ts = u_ts.copy()
            current_user_ts['user'] = new_id
            
            # Apply Jitter to Sensors
            for sensor in sensor_cols:
                # Vectorized jittering per sensor column
                # Note: This jitters the whole column trace. 
                # Ideally, we jitter per path, but doing it per user-block is faster and safe
                current_user_ts[sensor] = bounded_gaussian_jitter(
                    current_user_ts[sensor].values, 
                    noise_ratio=0.05, # 5% noise for TS
                    clip_ratio=0.1
                )
            
            new_ts_list.append(current_user_ts)

            # 2. Clone Paths
            current_user_paths = u_paths.copy()
            current_user_paths['user'] = new_id
            
            # 3. Augment "Original" Path Cols (Meta-info like Length/Difficulty)
            for p_col in original_path_related_cols:
                if p_col in current_user_paths.columns and pd.api.types.is_numeric_dtype(current_user_paths[p_col]):
                    # Small jitter for metadata
                    jitter = np.random.normal(0, 0.01 * current_user_paths[p_col].mean(), size=len(current_user_paths))
                    current_user_paths[p_col] += jitter

            # 4. Re-calculate Extracted Features from the NEW Time-Series
            # This ensures that "mean_force" matches the jittered "force" trace.
            
            # We iterate over every path for this new user
            for idx, row in current_user_paths.iterrows():
                path_id = row['path']
                
                # Extract the jittered TS slice for this path
                ts_slice = current_user_ts[current_user_ts['path'] == path_id]
                
                # For each sensor, compute features and update path row
                for sensor in sensor_cols:
                    if len(ts_slice) == 0: continue
                    
                    # Extract features
                    feats = extract_timeseries_features(ts_slice[sensor].values)
                    
                    # Update columns in path_df
                    # Convention: {sensor}_{feature_name}
                    for feat_name, feat_val in feats.items():
                        col_name = f"{sensor}_{feat_name}"
                        if col_name in current_user_paths.columns:
                            current_user_paths.at[idx, col_name] = feat_val
            
            new_paths.append(current_user_paths)

    # 7. Merge and Return
    print("Concatenating augmented data...")
    aug_user_df = pd.concat([user_df, pd.DataFrame(new_users)], ignore_index=True)
    aug_target_df = pd.concat([target_df, pd.DataFrame(new_targets)], ignore_index=True)
    aug_path_df = pd.concat([path_df] + new_paths, ignore_index=True)
    aug_ts_df = pd.concat([timeseries_df] + new_ts_list, ignore_index=True)

    print(f"Augmentation Complete. New User Count: {len(aug_user_df)}")
    return aug_target_df, aug_user_df, aug_path_df, aug_ts_df

if __name__ == "__main__":
    # Original signal
    x_original = np.array([...])  # shape (T,)

    # Augmented version
    x_augmented = bounded_gaussian_jitter(
        x_original,
        noise_ratio=0.03,
        clip_ratio=0.08,
        random_state=42
    )

    # Extract features from augmented signal
    features_aug = extract_timeseries_features(x_augmented, fs=1.0)
