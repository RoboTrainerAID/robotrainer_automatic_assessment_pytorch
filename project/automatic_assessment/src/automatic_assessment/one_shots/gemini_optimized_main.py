import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
MAX_TIMESTEPS = 300  # As per your domain knowledge (paths are 50-300 steps)
PAD_VALUE = 0.0

def prepare_data(ts_df, path_df, user_df, target_df):
    """
    Transforms raw dataframes into hierarchical tensors.
    """
    # 1. Identify Feature Columns
    # Exclude ID columns to get pure features
    ts_feat_cols = [c for c in ts_df.columns if c not in ['user', 'path', 'time']]
    path_feat_cols = [c for c in path_df.columns if c not in ['user', 'path']]
    user_feat_cols = [c for c in user_df.columns if c not in ['user']]
    target_cols = [c for c in target_df.columns if c not in ['user']]
    
    unique_users = sorted(user_df['user'].unique())
    
    # Storage
    X_ts = []    # Shape: (N_users, 20, MAX_TIME, N_ts_feats)
    X_path = []  # Shape: (N_users, 20, N_path_feats)
    X_user = []  # Shape: (N_users, N_user_feats)
    Y = []       # Shape: (N_users, N_targets)
    
    print(f"Processing {len(unique_users)} users...")
    
    for u_id in unique_users:
        # --- A. User Static Features ---
        u_static = user_df[user_df['user'] == u_id][user_feat_cols].values
        if len(u_static) == 0: raise ValueError(f"User {u_id} missing in user_df")
        X_user.append(u_static[0])
        
        # --- B. Targets ---
        targs = target_df[target_df['user'] == u_id][target_cols].values
        if len(targs) == 0: raise ValueError(f"User {u_id} missing in target_df")
        Y.append(targs[0])
        
        # --- C. Path & Timeseries Loop (Paths 1 to 20) ---
        user_ts_list = []
        user_path_list = []
        
        # Filter for this user once to speed up
        u_ts_df_all = ts_df[ts_df['user'] == u_id]
        u_path_df_all = path_df[path_df['user'] == u_id]
        
        for p_id in range(1, 21): # Paths 1..20
            # 1. Path Static
            p_static = u_path_df_all[u_path_df_all['path'] == p_id][path_feat_cols].values
            if len(p_static) == 0:
                # Fallback if path missing (should not happen based on N=560)
                p_static = np.zeros(len(path_feat_cols))
            else:
                p_static = p_static[0]
            user_path_list.append(p_static)
            
            # 2. Timeseries
            # Sort by time to ensure temporal order
            p_ts = u_ts_df_all[u_ts_df_all['path'] == p_id].sort_values('time')[ts_feat_cols].values
            
            # Padding / Truncating
            L = len(p_ts)
            if L >= MAX_TIMESTEPS:
                p_ts_padded = p_ts[:MAX_TIMESTEPS, :]
            else:
                # Pad with zeros at the end
                padding = np.zeros((MAX_TIMESTEPS - L, len(ts_feat_cols)))
                p_ts_padded = np.vstack([p_ts, padding])
                
            user_ts_list.append(p_ts_padded)
            
        X_ts.append(np.stack(user_ts_list))      # (20, 300, 50)
        X_path.append(np.stack(user_path_list))  # (20, 24)
        
    # Convert to Numpy
    return (np.array(X_ts, dtype=np.float32), 
            np.array(X_path, dtype=np.float32), 
            np.array(X_user, dtype=np.float32), 
            np.array(Y, dtype=np.float32))

# --- EXECUTION ---
# Assuming you have your dataframes loaded as: ts_df, path_df, user_df, target_df
# X_ts, X_path, X_user, Y = prepare_data(ts_df, path_df, user_df, target_df)

import torch.nn as nn
import torch.nn.functional as torch_functional

class HierarchicalMILNet(nn.Module):
    def __init__(self, ts_feats, path_feats, user_feats, n_targets):
        super().__init__()
        
        # --- 1. Path Encoder (Time-Distributed 1D CNN) ---
        # Input: (Batch * 20, ts_feats, Time)
        self.conv1 = nn.Conv1d(ts_feats, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Global Max Pooling effectively handles the "did a specific event occur?" question
        # Output becomes (Batch * 20, 64)
        
        # --- 2. Path Fusion (CNN embedding + Static Path Feats) ---
        self.fusion_dim = 64 + path_feats
        self.path_fc = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4) # High dropout for small data
        )
        
        # --- 3. Attention Aggregation (MIL) ---
        # Gated Attention Mechanism (Ilse et al. 2018)
        self.att_v = nn.Linear(64, 32)
        self.att_u = nn.Linear(64, 32)
        self.att_weights = nn.Linear(32, 1)
        
        # --- 4. User Encoder (Aggregated Path Vec + User Static) ---
        self.final_dim = 64 + user_feats
        
        self.regressor = nn.Sequential(
            nn.Linear(self.final_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, n_targets)
        )

    def forward(self, x_ts, x_path, x_user):
        """
        x_ts:   (Batch, 20, Time, Feats)
        x_path: (Batch, 20, Path_Feats)
        x_user: (Batch, User_Feats)
        """
        B, P, T, F = x_ts.size()
        
        # A. Path Encoding (Merge Batch & Path dims)
        x = x_ts.view(B * P, T, F).permute(0, 2, 1) # -> (B*P, F, T)
        
        h = torch_functional.relu(self.bn1(self.conv1(x)))
        h = torch_functional.relu(self.bn2(self.conv2(h)))
        
        # Global Max Pooling over time
        h = torch_functional.max_pool1d(h, kernel_size=h.size(2)).squeeze(2) # -> (B*P, 64)
        
        # B. Fusion with Path Static Features
        x_path_flat = x_path.view(B * P, -1)
        h_fused = torch.cat([h, x_path_flat], dim=1) # -> (B*P, 64 + Path_Feats)
        h_fused = self.path_fc(h_fused) # -> (B*P, 64)
        
        # Reshape back to (Batch, 20, 64)
        h_fused = h_fused.view(B, P, -1)
        
        # C. Attention Aggregation
        # Learn which paths are important for this specific user
        a_v = torch.tanh(self.att_v(h_fused))
        a_u = torch.sigmoid(self.att_u(h_fused))
        a_w = self.att_weights(a_v * a_u) # (B, 20, 1)
        a_score = torch.softmax(a_w, dim=1)
        
        user_vec = torch.sum(a_score * h_fused, dim=1) # (B, 64)
        
        # D. User Level Fusion & Output
        final_in = torch.cat([user_vec, x_user], dim=1)
        out = self.regressor(final_in)
        
        return out
    

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Helper for Augmentation ---
def add_noise_and_scale(tensor, noise_level=0.05, scale_range=(0.9, 1.1)):
    """
    Augmentation: Randomly scale signal magnitude and add Gaussian noise.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
        
    # 1. Scaling (simulates lighter/heavier users or different floor friction)
    # Apply random scaling 50% of the time
    if torch.rand(1).item() > 0.5:
        # Create scale factor on the same device as input
        scale_factor = (scale_range[0] - scale_range[1]) * torch.rand(1, device=tensor.device) + scale_range[1]
        tensor = tensor * scale_factor
        
    # 2. Add Gaussian Noise (simulates sensor jitter)
    noise = torch.randn_like(tensor) * noise_level
    return tensor + noise

# --- Main Training Loop ---
def train_loocv(X_ts: np.ndarray, X_path: np.ndarray, X_user: np.ndarray, Y: np.ndarray, epochs: int = 150, lr: float = 0.001):
    n_users = len(Y)
    all_preds_scaled = []
    all_actuals_scaled = []
    
    # Dimensions
    ts_feats = X_ts.shape[3]
    path_feats = X_path.shape[2]
    user_feats = X_user.shape[1]
    n_targets = Y.shape[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting LOOCV on {n_users} users using {device}...")
    
    for i in range(n_users):
        # 1. Split Data (Leave User i out)
        train_idx = [x for x in range(n_users) if x != i]
        test_idx = [i]
        
        # Raw Split
        X_ts_train, X_ts_test = X_ts[train_idx], X_ts[test_idx]
        X_p_train, X_p_test = X_path[train_idx], X_path[test_idx]
        X_u_train, X_u_test = X_user[train_idx], X_user[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        
        # 2. Scaling (Fit on TRAIN, transform TEST)
        # Flatten TS for scaling
        scaler_ts = StandardScaler()
        B_tr, P, T, F = X_ts_train.shape
        X_ts_train_flat = X_ts_train.reshape(-1, F)
        scaler_ts.fit(X_ts_train_flat)
        
        # Apply
        X_ts_train = scaler_ts.transform(X_ts_train_flat).reshape(B_tr, P, T, F)
        X_ts_test = scaler_ts.transform(X_ts_test.reshape(-1, F)).reshape(1, P, T, F)
        
        # Scale Targets
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        
        # 3. Convert to Tensors and move to device
        t_ts_train = torch.FloatTensor(X_ts_train).to(device)
        t_p_train = torch.FloatTensor(X_p_train).to(device)
        t_u_train = torch.FloatTensor(X_u_train).to(device)
        t_y_train = torch.FloatTensor(y_train).to(device)
        
        t_ts_test = torch.FloatTensor(X_ts_test).to(device)
        t_p_test = torch.FloatTensor(X_p_test).to(device)
        t_u_test = torch.FloatTensor(X_u_test).to(device)
        t_y_test_scaled = torch.FloatTensor(y_test_scaled).to(device)
        
        # 4. Initialize Model and move to device
        model = HierarchicalMILNet(ts_feats, path_feats, user_feats, n_targets).to(device)
        
        # CHANGE 1: Increased weight decay (0.001 -> 0.01) for stronger regularization
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # CHANGE 2: Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        criterion = nn.HuberLoss() 
        
        # 5. Training Loop
        model.train()
        train_loss = 0.0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # CHANGE 3: Apply Data Augmentation on the fly (Noise + Scaling)
            # We clone to avoid modifying the original data in place for subsequent epochs
            aug_ts = add_noise_and_scale(t_ts_train.clone(), noise_level=0.05)
            
            preds = model(aug_ts, t_p_train, t_u_train)
            loss = criterion(preds, t_y_train)
            loss.backward()
            optimizer.step()
            
            # Step the scheduler
            scheduler.step(loss)
            train_loss = loss.item()
            
        # 6. Evaluation
        model.eval()
        with torch.no_grad():
            pred_scaled_tensor = model(t_ts_test, t_p_test, t_u_test)
            val_loss = criterion(pred_scaled_tensor, t_y_test_scaled).item()
            pred_scaled = pred_scaled_tensor.cpu().numpy()
            
        all_preds_scaled.append(pred_scaled[0])
        all_actuals_scaled.append(y_test_scaled[0])
        
        print(f"User {i+1}/{n_users} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Final Metrics
    all_preds_scaled = np.array(all_preds_scaled)
    all_actuals_scaled = np.array(all_actuals_scaled)
    
    rmse = np.sqrt(mean_squared_error(all_actuals_scaled, all_preds_scaled))
    mae = mean_absolute_error(all_actuals_scaled, all_preds_scaled)
    
    # CHANGE 4: R2 Score Calculation
    r2 = r2_score(all_actuals_scaled, all_preds_scaled)
    
    print("\n=== Final LOOCV Results (Scaled) ===")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Overall MAE:  {mae:.4f}")
    print(f"Overall R2:   {r2:.4f}") # Aim for > 0.0
    
    return all_preds_scaled, all_actuals_scaled

if __name__ == "__main__":
    # Example usage with dummy dataframes
    # Load your actual dataframes here
    folder_path = "/data/dataset_conv@1s"
    print(f"Loading existing dataset from {folder_path}...")
    try:
        timeseries_df = pd.read_csv(f"{folder_path}/timeseries.csv")
        path_related_df = pd.read_csv(f"{folder_path}/path_related.csv")
        user_related_df = pd.read_csv(f"{folder_path}/user_related.csv")
        target_df = pd.read_csv(f"{folder_path}/target.csv")
        
        X_ts, X_path, X_user, Y = prepare_data(timeseries_df, path_related_df, user_related_df, target_df)
        
        # Increased epochs to 150 to accommodate the noise injection difficulty
        preds, actuals = train_loocv(X_ts, X_path, X_user, Y, epochs=150, lr=0.001)
    except FileNotFoundError:
        print("Dataset files not found. Ensure paths are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")