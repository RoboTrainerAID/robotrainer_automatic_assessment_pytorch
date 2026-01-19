import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

# ==========================================
# 1. The Dataset Class
# ==========================================
class RollatorDataset(Dataset):
    def __init__(self, sequences, static_data, labels):
        """
        sequences: List of np.arrays (each array is [Time_Steps, Dynamic_Features])
        static_data: np.array of shape [N_Samples, Static_Features]
        labels: np.array of shape [N_Samples, 1]
        """
        self.sequences = sequences
        self.static_data = torch.tensor(static_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert sequence to tensor and transpose to [Channels, Time] for 1D-CNN
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32).transpose(0, 1)
        return seq, self.static_data[idx], self.labels[idx]

def pad_collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Pads sequences in the batch to the length of the longest sequence in that batch.
    """
    sequences, static_data, labels = zip(*batch)
    
    # Pad sequences: [Batch, Channels, Time]
    # We transpose (2, 0, 1) because pad_sequence expects [Time, Batch, Channels] 
    # if batch_first=False, or we can just handle the permute manually.
    # Let's align dimensions first: List of [Channels, Time] -> List of [Time, Channels] for padding
    seq_list_permuted = [s.transpose(0, 1) for s in sequences]
    
    # Pad: Result is [Total_Max_Time, Batch, Channels]
    padded_seqs = torch.nn.utils.rnn.pad_sequence(seq_list_permuted, batch_first=True, padding_value=0.0)
    
    # Permute back to [Batch, Channels, Time] for CNN
    padded_seqs = padded_seqs.permute(0, 2, 1)
    
    static_data = torch.stack(static_data)
    labels = torch.stack(labels)
    
    return padded_seqs, static_data, labels

# ==========================================
# 2. The Model Architecture
# ==========================================
class DualStreamRollatorNet(nn.Module):
    def __init__(self, n_dynamic_features, n_static_features):
        super(DualStreamRollatorNet, self).__init__()
        
        # --- Stream 1: Temporal (CNN) ---
        # Input: [Batch, 50, Time]
        self.conv1 = nn.Conv1d(n_dynamic_features, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout_cnn = nn.Dropout(0.5)
        
        # --- Stream 2: Static (MLP) ---
        # Input: [Batch, 26]
        self.static_fc = nn.Linear(n_static_features, 16)
        self.static_bn = nn.BatchNorm1d(16)
        
        # --- Fusion ---
        # 64 (from CNN GAP) + 16 (from Static)
        self.fusion_fc1 = nn.Linear(64 + 16, 32)
        self.fusion_dropout = nn.Dropout(0.6)
        self.output_layer = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()

    def forward(self, x_ts, x_static):
        # 1. Temporal Stream
        x_t = self.relu(self.bn1(self.conv1(x_ts)))
        x_t = self.dropout_cnn(x_t)
        x_t = self.relu(self.bn2(self.conv2(x_t)))
        
        # Global Average Pooling: Mean over time dimension
        # Handles variable lengths and padding (zeros pull mean down slightly, 
        # but consistent within batch. Ideally use masking, but GAP is robust)
        x_t = x_t.mean(dim=2) 
        
        # 2. Static Stream
        x_s = self.relu(self.static_bn(self.static_fc(x_static)))
        
        # 3. Fusion
        combined = torch.cat((x_t, x_s), dim=1)
        out = self.relu(self.fusion_fc1(combined))
        out = self.fusion_dropout(out)
        prediction = self.output_layer(out)
        
        return prediction

# ==========================================
# 3. Data Preparation & Pipeline
# ==========================================
def prepare_data(ts_df, path_df, user_df, target_df, target_column='Single Leg Stance'):
    """
    Merges dataframes and prepares Train/Val splits based on USERS.
    """
    print("Preparing data...")
    
    # 1. Merge Static Data
    # user_df (User demographics) + path_df (Path summaries)
    static_df = pd.merge(path_df, user_df, on='user', how='inner')
    
    # 2. Merge Targets
    # We attach the target to the static dataframe
    targets = target_df[['user', target_column]].copy()
    static_df = pd.merge(static_df, targets, on='user', how='inner')
    
    # 3. Define Feature Columns
    # Dynamic: All columns in ts_df except identifiers
    dynamic_cols = [c for c in ts_df.columns if c not in ['user', 'path', 'time']]
    # Static: All columns in static_df except identifiers and target
    static_cols = [c for c in static_df.columns if c not in ['user', 'path', target_column]]
    
    print(f"Dynamic Features: {len(dynamic_cols)}")
    print(f"Static Features: {len(static_cols)}")
    
    # 4. Group Time Series
    # We need a list of arrays where each item is the matrix for one path
    # Sort to ensure time is ordered
    ts_df_sorted = ts_df.sort_values(by=['user', 'path', 'time'])
    
    # Efficient way to group:
    # Create a multi-index dictionary for fast lookup
    ts_grouped = ts_df_sorted.groupby(['user', 'path'])
    
    # 5. Build Final Lists
    X_seq_list = []
    X_static_list = []
    y_list = []
    groups = [] # For splitting by user
    
    # Iterate over the static_df (which has one row per path per user)
    # This ensures we align static data, labels, and sequences
    valid_indices = []
    
    for idx, row in static_df.iterrows():
        u, p = int(row['user']), int(row['path'])
        
        try:
            # Get corresponding time series
            group = ts_grouped.get_group((u, p))
            seq_data = group[dynamic_cols].values # Shape: [Time, Features]
            
            # Append
            X_seq_list.append(seq_data)
            X_static_list.append(row[static_cols].values)
            y_list.append(row[target_column])
            groups.append(u) # Keep track of user for splitting
            
        except KeyError:
            print(f"Warning: No time series found for User {u} Path {p}")
            continue

    # 6. Splitting (User-Level Split)
    # We must split by USER, not by sample, to prevent data leakage
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X_static_list, y_list, groups))
    
    # 7. Scaling
    # FIT scalers ONLY on Training data
    scaler_dynamic = StandardScaler()
    scaler_static = StandardScaler()
    # Scale labels too for better regression convergence
    scaler_target = StandardScaler() 
    
    # -- Process Train --
    # Flatten train sequences to fit scaler
    X_seq_train_flat = np.concatenate([X_seq_list[i] for i in train_idx])
    scaler_dynamic.fit(X_seq_train_flat)
    
    X_static_train = np.array([X_static_list[i] for i in train_idx])
    scaler_static.fit(X_static_train)
    
    y_train = np.array([y_list[i] for i in train_idx]).reshape(-1, 1)
    scaler_target.fit(y_train)
    
    # Apply transforms
    def get_dataset(indices):
        seqs = [scaler_dynamic.transform(X_seq_list[i]) for i in indices]
        stats = scaler_static.transform(np.array([X_static_list[i] for i in indices]))
        ys = scaler_target.transform(np.array([y_list[i] for i in indices]).reshape(-1, 1))
        return RollatorDataset(seqs, stats, ys)

    train_ds = get_dataset(train_idx)
    val_ds = get_dataset(val_idx)
    
    return train_ds, val_ds, scaler_target, len(dynamic_cols), len(static_cols)

# ==========================================
# 4. Training Loop
# ==========================================
def train_model(train_ds, val_ds, n_dyn, n_stat, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=pad_collate_fn)
    
    model = DualStreamRollatorNet(n_dyn, n_stat).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01) # L2 reg
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_ts, x_stat, y in train_loader:
            x_ts, x_stat, y = x_ts.to(device), x_stat.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x_ts, x_stat)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_ts, x_stat, y in val_loader:
                x_ts, x_stat, y = x_ts.to(device), x_stat.to(device), y.to(device)
                pred = model(x_ts, x_stat)
                val_loss += criterion(pred, y).item()
                
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    return model

# ==========================================
# 5. How to Run (Example)
# ==========================================
if __name__ == "__main__":
    # ASSUMPTION: You have your pandas dataframes loaded as:
    # timeseries_df, path_df, user_df, target_df
    
    # 1. Choose which score you want to predict
    target_metric = 'Single Leg Stance' # Change this to 'Balance Test', etc.
    folder_path = "/data/dataset_conv@1s"
    
    # 2. Prepare Data
    print(f"Loading existing dataset from {folder_path}...")
    timeseries_df = pd.read_csv(f"{folder_path}/timeseries.csv")
    path_related_df = pd.read_csv(f"{folder_path}/path_related.csv")
    user_related_df = pd.read_csv(f"{folder_path}/user_related.csv")
    target_df = pd.read_csv(f"{folder_path}/target.csv")
    
    train_dataset, val_dataset, label_scaler, n_dyn_feats, n_stat_feats = prepare_data(
        timeseries_df, path_related_df, user_related_df, target_df, target_column=target_metric
    )
    
    # 3. Train
    model = train_model(train_dataset, val_dataset, n_dyn_feats, n_stat_feats, epochs=30)
    
    # 4. Inference (Example on Validation Set)
    model.eval()
    
    # Define device again (or grab it from the model)
    device = next(model.parameters()).device 

    x_ts, x_stat, y_true_scaled = val_dataset[0] # Get one sample
    
    # Add batch dim AND move to device
    x_ts = x_ts.unsqueeze(0).to(device) 
    x_stat = x_stat.unsqueeze(0).to(device)
    
    with torch.no_grad():
        y_pred_scaled = model(x_ts, x_stat)
        
    # Move result back to CPU for numpy conversion
    y_pred_real = label_scaler.inverse_transform(y_pred_scaled.cpu().numpy())
    y_true_real = label_scaler.inverse_transform(y_true_scaled.reshape(1, -1).numpy())
    
    print(f"Predicted {target_metric}: {y_pred_real[0][0]:.2f}")
    print(f"Actual {target_metric}: {y_true_real[0][0]:.2f}")