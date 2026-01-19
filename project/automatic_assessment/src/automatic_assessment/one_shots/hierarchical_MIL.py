import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class HierarchicalRollatorDataset(Dataset):
    """
    One item = one user
    """
    def __init__(self, ts_df, user_df, target_df, target_name):
        self.ts_df = ts_df
        self.user_df = user_df.set_index("user")
        self.target_df = target_df.set_index("user")
        self.target_name = target_name

        self.users = sorted(self.target_df.index.unique())

        self.feature_cols = [
            c for c in ts_df.columns
            if c not in ["user", "path", "time"]
        ]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]

        user_ts = self.ts_df[self.ts_df.user == user_id]

        paths = []
        for path_id, df_p in user_ts.groupby("path"):
            x = torch.tensor(
                df_p[self.feature_cols].values,
                dtype=torch.float32
            )
            paths.append(x)

        user_meta = torch.tensor(
            self.user_df.loc[user_id].values,
            dtype=torch.float32
        )

        y = torch.tensor(
            self.target_df.loc[user_id, self.target_name],
            dtype=torch.float32
        )

        return paths, user_meta, y

def collate_users(batch):
    paths, user_meta, y = zip(*batch)
    return list(paths), torch.stack(user_meta), torch.stack(y)

import torch.nn as nn
import torch.nn.functional as F

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        # x: [N, D]
        a = self.attn(x)          # [N, 1]
        w = torch.softmax(a, dim=0)
        return (w * x).sum(dim=0), w

class HierarchicalMIL(nn.Module):
    def __init__(self, in_dim, user_dim, hidden=64):
        super().__init__()

        self.instance_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden)
        )

        self.path_pool = AttentionPool(hidden)
        self.user_pool = AttentionPool(hidden)

        self.regressor = nn.Sequential(
            nn.Linear(hidden + user_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, paths, user_meta):
        user_path_embeddings = []
        path_attentions = []

        for path in paths:
            h = self.instance_encoder(path)
            p_emb, p_att = self.path_pool(h)
            user_path_embeddings.append(p_emb)
            path_attentions.append(p_att)

        P = torch.stack(user_path_embeddings)      # [num_paths, hidden]
        user_emb, user_att = self.user_pool(P)

        full = torch.cat([user_emb, user_meta], dim=0)
        y_hat = self.regressor(full)

        return y_hat.squeeze(), path_attentions, user_att

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def train_loso(dataset: Dataset, epochs: int = 50, lr: float = 1e-3) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    users = dataset.users
    preds, trues = [], []
    
    # Calculate target std for scaled RMSE
    all_targets = dataset.target_df[dataset.target_name].values
    target_std = np.std(all_targets)

    for i, test_user in enumerate(users):
        train_ids = [u for u in users if u != test_user]

        train_idx = [users.index(u) for u in train_ids]
        test_idx = users.index(test_user)

        train_ds = torch.utils.data.Subset(dataset, train_idx)
        test_ds = torch.utils.data.Subset(dataset, [test_idx])

        train_loader = DataLoader(
            train_ds, batch_size=1,
            collate_fn=collate_users, shuffle=True
        )

        model = HierarchicalMIL(
            in_dim=len(dataset.feature_cols),
            user_dim=dataset.user_df.shape[1]
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Track losses for the last user to visualize training
        is_last_user = (i == len(users) - 1)
        train_losses = []
        val_losses = []

        for _ in range(epochs):
            model.train()
            epoch_loss = 0.0
            for paths, meta, y in train_loader:
                # Move inputs to device
                # paths[0] is the list of path tensors for the single user in batch
                paths_dev = [p.to(device) for p in paths[0]]
                meta_dev = meta[0].to(device)
                y_dev = y[0].to(device)

                y_hat, _, _ = model(paths_dev, meta_dev)
                loss = F.mse_loss(y_hat, y_dev)

                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            
            if is_last_user:
                train_losses.append(epoch_loss / len(train_loader))
                
                # Validation on test user
                model.eval()
                with torch.no_grad():
                    paths, meta, y = test_ds[0]
                    paths_dev = [p.to(device) for p in paths]
                    meta_dev = meta.to(device)
                    y_dev = y.to(device)
                    
                    y_hat_val, _, _ = model(paths_dev, meta_dev)
                    v_loss = F.mse_loss(y_hat_val, y_dev)
                    val_losses.append(v_loss.item())

        model.eval()
        with torch.no_grad():
            paths, meta, y = test_ds[0]
            # Move inputs to device
            paths_dev = [p.to(device) for p in paths]
            meta_dev = meta.to(device)
            
            # Capture user_att for visualization
            y_hat, _, user_att = model(paths_dev, meta_dev)

        preds.append(y_hat.item())
        trues.append(y.item())
        
        if is_last_user:
            print(f"Finished User {test_user}. Saving metrics plot...")
            plt.figure(figsize=(12, 5))
            
            # Plot Losses
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.title(f"Loss Curves (User {test_user})")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.legend()
            
            # Plot Attention
            plt.subplot(1, 2, 2)
            plot_path_attention(user_att, ax=plt.gca())
            
            plt.tight_layout()
            plt.savefig(f"training_metrics_user_{test_user}.png")
            plt.close()

    rmse = np.sqrt(mean_squared_error(trues, preds))
    scaled_rmse = rmse / target_std
    return rmse, scaled_rmse, preds, trues

def plot_path_attention(attn, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
        
    weights = attn.reshape(-1).cpu().numpy()
    ax.bar(range(len(weights)), weights)
    ax.set_title("Path Attention Weights")
    ax.set_xlabel("Path Index")
    ax.set_ylabel("Weight")
    
    if ax is None:
        plt.savefig("path_attention.png")
        plt.close()

def main():
    folder_path = "/data/dataset_conv@1s/pruned_random_forest"
    ts = pd.read_csv(f"{folder_path}/timeseries.csv")
    path = pd.read_csv(f"{folder_path}/path.csv")
    user = pd.read_csv(f"{folder_path}/user.csv")
    target = pd.read_csv(f"{folder_path}/target.csv")

    target_name = "Hand Grip Right"

    dataset = HierarchicalRollatorDataset(
        ts_df=ts,
        user_df=user,
        target_df=target,
        target_name=target_name
    )

    rmse, scaled_rmse, preds, trues = train_loso(dataset)

    print(f"\nLOSO RMSE ({target_name}): {rmse:.3f}")
    print(f"Scaled RMSE: {scaled_rmse:.3f}")

if __name__ == "__main__":
    main()
