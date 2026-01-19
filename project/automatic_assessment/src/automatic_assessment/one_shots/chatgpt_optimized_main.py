import numpy as np
import pandas as pd
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

# Load preprocessed data
timeseries_df = pd.read_csv("/data/dataset_conv@1s/timeseries.csv")
path_df = pd.read_csv("/data/dataset_conv@1s/path_related.csv")
user_df = pd.read_csv("/data/dataset_conv@1s/user_related.csv")
target_df = pd.read_csv("/data/dataset_conv@1s/target.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NUM_PATHS = 20
TIME_FEATURE_COLUMNS = [
    c for c in timeseries_df.columns
    if c not in ["user", "path", "time"]
]

PATH_FEATURE_COLUMNS = [
    c for c in path_df.columns
    if c not in ["user", "path"]
]

USER_FEATURE_COLUMNS = ["age", "sex_value"]

TARGET_COLUMNS = [
    c for c in target_df.columns
    if c != "user"
]

F_TIME = len(TIME_FEATURE_COLUMNS)
F_PATH = len(PATH_FEATURE_COLUMNS)
F_USER = len(USER_FEATURE_COLUMNS)
NUM_TASKS = len(TARGET_COLUMNS)

class UserMILDataset(Dataset):
    def __init__(
        self,
        user_ids: List[int],
        timeseries_df: pd.DataFrame,
        path_df: pd.DataFrame,
        user_df: pd.DataFrame,
        target_df: pd.DataFrame,
        time_scaler: StandardScaler = None,
        path_scaler: StandardScaler = None,
        user_scaler: StandardScaler = None,
        target_scaler: StandardScaler = None,
        fit_scalers: bool = False,
    ):
        self.user_ids = user_ids

        self.timeseries_df = timeseries_df
        self.path_df = path_df
        self.user_df = user_df
        self.target_df = target_df

        self.time_scaler = time_scaler
        self.path_scaler = path_scaler
        self.user_scaler = user_scaler
        self.target_scaler = target_scaler

        if fit_scalers:
            self._fit_scalers()

    def _fit_scalers(self):
        # --- Time-series scaler ---
        ts = self.timeseries_df[
            self.timeseries_df["user"].isin(self.user_ids)
        ][TIME_FEATURE_COLUMNS].values
        self.time_scaler.fit(ts)

        # --- Path-level scaler ---
        pf = self.path_df[
            self.path_df["user"].isin(self.user_ids)
        ][PATH_FEATURE_COLUMNS].values
        self.path_scaler.fit(pf)

        # --- User-level scaler ---
        uf = self.user_df[
            self.user_df["user"].isin(self.user_ids)
        ][USER_FEATURE_COLUMNS].values
        self.user_scaler.fit(uf)

        # --- Target scaler ---
        tf = self.target_df[
            self.target_df["user"].isin(self.user_ids)
        ][TARGET_COLUMNS].values
        self.target_scaler.fit(tf)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]

        # ---- User features ----
        user_row = self.user_df[self.user_df["user"] == user_id]
        user_feat = user_row[USER_FEATURE_COLUMNS].values.astype(np.float32)
        user_feat = self.user_scaler.transform(user_feat)[0]

        # ---- Targets ----
        target_row = self.target_df[self.target_df["user"] == user_id]
        targets = target_row[TARGET_COLUMNS].values.astype(np.float32)
        targets = self.target_scaler.transform(targets)[0]

        # ---- Path-level data ----
        user_paths = self.path_df[self.path_df["user"] == user_id].sort_values("path")
        path_features = user_paths[PATH_FEATURE_COLUMNS].values.astype(np.float32)
        path_features = self.path_scaler.transform(path_features)

        # ---- Time-series per path ----
        path_tensors = []
        lengths = []

        for path_id in user_paths["path"].values:
            ts_path = self.timeseries_df[
                (self.timeseries_df["user"] == user_id) &
                (self.timeseries_df["path"] == path_id)
            ].sort_values("time")

            ts_vals = ts_path[TIME_FEATURE_COLUMNS].values.astype(np.float32)
            ts_vals = self.time_scaler.transform(ts_vals)

            path_tensors.append(torch.tensor(ts_vals))
            lengths.append(ts_vals.shape[0])

        T_max = max(lengths)

        X = torch.zeros(NUM_PATHS, T_max, F_TIME)
        time_mask = torch.zeros(NUM_PATHS, T_max, dtype=torch.bool)

        for p, tensor in enumerate(path_tensors):
            L = tensor.shape[0]
            X[p, :L] = tensor
            time_mask[p, :L] = True

        path_mask = torch.ones(NUM_PATHS, dtype=torch.bool)

        return {
            "X": X,
            "time_mask": time_mask,
            "path_mask": path_mask,
            "path_features": torch.tensor(path_features),
            "user_features": torch.tensor(user_feat),
            "targets": torch.tensor(targets),
        }

class TemporalAttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x, mask):
        # x: (N, T, C), mask: (N, T)
        scores = torch.matmul(x, self.query)  # (N, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * alpha.unsqueeze(-1), dim=1)
        return pooled

class PathEncoder(nn.Module):
    def __init__(self, in_features, emb_dim=32):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=7, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(16, 128),
            nn.GELU(),
        )

        self.att_pool = TemporalAttentionPooling(128)
        self.proj = nn.Linear(128, emb_dim)

    def forward(self, x, mask):
        # x: (N, T, F)
        x = x.transpose(1, 2)  # (N, F, T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (N, T', C)

        mask = mask[:, :x.shape[1]]
        pooled = self.att_pool(x, mask)
        return self.proj(pooled)

class HierarchicalMIL(nn.Module):
    def __init__(self):
        super().__init__()

        self.path_encoder = PathEncoder(F_TIME, emb_dim=32)

        self.path_phi = nn.Sequential(
            nn.Linear(32 + F_PATH, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.4),
        )

        self.att_query = nn.Parameter(torch.randn(128))

        self.user_rho = nn.Sequential(
            nn.Linear(128 + F_USER, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(NUM_TASKS)
        ])

        self.log_vars = nn.Parameter(torch.zeros(NUM_TASKS))

    def forward(self, X, time_mask, path_mask, path_features, user_features):
        B, P, T, F = X.shape

        X_flat = X.view(B * P, T, F)
        mask_flat = time_mask.view(B * P, T)

        path_emb = self.path_encoder(X_flat, mask_flat)
        path_emb = path_emb.view(B, P, -1)

        path_cat = torch.cat([path_emb, path_features], dim=-1)
        phi = self.path_phi(path_cat.view(B * P, -1)).view(B, P, -1)

        scores = torch.matmul(phi, self.att_query)
        scores = scores.masked_fill(~path_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1)

        user_from_paths = torch.sum(phi * alpha.unsqueeze(-1), dim=1)
        user_input = torch.cat([user_from_paths, user_features], dim=-1)
        user_emb = self.user_rho(user_input)

        outputs = torch.cat(
            [head(user_emb) for head in self.heads], dim=1
        )

        return outputs

def multitask_loss(preds, targets, log_vars):
    loss = torch.tensor(0.0, device=preds.device)
    for i in range(preds.shape[1]):
        mse = F.mse_loss(preds[:, i], targets[:, i])
        loss += 0.5 * torch.exp(-log_vars[i]) * mse + 0.5 * log_vars[i]
    return loss

def train_loocv(
    timeseries_df: pd.DataFrame,
    path_df: pd.DataFrame,
    user_df: pd.DataFrame,
    target_df: pd.DataFrame,
    epochs: int = 200,
    lr: float = 3e-4,
):
    print(f"Training on device: {DEVICE}")

    loo = LeaveOneOut()
    user_ids = user_df["user"].values

    fold_rmse = []

    for fold, (train_idx, test_idx) in enumerate(loo.split(user_ids)):
        train_users = user_ids[train_idx]
        test_users = user_ids[test_idx]

        # Scalers
        time_scaler = StandardScaler()
        path_scaler = StandardScaler()
        user_scaler = StandardScaler()
        target_scaler = StandardScaler()

        train_ds = UserMILDataset(
            train_users, timeseries_df, path_df, user_df, target_df,
            time_scaler, path_scaler, user_scaler, target_scaler,
            fit_scalers=True,
        )

        test_ds = UserMILDataset(
            test_users, timeseries_df, path_df, user_df, target_df,
            time_scaler, path_scaler, user_scaler, target_scaler,
            fit_scalers=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=4,
            shuffle=True,
            collate_fn=mil_collate_fn,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=mil_collate_fn,
        )


        model = HierarchicalMIL().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_loss = np.inf
        patience, max_patience = 0, 15

        for epoch in range(epochs):
            model.train()
            train_loss_sum = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                preds = model(
                    batch["X"].to(DEVICE),
                    batch["time_mask"].to(DEVICE),
                    batch["path_mask"].to(DEVICE),
                    batch["path_features"].to(DEVICE),
                    batch["user_features"].to(DEVICE),
                )
                loss = multitask_loss(
                    preds,
                    batch["targets"].to(DEVICE),
                    model.log_vars,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss_sum += loss.item()
            
            avg_train_loss = train_loss_sum / len(train_loader)

            # Validation
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    preds = model(
                        batch["X"].to(DEVICE),
                        batch["time_mask"].to(DEVICE),
                        batch["path_mask"].to(DEVICE),
                        batch["path_features"].to(DEVICE),
                        batch["user_features"].to(DEVICE),
                    )
                    v_loss = multitask_loss(
                        preds,
                        batch["targets"].to(DEVICE),
                        model.log_vars,
                    )
                    val_loss_sum += v_loss.item()
            
            # Early stopping (using validation loss)
            if val_loss_sum < best_loss:
                best_loss = val_loss_sum
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    break

        # Evaluation
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                preds = model(
                    batch["X"].to(DEVICE),
                    batch["time_mask"].to(DEVICE),
                    batch["path_mask"].to(DEVICE),
                    batch["path_features"].to(DEVICE),
                    batch["user_features"].to(DEVICE),
                )
                # Use scaled values directly for better regression overview
                preds_np = preds.cpu().numpy()
                targets_np = batch["targets"].cpu().numpy()

                rmse = np.sqrt(np.mean((preds_np - targets_np) ** 2))
                fold_rmse.append(rmse)

        print(f"Fold {fold+1}/28 | Best Val Loss: {best_loss:.4f} | RMSE (Scaled): {rmse:.3f}")

    print(f"\nMean RMSE (Scaled): {np.mean(fold_rmse):.3f} ± {np.std(fold_rmse):.3f}")

def mil_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Pads users in a batch to the maximum T across users.
    """
    batch_size = len(batch)
    P = batch[0]["X"].shape[0]
    F = batch[0]["X"].shape[2]

    T_max_batch = max(item["X"].shape[1] for item in batch)

    X = torch.zeros(batch_size, P, T_max_batch, F)
    time_mask = torch.zeros(batch_size, P, T_max_batch, dtype=torch.bool)

    path_mask = torch.stack([item["path_mask"] for item in batch])
    path_features = torch.stack([item["path_features"] for item in batch])
    user_features = torch.stack([item["user_features"] for item in batch])
    targets = torch.stack([item["targets"] for item in batch])

    for i, item in enumerate(batch):
        T = item["X"].shape[1]
        X[i, :, :T, :] = item["X"]
        time_mask[i, :, :T] = item["time_mask"]

    return {
        "X": X,
        "time_mask": time_mask,
        "path_mask": path_mask,
        "path_features": path_features,
        "user_features": user_features,
        "targets": targets,
    }


if __name__ == "__main__":

    train_loocv(
        timeseries_df,
        path_df,
        user_df,
        target_df,
        epochs=150,
        lr=3e-4,
    )