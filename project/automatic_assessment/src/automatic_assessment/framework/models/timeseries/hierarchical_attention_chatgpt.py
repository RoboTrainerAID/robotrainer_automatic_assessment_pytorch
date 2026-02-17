import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from .base import BaseModel

class TimeSeriesEncoder(nn.Module):
    def __init__(self, feature_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GroupNorm(4, 8),
            nn.ReLU(),

            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
        )

        self.projection = nn.Linear(16, feature_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = h.mean(dim=-1)
        return self.projection(h)


class PathEncoder(nn.Module):
    def __init__(self, ts_dim, path_feat_dim):
        super().__init__()

        self.path_proj = nn.Linear(path_feat_dim, 8)
        self.fc = nn.Sequential(
            nn.Linear(ts_dim + 8, 16),
            nn.GELU()
        )

    def forward(self, ts_features, path_features):
        path_low = self.path_proj(path_features)
        x = torch.cat([ts_features, path_low], dim=-1)
        return self.fc(x)



class PathAttentionPooling(nn.Module):
    def __init__(self, d_path):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(d_path, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights



class HierarchicalTimeseriesChatGPT(BaseModel):
    model_name = "HierarchicalTimeseriesChatGPT"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        ts_shape, path_shape, user_shape = input_dims

        self.n_paths = ts_shape[1]
        self.n_ts = ts_shape[2]
        self.f_path = path_shape[2]
        self.f_user = user_shape[1]

        self.d_ts = hyperparams.get("d_ts", 32)
        self.d_path = hyperparams.get("d_path", 32)
        self.d_user = hyperparams.get("d_user", 32)
        self.dropout = hyperparams.get("dropout", 0.1)

        # Initialize storage for attention weights
        self.last_attn_weights = None

        self.ts_encoder = TimeSeriesEncoder(self.d_ts)

        self.path_encoder = PathEncoder(
            ts_dim=self.d_ts,
            path_feat_dim=self.f_path
        )

        self.path_attention = PathAttentionPooling(self.d_path)

        self.user_head = nn.Linear(16 + self.f_user, output_dim)

    def forward(self, x):
        x_ts, x_path, x_user = x
        B, P, T, L = x_ts.shape

        # --- Time-series encoding ---
        x_ts = x_ts.view(B * P * T, 1, L)
        ts_features = self.ts_encoder(x_ts)
        ts_features = ts_features.view(B, P, T, -1)

        ts_path_features = ts_features.mean(dim=2)

        # --- Path encoding ---
        path_embeddings = self.path_encoder(ts_path_features, x_path)

        # --- Attention pooling over paths ---
        user_embedding, attn_weights = self.path_attention(path_embeddings)
        self.last_attn_weights = attn_weights

        # --- User-level regression ---
        user_input = torch.cat([user_embedding, x_user], dim=-1)
        prediction = self.user_head(user_input)

        return prediction



    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "d_ts": trial.suggest_categorical("d_ts", [8, 16, 32]),
            "d_path": trial.suggest_categorical("d_path", [8, 32, 64]),
        }

    @staticmethod
    def get_default_parameters():
        return {
            "d_ts": 16,
            "d_path": 16,
            "weight_decay": 0.02,
            "lr": 0.001,
        }
