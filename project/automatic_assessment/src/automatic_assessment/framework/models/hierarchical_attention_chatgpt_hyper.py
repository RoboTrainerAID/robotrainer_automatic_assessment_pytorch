import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from .base import BaseModel


class TimeSeriesEncoder(nn.Module):
    def __init__(self, in_channels=1, feature_dim=16, dropout=0.0):
        super().__init__()
        # Fixed 2-layer CNN with predefined channels and kernel sizes
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 8, kernel_size=5, padding=2),
            nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
        )
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.projection = nn.Linear(16, feature_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = h.mean(dim=-1)  # global average pooling
        h = self.dropout(h)
        return self.projection(h)


class PathEncoder(nn.Module):
    def __init__(self, ts_dim, path_feat_dim, hidden_dim=16, dropout=0.0):
        super().__init__()
        self.path_proj = nn.Linear(path_feat_dim, hidden_dim)
        layers = [nn.Linear(ts_dim + hidden_dim, hidden_dim), nn.GELU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*layers)

    def forward(self, ts_features, path_features):
        path_low = self.path_proj(path_features)
        x = torch.cat([ts_features, path_low], dim=-1)
        return self.fc(x)


class PathAttentionPooling(nn.Module):
    def __init__(self, d_path, attn_hidden=8, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(d_path, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, 1)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.attn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)

        return pooled, weights


class HierarchicalTimeseriesChatGPT(BaseModel):
    model_name = "HierarchicalTimeseriesChatGPT"

    def __init__(self, input_dims, output_dim, hyperparams: Dict[str, Any]):
        super().__init__(input_dims, output_dim, hyperparams)

        ts_shape, path_shape, user_shape = input_dims
        self.n_paths, self.n_ts, self.f_path, self.f_user = ts_shape[1], ts_shape[2], path_shape[2], user_shape[1]

        # Hyperparameters
        self.d_ts = hyperparams.get("d_ts", 32)
        self.d_path = hyperparams.get("d_path", 32)
        self.d_user = hyperparams.get("d_user", 16)
        self.dropout = hyperparams.get("dropout", 0.1)
        self.use_attention = hyperparams.get("use_attention", True)

        # Initialize storage for attention weights
        self.last_attn_weights = None

        # Encoders
        self.ts_encoder = TimeSeriesEncoder(feature_dim=self.d_ts, dropout=self.dropout)
        self.path_encoder = PathEncoder(ts_dim=self.d_ts, path_feat_dim=self.f_path,
                                        hidden_dim=self.d_path, dropout=self.dropout)
        
        if self.use_attention:
            self.path_attention = PathAttentionPooling(d_path=self.d_path, attn_hidden=8, dropout=self.dropout)
        else:
            self.path_attention = None

        # User-level head
        self.user_head = nn.Linear(self.d_path + self.f_user, output_dim)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x
        B, P, T, L = x_ts.shape

        # --- Time-series encoding ---
        x_ts = x_ts.view(B * P * T, 1, L)
        ts_features = self.ts_encoder(x_ts)
        ts_features = ts_features.view(B, P, T, -1)
        ts_path_features = ts_features.mean(dim=2)

        # --- Path encoding ---
        path_embeddings = self.path_encoder(ts_path_features, x_path)

        # --- Pooling over paths ---
        if self.use_attention and self.path_attention is not None:
            user_embedding, attn_weights = self.path_attention(path_embeddings)
            self.last_attn_weights = attn_weights
        else:
            # Simple mean pooling
            user_embedding = path_embeddings.mean(dim=1)

        # --- User-level regression ---
        user_input = torch.cat([user_embedding, x_user], dim=-1)
        prediction = self.user_head(user_input)

        return prediction

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "d_ts": trial.suggest_categorical("d_ts", [8, 16, 32, 64]),
            "d_path": trial.suggest_categorical("d_path", [8, 16, 32, 64]),
            "d_user": trial.suggest_categorical("d_user", [8, 16, 32, 64]),
            "use_attention": trial.suggest_categorical("use_attention", [True, False]),
        }

    @staticmethod
    def get_default_parameters():
        return {
            "d_ts": 16,
            "d_path": 16,
            "d_user": 16,
            "dropout": 0.1,
            "use_attention": False,
            "weight_decay": 0.02,
            "lr": 0.001,
        }
