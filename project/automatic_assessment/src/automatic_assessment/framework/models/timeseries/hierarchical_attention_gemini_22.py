import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base import BaseModel


class HierarchicalTimeseriesGemini22(BaseModel):
    model_name = "HierarchicalTimeseriesGemini22"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        # -------------------------
        # Input dimensions
        # -------------------------
        ts_shape = input_dims[0]
        self.n_paths = ts_shape[1]        # 20
        self.n_ts_per_path = ts_shape[2]  # 35
        self.ts_len = ts_shape[3]         # 158
        self.f_path = input_dims[1][2]    # 105
        self.f_user = input_dims[2][1]    # 2

        # -------------------------
        # Hyperparameters
        # -------------------------
        ts_out_channels = hyperparams.get("ts_out_channels", 16)
        num_conv_layers = hyperparams.get("num_conv_layers", 1)
        path_dim = hyperparams.get("path_dim", 8)
        regressor_dim = hyperparams.get("regressor_dim", 128)

        dropout_path = hyperparams.get("dropout_path", 0.25)
        dropout_reg = hyperparams.get("dropout_reg", 0.25)

        self.path_aggregation = hyperparams.get("path_aggregation", "static")

        self.last_attn_weights = None

        # ============================================================
        # 1. Time-Series Extractor
        # ============================================================
        ts_layers = []
        in_channels = 1

        for i in range(num_conv_layers):
            kernel_size = 5 if i == 0 else 3
            stride = 2 if i == 0 else 1
            padding = 0 if i == 0 else 1

            ts_layers.append(
                nn.Conv1d(in_channels, ts_out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding)
            )
            ts_layers.append(nn.ReLU())
            in_channels = ts_out_channels

        ts_layers.append(nn.AdaptiveAvgPool1d(1))
        ts_layers.append(nn.Flatten())

        self.ts_extractor = nn.Sequential(*ts_layers)
        self.ts_feat_dim = ts_out_channels

        # ============================================================
        # 2. Path-Level Integration
        # ============================================================
        total_path_input = (self.n_ts_per_path * self.ts_feat_dim) + self.f_path

        self.path_bottleneck = nn.Sequential(
            nn.Linear(total_path_input, path_dim),
            nn.ReLU(),
            nn.Dropout(dropout_path)
        )

        # ============================================================
        # 3. Path Aggregation Options
        # ============================================================

        # Static learnable weights (global)
        if self.path_aggregation == "static":
            self.path_weights = nn.Parameter(torch.ones(self.n_paths))

        # Per-sample attention
        elif self.path_aggregation == "attention":
            self.path_attention = nn.Sequential(
                nn.Linear(path_dim, path_dim),
                nn.Tanh(),
                nn.Linear(path_dim, 1)
            )

        # Mean pooling requires no parameters

        # ============================================================
        # 4. Regressor
        # ============================================================
        total_user_input = path_dim + self.f_user

        self.regressor = nn.Sequential(
            nn.Linear(total_user_input, regressor_dim),
            nn.ReLU(),
            nn.Dropout(dropout_reg),
            nn.Linear(regressor_dim, output_dim)
        )

    # ================================================================
    # Forward
    # ================================================================
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x
        batch_size = x_ts.shape[0]

        # ---- TS extraction ----
        x_ts = x_ts.reshape(-1, 1, self.ts_len)
        ts_feats = self.ts_extractor(x_ts)

        ts_feats = ts_feats.view(batch_size, self.n_paths, -1)

        # ---- Path integration ----
        path_combined = torch.cat([ts_feats, x_path], dim=-1)
        path_feats = self.path_bottleneck(path_combined)

        # ---- Aggregation ----
        if self.path_aggregation == "static":
            weights = F.softmax(self.path_weights, dim=0)
            self.last_attn_weights = weights.unsqueeze(0).expand(batch_size, -1)
            path_aggr = torch.sum(
                path_feats * weights.view(1, self.n_paths, 1),
                dim=1
            )

        elif self.path_aggregation == "attention":
            attn_logits = self.path_attention(path_feats)  # (B, 20, 1)
            attn_weights = F.softmax(attn_logits, dim=1)
            self.last_attn_weights = attn_weights.squeeze(-1).detach()
            path_aggr = torch.sum(path_feats * attn_weights, dim=1)

        else:  # mean pooling
            self.last_attn_weights = None
            path_aggr = torch.mean(path_feats, dim=1)

        # ---- User integration ----
        user_combined = torch.cat([path_aggr, x_user], dim=-1)
        prediction = self.regressor(user_combined)

        return prediction

    # ================================================================
    # Updated Search Space
    # ================================================================
    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            # Optimization (narrowed around best result)
            "lr": trial.suggest_float("lr", 8e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),

            # Regularization split
            "dropout_path": trial.suggest_float("dropout_path", 0.15, 0.4),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.15, 0.35),

            # Architecture (centered around best values)
            "ts_out_channels": trial.suggest_categorical("ts_out_channels", [16, 32]),
            "num_conv_layers": trial.suggest_int("num_conv_layers", 1, 2),
            "path_dim": trial.suggest_categorical("path_dim", [8, 16]),
            "regressor_dim": trial.suggest_categorical("regressor_dim", [64, 128]),

            # Aggregation strategy
            "path_aggregation": trial.suggest_categorical(
                "path_aggregation",
                ["mean", "static", "attention"]
            ),

            # Batch Size
            "batch_size": trial.suggest_categorical("batch_size", [16])
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "lr": 0.0001699871281284228,
            "weight_decay": 0.0003325384737901396,
            "dropout_path": 0.3386564741107209,
            "dropout_reg": 0.21174325741397373,
            "ts_out_channels": 32,
            "num_conv_layers": 1,
            "path_dim": 8,
            "regressor_dim": 128,
            "path_aggregation": "attention",
            "batch_size": 16
        }
