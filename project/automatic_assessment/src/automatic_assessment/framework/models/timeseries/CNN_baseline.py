import torch
import torch.nn as nn
from typing import Dict, Any, List

from ..base import BaseModel


def masked_mean(x: torch.Tensor, lengths: torch.Tensor):
    device = x.device
    mask = torch.arange(x.size(1), device=device)[None, :] < lengths[:, None]
    mask = mask.float().unsqueeze(-1)
    x = x * mask
    return x.sum(1) / lengths.clamp(min=1).unsqueeze(-1)


def compute_lengths_flat(x_flat: torch.Tensor):
    valid = (x_flat.abs() > 1e-8)
    last_valid = valid.flip(1).float().argmax(dim=1)
    lengths = x_flat.size(1) - last_valid
    return lengths.clamp(min=1)


class CNNBaseline(BaseModel):
    model_name = "CNN_Baseline"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        ts_shape = input_dims[0]
        self.n_paths = ts_shape[1]
        self.n_ts = ts_shape[2]
        self.max_timesteps = ts_shape[3]

        path_shape = input_dims[1]
        self.f_path = path_shape[2]

        user_shape = input_dims[2]
        self.f_user = user_shape[1]

        hp = hyperparams
        channels = hp["cnn_channels"]

        self.cnn = nn.Sequential(
            nn.Conv1d(self.n_ts, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.path_dim = hyperparams["path_dim"]

        self.path_bottleneck = nn.Sequential(
            nn.Linear(channels + self.f_path, self.path_dim),
            nn.ReLU()
        )

        self.path_attention = nn.Sequential(
            nn.Linear(self.path_dim, self.path_dim),
            nn.Tanh(),
            nn.Linear(self.path_dim, 1)
        )

        fused_dim = self.path_dim + self.f_user

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, hp["regressor_dim"]),
            nn.ReLU(),
            nn.Dropout(hp["dropout_reg"]),
            nn.Linear(hp["regressor_dim"], output_dim),
        )

    # -----------------------------------------------------

    def encode_timeseries(self, x_ts):
        B, P, TS, T = x_ts.shape

        # compute path lengths
        x_path = x_ts.abs().sum(dim=2)
        x_path = x_path.view(B * P, T)
        l = compute_lengths_flat(x_path)

        # flatten → (B*P,TS,T)
        x = x_ts.reshape(B * P, TS, T)

        feat = self.cnn(x)
        feat = feat.permute(0, 2, 1)

        pooled = masked_mean(feat, l)

        pooled = pooled.view(B, P, pooled.shape[-1])
        return pooled

    # -----------------------------------------------------

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x

        ts_paths = self.encode_timeseries(x_ts)        # (B,P,C)

        path_combined = torch.cat([ts_paths, x_path], dim=-1)
        path_feat = self.path_bottleneck(path_combined)

        att = self.path_attention(path_feat)
        att = torch.softmax(att, dim=1)

        path_global = (path_feat * att).sum(dim=1)

        fused = torch.cat([path_global, x_user], dim=1)

        return self.regressor(fused)

    # =====================================================
    # OPTUNA
    # =====================================================

    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            "cnn_channels": trial.suggest_categorical("cnn_channels", [4, 8, 12]),
            "path_dim": trial.suggest_categorical("path_dim", [8, 12, 16, 24]),
            "regressor_dim": trial.suggest_categorical("regressor_dim", [96, 128, 192]),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.2, 0.5),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [6]),
            # "correlation_threshold": trial.suggest_float("correlation_threshold", 0.1, 0.4, step=0.01),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5)
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "cnn_channels": 8,
            "path_dim": 16,
            "regressor_dim": 128,
            "dropout_reg": 0.39377583985758274,
            "lr": 0.0008957476788727834,
            "weight_decay": 0.00954951015873937,
            "batch_size": 6,
            # "correlation_threshold": 0.1
            "n_path_features": 20
        }
