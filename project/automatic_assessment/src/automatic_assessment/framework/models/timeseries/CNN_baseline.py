import torch
import torch.nn as nn
from typing import Dict, Any, List

from ..base import BaseModel


def masked_mean(x: torch.Tensor, lengths: torch.Tensor):
    """
    Computes mean over time dimension considering variable lengths.
    x: (N, C, T) -> (N, C)
    lengths: (N,)
    """
    device = x.device
    # mask: (N, 1, T) - True where t < length
    mask = torch.arange(x.size(2), device=device)[None, None, :] < lengths[:, None, None]
    mask = mask.float()
    
    x_masked = x * mask
    # Sum over time (dim 2)
    summed = x_masked.sum(dim=2)
    # Divide by lengths
    return summed / lengths.clamp(min=1).unsqueeze(-1).to(device)


def compute_lengths(x: torch.Tensor) -> torch.Tensor:
    """
    Computes effective length. x: (N, T)
    """
    valid = (x.abs() > 1e-8)
    last_valid = valid.flip(1).float().argmax(dim=1)
    lengths = x.size(1) - last_valid
    return lengths.clamp(min=1).cpu()


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
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        manual_dim = self.n_paths * self.f_path + self.f_user
        fused_dim = self.n_paths * self.n_ts * channels + manual_dim

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, hp["regressor_dim"]),
            nn.ReLU(),
            nn.Dropout(hp["dropout_reg"]),
            nn.Linear(hp["regressor_dim"], output_dim),
        )

    # -----------------------------------------------------

    def encode_timeseries(self, x_ts: torch.Tensor) -> torch.Tensor:
        # x_ts: (B, P, TS, T)
        B, P, TS, T = x_ts.shape

        # Flatten to compute lengths: (B*P*TS, T)
        flat_ts = x_ts.view(B * P * TS, T)
        lengths = compute_lengths(flat_ts)

        # Prepare for Conv1d: (N, C_in, T) -> (B*P*TS, 1, T)
        x = flat_ts.unsqueeze(1) 

        # Apply CNN
        # feat: (N, C_out, T)
        feat = self.cnn(x)

        # Global Average Pooling (masked by length)
        # pooled: (N, C_out)
        pooled = masked_mean(feat, lengths)

        # Reshape back: (B, P * TS * C_out)
        pooled = pooled.view(B, P * TS * pooled.shape[-1])
        return pooled

    # -----------------------------------------------------

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x

        ts_feat = self.encode_timeseries(x_ts)

        x_path_flat = x_path.reshape(x_path.size(0), -1)

        manual = torch.cat([
            x_path_flat,
            x_user
        ], dim=1)

        fused = torch.cat([ts_feat, manual], dim=1)
        return self.regressor(fused)

    # =====================================================
    # OPTUNA
    # =====================================================

    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            "cnn_channels": trial.suggest_categorical("cnn_channels", [16, 32, 48, 64]),
            "regressor_dim": trial.suggest_categorical("regressor_dim", [64, 128, 256]),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [6]),
            "correlation_threshold": trial.suggest_float("correlation_threshold", 0.4, 0.8, step=0.01)
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "cnn_channels": 32,
            "regressor_dim": 256,
            "dropout_reg": 0.2,
            "lr": 2e-4,
            "weight_decay": 1e-3,
            "batch_size": 6,
            "correlation_threshold": 0.5
        }
