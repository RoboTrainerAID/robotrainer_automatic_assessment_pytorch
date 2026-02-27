import torch
import torch.nn as nn
from typing import Dict, Any, List

from ..base import BaseModel


def masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    device = x.device
    mask = torch.arange(x.size(1), device=device)[None, :] < lengths[:, None]
    mask = mask.float().unsqueeze(-1)
    x = x * mask
    return x.sum(1) / lengths.clamp(min=1).unsqueeze(-1)


def compute_lengths_flat(x_flat: torch.Tensor) -> torch.Tensor:
    valid = (x_flat.abs() > 1e-8)
    last_valid = valid.flip(1).float().argmax(dim=1)
    lengths = x_flat.size(1) - last_valid
    return lengths.clamp(min=1)


class CNNBaselineNOEMBED(BaseModel):
    model_name = "CNN_BaselineNOEMBED"

    def __init__(self, input_dims: list, output_dim: int, hyperparams: Dict[str, Any]) -> None:
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
        self.channels = channels

        self.cnn = nn.Sequential(
            nn.Conv1d(self.n_ts, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.path_aggregation = hp["path_aggregation"]

        # Combined dim after concatenating CNN output with path features
        # No bottleneck: directly feed (channels + f_path) into aggregation/regressor
        self.path_combined_dim = channels + self.f_path

        self.path_attention = nn.Sequential(
            nn.Linear(self.path_combined_dim, self.path_combined_dim),
            nn.Tanh(),
            nn.Linear(self.path_combined_dim, 1)
        )

        # Regressor input depends on aggregation strategy:
        # - "mean" / "attention": (channels + f_path + f_user)
        # - "flatten": (channels + f_path) * n_paths + f_user
        if self.path_aggregation == "flatten":
            fused_dim = self.path_combined_dim * self.n_paths + self.f_user
        else:
            fused_dim = self.path_combined_dim + self.f_user

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, hp["regressor_dim"]),
            nn.ReLU(),
            nn.Dropout(hp["dropout_reg"]),
            nn.Linear(hp["regressor_dim"], output_dim),
        )

    # -----------------------------------------------------

    def encode_timeseries(self, x_ts: torch.Tensor) -> torch.Tensor:
        """
        Encodes the timeseries input using a CNN and masked mean pooling.

        Args:
            x_ts: (B, P, TS, T) tensor of timeseries data.

        Returns:
            pooled: (B, P, channels) tensor of CNN-encoded path features.
        """
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
        """
        Forward pass. Encodes timeseries with CNN, concatenates with path features,
        aggregates across paths and fuses with user features before regression.

        Args:
            x: List of tensors [x_ts, x_path, x_user]
                - x_ts:   (B, P, TS, T)
                - x_path: (B, P, F)
                - x_user: (B, f_user)

        Returns:
            prediction: (B, output_dim)
        """
        x_ts, x_path, x_user = x

        B = x_path.size(0)

        ts_paths = self.encode_timeseries(x_ts)        # (B, P, channels)

        # Directly concatenate CNN output with path features — no bottleneck
        path_combined = torch.cat([ts_paths, x_path], dim=-1)  # (B, P, channels + f_path)

        if self.path_aggregation == "mean":
            path_global = path_combined.mean(dim=1)

        elif self.path_aggregation == "attention":
            att = self.path_attention(path_combined)
            att = torch.softmax(att, dim=1)
            path_global = (path_combined * att).sum(dim=1)

        elif self.path_aggregation == "flatten":
            # Flatten all path embeddings into a single vector: (B, P * (channels + f_path))
            path_global = path_combined.view(B, self.n_paths * self.path_combined_dim)

        else:
            raise ValueError(f"Invalid path_aggregation: '{self.path_aggregation}'. Choose 'mean', 'attention' or 'flatten'.")

        fused = torch.cat([path_global, x_user], dim=1)

        return self.regressor(fused)

    # =====================================================
    # OPTUNA
    # =====================================================

    @staticmethod
    def get_hyperparameter_space(trial: Any) -> Dict[str, Any]:
        return {
            "cnn_channels": trial.suggest_categorical("cnn_channels", [4, 8, 12]),
            "path_aggregation": trial.suggest_categorical("path_aggregation", ["mean", "flatten"]),
            "regressor_dim": trial.suggest_categorical("regressor_dim", [32, 64, 128, 256]),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 5e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [6]),
            # "correlation_threshold": trial.suggest_float("correlation_threshold", 0.1, 0.4, step=0.01),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "cnn_channels": 8,
            "path_aggregation": "mean",
            "regressor_dim": 48,
            "dropout_reg": 0.2,
            "lr": 0.0008957476788727834,
            "weight_decay": 0.00954951015873937,
            "batch_size": 6,
            # "correlation_threshold": 0.1
            "n_path_features": 50,
        }
