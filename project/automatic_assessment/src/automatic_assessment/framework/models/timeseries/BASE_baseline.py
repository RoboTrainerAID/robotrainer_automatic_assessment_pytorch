import torch
import torch.nn as nn
from typing import Dict, Any, List

from ..base import BaseModel

# Das selbe wie die CNN oder LSTM architektur aber ohne CNN oder LSTM, also nur mit linearen Schichten. 
# Damit können wir testen, ob die Komplexität der CNN/LSTM Architekturen überhaupt nötig ist oder ob ein simpler linearer Ansatz schon ausreicht.


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


class BASEBaseline(BaseModel):
    model_name = "BASE_Baseline"

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

        self.path_dim = hp["path_dim"]

        self.path_bottleneck = nn.Sequential(
            nn.Linear(self.f_path, self.path_dim),
            nn.ReLU(),
            nn.Dropout(hp["dropout_path"])
        )

        self.path_aggregation = hp["path_aggregation"]

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

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x

        path_feat = self.path_bottleneck(x_path)

        if self.path_aggregation == "mean":
            path_global = path_feat.mean(dim=1)
        else:
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
            "path_dim": trial.suggest_categorical("path_dim", [6, 8, 12, 16]),
            "dropout_path": trial.suggest_float("dropout_path", 0.1, 0.4),
            "path_aggregation": trial.suggest_categorical("path_aggregation", ["attention", "mean"]),
            "regressor_dim": trial.suggest_categorical("regressor_dim", [32, 48, 64, 128, 256]),
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
            "path_dim": 12,
            "dropout_path": 0.2,
            "path_aggregation": "mean",
            "regressor_dim": 48,
            "dropout_reg": 0.2,
            "lr": 0.0008957476788727834,
            "weight_decay": 0.00954951015873937,
            "batch_size": 6,
            # "correlation_threshold": 0.1
            "n_path_features": 50,
        }
