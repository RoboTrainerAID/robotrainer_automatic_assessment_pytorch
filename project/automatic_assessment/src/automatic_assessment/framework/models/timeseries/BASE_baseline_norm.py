import torch
import torch.nn as nn
from typing import Dict, Any, List

from ..base import BaseModel

# Das selbe wie die CNN oder LSTM architektur aber ohne CNN oder LSTM, also nur mit linearen Schichten.
# Damit können wir testen, ob die Komplexität der CNN/LSTM Architekturen überhaupt nötig ist oder ob ein simpler linearer Ansatz schon ausreicht.


class BASEBaselineNORM(BaseModel):
    model_name = "BASE_BaselineNORM"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        # Input convention: (x_path, x_user, *ts_groups) — this model uses path+user only
        path_shape = input_dims[0]
        self.n_paths = path_shape[1]
        self.f_path = path_shape[2]

        user_shape = input_dims[1]
        self.f_user = user_shape[1]

        hp = hyperparams

        self.path_dim = hp["path_dim"]

        self.path_bottleneck = nn.Sequential(
            nn.Linear(self.f_path, self.path_dim),
            nn.LayerNorm(self.path_dim),
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
        # Input convention: (x_path, x_user, *ts_groups) — this model uses path+user only
        x_path, x_user = x[0], x[1]

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
            "path_dim": trial.suggest_categorical("path_dim", [6, 8, 12, 16, 24]),
            "dropout_path": trial.suggest_float("dropout_path", 0.2, 0.5),
            "path_aggregation": trial.suggest_categorical("path_aggregation", ["mean"]), #"attention", 
            "regressor_dim": trial.suggest_categorical("regressor_dim", [64, 96, 128, 192]),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.2, 0.6),
            "lr": trial.suggest_float("lr", 5e-4, 5e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [6]),
            # "correlation_threshold": trial.suggest_float("correlation_threshold", 0.1, 0.4, step=0.01),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "path_dim": 12,
            "norm": "layer",
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
