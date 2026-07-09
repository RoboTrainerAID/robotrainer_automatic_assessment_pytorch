import torch
import torch.nn as nn
from typing import Dict, Any, List

from ..base import BaseModel

# Das slebe wie BASEBaseline aber mit keine aggregation sondern flattening


class BASEBaselineFLAT(BaseModel):
    model_name = "BASE_BaselineFLAT"

    def __init__(self, input_dims: list, output_dim: int, hyperparams: Dict[str, Any]) -> None:
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
            nn.ReLU(),
            nn.Dropout(hp["dropout_path"])
        )

        self.path_aggregation = hp["path_aggregation"]

        self.path_attention = nn.Sequential(
            nn.Linear(self.path_dim, self.path_dim),
            nn.Tanh(),
            nn.Linear(self.path_dim, 1)
        )

        # Regressor input depends on aggregation strategy:
        # - "mean" / "attention": (path_dim + f_user)
        # - "flatten": (path_dim * n_paths + f_user)
        if self.path_aggregation == "flatten":
            fused_dim = self.path_dim * self.n_paths + self.f_user
        else:
            fused_dim = self.path_dim + self.f_user

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, hp["regressor_dim"]),
            nn.ReLU(),
            nn.Dropout(hp["dropout_reg"]),
            nn.Linear(hp["regressor_dim"], output_dim),
        )

    # -----------------------------------------------------

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input list (x_path, x_user, *ts_groups)
                - x_path: (B, P, F)
                - x_user: (B, f_user)

        Returns:
            prediction: (B, output_dim)
        """
        # Input convention: (x_path, x_user, *ts_groups) — this model uses path+user only
        x_path, x_user = x[0], x[1]

        B = x_path.size(0)

        path_feat = self.path_bottleneck(x_path)

        if self.path_aggregation == "mean":
            path_global = path_feat.mean(dim=1)

        elif self.path_aggregation == "attention":
            att = self.path_attention(path_feat)
            att = torch.softmax(att, dim=1)
            path_global = (path_feat * att).sum(dim=1)

        elif self.path_aggregation == "flatten":
            # Flatten all path embeddings into a single vector: (B, P * path_dim)
            path_global = path_feat.view(B, self.n_paths * self.path_dim)

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
            "path_dim": trial.suggest_categorical("path_dim", [2, 4, 6, 8]),
            "dropout_path": trial.suggest_float("dropout_path", 0.1, 0.5),
            "path_aggregation": trial.suggest_categorical("path_aggregation", ["flatten"]), #"attention", "mean", 
            "regressor_dim": trial.suggest_categorical("regressor_dim", [384, 512, 640, 768]),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.2, 0.6),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 5e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [6]),
            # "correlation_threshold": trial.suggest_float("correlation_threshold", 0.1, 0.4, step=0.01),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "path_dim": 4,
            "dropout_path": 0.22655060308520866,
            "path_aggregation": "flatten",
            "regressor_dim": 512,
            "dropout_reg": 0.30335589862608825,
            "lr": 0.0008887240577211731,
            "weight_decay": 0.002222430039021746,
            "batch_size": 6,
            # "correlation_threshold": 0.1
            "n_path_features": 30,
        }
