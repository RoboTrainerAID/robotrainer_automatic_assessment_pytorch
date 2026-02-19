import torch
import torch.nn as nn
from typing import Dict, Any, List

from .base import BaseModel


class MLPBaseline(BaseModel):
    model_name = "MLP_Baseline"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        # ======================
        # DATA DIMENSIONS
        # ======================

        ts_shape = input_dims[0]
        self.n_paths = ts_shape[1]

        path_shape = input_dims[1]
        self.f_path = path_shape[2]

        user_shape = input_dims[2]
        self.f_user = user_shape[1]

        # Flattened input size
        self.input_dim = self.n_paths * self.f_path + self.f_user

        # ======================
        # HYPERPARAMETERS
        # ======================

        n_layers = hyperparams.get("n_layers", 2)
        hidden_dim = hyperparams.get("hidden_dim", 64)
        shrink_factor = hyperparams.get("shrink_factor", 1.0)
        dropout = hyperparams.get("dropout", 0.1)

        # ======================
        # BUILD FLEXIBLE MLP
        # ======================

        layers = []

        in_dim = self.input_dim
        current_dim = hidden_dim

        for i in range(n_layers):

            layers.append(nn.Linear(in_dim, current_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            in_dim = current_dim
            current_dim = max(4, int(current_dim * shrink_factor))

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    # ======================
    # FORWARD
    # ======================

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:

        _, x_path, x_user = x

        B = x_path.shape[0]

        # Flatten path dimension
        x_path_flat = x_path.view(B, -1)

        # Concatenate user features
        x_all = torch.cat([x_path_flat, x_user], dim=1)

        prediction = self.mlp(x_all)

        return prediction

    # ======================
    # HYPERPARAMETERS
    # ======================

    @staticmethod
    def get_hyperparameter_space(trial):

        return {
            "n_layers": trial.suggest_int("n_layers", 1, 4),

            "hidden_dim": trial.suggest_categorical(
                "hidden_dim",
                [32, 64, 128, 256]
            ),

            "shrink_factor": trial.suggest_categorical(
                "shrink_factor",
                [1.0, 0.75, 0.5]
            ),

            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            # "n_path_features": trial.suggest_int("n_path_features", 100, 500, step=10),
            "correlation_threshold": trial.suggest_float("correlation_threshold", 0.4, 0.8, step=0.01)
        }

    @staticmethod
    def get_default_parameters():

        return {
            "n_layers": 2,
            "hidden_dim": 64,
            "shrink_factor": 0.75,
            "dropout": 0.1,
            # "n_path_features": 100,
            "correlation_threshold": 0.5
        }

