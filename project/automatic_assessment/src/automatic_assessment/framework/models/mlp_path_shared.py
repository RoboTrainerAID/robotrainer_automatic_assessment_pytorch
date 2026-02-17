import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base import BaseModel


class MLPSharedEncoder(BaseModel):
    model_name = "MLP_SharedEncoder"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        # ========================
        # DATA DIMENSIONS
        # ========================

        ts_shape = input_dims[0]
        self.n_paths = ts_shape[1]

        path_shape = input_dims[1]
        self.f_path = path_shape[2]

        user_shape = input_dims[2]
        self.f_user = user_shape[1]

        # ========================
        # HYPERPARAMETERS
        # ========================

        self.embed_dim = hyperparams.get("embed_dim", 8)
        self.hidden_dim = hyperparams.get("hidden_dim", 16)
        self.reg_hidden_dim = hyperparams.get("reg_hidden_dim", 16)

        self.pooling = hyperparams.get("pooling", "mean")

        self.dropout = hyperparams.get("dropout", 0.1)

        # ========================
        # SHARED PATH ENCODER
        # ========================

        self.path_encoder = nn.Sequential(
            nn.Linear(self.f_path, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embed_dim),
            nn.ReLU()
        )

        # ========================
        # ATTENTION POOLING
        # ========================

        if self.pooling == "attention":

            self.attention = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Tanh(),
                nn.Linear(self.embed_dim, 1)
            )

        # ========================
        # REGRESSION HEAD
        # ========================

        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim + self.f_user, self.reg_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.reg_hidden_dim, output_dim)
        )

    # ========================
    # FORWARD
    # ========================

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:

        _, x_path, x_user = x

        B, P, F = x_path.shape

        # Flatten paths into batch
        x_flat = x_path.view(B * P, F)

        embeddings = self.path_encoder(x_flat)

        embeddings = embeddings.view(B, P, self.embed_dim)

        # ========================
        # POOLING
        # ========================

        if self.pooling == "mean":

            pooled = embeddings.mean(dim=1)

        elif self.pooling == "attention":

            attn_scores = self.attention(embeddings)

            attn_weights = torch.softmax(attn_scores, dim=1)

            pooled = torch.sum(attn_weights * embeddings, dim=1)

        else:
            raise ValueError("Invalid pooling")

        # ========================
        # FUSION
        # ========================

        fused = torch.cat([pooled, x_user], dim=1)

        prediction = self.regressor(fused)

        return prediction

    # ========================
    # HYPERPARAMETERS
    # ========================

    @staticmethod
    def get_hyperparameter_space(trial):

        return {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 6, 8, 16]),
            "embed_dim": trial.suggest_categorical("embed_dim", [4, 8, 16]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [8, 16, 32]),
            "reg_hidden_dim": trial.suggest_categorical("reg_hidden_dim", [8, 16, 32]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "pooling": trial.suggest_categorical("pooling", ["mean", "attention"]),
        }

    @staticmethod
    def get_default_parameters():

        return {
            "lr": 0.0002,
            "weight_decay": 0.002,
            "batch_size": 8,
            "embed_dim": 8,
            "hidden_dim": 16,
            "reg_hidden_dim": 16,
            "dropout": 0.1,
            "pooling": "mean",
        }
