import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base import BaseModel


class MLPPathSpecific(BaseModel):
    model_name = "MLP_PathSpecific"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        # ========================
        # DATA DIMENSIONS
        # ========================

        # Input convention: (x_path, x_user, *ts_groups) — this model uses path+user only
        path_shape = input_dims[0]
        self.n_paths = path_shape[1]
        self.f_path = path_shape[2]

        user_shape = input_dims[1]
        self.f_user = user_shape[1]

        # ========================
        # HYPERPARAMETERS
        # ========================

        self.embed_dim = hyperparams.get("embed_dim", 8)
        self.hidden_dim = hyperparams.get("hidden_dim", 16)
        self.reg_hidden_dim = hyperparams.get("reg_hidden_dim", 16)

        self.pooling = hyperparams.get("pooling", "mean")  # "mean" or "attention"

        self.dropout = hyperparams.get("dropout", 0.1)

        # ========================
        # PATH-SPECIFIC ENCODERS
        # ========================

        self.path_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.f_path, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.embed_dim),
                nn.ReLU()
            )
            for _ in range(self.n_paths)
        ])

        # ========================
        # ATTENTION POOLING (optional)
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

        # Input convention: (x_path, x_user, *ts_groups) — this model uses path+user only
        x_path, x_user = x[0], x[1]

        B = x_path.shape[0]

        # Encode each path separately
        embeddings = []

        for i in range(self.n_paths):

            path_feat = x_path[:, i, :]
            emb = self.path_encoders[i](path_feat)

            embeddings.append(emb)

        embeddings = torch.stack(embeddings, dim=1)

        # shape: (B, n_paths, embed_dim)

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
        # FUSION WITH USER FEATURES
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
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [6]),
            "embed_dim": trial.suggest_categorical("embed_dim", [24, 32, 48, 64]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [8, 16, 32]),
            "reg_hidden_dim": trial.suggest_categorical("reg_hidden_dim", [48, 64, 96, 128]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "pooling": trial.suggest_categorical("pooling", ["mean"]), #,"attention"
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
            # "correlation_threshold": trial.suggest_float("correlation_threshold", 0.4, 0.8, step=0.01)
        }

    @staticmethod
    def get_default_parameters():

        return {
            "lr": 0.0013384961181667743,
            "weight_decay": 0.0034329687492807917,
            "batch_size": 6,
            "embed_dim": 8,
            "hidden_dim": 16,
            "reg_hidden_dim": 16,
            "dropout": 0.04462387071386768,
            "pooling": "mean",
            "n_path_features": 50
        }
