import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base import BaseModel


# ============================================================
# Utility: Feature Aggregation (Path → User level)
# ============================================================

class PathFeatureAggregator(nn.Module):
    """
    Aggregates (B, n_paths, f_path) → (B, f_path)

    Modes:
        - mean        : simple average over paths
        - attention   : learned attention weights per path
    """

    def __init__(self, n_paths: int, f_path: int, mode: str = "mean"):
        super().__init__()
        self.mode = mode
        self.n_paths = n_paths
        self.last_attn_weights = None

        if mode == "attention":
            self.attn = nn.Sequential(
                nn.Linear(f_path, 1),
                nn.Tanh()
            )

    def forward(self, x_path):
        # x_path: (B, n_paths, f_path)

        if self.mode == "mean":
            self.last_attn_weights = None
            return torch.mean(x_path, dim=1)

        elif self.mode == "attention":
            scores = self.attn(x_path)          # (B, n_paths, 1)
            weights = torch.softmax(scores, dim=1)
            self.last_attn_weights = weights.squeeze(-1).detach()
            return torch.sum(x_path * weights, dim=1)

        else:
            raise ValueError(f"Unknown aggregation mode: {self.mode}")


# ============================================================
# 1️⃣ Dummy Mean Regressor
# ============================================================

class DummyMeanRegressor(BaseModel):
    model_name = "DummyMeanRegressor"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        self.output_dim = output_dim

        # Stored running mean
        self.register_buffer("running_mean", torch.zeros(output_dim))
        self.register_buffer("seen_samples", torch.tensor(0.0))

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # Input convention: (x_path, x_user, *ts_groups)
        batch_size = x[0].shape[0]

        # Always predict learned mean
        return self.running_mean.unsqueeze(0).expand(batch_size, -1)

    def update_running_mean(self, y: torch.Tensor):
        """
        Call this externally during training loop
        """
        batch_mean = torch.mean(y, dim=0)
        batch_size = y.shape[0]

        total = self.seen_samples + batch_size
        new_mean = (
            self.running_mean * self.seen_samples + batch_mean * batch_size
        ) / total

        self.running_mean = new_mean.detach()
        self.seen_samples = total.detach()

    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {}


# ============================================================
# 2️⃣ Linear Regression (OLS)
# ============================================================

class LinearRegressionModel(BaseModel):
    model_name = "LinearRegressionModel"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        # Input convention: (x_path, x_user, *ts_groups) — this model uses path+user only
        path_shape = input_dims[0]
        user_shape = input_dims[1]

        self.n_paths = path_shape[1]
        self.f_path = path_shape[2]
        self.f_user = user_shape[1]
        
        self.last_attn_weights = None

        aggregation = hyperparams.get("aggregation", "mean")

        self.aggregator = PathFeatureAggregator(
            self.n_paths, self.f_path, mode=aggregation
        )

        self.linear = nn.Linear(self.f_path + self.f_user, output_dim)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # Input convention: (x_path, x_user, *ts_groups) — this model uses path+user only
        x_path, x_user = x[0], x[1]

        path_features = self.aggregator(x_path)
        self.last_attn_weights = self.aggregator.last_attn_weights
        features = torch.cat([path_features, x_user], dim=-1)

        return self.linear(features)

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "aggregation": trial.suggest_categorical(
                "aggregation", ["mean", "attention"]
            ),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }

    @staticmethod
    def get_default_parameters():
        return {
            "aggregation": "mean",
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }


# ============================================================
# 3️⃣ ElasticNet-style Linear Model
# ============================================================

class ElasticNetModel(LinearRegressionModel):
    """
    Same architecture as LinearRegressionModel, trained with a combined
    L1 + L2 penalty (elastic net):

        penalty = alpha * (l1_ratio * ||w||_1 + 0.5 * (1 - l1_ratio) * ||w||_2^2)

    The penalty is added to the training loss via `regularization_loss()`,
    which the Trainer calls automatically when present. Biases are not
    penalized (standard elastic-net convention). `weight_decay` is fixed
    to 0.0 so AdamW's decoupled L2 does not double-penalize.
    """

    model_name = "ElasticNetModel"

    def regularization_loss(self) -> torch.Tensor:
        alpha = float(self.hyperparams.get("alpha", 0.0))
        l1_ratio = float(self.hyperparams.get("l1_ratio", 0.5))

        l1 = torch.tensor(0.0, device=next(self.parameters()).device)
        l2 = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, p in self.named_parameters():
            if not p.requires_grad or name.endswith("bias"):
                continue
            l1 = l1 + p.abs().sum()
            l2 = l2 + (p ** 2).sum()

        return alpha * (l1_ratio * l1 + 0.5 * (1.0 - l1_ratio) * l2)

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "aggregation": trial.suggest_categorical(
                "aggregation", ["mean", "attention"]
            ),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
            "weight_decay": trial.suggest_categorical("weight_decay", [0.0]),
        }

    @staticmethod
    def get_default_parameters():
        return {
            "aggregation": "mean",
            "lr": 1e-3,
            "alpha": 1e-2,
            "l1_ratio": 0.5,
            "weight_decay": 0.0,
        }


# ============================================================
# 4️⃣ RandomForest-like MLP (Tree Approximation)
# ============================================================

class RandomForestLikeMLP(BaseModel):
    """
    Small MLP approximating tree ensemble behavior.
    Shallow + regularized.
    """

    model_name = "RandomForestLikeMLP"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        # Input convention: (x_path, x_user, *ts_groups) — this model uses path+user only
        path_shape = input_dims[0]
        user_shape = input_dims[1]

        self.n_paths = path_shape[1]
        self.f_path = path_shape[2]
        self.f_user = user_shape[1]
        
        self.last_attn_weights = None

        hidden_dim = hyperparams.get("hidden_dim", 64)
        dropout = hyperparams.get("dropout", 0.2)

        self.aggregator = PathFeatureAggregator(
            self.n_paths, self.f_path,
            mode=hyperparams.get("aggregation", "mean")
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.f_path + self.f_user, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # Input convention: (x_path, x_user, *ts_groups) — this model uses path+user only
        x_path, x_user = x[0], x[1]

        path_features = self.aggregator(x_path)
        self.last_attn_weights = self.aggregator.last_attn_weights
        features = torch.cat([path_features, x_user], dim=-1)

        return self.mlp(features)

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "aggregation": trial.suggest_categorical(
                "aggregation", ["mean", "attention"]
            ),
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim", [32, 64, 128]
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-5, 1e-2, log=True
            ),
        }

    @staticmethod
    def get_default_parameters():
        return {
            "aggregation": "mean",
            "hidden_dim": 64,
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }


# ============================================================
# 5️⃣ Simple MLP Regressor (Stronger Baseline)
# ============================================================

class SimpleMLPRegressor(BaseModel):
    model_name = "SimpleMLPRegressor"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        # Input convention: (x_path, x_user, *ts_groups) — this model uses path+user only
        path_shape = input_dims[0]
        user_shape = input_dims[1]

        self.n_paths = path_shape[1]
        self.f_path = path_shape[2]
        self.f_user = user_shape[1]
        
        self.last_attn_weights = None

        hidden_dim = hyperparams.get("hidden_dim", 64)
        depth = hyperparams.get("depth", 2)
        dropout = hyperparams.get("dropout", 0.3)

        self.aggregator = PathFeatureAggregator(
            self.n_paths, self.f_path,
            mode=hyperparams.get("aggregation", "mean")
        )

        layers = []
        in_dim = self.f_path + self.f_user

        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Input convention: (x_path, x_user, *ts_groups) — this model uses path+user only
        x_path, x_user = x[0], x[1]

        path_features = self.aggregator(x_path)
        self.last_attn_weights = self.aggregator.last_attn_weights
        features = torch.cat([path_features, x_user], dim=-1)

        return self.mlp(features)

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "aggregation": trial.suggest_categorical(
                "aggregation", ["mean", "attention"]
            ),
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim", [32, 64, 128]
            ),
            "depth": trial.suggest_int("depth", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-5, 1e-2, log=True
            ),
        }

    @staticmethod
    def get_default_parameters():
        return {
            "aggregation": "mean",
            "hidden_dim": 64,
            "depth": 2,
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }
