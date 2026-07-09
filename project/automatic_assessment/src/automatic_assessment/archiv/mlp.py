import torch
import torch.nn as nn
from typing import Dict, Any, Union, List
from ..base import BaseModel

class SimpleMLP(BaseModel):
    model_name = "Simple_TimeAveraged_MLP"

    def __init__(self, input_dims: list, output_dim: int, hyperparams: Dict[str, Any]):
        super().__init__(input_dims, output_dim, hyperparams)
        
        # input_dims[0] is (N, P, F, T)
        ts_shape = input_dims[0]
        input_dim = ts_shape[1] * ts_shape[2]
        
        hidden_dim = hyperparams.get("hidden_dim", 64)
        dropout = hyperparams.get("dropout", 0.3)
        
        # 1. Collapse time dimension (Batch, Features, Time) -> (Batch, Features, 1) -> (Batch, Features)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 2. MLP
        self.mlp = nn.Sequential(
            nn.Flatten(), # (Batch, Features)
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Check for list/tuple input (new dataset structure)
        # Structure: [x_ts, x_path, x_user]
        if isinstance(x, (list, tuple)):
            x_ts = x[0] # (Batch, Paths, Features, Time)
            b, p, f, t = x_ts.shape
            # Flatten Paths and Features to match model expectation: (Batch, P*F, Time)
            x = x_ts.reshape(b, p*f, t)

        # x: (Batch, Features, Time)
        x = self.pool(x)
        return self.mlp(x)

    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1.0, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16])
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "lr": 0.001,
            "hidden_dim": 32,
            "dropout": 0.3,
            "weight_decay": 0.05,
            "batch_size": 16
        }