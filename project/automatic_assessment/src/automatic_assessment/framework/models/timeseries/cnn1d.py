import torch
import torch.nn as nn
from ..base import BaseModel
from typing import List, Union

class CNN1D(BaseModel):
    model_name = "CNN1D_Temporal"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        # Extract features from x_ts dimensions
        # input_dims[0] is shape of x_ts: (N, P, F, T)
        ts_shape = input_dims[0]
        curr_dims = ts_shape[1] * ts_shape[2] # P * F
        
        layers = []
        n_layers = hyperparams.get("n_layers", 2)
        hidden_dim = hyperparams.get("hidden_dim", 64)
        kernel_size = hyperparams.get("kernel_size", 3)
        dropout = hyperparams.get("dropout", 0.2)

        for i in range(n_layers):
            layers.extend([
                nn.Conv1d(curr_dims if i==0 else hidden_dim, hidden_dim, 
                          kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.feature_extractor = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1) # Handles variable length
        self.regressor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # Structure: [x_ts, x_path, x_user]
        # We only use x_ts for this model
        x_ts = x[0]
        
        # x_ts shape: (Batch, Paths, Features, Time)
        # Flatten Paths and Features: (Batch, Paths, Features, Time) -> (Batch, Paths*Features, Time)
        b, p, f, t = x_ts.shape
        x_in = x_ts.reshape(b, p*f, t)
        
        # X shape: (Batch, Features, TimeseriesLength) from data loader
        x_out = self.feature_extractor(x_in)
        x_out = self.adaptive_pool(x_out).squeeze(-1)
        return self.regressor(x_out)

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16])
        }
    
    @staticmethod
    def get_default_parameters():
        return {
            "lr": 0.001,
            "hidden_dim": 32,
            "n_layers": 3,
            "kernel_size": 7,
            "dropout": 0.4,
            "weight_decay": 0.05,
            "batch_size": 16
        }