import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base import BaseModel

class HierarchicalTimeseriesGemini2(BaseModel):
    model_name = "HierarchicalTimeseriesGemini2"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        ts_shape = input_dims[0]
        self.n_paths = ts_shape[1]      # 20
        self.n_ts_per_path = ts_shape[2] # 20
        self.ts_len = ts_shape[3]       # 79
        self.f_path = input_dims[1][2]  # 110
        self.f_user = input_dims[2][1]  # 2
        
        # Hyperparameters
        ts_out_channels = hyperparams.get("ts_out_channels", 4)
        num_conv_layers = hyperparams.get("num_conv_layers", 1)
        path_dim = hyperparams.get("path_dim", 16)
        regressor_dim = hyperparams.get("regressor_dim", 16)

        # Initialize storage for attention weights
        self.last_attn_weights = None
        
        # 1. Shared Time-Series Extractor (Very small)
        ts_layers = []
        in_channels = 1
        for i in range(num_conv_layers):
            k = 5 if i == 0 else 3
            s = 2 if i == 0 else 1
            p = 0 if i == 0 else 1
            ts_layers.append(nn.Conv1d(in_channels, ts_out_channels, kernel_size=k, stride=s, padding=p))
            ts_layers.append(nn.ReLU())
            in_channels = ts_out_channels
            
        ts_layers.append(nn.AdaptiveAvgPool1d(1))
        ts_layers.append(nn.Flatten())
        
        self.ts_extractor = nn.Sequential(*ts_layers)
        self.ts_feat_dim = ts_out_channels
        
        # 2. Path-Level Integration
        total_path_input = (self.n_ts_per_path * self.ts_feat_dim) + self.f_path
        self.path_bottleneck = nn.Sequential(
            nn.Linear(total_path_input, path_dim),
            nn.ReLU(),
            nn.Dropout(hyperparams.get('dropout', 0.2))
        )
        
        # 3. LEARNABLE PATH WEIGHTS
        # This vector learns which of the 20 paths are "important"
        self.path_weights = nn.Parameter(torch.ones(self.n_paths)) 
        
        # 4. User-Level Integration & Regressor
        total_user_input = path_dim + self.f_user
        self.regressor = nn.Sequential(
            nn.Linear(total_user_input, regressor_dim),
            nn.ReLU(),
            nn.Dropout(hyperparams.get('dropout', 0.1)),
            nn.Linear(regressor_dim, output_dim) 
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x[0], x[1], x[2]
        batch_size = x_ts.shape[0]

        # Step 1: Extract TS features (Shared across all paths/series)
        x_ts = x_ts.view(-1, 1, self.ts_len)
        ts_feats = self.ts_extractor(x_ts) 
        
        # Step 2: Merge into Path level
        ts_feats = ts_feats.view(batch_size, self.n_paths, -1) 
        path_combined = torch.cat([ts_feats, x_path], dim=-1) 
        path_feats = self.path_bottleneck(path_combined) # (B, 20, 16)
        
        # Step 3: Weighted Aggregation of Paths
        # Normalize weights so they sum to 1 (Softmax)
        normalized_weights = F.softmax(self.path_weights, dim=0)
        self.last_attn_weights = normalized_weights.detach().unsqueeze(0).expand(batch_size, -1)
        
        # Apply weights to each path: (B, 20, 16) * (20, 1) -> (B, 16)
        path_aggr = torch.sum(path_feats * normalized_weights.view(1, self.n_paths, 1), dim=1)
        
        # Step 4: Final User Integration & Regression
        user_combined = torch.cat([path_aggr, x_user], dim=-1)
        prediction = self.regressor(user_combined)
        
        return prediction
    
    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
            "ts_out_channels": trial.suggest_categorical("ts_out_channels", [4, 8, 16]),
            "num_conv_layers": trial.suggest_int("num_conv_layers", 1, 3),
            "path_dim": trial.suggest_categorical("path_dim", [8, 16, 32, 64]),
            "regressor_dim": trial.suggest_categorical("regressor_dim", [8, 16, 32, 64]),
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "dropout": 0.1,
            "lr": 0.00027,
            "ts_out_channels": 16,
            "num_conv_layers": 2,
            "path_dim": 8,
            "regressor_dim": 64,
            "weight_decay": 0.02,
            "batch_size": 16,
        }