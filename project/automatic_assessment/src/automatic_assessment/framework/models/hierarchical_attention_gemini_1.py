import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base import BaseModel

class HierarchicalTimeseriesGemini1(BaseModel):
    model_name = "HierarchicalTimeseriesGemini1"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        # input_dims example: [(B, 20, 20, 79), (B, 20, 110), (B, 2)]
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
        
        # 1. SHARED Time-Series Extractor (Very small to prevent overfitting)
        # We treat all 20*20 timeseries with the same weights
        ts_layers = []
        in_channels = 1
        for i in range(num_conv_layers):
            # First layer matches original (k=5, s=2), others preserve dims (k=3, s=1, p=1)
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
        # Each path has 20 TS. Each TS gives ts_feat_dim features
        # Plus 110 path features
        # We use a shared MLP across all 20 paths.
        total_path_input = (self.n_ts_per_path * self.ts_feat_dim) + self.f_path
        self.path_bottleneck = nn.Sequential(
            nn.Linear(total_path_input, path_dim),
            nn.ReLU(),
            nn.Dropout(hyperparams.get('dropout', 0.2))
        )
        
        # 3. User-Level Integration
        # We aggregate the 20 paths (20 * path_dim features) or use Global Average
        # Let's use Global Average Pool to keep params low.
        total_user_input = path_dim + self.f_user
        
        self.regressor = nn.Sequential(
            nn.Linear(total_user_input, regressor_dim),
            nn.ReLU(),
            nn.Linear(regressor_dim, output_dim) # output_dim = 4 (or 14)
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x[0], x[1], x[2]
        batch_size = x_ts.shape[0]

        # Step 1: Extract TS features
        # Reshape to (Batch * Paths * TS_count, 1, 79) to process all at once
        x_ts = x_ts.view(-1, 1, self.ts_len)
        ts_feats = self.ts_extractor(x_ts) # (B*20*20, ts_feat_dim)
        
        # Step 2: Merge into Path level
        ts_feats = ts_feats.view(batch_size, self.n_paths, -1) # (B, 20, ts_feat_dim * 20)
        path_combined = torch.cat([ts_feats, x_path], dim=-1) # (B, 20, ts_feat_dim * 20 + 110)
        
        path_feats = self.path_bottleneck(path_combined) # (B, 20, path_dim)
        
        # Step 3: Aggregate Paths (Global Average Pooling across paths)
        # This makes the model "path-order agnostic" and saves params
        path_aggr = torch.mean(path_feats, dim=1) # (B, path_dim)
        
        # Step 4: Final User Integration & Regression
        user_combined = torch.cat([path_aggr, x_user], dim=-1) # (B, path_dim + 2)
        prediction = self.regressor(user_combined)
        
        return prediction
    
    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "ts_out_channels": trial.suggest_categorical("ts_out_channels", [8, 16, 32]),
            "num_conv_layers": trial.suggest_int("num_conv_layers", 1, 3),
            "path_dim": trial.suggest_categorical("path_dim", [4, 8, 16, 32]),
            "regressor_dim": trial.suggest_categorical("regressor_dim", [16, 32, 64, 128]),
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