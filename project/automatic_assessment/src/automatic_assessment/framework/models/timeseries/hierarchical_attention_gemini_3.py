import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base import BaseModel

class HierarchicalTimeseriesGemini3(BaseModel):
    model_name = "HierarchicalTimeseriesGemini3"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        # Dimensions
        ts_shape = input_dims[0]
        self.n_paths = ts_shape[1]      # 20
        self.n_ts_per_path = ts_shape[2] # 20
        self.ts_len = ts_shape[3]       # 79
        self.f_path = input_dims[1][2]  # 110
        self.f_user = input_dims[2][1]  # 2

        # Initialize storage for attention weights
        self.last_attn_weights = None
        
        # 1. SHARED TS ENCODER: Using GroupNorm (4 groups for 8 channels)
        # No dropout here as per Tier-2 advice.
        self.ts_extractor = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2), 
            nn.GroupNorm(2, 8), # 2 groups of 4 channels
            nn.GELU(),
            nn.Conv1d(8, 4, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.ts_feat_dim = 4
        
        # 2. PATH BOTTLE NECK (Early Compression)
        path_input_dim = (self.n_ts_per_path * self.ts_feat_dim) + self.f_path
        self.path_mlp = nn.Sequential(
            nn.Linear(path_input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16) # Compress to 16-dim path embeddings
        )
        
        # 3. PATH ATTENTION (Tier-1.2 Recommendation)
        # Learns a query vector to score the importance of each path
        self.attention_query = nn.Parameter(torch.randn(16, 1))
        self.attn_dropout = nn.Dropout(p=hyperparams.get('dropout', 0.1))

        # 4. FINAL REGRESSOR HEAD
        user_input_dim = 16 + self.f_user
        self.regressor = nn.Sequential(
            nn.Linear(user_input_dim, 16),
            nn.GELU(),
            nn.Dropout(p=hyperparams.get('dropout', 0.1)),
            nn.Linear(16, output_dim)
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x[0], x[1], x[2]
        batch_size = x_ts.shape[0]

        # Step 1: Time Series Feature Extraction
        # Flatten paths and series into one big batch for shared processing
        x_ts = x_ts.view(-1, 1, self.ts_len) 
        ts_feats = self.ts_extractor(x_ts) 
        
        # Step 2: Path-Level merging
        ts_feats = ts_feats.view(batch_size, self.n_paths, -1) 
        path_combined = torch.cat([ts_feats, x_path], dim=-1) 
        path_embeddings = self.path_mlp(path_combined) # (B, 20, 16)
        
        # Step 3: Attention Pooling over Paths
        # Score each path's embedding against the learnable query
        # scores shape: (B, 20, 1)
        attn_scores = torch.matmul(path_embeddings, self.attention_query)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = self.attn_dropout(attn_weights) # Mild dropout on weights
        self.last_attn_weights = attn_weights
        
        # Weighted sum of path embeddings: (B, 16)
        path_aggr = torch.sum(path_embeddings * attn_weights, dim=1)
        
        # Step 4: User-Level Integration & Regression
        user_combined = torch.cat([path_aggr, x_user], dim=-1)
        return self.regressor(user_combined)

    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16])
        }
    
    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "dropout": 0.2,
            "lr": 0.001,
            "weight_decay": 0.01,
        }