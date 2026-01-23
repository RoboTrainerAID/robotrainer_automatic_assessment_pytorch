import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base import BaseModel

class HierarchicalTimeseriesGemini(BaseModel):
    model_name = "HierarchicalTimeseriesGemini"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        # Dimensions from your summary: [(140, 20, 19, 79), (140, 20, 110), (140, 2)]
        ts_shape = input_dims[0]
        self.n_paths = ts_shape[1]        # 20
        self.n_ts_per_path = ts_shape[2]   # 19
        self.ts_len = ts_shape[3]         # 79
        self.f_path = input_dims[1][2]    # 110
        self.f_user = input_dims[2][1]    # 2

        # Initialize storage for attention weights
        self.last_attn_weights = None
        
        # 1. SHARED TS ENCODER (Remains small)
        # Reduced to 4 channels to save parameters downstream
        self.ts_extractor = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=2), 
            nn.GroupNorm(2, 4), 
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.ts_feat_dim = 4 # 4 features per time-series
        
        # 2. PATH BOTTLE NECK (The biggest parameter saver)
        # Instead of 186 -> 32 -> 16, we go 186 -> 8.
        # This reduces params from ~6,500 to ~1,500.
        path_input_dim = (self.n_ts_per_path * self.ts_feat_dim) + self.f_path # 19*4 + 110 = 186
        self.path_projector = nn.Sequential(
            nn.Linear(path_input_dim, 8),
            nn.GELU()
        )
        
        # 3. LEARNABLE ATTENTION QUERY
        # Vector size matches the 8-dim path projection
        self.attention_query = nn.Parameter(torch.randn(8, 1))
        
        # 4. FINAL REGRESSOR HEAD
        # user_input_dim: 8 (from path attention) + 2 (user features) = 10
        self.regressor = nn.Sequential(
            nn.Linear(10, 8),
            nn.GELU(),
            nn.Dropout(p=hyperparams.get('dropout', 0.1)),
            nn.Linear(8, output_dim)
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x[0], x[1], x[2]
        batch_size = x_ts.shape[0]

        # Step 1: Shared TS Extraction
        # Shape: (Batch * 20 * 19, 1, 79)
        x_ts = x_ts.view(-1, 1, self.ts_len)
        ts_feats = self.ts_extractor(x_ts) 
        
        # Step 2: Path-Level Projection
        # Re-group: (Batch, 20 paths, 19*4 features)
        ts_feats = ts_feats.view(batch_size, self.n_paths, -1) 
        path_combined = torch.cat([ts_feats, x_path], dim=-1) # (B, 20, 186)
        path_embeddings = self.path_projector(path_combined) # (B, 20, 8)
        
        # Step 3: Attention Pooling
        # scores: (B, 20, 1)
        attn_scores = torch.matmul(path_embeddings, self.attention_query)
        attn_weights = F.softmax(attn_scores, dim=1)
        self.last_attn_weights = attn_weights
        
        # Weighted sum: (B, 8)
        path_aggr = torch.sum(path_embeddings * attn_weights, dim=1)
        
        # Step 4: Regressor
        user_combined = torch.cat([path_aggr, x_user], dim=-1) # (B, 10)
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
        return {"dropout": 0.1,
                "lr": 0.001,
                "weight_decay": 0.01,
        }