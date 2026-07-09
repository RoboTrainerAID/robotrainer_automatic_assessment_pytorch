import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from .base import BaseModel

class GatedAttention(nn.Module):
    """
    Gated Attention Mechanism (Ilse et al., 2018).
    """
    def __init__(self, in_features: int, hidden_dim: int = 8):
        super().__init__()
        # Input: (Batch, Paths, Features=in_features)
        # Output latent: (Batch, Paths, hidden_dim)
        self.attention_V = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid()
        )
        # Reduction to scalar weight per path: (Batch, Paths, 1)
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (Batch, Paths, Features)
        a_v = self.attention_V(x)
        a_u = self.attention_U(x)
        scores = self.attention_weights(a_v * a_u)
        alpha = F.softmax(scores, dim=1) # Softmax over Paths dimension to weight them
        context = torch.sum(alpha * x, dim=1) # Weighted sum: (Batch, Features)
        return context, alpha

class HierarchicalAttentionNetwork(BaseModel):
    model_name = "Hierarchical_Attention_Network"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        # Dimensions based on summary example:
        # ts_shape: (Batch, 20, 19, 79)
        # path_shape: (Batch, 20, 110)
        # user_shape: (Batch, 2)
        
        ts_shape = input_dims[0]
        self.n_paths = ts_shape[1]
        self.f_ts = ts_shape[2]      # 19 (Filtered features)
        
        path_shape = input_dims[1]
        self.f_path = path_shape[2]  # 110 (Based on summary) or 7 (PCA)
        
        user_shape = input_dims[2]
        self.f_user = user_shape[1]  # 2 (Demographics)
        
        # Hyperparameters
        hidden_ts = hyperparams.get("hidden_ts", 8)      # e.g., 32
        hidden_fusion = hyperparams.get("hidden_fusion", 16) # e.g., 8
        dropout_p = hyperparams.get("dropout", 0.3)

        # Storage for visualization
        self.last_attn_weights = None
        
        # 1. TS Encoder (No change)
        # Input: (Batch * Paths, f_ts, Time) -> (B*20, 19, 79)
        # Output: (Batch * Paths, hidden_ts * 2, Time) -> (B*20, 64, 79)
        self.ts_encoder = nn.Sequential(
            nn.Conv1d(self.f_ts, hidden_ts, kernel_size=5, padding=2),
            nn.GroupNorm(1, hidden_ts), # LayerNorm equivalent for (N, C, L)
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_ts, hidden_ts * 2, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_ts * 2), # LayerNorm equivalent
            nn.ReLU(),
        )
        self.ts_out_dim = hidden_ts * 2
        
        # 2. Path Projection -> REMOVED
        # We will use self.f_path (e.g., 110) directly.
        
        # 3. Path Mixer (Fusion)
        # Input: ts_out_dim (64) + f_path (110) = 174
        # Output: hidden_fusion (8)
        self.fusion_dim = self.ts_out_dim + self.f_path
        self.path_mixer = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_fusion),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        
        # 4. Attention (Restored)
        # Input: hidden_fusion (8). Internal Hidden: hidden_fusion // 2 (4).
        # Output: Context Vector (8)
        self.attention = GatedAttention(hidden_fusion, hidden_dim=hidden_fusion // 2)
        
        # 5. MLP Regressor (Configurable)
        # Input: Attention Context (8) + User (2) = 10
        final_dim = hidden_fusion + self.f_user
        
        # Strategy: Start with input dimension and half it for each subsequent layer
        # until we hit the output dimension.
        n_regressor_layers = hyperparams.get("n_regressor_layers", 1)
        
        layers = []
        curr_dim = final_dim
        
        for _ in range(n_regressor_layers):
            next_dim = curr_dim // 2
            
            # If halving would result in fewer neurons than output targets, stop halving.
            # We want to maintain at least output_dim capacity until the final layer.
            if next_dim < output_dim:
                break
                
            layers.append(nn.Linear(curr_dim, next_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            curr_dim = next_dim
            
        layers.append(nn.Linear(curr_dim, output_dim))
        
        self.regressor = nn.Sequential(*layers)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x[0], x[1], x[2]
        batch_size = x_ts.shape[0]
        
        # --- TS Encoding ---
        t_len = x_ts.shape[-1]
        # x_ts: (Batch, Paths, Features, Time)
        x_ts_flat = x_ts.view(batch_size * self.n_paths, self.f_ts, t_len) # (B*20, 19, 79)
        
        ts_feat = self.ts_encoder(x_ts_flat) # (B*20, 64, 79)
        ts_embed = F.adaptive_max_pool1d(ts_feat, 1).squeeze(2) # (B*P, 64)
        
        # --- Path Features (Direct) ---
        x_path_flat = x_path.view(batch_size * self.n_paths, self.f_path) # (B*P, 110)
        
        # --- Fusion ---
        # Concatenate directly: [64] + [110]
        combined = torch.cat([ts_embed, x_path_flat], dim=1) # (B*P, 174)
        path_rep = self.path_mixer(combined) # (B*P, 8)
        
        # --- Attention Aggregation ---
        path_rep_seq = path_rep.view(batch_size, self.n_paths, -1) # (B, 20, 8)
        user_rep, att_weights = self.attention(path_rep_seq) # (B, 8)

        # Store attention weights for visualization (detach to avoid graph retention)
        self.last_attn_weights = att_weights
        
        # --- Regression ---
        final_input = torch.cat([user_rep, x_user], dim=1) # (B, 8 + 2 = 10)
        prediction = self.regressor(final_input) # (B, Output)
        
        return prediction
    
    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            # --- Architecture (Capacity Control) ---
            # Keep dimensions TINY. 
            # 8 features is often enough to distinguish "stumbling" from "walking".
            "hidden_ts": trial.suggest_categorical("hidden_ts", [4, 8, 12, 16, 32]),
            
            # The bottleneck before attention. Keep it small.
            "hidden_fusion": trial.suggest_categorical("hidden_fusion", [8, 16, 32]),

            # --- Regularization (The "Anti-Overfitting" Knobs) ---
            # With N=28, standard dropout (0.2) is useless. 
            # You need high dropout to force redundancy.
            "dropout": trial.suggest_float("dropout", 0.3, 0.7),

            # --- Optimization ---
            # AdamW loves higher learning rates than SGD.
            # We search around your default of 1e-3.
            "lr": trial.suggest_float("lr", 5e-4, 1e-2, log=True),
            
            # CRITICAL: High weight decay is your main defense against noise.
            # We search from "Strong" (0.01) to "Very Strong" (0.1).
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1, log=True),

            # Batch Size Strategy:
            # Batch=4: High noise, good for escaping local minima in small data.
            # Batch=8: Balanced.
            # Avoid Batch=16/32 (which is >50% of your data) as it leads to sharp minima.
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
            
            # MLP Regressor Structure
            # Replaced explicit tuples with number of layers for automatic halving
            "n_regressor_layers": trial.suggest_int("n_regressor_layers", 1, 3)
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "batch_size": 32,           # Balanced
            "dropout": 0.3,            # High regularization
            "hidden_fusion": 8,        # Minimal capacity
            "hidden_ts": 32,             # Minimal capacity
            "lr": 0.005,               # Standard AdamW start
            "weight_decay": 0.015,      # Strong regularization
            "n_regressor_layers": 1     # Simple projection by default
        }