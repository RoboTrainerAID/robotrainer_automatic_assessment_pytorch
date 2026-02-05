import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from .base import BaseModel

class HierarchicalTimeseriesGemini21(BaseModel):
    model_name = "HierarchicalTimeseriesGemini21"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        # Dimensions
        ts_shape = input_dims[0]
        self.n_paths = ts_shape[1]        # 20
        self.n_ts_sensors = ts_shape[2]   # 35 (multivariate channels)
        self.ts_len = ts_shape[3]         # 158
        
        self.f_path = input_dims[1][2]    # 105
        self.f_user = input_dims[2][1]    # 2
        
        # Hyperparameters
        ts_channels = hyperparams.get("ts_out_channels", 16)
        num_conv_layers = hyperparams.get("num_conv_layers", 1)
        path_dim = hyperparams.get("path_dim", 8)
        regressor_dim = hyperparams.get("regressor_dim", 128)
        dropout_rate = hyperparams.get("dropout", 0.26)
        
        # New: Attention Strategy
        # Options: "gated" (per-user dynamic), "static" (global fixed weights), "mean" (simple average)
        self.attn_type = hyperparams.get("path_attention_type", "mean") 
        # Store attention weights from last forward pass for external inspection
        self.last_attn_weights = None
        
        # 1. Dynamic Time-Series Extractor
        ts_layers = []
        in_c = self.n_ts_sensors
        
        for i in range(num_conv_layers):
            # First layer uses Kernel 5, subsequent use 3
            k = 5 if i == 0 else 3
            p = 2 if i == 0 else 1
            
            ts_layers.append(nn.Conv1d(in_c, ts_channels, kernel_size=k, padding=p))
            ts_layers.append(nn.BatchNorm1d(ts_channels))
            ts_layers.append(nn.ReLU())
            ts_layers.append(nn.Dropout(dropout_rate))
            in_c = ts_channels # Update for next layer

        # Global Max Pooling is applied in forward()
        self.ts_extractor = nn.Sequential(*ts_layers)
        self.ts_out_dim = ts_channels # Output of MaxPool is just channels
        
        # 2. Path-Level Integration (Bottleneck)
        total_path_input = self.ts_out_dim + self.f_path
        
        self.path_bottleneck = nn.Sequential(
            nn.Linear(total_path_input, path_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(path_dim * 2, path_dim),
            nn.ReLU()
        )
        
        # 3. Path Aggregation Mechanisms
        if self.attn_type == "gated":
            # Dynamic Gated Attention (MIL)
            self.attention_V = nn.Linear(path_dim, path_dim // 2)
            self.attention_U = nn.Linear(path_dim, path_dim // 2)
            self.attention_w = nn.Linear(path_dim // 2, 1)
            
        elif self.attn_type == "static":
            # Global Static Attention (One weight per path index 0..19)
            self.static_weights = nn.Parameter(torch.ones(self.n_paths))
            
        # "mean" requires no parameters
        
        # 4. Final Regressor
        total_user_input = path_dim + self.f_user
        self.regressor = nn.Sequential(
            nn.Linear(total_user_input, regressor_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(regressor_dim, output_dim)
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x[0], x[1], x[2]
        batch_size = x_ts.shape[0]

        # --- Step 1: TS Extraction ---
        # Flatten: (B*20, 35, 158)
        x_ts = x_ts.view(batch_size * self.n_paths, self.n_ts_sensors, self.ts_len)
        ts_feats = self.ts_extractor(x_ts) 
        
        # Global Max Pooling (Collapse Time)
        # (B*20, Channels, Time) -> (B*20, Channels)
        ts_embeds = F.adaptive_max_pool1d(ts_feats, 1).view(batch_size * self.n_paths, -1)
        
        # --- Step 2: Path Integration ---
        x_path_flat = x_path.view(batch_size * self.n_paths, -1)
        path_combined = torch.cat([ts_embeds, x_path_flat], dim=1)
        
        # Bottleneck: (B*20, path_dim)
        path_embeds = self.path_bottleneck(path_combined)
        
        # Reshape to (B, 20, path_dim)
        path_embeds = path_embeds.view(batch_size, self.n_paths, -1)
        
        # --- Step 3: Aggregation ---
        if self.attn_type == "gated":
            # Per-User Dynamic Attention
            att_v = torch.tanh(self.attention_V(path_embeds))
            att_u = torch.sigmoid(self.attention_U(path_embeds))
            att_scores = self.attention_w(att_v * att_u) # (B, 20, 1)
            att_weights = F.softmax(att_scores, dim=1)
            # Save attention weights (B, 20)
            self.last_attn_weights = att_weights.squeeze(-1)
            user_rep = torch.sum(path_embeds * att_weights, dim=1)
            
        elif self.attn_type == "static":
            # Global Static Weights
            weights = F.softmax(self.static_weights, dim=0) # (20)
            # Expand to per-batch weights and save (B, 20)
            weights_exp = weights.view(1, -1).repeat(batch_size, 1)
            self.last_attn_weights = weights_exp
            # Broadcasting: (B, 20, Dim) * (1, 20, 1)
            user_rep = torch.sum(path_embeds * weights.view(1, -1, 1), dim=1)
            
        else: # "mean"
            # Simple Average
            mean_w = 1.0 / float(self.n_paths)
            self.last_attn_weights = torch.full((batch_size, self.n_paths), mean_w, device=path_embeds.device)
            user_rep = torch.mean(path_embeds, dim=1)
        
        # --- Step 4: Regression ---
        final_input = torch.cat([user_rep, x_user], dim=1)
        return self.regressor(final_input)

    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            # Architecture
            # Centered on your optimal values: 16 channels, 1 layer
            "ts_out_channels": trial.suggest_categorical("ts_out_channels", [16]),
            "num_conv_layers": trial.suggest_int("num_conv_layers", 2, 2),
            
            # Bottleneck & Regressor
            # Your best was 8 and 128. We allow slight variation around that.
            "path_dim": trial.suggest_categorical("path_dim", [4, 8, 16]), 
            "regressor_dim": trial.suggest_categorical("regressor_dim", [32, 64, 128, 256]),
            
            # New Attention Parameter
            "path_attention_type": trial.suggest_categorical("path_attention_type", ["mean"]), #["gated", "static", "mean"]
            
            # Optimization
            # Centered on best: LR ~1.6e-4, Decay ~5e-3
            "lr": trial.suggest_float("lr", 5e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-2, log=True),
            
            # Regularization
            # Centered on best: 0.26
            "dropout": trial.suggest_float("dropout", 0.2, 0.5),
            
            # Batch Size
            "batch_size": trial.suggest_categorical("batch_size", [16])
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        # Your specific optimal configuration
        return {
            "batch_size": 16,
            "dropout": 0.2480023388377924,
            "lr": 0.00015565288829063982,
            "weight_decay": 0.0018028150752207402,
            "ts_out_channels": 16,
            "num_conv_layers": 2,
            "path_dim": 4,
            "regressor_dim": 256,
            "path_attention_type": "mean"
        }