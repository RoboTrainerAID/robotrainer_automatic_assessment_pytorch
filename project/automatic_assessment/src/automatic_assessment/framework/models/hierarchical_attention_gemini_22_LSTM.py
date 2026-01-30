import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base import BaseModel


class HierarchicalTimeseriesLSTM(BaseModel):
    model_name = "HierarchicalTimeseriesLSTM"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        # ============================================================
        # Input Dimensions
        # ============================================================
        ts_shape = input_dims[0]
        self.n_paths = ts_shape[1]        # e.g. 20
        self.n_ts_per_path = ts_shape[2]  # e.g. 35
        self.ts_len = ts_shape[3]         # e.g. 158
        self.f_path = input_dims[1][2]    # e.g. 105
        self.f_user = input_dims[2][1]    # e.g. 2

        # ============================================================
        # Hyperparameters
        # ============================================================
        lstm_hidden_dim = hyperparams.get("lstm_hidden_dim", 32)
        lstm_layers = hyperparams.get("lstm_layers", 1)
        bidirectional = hyperparams.get("bidirectional", False)

        path_dim = hyperparams.get("path_dim", 16)
        regressor_dim = hyperparams.get("regressor_dim", 128)

        dropout_lstm = hyperparams.get("dropout_lstm", 0.0)
        dropout_path = hyperparams.get("dropout_path", 0.25)
        dropout_reg = hyperparams.get("dropout_reg", 0.25)

        self.path_aggregation = hyperparams.get("path_aggregation", "mean")

        self.last_attn_weights = None

        # ============================================================
        # 1. Shared Time-Series LSTM Encoder
        # ============================================================

        self.bidirectional = bidirectional
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(
            input_size=self.n_ts_per_path,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_lstm if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.ts_feat_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        # ============================================================
        # 2. Path-Level Integration
        # ============================================================

        total_path_input = self.ts_feat_dim + self.f_path

        self.path_bottleneck = nn.Sequential(
            nn.Linear(total_path_input, path_dim),
            nn.ReLU(),
            nn.Dropout(dropout_path)
        )

        # ============================================================
        # 3. Path Aggregation Mechanisms
        # ============================================================

        if self.path_aggregation == "static":
            self.path_weights = nn.Parameter(torch.ones(self.n_paths))

        elif self.path_aggregation == "attention":
            self.path_attention = nn.Sequential(
                nn.Linear(path_dim, path_dim),
                nn.Tanh(),
                nn.Linear(path_dim, 1)
            )

        # mean pooling requires no parameters

        # ============================================================
        # 4. User-Level Regressor
        # ============================================================

        total_user_input = path_dim + self.f_user

        self.regressor = nn.Sequential(
            nn.Linear(total_user_input, regressor_dim),
            nn.ReLU(),
            nn.Dropout(dropout_reg),
            nn.Linear(regressor_dim, output_dim)
        )

    # ================================================================
    # Forward
    # ================================================================
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x
        batch_size = x_ts.shape[0]

        # ------------------------------------------------------------
        # Step 1: LSTM feature extraction
        # ------------------------------------------------------------
        # Reshape:
        # (B, n_paths, n_ts_per_path, ts_len)
        # → (B * n_paths, ts_len, n_ts_per_path)
        # We need (Batch, Seq_Len, Input_Size) for batch_first=True
        x_ts = x_ts.permute(0, 1, 3, 2)  # B, n_paths, ts_len, n_ts_per_path
        x_ts = x_ts.reshape(batch_size * self.n_paths, self.ts_len, self.n_ts_per_path)

        lstm_out, (h_n, c_n) = self.lstm(x_ts)

        if self.bidirectional:
            # concatenate last forward and backward hidden states
            forward_hidden = h_n[-2]
            backward_hidden = h_n[-1]
            ts_feats = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            ts_feats = h_n[-1]

        # Reshape back to (B, n_paths, -1)
        ts_feats = ts_feats.view(batch_size, self.n_paths, -1)

        # ------------------------------------------------------------
        # Step 2: Path-Level Integration
        # ------------------------------------------------------------
        path_combined = torch.cat([ts_feats, x_path], dim=-1)
        path_feats = self.path_bottleneck(path_combined)

        # ------------------------------------------------------------
        # Step 3: Path Aggregation
        # ------------------------------------------------------------
        if self.path_aggregation == "static":
            weights = F.softmax(self.path_weights, dim=0)
            self.last_attn_weights = weights.unsqueeze(0).expand(batch_size, -1)
            path_aggr = torch.sum(
                path_feats * weights.view(1, self.n_paths, 1),
                dim=1
            )

        elif self.path_aggregation == "attention":
            attn_logits = self.path_attention(path_feats)
            attn_weights = F.softmax(attn_logits, dim=1)
            self.last_attn_weights = attn_weights.squeeze(-1).detach()
            path_aggr = torch.sum(path_feats * attn_weights, dim=1)

        else:  # mean pooling
            self.last_attn_weights = None
            path_aggr = torch.mean(path_feats, dim=1)

        # ------------------------------------------------------------
        # Step 4: User-Level Regression
        # ------------------------------------------------------------
        user_combined = torch.cat([path_aggr, x_user], dim=-1)
        prediction = self.regressor(user_combined)

        return prediction

    # ================================================================
    # Hyperparameter Search Space
    # ================================================================
    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {

            # Optimization
            "lr": trial.suggest_float("lr", 5e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),

            # LSTM architecture
            "lstm_hidden_dim": trial.suggest_categorical("lstm_hidden_dim", [8, 16, 32]),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
            "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
            "dropout_lstm": trial.suggest_float("dropout_lstm", 0.0, 0.3),

            # Path-level
            "path_dim": trial.suggest_categorical("path_dim", [8, 16, 32]),
            "dropout_path": trial.suggest_float("dropout_path", 0.1, 0.35),

            # Regressor
            "regressor_dim": trial.suggest_categorical("regressor_dim", [64, 128, 256, 512]),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.1, 0.35),

            # Aggregation mechanism
            "path_aggregation": trial.suggest_categorical(
                "path_aggregation", ["mean", "static", "attention"]
            ),

            # Batch Size
            "batch_size": trial.suggest_categorical("batch_size", [16])
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "lr": 0.00024131443163738766,
            "weight_decay": 0.002209466629569444,
            "lstm_hidden_dim": 16,
            "lstm_layers": 1,
            "bidirectional": False,
            "dropout_lstm": 0.15286409932413358,
            "path_dim": 16,
            "dropout_path": 0.3159279699167164,
            "regressor_dim": 256,
            "dropout_reg": 0.2796310905629553,
            "path_aggregation": "attention",
            "batch_size": 16
        }
