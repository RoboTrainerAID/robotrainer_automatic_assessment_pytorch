import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from ..base import BaseModel
from .masking import mask_lengths
from ...data.schema import split_inputs, group_shapes


class HierarchicalTimeseriesLSTM(BaseModel):
    model_name = "LSTM_v2_masked"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        # ============================================================
        # Input Dimensions
        # ============================================================
        path_shape, user_shape, groups = group_shapes(input_dims)
        self.n_paths = path_shape[1]
        self.f_path = path_shape[2]
        self.f_user = user_shape[1]
        # total number of channels across all groups (shared 1-ch LSTM)
        self.n_ts_per_path = sum(g_x_shape[2] for (g_x_shape, _) in groups)

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

        self.bidirectional = bidirectional
        self.lstm_hidden_dim = lstm_hidden_dim

        # ============================================================
        # 1. Shared Timeseries Encoder (input_size = 1)
        # ============================================================
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_lstm if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.ts_feat_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        # Timeseries → path aggregation
        self.ts_pool = nn.Sequential(
            nn.Linear(self.ts_feat_dim, self.ts_feat_dim),
            nn.ReLU()
        )

        # ============================================================
        # 2. Path Integration
        # ============================================================
        total_path_input = self.ts_feat_dim + self.f_path

        self.path_bottleneck = nn.Sequential(
            nn.Linear(total_path_input, path_dim),
            nn.ReLU(),
            nn.Dropout(dropout_path)
        )

        # ============================================================
        # 3. Path Aggregation
        # ============================================================
        if self.path_aggregation == "static":
            self.path_weights = nn.Parameter(torch.ones(self.n_paths))

        elif self.path_aggregation == "attention":
            self.path_attention = nn.Sequential(
                nn.Linear(path_dim, path_dim),
                nn.Tanh(),
                nn.Linear(path_dim, 1)
            )

        # ============================================================
        # 4. User Regressor
        # ============================================================
        total_user_input = path_dim + self.f_user

        self.regressor = nn.Sequential(
            nn.Linear(total_user_input, regressor_dim),
            nn.ReLU(),
            nn.Dropout(dropout_reg),
            nn.Linear(regressor_dim, output_dim)
        )

        self.last_attn_weights = None

    # ================================================================
    # Forward
    # ================================================================
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_path, x_user, groups = split_inputs(x)
        B = x_path.shape[0]

        # ------------------------------------------------------------
        # Shared single-channel LSTM over EVERY channel of EVERY group.
        # Groups have different rates/lengths — each channel is packed
        # to its own mask-derived length, so the recurrence skips
        # padding entirely.
        # ------------------------------------------------------------
        all_ts_feats = []

        for x_g, mask_g in groups:
            T = x_g.shape[3]
            n_channels = x_g.shape[2]

            for ts_idx in range(n_channels):

                ts = x_g[:, :, ts_idx, :]         # (B, paths, len)
                ts = ts.reshape(-1, T)            # (B*paths, len)

                # per-channel validity mask (never infer lengths from values!)
                ch_mask = mask_g[:, :, ts_idx, :].reshape(-1, T)
                lengths = mask_lengths(ch_mask)
                ts = ts.unsqueeze(-1)

                packed = nn.utils.rnn.pack_padded_sequence(
                    ts, lengths, batch_first=True, enforce_sorted=False
                )

                _, (h_n, _) = self.lstm(packed)

                if self.bidirectional:
                    forward_hidden = h_n[-2]
                    backward_hidden = h_n[-1]
                    feat = torch.cat([forward_hidden, backward_hidden], dim=1)
                else:
                    feat = h_n[-1]

                feat = feat.view(B, self.n_paths, -1)
                all_ts_feats.append(feat)

        ts_feats = torch.stack(all_ts_feats, dim=2)
        ts_feats = ts_feats.mean(dim=2)

        # ------------------------------------------------------------
        # Path integration
        # ------------------------------------------------------------
        path_combined = torch.cat([ts_feats, x_path], dim=-1)
        path_feats = self.path_bottleneck(path_combined)

        # ------------------------------------------------------------
        # Path aggregation
        # ------------------------------------------------------------
        if self.path_aggregation == "static":
            weights = F.softmax(self.path_weights, dim=0)
            self.last_attn_weights = weights.unsqueeze(0).expand(B, -1)
            path_aggr = torch.sum(
                path_feats * weights.view(1, self.n_paths, 1),
                dim=1
            )

        elif self.path_aggregation == "attention":
            attn_logits = self.path_attention(path_feats)
            attn_weights = F.softmax(attn_logits, dim=1)
            self.last_attn_weights = attn_weights.squeeze(-1).detach()
            path_aggr = torch.sum(path_feats * attn_weights, dim=1)

        else:
            self.last_attn_weights = None
            path_aggr = torch.mean(path_feats, dim=1)

        # ------------------------------------------------------------
        # User regression
        # ------------------------------------------------------------
        user_combined = torch.cat([path_aggr, x_user], dim=-1)
        prediction = self.regressor(user_combined)

        return prediction

    # ================================================================
    # Hyperparameter Space
    # ================================================================
    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            "lr": trial.suggest_float("lr", 5e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),

            "lstm_hidden_dim": trial.suggest_categorical("lstm_hidden_dim", [8, 16, 32]),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
            "bidirectional": trial.suggest_categorical("bidirectional", [False]), # True
            "dropout_lstm": trial.suggest_float("dropout_lstm", 0.0, 0.3),

            "path_dim": trial.suggest_categorical("path_dim", [8, 16, 32]),
            "dropout_path": trial.suggest_float("dropout_path", 0.2, 0.4),

            "regressor_dim": trial.suggest_categorical("regressor_dim", [128, 256, 512]),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.2, 0.4),

            "path_aggregation": trial.suggest_categorical(
                "path_aggregation", ["mean"] #, "static", "attention"
            ),

            "batch_size": trial.suggest_categorical("batch_size", [6]),
            "correlation_threshold": trial.suggest_float("correlation_threshold", 0.4, 0.8, step=0.01)
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "lr": 2e-4,
            "weight_decay": 1e-3,
            "lstm_hidden_dim": 8,
            "lstm_layers": 1,
            "bidirectional": False,
            "dropout_lstm": 0.1,
            "path_dim": 16,
            "dropout_path": 0.3,
            "regressor_dim": 256,
            "dropout_reg": 0.25,
            "path_aggregation": "mean",
            "batch_size": 6,
            "correlation_threshold": 0.5
        }

