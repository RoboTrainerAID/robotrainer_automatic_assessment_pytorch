import torch
import torch.nn as nn
from typing import Dict, Any, List

from ..base import BaseModel
from .masking import timestep_mask, masked_mean_over_time, mask_lengths
from ...data.schema import split_inputs, group_shapes


class LSTMBaseline(BaseModel):
    """
    One LSTM encoder PER CHANNEL GROUP (mechanical / physiological /
    gait — see config.TS_MODEL_GROUPS). Groups keep their native sampling
    rates and lengths. Sequences are packed with mask-derived lengths, so
    the recurrence skips the padded tails entirely (no wasted compute on
    padding); pooling uses the group's validity mask.
    """

    model_name = "LSTM_Baseline"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        path_shape, user_shape, groups = group_shapes(input_dims)
        self.n_paths = path_shape[1]
        self.f_path = path_shape[2]
        self.f_user = user_shape[1]
        self.n_groups = len(groups)

        hp = hyperparams

        self.lstm_hidden = hp["lstm_hidden"]

        # One LSTM per group (input size = group channel count)
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=g_x_shape[2],
                hidden_size=self.lstm_hidden,
                num_layers=hp["lstm_layers"],
                dropout=hp["dropout_lstm"] if hp["lstm_layers"] > 1 else 0.0,
                batch_first=True,
                bidirectional=False,
            )
            for (g_x_shape, _) in groups
        ])

        self.ts_feat_dim = self.lstm_hidden * self.n_groups

        self.path_dim = hp["path_dim"]

        self.path_bottleneck = nn.Sequential(
            nn.Linear(self.ts_feat_dim + self.f_path, self.path_dim),
            nn.ReLU(),
            nn.Dropout(hp["dropout_path"])
        )

        self.path_aggregation = hp["path_aggregation"]

        self.path_attention = nn.Sequential(
            nn.Linear(self.path_dim, self.path_dim),
            nn.Tanh(),
            nn.Linear(self.path_dim, 1)
        )

        fused_dim = self.path_dim + self.f_user

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, hp["regressor_dim"]),
            nn.ReLU(),
            nn.Dropout(hp["dropout_reg"]),
            nn.Linear(hp["regressor_dim"], output_dim),
        )

    # -----------------------------------------------------

    def encode_timeseries(self, groups) -> torch.Tensor:
        """
        Encodes every channel group with its own LSTM (packed to the
        mask-derived true lengths) and masked mean pooling, then
        concatenates the group embeddings.

        Args:
            groups: list of (x_g, mask_g) with x_g: (B, P, C_g, T_g)

        Returns:
            (B, P, lstm_hidden * n_groups)
        """
        pooled_groups = []
        for lstm, (x_g, mask_g) in zip(self.lstms, groups):
            B, P, C, T = x_g.shape

            # per-timestep validity from the explicit mask (never from values!)
            t_mask = timestep_mask(mask_g).view(B * P, T)
            lengths = mask_lengths(t_mask)

            # (B,P,C,T) -> (B*P, T, C)
            seq = x_g.permute(0, 1, 3, 2).reshape(B * P, T, C)

            packed = nn.utils.rnn.pack_padded_sequence(
                seq, lengths, batch_first=True, enforce_sorted=False,
            )

            out, _ = lstm(packed)

            padded, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=T,
            )

            pooled = masked_mean_over_time(padded, t_mask)
            pooled_groups.append(pooled.view(B, P, -1))

        return torch.cat(pooled_groups, dim=-1)

    # -----------------------------------------------------

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_path, x_user, groups = split_inputs(x)

        ts_paths = self.encode_timeseries(groups)               # (B,P,ts_feat_dim)

        path_combined = torch.cat([ts_paths, x_path], dim=-1)   # (B,P,ts+f_path)
        path_feat = self.path_bottleneck(path_combined)         # (B,P,path_dim)

        if self.path_aggregation == "mean":
            path_global = path_feat.mean(dim=1)
        else:
            att = self.path_attention(path_feat)
            att = torch.softmax(att, dim=1)

            path_global = (path_feat * att).sum(dim=1)          # (B,path_dim)

        fused = torch.cat([path_global, x_user], dim=1)

        return self.regressor(fused)

    # =====================================================
    # OPTUNA
    # =====================================================

    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            "lstm_hidden": trial.suggest_categorical("lstm_hidden", [6, 8, 12, 18]),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 1),
            "dropout_lstm": trial.suggest_float("dropout_lstm", 0.1, 0.4),
            "path_dim": trial.suggest_categorical("path_dim", [8, 12, 16]),
            "dropout_path": trial.suggest_float("dropout_path", 0.1, 0.6),
            "path_aggregation": trial.suggest_categorical("path_aggregation", ["mean"]), #"attention",
            "regressor_dim": trial.suggest_categorical("regressor_dim", [32, 48, 64]),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.1, 0.4),
            "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [6]),
            # "correlation_threshold": trial.suggest_float("correlation_threshold", 0.1, 0.4, step=0.01),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "lstm_hidden": 8,
            "lstm_layers": 1,
            "dropout_lstm": 0.2596053561405025,
            "path_dim": 12,
            "dropout_path": 0.30807648872337967,
            "path_aggregation": "mean",
            "regressor_dim": 48,
            "dropout_reg": 0.1727335218342718,
            "lr": 0.0009883405876509705,
            "weight_decay": 0.0009484230000347363,
            "batch_size": 6,
            # "correlation_threshold": 0.3,
            "n_path_features": 70,
        }
