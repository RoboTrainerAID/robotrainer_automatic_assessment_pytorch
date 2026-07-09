import torch
import torch.nn as nn
from typing import Dict, Any, List

from ..base import BaseModel
from .masking import timestep_mask, masked_mean_over_time
from ...data.schema import split_inputs, group_shapes


class CNNBaseline(BaseModel):
    """
    One Conv1d encoder PER CHANNEL GROUP (mechanical / physiological /
    gait — see config.TS_MODEL_GROUPS). Groups keep their native sampling
    rates and lengths; each encoder pools over time with the group's
    validity mask, and the pooled group embeddings are concatenated
    before fusion with the path features.
    """

    model_name = "CNN_Baseline"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        path_shape, user_shape, groups = group_shapes(input_dims)
        self.n_paths = path_shape[1]
        self.f_path = path_shape[2]
        self.f_user = user_shape[1]
        self.n_groups = len(groups)

        hp = hyperparams
        channels = hp["cnn_channels"]

        # One CNN per group (input channels = group channel count)
        self.cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(g_x_shape[2], channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            for (g_x_shape, _) in groups
        ])

        self.ts_feat_dim = channels * self.n_groups

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
        Encodes every channel group with its own CNN and masked mean
        pooling, then concatenates the group embeddings.

        Args:
            groups: list of (x_g, mask_g) with x_g: (B, P, C_g, T_g)

        Returns:
            (B, P, cnn_channels * n_groups)
        """
        pooled_groups = []
        for cnn, (x_g, mask_g) in zip(self.cnns, groups):
            B, P, C, T = x_g.shape

            # per-timestep validity from the explicit mask (never from values!)
            t_mask = timestep_mask(mask_g).view(B * P, T)

            feat = cnn(x_g.reshape(B * P, C, T))
            feat = feat.permute(0, 2, 1)

            pooled = masked_mean_over_time(feat, t_mask)
            pooled_groups.append(pooled.view(B, P, -1))

        return torch.cat(pooled_groups, dim=-1)

    # -----------------------------------------------------

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_path, x_user, groups = split_inputs(x)

        ts_paths = self.encode_timeseries(groups)      # (B,P,ts_feat_dim)

        path_combined = torch.cat([ts_paths, x_path], dim=-1)
        path_feat = self.path_bottleneck(path_combined)

        if self.path_aggregation == "mean":
            path_global = path_feat.mean(dim=1)
        else:
            att = self.path_attention(path_feat)
            att = torch.softmax(att, dim=1)

            path_global = (path_feat * att).sum(dim=1)

        fused = torch.cat([path_global, x_user], dim=1)

        return self.regressor(fused)

    # =====================================================
    # OPTUNA
    # =====================================================

    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            "cnn_channels": trial.suggest_categorical("cnn_channels", [2, 4, 8]),
            "path_dim": trial.suggest_categorical("path_dim", [4, 6, 8, 12]),
            "dropout_path": trial.suggest_float("dropout_path", 0.1, 0.6),
            "path_aggregation": trial.suggest_categorical("path_aggregation", ["mean"]), #"attention",
            "regressor_dim": trial.suggest_categorical("regressor_dim", [128, 192, 256, 384, 512]),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.1, 0.6),
            "lr": trial.suggest_float("lr", 1e-5, 5e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 5e-4, 5e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [6]),
            # "correlation_threshold": trial.suggest_float("correlation_threshold", 0.1, 0.4, step=0.01),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "cnn_channels": 8,
            "path_dim": 12,
            "dropout_path": 0.2,
            "path_aggregation": "mean",
            "regressor_dim": 48,
            "dropout_reg": 0.2,
            "lr": 0.0008957476788727834,
            "weight_decay": 0.00954951015873937,
            "batch_size": 6,
            # "correlation_threshold": 0.1
            "n_path_features": 50,
        }
