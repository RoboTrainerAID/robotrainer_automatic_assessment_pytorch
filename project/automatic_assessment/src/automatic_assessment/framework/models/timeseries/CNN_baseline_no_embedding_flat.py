import torch
import torch.nn as nn
from typing import Dict, Any, List

from ..base import BaseModel
from .masking import timestep_mask, masked_mean_over_time
from ...data.schema import split_inputs, group_shapes


class CNNBaselineNOEMBEDFLAT(BaseModel):
    model_name = "CNN_BaselineNOEMBEDFLAT"

    def __init__(self, input_dims: list, output_dim: int, hyperparams: Dict[str, Any]) -> None:
        super().__init__(input_dims, output_dim, hyperparams)

        path_shape, user_shape, groups = group_shapes(input_dims)
        self.n_paths = path_shape[1]
        self.f_path = path_shape[2]
        self.f_user = user_shape[1]
        self.n_groups = len(groups)

        hp = hyperparams
        channels = hp["cnn_channels"]
        self.channels = channels

        # One CNN per channel group (mechanical / physiological / gait)
        self.cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(g_x_shape[2], channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            for (g_x_shape, _) in groups
        ])

        self.path_aggregation = hp["path_aggregation"]

        # Combined dim after concatenating CNN outputs with path features
        # No bottleneck: directly feed (channels*n_groups + f_path) into aggregation/regressor
        self.path_combined_dim = channels * self.n_groups + self.f_path

        self.path_attention = nn.Sequential(
            nn.Linear(self.path_combined_dim, self.path_combined_dim),
            nn.Tanh(),
            nn.Linear(self.path_combined_dim, 1)
        )

        # Regressor input depends on aggregation strategy:
        # - "mean" / "attention": (channels + f_path + f_user)
        # - "flatten": (channels + f_path) * n_paths + f_user
        if self.path_aggregation == "flatten":
            fused_dim = self.path_combined_dim * self.n_paths + self.f_user
        else:
            fused_dim = self.path_combined_dim + self.f_user

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
            pooled: (B, P, channels * n_groups) tensor of CNN-encoded path features.
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
        """
        Forward pass. Encodes timeseries with CNN, concatenates with path features,
        aggregates across paths and fuses with user features before regression.

        Args:
            x: Input list (x_path, x_user, g0_x, g0_mask, ...)
                - x_path: (B, P, F)
                - x_user: (B, f_user)

        Returns:
            prediction: (B, output_dim)
        """
        x_path, x_user, groups = split_inputs(x)

        B = x_path.size(0)

        ts_paths = self.encode_timeseries(groups)      # (B, P, channels*n_groups)

        # Directly concatenate CNN output with path features — no bottleneck
        path_combined = torch.cat([ts_paths, x_path], dim=-1)  # (B, P, channels + f_path)

        if self.path_aggregation == "mean":
            path_global = path_combined.mean(dim=1)

        elif self.path_aggregation == "attention":
            att = self.path_attention(path_combined)
            att = torch.softmax(att, dim=1)
            path_global = (path_combined * att).sum(dim=1)

        elif self.path_aggregation == "flatten":
            # Flatten all path embeddings into a single vector: (B, P * (channels + f_path))
            path_global = path_combined.view(B, self.n_paths * self.path_combined_dim)

        else:
            raise ValueError(f"Invalid path_aggregation: '{self.path_aggregation}'. Choose 'mean', 'attention' or 'flatten'.")

        fused = torch.cat([path_global, x_user], dim=1)

        return self.regressor(fused)

    # =====================================================
    # OPTUNA
    # =====================================================

    @staticmethod
    def get_hyperparameter_space(trial: Any) -> Dict[str, Any]:
        return {
            "cnn_channels": trial.suggest_categorical("cnn_channels", [8, 12, 16]),
            "path_aggregation": trial.suggest_categorical("path_aggregation", ["flatten"]),
            "regressor_dim": trial.suggest_categorical("regressor_dim", [256, 384, 512, 768]),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.2, 0.6),
            "lr": trial.suggest_float("lr", 5e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [6]),
            # "correlation_threshold": trial.suggest_float("correlation_threshold", 0.1, 0.4, step=0.01),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "cnn_channels": 8,
            "path_aggregation": "mean",
            "regressor_dim": 48,
            "dropout_reg": 0.2,
            "lr": 0.0008957476788727834,
            "weight_decay": 0.00954951015873937,
            "batch_size": 6,
            # "correlation_threshold": 0.1
            "n_path_features": 50,
        }
