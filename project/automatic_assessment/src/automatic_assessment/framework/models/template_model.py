import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base import BaseModel
from .timeseries.masking import timestep_mask, masked_mean_over_time, mask_lengths
from ..data.schema import split_inputs, group_shapes


class ModelTemplate(BaseModel):
    model_name = "ModelTemplate"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        # X tuple convention: (x_path, x_user, g0_x, g0_mask, g1_x, g1_mask, ...)
        #     1. x_path: Path-Level Features
        #         - Shape: (n_samples, n_paths, n_path_features), e.g. (25, 20, 88)
        #     2. x_user: User-Level Features
        #         - Shape: (n_samples, n_user_features), e.g. (25, 2)
        #     3.+ Timeseries channel groups (config.TS_MODEL_GROUPS order:
        #         mechanical @ 50 Hz, physiological @ 2 Hz, gait @ 2 Hz).
        #         Each group is a PAIR of tensors:
        #         - g_x:    (n_samples, n_paths, n_group_channels, T_group) float32,
        #                   0.0 where no data
        #         - g_mask: same shape, bool, True where the bin holds data
        #         Groups have DIFFERENT rates/lengths and are not aligned.
        #
        # y: Targets — Shape: (n_samples, n_targets), e.g. (25, 8)

        path_shape, user_shape, groups = group_shapes(input_dims)
        self.n_paths = path_shape[1]
        self.f_path = path_shape[2]
        self.f_user = user_shape[1]
        self.group_channel_counts = [g_x_shape[2] for (g_x_shape, _) in groups]

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_path, x_user, groups = split_inputs(x)
        batch_size = x_path.shape[0]

        # Example per-group processing:
        # for x_g, mask_g in groups:
        #     B, P, C, T = x_g.shape
        #     t_mask = timestep_mask(mask_g).view(B * P, T)   # (B*P, T) validity
        #     lengths = mask_lengths(t_mask)                  # for pack_padded_sequence
        #     ...encode, then pool with masked_mean_over_time(feat, t_mask)

        prediction = None  # build your prediction here: (batch_size, output_dim)
        return prediction

    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {}
