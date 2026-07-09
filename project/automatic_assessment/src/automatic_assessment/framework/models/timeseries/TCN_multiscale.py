"""
MultiScaleTCN — multi-scale dilated temporal-convolution encoder inside the
simple dual-track topology that is tied with BASE_BaselineFLAT.

History / lesson (v1 "MultiScaleTCNMIL" 0.935, v2 1.009 val RMSE vs
BASE_FLAT 0.863): the novel machinery around the encoder — per-target MIL
attention over paths, coverage gating with missing embeddings, path-slot
embeddings, attentive statistics pooling, per-target heads — never earned
its complexity on 24 training users. v2's stability fixes (zero-init
attention, heavy LayerScale, lr 5e-4) flattened the loss landscape so much
that training became glacial: 15/25 folds were still improving at the
30-epoch cap while 8/25 never left the epoch-1 plateau and were executed
by patience-2 early stopping (bimodal best-epochs, near-dummy predictions).

v3 therefore keeps ONLY the core hypothesis — the multi-scale dilated
depthwise-separable encoder (stem at native rate for the tremor band,
masked downsampling, exponentially dilated residual blocks for seconds of
context) — and drops it into the exact topology of HybridCNNGRUFusion,
which ties BASE_FLAT (0.8636 vs 0.8628) and clearly beats it on the
disturbance targets:

  per-group encoder -> masked mean+std pooling
  -> [ts embeddings * LayerScale | coverage fractions | path features]
  -> feature dropout -> compressed-flatten track + mean track
  -> plain 2-layer regressor.

The two models are now a controlled encoder A/B: CNN+GRU (Hybrid) vs
dilated TCN (this file) in an otherwise identical, deliberately simple
harness. Masks drive all pooling/validity decisions (never values).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from ..base import BaseModel
from .masking import timestep_mask, masked_mean_std_over_time, masked_avg_pool1d
from ...data.schema import split_inputs, group_shapes


class _DilatedBlock(nn.Module):
    """Residual depthwise-separable Conv1d block with dilation."""

    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.depthwise = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(F.gelu(self.pointwise(self.depthwise(x))))


class MultiScaleTCN(BaseModel):
    model_name = "MultiScale_TCN"

    # Groups longer than this get the downsample stride after the stem
    # (in practice: only the 50 Hz mechanical group).
    DOWNSAMPLE_MIN_LEN = 500

    def __init__(self, input_dims: list, output_dim: int, hyperparams: Dict[str, Any]) -> None:
        super().__init__(input_dims, output_dim, hyperparams)

        path_shape, user_shape, groups = group_shapes(input_dims)
        self.n_paths = path_shape[1]
        self.f_path = path_shape[2]
        self.f_user = user_shape[1]
        self.n_groups = len(groups)

        hp = hyperparams
        ts_dim = hp.get("ts_dim", 16)
        kernel = hp.get("kernel_size", 5)
        n_blocks = hp.get("n_blocks", 3)
        stride = hp.get("downsample_stride", 4)
        flat_dim = hp.get("flat_dim", 8)

        # --- Per-group multi-scale encoder: stem at native rate (keeps the
        # tremor band) -> masked downsample -> dilated residual blocks ---
        self.stems = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.strides = []
        for (g_x_shape, _) in groups:
            c_g, t_g = g_x_shape[2], g_x_shape[3]
            stride_g = stride if t_g > self.DOWNSAMPLE_MIN_LEN else 1
            self.strides.append(stride_g)
            self.stems.append(nn.Conv1d(c_g, ts_dim, kernel_size=7, padding=3))
            t_ds = math.ceil(t_g / stride_g)
            group_blocks = []
            for i in range(n_blocks):
                # Exponential dilation 1, 4, 16, ...; capped so the receptive
                # span never exceeds the (downsampled) sequence.
                dilation = min(4 ** i, max(1, (t_ds - 1) // max(1, kernel - 1)))
                group_blocks.append(_DilatedBlock(ts_dim, kernel, dilation, hp.get("dropout_ts", 0.1)))
            self.blocks.append(nn.Sequential(*group_blocks))

        # LayerScale: ts embeddings + coverages fade in from a small factor,
        # so early training is dominated by the proven path features.
        ts_feat_dim = 2 * ts_dim * self.n_groups + self.n_groups
        self.ts_gamma = nn.Parameter(torch.full((ts_feat_dim,), 0.3))

        # Fused per-path vector: [scaled group embeddings | coverages | path
        # features] — no per-sample normalization (a LayerNorm here made the
        # sibling model overfit by epoch 2).
        self.path_combined_dim = ts_feat_dim + self.f_path

        # Feature dropout before BOTH aggregation tracks
        self.path_drop = nn.Dropout(hp.get("dropout_path", 0.15))

        # Compressed flatten track (BASE_FLAT's trick): compress first, then flatten
        self.flat_compress = nn.Sequential(
            nn.Linear(self.path_combined_dim, flat_dim),
            nn.ReLU(),
        )

        # Regressor sees BOTH tracks: flatten (path identity) + mean (winner route)
        fused_dim = flat_dim * self.n_paths + self.path_combined_dim + self.f_user
        regressor_dim = hp.get("regressor_dim", 384)
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, regressor_dim),
            nn.ReLU(),
            nn.Dropout(hp.get("dropout_reg", 0.5)),
            nn.Linear(regressor_dim, output_dim),
        )

    # -----------------------------------------------------

    def encode_timeseries(self, groups) -> torch.Tensor:
        """
        Per group: multi-scale TCN encoder -> masked mean+std pooling;
        plus the coverage fraction per path.

        Args:
            groups: list of (x_g, mask_g) with x_g: (B, P, C_g, T_g)

        Returns:
            (B, P, 2*ts_dim*n_groups + n_groups)
        """
        embs, covs = [], []
        for stem, blocks, stride_g, (x_g, mask_g) in zip(self.stems, self.blocks, self.strides, groups):
            B, P, C, T = x_g.shape

            # per-timestep validity from the explicit mask (never from values!)
            t_mask = timestep_mask(mask_g).view(B * P, T)
            covs.append((t_mask.sum(dim=1) / T).view(B, P, 1))

            h = F.gelu(stem(x_g.reshape(B * P, C, T)))
            h, t_mask = masked_avg_pool1d(h, t_mask, stride_g)
            h = blocks(h)

            pooled = masked_mean_std_over_time(h.permute(0, 2, 1), t_mask)  # (B*P, 2*ts_dim)
            embs.append(pooled.view(B, P, -1))

        return torch.cat(embs + covs, dim=-1)

    # -----------------------------------------------------

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Input list (x_path, x_user, g0_x, g0_mask, ...)

        Returns:
            prediction: (B, output_dim)
        """
        x_path, x_user, groups = split_inputs(x)
        B = x_path.size(0)

        ts_paths = self.encode_timeseries(groups) * self.ts_gamma    # (B, P, 2*ts_dim*G + G)

        path_combined = self.path_drop(
            torch.cat([ts_paths, x_path], dim=-1))                   # (B, P, D_path)

        flat_track = self.flat_compress(path_combined).view(B, -1)   # (B, P*flat_dim)
        mean_track = path_combined.mean(dim=1)                       # (B, D_path)

        fused = torch.cat([flat_track, mean_track, x_user], dim=1)
        return self.regressor(fused)

    # =====================================================
    # OPTUNA
    # =====================================================

    @staticmethod
    def get_hyperparameter_space(trial: Any) -> Dict[str, Any]:
        return {
            "ts_dim": trial.suggest_categorical("ts_dim", [8, 16, 24]),
            "n_blocks": trial.suggest_int("n_blocks", 2, 4),
            "downsample_stride": trial.suggest_categorical("downsample_stride", [2, 4, 8]),
            "flat_dim": trial.suggest_categorical("flat_dim", [4, 6, 8, 12]),
            "dropout_ts": trial.suggest_float("dropout_ts", 0.05, 0.25),
            "dropout_path": trial.suggest_float("dropout_path", 0.05, 0.4),
            "regressor_dim": trial.suggest_categorical("regressor_dim", [256, 384, 512, 768]),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.2, 0.6),
            "lr": trial.suggest_float("lr", 2e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 5e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [6]),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "ts_dim": 16,
            "kernel_size": 5,
            "n_blocks": 3,
            "downsample_stride": 4,
            "flat_dim": 8,
            "dropout_ts": 0.1,
            "dropout_path": 0.15,
            "regressor_dim": 384,
            "dropout_reg": 0.5,
            "lr": 1e-3,
            "weight_decay": 3e-3,
            "batch_size": 6,
            "n_path_features": 35,
        }
